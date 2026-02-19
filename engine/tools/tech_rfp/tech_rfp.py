# pylint: skip-file
import re
import os
import json
import time
import fitz

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)
import asyncio
import warnings
import threading
from dataclasses import replace as dc_replace
from utils.document.docingest import get_documents
from typing import List, Dict, Any, Optional, Tuple

from tools.tech_rfp.tech_rfp_ptc import _call_llm_json
from utils.core.slack import SlackActivityLogger, fmt_dur
from tools.tech_rfp.tech_rfp_generate_report import build_technical_report
from utils.core.log import pid_tool_logger, set_logger, get_logger
from utils.storage.bucket import (
    list_files,
    list_subdirectories,
    build_public_url_for_key,
)
from google.genai import types
from google.genai.errors import ClientError
from google.genai.types import Part, Content
from utils.document.doc import _sanitize_parts_keep_media
from utils.core.fuzzy_search import (
    normalize_evidence_sources,
    fuzzy_search_tech_rfp,
    promote_fuzzy_key,
)
from utils.core.jsonval import validate_llm_response_tech_rfp, clean_malformed_json

warnings.filterwarnings(
    "ignore",
    message="style lookup by style_id is deprecated. Use style name as key instead.",
    module="docx.styles.styles",
)
from tools.tech_rfp.prompts_tech_rfp import (
    build_prompt_for_criterion_grade,
    build_ptc_prompt,
    EVAL_SYS_PROMPT,
    SENIOR_SYS_PROMPT,
)

from utils.llm.context_cache import (
    cache_enabled,
    create_cache,
    delete_cache,
    make_composite_contents,
    _min_cache_tokens,
    count_cached_tokens,
)

from utils.storage.gcs import (
    spill_media_and_text_to_gcs,
    preflight_inline_sizes,
    load_or_build_contractor_context,
    INLINE_BODY_BUDGET,
    TEXT_PART_LIMIT,
    MEDIA_PART_LIMIT,
)
from utils.llm.LLM import call_llm_async_cache, get_global_limiter, COMPRESSION_SETTINGS
from utils.llm.context_compactor import (
    compress_if_needed_document_aware,
    compress_with_memo,
    format_eval_criteria_for_summary,
)
from utils.llm.compactor_cache import set_compactor_context, _norm_name
from utils.vault import secrets

# One-time spill guard per contractor prefix
_SPILL_MEMO: set[str] = set()


def extract_criteria_from_structure(
    data: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str]]:
    """
    Flattens the evaluation structure to:
      - flat_criteria: {lower(criterion): weight_float}
      - name_map:      {lower(criterion): original_criterion_name}
      - desc_map:      {lower(criterion): description}  # optional

    Accepts either:
      { Scope: { Criterion: <number|string> } }
    or
      { Scope: { Criterion: { "weight": <number|string>, "description"?: <string> } } }
    and arbitrarily nested scopes.
    """
    flat_criteria: Dict[str, float] = {}
    name_map: Dict[str, str] = {}
    desc_map: Dict[str, str] = {}

    def _coerce_weight(w: Any) -> Optional[float]:
        if w is None:
            return None
        try:
            return float(w)
        except (ValueError, TypeError):
            # tolerate strings like "15%" or "0.15" by stripping non-numeric except dot
            if isinstance(w, str):
                s = w.strip().replace(",", ".")
                s = re.sub(r"[^\d.]", "", s)
                try:
                    return float(s) if s else None
                except Exception:
                    return None
            return None

    def walk(obj: Any):
        if not isinstance(obj, dict):
            return
        for key, val in obj.items():
            # Leaf in new schema: dict with weight(/description)
            if isinstance(val, dict) and ("weight" in val or "description" in val):
                crit_name = key
                lc = crit_name.strip().lower()
                w = _coerce_weight(val.get("weight"))
                if w is None:
                    continue
                flat_criteria[lc] = w
                name_map[lc] = crit_name
                desc = val.get("description")
                if isinstance(desc, str) and desc.strip():
                    desc_map[lc] = desc.strip()
                continue

            # Legacy leaf: direct number/string weight
            if isinstance(val, (int, float, str)):
                crit_name = key
                lc = crit_name.strip().lower()
                w = _coerce_weight(val)
                if w is None:
                    continue
                flat_criteria[lc] = w
                name_map[lc] = crit_name
                continue

            # Otherwise keep walking (scope/group)
            if isinstance(val, dict):
                walk(val)

    walk(data)

    if not flat_criteria:
        raise ValueError("No valid criteria were found in the provided data structure.")

    return flat_criteria, name_map, desc_map


async def _run_llm_async(
    prompt_parts: list[Part],
    *,
    model: str,
    system_instruction: str,
    thinking_budget: int | None = None,
    cached_content: str | None = None,
    config: dict | None = None,
    debug_caller: str | None = None,
    debug_prompt_hint: str | None = None,
) -> str:
    """
    Make an async Gemini call using the shared wrapper (Tenacity retry + rate-limit built-in).
    """
    cfg = {
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        "seed": 12345,
    }
    if config:
        cfg.update(config)

    if thinking_budget:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    use_cache = bool(cached_content) and cache_enabled()

    return await call_llm_async_cache(
        prompt_parts,
        model=model,
        system_instruction=None if use_cache else system_instruction,
        cfg=cfg,
        limiter=None,
        cached_content=cached_content,
        debug_caller=debug_caller,
        debug_prompt_hint=debug_prompt_hint,
    )


def build_cached_context_parts(
    rfp_parts: list[Part], contractor_parts: list[Part]
) -> list[Part]:
    return (
        [Part.from_text(text="### BEGIN RFP-REFERENCE\n")]
        + list(rfp_parts or [])
        + [Part.from_text(text="### END RFP-REFERENCE\n")]
        + [Part.from_text(text="\n### BEGIN CONTRACTOR-SUBMISSION\n")]
        + list(contractor_parts or [])
        + [Part.from_text(text="\n### END CONTRACTOR-SUBMISSION\n")]
    )


def prune_buckets_for_cached_call(
    original_parts: list[Part],
) -> tuple[list[Part], list[Part]]:
    """
    Keep only text Parts OUTSIDE the bucket sections delimited by our markers.
    Drops everything from '### BEGIN RFP-REFERENCE' through '### END CONTRACTOR-SUBMISSION'.
    """
    OUT, DROPPED = [], []
    dropping = False
    for p in original_parts or []:
        t = getattr(p, "text", None)
        # Start dropping at BEGIN marker
        if isinstance(t, str) and t.strip().startswith("### BEGIN RFP-REFERENCE"):
            dropping = True
            DROPPED.append(p)
            continue
        # Stop dropping after END CONTRACTOR
        if (
            dropping
            and isinstance(t, str)
            and t.strip().startswith("### END CONTRACTOR-SUBMISSION")
        ):
            dropping = False
            DROPPED.append(p)
            continue
        # While dropping, skip everything (including media/file parts)
        if dropping:
            DROPPED.append(p)
            continue
        # Outside bucket sections: keep only non-empty text Parts
        if isinstance(t, str) and t.strip():
            OUT.append(Part.from_text(text=t))
    return OUT, DROPPED


def _summarize_parts_quick(parts: list[Part]) -> dict:
    text = uri = inline = 0
    inline_bytes = 0
    for p in parts or []:
        if getattr(p, "text", None):
            text += 1
        else:
            fd = getattr(p, "file_data", None)
            if fd and getattr(fd, "file_uri", None):
                uri += 1
            else:
                il = getattr(p, "inline_data", None)
                if il:
                    inline += 1
                    inline_bytes += len(il.data or b"")
    return {
        "total": len(parts or []),
        "text": text,
        "uri": uri,
        "inline": inline,
        "inline_bytes": inline_bytes,
    }


def norm(k: str) -> str:
    return k.strip().lower()


GRADE_LEVELS: dict[str, int] = {
    "Major Concerns": 1,
    "Serious Concerns": 2,
    "High Concerns": 3,
    "Minor-Moderate Concerns": 4,
    "Moderate Confidence": 5,
    "Sound Confidence": 6,
    "Good Confidence": 7,
    "Very Good Confidence": 8,
    "Excellent Confidence": 9,
    "Outstanding": 10,
}

BINARY_GRADES: dict[str, float] = {
    "Non-Compliant": 0.0,
    "Compliant": 100.0,
}


def _level_to_score_0_100(level: int) -> float:
    return float(max(1, min(10, int(level))) * 10)


GRADE_TO_SCORE: dict[str, float] = {
    label: _level_to_score_0_100(level) for label, level in GRADE_LEVELS.items()
}
GRADE_TO_SCORE.update(BINARY_GRADES)
LEVEL_TO_GRADE: dict[int, str] = {v: k for k, v in GRADE_LEVELS.items()}


def _norm_label(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s+", " ", s)
    return s


_GRADE_NORM_MAP: dict[str, str] = {_norm_label(k): k for k in GRADE_LEVELS.keys()}
_GRADE_NORM_MAP.update({_norm_label(k): k for k in BINARY_GRADES.keys()})


def normalize_grade(label: Any) -> str | None:
    if isinstance(label, (int, float)) and float(label).is_integer():
        n = int(label)
        if 1 <= n <= 10:
            return LEVEL_TO_GRADE[n]
        return None

    if not isinstance(label, str):
        return None
    raw = label.strip()
    if re.fullmatch(r"\d{1,2}", raw):
        n = int(raw)
        if 1 <= n <= 10:
            return LEVEL_TO_GRADE[n]
        return None

    m = re.match(r"^\s*(\d{1,2})\s*[-–—:]\s*(.+)$", raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            return LEVEL_TO_GRADE[n]

    key = _norm_label(raw)
    if key in _GRADE_NORM_MAP:
        return _GRADE_NORM_MAP[key]

    key2 = re.sub(r"[^\w\s/+-]", "", key)
    if key2 in _GRADE_NORM_MAP:
        return _GRADE_NORM_MAP[key2]
    return None


def next_higher_grade(label: str) -> str | None:
    """
    Legacy helper retained for old evaluation fallback logic.
    Returns the next better grade on the 10-level scale, or None at ceiling.
    """
    g = normalize_grade(label)
    if not g:
        return None
    lvl = GRADE_LEVELS.get(g)
    if not lvl:
        return None
    if lvl >= 10:
        return None
    return LEVEL_TO_GRADE.get(lvl + 1)


def _crit_key(path: list[str], leaf: str) -> str:
    """Legacy criteria key formatter."""
    bits = [*(path or []), leaf]
    return " / ".join(str(x).strip() for x in bits if str(x).strip()).lower()


def _crit_display(path: list[str], leaf: str) -> str:
    """Legacy criteria display formatter."""
    bits = [*(path or []), leaf]
    return " / ".join(str(x).strip() for x in bits if str(x).strip())


def _part_descriptor(p: Part) -> dict:
    """Compact descriptor used by legacy cache/debug audit helpers."""
    t = getattr(p, "text", None)
    if isinstance(t, str):
        b = len(t.encode("utf-8"))
        return {"type": "text", "bytes": b, "preview": t[:120]}
    fd = getattr(p, "file_data", None)
    if fd and getattr(fd, "file_uri", None):
        return {
            "type": "uri",
            "uri": getattr(fd, "file_uri", ""),
            "mime": getattr(fd, "mime_type", ""),
        }
    il = getattr(p, "inline_data", None)
    if il:
        return {
            "type": "inline",
            "bytes": len(getattr(il, "data", b"") or b""),
            "mime": getattr(il, "mime_type", ""),
        }
    return {"type": "unknown"}


def _markers_present(parts: list[Part]) -> bool:
    txt = "\n".join(
        getattr(p, "text", "")
        for p in (parts or [])
        if isinstance(getattr(p, "text", None), str)
    )
    return ("### BEGIN RFP-REFERENCE" in txt) and ("### END CONTRACTOR-SUBMISSION" in txt)


async def ask_llm_for_criterion(
    prompt_input: str | Content,
    semaphore: asyncio.Semaphore,
    *,
    purpose: str = "evaluation",
    criterion_name: str | None = None,
    retries: int = 3,
    cached_content: str | None = None,
    spill_prefix: str | None = None,
    prepared_context_parts: Optional[List[Part]] = None,
) -> str:
    """
    Async wrapper that:
    1. Chooses persona model + system prompt.
    2. Calls Gemini with shared util.
    3. Sanitises / validates JSON, with up-to-{retries} attempts.
    Returns the raw (possibly fixed) JSON string.
    """
    logger = get_logger()
    # persona selection
    if purpose in ("pro"):
        model = "gemini-2.5-pro"
        system_instruction = EVAL_SYS_PROMPT
        tb = 5000
    else:
        model = "gemini-2.5-pro"
        system_instruction = EVAL_SYS_PROMPT
        tb = 5000

    # If we have a prepared (spilled/compressed) context, always prefer it.
    # We'll append only the small, dynamic criterion instruction inline.
    if prepared_context_parts:
        # Build minimal dynamic instruction from prompt_input
        if isinstance(prompt_input, Content):
            # Extract only the small textual instruction bits
            tiny: List[Part] = []
            for p in list(prompt_input.parts or []):
                t = getattr(p, "text", None)
                if isinstance(t, str) and len(t.encode("utf-8")) <= 8_192:
                    tiny.append(Part.from_text(text=t))
            parts = list(prepared_context_parts) + tiny
        else:
            parts = list(prepared_context_parts) + [
                Part.from_text(text=str(prompt_input))
            ]

    else:
        # unify prompt parts
        parts: list[Part]
        use_cache = bool(cached_content) and cache_enabled()

        if isinstance(prompt_input, Content):
            original = list(prompt_input.parts or [])
            if use_cache:
                # When referencing a cache, send only the small dynamic text prompt
                kept, dropped = prune_buckets_for_cached_call(original)
                parts = kept
                dsum = _summarize_parts_quick(dropped)
                ksum = _summarize_parts_quick(kept)
                logger.debug(
                    "cache pruned prompt: kept=%s | dropped=%s",
                    ksum,
                    dsum,
                )
            else:
                # No cache - keep media + non-empty text, but DO NOT respill when a contractor prefix is provided.
                parts = _sanitize_parts_keep_media(original)

                # If we have a per-contractor spill_prefix, assume the contractor context already spilled:
                if spill_prefix:
                    # Keep only tiny inline glue; anything big should already be in URIs.
                    compact: List[Part] = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str):
                            if len(t.encode("utf-8")) <= 8_192:
                                compact.append(Part.from_text(text=t))
                            # else drop; it should have been included via text_uri in the spilled context
                        else:
                            fd = getattr(p, "file_data", None)
                            if fd and getattr(fd, "file_uri", None):
                                compact.append(p)
                            else:
                                inline = getattr(p, "inline_data", None)
                                if inline and inline.data:
                                    # Drop large inline media; should have been spilled earlier
                                    continue
                    parts = compact
                else:
                    # Fallback: spill to a stable, tiny prefix (rare path)
                    try:
                        logger.debug(
                            "Fallback: spill to a stable, tiny prefix (rare path). Spilling"
                        )
                        base = f"tmp/rare_case"
                        tmp_prefix = f"{base}/nocache/{criterion_name or 'criterion'}"
                        parts, _m, _ = spill_media_and_text_to_gcs(
                            parts, prefix=tmp_prefix, force_spill_all_text=False
                        )
                    except Exception:
                        # fall back to inline if spill fails
                        pass
        else:
            # Plain string prompt
            parts = [Part.from_text(text=str(prompt_input))]

    # retry loop with semaphore
    async with semaphore:
        last_raw = ""
        for attempt in range(1, retries + 1):
            try:
                raw = await _run_llm_async(
                    parts,
                    model=model,
                    system_instruction=system_instruction,
                    thinking_budget=tb,
                    cached_content=cached_content,
                    debug_caller=f"tech_rfp.ask_llm_for_criterion:{purpose}",
                    debug_prompt_hint=f"criterion={criterion_name or 'unknown'}",
                )
                if not raw:
                    raise ValueError("LLM returned empty response")

                last_raw = raw
                fixed = clean_malformed_json(raw, label=criterion_name or "criterion")

                # structural check
                parsed = json.loads(fixed)
                if purpose in ("pro"):
                    ok = (
                        isinstance(parsed, dict)
                        and "grade" in parsed
                        and "reasoning" in parsed
                    )
                else:  # consolidation
                    ok = isinstance(parsed, dict) and "reasoning" in parsed

                if ok:
                    return fixed
            except Exception as e:
                logger.exception(
                    f"[{criterion_name}] LLM attempt {attempt}/{retries} failed: {e}"
                )
                if attempt < retries:
                    await asyncio.sleep(attempt)

        # fall-back after all retries
        return clean_malformed_json(
            last_raw
            or json.dumps(
                {
                    "criterion": "INTERNAL_FAILED",
                    "grade": "Major Concerns",
                    "reasoning": [
                        {
                            "text": "Internal error after retries.",
                            "source": None,
                            "pageNumber": None,
                        }
                    ],
                }
            ),
            label=criterion_name or "criterion",
        )


async def _summarize_reasoning_for_client(
    *,
    criterion: str,
    grade: Any,
    reasoning: Any,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Uses gemini-2.5-flash to compress a senior response's reasoning into a
    short client-facing summary (<=3 sentences). Returns a plain string.

    Falls back to a simple join of the first couple of reasoning items if the LLM fails.
    """

    # Extract up to 5 short points from reasoning
    def _collect_points(r: Any) -> list[str]:
        pts: list[str] = []
        if isinstance(r, list):
            for it in r:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    t = it["text"].strip()
                    if t:
                        pts.append(t)
                elif isinstance(it, str):
                    s = it.strip()
                    if s:
                        pts.append(s)
        elif isinstance(r, dict):
            t = str(r.get("text", "")).strip()
            if t:
                pts.append(t)
        elif isinstance(r, str):
            s = r.strip()
            if s:
                pts.append(s)
        return pts[:5]

    points = _collect_points(reasoning)

    prompt = (
        "You are drafting an executive-facing summary for a client.\n"
        "Summarize the evaluator's reasoning for the criterion below in AT MOST 3 sentences, "
        "focusing on the most important strengths and the key gaps that affected the grade. "
        "Do NOT mention numeric scores or levels. Avoid citations, file names, or page numbers. Be clear and neutral.\n\n"
        f"Criterion: {criterion}\n"
        f"Final Grade (verbatim): {grade}\n"
        "Reasoning points:\n" + "\n".join(f"- {p}" for p in points) + "\n\n"
        'Return ONLY valid JSON: {"summary": "<<=3 sentences>"}'
    )

    try:
        async with semaphore:
            json_str = await _run_llm_async(
                [Part.from_text(text=prompt)],
                model="gemini-2.5-flash",
                system_instruction="You write concise executive summaries. Output JSON with one key: summary.",
                thinking_budget=512,
                cached_content=None,
                debug_caller="tech_rfp._summarize_reasoning_for_client",
                debug_prompt_hint=f"criterion={criterion}",
            )

        fixed = clean_malformed_json(json_str, label="client-summary")
        parsed = json.loads(fixed)
        summary = (parsed or {}).get("summary")
        if isinstance(summary, str) and summary.strip():
            # defensively cap to 3 sentences
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", summary.strip())
                if s.strip()
            ]
            return " ".join(sentences[:3])
    except Exception:
        pass

    # Fallback: brief, local summary from first couple points
    return " ".join(points[:2]) if points else "Summary unavailable."


async def _justify_final_score(
    *,
    resp_senior: dict,
    criterion: str,
    semaphore: asyncio.Semaphore,
    max_points: float = 100.0,
) -> str:
    """
    Produces one plain-text paragraph (in the senior's voice) justifying the final score,
    focusing on strengths and on where/why points were lost.
    """

    score = resp_senior.get("score", 0)

    # score handling
    try:
        s = float(score) if score is not None else 0.0
    except Exception:
        s = 0.0
    lost = max(0.0, float(max_points) - s)

    system_instruction = (
        "You are the SAME senior evaluator who assigned the final score. "
        "Write in the senior evaluator's voice (first person plural or neutral). "
        "Do NOT mention that you're an AI or that another model was used. "
        "Output must be PLAIN TEXT (no JSON, no lists, no headings). "
        "Be concise, factual, and focus on where and why points were lost."
    )

    prefix = Part.from_text(
        text=(
            "Task: Write ONE cohesive paragraph as the senior evaluator justifying the score.\n"
            f"Criterion: {criterion}\n"
            f"Final score (0–{int(max_points)}): {s:.1f}\n"
            f"Points not awarded: {lost:.1f}\n"
            "Requirements:\n"
            f"1) Start with: 'Final score: {s:.1f}/{int(max_points)}.'\n"
            "2) Briefly note strengths and why points were awarded.\n"
            "3) Provide a numeric deduction breakdown summing exactly to the points lost (one decimal place).\n"
            "4) ONE paragraph, plain text, no bullets, no citations, no file/page names.\n"
        )
    )

    sr_part = Part.from_text(
        text="### BEGIN EVALUATOR-JSON\n"
        + json.dumps(resp_senior, indent=2)
        + "\n### END EVALUATOR-JSON\n"
    )

    parts = [prefix, sr_part]

    async with semaphore:
        text = await _run_llm_async(
            parts,
            model="gemini-2.5-flash",
            system_instruction=system_instruction,
            thinking_budget=1024,
            config={
                "temperature": 0,
                "top_p": 1.0,
                "top_k": 1,
                "max_output_tokens": 1024,
                "response_mime_type": "text/plain",
                "seed": 12345,
            },
            debug_caller="tech_rfp._justify_final_score",
            debug_prompt_hint=f"criterion={criterion}",
        )

    return (text or "").strip()


async def evaluate_proposal(
    package_id: str,
    evaluation_criteria: Dict[str, float],
    proposal_parts: List[Part],
    file_names: List[str],
    rfp_parts: List[Part],
    name_map: dict[str, str],
    *,
    description_map: Optional[Dict[str, str]] = None,
    pro_cache_name: str | None = None,
    act: "SlackActivityLogger | None" = None,
    spill_prefix: str | None = None,
    prepared_context_parts: Optional[List[Part]] = None,
    progress_hook: Optional[callable] = None,
) -> Dict[str, Any]:
    """Evaluates the proposal against the criteria and returns a structured result."""
    logger = get_logger()

    total_criteria = len(evaluation_criteria)
    logger.debug(f"Starting evaluation for {total_criteria} criteria.")
    act.sub(f"🔄 Starting evaluation for {total_criteria} criteria.")

    CONCURRENCY_LIMIT = 10
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    logger.debug(
        f"Starting evaluation with async concurrency limit of {CONCURRENCY_LIMIT}"
    )

    async def evaluate_with_retries(
        prompt,
        purpose: str,
        criterion_name: str,
        cached_content: str | None,
        prepared_context_parts: Optional[List[Part]],
    ) -> Optional[Dict[str, Any]]:
        """Runs retry loop, JSON parsing, validation, grade mapping, and salvage logic."""
        max_retries = 3
        attempt = 0
        valid_response = None
        last_response_text = ""
        last_error = None
        logger = get_logger()
        while attempt < max_retries and valid_response is None:
            response_text = await ask_llm_for_criterion(
                prompt,
                semaphore=semaphore,
                criterion_name=criterion_name,
                purpose=purpose,
                cached_content=cached_content,
                spill_prefix=spill_prefix,
                prepared_context_parts=prepared_context_parts,
            )
            last_response_text = response_text
            try:
                response_json = json.loads(response_text)

                # if it's a list of dicts, pick the first one
                if (
                    isinstance(response_json, list)
                    and response_json
                    and isinstance(response_json[0], dict)
                ):
                    response_json = response_json[0]

                if not isinstance(response_json, dict):
                    raise ValueError("LLM response must be a JSON object.")

                # fix mangled grade key
                try:
                    promote_fuzzy_key(response_json, "grade", score_threshold=70)
                except KeyError as e:
                    raise ValueError("No grade key found in LLM response.") from e

                grade_raw = response_json.get("grade")
                grade = normalize_grade(grade_raw)
                if not grade:
                    raise ValueError(f"Invalid grade: {grade_raw}")

                response_json["grade"] = grade
                response_json["score"] = float(GRADE_TO_SCORE[grade])

                if not validate_llm_response_tech_rfp(response_json):
                    raise ValueError("LLM response structure failed validation.")

                valid_response = response_json
                return valid_response

            except (json.JSONDecodeError, ValueError) as e:
                logger.exception(
                    f"Attempt {attempt+1} for criterion '{criterion_name}' failed: {e}"
                )
                logger.error(f"Raw LLM Response for {criterion_name}:\n{response_text}")

                if attempt == max_retries - 1:
                    try:
                        grade_match = re.search(r'"grade"\s*:\s*"([^"]+)"', last_response_text)
                        reasoning_match = re.search(r'"reasoning"\s*:\s*(\[[\s\S]*\])', last_response_text)

                        extracted_grade_raw = grade_match.group(1) if grade_match else None
                        extracted_grade = normalize_grade(extracted_grade_raw) if extracted_grade_raw else None
                        if not extracted_grade:
                            extracted_grade = "Major Concerns"

                        extracted_reasoning: list[dict] = []
                        if reasoning_match:
                            try:
                                extracted_reasoning = json.loads(reasoning_match.group(1))
                            except Exception:
                                extracted_reasoning = []

                        if not isinstance(extracted_reasoning, list) or not extracted_reasoning:
                            extracted_reasoning = [
                                {
                                    "text": "Extracted from incomplete/invalid model output; detailed reasoning unavailable.",
                                    "source": None,
                                    "pageNumber": None,
                                }
                            ]

                        valid_response = {
                            "criterion": criterion_name,
                            "grade": extracted_grade,
                            "score": float(GRADE_TO_SCORE.get(extracted_grade, 0.0)),
                            "reasoning": extracted_reasoning,
                        }
                        logger.warning(
                            f"Salvaged partial response for {criterion_name} with grade '{extracted_grade}'"
                        )
                    except Exception as salvage_error:
                        logger.exception(f"Failed to salvage response: {salvage_error}")
                        valid_response = None
                else:
                    valid_response = None

                last_error = e
                attempt += 1

                if attempt < max_retries:
                    act.sub(
                        f"Retrying criterion '{criterion_name}' (attempt {attempt+1})..."
                    )

        return (
            valid_response
            if valid_response is not None
            else {
                "criterion": criterion_name,
                "grade": "Major Concerns",
                "score": float(GRADE_TO_SCORE["Major Concerns"]),
                "reasoning": [
                    {
                        "text": (
                            f"Error: Failed to process grade after {max_retries} attempts. "
                            f"Last error: {last_error}. Raw response: {last_response_text}"
                        )[:2000],
                        "source": None,
                        "pageNumber": None,
                    }
                ],
            }
        )

    async def process_criterion(criterion_lc: str, weight: float):
        logger = get_logger()
        try:
            criterion = name_map.get(criterion_lc, criterion_lc)
            crit_desc = (description_map or {}).get(criterion_lc)

            prompt_grade = build_prompt_for_criterion_grade(
                criterion, weight, proposal_parts, rfp_parts, description=crit_desc
            )

            resp = await evaluate_with_retries(
                prompt_grade,
                purpose="pro",
                criterion_name=criterion,
                cached_content=pro_cache_name if pro_cache_name else None,
                prepared_context_parts=prepared_context_parts,
            )
            if progress_hook:
                await progress_hook()

            client_summary = "Summary unavailable."
            try:
                client_summary = await _summarize_reasoning_for_client(
                    criterion=criterion,
                    grade=resp.get("grade", "NA"),
                    reasoning=resp.get("reasoning", []),
                    semaphore=semaphore,
                )
            except Exception as e:
                logger.exception(f"Client summary failed for '{criterion}': {e}")

            ptc_final = "NA"
            try:
                evaluation_entry = {
                    "criterion": criterion,
                    "llm_responses": [{"persona": "pro", "response": resp}],
                }
                evaluation_json = json.dumps(evaluation_entry, indent=2)
                ptc_prompt, ptc_schema = build_ptc_prompt(evaluation_json)

                ptc_dict = await asyncio.to_thread(
                    _call_llm_json, ptc_prompt, ptc_schema
                )

                if ptc_dict and "ptc" in ptc_dict:
                    ptc_final = ptc_dict["ptc"]
                else:
                    ptc_final = {"queryDescription": "NA", "refType": "N/A"}

            except Exception as e:
                logger.exception(
                    f"PTC generation for criterion '{criterion}' failed: {e}. Defaulting to 'NA'."
                )

            result_package = {
                "raw_llm_responses": {
                    "criterion": criterion,
                    "llm_responses": [
                        {"persona": "pro", "response": resp},
                    ],
                },
                "tenderEvaluation": {
                    "score": resp.get("score", 0),
                    "grade": resp.get("grade", "NA"),
                    "reasoning": resp.get("reasoning", []),
                    "ptc": ptc_final,
                    "clientSummary": client_summary,
                },
            }

            return (criterion_lc, result_package)

        except Exception as e:
            logger.exception(
                f"FATAL ERROR in thread for criterion '{criterion_lc}'. The thread will now terminate."
            )

            if progress_hook:
                await progress_hook()

            return (
                criterion_lc,
                {
                    "error": str(e),
                    "raw_llm_responses": {},
                    "tenderEvaluation": {
                        "score": 0,
                        "reasoning": [{"text": "Processing error"}],
                    },
                },
            )

    tasks = [
        process_criterion(criterion, weight)
        for criterion, weight in evaluation_criteria.items()
    ]
    all_results_list = await asyncio.gather(*tasks)

    final_output = {"tenderEvaluation": {}, "raw_llm_responses": []}
    for criterion_lc, result_package in all_results_list:
        if result_package and "error" not in result_package:
            final_output["tenderEvaluation"][criterion_lc] = result_package[
                "tenderEvaluation"
            ]
            final_output["raw_llm_responses"].append(
                result_package["raw_llm_responses"]
            )
        else:
            logger.error(
                f"Could not process criterion '{criterion_lc}': {result_package.get('error')}"
            )
            final_output["tenderEvaluation"][criterion_lc] = {
                "score": 0,
                "reasoning": [{"text": "An error occurred during evaluation."}],
                "ptc": "NA",
            }

    logger.debug(
        f"All raw LLM responses:\n{json.dumps(final_output['raw_llm_responses'], indent=2)}"
    )

    return final_output


def scale_score(
    data_structure: Dict[str, Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
    name_map: Dict[str, str],
    doc_index: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Calculate scaled scores for tender evaluations. This now uses contractor-specific file names.
    """
    logger = get_logger()
    result = {"result": {"tenderReport": []}}
    # Convert weights to float and create a lowercase lookup
    normalized_data_structure = {}
    original_criteria_map = {}
    for category, criteria in data_structure.items():
        normalized_category = {}
        for criterion, weight_raw in criteria.items():
            if isinstance(weight_raw, dict):
                weight_val = weight_raw.get("weight")
            else:
                weight_val = weight_raw

            try:
                weight = float(weight_val)
            except (ValueError, TypeError):
                logger.exception(
                    f"Invalid weight '{weight_val}' for criterion '{criterion}'. Skipping."
                )
                weight = 0.0

            key = norm(criterion)
            normalized_category[key] = weight
            original_criteria_map[key] = criterion
        normalized_data_structure[category] = normalized_category

    original_criteria_map.update(name_map)

    for evaluation in evaluations:
        contractor_name = evaluation.get("contractorName", "Unknown Contractor")
        contractor_files = evaluation.get("file_names", [])
        contractor_errors = evaluation.get("documentProcessingErrors", [])
        raw_llm_responses = evaluation.get("raw_llm_responses", [])

        contractor_scores = {
            "contractorName": contractor_name,
            "tenderEvaluation": {"scopes": [], "totalScore": 0.0},
            "documentProcessingErrors": contractor_errors,
            "raw_llm_responses": raw_llm_responses,
        }

        for category_name, criteria in normalized_data_structure.items():
            scope = {
                "scopeName": category_name,
                "scopeTotal": 0.0,
                "evaluationBreakdown": [],
            }

            for criterion, weight in criteria.items():
                eval_data = evaluation["tenderEvaluation"].get(criterion)

                if eval_data is None:
                    logger.warning(
                        f"No evaluation data found for criterion '{criterion}' in {contractor_name}'s evaluation. Skipping."
                    )
                    continue

                score = eval_data.get(fuzzy_search_tech_rfp("score", eval_data.keys()))
                try:
                    if isinstance(score, str):
                        score = float(score)
                    elif not isinstance(score, (int, float)):
                        raise ValueError("Invalid score type")
                except ValueError:
                    logger.error(
                        f"Invalid score for criterion '{criterion}' in {contractor_name}. Found: {score} ({type(score).__name__})"
                    )
                    score = None

                try:
                    weight = float(weight)
                except ValueError:
                    logger.error(
                        f"Invalid weight '{weight}' for criterion '{criterion}'. Skipping."
                    )
                    continue

                if score is None:
                    score = 0
                    scaled_score = 0
                else:
                    weight = weight / 100
                    scaled_score = score * weight

                evidence = eval_data.get("reasoning", [])
                ptc = eval_data.get("ptc", "NA")
                if not isinstance(evidence, list):
                    evidence = [
                        {
                            "text": str(evidence)[:200],
                            "source": "Unknown Source",
                            "pageNumber": None,
                        }
                    ]

                # match against all docs (RFP + tender) and attach URLs
                evidence = normalize_evidence_sources(
                    evidence,
                    doc_index,
                    contractor_name=contractor_name,
                )

                client_summary = eval_data.get("clientSummary")
                scope["evaluationBreakdown"].append(
                    {
                        "criteria": original_criteria_map.get(criterion, criterion),
                        "ptc": ptc,
                        "score": score,
                        "grade": eval_data.get("grade", "NA"),
                        "result": scaled_score,
                        "evidence": evidence,
                        "clientSummary": client_summary,
                    }
                )

                scope["scopeTotal"] += scaled_score

            contractor_scores["tenderEvaluation"]["totalScore"] += scope["scopeTotal"]
            contractor_scores["tenderEvaluation"]["scopes"].append(scope)

        # flag contractor if >50% of criteria have score == 0
        zero_scores = 0
        total_scored = 0
        for scope in contractor_scores["tenderEvaluation"]["scopes"]:
            for row in scope.get("evaluationBreakdown", []):
                s = row.get("score")
                try:
                    if isinstance(s, str):
                        s_val = float(s)
                    elif isinstance(s, (int, float)):
                        s_val = float(s)
                    else:
                        continue
                except Exception:
                    continue

                total_scored += 1
                if s_val == 0.0:
                    zero_scores += 1

        zero_ratio = (zero_scores / total_scored) if total_scored else 0.0
        if total_scored > 0 and zero_ratio > 0.5:
            contractor_scores.setdefault("warnings", {})
            contractor_scores["warnings"]["tooManyZeroScores"] = {
                "flag": True,
                "zeroScoreCount": zero_scores,
                "totalCriteriaCount": total_scored,
                "zeroScoreRatio": zero_ratio,
                "message": (
                    "More than 50% of criteria for this contractor received a score of 0. "
                    "This may indicate missing content in the submission, extraction issues, "
                    "or that the proposal is not responsive to the RFP."
                ),
            }

        result["result"]["tenderReport"].append(contractor_scores)

    # Normalize and format document errors for frontend
    document_errors = []
    for contractor in result["result"]["tenderReport"]:
        contractor_name = contractor.get("contractorName", "Unknown Contractor")
        contractor_errors = contractor.pop("documentProcessingErrors", [])

        for item in contractor_errors:
            file_name = item.get("file", "unknown")
            raw_msgs = []
            if "error" in item:
                raw_msgs.append(item["error"])
            raw_msgs.extend(item.get("errors", []))

            # Log truncated raw errors
            for raw_msg in raw_msgs:
                clean_msg = str(raw_msg)
                if len(clean_msg) > 500:
                    clean_msg = clean_msg[:500] + " ...[truncated]"
                logger.error(f"Error for file '{file_name}': {clean_msg}")

            # Always include failedPages (can be empty)
            document_errors.append(
                {
                    "file": file_name,
                    "contractorName": contractor_name,
                    "errors": [
                        "We couldn't process this document. The file may be damaged or in an unsupported format. Please check the file and try uploading it again."
                    ],
                    "failedPages": item.get("failedPages", []),
                }
            )

    result["result"]["documentProcessingErrors"] = document_errors
    return result


async def score_proposal_main(
    package_id: str,
    proposal_parts: List[Part],
    rfp_parts: List[Part],
    evaluation_criteria: Dict,
    name_map: Dict,
    contractor_name: str,
    proposal_file_names: List[str],
    *,
    description_map: Optional[Dict[str, str]] = None,
    pro_cache_name: str | None = None,
    act: "SlackActivityLogger | None" = None,
    spill_prefix: str | None = None,
    prepared_context_parts: Optional[List[Part]] = None,
    progress_hook: Optional[callable] = None,
) -> Dict[str, Any]:
    """Main function to process a single proposal."""

    logger = get_logger()
    start_time = time.time()
    p4 = package_id[-4:]
    results = await evaluate_proposal(
        package_id,
        evaluation_criteria,
        proposal_parts,
        proposal_file_names,
        rfp_parts,
        name_map,
        description_map=description_map,
        pro_cache_name=pro_cache_name,
        act=act,
        spill_prefix=spill_prefix,
        prepared_context_parts=prepared_context_parts,
        progress_hook=progress_hook,
    )
    results["contractorName"] = contractor_name

    end_time = time.time()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    readable_time = f"{int(minutes)}m {int(seconds)}s"

    slack_msg = f"⏱️ {p4} Contractor *{contractor_name}* processed in {readable_time}."
    act.sub(slack_msg)

    logger.debug(f"Contractor {contractor_name} processed in {readable_time}.")
    results["file_names"] = proposal_file_names.copy()
    return results


def _mk_progress(
    *,
    current_contractor: str | None,
    finished_for_current: int,
    total_criteria_per_contractor: int,
    contractor_index: int,
    total_contractors: int,
    contractor_names: list[str],
) -> dict:
    total_done = (
        contractor_index * total_criteria_per_contractor
    ) + finished_for_current
    total_all = total_contractors * total_criteria_per_contractor
    overall = (total_done / total_all) if total_all else 0.0

    next_name = None
    if contractor_index + 1 < total_contractors:
        next_name = contractor_names[contractor_index + 1]

    pct_current = (
        (finished_for_current / total_criteria_per_contractor)
        if total_criteria_per_contractor
        else 0.0
    )

    return {
        "currentContractor": current_contractor or "",
        "currentContractorPercentageCompletion": float(pct_current),
        "numberOfCriterionFinishedForCurrentContractor": int(finished_for_current),
        "overallPercentageCompletion": float(overall),
        "totalNumberOfCriteriaToBeAnalyzed": int(total_all),
        "nextContractorToBeProcessed": next_name or "",
    }


async def tech_rfp_analysis_main(
    *,
    package_id: str | None = None,
    company_id: str | None = None,
    country_code: str = "USA",
    request_method: str | None = None,
    remote_ip: str | None = None,
    user_name: str | None = None,
    compute_reanalysis: bool = True,
    user_id: str | None = None,
    package_name: str | None = None,
    evaluation_criteria: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Entry-point and router for the Tech-RFP tool.
    Uses job queue system for processing.
    """
    from utils.db.job_queue import JobType
    from tools.tech_rfp.job_queue_integration import (
        create_tech_rfp_job,
        get_job_status_response,
    )

    base_logger = pid_tool_logger(package_id, "tech_rfp_analysis")
    set_logger(
        base_logger,
        tool_name="tech_rfp_analysis_main",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()
    country_code = country_code or "USA"

    if request_method == "GET":
        return get_job_status_response(
            package_id=package_id,
            company_id=company_id,
            job_type=JobType.TECH_RFP_ANALYSIS,
        )

    if request_method == "POST":
        if not compute_reanalysis:
            response = get_job_status_response(
                package_id=package_id,
                company_id=company_id,
                job_type=JobType.TECH_RFP_ANALYSIS,
            )
            if response.get("status") == "completed":
                return response

        if evaluation_criteria is None:
            return {
                "status": "error",
                "error": "evaluation_criteria is required for tech_rfp analysis jobs",
            }

        job_id = create_tech_rfp_job(
            job_type=JobType.TECH_RFP_ANALYSIS,
            package_id=package_id,
            company_id=company_id,
            user_id=user_id,
            user_name=user_name,
            payload_fields={"evaluation_criteria": evaluation_criteria},
            remote_ip=remote_ip,
            country_code=country_code,
            package_name=package_name,
        )

        logger.info(f"Created tech_rfp_analysis job {job_id} for project {package_id}")
        return {
            "status": "in progress",
            "message": f"RFP analysis job created for {package_id}",
            "job_id": job_id,
        }

    return {"error": f"Unsupported request method: {request_method}", "status": "error"}


# pre-cache constants (if file to big -> don't cache -> compress )
MAX_TOKENS_PRO = 1_048_576
CACHE_HEADROOM = 50_000


def _cache_safe_tokens() -> int:
    try:
        reserve = int(getattr(COMPRESSION_SETTINGS, "prompt_reserve_tokens", 20_000))
    except Exception:
        reserve = 20_000
    return MAX_TOKENS_PRO - CACHE_HEADROOM - reserve


def _count_tokens_cache_pro(parts: list[Part]) -> int:
    contents = make_composite_contents(parts)
    return count_cached_tokens("gemini-2.5-pro", contents)


def _count_cache_tokens(parts: list[Part]) -> int:
    """Legacy name kept for compatibility with old scripts."""
    return _count_tokens_cache_pro(parts)


def _cache_aware_count_tokens(model: str, parts) -> int:
    """
    Count tokens on Content(role='user', parts=...) shape used by cache creation.
    """
    contents = make_composite_contents(list(parts))
    return count_cached_tokens(model, contents)


def drop_image_parts(parts: list[Part]) -> list[Part]:
    """Legacy utility: keep only text/uri parts, drop inline image/media blobs."""
    kept: list[Part] = []
    for p in parts or []:
        if isinstance(getattr(p, "text", None), str):
            kept.append(p)
            continue
        fd = getattr(p, "file_data", None)
        if fd and getattr(fd, "file_uri", None):
            kept.append(p)
    return kept


def _dump_cache_audit(*args, **kwargs) -> None:
    """Legacy debug hook preserved as no-op in queue architecture."""
    get_logger().debug("cache audit hook called (compat mode).")


def _dump_failed_criterion_call(*args, **kwargs) -> None:
    """Legacy debug hook preserved as no-op in queue architecture."""
    get_logger().debug("failed criterion dump hook called (compat mode).")


def _upload_error_artifacts(*, err: Exception | str, stage: str = "process", **_) -> None:
    """Legacy compatibility hook for old non-queue callers."""
    get_logger().warning("error artifact (compat): stage=%s err=%s", stage, err)


def _derive_dynamic_budget(
    n_docs: int = 1,
    *,
    cache_safe: int | None = None,
) -> dict:
    if cache_safe is None:
        cache_safe = _cache_safe_tokens()

    hard = cache_safe
    target = cache_safe - 30_000
    frac = max(0.15, 0.40 / max(1, (n_docs / 10)))
    per_doc = max(50_000, int(target * frac))
    return {
        "hard_limit_tokens": hard,
        "target_total_tokens": target,
        "per_doc_ceiling_tokens": per_doc,
    }


def _estimate_doc_count(parts: list[Part]) -> int:
    tag_re = re.compile(r"^\s*<Start of File:\s*.+?>\s*$")
    n = 0
    for p in parts or []:
        t = getattr(p, "text", None)
        if isinstance(t, str) and tag_re.match(t):
            n += 1
    return max(1, n)


def ensure_cacheable_parts(
    *,
    parts: list[Part],
    limiter,
    name_to_source_key: dict[str, str] | None = None,
    eval_criteria_text: str | None = None,
) -> tuple[list[Part], int, str]:
    logger = get_logger()
    cache_safe = _cache_safe_tokens()
    n_docs = _estimate_doc_count(parts)

    toks0 = _count_tokens_cache_pro(parts)
    if toks0 is not None and toks0 >= 0 and toks0 <= cache_safe:
        return parts, int(toks0), "no extra compression needed"

    budget = _derive_dynamic_budget(n_docs, cache_safe=cache_safe)
    settings = dc_replace(
        COMPRESSION_SETTINGS,
        hard_limit_tokens=budget["hard_limit_tokens"],
        target_total_tokens=budget["target_total_tokens"],
        per_doc_ceiling_tokens=budget["per_doc_ceiling_tokens"],
    )
    logger.debug(
        "[ENSURE] dynamic budget: docs=%d cache_safe=%d target=%d per_doc_ceiling=%d toks0=%s",
        n_docs,
        cache_safe,
        budget["target_total_tokens"],
        budget["per_doc_ceiling_tokens"],
        str(toks0),
    )

    compressed = compress_with_memo(
        parts=parts,
        model_for_counting="gemini-2.5-pro",
        limiter=limiter,
        settings=settings,
        name_to_source_key=name_to_source_key,
        count_tokens_fn=_cache_aware_count_tokens,
        eval_criteria_text=eval_criteria_text,
    )
    toks = _count_tokens_cache_pro(compressed)
    if toks is not None and toks >= 0 and toks <= cache_safe:
        return compressed, int(toks), "compressed (dynamic loop)"

    logger.warning(
        "[ENSURE] still over cache-safe after dynamic compress (%s > %d); "
        "running emergency pass.",
        str(toks),
        cache_safe,
    )
    emergency = dc_replace(
        settings,
        hard_limit_tokens=cache_safe,
        target_total_tokens=min(int(settings.target_total_tokens), cache_safe - 10_000),
        chunk_target_tokens=50_000,
        summary_max_output_tokens=5_000,
        max_compress_passes=3,
    )
    compressed2 = compress_with_memo(
        parts=compressed,
        model_for_counting="gemini-2.5-pro",
        limiter=limiter,
        settings=emergency,
        name_to_source_key=name_to_source_key,
        count_tokens_fn=_cache_aware_count_tokens,
        eval_criteria_text=eval_criteria_text,
    )
    toks2 = _count_tokens_cache_pro(compressed2)
    if toks2 is None:
        toks2 = -1
    return compressed2, int(toks2), "emergency ensure"


def _pre_compress_rfp_parts(
    *,
    rfp_parts: list[Part],
    rfp_budget_tokens: int,
    limiter,
    rfp_name_to_source_key: dict[str, str] | None = None,
    eval_criteria_text: str | None = None,
) -> tuple[list[Part], int, str]:
    """
    Compress RFP context once and reuse it for every contractor.
    """
    from utils.llm.LLM import get_client

    logger = get_logger()
    rfp_toks = _count_tokens_cache_pro(rfp_parts)
    if rfp_toks >= 0 and rfp_toks <= rfp_budget_tokens:
        logger.debug(
            "[RFP-PRE] RFP fits budget (%d <= %d); no pre-compression needed.",
            rfp_toks,
            rfp_budget_tokens,
        )
        return rfp_parts, rfp_toks, f"rfp_ok ({rfp_toks} toks)"

    logger.debug(
        "[RFP-PRE] RFP over budget (%d > %d); starting pre-compression.",
        rfp_toks,
        rfp_budget_tokens,
    )
    n_docs = _estimate_doc_count(rfp_parts)
    budget = _derive_dynamic_budget(n_docs, cache_safe=rfp_budget_tokens)
    settings = dc_replace(
        COMPRESSION_SETTINGS,
        hard_limit_tokens=rfp_budget_tokens,
        target_total_tokens=budget["target_total_tokens"],
        per_doc_ceiling_tokens=budget["per_doc_ceiling_tokens"],
    )
    try:
        compressed = compress_if_needed_document_aware(
            prefix_parts=[],
            context_parts=list(rfp_parts),
            suffix_parts=[],
            model_for_counting="gemini-2.5-pro",
            get_client=get_client,
            count_tokens=_cache_aware_count_tokens,
            limiter=limiter,
            settings=settings,
            name_to_source_key=rfp_name_to_source_key,
            eval_criteria_text=eval_criteria_text,
        )
    except Exception:
        logger.exception("[RFP-PRE] compression raised; using original RFP parts.")
        return rfp_parts, rfp_toks, "rfp_compress_error"

    toks = _count_tokens_cache_pro(compressed)
    if toks >= 0 and toks <= rfp_budget_tokens:
        return compressed, toks, f"rfp_compressed ({rfp_toks}->{toks})"
    logger.warning(
        "[RFP-PRE] still over budget after compression (%d > %d); returning best-effort.",
        toks,
        rfp_budget_tokens,
    )
    return compressed, toks, f"rfp_best_effort ({rfp_toks}->{toks})"


def _should_cache(contents) -> bool:
    logger = get_logger()
    try:
        toks_pro = count_cached_tokens("gemini-2.5-pro", contents)
    except Exception:
        toks_pro = -1

    # Normalize possible None from count_cached_tokens (no exceptions thrown)
    if toks_pro is None:
        toks_pro = -1

    def over_cap(tok: int, cap: int) -> bool:
        # treat unknown/failed counts as "over" to force skip
        return (tok is None) or (tok < 0) or (tok > (cap - CACHE_HEADROOM))

    ok_pro = not over_cap(toks_pro, MAX_TOKENS_PRO)

    decision = "USE CACHE" if ok_pro else "SKIP CACHE"

    # Never compare None to ints in logging; only display if non-negative numbers

    pro_disp = (
        toks_pro if isinstance(toks_pro, (int, float)) and toks_pro >= 0 else "UNKNOWN"
    )

    logger.debug(
        "CACHE PREFLIGHT: pro=%s tokens=%s (cap=%d, headroom=%d) => decision=%s",

        "OK" if ok_pro else "OVER",
        pro_disp,
        MAX_TOKENS_PRO,
        CACHE_HEADROOM,
        decision,
    )

    return ok_pro


# Extracted workflow function for job queue
async def _do_analysis_workflow(
    package_id: str,
    company_id: str,
    user_name: str | None = None,
    user_id: str | None = None,
    package_name: str | None = None,
    evaluation_criteria: Dict[str, Any] | str | bytes | bytearray | None = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Extracted workflow for tech RFP analysis.
    Can be called directly by job processors.

    Args:
        package_id: Project identifier
        company_id: Company identifier
        user_name: User name (optional)
        user_id: User ID (optional)
        package_name: Package name (optional)
        progress_callback: Optional callback function(progress_dict) for progress updates

    Returns:
        Dict with 'result' and 'report' keys
    """
    set_compactor_context(package_id=package_id, company_id=company_id)

    worker_logger = pid_tool_logger(package_id, "tech_rfp_analysis")
    set_logger(
        worker_logger,
        tool_name="tech_rfp_analysis_worker",
        package_id=package_id or "unknown",
        ip_address="no_ip",
        request_type="WORKER",
    )
    logger = get_logger()

    # Slack logger
    try:
        from utils.core.slack import SlackActivityLogger, SlackActivityMeta

        wact = SlackActivityLogger(
            SlackActivityMeta(
                package_id=package_id,
                tool="TECH-RFP",
                user=user_name,
                company=company_id,
            )
        )
        parent_ts = wact.start()
        wact.sub("🛠️ Starting Technical RFP evaluation")
    except Exception:
        wact = None
        parent_ts = None
        logger.debug("Slack logger not available, continuing without it")

    progress_lock = asyncio.Lock()
    start_t = time.perf_counter()

    try:
        if progress_callback:
            progress_callback({"stage": "loading", "message": "Loading documents"})

        # Download / locate files
        rfp_source_dir = "tech_rfp/rfp/"
        tender_source_dir = "tech_rfp/tender/"

        contractor_names = list_subdirectories(
            package_id, tender_source_dir, company_id
        )
        if not contractor_names:
            raise RuntimeError(
                f"No contractor tender directories found in '{tender_source_dir}'"
            )

        all_doc_metas = []
        rfp_file_keys = list_files(package_id, rfp_source_dir, company_id)
        for rfp_key in rfp_file_keys:
            all_doc_metas.append(
                {
                    "name": os.path.basename(rfp_key),
                    "source_key": rfp_key,
                    "context": "rfp",
                }
            )

        for name in contractor_names:
            contractor_dir_key = f"{tender_source_dir}{name}/"
            tender_file_keys = list_files(package_id, contractor_dir_key, company_id)
            for tender_key in tender_file_keys:
                all_doc_metas.append(
                    {
                        "name": os.path.basename(tender_key),
                        "source_key": tender_key,
                        "context": "tender",
                        "contractor": name,
                    }
                )

        doc_index: list[dict] = []
        for meta in all_doc_metas:
            source_key = meta["source_key"]  # full key: company/project/tech_rfp/...
            public_url = build_public_url_for_key(source_key)

            doc_index.append(
                {
                    "display": meta["name"],  # filename the LLM will say
                    "location": source_key,  # full key in the bucket
                    "url": public_url,  # sharable link
                    "context": meta["context"],  # 'rfp' or 'tender'
                    "contractor": meta.get("contractor"),
                }
            )

        if evaluation_criteria is None:
            raise RuntimeError(
                "Missing evaluation_criteria in job payload for tech_rfp_analysis."
            )

        if isinstance(evaluation_criteria, (bytes, bytearray)):
            structure = json.loads(evaluation_criteria.decode("utf-8"))
        elif isinstance(evaluation_criteria, str):
            structure = json.loads(evaluation_criteria)
        elif isinstance(evaluation_criteria, dict):
            structure = evaluation_criteria
        else:
            raise ValueError(
                "Unexpected type for evaluation_criteria in job payload: "
                f"{type(evaluation_criteria).__name__}"
            )

        flat_criteria, name_map, desc_map = extract_criteria_from_structure(structure)
        eval_criteria_text = format_eval_criteria_for_summary(structure)

        # Progress trackers
        criteria_per_contractor = len(flat_criteria)
        total_contractors = len(contractor_names)
        finished_for_current = 0
        contractor_index = 0

        progress = _mk_progress(
            current_contractor=(contractor_names[0] if contractor_names else None),
            finished_for_current=0,
            total_criteria_per_contractor=criteria_per_contractor,
            contractor_index=0,
            total_contractors=total_contractors,
            contractor_names=contractor_names,
        )
        if progress_callback:
            progress_callback(progress)

        logger.debug(f"Loading a total of {len(all_doc_metas)} documents...")
        all_parts, aux = await get_documents(
            package_id, company_id, all_doc_metas, force_refresh=False
        )
        logger.debug("All documents loaded successfully.")

        aux = aux or {}
        parts_by_source_key = aux.get("parts_by_source_key", {})
        not_processed_by_source_key = (
            aux.get("not_processed_by_source_key", {}) or {}
        )
        parts_map: Dict[str, list] = {m["source_key"]: [] for m in all_doc_metas}
        parts_map.update(
            {k: v for k, v in parts_by_source_key.items() if k in parts_map}
        )

        empties = [
            m["source_key"] for m in all_doc_metas if not parts_map.get(m["source_key"])
        ]
        if empties:
            logger.warning(
                "No parts loaded for %d files (first few): %s",
                len(empties),
                [os.path.basename(x) for x in empties[:5]],
            )

        rfp_parts = []
        for meta in all_doc_metas:
            if meta["context"] == "rfp":
                rfp_parts.extend(parts_map.get(meta["source_key"], []))

        # Shared map: normalized file name -> source_key for all docs.
        name_to_keys: dict[str, set[str]] = {}
        for m in all_doc_metas:
            nm = _norm_name(m.get("name"))
            sk = (m.get("source_key") or "").strip()
            if nm and sk:
                name_to_keys.setdefault(nm, set()).add(sk)
        shared_name_to_source_key: dict[str, str] = {
            nm: next(iter(ks)) for nm, ks in name_to_keys.items() if len(ks) == 1
        }

        # RFP-only subset for pre-compression
        rfp_name_to_source_key: dict[str, str] = {}
        for m in all_doc_metas:
            if m["context"] == "rfp":
                nm = _norm_name(m.get("name"))
                sk = (m.get("source_key") or "").strip()
                if nm and sk:
                    rfp_name_to_source_key[nm] = sk

        limiter = get_global_limiter()
        cache_safe = _cache_safe_tokens()
        contractor_min_reserve = int(cache_safe * 0.70)
        largest_contractor_toks = 0
        for cname in contractor_names:
            c_parts: list[Part] = []
            for m in all_doc_metas:
                if m.get("contractor") == cname:
                    c_parts.extend(parts_map.get(m["source_key"], []))
            if c_parts:
                c_toks = sum(len(getattr(p, "text", "") or "") // 4 for p in c_parts)
                if c_toks > largest_contractor_toks:
                    largest_contractor_toks = c_toks

        contractor_need = max(contractor_min_reserve, int(largest_contractor_toks * 1.10))
        rfp_floor = int(cache_safe * 0.10)
        rfp_budget = max(rfp_floor, cache_safe - contractor_need)

        if wact:
            wact.sub("📄 Measuring RFP size...")
        try:
            compacted_rfp_parts, rfp_toks, rfp_note = _pre_compress_rfp_parts(
                rfp_parts=rfp_parts,
                rfp_budget_tokens=rfp_budget,
                limiter=limiter,
                rfp_name_to_source_key=rfp_name_to_source_key,
                eval_criteria_text=eval_criteria_text,
            )
        except Exception:
            logger.exception("[RFP-PRE] pre-compression failed; using original RFP parts.")
            compacted_rfp_parts = rfp_parts
            rfp_toks = -1
            rfp_note = "rfp_precompress_error"
        rfp_was_compressed = compacted_rfp_parts is not rfp_parts
        if wact:
            rfp_pct = int(100 * rfp_budget / max(1, cache_safe))
            wact.sub(
                f"📄 RFP: {rfp_note}"
                f" | budget={rfp_budget} ({rfp_pct}% of ceiling)"
                + (
                    f" | largest contractor≈{largest_contractor_toks} toks"
                    if largest_contractor_toks > 0
                    else ""
                )
            )

        # Loop over contractors
        tender_results = []
        for idx, contractor_name in enumerate(contractor_names, start=1):
            contractor_index = idx - 1
            finished_for_current = 0
            # heartbeat at contractor start
            progress = _mk_progress(
                current_contractor=contractor_name,
                finished_for_current=0,
                total_criteria_per_contractor=criteria_per_contractor,
                contractor_index=contractor_index,
                total_contractors=total_contractors,
                contractor_names=contractor_names,
            )
            if progress_callback:
                progress_callback(progress)
            logger.debug(
                f"Processing contractor {contractor_name} ({idx}/{len(contractor_names)})"
            )
            if wact:
                wact.sub(f"🧩 Preparing context for {contractor_name} ({idx}/{len(contractor_names)}).")

            proposal_parts = []
            proposal_file_names = []
            # Get metas for just this contractor
            current_contractor_metas = [
                m for m in all_doc_metas if m.get("contractor") == contractor_name
            ]
            for meta in current_contractor_metas:
                proposal_parts.extend(parts_map.get(meta["source_key"], []))
                proposal_file_names.append(meta["name"])

            if not proposal_parts:
                logger.warning(
                    f"No document parts found for contractor {contractor_name}. Skipping."
                )
                continue

            submitted_not_processed: list[str] = []
            seen_missing = set()
            for meta in current_contractor_metas:
                for fn in (
                    not_processed_by_source_key.get(meta["source_key"], []) or []
                ):
                    fn = str(fn).strip()
                    if fn and fn not in seen_missing:
                        seen_missing.add(fn)
                        submitted_not_processed.append(fn)

            # Build combined contents (RFP + proposal) ONCE per contractor
            combined_context_parts = build_cached_context_parts(
                compacted_rfp_parts, proposal_parts
            )
            pre = preflight_inline_sizes(combined_context_parts)
            must_spill = pre["inline_total"] > INLINE_BODY_BUDGET or pre[
                "max_part"
            ] > max(TEXT_PART_LIMIT, MEDIA_PART_LIMIT)

            spilled_parts, used_ctx_prefix, ctx_manifest = (
                load_or_build_contractor_context(
                    package_id=package_id,
                    company_id=company_id,
                    contractor_name=contractor_name,
                    rfp_parts=compacted_rfp_parts,
                    proposal_parts=proposal_parts,
                    bucket=secrets.get("GCS_BUCKET", default="") or None,
                    force_rebuild=rfp_was_compressed,
                )
            )

            pre_contents = make_composite_contents(spilled_parts)
            try:
                tokens_before = count_cached_tokens("gemini-2.5-pro", pre_contents)
            except Exception:
                # If counting fails, force compression for safety
                tokens_before = -1

            cache_safe = _cache_safe_tokens()
            need_compress = (
                (tokens_before is None)
                or (tokens_before < 0)
                or (tokens_before > cache_safe)
            )
            delta_note = ""
            parts_for_eval = spilled_parts
            limiter = get_global_limiter()
            header_text = None
            pinned_header_part = None
            if submitted_not_processed:
                lines = "\n".join(f"- {fn}" for fn in submitted_not_processed[:200])
                extra = ""
                if len(submitted_not_processed) > 200:
                    extra = (
                        f"\n- ...and {len(submitted_not_processed) - 200} more"
                    )
                header_text = (
                    "### SUBMITTED BUT NOT PROCESSED FILES\n"
                    "The following files were submitted but could not be processed/extracted:\n"
                    f"{lines}{extra}\n"
                )
                pinned_header_part = Part.from_text(text=header_text)
                parts_for_eval = [pinned_header_part] + list(parts_for_eval)

            if need_compress:
                if wact:
                    wact.sub(
                        "✂️ Running one-time marker-based compression for this contractor."
                    )

                # Ensure cache-safe size with dynamic budget and emergency fallback.
                compressed_parts, ensured_tokens, ensure_note = ensure_cacheable_parts(
                    parts=spilled_parts,
                    limiter=limiter,
                    name_to_source_key=shared_name_to_source_key,
                    eval_criteria_text=eval_criteria_text,
                )
                parts_for_eval = compressed_parts
                if pinned_header_part is not None:
                    first_text = (
                        getattr(parts_for_eval[0], "text", None)
                        if parts_for_eval
                        else None
                    )
                    if first_text != header_text:
                        parts_for_eval = [pinned_header_part] + list(parts_for_eval)
                total_tokens = ensured_tokens
                logger.debug(
                    "[ENSURE] contractor=%s note=%s tokens=%s cache_safe=%d",
                    contractor_name,
                    ensure_note,
                    str(ensured_tokens),
                    cache_safe,
                )
                if (
                    isinstance(tokens_before, (int, float))
                    and tokens_before >= 0
                    and isinstance(total_tokens, (int, float))
                    and total_tokens >= 0
                ):
                    delta_note = f"; compressed from {tokens_before}"
            else:
                total_tokens = tokens_before
                logger.debug(
                    "Compression skipped: token budget fits (before=%s, cache_safe=%d).",
                    str(tokens_before),
                    cache_safe,
                )

            # Always share a single token message
            if isinstance(total_tokens, (int, float)) and total_tokens >= 0:
                if wact:
                    wact.sub(
                        f"✅ Token budget: {total_tokens} (target ≤ {cache_safe}){delta_note}."
                    )
            else:
                if wact:
                    wact.sub(f"✅ Token budget: UNKNOWN (target ≤ {cache_safe}).")

            # cache creation
            pro_cache_name = None
            use_caching = False
            min_needed = _min_cache_tokens("gemini-2.5-pro")

            if (
                cache_enabled()
                and isinstance(total_tokens, (int, float))
                and total_tokens >= min_needed
            ):
                contents = make_composite_contents(parts_for_eval)
                logger.debug(
                    "CACHE PREFLIGHT (initial) for contractor=%s",
                    contractor_name,
                )
                ok_for_cache = _should_cache(contents)
                # Preflight tokens for the exact cache models; skip caching if too big
                if not ok_for_cache:
                    logger.debug(
                        "CACHE DECISION (initial): SKIP (token cap hit on at least one model)."
                    )
                    if wact:
                        try:
                            wact.sub(
                                f"⚠️ {contractor_name}: "
                                f"cache preflight: over token cap for cache model(s); skipping caching."
                            )
                        except Exception:
                            logger.debug("Slack sub failed (non-fatal).", exc_info=True)
                        try:
                            wact.sub(
                                f"⚠️ {contractor_name}: still fails cache preflight after tightening; proceeding uncached with cacheable parts."
                            )
                        except Exception:
                            logger.debug("Slack sub failed (non-fatal).", exc_info=True)
                    use_caching = False
                else:
                    # 3) Create both caches; on size error, force-spill all text and retry once
                    try:
                        pro_cache_name = create_cache(
                            model="gemini-2.5-pro",
                            contents=contents,
                            display=f"{package_id}:{contractor_name}:pro",
                            system_instruction=SENIOR_SYS_PROMPT,
                        )
                        use_caching = bool(pro_cache_name)
                        logger.debug(
                            "CACHE CREATE success: pro=%s use_caching=%s",
                            pro_cache_name,
                            use_caching,
                        )
                        if wact and use_caching:
                            wact.sub(f"🧠 Cache created for {contractor_name}.")

                    except ClientError as e:
                        msg = (getattr(e, "response_json", {}) or {}).get(
                            "error", {}
                        ).get("message", "") or str(e)
                        is_size_limit = "exceeds the data size limit" in msg
                        is_token_limit = (
                            "input token count is" in msg
                            and "only supports up to" in msg
                        )

                        if is_token_limit:
                            # Spilling won't reduce tokens; we already spilled. Skip caching.
                            logger.debug(
                                "CACHE CREATE: TOKEN_LIMIT for %s; skipping caching. Detail=%s",
                                contractor_name,
                                msg,
                            )
                            use_caching = False
                            if wact:
                                wact.sub(
                                    f"⚠️ {contractor_name}: cache create hit token limit; proceeding uncached."
                                )
                        elif is_size_limit:
                            # We already use URIs, so inline body should be small; rarely hit here. Skip if it happens.
                            logger.debug(
                                "CACHE CREATE: SIZE_LIMIT for %s even with URIs; skipping caching. Detail=%s",
                                contractor_name,
                                msg,
                            )
                            use_caching = False
                            if wact:
                                wact.sub(
                                    f"⚠️ {contractor_name}: cache create hit size limit; proceeding uncached."
                                )
                        else:
                            raise

            else:
                logger.debug(
                    f"cache skip: total tokens {total_tokens} < {min_needed} (no caching for {contractor_name})"
                )

            try:

                async def _criterion_done_hook():
                    nonlocal finished_for_current, contractor_index
                    async with progress_lock:
                        finished_for_current += 1
                        pr = _mk_progress(
                            current_contractor=contractor_name,
                            finished_for_current=finished_for_current,
                            total_criteria_per_contractor=criteria_per_contractor,
                            contractor_index=contractor_index,
                            total_contractors=total_contractors,
                            contractor_names=contractor_names,
                        )
                    if progress_callback:
                        progress_callback(pr)
                    if wact:
                        try:
                            wact.sub(
                                f"✅ {contractor_name}: "
                                f"{finished_for_current}/{criteria_per_contractor} criteria done "
                                f"(overall {int(pr['overallPercentageCompletion']*100)}%)."
                            )
                        except Exception:
                            logger.debug("Slack sub failed (non-fatal).", exc_info=True)

                contractor_result = await score_proposal_main(
                    package_id=package_id,
                    proposal_parts=proposal_parts,
                    rfp_parts=compacted_rfp_parts,
                    evaluation_criteria=flat_criteria,
                    name_map=name_map,
                    contractor_name=contractor_name,
                    proposal_file_names=proposal_file_names,
                    description_map=desc_map,
                    pro_cache_name=pro_cache_name,
                    act=wact,
                    spill_prefix=used_ctx_prefix,
                    prepared_context_parts=parts_for_eval,
                    progress_hook=_criterion_done_hook,
                )

            finally:
                if cache_enabled() and use_caching:
                    delete_cache(pro_cache_name)

            tender_results.append(contractor_result)

            if wact:
                wact.sub(
                    f"✅ Finished {contractor_name} ({idx}/{len(contractor_names)})."
                )

        # Scale & save
        scaled = scale_score(
            data_structure=structure,
            evaluations=tender_results,
            name_map=name_map,
            doc_index=doc_index,
        )

        # 6. Technical report
        tech_data = await build_technical_report(
            rfp_result_data=scaled,
            evaluation_criteria=structure,
        )

        total_dur = fmt_dur(time.perf_counter() - start_t)
        if wact:
            wact.sub(
                f"🏁 Finished processing project in {total_dur} "
                f"({len(contractor_names)} contractors)."
            )
            wact.done()

        logger.info(f"Technical evaluation completed in {total_dur}.")
        progress = _mk_progress(
            current_contractor=contractor_name,
            finished_for_current=criteria_per_contractor,
            total_criteria_per_contractor=criteria_per_contractor,
            contractor_index=contractor_index,
            total_contractors=total_contractors,
            contractor_names=contractor_names,
        )
        if progress_callback:
            progress_callback(progress)

        # Return both result and report
        return {
            "result": scaled,
            "report": tech_data,
        }

    except Exception as exc:
        total_dur = fmt_dur(time.perf_counter() - start_t)
        logger.error(f"Technical evaluation FAILED after {total_dur}: {exc}")
        if wact:
            wact.error(f"🔥 Technical evaluation FAILED after {total_dur}.")
        raise


def smoke_test_workflow() -> bool:
    """
    Smoke-test the end-to-end analysis workflow with in-memory stubs.
    Validates that `_do_analysis_workflow` consumes payload-provided
    `evaluation_criteria` and produces both `result` and `report`.
    """
    import asyncio

    set_logger(pid_tool_logger("SYSTEM_CHECK", "tech_rfp_workflow"))
    print("TECH_RFP_WORKFLOW TEST START")

    patch_names = [
        "list_subdirectories",
        "list_files",
        "build_public_url_for_key",
        "get_documents",
        "load_or_build_contractor_context",
        "score_proposal_main",
        "scale_score",
        "build_technical_report",
        "count_cached_tokens",
        "cache_enabled",
    ]
    originals = {name: globals().get(name) for name in patch_names}

    try:
        rfp_key = "tech_rfp/rfp/rfp.pdf"
        tender_key = "tech_rfp/tender/Dummy Contractor/proposal.pdf"

        def _fake_list_subdirectories(package_id, prefix, company_id):
            if prefix == "tech_rfp/tender/":
                return ["Dummy Contractor"]
            return []

        def _fake_list_files(package_id, prefix, company_id):
            if prefix == "tech_rfp/rfp/":
                return [rfp_key]
            if prefix == "tech_rfp/tender/Dummy Contractor/":
                return [tender_key]
            return []

        def _fake_build_public_url_for_key(source_key):
            return f"https://example.test/{source_key}"

        async def _fake_get_documents(package_id, company_id, metas, force_refresh=False):
            parts_by_source_key = {}
            for meta in metas:
                source_key = meta.get("source_key")
                if source_key == rfp_key:
                    parts_by_source_key[source_key] = [
                        Part.from_text(text="RFP text for workflow smoke test.")
                    ]
                elif source_key == tender_key:
                    parts_by_source_key[source_key] = [
                        Part.from_text(text="Proposal text for workflow smoke test.")
                    ]
                else:
                    parts_by_source_key[source_key] = []
            return [], {"parts_by_source_key": parts_by_source_key}

        def _fake_load_or_build_contractor_context(
            package_id,
            company_id,
            contractor_name,
            rfp_parts,
            proposal_parts,
            bucket=None,
            force_rebuild=False,
        ):
            combined = list(rfp_parts or []) + list(proposal_parts or [])
            return combined, "smoke/context/prefix", {"ok": True}

        async def _fake_score_proposal_main(
            package_id,
            proposal_parts,
            rfp_parts,
            evaluation_criteria,
            name_map,
            contractor_name,
            proposal_file_names,
            **_,
        ):
            tender_eval = {}
            for criterion_lc in evaluation_criteria.keys():
                tender_eval[criterion_lc] = {
                    "score": 8.0,
                    "reasoning": [{"text": "stub reasoning"}],
                    "ptc": {"queryDescription": "NA", "refType": "N/A"},
                    "clientSummary": "stub summary",
                }
            return {
                "contractorName": contractor_name,
                "tenderEvaluation": tender_eval,
                "raw_llm_responses": [],
                "file_names": list(proposal_file_names),
                "documentProcessingErrors": [],
            }

        def _fake_scale_score(data_structure, evaluations, name_map, doc_index=None):
            contractor = evaluations[0] if evaluations else {}
            return {
                "result": {
                    "tenderReport": [
                        {
                            "contractorName": contractor.get(
                                "contractorName", "Dummy Contractor"
                            ),
                            "tenderEvaluation": {"scopes": [], "totalScore": 80.0},
                            "documentProcessingErrors": contractor.get(
                                "documentProcessingErrors", []
                            ),
                            "raw_llm_responses": contractor.get("raw_llm_responses", []),
                        }
                    ]
                }
            }

        async def _fake_build_technical_report(*, rfp_result_data, evaluation_criteria):
            return {
                "executive_summary": "stub summary",
                "contractorEvaluations": [],
                "evaluationScopes": {"scopesCount": len(evaluation_criteria or {})},
            }

        globals()["list_subdirectories"] = _fake_list_subdirectories
        globals()["list_files"] = _fake_list_files
        globals()["build_public_url_for_key"] = _fake_build_public_url_for_key
        globals()["get_documents"] = _fake_get_documents
        globals()["load_or_build_contractor_context"] = _fake_load_or_build_contractor_context
        globals()["score_proposal_main"] = _fake_score_proposal_main
        globals()["scale_score"] = _fake_scale_score
        globals()["build_technical_report"] = _fake_build_technical_report
        globals()["count_cached_tokens"] = lambda *args, **kwargs: 1000
        globals()["cache_enabled"] = lambda: False

        payload_criteria = {
            "Technical": {
                "Experience": {"weight": 60.0, "description": "Relevant track record"},
                "Cost": {"weight": 40.0, "description": "Commercial value"},
            }
        }

        async def _run():
            return await _do_analysis_workflow(
                package_id="SELFTEST",
                company_id="SELFTEST",
                evaluation_criteria=payload_criteria,
                progress_callback=None,
            )

        out = asyncio.run(_run())
        ok = (
            isinstance(out, dict)
            and isinstance(out.get("result"), dict)
            and isinstance(out.get("report"), dict)
            and "tenderReport" in (out.get("result", {}).get("result", {}))
        )
        print("TECH_RFP_WORKFLOW OK" if ok else "TECH_RFP_WORKFLOW FAIL")
        return ok

    except Exception as e:
        print(f"TECH_RFP_WORKFLOW ERROR: {e}")
        return False
    finally:
        for name, original in originals.items():
            if original is not None:
                globals()[name] = original


def _test_pre_compress_rfp() -> bool:
    """
    Legacy compatibility hook.
    The dedicated regression coverage now lives in engine/tests/test_tech_rfp_workflows.py.
    """
    return True


def main() -> bool:
    """Legacy CLI entrypoint retained for compatibility."""
    return smoke_test_workflow()


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
