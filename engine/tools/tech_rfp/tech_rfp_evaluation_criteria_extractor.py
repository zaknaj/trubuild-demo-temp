"""
TruBuild - Evaluation-Criteria Spreadsheet Processor

Purpose
- Convert any submitted evaluation spreadsheet / table (CSV, XLS/XLSX, PDF
  converted to table, etc.) into a clean JSON structure:

    {
        "Evaluation Scope": {
            "Criterion": 5.0,
            ...
        },
        ...
    }

- If a valid JSON file is already present in the evaluation folder, re-use it.
- Handle multiple files in the evaluation directory and stop at the first
  successfully-processed spreadsheet.

Design
1. The public entry-point :func:`process_eval_spreadsheet` pulls (or opens) the
   evaluation directory.
2. If the directory already contains **one** ``.json`` file, that file is
   returned verbatim (no LLM call).
3. Otherwise, it walks every supported spreadsheet in alphabetical order until
   the first one is parsed successfully.
4. We keep the original *datetime -> ISO-string* helper so DataFrame dumps never
   choke on ``Timestamp`` objects.

"""

from __future__ import annotations

import os
import re
import json
import asyncio
import datetime
import threading
from typing import Any, Dict, List
import pandas.api.types as ptypes

import pandas as pd
from google.genai.types import Part
from utils.document.docingest import get_documents
from utils.llm.LLM import call_llm_sync, call_llm_async
from utils.storage.bucket import (
    list_files,
)
from utils.core.log import pid_tool_logger, set_logger, get_logger
from tools.tech_rfp.prompts_tech_rfp import build_eval_extractor_prompt

_PROMPT_HEADER = build_eval_extractor_prompt()
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_RETRY_TEMPS = [0.0, 0.1, 0.3, 0.5]
_RETRY_ATTEMPTS = len(_RETRY_TEMPS)
_RETRY_BASE_DELAY = 0.2  # seconds
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-pro"

PDF_GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 1,
    "max_output_tokens": 20000,
    "thinking_config": {"thinking_budget": 0},
    "response_mime_type": "application/json",
    "seed": 12345,
}

PDF_SYSTEM_INSTRUCTIONS = """
You are an expert at extracting evaluation criteria from RFP and tender documents.

Your task is to:
1. Read through the entire PDF document carefully
2. Identify all evaluation criteria, requirements, and scoring elements
3. Organize them into logical categories/scopes
4. Extract weights, percentages, or point values where available
5. Handle bilingual content (Arabic/English)
6. Return a well-structured JSON format suitable for contractor evaluation

Focus on creating a professional evaluation matrix that procurement teams would use to assess contractor qualifications and proposals.
"""
EVALUATION_EXTRACTION_PROMPT = """
Extract ALL evaluation criteria from this technical RFP/tender PDF document.

For each evaluation criterion found, extract:
- Evaluation scope/category (e.g., "Documentation", "Technical", "Experience", etc.)
- Criterion name/description
- Weight/percentage (if specified)
- Any additional requirements or details

Return a structured JSON format like:
{
  "Documentation": {
    "Tender Documents Submission": {
      "weight": 5.0,
      "description": "Complete tender documentation as required"
    },
    "Required Licenses": {
      "weight": 3.0,
      "description": "Valid licenses and certifications"
    }
  },
  "Technical Capability": {
    "Previous Similar Projects": {
      "weight": 15.0,
      "description": "Experience with similar project types and scale"
    }
  }
}

Extract from both Arabic and English text, but make sure the final output is ONLY in english. Focus on:
1. Technical evaluation criteria
2. Document requirements with weights/percentages
3. Experience and qualification requirements
4. Any scoring matrices or evaluation tables

Be comprehensive - extract all evaluation criteria you can find, even if weights are not specified.
"""


# LLM helper
async def _ask_LLM(
    prompt: str, cfg_override: dict[str, Any] = None, *, model: str = PRIMARY_MODEL
) -> Any:

    cfg: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1.0,
        "max_output_tokens": 20000,
        "thinking_config": {"thinking_budget": 0},
        "response_mime_type": "application/json",
        "seed": 12345,
    }
    if cfg_override:
        cfg.update(cfg_override)
    prompt_parts = [Part.from_text(text=prompt)]
    if asyncio.get_event_loop().is_running():
        raw = await call_llm_async(
            prompt_parts=prompt_parts,
            model=model,
            cfg=cfg,
            system_instruction=(
                "You are an expert who reviews files and extracts the evaluation "
                "criteria as it is. Do not wrap the result in an array or list. The "
                "output must strictly follow the defined JSON schema."
            ),
        )
    else:
        raw = call_llm_sync(
            prompt_parts=prompt_parts,
            model=model,
            cfg=cfg,
            system_instruction=(
                "You are an expert who reviews files and extracts the evaluation "
                "criteria as it is. Do not wrap the result in an array or list. The "
                "output must strictly follow the defined JSON schema."
            ),
        )

    return raw


async def _ask_LLM_with_retries(
    prompt: str,
    attempts: int = _RETRY_ATTEMPTS,
    temps: list[float] = _RETRY_TEMPS,
    base_delay: float = _RETRY_BASE_DELAY,
) -> Dict[str, Any]:
    logger = get_logger()
    last_err: Exception | None = None

    phases = [
        {"model": PRIMARY_MODEL, "attempts": attempts, "temps": temps},
        {"model": FALLBACK_MODEL, "attempts": 2, "temps": [0.1, 0.3]},
    ]

    for phase in phases:
        model = phase["model"]
        atts = phase["attempts"]
        tlist = phase["temps"]

        logger.debug(f"LLM phase start: model={model}, attempts={atts}")
        for i in range(atts):
            t = tlist[i] if i < len(tlist) else tlist[-1]
            try:
                raw = await _ask_LLM(
                    prompt, cfg_override={"temperature": t}, model=model
                )
                logger.debug(
                    "LLM raw response (attempt %d, temp=%.2f):\n%s", i + 1, t, raw
                )

                def _dedupe_pairs(pairs):
                    out = {}
                    counts = {}
                    for k, v in pairs:
                        if k in out:
                            counts[k] = counts.get(k, 1) + 1
                            k2 = f"{k} ({counts[k]})"
                            while k2 in out:
                                counts[k] += 1
                                k2 = f"{k} ({counts[k]})"
                            out[k2] = v
                        else:
                            out[k] = v
                            counts[k] = 1
                    return out

                parsed = json.loads(raw, object_pairs_hook=_dedupe_pairs)
                logger.debug(f"LLM parse OK (model={model}, attempt={i+1}, temp={t})")
                return parsed
            except Exception as e:
                last_err = e
                delay = base_delay * (2**i)
                jitter = (i % 3) * 0.03
                logger.debug(
                    f"LLM attempt failed (model={model}, attempt={i+1}/{atts}, temp={t}): {e}. "
                    f"Retrying in {delay + jitter:.2f}s"
                )
                await asyncio.sleep(delay + jitter)
        logger.debug(f"LLM phase exhausted: model={model}")
    raise last_err or RuntimeError(
        "LLM retries exhausted across primary+fallback models"
    )


# TODO: doesn't follow protocol
# LLM helper for PDFs (ADD AFTER _ask_LLM_with_retries)
async def _extract_from_pdf_bytes(
    pdf_parts: List[Part],
    file_name: str,
    attempts: int = _RETRY_ATTEMPTS,
    base_delay: float = _RETRY_BASE_DELAY,
) -> Dict[str, Any]:
    """Extract evaluation criteria from PDF bytes using direct LLM call."""
    logger = get_logger()

    # Build prompt parts: text prompt + PDF bytes
    prompt_parts = [Part.from_text(text=EVALUATION_EXTRACTION_PROMPT)] + pdf_parts

    last_err: Exception | None = None
    temps = [0.2, 0.3, 0.4, 0.5]

    for i in range(attempts):
        t = temps[i] if i < len(temps) else temps[-1]
        try:
            logger.debug(f"PDF extraction attempt {i+1} for {file_name} at temp={t}")

            cfg = PDF_GENERATION_CONFIG.copy()
            cfg["temperature"] = t

            raw = await call_llm_async(
                prompt_parts=prompt_parts,
                model="gemini-2.5-flash",
                system_instruction=PDF_SYSTEM_INSTRUCTIONS,
                cfg=cfg,
            )

            if not raw:
                raise ValueError("Empty response from LLM")

            # Clean up response
            response_text = str(raw).strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                first_newline = response_text.find("\n")
                if first_newline != -1:
                    response_text = response_text[first_newline + 1 :]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            parsed = json.loads(response_text)

            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed)}")

            logger.debug(f"PDF extraction OK on attempt {i+1} for {file_name}")
            return parsed

        except Exception as e:
            last_err = e
            delay = base_delay * (2**i)
            jitter = (i % 3) * 0.03
            logger.warning(
                f"PDF extraction attempt {i+1}/{attempts} failed for {file_name}: {e}. "
                f"Retrying in {delay + jitter:.2f}s"
            )
            await asyncio.sleep(delay + jitter)

    raise last_err or RuntimeError(f"PDF extraction retries exhausted for {file_name}")


def make_json_serializable(obj: Any):  # noqa: ANN401 - generic helper
    """Convert datetime/date objects so ``json.dumps`` does not fail."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return str(obj)


# Helper function to load into dataframe
def _load_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Recognizes the file type (CSV, XLS, XLSX) and loads it into a pandas DataFrame.

    Args:
        file_path (str): The path to the file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the file type is not supported or the file cannot be loaded.
    """
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)

    # Load the file based on its extension
    try:
        if file_extension.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_extension.lower() in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            has_datetime = any(
                ptypes.is_datetime64_any_dtype(df[col]) for col in df.columns
            )
            if has_datetime:
                return df
            else:
                try:
                    json_data = df.fillna("").to_dict(orient="records")
                    return json_data
                except Exception as e:
                    return df
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def _ffill_scope(
    rows,
    scope_keys=("Evaluation Scope", "evaluation_scope", "scope", "section", "theme"),
):
    last = None
    for r in rows:
        # find first present scope key
        for k in scope_keys:
            if k in r and str(r[k]).strip():
                last = r[k]
                break
        else:
            # no scope value in this row -> fill it
            if last is not None:
                for k in scope_keys:
                    if k in r and not str(r[k]).strip():
                        r[k] = last
    return rows


def _parse_weight_value(raw) -> tuple[float, bool] | None:
    """
    Returns a tuple of (numeric_value, had_percent_literal).
    Handles Arabic-Indic digits and common punctuation.
    """
    if isinstance(raw, (int, float)):
        return (float(raw), False)  # Return tuple
    s = str(raw).strip().translate(_ARABIC_DIGITS)
    # remember if the string had an explicit percent sign
    is_percent_literal = "%" in s
    # normalize decimal comma
    s = s.replace(",", ".")
    # keep digits, sign, and dot only
    s = re.sub(r"[^\d.\-]", "", s)
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    # If the original had a %, it's already percent-scale
    return (v, True) if is_percent_literal else (v, False)


def _to_percent(v: float, had_percent_literal: bool) -> float:
    """
    Convert v to percent scale (0..100) using simple heuristics:
    - explicit '%' => already percent
    - <= 1.0       => fraction, multiply by 100
    - > 100 and looks like double-scaled (e.g., 2000) => divide by 100
    - else keep as-is
    """
    if had_percent_literal:
        return v
    if v <= 1.0:
        return v * 100.0
    # treat 100..10000 with near-multiple-of-100 as double-scaled
    if (
        v > 100.0
        and abs((v / 100.0) - round(v / 100.0)) < 1e-9
        and (v / 100.0) <= 100.0
    ):
        return v / 100.0
    return v


_SCOPE_COL_CANDIDATES = (
    "Evaluation Scope",
    "evaluation_scope",
    "scope",
    "section",
    "theme",
    "Scope",
    "Section",
    "Theme",
    "Category",
    "category",
)
_SCOPE_BY_SCOPE_THRESHOLD = 50


def _detect_scope_column(rows: List[Dict]) -> str | None:
    if not rows:
        return None
    sample_lower = {k.lower().strip(): k for k in rows[0].keys()}
    for candidate in _SCOPE_COL_CANDIDATES:
        actual = sample_lower.get(candidate.lower().strip())
        if actual is not None:
            return actual
    return None


def _group_rows_by_scope(rows: List[Dict], scope_col: str) -> Dict[str, List[Dict]]:
    filled = _ffill_scope(list(rows), scope_keys=(scope_col,))
    groups: Dict[str, List[Dict]] = {}
    for r in filled:
        val = str(r.get(scope_col, "")).strip()
        if not val:
            val = "Unscoped"
        groups.setdefault(val, []).append(r)
    return groups


async def _extract_scope_by_scope(
    rows: List[Dict], file_name: str
) -> Dict[str, Dict[str, Dict[str, Any]]] | None:
    logger = get_logger()
    scope_col = _detect_scope_column(rows)
    if scope_col is None:
        logger.warning(
            "Cannot detect scope column for scope-by-scope extraction (file=%s); "
            "falling back to full-table path.",
            file_name,
        )
        return None

    groups = _group_rows_by_scope(rows, scope_col)
    logger.debug(
        "Scope-by-scope extraction: %d scopes, %d total rows, file=%s",
        len(groups),
        len(rows),
        file_name,
    )

    merged_raw: Dict[str, Any] = {}
    for scope_name, scope_rows in groups.items():
        logger.debug("  extracting scope '%s' (%d rows)", scope_name, len(scope_rows))
        table_json = json.dumps(scope_rows, indent=2, default=make_json_serializable)
        prompt = f"{_PROMPT_HEADER}\n\nTable:\n{table_json}"
        try:
            raw = await _ask_LLM_with_retries(prompt)
        except Exception as exc:
            logger.exception(
                "  scope '%s' extraction failed (%s); skipping scope.",
                scope_name,
                exc,
            )
            continue

        if not isinstance(raw, dict):
            logger.warning("  scope '%s' returned non-dict; skipping.", scope_name)
            continue

        for s, criteria in raw.items():
            if s in merged_raw:
                if isinstance(merged_raw[s], dict) and isinstance(criteria, dict):
                    merged_raw[s].update(criteria)
                else:
                    merged_raw[s] = criteria
            else:
                merged_raw[s] = criteria

    if not merged_raw:
        logger.warning(
            "Scope-by-scope extraction produced no raw results for %s; "
            "falling back to full-table path.",
            file_name,
        )
        return None

    merged = _postprocess_eval_output(merged_raw)
    if not merged:
        logger.warning(
            "Scope-by-scope post-processing produced no results for %s; "
            "falling back to full-table path.",
            file_name,
        )
        return None

    logger.debug(
        "Scope-by-scope extraction complete: %d scopes, %d total criteria",
        len(merged),
        sum(len(v) for v in merged.values()),
    )
    return merged


def _postprocess_eval_output(raw_json: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    - Unwrap per-scope JSON strings
    - Sanitize weight text
    - Convert to float and multiply by 100 (fraction -> percent)

    Output:
      { scope: { criterion: { "weight_pct": <float>, ["description"]: "<str>" } } }
    """
    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except Exception:
            return {}

    if not isinstance(raw_json, dict):
        return {}

    parsed_all = []
    global_sum = 0.0

    for scope, block in raw_json.items():
        if isinstance(block, str):
            try:
                block = json.loads(block)
            except Exception:
                continue
        if not isinstance(block, dict):
            continue

        for crit, val in block.items():
            desc = None
            w = val
            if isinstance(val, dict):
                if "weight" in val:
                    w = val["weight"]
                if "description" in val and val["description"] is not None:
                    s = str(val["description"]).strip()
                    if s:
                        desc = s

            parsed = _parse_weight_value(w)
            if not parsed:
                continue

            w_num, had_pct = parsed
            parsed_all.append((scope, crit, w_num, had_pct, desc))
            global_sum += w_num

    if not parsed_all:
        return {}

    if 80.0 <= global_sum <= 120.0:
        mode = "percent"
    elif 0.8 <= global_sum <= 1.2:
        mode = "fraction"
    elif 8000.0 <= global_sum <= 12000.0:
        mode = "double_scaled"
    else:
        mode = "fallback"

    def _to_percent_mode(v: float, had_percent_literal: bool, mode: str) -> float:
        if had_percent_literal:
            return v
        if mode == "percent":
            return v
        if mode == "fraction":
            return v * 100.0
        if mode == "double_scaled":
            return v / 100.0
        return _to_percent(v, had_percent_literal)

    res: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scope, crit, w_num, had_pct, desc in parsed_all:
        w_final = round(_to_percent_mode(w_num, had_pct, mode), 6)
        entry: Dict[str, Any] = {"weight": w_final}
        if desc:
            entry["description"] = desc
        res.setdefault(scope, {})[crit] = entry

    return res


# public API
async def process_eval_spreadsheet(package_id: str, company_id: str) -> Dict | None:
    """
    Reads evaluation inputs from cloud via the docstore:
    - Prefer a single JSON file if present (returned as-is).
    - Otherwise, return the first successfully ingested spreadsheet table.
    """
    logger = get_logger()

    # 1) Discover candidate files (cloud)
    supported_exts = (".csv", ".xls", ".xlsx", ".json", ".pdf")
    keys = list_files(package_id, "tech_rfp/evaluation/", company_id)
    metas = [
        {
            "name": os.path.basename(k),
            "source_key": k,
        }
        for k in keys
        if os.path.splitext(k)[1].lower() in supported_exts
    ]

    if not metas:
        logger.error(
            "No supported eval files (.csv/.xls/.xlsx/.json/.pdf) under tech_rfp/evaluation/"
        )
        return None

    # 2) JIT ingest + load via the gatekeeper
    eval_tables, _ = await get_documents(
        package_id,
        company_id,
        metas,
        tool_context="tech_eval",
        force_refresh=True,
    )

    if not eval_tables:
        logger.error("Ingestion returned no evaluation tables")
        return None

    # 3) Immediate JSON reuse (single JSON present -> return as-is)
    json_metas = [m for m in metas if m["name"].lower().endswith(".json")]
    if len(json_metas) == 1:
        chosen_meta = json_metas[0]
        table = eval_tables.get(chosen_meta["source_key"])
        if isinstance(table, dict) and table:
            return table
        logger.warning(
            f"JSON file '{chosen_meta['source_key']}' was empty/invalid; falling back to spreadsheets."
        )

    # 4) Walk spreadsheets/PDFs deterministically
    all_files = sorted(
        [m for m in metas if not m["name"].lower().endswith(".json")],
        key=lambda m: m["name"],
    )

    any_failure = False
    for meta in all_files:
        file_name = meta["name"]
        source_key = meta["source_key"]
        file_ext = os.path.splitext(file_name)[1].lower()

        try:
            # PDF files: use Parts from ingestion
            if file_ext == ".pdf":
                logger.debug(f"Processing PDF file: {file_name}")

                parts_data = eval_tables.get(source_key)
                if not parts_data:
                    logger.warning(f"No Parts for {file_name}")
                    continue

                raw = await _extract_from_pdf_bytes(parts_data, file_name)
                processed = _postprocess_eval_output(raw)
                logger.info(f"Criteria extracted from PDF {file_name}")
                return processed

            # Spreadsheet files: use existing ingestion path
            else:
                logger.debug(f"Processing spreadsheet: {file_name}")
                rows = eval_tables.get(source_key) or eval_tables.get(file_name)
                if not rows:
                    logger.warning(f"No data for {file_name}")
                    continue

                if isinstance(rows, list) and len(rows) > _SCOPE_BY_SCOPE_THRESHOLD:
                    logger.debug(
                        "Table has %d rows (> %d); attempting scope-by-scope extraction.",
                        len(rows),
                        _SCOPE_BY_SCOPE_THRESHOLD,
                    )
                    scope_result = await _extract_scope_by_scope(rows, file_name)
                    if scope_result is not None:
                        logger.debug(
                            "Criteria extracted from %s (scope-by-scope)", file_name
                        )
                        return scope_result
                    logger.debug(
                        "Scope-by-scope unavailable; falling back to full-table extraction."
                    )

                table_json = json.dumps(rows, indent=2, default=make_json_serializable)
                prompt = f"{_PROMPT_HEADER}\n\nTable:\n{table_json}"

                raw = await _ask_LLM_with_retries(prompt)
                processed = _postprocess_eval_output(raw)
                logger.debug(f"Criteria extracted from {file_name}")
                return processed

        except Exception as exc:
            any_failure = True
            logger.exception("Failed to process %s: %s", file_name, exc)

    logger.error(
        "Evaluation extraction failed for all inputs%s.",
        " (at least one attempt raised)" if any_failure else "",
    )
    raise RuntimeError(
        "No evaluation results extracted from any provided evaluation files."
    )


## EVAL PATH global definition
EVAL_PATH = "data/evaluation.json"


async def eval_analysis_main(
    *,
    package_id: str | None = None,
    company_id: str | None = None,
    user_name: str | None = None,
    country_code: str = "USA",
    request_method: str | None = None,
    remote_ip: str | None = None,
    compute_reanalysis: bool = True,
    **_,
) -> Dict[str, Any]:
    """
    Entry-point for evaluation criteria extraction.
    Uses job queue system for processing.
    """
    from utils.db.job_queue import JobType
    from tools.tech_rfp.job_queue_integration import (
        create_tech_rfp_job,
        get_job_status_response,
    )

    set_logger(
        pid_tool_logger(package_id=package_id, tool_name="eval_analysis"),
        tool_name="eval_analysis_main",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()
    country_code = country_code or "USA"

    if not package_id:
        return {"error": "packageId is required", "status": "error"}

    if request_method == "GET":
        # Check job status from database
        return get_job_status_response(
            package_id=package_id,
            company_id=company_id,
            job_type=JobType.TECH_RFP_EVALUATION_EXTRACT,
        )

    if request_method == "POST":
        # Create job in queue
        if not compute_reanalysis:
            response = get_job_status_response(
                package_id=package_id,
                company_id=company_id,
                job_type=JobType.TECH_RFP_EVALUATION_EXTRACT,
            )
            if response.get("status") == "completed":
                return response

        job_id = create_tech_rfp_job(
            job_type=JobType.TECH_RFP_EVALUATION_EXTRACT,
            package_id=package_id,
            company_id=company_id,
            user_name=user_name,
            remote_ip=remote_ip,
            country_code=country_code,
        )

        logger.info(f"Created eval_analysis job {job_id} for project {package_id}")
        return {
            "status": "in progress",
            "message": f"Evaluation extraction job created for {package_id}",
            "job_id": job_id,
        }

    return {"error": f"Unsupported request method: {request_method}", "status": "error"}


# Extracted workflow function for job queue
async def _do_eval_extract_workflow(
    package_id: str,
    company_id: str,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Extracted workflow for evaluation criteria extraction.
    Can be called directly by job processors.

    Args:
        package_id: Project identifier
        company_id: Company identifier
        progress_callback: Optional callback function(progress_dict) for progress updates

    Returns:
        Evaluation criteria dict
    """
    set_logger(
        pid_tool_logger(package_id, "eval_analysis"),
        tool_name="eval_analysis_worker",
        package_id=package_id or "unknown",
        ip_address="no_ip",
        request_type="WORKER",
    )
    log = get_logger()

    if progress_callback:
        progress_callback(
            {"stage": "extracting", "message": "Extracting evaluation criteria"}
        )

    try:
        eval_res = await process_eval_spreadsheet(
            package_id=package_id, company_id=company_id
        )
        if not eval_res:
            raise RuntimeError("No evaluation results extracted")

        eval_payload = (
            json.loads(eval_res.to_json(orient="records"))
            if isinstance(eval_res, pd.DataFrame)
            else eval_res
        )

        log.info("Evaluation extraction completed")
        return eval_payload
    except Exception as exc:
        log.exception("Evaluation extraction failed: %s", exc)
        raise


async def evaluation_fetch_main(
    *,
    package_id: str | None = None,
    company_id: str | None = None,
    user_name: str | None = None,
    country_code: str = "USA",
    remote_ip: str | None = None,
    **_,
) -> Dict[str, Any]:
    """
    Fetch evaluation results - now uses job queue.
    """
    from utils.db.job_queue import JobType
    from tools.tech_rfp.job_queue_integration import get_job_status_response

    set_logger(
        pid_tool_logger(package_id=package_id, tool_name="evaluation_fetch"),
        tool_name="evaluation_fetch_main",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type="GET",
    )
    logger = get_logger()
    if not package_id:
        return {"error": "packageId is required", "status": "error"}

    return get_job_status_response(
        package_id=package_id,
        company_id=company_id,
        job_type=JobType.TECH_RFP_EVALUATION_EXTRACT,
    )


def main() -> bool:
    """
    Lightweight self-test for the evaluation-criteria extractor:
      - Fakes cloud listing and document ingestion
      - Stubs LLM to return deterministic JSON
      - Runs process_eval_spreadsheet() and sanity-checks the result
      - Prints OK/FAIL and returns a boolean
    """
    import asyncio

    set_logger(
        pid_tool_logger("SYSTEM_CHECK", "tech_rfp_evaluation_criteria_extractor")
    )
    print("TECH_RFP_EVALUATION_CRITERIA_EXTRACTOR TEST START")
    try:
        # Stubs to avoid I/O and network
        fake_keys = ["tech_rfp/evaluation/test_eval.csv"]

        def _fake_list_files(package_id, prefix, company_id):
            # Simulate one spreadsheet in the evaluation dir
            return list(fake_keys)

        async def _fake_get_documents(package_id, company_id, metas, tool_context=None):
            # Simulate ingestion result: rows per file name & source_key
            rows = [
                {"Evaluation Scope": "Technical", "Criterion": "Design", "Weight": 0.4},
                {
                    "Evaluation Scope": "Technical",
                    "Criterion": "Methodology",
                    "Weight": 0.6,
                },
                {
                    "Evaluation Scope": "Commercial",
                    "Criterion": "Pricing",
                    "Weight": 1.0,
                },
            ]
            table_by = {}
            for m in metas:
                table_by[m["name"]] = rows
                table_by[m["source_key"]] = rows
            # Return structure compatible with get_documents()
            return table_by, {}

        async def _fake_ask_llm(prompt: str):
            # Return JSON shaped like the extractor expects; weights are FRACTIONS
            return {
                "Technical": {"Design": 0.4, "Methodology": 0.6},
                "Commercial": {"Pricing": 1.0},
            }

        # Monkey-patch the names this module uses
        globals()["list_files"] = _fake_list_files
        globals()["get_documents"] = _fake_get_documents
        globals()["_ask_LLM"] = _fake_ask_llm

        # Run the self-test
        result = asyncio.run(
            process_eval_spreadsheet(package_id="SELFTEST", company_id="SELFTEST")
        )

        ok = (
            isinstance(result, dict)
            and isinstance(result.get("Technical"), dict)
            and result["Technical"].get("Design") == 40.0  # fraction -> percentage
            and result["Commercial"].get("Pricing") == 100.0
        )

        if ok:
            print("TECH_RFP_EVALUATION_CRITERIA_EXTRACTOR OK")
            return True
        else:
            print("TECH_RFP_EVALUATION_CRITERIA_EXTRACTOR FAIL: unexpected result")
            return False

    except Exception as e:
        print(f"TECH_RFP_EVALUATION_CRITERIA_EXTRACTOR ERROR: {e}")
        return False


def smoke_test():
    import os
    import json
    import uuid
    import tempfile
    import asyncio
    import shutil
    from pathlib import Path
    import pandas as pd
    from utils.document.docingest import (
        _as_json_dict,
        LOCK_FILE_PATH,
        MANIFEST_PATH,
        DOCSTORE_DIR,
    )
    from utils.core.log import set_logger, pid_tool_logger, get_logger
    from utils.storage.bucket import upload_file, delete_file, get_file

    EXCEL_PATH = "/home/development/TruBuildBE/tools/Evaluation_with_description.xlsx"
    COUNTRY_CODE = "USA"
    COMPANY_ID = None
    DEST_KEY = "tech_rfp/evaluation/smoke_eval.xlsx"
    CLEANUP = True

    # Minimal logging
    set_logger(pid_tool_logger(package_id="SMOKE", tool_name="eval_smoke"))
    log = get_logger()

    package_id = f"SMOKETEST_EXTRACTEVAL"

    tmpdir = None
    try:
        # 1) Prepare an Excel file
        if EXCEL_PATH and os.path.exists(EXCEL_PATH):
            src_path = EXCEL_PATH
            log.debug(f"Using provided Excel at {src_path}")

        upload_file(package_id=package_id, location=DEST_KEY, file_path=src_path)
        log.debug(f"Uploaded {src_path} -> {package_id}/{DEST_KEY}")

        try:
            delete_file(package_id=package_id, file_path=EVAL_PATH)
        except Exception:
            pass

        async def _run():
            res = await process_eval_spreadsheet(
                package_id=package_id, company_id=COMPANY_ID
            )
            if not res:
                print("SMOKE TEST: FAILED (no extraction result)")
                return False

            print("\n=== Extracted Evaluation JSON ===")
            print(json.dumps(res, indent=2, ensure_ascii=False))

            upload_file(
                package_id=package_id,
                location=EVAL_PATH,
                json_string=json.dumps(res, ensure_ascii=False),
            )
            print(f"\nWrote evaluation to '{EVAL_PATH}' in project '{package_id}'.")
            return True

        ok = asyncio.run(_run())

        if CLEANUP:
            try:
                manifest = _as_json_dict(get_file(package_id, MANIFEST_PATH))
                if isinstance(manifest, dict):
                    for _src, fp_name in manifest.items():
                        try:
                            delete_file(
                                package_id, f"{DOCSTORE_DIR}{fp_name}"
                            )
                        except Exception:
                            pass
                try:
                    delete_file(package_id, MANIFEST_PATH)
                except Exception:
                    pass
                try:
                    delete_file(package_id, LOCK_FILE_PATH)
                except Exception:
                    pass
            except Exception:
                pass

            try:
                delete_file(package_id, DEST_KEY)
            except Exception:
                pass

            try:
                delete_file(package_id, EVAL_PATH)
            except Exception:
                pass

        print("\nSMOKE TEST:", "OK" if ok else "FAIL")
        return ok

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    # Self-test with mocked storage/LLM (no bucket or API calls)
    ok = main()
    raise SystemExit(0 if ok else 1)
