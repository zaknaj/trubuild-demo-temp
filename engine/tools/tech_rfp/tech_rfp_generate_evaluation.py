"""
TruBuild - Technical-Evaluation Criteria Generator

This module turns an uploaded set of RFP documents into a weighted JSON rubric
for technical scoring.  It re-uses the same cache layout as the new chat engine
(`data/chat_docs/<file>.json`) and the shared Gemini helper utilities.

Typical usage
-------------
>>> files = [
...     {"name": "tech_rfp_cleaned.pdf", "category": "techRfp"},
... ]
>>> gen = TechCriteriaGenerator(package_id="demo", base_path="..", doc_meta=files)
>>> rubric = gen.build_rubric()          # returns dict[str, dict[str, float]]
"""

from __future__ import annotations

import json
import os, re
import time, random
from typing import Any, Dict, List
from utils.storage.bucket import list_files
from utils.document.docingest import get_documents
from decimal import Decimal, ROUND_HALF_UP
from utils.core.log import pid_tool_logger, set_logger, get_logger

from google.genai.types import Part

from utils.llm.LLM import call_llm_sync
from tools.tech_rfp.prompts_tech_rfp import generate_evaluation_criteria_prompt


_MODEL = "gemini-2.5-flash"
_ALT_MODEL = "gemini-2.5-flash"
_CFG: dict[str, Any] = {
    "temperature": 0.0,
    "max_output_tokens": 20000,
    "thinking_config": {"thinking_budget": 0},
    "response_mime_type": "application/json",
    "top_p": 1.0,
    "top_k": 1.0,
    "seed": 12345,
}


def normalize_weights(
    response_json: dict[str, dict[str, Any]],
    step: float = 0.5,
    min_weight: float = 1.0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Normalize weights to the below schema:
      { Scope: { Criterion: { "weight": <float>, "description"?: <str> } } }

    - Accepts legacy flat form (Criterion: <number|string>) or new form
      (Criterion: {"weight": <number|string>, "description"?: <str>}).
    - Scales so the total = 100.0, rounds to `step`, enforces `min_weight`.
    - Preserves any provided descriptions.
    """
    quant = Decimal(str(step))
    min_w = Decimal(str(min_weight))

    # 1) Flatten and coerce incoming weights; carry descriptions
    flat_rows: list[tuple[str, str, Decimal, str | None]] = []

    def _coerce_num(x: Any) -> Decimal | None:
        if isinstance(x, (int, float, Decimal)):
            return Decimal(str(x))
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            s = re.sub(r"[^\d.]", "", s)
            if not s:
                return None
            try:
                return Decimal(s)
            except Exception:
                return None
        return None

    for scope, block in (response_json or {}).items():
        if not isinstance(block, dict):
            continue
        for crit, v in block.items():
            desc = None
            if isinstance(v, dict) and ("weight" in v or "description" in v):
                w = _coerce_num(v.get("weight"))
                d = v.get("description")
                if isinstance(d, str) and d.strip():
                    desc = d.strip()
            else:
                w = _coerce_num(v)
            if w is None:
                continue
            flat_rows.append((scope, crit, w, desc))

    if not flat_rows:
        raise ValueError("No valid criteria found in model response.")

    # 2) Proportional scaling to 100
    total_raw = sum(w for *_, w, _ in flat_rows)
    if total_raw == 0:
        raise ValueError("All weights are zero.")

    scaled: list[list[Any]] = []
    for scope, crit, w, desc in flat_rows:
        exact = w * Decimal("100") / total_raw
        rounded = (exact / quant).quantize(0, ROUND_HALF_UP) * quant
        scaled.append([scope, crit, rounded, desc])

    # 3) Enforce minimum per item
    for row in scaled:
        if row[2] < min_w:
            row[2] = min_w

    # 4) Fix rounding drift to sum exactly 100
    total_now = sum(r[2] for r in scaled)
    drift = Decimal("100") - total_now
    step_unit = quant if drift > 0 else -quant

    while drift != 0:
        for r in scaled:
            if drift == 0:
                break
            # Do not drop below min_w
            if r[2] + step_unit >= min_w:
                r[2] += step_unit
                drift -= step_unit

    # 5) Rebuild out: keep descriptions if present
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for scope, crit, w, desc in scaled:
        out.setdefault(scope, {})
        entry: dict[str, Any] = {"weight": float(w)}
        if isinstance(desc, str) and desc.strip():
            entry["description"] = desc.strip()
        out[scope][crit] = entry

    return out


class TechCriteriaGenerator:
    """
    Build weighted technical-evaluation criteria from cached RFP document Parts.

    Parameters
    ----------
    package_id : str
    base_path  : str | Path
        Root folder containing `data/chat_docs`.
    doc_meta   : List[dict]
        Same structure as the chat engine - list of {"name": ..., "category": ...}.
    """

    def __init__(
        self,
        package_id: str,
        *,
        model_name: str = _MODEL,
    ):
        self.logger = get_logger()
        self.package_id = package_id
        self.model = model_name

    def build_rubric(self, doc_parts: List[Part]) -> Dict[str, Dict[str, float]]:
        """Return normalised rubric {theme: {criterion: weight}}."""
        rfp_text = "\n".join(p.text for p in doc_parts if getattr(p, "text", None))
        prompt = generate_evaluation_criteria_prompt(rfp_text)

        raw_json = self._ask_llm(prompt)
        self._validate_raw_rubric(raw_json)
        normalised = normalize_weights(raw_json)
        self.logger.debug(f"Rubric built with {len(normalised)} themes")
        return normalised

    def _validate_raw_rubric(self, data: Dict[str, Any]) -> None:
        """Raise ValueError if JSON rubric is structurally invalid."""
        if not isinstance(data, dict) or not data:
            raise ValueError("LLM response is not a non-empty JSON object")

        saw_any = False

        for theme, criteria in data.items():
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(f"Theme '{theme}' has no valid criteria")

            for crit, w in criteria.items():
                # SCHEMA: {"weight": <num|str>, "description"?: <str>}
                if isinstance(w, dict):
                    if "weight" not in w:
                        raise ValueError(
                            f"Criterion '{crit}' object missing 'weight' field"
                        )

                    try:
                        w_num = float(w["weight"])
                    except Exception:
                        raise ValueError(f"Weight for '{crit}' is not numeric")

                    if not (0 <= w_num <= 100):
                        raise ValueError(
                            f"Weight for '{crit}' out of 1-100 range: {w_num}"
                        )

                    # basic sanity for description if present
                    if (
                        "description" in w
                        and w["description"] is not None
                        and not isinstance(w["description"], str)
                    ):
                        raise ValueError(
                            f"Description for '{crit}' must be a string if provided"
                        )

                    saw_any = True
                    continue

        if not saw_any:
            raise ValueError("No numeric weights found")

    def _ask_llm(self, prompt: str) -> Dict[str, Dict[str, float]]:
        """
        Retry policy:
          Try 1 (primary model): 4 attempts with exponential backoff and varied temperatures.
            The last two attempts draw temperatures randomly from lower-end values.
          Try 2 (fallback model): repeat the same strategy with gemini-2.5-flash.
        """
        sys_inst = (
            "You are TruBuild, a professional evaluator in the construction industry. "
            "Output strictly JSON according to the prompt."
        )

        def _attempt_round(model_name: str) -> Dict[str, Dict[str, float]] | None:
            # First two are fixed, last two are randomly sampled from the low end.
            temps = [0.9, 0.6, random.uniform(0.05, 0.35), random.uniform(0.0, 0.25)]
            delay = 1.0  # seconds
            for i, t in enumerate(temps, start=1):
                cfg = dict(_CFG)
                cfg["temperature"] = float(t)
                try:
                    raw = call_llm_sync(
                        [Part.from_text(text=prompt)],
                        model=model_name,
                        system_instruction=sys_inst,
                        cfg=cfg,
                    )
                    return json.loads(raw)
                except Exception as exc:
                    self.logger.warning(
                        f"ask_llm attempt {i}/4 failed (model={model_name}, temp={t:.2f}): {exc}"
                    )
                    if i < 4:
                        time.sleep(delay)
                        delay *= 2
            return None

        # Try 1 primary model
        out = _attempt_round(self.model)
        if out is not None:
            return out

        # Try 2 fallback model
        out = _attempt_round(_ALT_MODEL)
        if out is not None:
            return out

        raise RuntimeError(
            "generating evaluation criteria failed after retries on both models"
        )

## EVAL PATH global definition
EVAL_PATH = "data/evaluation.json"


async def generate_eval_main(
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
    Router for criteria-generation tool.
    Uses job queue system for processing.
    """
    from utils.db.job_queue import JobType
    from tools.tech_rfp.job_queue_integration import (
        create_tech_rfp_job,
        get_job_status_response,
    )

    base_logger = pid_tool_logger(package_id=package_id, tool_name="generate_eval")
    set_logger(
        base_logger,
        tool_name="generate_eval_main",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()

    country_code = country_code or "USA"

    if not package_id:
        return {"error": "packageId is required", "status": "error"}

    if request_method == "GET":
        return get_job_status_response(
            package_id=package_id,
            company_id=company_id,
            job_type=JobType.TECH_RFP_GENERATE_EVAL,
        )

    if request_method == "POST":
        if not compute_reanalysis:
            response = get_job_status_response(
                package_id=package_id,
                company_id=company_id,
                job_type=JobType.TECH_RFP_GENERATE_EVAL,
            )
            if response.get("status") == "completed":
                return response

        job_id = create_tech_rfp_job(
            job_type=JobType.TECH_RFP_GENERATE_EVAL,
            package_id=package_id,
            company_id=company_id,
            user_name=user_name,
            remote_ip=remote_ip,
            country_code=country_code,
        )

        logger.info(f"Created generate_eval job {job_id} for project {package_id}")
        return {
            "status": "in progress",
            "message": f"Evaluation generation job created for {package_id}",
            "job_id": job_id,
        }

    return {"error": f"Unsupported request method: {request_method}", "status": "error"}


# Extracted workflow function for job queue
async def _do_generate_eval_workflow(
    package_id: str,
    company_id: str,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Extracted workflow for evaluation criteria generation.
    Can be called directly by job processors.

    Args:
        package_id: Project identifier
        company_id: Company identifier
        progress_callback: Optional callback function(progress_dict) for progress updates

    Returns:
        Generated evaluation criteria dict
    """
    worker_logger = pid_tool_logger(package_id, "generate_eval")
    set_logger(
        worker_logger,
        tool_name="generate_eval_worker",
        package_id=package_id or "unknown",
        ip_address="no_ip",
        request_type="WORKER",
    )
    logger = get_logger()
    try:
        if progress_callback:
            progress_callback({"stage": "loading", "message": "Loading RFP documents"})

        rfp_source_dir = "tech_rfp/rfp/"
        rfp_file_keys = list_files(package_id, rfp_source_dir, company_id)

        if not rfp_file_keys:
            raise RuntimeError("No files found in tech_rfp/rfp/")

        doc_meta = [
            {"name": os.path.basename(key), "source_key": key} for key in rfp_file_keys
        ]

        if progress_callback:
            progress_callback({"stage": "processing", "message": "Processing documents"})

        rfp_parts, _ = await get_documents(
            package_id, company_id, doc_meta, tool_context=None, force_refresh=True
        )

        if not rfp_parts:
            raise RuntimeError("Document loader returned no usable parts.")

        if progress_callback:
            progress_callback(
                {"stage": "generating", "message": "Generating evaluation criteria"}
            )

        generator = TechCriteriaGenerator(package_id=package_id)
        rubric = generator.build_rubric(doc_parts=rfp_parts)

        logger.info("Evaluation criteria generated")
        return rubric
    except Exception as exc:
        logger.exception("Generate evaluation workflow failed: %s", exc)
        raise


def main() -> bool:
    """
    Lightweight self-test for TechCriteriaGenerator:
      - Builds tiny in-memory doc_parts
      - Monkey-patches _ask_llm to return a deterministic rubric
      - Runs build_rubric(doc_parts) and sanity-checks the normalized result
      - Prints OK/FAIL and returns a boolean
    """
    set_logger(pid_tool_logger("SYSTEM_CHECK", "tech_rfp_generate_evaluation"))
    print("TECH_RFP_GENERATE_EVALUATION TEST START")
    try:
        # 1) Minimal in-memory "RFP" content
        doc_parts = [
            Part.from_text(
                text=(
                    "Evaluation will consider Technical Capability, Team Experience, and Innovation. "
                    "Vendors must describe architecture, scalability, security, and team credentials."
                )
            )
        ]

        # 2) Stub the LLM call to be deterministic & fast
        def _fake_ask_llm(self, prompt: str) -> Dict[str, Dict[str, float]]:
            # Deliberately not summing to 100; normalize_weights should fix it.
            return {
                "Technical Capability": {
                    "Architecture": 35,
                    "Scalability": 20,
                    "Security": 15,
                },
                "Team Experience": {
                    "Experience": 18,
                    "Certifications": 7,
                },
                "Innovation": {
                    "R&D": 3,
                    "Differentiators": 2,
                },
            }

        TechCriteriaGenerator._ask_llm = _fake_ask_llm

        # 3) Run generator
        gen = TechCriteriaGenerator(package_id="SELFTEST")
        rubric = gen.build_rubric(doc_parts=doc_parts)

        # 4) Sanity checks -> boolean outcome
        def _sum_weights(r: Dict[str, Dict[str, float]]) -> float:
            return sum(v for section in r.values() for v in section.values())

        ok = (
            isinstance(rubric, dict)
            and rubric
            and abs(_sum_weights(rubric) - 100.0)
            < 1e-6  # should be exactly 100 after normalization
            and all(
                isinstance(v, (int, float))
                for sec in rubric.values()
                for v in sec.values()
            )
        )

        if ok:
            print("TECH_RFP_GENERATE_EVALUATION OK")
            return True
        else:
            print(
                "TECH_RFP_GENERATE_EVALUATION FAIL: unexpected rubric shape or weights"
            )
            return False

    except Exception as e:
        print(f"TECH_RFP_GENERATE_EVALUATION ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
