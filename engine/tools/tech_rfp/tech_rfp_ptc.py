"""
TruBuild - Post-Tender Clarification (PTC) Generator

This module exposes :func:`generate_ptc_dataframe`, a wrapper that evaluates
each tender-criterion 'raw LLM response' and decides whether a Post-Tender
Clarification is required.

Key Features
- Prompt builder -`get_ptc_prompt()` converts an evaluator-s JSON blob into
  a single natural-language prompt.
- Single-shot LLM call -`_call_llm_json()` invokes Gemini via
  `utils.LLM.call_llm_sync`, requesting a strict JSON schema and letting the
  Tenacity retry policy handle transient errors.
- JSON schema validation - LLM response is parsed and optionally validated
  by `validate_ptc_json`.
- CSV output - The final nested dict is converted to a `pandas.DataFrame`
  and written to disk.

Usage Example
-------------
>>> df = generate_ptc_dataframe(
...         evaluations_json="/path/to/result_xx.json",
...         package_id="demo",
...         out_csv="ptc_data.csv",
...     )
>>> print(df.head())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from google.genai.types import Part
from utils.llm.LLM import call_llm_sync
from utils.core.jsonval import clean_malformed_json
from tools.tech_rfp.prompts_tech_rfp import build_ptc_prompt
from utils.core.log import pid_tool_logger, set_logger, get_logger

_MODEL_DEFAULT = "gemini-2.5-flash"


# low-level LLM helper -
def _call_llm_json(prompt: str, ptc_schema: Dict) -> Dict[str, Any] | None:
    """
    Send prompt to Gemini with strict JSON schema.  Returns parsed JSON or None.
    """
    logger = get_logger()
    try:

        raw = call_llm_sync(
            prompt_parts=[Part.from_text(text=prompt)],
            model=_MODEL_DEFAULT,
            system_instruction=(
                "You are TruBuild, a professional in the construction industry producing "
                "Post-Tender Clarification (PTC) comments in master developer style. "
                "Use directive/contractual language such as 'The Contractor shall...', 'The Contractor is required to...', "
                "'Bidder to...', 'Please revise and resubmit...', 'Provide compliant bid.' "
                "Respond ONLY with valid JSON matching the provided response_schema."
            ),
            cfg={
                "temperature": 0.0,
                "response_schema": ptc_schema,
                "max_output_tokens": 20000,
                "thinking_config": {"thinking_budget": 0},
                "top_p": 1.0,
                "top_k": 1,
                "response_mime_type": "application/json",
                "seed": 12345,
            },
            limiter=None,
        )
        fixed = clean_malformed_json(raw)
        return json.loads(fixed)
    except Exception as e:
        logger.exception("LLM call failed: %s" % e)
        logger.error(raw)
    return None


def validate_ptc_json(payload: dict[str, Any]) -> bool:
    """
    Very light validation that JSON structurally matches _PTC_SCHEMA.
    """
    try:
        return (
            isinstance(payload, dict)
            and "ptc" in payload
            and "queryDescription" in payload["ptc"]
            and "refType" in payload["ptc"]
        )
    except Exception:
        return False


# public batch-processing API
def generate_ptc_dataframe(
    *,
    evaluations_json: str | Path,
    package_id: str,
) -> pd.DataFrame:
    """
    Batch-process the tender evaluations JSON -> PTC CSV.

    Parameters
    ----------
    evaluations_json : str | Path
        Path to JSON produced by the tender-scoring tool
        (same shape as your original script expected).
    package_id : str
        Used only for logging prefix.
    out_csv : str | Path, optional
        If provided, the resulting DataFrame is saved to this path.

    Returns
    -------
    pandas.DataFrame
        Index = evaluation criterion
        Columns = contractor names
        Cells  = queryDescription string
    """
    set_logger(pid_tool_logger(package_id, "ptc"))
    logger = get_logger()
    logger.info(f"Loading evaluations JSON: {evaluations_json}")
    if isinstance(evaluations_json, dict):
        data = evaluations_json
    else:
        text = (
            evaluations_json.read_text()
            if isinstance(evaluations_json, Path)
            else str(evaluations_json)
        )
        data = json.loads(text)

    table: Dict[str, Dict[str, str]] = {}

    for tender in data["result"]["tenderReport"]:
        contractor = tender["contractorName"]
        table[contractor] = {}

        for eval_block in tender["raw_llm_responses"]:
            criterion = eval_block["criterion"]
            ptc_prompt, ptc_schema = build_ptc_prompt(json.dumps(eval_block, indent=2))
            resp = _call_llm_json(prompt=ptc_prompt, ptc_schema=ptc_schema)

            if resp and validate_ptc_json(resp):
                table[contractor][criterion] = resp["ptc"]["queryDescription"]
            else:
                logger.warning(
                    f"Invalid or missing response for {contractor} / {criterion}"
                )
                table[contractor][criterion] = "N/A"

    df = pd.DataFrame(table).reindex(sorted(table), axis=1).sort_index()

    return df


def main(*, package_id: str = "system_check") -> bool:
    """
    Lightweight self-test for the PTC generator:
      - Uses an in-memory sample evaluations payload (no file I/O)
      - Monkey-patches _call_llm_json to return deterministic JSON (no LLM)
      - Verifies the returned DataFrame has rows/cols
      - Prints OK/FAIL and returns a boolean
    """
    print("TECH_RFP_GENERATE_PTC TEST START")
    try:
        # Stub the LLM call so the test is fast and offline
        def _fake_call_llm_json(prompt: str, ptc_schema: Dict) -> Dict[str, Any]:
            # Minimal valid payload matching validate_ptc_json expectations
            return {
                "ptc": {
                    "queryDescription": "Please provide more detail on the proposed approach.",
                    "refType": "RFI",
                    "refId": "PTC-001",
                }
            }

        globals()["_call_llm_json"] = _fake_call_llm_json

        # In-memory sample evaluations (exercises multiple contractors/criteria)
        sample_data = {
            "result": {
                "tenderReport": [
                    {
                        "contractorName": "BuildCorp",
                        "raw_llm_responses": [
                            {
                                "criterion": "Technical Capability",
                                "score": 7.5,
                                "comments": "…",
                            },
                            {"criterion": "Innovation", "score": 5.0, "comments": "…"},
                        ],
                    },
                    {
                        "contractorName": "EcoConstruct",
                        "raw_llm_responses": [
                            {
                                "criterion": "Technical Capability",
                                "score": 8.0,
                                "comments": "…",
                            },
                            {"criterion": "Innovation", "score": 4.0, "comments": "…"},
                        ],
                    },
                ]
            }
        }

        # Run the generator directly with the dict (no temp file needed)
        df = generate_ptc_dataframe(evaluations_json=sample_data, package_id=package_id)

        ok = (df is not None) and (not df.empty) and df.shape[0] > 0 and df.shape[1] > 0
        if ok:
            print("TECH_RFP_GENERATE_PTC OK")
            return True
        else:
            print("TECH_RFP_GENERATE_PTC FAIL: empty or invalid DataFrame")
            return False

    except Exception as e:
        print(f"TECH_RFP_GENERATE_PTC ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
