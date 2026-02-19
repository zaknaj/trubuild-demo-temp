"""
TruBuild RFP Summarizer

This module exposes :class:`TBARFPSummarizer`, a utility that scans a folder of
RFP documents (PDF, DOC/DOCX, TXT, XLSX), extracts their textual + image
content, and asks a Gemini model to return structured JSON summaries
(`rfpDates`, `scopeOfWork`) or a narrative `rfp_with_details`.

Key Components
--------------
* Document extraction - delegates to shared helpers in ``doc.py``:
  - `process_document_parts()` from that module handles every file extension,
    returns a list of ``google.genai.types.Part``.
* Token-aware LLM calls - uses `llm.call_llm_sync()` to do the heavy
  lifting, including retries and rate-limiting.
* Prompt / schema library* - in-code dictionary mapping prompt-keys to
  JSON-schemas and natural-language instructions.


usage
-----------
>>> summarizer = TBARFPSummarizer(package_id="demo",
...                               rfp_directory="/data/demo/rfp",
...                               log=True)
>>> result = summarizer.summarize_rfp(prompt_key="rfpDates")
>>> print(result["rfpDates"]["proposalSubmissionDeadline"])
"""

from __future__ import annotations

import os
import json
from google.genai.types import Part
from typing import Any, Dict, List, Literal

from utils.llm.LLM import call_llm_sync
from utils.storage.bucket import list_files
from utils.document.docingest import get_documents
from tools.tech_rfp.prompts_tech_rfp import summarize_rfp_prompt_library


from utils.core.log import pid_tool_logger, set_logger, get_logger

from utils.llm.compactor_cache import set_compactor_context, _norm_name


class TBARFPSummarizer:
    """
    Summarise all RFP documents  into structured JSON via Gemini.

    Parameters
    ----------
    package_id : str
        TruBuild package identifier - used for on-disk cache path.
    rfp_directory : str | Path
        Folder containing raw RFP files.
    model_name : str, default 'gemini-2.5-flash'
        Gemini model to call.
    log : bool, default False
        Enables file logging.

    Notes
    -----
    * All extraction of document pages / images is delegated to ``doc.load_dir_as_parts``.
    * LLM calls are made through the shared ``utils.LLM.call_llm_sync`` wrapper,
      which already implements token counting, retries, and rate-limiting.
    """

    def __init__(
        self,
        package_id: str,
        *,
        model_name: str = "gemini-2.5-flash",
        name_to_source_key: dict[str, str] | None = None,
    ):
        set_logger(pid_tool_logger(package_id, "rfp_summary"))
        self.logger = get_logger()
        self.logger.debug("TBARFPSummarizer initialized")
        self.package_id = package_id
        self.model = model_name
        self._name_to_source_key = {
            _norm_name(k): v for k, v in (name_to_source_key or {}).items()
        }

        # default generation config
        self._cfg: dict[str, Any] = {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1.0,
            "max_output_tokens": 20000,
            "thinking_config": {"thinking_budget": 0},
            "response_mime_type": "application/json",
            "seed": 12345,
        }

    def _unwrap_payload(self, expected_key: str, payload):
        """
        If the model wrapped the result like {"rfpDates": {...}},
        return just the inner object. Otherwise return as-is.
        """
        if isinstance(payload, dict) and expected_key in payload and len(payload) == 1:
            return payload[expected_key]
        return payload

    # public API
    def summarize_rfp(
        self, doc_parts: List[Part], prompt_key: str | None = None
    ) -> Dict[str, Any]:
        """
        Run the RFP summariser for the chosen prompt.

        Parameters
        ----------
        prompt_key : {'rfpDates', 'scopeOfWork', 'rfp_with_details'}, optional
            If None, runs both *rfpDates* and *scopeOfWork* prompts.

        Returns
        -------
        dict
            Keys are prompt keys; values are parsed JSON or raw narrative text.
        """

        prompts, schemas = summarize_rfp_prompt_library()
        keys_to_run = [prompt_key] if prompt_key else ["rfpDates", "scopeOfWork"]

        # iterate prompts, call LLM
        result: Dict[str, Any] = {}
        for key in keys_to_run:
            output = self._run_single_prompt(
                prompt_text=prompts[key],
                response_schema=schemas.get(key),
                parts=doc_parts,
            )
            if key in ("rfpDates", "scopeOfWork"):
                output = self._unwrap_payload(key, output)

            result[key] = output
        return result

    def _run_single_prompt(
        self,
        *,
        prompt_text: str,
        response_schema: dict | None,
        parts: List[Part],
    ) -> str | Dict[str, Any]:
        """
        Dispatch one prompt to Gemini.

        Returns raw text (for narrative prompt) or parsed JSON (for schema prompts).
        """

        system_instruction = (
            "You are an RFP summarization assistant. "
            "Process the document parts (text and images) "
            "and generate a comprehensive summary."
        )

        if response_schema:
            cfg = {**self._cfg, "response_schema": response_schema}
        else:
            cfg = self._cfg.copy()

        try:
            reply = call_llm_sync(
                [Part.from_text(text=prompt_text)] + parts,
                model=self.model,
                system_instruction=system_instruction,
                cfg=cfg,
                name_to_source_key=self._name_to_source_key,
            )
            if response_schema:
                try:
                    return json.loads(reply)
                except json.JSONDecodeError:
                    raise ValueError("LLM did not return valid JSON")
            return reply
        except Exception as e:
            self.logger.exception("Prompt failed: %s", e)
            return {"error": "Failed after several attempts"}


def _paths(variant: str) -> tuple[str, str]:
    """
    Returns (rfp_source_dir, summary_key) for the chosen variant.
    """
    # where raw PDFs / DOCX were uploaded
    src_dir = f"{variant}_rfp/rfp/"  # "tech_rfp/rfp/" or "comm_rfp/rfp/"
    # legacy key path kept for compatibility/logging only; final result is stored in DB artifacts
    key_json = f"data/{variant}_rfp_summary.json"
    return src_dir, key_json


async def rfp_summary_main(
    *,
    package_id: str | None = None,
    company_id: str | None = None,
    user_name: str | None = None,
    rfp_variant: Literal["tech", "comm"] = "tech",
    country_code: str = "USA",
    request_method: str | None = None,
    remote_ip: str | None = None,
    compute_reanalysis: bool = True,
    package_name: str = "",
    company_name: str = "",
    **_,
) -> Dict[str, Any]:
    """
    Dispatcher that routes to job queue system.
    Only tech variant uses job queue for now.
    """
    from utils.db.job_queue import JobType
    from tools.tech_rfp.job_queue_integration import (
        create_tech_rfp_job,
        get_job_status_response,
    )

    tool_base = "COMM_RFP" if rfp_variant == "comm" else "TECH_RFP"
    logger = pid_tool_logger(package_id=package_id, tool_name="rfp_summary")
    set_logger(
        logger,
        tool_name="rfp_summary_main",
        tool_base=tool_base,
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    country_code = country_code or "USA"

    if not package_id:
        return {"error": "packageId is required", "status": "error"}

    # Use job queue for all variants (comm variant will be migrated later)
    if request_method == "GET":
        return get_job_status_response(
            package_id=package_id,
            company_id=company_id,
            job_type=JobType.TECH_RFP_SUMMARY,
        )

    if request_method == "POST":
        if not compute_reanalysis:
            response = get_job_status_response(
                package_id=package_id,
                company_id=company_id,
                job_type=JobType.TECH_RFP_SUMMARY,
            )
            if response.get("status") == "completed":
                return response

        job_id = create_tech_rfp_job(
            job_type=JobType.TECH_RFP_SUMMARY,
            package_id=package_id,
            company_id=company_id,
            user_name=user_name,
            remote_ip=remote_ip,
            country_code=country_code,
            package_name=package_name,
            company_name=company_name,
            rfp_variant=rfp_variant,
        )

        logger.info(f"Created rfp_summary job {job_id} for project {package_id}")
        return {
            "status": "in progress",
            "message": f"RFP summary job created for {package_id}",
            "job_id": job_id,
        }

    return {"error": f"Unsupported method {request_method}", "status": "error"}


async def _do_summary_workflow(
    package_id: str,
    company_id: str,
    rfp_variant: Literal["tech", "comm"] = "tech",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Extracted workflow for RFP summary generation.
    Can be called directly by job processors.

    Args:
        package_id: Project identifier
        company_id: Company identifier
        rfp_variant: "tech" or "comm"
        progress_callback: Optional callback function(progress_dict) for progress updates

    Returns:
        Summary dict
    """
    set_compactor_context(package_id=package_id, company_id=company_id)

    worker_logger = pid_tool_logger(package_id, "summarize_rfp")
    set_logger(
        worker_logger,
        tool_name="summarize_rfp",
        tool_base="TECH_RFP" if rfp_variant == "tech" else "COMM_RFP",
        package_id=package_id or "unknown",
        ip_address="no_ip",
        request_type="WORKER",
    )
    logger = get_logger()

    if progress_callback:
        progress_callback({"stage": "loading", "message": "Loading RFP documents"})

    src_dir, summary_key = _paths(rfp_variant)
    try:
        rfp_file_keys = list_files(package_id, src_dir, company_id)

        if not rfp_file_keys:
            raise RuntimeError(f"No RFP files found in '{src_dir}'")

        doc_meta = [
            {"name": os.path.basename(key), "source_key": key} for key in rfp_file_keys
        ]
        name_to_source_key = {
            _norm_name(m["name"]): m["source_key"]
            for m in doc_meta
            if m.get("name") and m.get("source_key")
        }

        rfp_parts, _ = await get_documents(
            package_id, company_id, doc_meta, force_refresh=True
        )

        if not rfp_parts:
            raise RuntimeError("Document loader returned no usable parts.")

        if progress_callback:
            progress_callback({"stage": "summarizing", "message": "Generating summary"})

        summariser = TBARFPSummarizer(
            package_id=package_id, name_to_source_key=name_to_source_key
        )
        summary = summariser.summarize_rfp(doc_parts=rfp_parts)

        logger.info("RFP summary completed")
        return summary
    except Exception as exc:
        logger.exception("RFP summary workflow failed: %s", exc)
        raise


def main(*, package_id: str = "system_check") -> bool:
    """
    Lightweight self-test for TBARFPSummarizer:
      - Monkey-patches prompt library + LLM call
      - Uses in-memory Parts (no I/O)
      - Verifies output shape
      - Prints OK/FAIL and returns a boolean
    """
    from google.genai.types import Part
    import json

    print("RFP_SUMMARIZER TEST START")
    try:
        # Fake prompt library so we don't depend on external module contents
        def _fake_prompt_lib():
            prompts = {
                "rfpDates": "RFP_SUMMARY::rfpDates:: Extract key RFP dates.",
                "scopeOfWork": "RFP_SUMMARY::scopeOfWork:: Summarize the scope of work.",
                # optional narrative prompt if you ever choose to run it
                "rfp_with_details": "RFP_SUMMARY::rfp_with_details:: Narrative summary.",
            }
            # minimal schemas; the code doesn't validate them, just routes JSON
            schemas = {
                "rfpDates": {
                    "type": "object",
                    "properties": {"proposalSubmissionDeadline": {"type": "string"}},
                },
                "scopeOfWork": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "keyDeliverables": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            }
            return prompts, schemas

        # monkey-patch the imported symbol used by TBARFPSummarizer
        globals()["summarize_rfp_prompt_library"] = _fake_prompt_lib

        # Fake LLM that returns deterministic JSON based on the prompt key tag
        def _fake_call_llm_sync(parts, model, system_instruction, cfg):
            prompt_text = getattr(parts[0], "text", "")
            if "::rfpDates::" in prompt_text:
                return json.dumps(
                    {
                        "rfpDates": {
                            "proposalSubmissionDeadline": "2025-09-30",
                            "preBidMeetingDate": "2025-09-10",
                        }
                    }
                )
            if "::scopeOfWork::" in prompt_text:
                return json.dumps(
                    {
                        "scopeOfWork": {
                            "summary": "Provide design, implementation, and support.",
                            "keyDeliverables": ["Design docs", "MVP", "User training"],
                        }
                    }
                )
            # narrative fallback
            return "This is a narrative RFP summary."

        # monkey-patch the imported LLM wrapper
        globals()["call_llm_sync"] = _fake_call_llm_sync

        # In-memory document parts (no file I/O)
        doc_parts = [
            Part.from_text(
                text=(
                    "Sample RFP: Proposal due September 30, 2025. "
                    "Scope includes design, implementation, and training."
                )
            )
        ]

        # Run summarizer for default prompts (rfpDates + scopeOfWork)
        summarizer = TBARFPSummarizer(package_id=package_id)
        result = summarizer.summarize_rfp(doc_parts=doc_parts)

        # Minimal shape checks -> boolean outcome
        ok = (
            isinstance(result, dict)
            and isinstance(result.get("rfpDates"), dict)
            and isinstance(result.get("scopeOfWork"), dict)
            and "proposalSubmissionDeadline" in result["rfpDates"]
        )

        if ok:
            print("RFP_SUMMARIZER OK")
            return True
        else:
            print("RFP_SUMMARIZER FAIL: unexpected result shape")
            return False

    except Exception as e:
        print(f"RFP_SUMMARIZER ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
