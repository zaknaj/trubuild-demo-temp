import os
import inspect
import asyncio
from pathlib import Path

# utils main functions
from utils.document.doc import main as doc_main
from utils.llm.LLM import main as llm_main
from utils.core.slack import main as slack_main
from utils.storage.bucket import main as bucket_main
from utils.core.jsonval import main as jsonval_main
from utils.core.fuzzy_search import main as fuzzy_main
from utils.document.docingest import main as docingest_main
from utils.core.web_search import main as web_search_main
from utils.llm.context_cache import main as context_cache_main
from utils.core.log import setup_logging, pid_tool_logger, set_logger, get_logger


# tools main functions
from tools.chat.chat import main as chat_main
from tools.tech_rfp.tech_rfp import main as tech_rfp_main
from tools.tech_rfp.tech_rfp import main as tech_rfp_ptc_main
from tools.comm_rfp.comm_rfp import main as comm_rfp_test_main
from tools.tech_rfp.rfp_summarizer import main as rfp_summarizer_main
from tools.contract.contract_review import main as contract_review_main
from tools.comm_rfp.comm_rfp_extract import main as comm_rfp_extract_main
from tools.comm_rfp.comm_rfp_process import main as comm_rfp_process_main
from tools.contract.contract_analyzer import main as contract_analyzer_main
from tools.tech_rfp.tech_rfp_generate_report import (
    main as tech_rfp_generate_report_main,
)
from tools.tech_rfp.tech_rfp_generate_evaluation import (
    main as tech_rfp_generate_eval_criteria_main,
)
from tools.tech_rfp.tech_rfp_evaluation_criteria_extractor import (
    main as tech_rfp_eval_criteria_extraction_main,
)


import logging

logger = logging.getLogger("TruBuildBE")

# List of main functions

main_functions = [
    # Utils
    ("bucket_main", bucket_main),
    ("context_cache_main", context_cache_main),
    ("doc_main", doc_main),
    ("doc_ingest_main", docingest_main),
    ("fuzzy_main", fuzzy_main),
    ("jsonval_main", jsonval_main),
    ("llm_main", llm_main),
    ("slack_main", slack_main),
    ("web_search", web_search_main),
    # Not present: check.py, sim.py, log.py
    # Tools
    ("chat_main", chat_main),
    ("comm_rfp_extract", comm_rfp_extract_main),
    ("comm_rfp_process", comm_rfp_process_main),
    ("comm_rfp_main", comm_rfp_test_main),
    ("contract_analyzer_main", contract_analyzer_main),
    ("contract_review_main", contract_review_main),
    ("rfp_summarizer", rfp_summarizer_main),
    ("tech_rfp_eval_criteria_extraction", tech_rfp_eval_criteria_extraction_main),
    ("tech_rfp_generate_eval_criteria", tech_rfp_generate_eval_criteria_main),
    ("tech_rfp_generate_report", tech_rfp_generate_report_main),
    ("tech_rfp_ptc", tech_rfp_ptc_main),
    ("tech_rfp_main", tech_rfp_main),
]


class _OnlyThisProjectOnConsole(logging.Filter):
    def __init__(
        self,
        package_id: str,
        request_type: str | None = None,
        tool_base: str | None = None,
        min_always: int = logging.ERROR,
    ):
        super().__init__()
        self.package_id = package_id
        self.request_type = request_type
        self.tool_base = tool_base
        self.min_always = min_always  # show >= this level even if context missing

    def filter(self, record: logging.LogRecord) -> bool:
        # Always allow high-severity records (ERROR by default)
        if record.levelno >= self.min_always:
            return True
        # Otherwise, require our CHECK context
        if getattr(record, "package_id", None) != self.package_id:
            return False
        if (
            self.request_type
            and getattr(record, "request_type", None) != self.request_type
        ):
            return False
        if self.tool_base and getattr(record, "tool_base", None) != self.tool_base:
            name = getattr(record, "name", "")
            if not (name.endswith(".CHECK") or name.endswith(".check")):
                return False
        return True


def _attach_filter_to_all_stream_handlers(console_filter: logging.Filter):
    """Attach filter to ALL existing loggers' StreamHandlers (not file handlers)."""
    # root handlers
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            h.addFilter(console_filter)
    # named loggers created so far
    for _name, obj in logging.root.manager.loggerDict.items():
        if isinstance(obj, logging.Logger):
            for h in obj.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, logging.FileHandler
                ):
                    h.addFilter(console_filter)


def _prepare_check_logging(ip_address: str):
    setup_logging()

    base = pid_tool_logger(package_id="SYSTEM_CHECK", tool_name="check")
    base.propagate = True

    ctx = {
        "tool_name": "CHECK",
        "tool_base": "CHECK",
        "package_id": "system_check",
        "ip_address": ip_address or "127.0.0.1",
        "request_type": "CHECK",
    }
    adapter = logging.LoggerAdapter(base, ctx)
    set_logger(base, **ctx)

    # Console filter: show only this run’s logs; always allow ERROR+
    only_check = _OnlyThisProjectOnConsole(
        package_id="system_check",
        request_type="CHECK",
        tool_base="CHECK",
        min_always=logging.ERROR,
    )
    _attach_filter_to_all_stream_handlers(only_check)

    return adapter


def check_main(ip_address: str):
    log_adapter = _prepare_check_logging(ip_address)

    passed = 0
    failed = 0
    total = len(main_functions)
    failed_items = []

    for name, fn in main_functions:
        try:
            result = fn()
            if inspect.isawaitable(result):
                result = asyncio.run(result)

            if result is None:
                reason = "returned None — unable to verify success"
                log_adapter.error(f"{name} {reason}.")
                failed += 1
                failed_items.append((name, reason))
            elif isinstance(result, bool) and not result:
                reason = "returned False — indicates failure"
                log_adapter.error(f"{name} {reason}.")
                failed += 1
                failed_items.append((name, reason))
            else:
                log_adapter.info(f"{name} succeeded with result: {result}")
                passed += 1

        except Exception as exc:
            reason = f"exception: {exc.__class__.__name__}: {exc}"
            log_adapter.exception(f"{name} FAILED with {reason}")
            failed += 1
            failed_items.append((name, reason))

    log_adapter.info("================= CHECK SUMMARY =================")
    log_adapter.info(f"Passed:   {passed}")
    log_adapter.info(f"Failed:   {failed}")
    log_adapter.info(f"Total:    {total}")
    if failed_items:
        log_adapter.info("----------- Failed items -----------")
        for i, (n, why) in enumerate(failed_items, 1):
            log_adapter.info(f"{i}. {n} — {why}")
    log_adapter.info("=================================================")


if __name__ == "__main__":
    setup_logging()
    check_main(ip_address="1")
