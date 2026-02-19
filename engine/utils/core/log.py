import re
import json
import logging
import pathlib
import datetime
import logging.config
from typing import Union
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler


_logger_var: ContextVar[Union[logging.Logger, logging.LoggerAdapter]] = ContextVar(
    "pid_tool_logger", default=None
)

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
ORANGE = "\033[33m"
GREY = "\033[90m"
WHITE = "\033[97m"
PURPLE = "\033[35m"
RESET = "\033[0m"


def set_logger(logger: logging.Logger, **extra):
    _logger_var.set(logging.LoggerAdapter(logger, extra))


def get_logger() -> logging.Logger:
    logger = _logger_var.get()
    if logger is None:
        raise RuntimeError("Tool-specific logger not set in this context")
    return logger


class NoDebugFilter(logging.Filter):
    """Filter that blocks DEBUG messages"""

    def filter(self, record):
        return record.levelno > logging.DEBUG


def setup_logging():
    config_file = pathlib.Path("utils/logging_config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    # Expand ~ to full home directory for main log file
    if "handlers" in config:
        for handler in config["handlers"].values():
            if "filename" in handler:
                path = pathlib.Path(handler["filename"]).expanduser()
                handler["filename"] = str(path)
                # Ensure the parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
    noisy_libs = [
        "google_genai",
        "google_genai.models",
        "google.genai",
        "google.genai.models",
        "grpc",
        "httpx",
        "absl",
        "google",
    ]

    for name in noisy_libs:
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.ERROR)  # show only ERROR+
        lib_logger.propagate = False

    # Add contextual filter and no-debug filter to root logger
    context_filter = ContextFilter()
    no_debug_filter = NoDebugFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(context_filter)
    for handler in root_logger.handlers:
        handler.addFilter(context_filter)
        handler.addFilter(no_debug_filter)  # Add no-debug filter to all root handlers


class ContextFilter(logging.Filter):
    def filter(self, record):
        # Try to pull extras from the current adapter, if any
        try:
            current = _logger_var.get()
            if isinstance(current, logging.LoggerAdapter):
                extra = getattr(current, "extra", {}) or {}
                for k in (
                    "tool_name",
                    "package_id",
                    "ip_address",
                    "request_type",
                    "user_name",
                ):
                    if not hasattr(record, k) and k in extra:
                        setattr(record, k, extra[k])
        except LookupError:
            pass  # no adapter set in this context

        # Final defaults
        if not hasattr(record, "tool_name"):
            record.tool_name = "N/A"
        if not hasattr(record, "package_id"):
            record.package_id = "N/A"
        if not hasattr(record, "ip_address"):
            record.ip_address = "no_ip"
        if not hasattr(record, "request_type"):
            record.request_type = "N/A"
        if not hasattr(record, "user_name"):
            record.user_name = "Anonymous"
        return True


class PidToolHandlerFilter(logging.Filter):
    """Filter for pid_tool handler - only allows DEBUG, ERROR, CRITICAL"""

    def filter(self, record):
        # Allow DEBUG, ERROR (40), and CRITICAL (50)
        # This will also catch exceptions since they're at ERROR level
        return record.levelno == logging.DEBUG or record.levelno >= logging.ERROR


def pid_tool_logger(package_id: str, tool_name: str):
    base = pathlib.Path.home() / "process_logs"
    log_dir = base / package_id
    log_dir.mkdir(parents=True, exist_ok=True)

    log_name = log_dir / f"{tool_name}.log"
    handler = RotatingFileHandler(filename=log_name, maxBytes=5_000_000, backupCount=1)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    # Allow DEBUG + ERROR/CRITICAL (and thus EXCEPTION) to this file
    handler.addFilter(PidToolHandlerFilter())

    logger_name = f"{package_id}.{tool_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if this is called multiple times
    logger.handlers.clear()
    logger.addHandler(handler)

    logger.propagate = True

    return logger


class DynamicPrefixFormatter(logging.Formatter):
    """
    Color-aware formatter. Pass color=True/False from logging config.
    """

    PID_W = 26  # package_id
    IP_W = 15  # IPv4
    PROC_W = 6  # POST/GET/WORKER (max 7)
    TOOL_W = 9  # tool base label
    FUNC_W = 20  # function name
    LEVEL_W = 7  # INFO/WARNING/ERROR

    def __init__(self, color: bool = True):
        super().__init__()
        self.color = bool(color)

    def _c(self, code: str) -> str:
        """Return ANSI code only if color mode is enabled."""
        return code if self.color else ""

    @staticmethod
    def _derive_tool_base(record: logging.LogRecord) -> str:
        tb = getattr(record, "tool_base", None)
        if tb:
            return str(tb).upper()

        name = getattr(record, "name", "")
        if "." in name:
            tool = name.split(".", 1)[1].lower()
        else:
            tool = (getattr(record, "tool_name", "") or "").lower()

        tool = re.sub(r"(_main|_worker)$", "", tool)

        if any(k in tool for k in ("tech", "eval", "rfp_", "summary", "report")):
            return "COMM_RFP" if "comm" in tool else "TECH_RFP"
        if "comm" in tool:
            return "COMM_RFP"
        if "chat" in tool:
            return "CHAT"
        if "contract" in tool:
            return "CONTRACT"
        return "-"

    def format(self, record: logging.LogRecord) -> str:
        # Gather fields
        is_error_or_warn = record.levelno >= logging.WARNING
        is_error = record.levelno >= logging.ERROR
        request_type = (getattr(record, "request_type", "") or "").upper()
        is_get = request_type == "GET"
        process = (getattr(record, "request_type", "N/A") or "N/A")[: self.PROC_W]
        package_id = (getattr(record, "package_id", "N/A") or "N/A")[: self.PID_W]
        ip_address = (getattr(record, "ip_address", "no_ip") or "no_ip")[: self.IP_W]
        user_name = (getattr(record, "user_name", "Anonymous") or "Anonymous")[
            :15
        ]  # Add width limit
        tool_base = self._derive_tool_base(record)
        func_name = (getattr(record, "tool_name", "N/A") or "N/A")[: self.FUNC_W]
        ts = datetime.datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Prefix token
        prefix = "[-]" if is_error_or_warn else "[+]"
        if prefix == "[+]":
            prefix_colored = f"{self._c(GREY) if is_get else self._c(GREEN)}{prefix}"
        else:
            prefix_colored = f"{self._c(RED)}{prefix}"

        # Timestamp -> White
        ts_colored = f"{self._c(WHITE)}{ts}"

        # Project ID -> Blue
        proj_colored = f"{self._c(BLUE)}{package_id:<{self.PID_W}}"

        # IP -> Orange
        ip_colored = f"{self._c(ORANGE)}{ip_address:<{self.IP_W}}"

        # User Name -> Add a color (e.g., Cyan/Blue)
        user_colored = (
            f"{self._c(BLUE)}{user_name:<15}"  # Using BLUE, or define a new color
        )

        # Process: POST -> Green; otherwise White
        if process == "POST":
            proc_colored = f"{self._c(GREEN)}{process:<{self.PROC_W}}"
        else:
            proc_colored = f"{self._c(WHITE)}{process:<{self.PROC_W}}"

        # Dashes are Red
        dash = f"{self._c(RED)} - "

        # Level: INFO -> Purple; ERROR/CRITICAL -> Red; others -> Purple
        if is_error:
            level_colored = f"{self._c(RED)}{record.levelname:<{self.LEVEL_W}}"
        else:
            level_colored = f"{self._c(PURPLE)}{record.levelname:<{self.LEVEL_W}}"

        # Tail (tool base, function, message) -> Grey
        tool_colored = f"{self._c(GREY)}{tool_base:<{self.TOOL_W}}"
        func_colored = f"{self._c(GREY)}{func_name:<{self.FUNC_W}}"
        msg = record.getMessage()
        tail_msg_colored = f"{self._c(GREY)}{msg}"

        # Assemble line with user_name added after IP or wherever you prefer
        line = (
            f"{prefix_colored} "
            f"{ts_colored} "
            f"{proj_colored} "
            f"{ip_colored} "
            f"{user_colored} "  # Added user_name here
            f"{proc_colored}"
            f"{dash}"
            f"{level_colored}"
            f"{dash}"
            f"{tool_colored}: {func_colored} "
            f"{tail_msg_colored}"
        )
        if self.color:
            line += RESET

        # Exceptions/stack
        if record.exc_info:
            line += "\n" + super().formatException(record.exc_info)
            if self.color:
                line += RESET
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
            if self.color:
                line += RESET

        return line


def main():
    setup_logging()

    context = {
        "tool_name": "detector",
        "package_id": "123",
        "ip_address": "10.11,12",
        "request_type": "POST",
    }
    logger = logging.getLogger("TrubuildBE")
    logger = logging.LoggerAdapter(logger, context)
    logger.info("info message")
    logger.error("error message")
    try:
        raise ValueError("simulated error")
    except Exception:
        logger.exception("exception message")

    print("\n--- PID TOOL LOGGER ---")
    debug_logger = pid_tool_logger("123", "detector")

    print("Sending debug message...")
    debug_logger.debug("this is a debug message")

    print("Sending error message...")
    debug_logger.error("This is a pid_tool error log")

    print("Sending info message...")
    debug_logger.info("pid_tool info message")

    print("Sending exception...")
    try:
        raise ValueError("simulated error")
    except Exception:
        debug_logger.exception("pid_tool logger exception message")


if __name__ == "__main__":
    main()
