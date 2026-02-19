"""
Central warning filter configuration for engine runtime processes.
"""

import warnings


def configure_warning_filters() -> None:
    """Silence known noisy library warnings."""
    warnings.filterwarnings(
        "ignore",
        message="style lookup by style_id is deprecated. Use style name as key instead.",
        module=r"docx\.styles\.styles",
    )
    warnings.filterwarnings(
        "ignore",
        message="Cannot parse header or footer",
        module=r"openpyxl\.worksheet\.header_footer",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*extension is not supported and will be removed",
        module=r"openpyxl\.worksheet\._reader",
    )

