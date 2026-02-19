from datetime import datetime, UTC


def _make_error_payload(
    stage: str, err: Exception | str, extra: dict | None = None
) -> dict:
    msg = str(err)
    base = {
        "status": "error",
        "error": msg,
        "stage": stage,
        "timestamp": datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }
    if extra:
        base.update(extra)
    return base
