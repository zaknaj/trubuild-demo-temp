import os
import threading
from typing import Optional
from slack_sdk import WebClient
from dataclasses import dataclass
from slack_sdk.errors import SlackApiError
from utils.vault import secrets

env = secrets.get("CLIENT_API_URL", default="Unknown") or "Unknown"
_ENV = "Unknown"

if env == "https://dev.trubuild.io":
    _ENV = "Dev"
elif env == "https://staging.trubuild.io":
    _ENV = "Stage"
elif env == "https://app.trubuild.io":
    _ENV = "Prod"

_CHANNEL_ID = secrets.get("channel_id", default="") or ""
_TOKEN = secrets.get("slack_token", default="") or ""
_client = WebClient(token=_TOKEN)


@dataclass
class SlackActivityMeta:
    package_id: str
    tool: str
    user: str
    environment: str = _ENV
    company: Optional[str] = None


def fmt_dur(sec: float) -> str:
    if sec is None:
        return "0s"
    total_seconds = int(round(float(sec)))
    return f"{total_seconds}s"


class SlackActivityLogger:
    """
    Structured Slack activity logger:
      - start(): parent message (first line)
      - sub():   sub tool line(s) in thread
      - done():  final DONE
      - error(): final ERROR — details
    """

    def __init__(
        self,
        meta: SlackActivityMeta,
        channel_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ):
        if not _TOKEN or not (_CHANNEL_ID or channel_id):
            raise EnvironmentError("Missing SLACK_TOKEN or CHANNEL_ID")
        self.meta = meta
        self.channel_id = channel_id or _CHANNEL_ID
        self.thread_ts = thread_ts

    @property
    def header_text(self) -> str:
        company = self.meta.company or "-"
        return f"ENV={self.meta.environment} | USER={self.meta.user} | TOOL={self.meta.tool} | PACKAGE={self.meta.package_id} | COMPANY={company}"

    def start(self) -> str:
        """Post the parent message (first line) and capture thread_ts."""
        resp = _client.chat_postMessage(channel=self.channel_id, text=self.header_text)
        self.thread_ts = resp["ts"]
        return self.thread_ts

    def sub(self, text: str) -> None:
        """Post a sub-tool log line as a thread reply."""
        if not self.thread_ts:
            self.start()
        _client.chat_postMessage(
            channel=self.channel_id, text=text, thread_ts=self.thread_ts
        )

    def done(self) -> None:
        """Post explicit DONE as final line."""
        if not self.thread_ts:
            self.start()
        _client.chat_postMessage(
            channel=self.channel_id, text="DONE", thread_ts=self.thread_ts
        )

    def error(self, error_text: str) -> None:
        """Post explicit ERROR as final line."""
        if not self.thread_ts:
            self.start()
        _client.chat_postMessage(
            channel=self.channel_id,
            text=f"ERROR — {error_text}",
            thread_ts=self.thread_ts,
        )

    def run_block(self, fn, *args, **kwargs):
        """
        Convenience for sync blocks:
            def work(): ...
            log.run_block(work)
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self.error(str(e))
            raise
        finally:
            pass


def main() -> bool:

    from utils.core.log import pid_tool_logger, set_logger

    print("SLACK LOGGER TEST START")

    logger = pid_tool_logger(package_id="SYSTEM_CHECK", tool_name="activity")
    set_logger(
        logger,
        tool_name="activity_main",
        tool_base="SLACK",
        package_id="SYSTEM_CHECK",
        ip_address="local",
        request_type="N/A",
    )

    ok = True

    token = secrets.get("slack_token", default="") or ""
    channel_id = secrets.get("channel_id", default="") or ""
    if not token or not channel_id:
        logger.warning(
            "SLACK_TOKEN or CHANNEL_ID missing; skipping ActivityLogger Slack test."
        )
        print("ACTIVITY LOGGER OK")
        return True

    try:
        _client.auth_test()
    except SlackApiError as e:
        ok = False
        err = getattr(e, "response", {}).get("error", str(e))
        logger.exception(f"Slack auth_test failed: {err}")
        print("SLACK LOGGER FAIL", flush=True)
        return ok

    try:
        user = "system"
        test_package_id = "SYSTEM_CHECK"

        act = SlackActivityLogger(
            SlackActivityMeta(package_id=test_package_id, tool="TEST", user=user)
        )
        thread_ts = act.start()

        act.sub("Sub-step 1: parent message posted.")

        nonlocal_ok = [True]

        def _thread_fn(ts: str):
            try:
                wact = SlackActivityLogger(
                    SlackActivityMeta(
                        package_id=test_package_id, tool="TEST", user=user
                    ),
                    thread_ts=ts,
                )
                wact.sub("Sub-step 2: reply from worker thread.")
            except Exception as e:
                logger.exception(f"Thread logging failed: {e}")
                nonlocal_ok[0] = False

        t = threading.Thread(target=_thread_fn, args=(thread_ts,), daemon=True)
        t.start()
        t.join(timeout=2.0)

        ok = ok and nonlocal_ok[0]

        # Final line (explicit DONE)
        act.done()

    except SlackApiError as e:
        ok = False
        err = getattr(e, "response", {}).get("error", str(e))
        logger.exception(f"Slack API error: {err}")
    except Exception as e:
        ok = False
        logger.exception(f"SLACK LOGGER test failed: {e}")

    print("SLACK LOGGER OK" if ok else "SLACK LOGGER FAIL")
    return ok


if __name__ == "__main__":
    main()
