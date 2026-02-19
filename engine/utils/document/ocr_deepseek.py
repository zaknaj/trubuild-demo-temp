import os
import json
import time
import base64
import httpx
import threading
import google.auth
from utils.core.log import get_logger
from google.auth.transport.requests import Request as GoogleRequest


ENABLE_IMAGE_OCR = False

DEEPSEEK_OCR_MODEL = "deepseek-ai/deepseek-ocr-maas"
DEEPSEEK_ENDPOINT = "https://aiplatform.googleapis.com"
DEEPSEEK_REGION = "global"
DEEPSEEK_PROJECT = "gen-lang-client-0502418869"

_OCR_MAX_ATTEMPTS = 3

# Per-request network timeouts
_OCR_HTTP_TIMEOUT_SEC = 60
_OCR_AUTH_TIMEOUT_SEC = 30

_OCR_CONCURRENCY = 3
_OCR_SEM = threading.Semaphore(_OCR_CONCURRENCY)


class ShortTimeoutRequest(GoogleRequest):
    """Request wrapper that enforces a sane timeout for auth HTTP calls."""

    def __init__(self, timeout: float = _OCR_AUTH_TIMEOUT_SEC, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeout_override = timeout

    def __call__(
        self, url, method="GET", body=None, headers=None, timeout=None, **kwargs
    ):
        if timeout is None:
            timeout = self._timeout_override
        return super().__call__(
            url, method=method, body=body, headers=headers, timeout=timeout, **kwargs
        )


def _get_gcloud_access_token(timeout: float = _OCR_AUTH_TIMEOUT_SEC) -> str:
    """
    Get an OAuth2 access token using application default credentials.

    - Uses a ShortTimeoutRequest so auth cannot hang forever.
    - Logs timing and failures.
    """
    logger = get_logger()
    start = time.time()
    logger.debug("[OCR] Fetching GCP access token (timeout=%.1fs)", timeout)

    try:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not creds.valid:
            creds.refresh(ShortTimeoutRequest(timeout=timeout))
        token = creds.token or ""
        elapsed = time.time() - start
        logger.debug("[OCR] Got GCP access token in %.2fs", elapsed)
        return token
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "[OCR] Failed to obtain GCP access token after %.2fs: %s",
            elapsed,
            e,
        )
        return ""


def _deepseek_ocr_call_from_bytes(
    image_bytes: bytes, mime_type: str = "image/png"
) -> str:
    """
    Calls DeepSeek OCR via the Vertex AI endpoint, with:

    - bounded auth + HTTP timeouts
    - a small, explicit retry loop
    - detailed logging per attempt
    """
    logger = get_logger()

    if not DEEPSEEK_PROJECT:
        logger.error(
            "[OCR] DEEPSEEK_PROJECT / GOOGLE_CLOUD_PROJECT not set; cannot call DeepSeek OCR."
        )
        return ""

    url = (
        f"{DEEPSEEK_ENDPOINT}/v1/projects/{DEEPSEEK_PROJECT}"
        f"/locations/{DEEPSEEK_REGION}/endpoints/openapi/chat/completions"
    )

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime_type};base64,{b64}"

    payload = {
        "model": DEEPSEEK_OCR_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Perform OCR on the attached document. "
                            "Return the full text in logical reading order. "
                            "Do NOT add any explanation, only the extracted text. "
                            "Extract in the language you see."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": data_uri,
                    },
                ],
            }
        ],
    }

    for attempt in range(1, _OCR_MAX_ATTEMPTS + 1):
        start = time.time()
        logger.debug(
            "[OCR] Attempt %d/%d: calling DeepSeek OCR (bytes=%d, mime=%s)",
            attempt,
            _OCR_MAX_ATTEMPTS,
            len(image_bytes),
            mime_type,
        )

        try:
            token = _get_gcloud_access_token(timeout=_OCR_AUTH_TIMEOUT_SEC)
            if not token:
                logger.error(
                    "[OCR] Attempt %d/%d: no access token; aborting.",
                    attempt,
                    _OCR_MAX_ATTEMPTS,
                )
                return ""

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            resp = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=_OCR_HTTP_TIMEOUT_SEC,
            )
            elapsed = time.time() - start
            logger.debug(
                "[OCR] Attempt %d/%d: HTTP %s in %.2fs",
                attempt,
                _OCR_MAX_ATTEMPTS,
                resp.status_code,
                elapsed,
            )

            if resp.status_code != 200:
                logger.error(
                    "[OCR] Attempt %d/%d: HTTP error %s %s | body=%s",
                    attempt,
                    _OCR_MAX_ATTEMPTS,
                    resp.status_code,
                    resp.reason_phrase,
                    resp.text[:500],
                )
                # fall through to retry if any left
                if attempt < _OCR_MAX_ATTEMPTS:
                    time.sleep(min(2**attempt, 10))
                    continue
                return ""

            try:
                data = resp.json()
            except Exception as e:
                logger.error(
                    "[OCR] Attempt %d/%d: failed to parse JSON (%s) | body=%s",
                    attempt,
                    _OCR_MAX_ATTEMPTS,
                    e,
                    resp.text[:500],
                )
                if attempt < _OCR_MAX_ATTEMPTS:
                    time.sleep(min(2**attempt, 10))
                    continue
                return ""

            # choices[0].message.content
            choices = data.get("choices") or []
            if not choices:
                logger.warning(
                    "[OCR] Attempt %d/%d: no choices in response.",
                    attempt,
                    _OCR_MAX_ATTEMPTS,
                )
                return ""

            msg = (choices[0] or {}).get("message", {}) or {}
            content = msg.get("content")

            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                texts = []
                for seg in content:
                    if isinstance(seg, str):
                        texts.append(seg)
                    elif isinstance(seg, dict):
                        t = seg.get("text") or seg.get("content") or ""
                        if t:
                            texts.append(str(t))
                text = "\n".join(t.strip() for t in texts if t and str(t).strip())
            else:
                text = ""

            logger.debug(
                "[OCR] Attempt %d/%d: extracted text length=%d",
                attempt,
                _OCR_MAX_ATTEMPTS,
                len(text or ""),
            )
            return text.strip()

        except httpx.RequestError as e:
            elapsed = time.time() - start
            logger.error(
                "[OCR] Attempt %d/%d: network error after %.2fs: %s",
                attempt,
                _OCR_MAX_ATTEMPTS,
                elapsed,
                e,
            )
            if attempt < _OCR_MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 10))
                continue
            return ""
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "[OCR] Attempt %d/%d: unexpected error after %.2fs: %s",
                attempt,
                _OCR_MAX_ATTEMPTS,
                elapsed,
                e,
            )
            if attempt < _OCR_MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 10))
                continue
            return ""

    logger.error("[OCR] Exhausted all %d attempts with no success.", _OCR_MAX_ATTEMPTS)
    return ""


def collapse_consecutive_duplicate_lines(text: str) -> str:
    lines = text.splitlines()
    out = []
    prev = None
    for line in lines:
        if line.strip() and line == prev:
            # skip exact consecutive duplicate
            continue
        out.append(line)
        prev = line
    return "\n".join(out)


def ocr_image_to_text(
    image_bytes: bytes,
    mime_type: str = "image/png",
    *,
    filename_hint: str = "ocr_image.png",
) -> str:
    """
    Public entry point for OCR.

    - Honors ENABLE_IMAGE_OCR.
    - Limits concurrency via _OCR_SEM.
    - Logs start/end, duration, and text length.
    - Never raises; returns "" on any failure.
    """
    logger = get_logger()

    if not ENABLE_IMAGE_OCR:
        logger.debug(
            "[OCR] Skipping OCR for %s (ENABLE_IMAGE_OCR disabled).", filename_hint
        )
        return ""

    # Concurrency guard so we don't have 6+ threads all doing OCR at the same time.
    with _OCR_SEM:
        start = time.time()
        logger.debug(
            "[OCR] Starting OCR for %s (bytes=%d, mime=%s)",
            filename_hint,
            len(image_bytes),
            mime_type,
        )

        try:
            raw_text = _deepseek_ocr_call_from_bytes(image_bytes, mime_type=mime_type)
            cleaned = collapse_consecutive_duplicate_lines(raw_text or "")
            elapsed = time.time() - start
            logger.debug(
                "[OCR] Finished OCR for %s in %.2fs (len(text)=%d)",
                filename_hint,
                elapsed,
                len(cleaned),
            )
            return cleaned
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "[OCR] OCR failed for %s after %.2fs: %s",
                filename_hint,
                elapsed,
                e,
            )
            return ""


# Testing
import os
import io
import argparse
import json
from typing import List
from google.genai.types import Part

from utils.document.doc import process_document_parts
from utils.core.log import get_logger, pid_tool_logger, set_logger

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


def init_logger() -> None:
    package_id = "ocr_pdf_check"
    base_logger = pid_tool_logger(package_id, "ocr_pdf_check")
    set_logger(
        base_logger,
        tool_name="ocr_pdf_check",
        package_id=package_id,
        ip_address="local",
        request_type="LOCAL",
    )


def _summarize_parts_for_text_dump(parts: List[Part]) -> str:
    """
    Produce a human-readable summary of all TEXT parts, including OCR text,
    for inspection. This is what we write to the .txt file.
    """
    out_lines: List[str] = []
    for idx, p in enumerate(parts):
        text = getattr(p, "text", None)
        if isinstance(text, str) and text.strip():
            out_lines.append(f"\n=== PART #{idx} [TEXT] ===\n")
            out_lines.append(text)
    if not out_lines:
        return "[NO TEXT PARTS]\n"
    return "".join(out_lines)


def debug_ocr_pdf_pipeline(pdf_path: str) -> None:
    """
    Run the full document extraction pipeline (including OCR) on a single PDF
    and dump all extracted text (native + OCR) into a .txt file in the CWD.
    """
    logger = get_logger()

    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return

    doc_name = os.path.basename(pdf_path)
    print(f"[INFO] Running OCR PDF pipeline test on: {pdf_path} (name={doc_name})")

    parts, errors, tokenized_pages = process_document_parts(pdf_path, doc_name, 0)

    # Log / print any errors
    if errors:
        print("\n[WARN] Extraction errors:")
        for e in errors:
            print(f"  - file={e.get('file')}: {e.get('errors') or []}")
            logger.debug("OCR PDF test error: %s", e.get("technical", ""))

    # Build a text dump of all textual content (including OCR)
    dump_text = _summarize_parts_for_text_dump(parts)

    # Save to txt in the root (current working directory)
    safe_name = os.path.splitext(doc_name)[0].replace(" ", "_")
    out_filename = f"ocr_pdf_{safe_name}.txt"
    out_path = os.path.join(os.getcwd(), out_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"[PDF] {pdf_path}\n")
        f.write(f"[TOTAL PARTS] {len(parts)}\n")
        f.write(f"[TOKENIZED PAGES] {len(tokenized_pages)}\n\n")
        f.write(dump_text)

        if tokenized_pages:
            f.write("\n\n=== TOKENIZED PAGES (first 30) ===\n")
            for t in tokenized_pages[:30]:
                f.write(json.dumps(t, ensure_ascii=False))
                f.write("\n")

    print(f"[INFO] OCR PDF pipeline result written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek OCR PDF pipeline tester")
    parser.add_argument(
        "pdf",
        help="Path to a local PDF file to run through the extraction + OCR pipeline.",
    )
    args = parser.parse_args()

    init_logger()
    debug_ocr_pdf_pipeline(args.pdf)
