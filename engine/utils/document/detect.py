"""
AI Detector using Sapling

- Calls Sapling AI Detector API to score likelihood text is AI-generated.
- Supports inline text, file(s), or directories (recursive).
- Automatically chunks long inputs (<= 200k chars / request).
- Aggregates chunk scores (length-weighted mean) for an overall score.
- Optional per-sentence scores and token heatmap string.
- Character-rate limiting (default: 120k chars per 120s) and concurrency control.
- Retries with exponential backoff + jitter on 429/5xx/timeouts.
- Outputs JSON Lines (JSONL).

Auth
- Reads API key from env SAPLING_API_KEY.
- Uses Authorization: Bearer <key> header by default.

Docs:
- Endpoint & params: https://api.sapling.ai/api/v1/aidetect
- Request fields: key or Authorization header, text, sent_scores, score_string, version
- Limits: <= 200,000 characters / request
- Rate limits: ~120,000 characters / 2 minutes, 429 on exceed

"""
from __future__ import annotations

import json
import time
import random
import asyncio
import dataclasses
from pathlib import Path
from collections import deque
from utils.core.log import get_logger
from utils.document.doc import process_document_parts
from typing import Iterable, List, Optional, Dict, Any, Tuple

import httpx

API_URL = "https://api.sapling.ai/api/v1/aidetect"
DEFAULT_TIMEOUT = 30.0
DEFAULT_CHAR_BUDGET = 120_000
DEFAULT_CHAR_WINDOW_SEC = 120
MAX_REQUEST_CHARS = 200_000



class CharRateLimiter:
    """
    Sliding-window char budget limiter.
    Tracks (timestamp, chars) for last `window_sec` seconds.
    Before sending a request with N chars, wait until sum+N <= budget.
    """
    def __init__(self, budget_chars: int, window_sec: int):
        self.budget_chars = budget_chars
        self.window_sec = window_sec
        self.events = deque()
        self._lock = asyncio.Lock()
        self.logger = get_logger()
    async def acquire(self, needed_chars: int) -> None:
        async with self._lock:
            now = time.monotonic()
            self._evict_old(now)
            while self._current_sum() + needed_chars > self.budget_chars:
                # Sleep until the oldest event falls out of the window.
                oldest_ts, _ = self.events[0]
                sleep_for = max(0.0, (oldest_ts + self.window_sec) - now)
                self.logger.debug(f"Rate limit: sleeping {sleep_for:.2f}s")
                await asyncio.sleep(min(sleep_for, 2.0))
                now = time.monotonic()
                self._evict_old(now)
            # Reserve
            self.events.append((now, needed_chars))

    def _evict_old(self, now: float) -> None:
        while self.events and (now - self.events[0][0]) > self.window_sec:
            self.events.popleft()

    def _current_sum(self) -> int:
        return sum(chars for _, chars in self.events)

# Data Structures

@dataclasses.dataclass
class DetectRequest:
    source_id: str
    text: str
    idx: int = 0
    start: int = 0
    end: int = 0

@dataclasses.dataclass
class DetectResponse:
    source_id: str
    chunk_index: int
    start: int
    end: int
    length: int
    score: float
    sentence_scores: Optional[List[Dict[str, Any]]] = None

def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    with path.open("r", encoding=encoding, errors="ignore") as f:
        return f.read()

def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        for p in path.rglob("*"):
            if p.is_file():
                yield p

def chunk_text(text: str, max_len: int = MAX_REQUEST_CHARS) -> List[Tuple[int, int, str]]:
    """
    Return list of (start, end, chunk_text) where end is exclusive.
    Prefer paragraph boundaries; fall back to hard splits.
    """
    if len(text) <= max_len:
        return [(0, len(text), text)]

    # Try paragraph-aware chunking
    paras = text.split("\n\n")
    chunks: List[Tuple[int, int, str]] = []
    buf, start_idx = [], 0
    current_len = 0

    idx = 0
    for para in paras:
        add_len = len(para) + (2 if buf else 0)  # account for "\n\n" joins
        if current_len + add_len <= max_len:
            buf.append(para)
            current_len += add_len
        else:
            # flush buffer
            chunk_str = "\n\n".join(buf)
            end_idx = start_idx + len(chunk_str)
            chunks.append((start_idx, end_idx, chunk_str))
            # start new buffer with current para
            start_idx = end_idx + 2  # approximate; okay for aggregation
            buf = [para]
            current_len = len(para)

    if buf:
        chunk_str = "\n\n".join(buf)
        end_idx = start_idx + len(chunk_str)
        chunks.append((start_idx, end_idx, chunk_str))

    # If any chunk > max_len due to weird inputs, hard-split them.
    fixed: List[Tuple[int, int, str]] = []
    for s, e, ch in chunks:
        if len(ch) <= max_len:
            fixed.append((s, e, ch))
        else:
            offs = 0
            while offs < len(ch):
                sub = ch[offs:offs + max_len]
                fixed.append((s + offs, s + offs + len(sub), sub))
                offs += max_len
    return fixed

def length_weighted_mean(pairs: Iterable[Tuple[float, int]]) -> float:
    total_weight = 0
    total = 0.0
    for score, weight in pairs:
        total += score * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0

async def call_aidetect(
    client: httpx.AsyncClient,
    text: str,
    *,
    api_key: str,
    sent_scores: bool,
    score_string: bool,
    version: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    logger = get_logger()
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "text": text,
        "sent_scores": sent_scores,
        "score_string": score_string,
    }
    if version:
        payload["version"] = version

    # Retry policy
    max_attempts = 6
    backoff = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 200 and resp.status_code < 300:
                return resp.json()
            elif resp.status_code in (408, 409, 425, 429) or 500 <= resp.status_code < 600:
                # Retryable
                jitter = random.uniform(0, 0.25 * backoff)
                sleep_for = min(8.0, backoff + jitter)
                logger.warning(f"Retryable error {resp.status_code}; attempt {attempt}/{max_attempts}; sleeping {sleep_for:.2f}s")
                await asyncio.sleep(sleep_for)
                backoff *= 2
                continue
            else:
                raise httpx.HTTPStatusError(f"Non-retryable status {resp.status_code}: {resp.text}", request=resp.request, response=resp)
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ConnectError) as e:
            if attempt == max_attempts:
                raise
            jitter = random.uniform(0, 0.25 * backoff)
            sleep_for = min(8.0, backoff + jitter)
            logger.warning(f"{e.__class__.__name__}: attempt {attempt}/{max_attempts}; sleeping {sleep_for:.2f}s")
            await asyncio.sleep(sleep_for)
            backoff *= 2
    raise RuntimeError("Exhausted retries")


async def process_request(
    req: DetectRequest,
    client: httpx.AsyncClient,
    limiter: CharRateLimiter,
    *,
    api_key: str,
    sent_scores: bool,
    score_string: bool,
    version: Optional[str],
    timeout: float,
) -> DetectResponse:
    await limiter.acquire(len(req.text))
    result = await call_aidetect(
        client, req.text,
        api_key=api_key,
        sent_scores=sent_scores,
        score_string=score_string,
        version=version,
        timeout=timeout,
    )
    score = float(result.get("score", 0.0))
    return DetectResponse(
        source_id=req.source_id,
        chunk_index=req.idx,
        start=req.start,
        end=req.end,
        length=len(req.text),
        score=score,
        sentence_scores=result.get("sentence_scores"),
    )

async def run_detection(
    inputs: List[Tuple[str, str]],  # (source_id, text)
    *,
    api_key: str,
    sent_scores: bool,
    score_string: bool,
    version: Optional[str],
    timeout: float,
    concurrency: int,
    char_budget: int,
    char_window_sec: int,
) -> List[Dict[str, Any]]:
    logger = get_logger()
    # Build chunked request list
    requests: List[DetectRequest] = []
    for source_id, text in inputs:
        chunks = chunk_text(text, MAX_REQUEST_CHARS)
        for idx, (s, e, ch) in enumerate(chunks):
            requests.append(DetectRequest(source_id=source_id, text=ch, idx=idx, start=s, end=e))

    limiter = CharRateLimiter(char_budget, char_window_sec)
    results: List[DetectResponse] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(http2=True) as client:
        async def worker(req: DetectRequest):
            async with sem:
                try:
                    return await process_request(
                        req, client, limiter,
                        api_key=api_key,
                        sent_scores=sent_scores,
                        score_string=score_string,
                        version=version,
                        timeout=timeout,
                    )
                except Exception as e:
                    logger.error(f"[{req.source_id}#{req.idx}] failed: {e}")
                    raise

        tasks = [asyncio.create_task(worker(r)) for r in requests]
        for t in asyncio.as_completed(tasks):
            res = await t
            results.append(res)

    # Aggregate by source_id
    aggregated: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, List[DetectResponse]] = {}
    for r in results:
        by_source.setdefault(r.source_id, []).append(r)

    for source_id, items in by_source.items():
        items.sort(key=lambda x: x.chunk_index)
        overall = length_weighted_mean((it.score, it.length) for it in items)
        aggregated[source_id] = {
            "source": source_id,
            "score": overall,
            "chunks": [
                {
                    "index": it.chunk_index,
                    "start": it.start,
                    "end": it.end,
                    "length": it.length,
                    "score": it.score,
                }
                for it in items
            ],
        }
    # Return list of per-source summaries
    return list(aggregated.values())

API_KEY = "305L74K86A5JJP11SXP3RTTWHFDRI4OF"
VERSION = None                          # None for default
SENT_SCORES = True                      # per-sentence probabilities
SCORE_STRING = False                    # token-level heatmap string (HTML)

CONCURRENCY = 4
TIMEOUT = DEFAULT_TIMEOUT
CHAR_BUDGET = DEFAULT_CHAR_BUDGET       # ~120_000 chars / 120s by default
CHAR_WINDOW_SEC = DEFAULT_CHAR_WINDOW_SEC

TEXT_INPUT: Optional[str] = None
# TEXT_INPUT = "Paste a long passage here to test Sapling AI detection."

DOCUMENTS  = ['/home/development/TruBuildBE/utils/rfp.docx']

# Output controls
JSONL_OUT: Optional[Path] = Path("results.jsonl")
SUMMARY = True
VERBOSE = True
def _list_files(paths: List[str]) -> List[Path]:
    """Expand directories recursively; return a de-duplicated, deterministic file list."""
    logger = get_logger()
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in iter_files(p):      # uses earlier helper
                files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            logger.warning(f"Path not found or not a file/dir: {raw}")
    # deterministic order for reproducibility
    uniq = sorted({str(f.resolve()) for f in files})
    return [Path(u) for u in uniq]


def _extract_text_from_outputs(
    doc_name: str,
    parts: List[Any],
    tokenized_pages: List[dict],
) -> str:
    """
    Prefer structured text from tokenized_pages (type == 'text').
    Fall back to text-bearing Part objects if present.
    Excludes the open/close tags injected by process_document_parts.
    """
    # 1) Preferred: tokenized_pages content
    page_texts: List[str] = []
    for t in tokenized_pages:
        if isinstance(t, dict) and t.get("type") == "text":
            c = t.get("content")
            if isinstance(c, str) and c.strip():
                page_texts.append(c)
    if page_texts:
        return "\n\n".join(page_texts)

    # 2) Fallback: scan Part objects for text payloads
    texts: List[str] = []
    for p in parts or []:
        # try common attribute
        txt = getattr(p, "text", None)
        if isinstance(txt, str) and txt.strip():
            # skip the synthetic open/close markers
            if txt.strip() == f"<{doc_name}>" or txt.strip() == f"</{doc_name}>":
                continue
            texts.append(txt)
    return "\n\n".join(texts)


def _collect_inputs_via_document_processor(doc_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Run your utils.doc.process_document_parts on each file and return
    a list of (source_id, extracted_text) for Sapling.
    """
    logger = get_logger()
    inputs: List[Tuple[str, str]] = []
    files = _list_files(doc_paths)
    if not files:
        raise SystemExit("No input files found in DOCUMENTS.")

    for idx, fpath in enumerate(files):
        doc_path = str(fpath)
        doc_name = fpath.name
        try:
            parts, errors, tokenized_pages = process_document_parts(doc_path, doc_name, idx)

            # Log any user-friendly errors returned by your processor
            if errors:
                for e in errors:
                    # e is expected like {"file": ..., "error": ..., "user_friendly": True}
                    logger.warning(f"[{doc_name}] extractor warning: {e.get('error') or e}")

            text = _extract_text_from_outputs(doc_name, parts, tokenized_pages)
            if not text.strip():
                logger.info(f"[{doc_name}] no text extracted; skipping.")
                continue

            # Use the absolute path as the source_id for traceability
            inputs.append((str(fpath.resolve()), text))
        except Exception as ex:
            logger.error(f"[{doc_name}] processing failed: {ex}")

    if not inputs:
        raise SystemExit("No text could be extracted from the provided DOCUMENTS.")
    return inputs


def print_summary(results: List[Dict[str, Any]]) -> None:
    for r in results:
        print(f"{r['source']}: score={r['score']:.4f}  (chunks={len(r['chunks'])})")


def main() -> None:
    from utils.core.log import set_logger, pid_tool_logger
    logger = pid_tool_logger("SYSTEM_CHECK", "DETECT")
    set_logger(logger)

    if not API_KEY or API_KEY.startswith("YOUR_SAPLING_API_KEY"):
        raise SystemExit("Missing API key. Set API_KEY in the config section.")

    # Build inputs from your document pipeline
    inputs = _collect_inputs_via_document_processor(DOCUMENTS)
    logger.info(f"Prepared {len(inputs)} document(s) with extracted text.")

    # Run Sapling detection (handles chunking, retries, rate limits)
    results = asyncio.run(
        run_detection(
            inputs,
            api_key=API_KEY,
            sent_scores=SENT_SCORES,
            score_string=SCORE_STRING,
            version=VERSION,
            timeout=TIMEOUT,
            concurrency=CONCURRENCY,
            char_budget=CHAR_BUDGET,
            char_window_sec=CHAR_WINDOW_SEC,
        )
    )

    # Persist machine-readable results if requested
    if JSONL_OUT:
        with JSONL_OUT.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Wrote JSONL to {JSONL_OUT}")

    # Human-readable summary
    if SUMMARY or not JSONL_OUT:
        print_summary(results)


if __name__ == "__main__":
    main()
