# Embedding-free, recursive compression
# - Falls back to 4 chars/token when count_tokens() fails
# - Summarizes ~500k-token chunks with overlap
# - Recursively compresses until under target; then hard-cap if needed
# - Logs whenever compression is used
# - summarizes images

# TODO: need to remove all legacy behavior

from __future__ import annotations
import re, os
import time
import random
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
import asyncio, concurrent.futures, contextvars
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from google import genai
from google.genai import errors as gerrors
from google.genai.types import GenerateContentConfig, ThinkingConfig, Part


from utils.core.log import get_logger

from collections import OrderedDict
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import field
from typing import Dict


from utils.llm.compactor_cache import (
    load_doc_summary_from_store,
    save_doc_summary_to_store,
    make_doccache_key_from_source_key,
    _norm_name
)

# Parallel preflight configuration
PREFLIGHT_MAX_WORKERS = 32
PREFLIGHT_TIMEOUT_PER_DOC = 10.0
SMALL_DOC_BATCH_TARGET_TOKENS = 50_000


MARKER_BEGIN_RFP = "### BEGIN RFP-REFERENCE"
MARKER_END_RFP = "### END RFP-REFERENCE"
MARKER_BEGIN_CONTRACTOR = "### BEGIN CONTRACTOR-SUBMISSION"
MARKER_END_CONTRACTOR = "### END CONTRACTOR-SUBMISSION"
TAG_OPEN_RE_SOFA     = re.compile(r'^\s*<Start of File:\s*(.+?)>\s*$')     # Start Of File Anchor from process document parts
TAG_CLOSE_RE_EOFA    = re.compile(r'^\s*</End of File:\s*(.+?)>\s*$')      # End Of File Anchor from process document parts

COMPACTOR_429_INITIAL_BACKOFF = 5
COMPACTOR_429_MAX_BACKOFF     = 90
COMPACTOR_429_MAX_ATTEMPTS    = 12
COMPACTOR_429_MAX_TOTAL       = 5 * 60      # total retry window per chunk: 5 min


COMPACTOR_5XX_MAX_ATTEMPTS = 5
COMPACTOR_5XX_MAX_BACKOFF  = 60

_CTX_COMPRESS_MEMO_MAX = 128
_CTX_COMPRESS_MEMO: "OrderedDict[str, List[Part]]" = OrderedDict()
_CTX_MEMO_LOCK = threading.Lock()
_CTX_INFLIGHT: dict[str, threading.Event] = {}

# Per-document compression cache
_DOC_COMPRESS_MEMO_MAX = 512
_DOC_COMPRESS_MEMO: "OrderedDict[str, List[Part]]" = OrderedDict()
_DOC_MEMO_LOCK = threading.Lock()
_DOC_INFLIGHT: dict[str, threading.Event] = {}

_DOC_CACHE_VERSION = "v1"
COMPACTOR_FALLBACK_MODEL = "gemini-2.5-flash"
COMPACTOR_FALLBACK_MODEL_ATTEMPTS = 2

def _settings_fingerprint(s: CompressionSettings) -> str:
    return "|".join(map(str, [
        s.summary_model,
        s.hard_limit_tokens,
        s.target_total_tokens,
        s.prompt_reserve_tokens,
        s.per_doc_ceiling_tokens,
        s.chunk_target_tokens,
        s.summary_max_output_tokens,
        s.max_rounds,
        s.max_compress_passes,
    ]))

def _is_resource_exhausted_err(e: Exception) -> bool:
    """
    Google genai errors vary across versions. We detect:
    - explicit codes
    - message contains RESOURCE_EXHAUSTED / quota language
    """
    code = getattr(e, "code", None)
    if code in (429, 499):
        return True

    msg = str(e).lower()
    if "resource_exhausted" in msg:
        return True
    if "quota" in msg and ("exceed" in msg or "exhaust" in msg):
        return True
    if "rate limit" in msg or "too many requests" in msg:
        return True

    # Sometimes there is a status field
    status = getattr(e, "status", None)
    if isinstance(status, str) and "resource_exhausted" in status.lower():
        return True

    return False

def _fallback_extract_from_parts(
    parts: list[Part],
    *,
    max_chars: int,
    chars_per_token: int,
) -> str:
    """
    Last-resort fallback: produce a truncated raw extract of the input.
    - Includes text parts
    - Includes minimal image/file references (file_uri or mime)
    - Keeps both head + tail for better recall
    """
    lines: list[str] = []

    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str) and t.strip():
            lines.append(t)
            continue

        # Non-text: include minimal stable reference
        fd = getattr(p, "file_data", None)
        if fd and getattr(fd, "file_uri", None):
            lines.append(f"[NON-TEXT PART: file_uri={fd.file_uri}]")
            continue

        mt = getattr(p, "mime_type", None)
        if mt:
            lines.append(f"[NON-TEXT PART: mime_type={mt}]")
        else:
            lines.append("[NON-TEXT PART]")

    raw = "\n".join(lines).strip()
    if not raw:
        return "[FALLBACK-EXTRACT] (empty content)"

    # Ensure max_chars is sensible even if caller passes tiny
    max_chars = max(256, int(max_chars))

    if len(raw) <= max_chars:
        return "[FALLBACK-EXTRACT]\n" + raw

    # Keep head + tail
    half = max_chars // 2
    head = raw[:half]
    tail = raw[-half:]
    return "[FALLBACK-EXTRACT]\n" + head + "\n...\n" + tail

def _match_open_tag(p: Part) -> tuple[str, Part] | None:
    t = getattr(p, "text", None)
    if not isinstance(t, str):
        return None
    m = TAG_OPEN_RE_SOFA.match(t)
    if m: return (m.group(1), p)
    return None

def _match_close_tag(p: Part) -> tuple[str, Part] | None:
    t = getattr(p, "text", None)
    if not isinstance(t, str):
        return None
    m = TAG_CLOSE_RE_EOFA.match(t)
    if m: return (m.group(1), p)
    return None

def _count_tokens_strict(model: str, parts: Sequence[Part], count_tokens) -> int:
    """
    count_tokens() returns -1 on failure in utils.LLM.
    Treat non-positive values as failures so we fall back to heuristics.
    """
    v = count_tokens(model, parts)
    if isinstance(v, int) and v > 0:
        return v
    raise RuntimeError(f"count_tokens failed (got {v})")

def _count_parts_safe_mixed(
    *, model: str, parts: list[Part], count_tokens, chars_per_token: int
) -> int:
    try:
        return _count_tokens_strict(model, parts, count_tokens)
    except Exception:
        txt = 0
        media = 0
        for p in parts:
            s = getattr(p, "text", None)
            if isinstance(s, str):
                txt += len(s)
            else:
                media += 1
        return max(1, txt // max(1, chars_per_token)) + media * 512  # heuristic


def _submit_with_context(executor: ThreadPoolExecutor, fn, *args, **kwargs):
    """
    Submit work to a thread while preserving ContextVars (logger context).
    """
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


def _normalize_key_for_prefix_check(s: str) -> str:
    # lower, forward slashes, strip spaces; drop URL scheme/bucket if present
    s = (s or "").strip().lower().replace("\\", "/")
    if "://" in s:
        s = s.split("://", 1)[1]
        s = s.split("/", 1)[1] if "/" in s else ""
    return re.sub(r"/+", "/", s)

def _is_tech_rfp_rfp_source(source_key: str) -> bool:
    prefix = _normalize_key_for_prefix_check(
        os.getenv("COMPACTOR_RFP_PREFIX", "tech_rfp/rfp/")
    )
    key = _normalize_key_for_prefix_check(source_key)
    return key.startswith(prefix)

@dataclass
class DocNode:
    """Represents a parsed document with its open/close tags and content."""
    name: str
    open_part: Part
    close_part: Optional[Part]  # None if unterminated
    content: List[Part]
    source_key: Optional[str] = None
    # Preflight results (populated by parallel preflight)
    tokens: int = 0
    cached_summary: Optional[List[Part]] = None
    cache_key: Optional[str] = None
    needs_summary: bool = False
    force_summarize: bool = False  # Set by budget-aware planner
    is_batched: bool = False  # True if this doc is part of a small-doc batch


@dataclass
class DocBatch:
    """Represents a batch of small documents to be summarized together."""
    docs: List[DocNode]
    total_tokens: int = 0
    batch_name: str = ""


def _parse_documents_from_context(
    context_parts: List[Part],
    name_to_source_key: Dict[str, str] | None = None,
) -> Tuple[List[DocNode], List[Part]]:
    """
    Single-pass parsing of context_parts into a list of DocNodes.
    Returns (doc_nodes, non_doc_parts) where non_doc_parts are parts outside any document.
    """
    doc_nodes: List[DocNode] = []
    non_doc_parts: List[Part] = []
    stack: List[dict] = []

    for p in context_parts:
        o = _match_open_tag(p)
        if o:
            name, open_part = o
            stack.append({"name": name, "open": open_part, "content": []})
            continue

        c = _match_close_tag(p)
        if c and stack:
            close_name, close_part = c
            if stack[-1]["name"] == close_name:
                node_dict = stack.pop()
                nm = _norm_name(node_dict["name"])
                sk = (name_to_source_key or {}).get(nm)

                doc_node = DocNode(
                    name=node_dict["name"],
                    open_part=node_dict["open"],
                    close_part=close_part,
                    content=node_dict["content"],
                    source_key=sk,
                )

                if stack:
                    # Nested document - add as content to parent
                    # We'll flatten the structure for the parent
                    rebuilt = [doc_node.open_part] + doc_node.content + [close_part]
                    stack[-1]["content"].extend(rebuilt)
                else:
                    doc_nodes.append(doc_node)
                continue

        # Normal part
        if stack:
            stack[-1]["content"].append(p)
        else:
            non_doc_parts.append(p)

    # Flush unterminated docs
    while stack:
        node_dict = stack.pop()
        nm = _norm_name(node_dict["name"])
        sk = (name_to_source_key or {}).get(nm)
        doc_node = DocNode(
            name=node_dict["name"],
            open_part=node_dict["open"],
            close_part=None,  # Unterminated
            content=node_dict["content"],
            source_key=sk,
        )
        doc_nodes.append(doc_node)

    return doc_nodes, non_doc_parts


def _estimate_tokens_heuristic(parts: List[Part], chars_per_token: int = 4) -> int:
    """
    FAST heuristic token estimation
    Used for preflight budget planning.
    """
    total_chars = 0
    media_count = 0
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            total_chars += len(t)
        else:
            media_count += 1
    # ~4 chars per token for text, ~512 tokens per media item (conservative)
    return max(1, total_chars // max(1, chars_per_token)) + media_count * 512


def _preflight_single_doc(
    doc: DocNode,
    model_for_counting: str,
    count_tokens,
    settings: "CompressionSettings",
) -> DocNode:
    """
    Preflight a single document: estimate tokens (heuristic) and check cache.
    This is designed to be called in parallel via ThreadPoolExecutor.

    IMPORTANT: Uses fast heuristic for token estimation, not the real count_tokens() API call.
    This keeps preflight fast for thousands of documents.
    """
    logger = get_logger()
    try:
        # FAST heuristic token estimation
        doc.tokens = _estimate_tokens_heuristic(doc.content, settings.chars_per_token)

        # Generate cache key
        doc.cache_key = _doc_cache_key(doc.name, doc.content, settings)

        # Check in-memory cache first
        with _DOC_MEMO_LOCK:
            cached_mem = _DOC_COMPRESS_MEMO.get(doc.cache_key)
            if cached_mem is not None:
                _DOC_COMPRESS_MEMO.move_to_end(doc.cache_key)
                doc.cached_summary = list(cached_mem)
                logger.debug("[PREFLIGHT] mem hit: name=%s tokens=%d", doc.name, doc.tokens)
                return doc

        # Check persistent store
        persisted = load_doc_summary_from_store(doc.cache_key)
        if persisted:
            doc.cached_summary = persisted
            # Also store in memory for future lookups
            with _DOC_MEMO_LOCK:
                _DOC_COMPRESS_MEMO[doc.cache_key] = list(persisted)
                _DOC_COMPRESS_MEMO.move_to_end(doc.cache_key)
                if len(_DOC_COMPRESS_MEMO) > _DOC_COMPRESS_MEMO_MAX:
                    _DOC_COMPRESS_MEMO.popitem(last=False)
            logger.debug("[PREFLIGHT] store hit: name=%s tokens=%d", doc.name, doc.tokens)
            return doc

        logger.debug("[PREFLIGHT] miss: name=%s tokens=%d", doc.name, doc.tokens)

    except Exception as e:
        logger.warning("[PREFLIGHT] error for %s: %s", doc.name, e)
        # Fallback: estimate tokens from content length
        txt_len = sum(len(getattr(p, "text", "") or "") for p in doc.content)
        doc.tokens = max(1, txt_len // settings.chars_per_token)

    return doc


def _preflight_documents_parallel(
    docs: List[DocNode],
    model_for_counting: str,
    count_tokens,
    settings: "CompressionSettings",
) -> List[DocNode]:
    """
    Run parallel preflight on all documents.
    Returns the same docs with preflight results populated.
    """
    logger = get_logger()
    if not docs:
        return docs

    t_start = time.perf_counter()
    n_docs = len(docs)

    # For small number of docs, don't bother with threading overhead
    if n_docs <= 3:
        for doc in docs:
            _preflight_single_doc(doc, model_for_counting, count_tokens, settings)
        dur = time.perf_counter() - t_start
        logger.debug("[PREFLIGHT] sequential: docs=%d dur=%.2fs", n_docs, dur)
        return docs

    # Parallel preflight
    max_workers = min(PREFLIGHT_MAX_WORKERS, n_docs)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            _submit_with_context(
                executor,
                _preflight_single_doc, doc, model_for_counting, count_tokens, settings
            ): i
            for i, doc in enumerate(docs)
        }

        for future in as_completed(futures, timeout=PREFLIGHT_TIMEOUT_PER_DOC * n_docs):
            try:
                future.result()
                completed += 1
            except Exception as e:
                idx = futures[future]
                logger.warning("[PREFLIGHT] failed for doc %d: %s", idx, e)

    dur = time.perf_counter() - t_start
    cache_hits = sum(1 for d in docs if d.cached_summary)
    logger.debug(
        "[PREFLIGHT] parallel: docs=%d workers=%d dur=%.2fs cache_hits=%d",
        n_docs, max_workers, dur, cache_hits
    )

    return docs


def _compute_summarization_plan(
    docs: List[DocNode],
    target_tokens: int,
    settings: "CompressionSettings",
) -> Tuple[List[DocNode], List[DocBatch]]:
    """
    Budget-aware planning: decide which documents need summarization.

    Returns:
        - docs_to_summarize: individual docs that need summarization
        - batches: batches of small docs to be summarized together
    """
    logger = get_logger()

    # Calculate current total
    total_tokens = 0
    for doc in docs:
        if doc.cached_summary:
            # Use cached summary token count
            cached_tokens = sum(
                len(getattr(p, "text", "") or "") // settings.chars_per_token
                for p in doc.cached_summary
            )
            total_tokens += max(1, cached_tokens)
        else:
            total_tokens += doc.tokens

    logger.debug("[PLAN] total_tokens=%d target=%d", total_tokens, target_tokens)

    if total_tokens <= target_tokens:
        # Already under budget, no summarization needed
        logger.debug("[PLAN] already under budget, no summarization needed")
        return [], []

    excess = total_tokens - target_tokens
    logger.debug("[PLAN] excess=%d tokens to reduce", excess)

    # Apply per-document hard ceiling first (if enabled).
    per_doc_ceiling = getattr(settings, "per_doc_ceiling_tokens", 0)
    if per_doc_ceiling > 0:
        forced = 0
        for doc in docs:
            if doc.cached_summary:
                continue
            if doc.tokens > per_doc_ceiling:
                doc.force_summarize = True
                doc.needs_summary = True
                forced += 1
        if forced:
            logger.debug(
                "[PLAN] forced by per-doc ceiling=%d: docs=%d",
                per_doc_ceiling,
                forced,
            )

    # Sort docs by size (descending) - biggest savings first
    # Only consider docs without cached summaries
    uncached_docs = [d for d in docs if not d.cached_summary]
    uncached_docs.sort(key=lambda d: d.tokens, reverse=True)

    docs_to_summarize: List[DocNode] = []
    small_docs: List[DocNode] = []
    projected_savings = 0

    # Minimum size to consider for individual summarization
    min_individual_threshold = max(1000, settings.chunk_target_tokens // 100)

    for doc in uncached_docs:
        if projected_savings >= excess:
            break

        if doc.force_summarize or doc.tokens >= min_individual_threshold:
            # Large enough for individual summarization
            doc.force_summarize = True
            doc.needs_summary = True
            docs_to_summarize.append(doc)
            # Estimate 80% reduction from summarization
            projected_savings += int(doc.tokens * 0.8)
        else:
            # Small doc - candidate for batching
            small_docs.append(doc)

    logger.debug(
        "[PLAN] individual summarization: %d docs, projected_savings=%d",
        len(docs_to_summarize), projected_savings
    )

    # If still not enough, batch small docs together
    batches: List[DocBatch] = []
    if projected_savings < excess and small_docs:
        remaining_excess = excess - projected_savings

        # Sort small docs by size (descending) for better batching
        small_docs.sort(key=lambda d: d.tokens, reverse=True)

        current_batch: List[DocNode] = []
        current_batch_tokens = 0
        batch_idx = 0

        for doc in small_docs:
            if projected_savings >= excess:
                break

            # If adding this doc would exceed batch target, finalize current batch
            if current_batch_tokens + doc.tokens > SMALL_DOC_BATCH_TARGET_TOKENS and current_batch:
                batch = DocBatch(
                    docs=current_batch,
                    total_tokens=current_batch_tokens,
                    batch_name=f"small_batch_{batch_idx}",
                )
                batches.append(batch)
                for d in current_batch:
                    d.is_batched = True
                    d.force_summarize = True
                # Estimate 70% reduction for batched small docs
                projected_savings += int(current_batch_tokens * 0.7)

                current_batch = []
                current_batch_tokens = 0
                batch_idx += 1

            current_batch.append(doc)
            current_batch_tokens += doc.tokens

        # Finalize last batch if it has content
        if current_batch and projected_savings < excess:
            batch = DocBatch(
                docs=current_batch,
                total_tokens=current_batch_tokens,
                batch_name=f"small_batch_{batch_idx}",
            )
            batches.append(batch)
            for d in current_batch:
                d.is_batched = True
                d.force_summarize = True
            projected_savings += int(current_batch_tokens * 0.7)

    logger.debug(
        "[PLAN] batches: %d batches with %d small docs, total projected_savings=%d",
        len(batches), sum(len(b.docs) for b in batches), projected_savings
    )

    return docs_to_summarize, batches


def _summarize_batch(
    batch: DocBatch,
    client,
    model_for_summary: str,
    model_for_counting: str,
    limiter,
    count_tokens,
    settings: "CompressionSettings",
) -> Dict[str, List[Part]]:
    """
    Summarize a batch of small documents together.
    Returns a dict mapping doc name to its summary parts.
    """
    logger = get_logger()

    # Combine all docs in batch with clear separators
    combined_parts: List[Part] = []
    for doc in batch.docs:
        # Add doc separator
        combined_parts.append(Part.from_text(text=f"\n--- Document: {doc.name} ---\n"))
        combined_parts.extend(doc.content)

    # Summarize the combined batch
    blocks = _chunk_parts_by_true_count(
        parts=combined_parts,
        target_tokens=settings.chunk_target_tokens,
        model_for_counting=model_for_counting,
        count_tokens=count_tokens,
    )

    summaries = _run_coro_blocking(
        _summarize_mixed_chunks_parallel(
            client=client,
            model_for_summary=model_for_summary,
            model_for_counting=model_for_counting,
            chunks=blocks,
            limiter=limiter,
            count_tokens=count_tokens,
            settings=settings,
            doc_name=batch.batch_name,
        )
    )

    # Create a single summary for the batch
    batch_summary = "\n\n---\n\n".join(s for s in summaries if s.strip())

    # Parse the batch summary to extract per-doc summaries
    # For now, we'll assign the entire batch summary to each doc
    # (could be improved with more sophisticated parsing)
    result: Dict[str, List[Part]] = {}
    for doc in batch.docs:
        # Create a doc-specific summary header
        doc_summary = f"[Summary from batch {batch.batch_name}]\n\n{batch_summary}"
        result[doc.name] = [Part.from_text(text=doc_summary)]

    logger.debug("[BATCH] summarized: %s docs=%d", batch.batch_name, len(batch.docs))
    return result


def _reassemble_context(
    docs: List[DocNode],
    non_doc_parts: List[Part],
    doc_summaries: Dict[str, List[Part]],
) -> List[Part]:
    """
    Reassemble context parts from processed documents.
    Preserves original tag structure.
    """
    result: List[Part] = list(non_doc_parts)  # Non-doc parts go first

    for doc in docs:
        # Use summary if available, otherwise use original content
        if doc.name in doc_summaries:
            inner = doc_summaries[doc.name]
        elif doc.cached_summary:
            inner = doc.cached_summary
        else:
            inner = doc.content

        # Rebuild with tags
        result.append(doc.open_part)
        result.extend(inner)
        if doc.close_part:
            result.append(doc.close_part)

    return result


def _wait_for_existing_summary(
    *, content_key: str, timeout_s: int | None = None, interval_s: int = 30, log_interval_s: int = 30
) -> Optional[List[Part]]:
    """
    Poll every `interval_s` seconds for a summary written by the RFP summarizer.
    Returns parts if found within timeout, else None.
    """

    logger = get_logger()
    timeout_s = 3600 if timeout_s is None else int(timeout_s)
    if timeout_s <= 0:
        logger.debug("[DOCCACHE] wait skipped: timeout<=0 key=%s", content_key[:8])
        return None

    start = time.monotonic()
    next_tick = start
    last_log = start
    polls = 0

    logger.debug("[DOCCACHE] wait start: key=%s timeout=%ss interval=%ss",
                 content_key[:8], timeout_s, interval_s)

    while True:
        # Poll
        persisted = load_doc_summary_from_store(content_key)
        now = time.monotonic()

        if persisted:
            elapsed = now - start
            logger.debug("[DOCCACHE] wait success: key=%s elapsed=%.2fs polls=%d",
                         content_key[:8], elapsed, polls)
            return persisted

        if (now - start) >= timeout_s:
            logger.debug("[DOCCACHE] wait timeout: key=%s elapsed=%.2fs polls=%d",
                         content_key[:8], now - start, polls)
            return None

        if (now - last_log) >= log_interval_s:
            logger.debug("[DOCCACHE] waiting: key=%s elapsed=%.1fs polls=%d next_sleep=%.2fs",
                         content_key[:8], now - start, polls,
                         max(0.0, next_tick + interval_s - now))
            last_log = now

        # Poll every 30 seconds
        polls += 1
        next_tick += interval_s
        sleep_for = max(0.0, next_tick - time.monotonic())
        time.sleep(sleep_for)

@dataclass
class CompressionSettings:
    enabled: bool = True

    # Limits
    hard_limit_tokens: int = 1_000_000           # absolute cap
    target_total_tokens: int = 900_000           # aim to stay under this
    prompt_reserve_tokens: int = 20_000          # budget we try to leave for prompt/system

    # Per-document ceiling: if a single doc exceeds this, it is force-summarized
    # regardless of overall budget. 0 = disabled.
    per_doc_ceiling_tokens: int = 0

    # Chunking (token-based; overlap also in tokens)
    chunk_target_tokens: int = 500_000           # summarize ~500k tokens at once

    # Summarizer model/config
    summary_model: str = "gemini-2.5-flash"
    summary_temperature: float = 0.0
    summary_max_output_tokens: int = 30000        # per-block output cap
    max_rounds: int = 3

    # Output formatting
    header_label: str = "CONTEXT"

    # Extreme fallback truncation
    truncation_seed: Optional[int] = None

    # Heuristic conversion when count_tokens() fails
    chars_per_token: int = 4
    max_parallel_requests: int = 3

    # Dynamic re-loop: max outer compression passes in compress_if_needed_document_aware.
    max_compress_passes: int = 10


def _auto_tighten_settings(
    current: "CompressionSettings",
    original: "CompressionSettings",
    pass_num: int,
    overshoot_ratio: float,
) -> "CompressionSettings":
    """
    Progressively tighten settings per pass.
    """
    base_factor = 1.0 / max(1.0, overshoot_ratio)
    decay = 0.7 ** (pass_num - 1)
    factor = max(0.05, base_factor * decay)

    per_doc_ceiling = current.per_doc_ceiling_tokens
    if per_doc_ceiling > 0:
        per_doc_ceiling = max(10_000, int(per_doc_ceiling * factor))

    return _dc_replace(
        current,
        chunk_target_tokens=max(30_000, int(original.chunk_target_tokens * factor)),
        summary_max_output_tokens=max(3_000, int(original.summary_max_output_tokens * factor)),
        per_doc_ceiling_tokens=per_doc_ceiling,
    )


def format_eval_criteria_for_summary(structure: dict | None) -> str | None:
    """
    Convert nested evaluation criteria JSON into concise hierarchical text.
    """
    if not structure or not isinstance(structure, dict):
        return None

    lines: list[str] = []

    def _walk(obj: dict, depth: int = 0) -> None:
        if not isinstance(obj, dict):
            return
        for key, val in obj.items():
            indent = "  " * depth
            if isinstance(val, dict) and "weight" in val:
                nested = any(
                    isinstance(vv, dict)
                    for kk, vv in val.items()
                    if kk not in ("weight", "description")
                )
                if not nested:
                    lines.append(f"{indent}- {key}")
                    desc = val.get("description")
                    if isinstance(desc, str) and desc.strip():
                        lines.append(f"{indent}  {desc.strip()}")
                    continue

            if isinstance(val, (int, float, str)):
                lines.append(f"{indent}- {key}")
                continue

            if isinstance(val, dict):
                label = "Scope" if depth == 0 else "Category"
                lines.append(f"{indent}{label}: {key}")
                _walk(val, depth + 1)

    _walk(structure)
    if not lines:
        return None

    header = "EVALUATION CRITERIA (preserve information relevant to these areas)"
    return header + "\n" + "\n".join(lines)


def _true_count_range(
    *,
    model: str,
    parts: list[Part],
    i: int,
    j: int,  # exclusive
    count_tokens,
) -> int:
    # One call to Vertex count_tokens() for parts[i:j]
    try:
        if i >= j:
            return 0
        return _count_tokens_strict(model, parts[i:j], count_tokens)
    except Exception:
        txt = 0; media = 0
        for p in parts[i:j]:
            s = getattr(p, "text", None)
            if isinstance(s, str):
                txt += len(s)
            else:
                media += 1
        return (txt // 4) + media * 512


def _chunk_parts_by_true_count(
    *,
    parts: list[Part],
    target_tokens: int,
    model_for_counting: str,
    count_tokens,
) -> list[list[Part]]:
    """
    Make chunks by true token counts using exponential growth + binary search.
    - Images are naturally included (we pass original Parts into count_tokens).
    - Complexity ~ O(k log n) count calls per document.
    """
    chunks: list[list[Part]] = []
    n = len(parts)
    i = 0
    logger = get_logger()
    logger.debug("[CHUNKER] start parts=%d target≈%d", len(parts), target_tokens)
    # memo to avoid recounting the same [i:j] in bsearch
    cache: dict[tuple[int, int], int] = {}

    def _ct(i_: int, j_: int) -> int:
        key = (i_, j_)
        if key in cache:
            return cache[key]
        v = _true_count_range(
            model=model_for_counting,
            parts=parts,
            i=i_, j=j_,
            count_tokens=count_tokens
        )
        cache[key] = v
        return v

    while i < n:
        # If a single part already exceeds target, emit it alone and move on.
        if _ct(i, i+1) > target_tokens:
            chunks.append(parts[i:i+1])
            i += 1
            continue

        # Exponential growth from i to find an upper bound that exceeds target (or hit n)
        lo = i + 1               # known OK end (exclusive)
        hi = min(n, i + 1)       # probe end
        last_ok = lo             # best OK boundary
        last_ok_tokens = _ct(i, lo)

        step = 1
        while hi < n and last_ok_tokens < target_tokens:
            hi = min(n, i + step*2)
            t = _ct(i, hi)
            if t <= target_tokens:
                last_ok = hi
                last_ok_tokens = t
                step *= 2
            else:
                break

        # If we stopped because we hit n and still under target, take it all
        if last_ok == n or last_ok_tokens <= target_tokens and hi == n:
            logger.debug("[CHUNKER] emit [%d:%d) tokens≈%d", i, last_ok, cache.get((i, last_ok), -1))
            chunks.append(parts[i:last_ok])
            i = last_ok
            continue

        # Otherwise we overshot at 'hi': binary search (last_ok, hi] for tightest boundary <= target
        lo_b, hi_b = last_ok, hi
        while lo_b + 1 < hi_b:
            mid = (lo_b + hi_b) // 2
            tmid = _ct(i, mid)
            if tmid <= target_tokens:
                lo_b = mid
            else:
                hi_b = mid

        split = lo_b  # maximal j s.t. count(i, j) <= target
        if split <= i:  # extreme edge case safeguard
            split = i + 1

        chunks.append(parts[i:split])
        logger.debug("[CHUNKER] emit [%d:%d) tokens≈%d", i, split, cache.get((i, split), -1))
        i = split

    logger.debug("[CHUNKER] done chunks=%d", len(chunks))
    return chunks


def _build_summary_prompts(
    doc_name: str | None,
    eval_criteria_text: str | None = None,
) -> tuple[str, str]:
    sys_instr = (
        "You are a meticulous document condenser used in a retrieval system.\n"
        "Your goal is HIGH RECALL, not brevity.\n"
        "Produce a dense, self-contained summary that preserves ALL important facts, "
        "including people/CV details, dates, numbers, names, and requirements. Include any important information that can be used for and can affect evaluation.\n"
        "\n"
        "Guidelines:\n"
        "- Prefer coverage over compression; it is OK if the summary is relatively long.\n"
        "- Never drop whole sections such as CVs, profiles, evaluation criteria, or checklists.\n"
        "- When CVs, bios, or staff profiles appear, capture for EACH person: "
        "  name, role/title, years of experience (if given), key skills, qualifications, "
        "  notable projects, and any mandatory/desired requirements they satisfy.\n"
        "- Explicitly list requirements, evaluation criteria, weights, scoring scales, "
        "  constraints, assumptions, deadlines, and dependencies.\n"
        "- Preserve important numbers (prices, weights, scores, dates, quantities, thresholds).\n"
        "- Keep section headings or a simplified outline when obvious.\n"
        "- Include key tables as CSV-like lines (one line per row, comma-separated cells) when possible.\n"
        "- For images, summarize any text-like information they contain: tables, forms, CVs, diagrams, "
        "  certificates, signatures, or schematics. Do not describe decorative imagery.\n"
        "- Do NOT invent or infer facts that are not present in the input.\n"
        "- If you must omit something due to length, remove redundancy and wording first, "
        '  NOT distinct facts or people.\n'
    )

    if eval_criteria_text:
        sys_instr += (
            "\n"
            "IMPORTANT — The document(s) you are summarizing will be evaluated against\n"
            "specific criteria. Prioritize preserving information relevant to these areas.\n"
            "Do NOT omit content that could help answer or score any of the following:\n"
            "\n"
            f"{eval_criteria_text}\n"
        )

    head = (
        f"Summarize the following block from document: {doc_name or 'unknown'}.\n"
        "Return a dense, high-coverage summary that can be used WITHOUT the original text.\n"
        "Contain as much as important details as possible.\n"
        "Use clear bullets and descriptive paragraphs. Preserve raw values and details.\n"
    )
    return sys_instr, head


def _summarize_mixed_block_raw(
    *,
    client: genai.Client,
    model: str,
    parts: list[Part],
    temperature: float,
    max_output_tokens: int,
    doc_name: str | None = None,
    eval_criteria_text: str | None = None,
):
    sys_instr, head = _build_summary_prompts(
        doc_name,
        eval_criteria_text=eval_criteria_text,
    )

    # Explicitly disable thinking for the summarizer call path.
    thinking = ThinkingConfig(thinking_budget=0)

    # Chat create config
    chat_cfg_kwargs = {
        "system_instruction": sys_instr,
        "thinking_config": thinking,
    }
    chat = client.chats.create(
        model=model,
        config=GenerateContentConfig(**chat_cfg_kwargs),
    )

    # Send message config
    msg_cfg_kwargs = dict(
        temperature=temperature,
        top_p=1.0,
        top_k=1,
        max_output_tokens=max_output_tokens,
        response_mime_type="text/plain",
        thinking_config=thinking,
    )

    resp = chat.send_message(
        [Part.from_text(text=head)] + parts,
        config=GenerateContentConfig(**msg_cfg_kwargs),
    )
    return resp, sys_instr, head

def _hash_parts_list_minimal(parts: Sequence[Part]) -> str:
    """
    Hash inner document content only (stable across contexts).
    """
    h = hashlib.sha256()
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            h.update(b"T|"); h.update(t.encode("utf-8", "ignore")); continue
        fd = getattr(p, "file_data", None)
        if fd and getattr(fd, "file_uri", None):
            h.update(b"U|"); h.update(fd.file_uri.encode("utf-8", "ignore")); continue
        il = getattr(p, "inline_data", None)
        if il and getattr(il, "data", None):
            h.update(b"I|"); h.update(hashlib.sha1(il.data).digest()); continue
        mt = getattr(p, "mime_type", "") or ""
        h.update(b"M|"); h.update(mt.encode("utf-8", "ignore"))
    return h.hexdigest()

def _doc_cache_key(name: str | None, inner_parts: Sequence[Part], settings: CompressionSettings) -> str:
    """
    Include name, a hash of the inner parts, and a small fingerprint of settings that affect output size.
    """
    inner_h = _hash_parts_list_minimal(inner_parts)
    fp = f"{_DOC_CACHE_VERSION}|{settings.summary_model}|{settings.chunk_target_tokens}|{settings.summary_max_output_tokens}"
    return hashlib.sha256(f"{name or 'noname'}|{inner_h}|{fp}".encode("utf-8")).hexdigest()


def _hash_parts_list(parts: Sequence[Part]) -> str:
    h = hashlib.sha256()
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            h.update(b"T|"); h.update(t.encode("utf-8", "ignore")); continue
        fd = getattr(p, "file_data", None)
        if fd and getattr(fd, "file_uri", None):
            h.update(b"U|"); h.update(fd.file_uri.encode("utf-8", "ignore")); continue
        il = getattr(p, "inline_data", None)
        if il and getattr(il, "data", None):
            h.update(b"I|"); h.update(hashlib.sha1(il.data).digest()); continue
        mt = getattr(p, "mime_type", "") or ""
        h.update(b"M|"); h.update(mt.encode("utf-8", "ignore"))
    return h.hexdigest()

def _settings_fp(s: CompressionSettings) -> str:
    return (
        f"{s.summary_model}|{s.hard_limit_tokens}|{s.target_total_tokens}|"
        f"{s.prompt_reserve_tokens}|{s.per_doc_ceiling_tokens}|"
        f"{s.chunk_target_tokens}|{s.summary_max_output_tokens}|"
        f"{s.max_rounds}|{s.max_compress_passes}"
    )

def compress_with_memo(
    parts: Sequence[Part],
    model_for_counting: str,
    limiter,
    settings: CompressionSettings,
    name_to_source_key: dict[str, str] | None = None,
    count_tokens_fn: Optional[Callable[[str, Sequence[Part]], int]] = None,
    eval_criteria_text: str | None = None,
) -> List[Part]:
    from utils.llm.LLM import get_client, count_tokens as _default_count_tokens
    count_tokens = count_tokens_fn or _default_count_tokens
    logger = get_logger()

    prefix, ctx, suffix = _split_prompt_context_suffix(list(parts))
    if not ctx:
        return list(parts)

    ctx_h = _hash_parts_list(ctx)
    fp = _settings_fingerprint(settings)
    ec_h = hashlib.sha256((eval_criteria_text or "").encode("utf-8")).hexdigest()[:16]
    key = hashlib.sha256(f"{ctx_h}|{fp}|ec={ec_h}".encode("utf-8")).hexdigest()
    key8 = key[:8]
    # memo hit
    with _CTX_MEMO_LOCK:
        cached_ctx = _CTX_COMPRESS_MEMO.get(key)
        if cached_ctx is not None:
            logger.debug("CTX-COMPRESS memo hit key=%s (cached_ctx_parts=%d, orig_ctx_parts=%d)", key[:8], len(cached_ctx), len(ctx))
            _CTX_COMPRESS_MEMO.move_to_end(key)
            if not cached_ctx:
                get_logger().warning("CTX-COMPRESS memo was empty; returning ORIGINAL context key=%s", key[:8])
                return list(prefix) + list(ctx) + list(suffix)
            return list(prefix) + list(cached_ctx) + list(suffix)

        # In-flight guard
        ev = _CTX_INFLIGHT.get(key)
        if ev is None:
            ev = threading.Event()
            _CTX_INFLIGHT[key] = ev
            leader = True
        else:
            leader = False

    if not leader:
        # Wait for the leader to finish compressing this context
        logger.debug("CTX-COMPRESS follower waiting key=%s", key8)
        _t0 = time.perf_counter()
        ev.wait()
        _dt = time.perf_counter() - _t0
        logger.debug("CTX-COMPRESS follower resume key=%s waited=%.2fs", key8, _dt)

        with _CTX_MEMO_LOCK:
            cached_ctx = _CTX_COMPRESS_MEMO.get(key)
        if cached_ctx is not None:
            logger.debug(
                "CTX-COMPRESS follower resume key=%s (ctx_parts=%d)",
                key8,
                len(cached_ctx),
            )
            if not cached_ctx:
                logger.warning(
                    "CTX-COMPRESS follower: memo was empty for key=%s; returning original context",
                    key8,
                )
                return list(prefix) + list(ctx) + list(suffix)
            return list(prefix) + list(cached_ctx) + list(suffix)

        logger.warning(
            "CTX-COMPRESS follower: leader failed for key=%s; returning original (uncompressed) context (%d parts)",
            key8,
            len(ctx),
        )
        return list(prefix) + list(ctx) + list(suffix)

    # Leader path: do the compression once (guard since we start in parallel)
    try:
        logger.debug("CTX-COMPRESS leader computing key=%s", key8)
        _t0 = time.perf_counter()
        out = compress_if_needed_document_aware(
            prefix_parts=list(prefix),
            context_parts=list(ctx),
            suffix_parts=list(suffix),
            model_for_counting=model_for_counting,
            get_client=get_client,
            count_tokens=count_tokens,
            limiter=limiter,
            settings=settings,
            name_to_source_key=name_to_source_key,
            eval_criteria_text=eval_criteria_text,
        )
        _dt = time.perf_counter() - _t0
        logger.debug("CTX-COMPRESS leader done key=%s dur=%.1fs", key8, _dt)
        new_prefix, new_ctx, new_suffix = _split_prompt_context_suffix(out)
        if not new_ctx:
            # Fallback (shouldn't happen with doc-aware path): keep original ctx
            new_ctx = list(ctx)

        with _CTX_MEMO_LOCK:
            _CTX_COMPRESS_MEMO[key] = list(new_ctx)
            _CTX_COMPRESS_MEMO.move_to_end(key)
            if len(_CTX_COMPRESS_MEMO) > _CTX_COMPRESS_MEMO_MAX:
                _CTX_COMPRESS_MEMO.popitem(last=False)
        logger.debug("CTX-COMPRESS stored key=%s (memo_size=%d)", key8, len(_CTX_COMPRESS_MEMO))
        return list(new_prefix) + list(new_ctx) + list(new_suffix)
    finally:
        # Release all waiters even on error
        with _CTX_MEMO_LOCK:
            ev = _CTX_INFLIGHT.pop(key, None)
        if ev:
            ev.set()


def _split_prompt_context_suffix(parts: Sequence[Part]) -> Tuple[List[Part], List[Part], List[Part]]:
    """
    Split the incoming Parts into:
      - prefix_parts: non-context (instructions etc.) BEFORE the context buckets
      - context_parts: ONLY the RFP+CONTRACTOR buckets (between BEGIN RFP and END CONTRACTOR)
      - suffix_parts: non-context (instructions etc.) AFTER the context buckets

    Also handles the case where all content is in a single text Part (split on header).
    """
    prefix_parts: List[Part] = []
    context_parts: List[Part] = []
    suffix_parts: List[Part] = []
    logger = get_logger()
    # Single-Part fast path: if one text Part contains the header, split text
    if len(parts) == 1:
        t = getattr(parts[0], "text", None)
        if isinstance(t, str) and (MARKER_BEGIN_RFP in t or MARKER_BEGIN_CONTRACTOR in t):
            # Split on the first context header we find; keep prefix intact, compress tail
            idx = t.find(MARKER_BEGIN_RFP) if MARKER_BEGIN_RFP in t else t.find(MARKER_BEGIN_CONTRACTOR)
            if idx > 0:
                prefix_text = t[:idx]
                tail_text = t[idx:]
                if prefix_text.strip():
                    prefix_parts.append(Part.from_text(text=prefix_text))
                context_parts.append(Part.from_text(text=tail_text))
                return prefix_parts, context_parts, suffix_parts
            # header is at start, whole part is context
            context_parts.append(parts[0])
            return prefix_parts, context_parts, suffix_parts

    # Multi-Part path: bucketize by markers
    STATE_PREFIX, STATE_CONTEXT, STATE_SUFFIX = 0, 1, 2
    state = STATE_PREFIX

    for p in parts:
        txt = getattr(p, "text", None)
        if isinstance(txt, str):
            if state == STATE_PREFIX:
                # Flip to context once we see either BEGIN marker
                if MARKER_BEGIN_RFP in txt or MARKER_BEGIN_CONTRACTOR in txt:
                    state = STATE_CONTEXT
                    context_parts.append(p)
                else:
                    prefix_parts.append(p)
            elif state == STATE_CONTEXT:
                context_parts.append(p)
                # Leave context once we see END CONTRACTOR
                if MARKER_END_CONTRACTOR in txt:
                    state = STATE_SUFFIX
            else:
                suffix_parts.append(p)
        else:
            # Non-text parts (e.g., images) should follow the current state
            if state == STATE_PREFIX:
                prefix_parts.append(p)
            elif state == STATE_CONTEXT:
                context_parts.append(p)
            else:
                suffix_parts.append(p)

    # If no context markers present at all, treat the first text Part as “prompt” and the rest as “context” (legacy behavior)
    if not context_parts:
        if parts and isinstance(getattr(parts[0], "text", None), str):
            return [parts[0]], list(parts[1:]), []
        return [], list(parts), []

    logger.debug(
    "[SPLIT] prefix=%d context=%d suffix=%d markers_present=%s",
    len(prefix_parts), len(context_parts), len(suffix_parts),
    bool(context_parts)
)
    return prefix_parts, context_parts, suffix_parts

def _count_parts_safe(
    *,
    model: str,
    parts: Iterable[Part],
    count_tokens: Callable[[str, Iterable[Part]], int],
    chars_per_token: int,
) -> int:
    try:
        return _count_tokens_strict(model, list(parts), count_tokens)
    except Exception:
        total_chars = 0
        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                total_chars += len(t)
        return max(1, total_chars // max(1, chars_per_token))

_SUMMARY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def _run_coro_blocking(coro):
    """
    Run `coro` whether or not an event loop is already running.
    - If no loop: asyncio.run(coro)
    - If a loop exists: execute in a fresh loop inside a worker thread,
      but *propagate ContextVars* so get_logger() still works.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    ctx = contextvars.copy_context()
    def _runner():
        return asyncio.run(coro)

    # Run with the copied ContextVars so get_logger() sees the tool adapter
    return _SUMMARY_EXECUTOR.submit(lambda: ctx.run(_runner)).result()

async def _summarize_mixed_chunks_parallel(
    *,
    client: genai.Client,
    model_for_summary: str,
    model_for_counting: str,
    chunks: list[list[Part]],
    limiter,
    count_tokens,
    settings: CompressionSettings,
    doc_name: str | None = None,
    doc_key: str | None = None,
    doc_source_key: str | None = None,
    eval_criteria_text: str | None = None,
) -> list[str]:
    """
    Summarize each chunk (text + images) in parallel with:
      - token-aware throttling via `limiter`
      - capped concurrency via asyncio.Semaphore(max_parallel_requests)
      - retries on 429 and server errors
      - on resource exhausted: retry with gemini-2.5-flash 2 more times
      - NEVER raise: final fallback is a truncated raw extract; logs a warning
    """
    logger = get_logger()

    total = len(chunks)
    out: list[str] = [""] * total
    if total == 0:
        logger.debug("[DOC] summarize: doc=%s chunks=0 (nothing to do)", doc_name or "unknown")
        return out

    LARGE = 120000
    max_chunk = max(
        _count_parts_safe_mixed(
            model=model_for_counting,
            parts=c,
            count_tokens=count_tokens,
            chars_per_token=settings.chars_per_token,
        )
        for c in chunks
    )
    parallel = 1 if max_chunk >= LARGE else settings.max_parallel_requests
    sem = asyncio.Semaphore(max(1, parallel))
    lock = asyncio.Lock()
    completed = 0
    start_doc = time.perf_counter()

    logger.debug("[DOC] parallel=%d max_chunk≈%d threshold=%d", parallel, max_chunk, LARGE)

    async def _one(i: int, block: list[Part]):
        nonlocal completed

        # Stable per-block folder name: idx + short content-hash
        block_hash10 = _hash_parts_list_minimal(block)[:10]
        block_key = f"{i+1:04d}_{block_hash10}"

        # Estimate tokens (fallback heuristic if needed)
        toks = _count_parts_safe_mixed(
            model=model_for_counting,
            parts=block,
            count_tokens=count_tokens,
            chars_per_token=settings.chars_per_token,
        )

        logger.debug(
            "[CHUNK] start: doc=%s idx=%d/%d est_tokens=%d",
            doc_name or "unknown", i + 1, total, toks
        )

        async with sem:
            logger.debug(
                "[CHUNK] queue: doc=%s idx=%d/%d tokens=%d",
                doc_name or "unknown", i + 1, total, toks
            )

            # Build prompts once (audit uses these)
            sys_instr, head = _build_summary_prompts(
                doc_name, eval_criteria_text=eval_criteria_text
            )

            # Audit input (best-effort; never crash)
            if doc_key:
                try:
                    from utils.llm.compactor_cache import save_block_audit_input
                    await asyncio.to_thread(
                        save_block_audit_input,
                        doc_key=doc_key,
                        block_key=block_key,
                        doc_name=doc_name,
                        source_key=doc_source_key,
                        model=model_for_summary,  # primary model recorded; fallback noted in logs
                        system_instruction=sys_instr,
                        head_text=head,
                        block_parts=block,
                    )
                except Exception:
                    logger.debug("[AUDIT] save_block_audit_input failed (non-fatal)", exc_info=True)

            # Retry loop
            t_retry_start = time.monotonic()
            backoff = float(globals().get("COMPACTOR_429_INITIAL_BACKOFF", 5))
            attempts_429 = 0
            attempts_5xx = 0
            using_fallback = False
            current_model = model_for_summary
            fallback_remaining = COMPACTOR_FALLBACK_MODEL_ATTEMPTS

            dur = 0.0
            resp_obj = None
            text_out = ""

            req_cfg = {
                "temperature": settings.summary_temperature,
                "top_p": 1.0,
                "top_k": 1,
                "max_output_tokens": settings.summary_max_output_tokens,
                "response_mime_type": "text/plain",
            }

            # We switch models only on resource exhausted; otherwise keep primary
            current_model = model_for_summary

            # Final fallback: truncate raw extract to something roughly bounded by output tokens
            # (keeps context from exploding if the model is down)
            fallback_chars = int(settings.summary_max_output_tokens * max(1, settings.chars_per_token))

            def _raw_fallback(reason: str, err: Exception | None = None) -> str:
                if err is not None:
                    logger.warning(
                        "[CHUNK] FALLBACK-EXTRACT: doc=%s idx=%d/%d reason=%s err=%s",
                        doc_name or "unknown", i + 1, total, reason, repr(err),
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "[CHUNK] FALLBACK-EXTRACT: doc=%s idx=%d/%d reason=%s",
                        doc_name or "unknown", i + 1, total, reason,
                    )
                return _fallback_extract_from_parts(
                    block,
                    max_chars=fallback_chars,
                    chars_per_token=settings.chars_per_token,
                )

            while True:
                try:
                    def _call_sync_raw():
                        return _summarize_mixed_block_raw(
                            client=client,
                            model=current_model,
                            parts=block,
                            temperature=settings.summary_temperature,
                            max_output_tokens=settings.summary_max_output_tokens,
                            doc_name=doc_name,
                            eval_criteria_text=eval_criteria_text,
                        )

                    # Budget: input toks + output headroom
                    budget = toks + int(settings.summary_max_output_tokens * 1.1) + 1024

                    t_acq = time.perf_counter()
                    await limiter.acquire(max(1, budget))
                    waited = time.perf_counter() - t_acq
                    if waited > 0.5:
                        logger.debug(
                            "[CHUNK] throttle: doc=%s idx=%d waited=%.2fs budget=%d (in=%d out≈%d) model=%s",
                            doc_name or "unknown", i + 1, waited, budget, toks, settings.summary_max_output_tokens, current_model
                        )

                    t_start = time.perf_counter()
                    resp_obj, _sys, _head = await asyncio.to_thread(_call_sync_raw)
                    dur = time.perf_counter() - t_start

                    text_out = getattr(resp_obj, "text", "") or getattr(resp_obj, "content", "") or ""
                    break

                except gerrors.ClientError as e:
                    code = getattr(e, "code", None)
                    is_exhausted = _is_resource_exhausted_err(e)
                    is_quota_related = (code in (429, 499)) or is_exhausted

                    # How long have we been retrying this chunk?
                    elapsed = time.monotonic() - t_retry_start
                    remaining_total = COMPACTOR_429_MAX_TOTAL - elapsed

                    # If we've exceeded the total retry budget, stop retrying:
                    if remaining_total <= 0:
                        if (not using_fallback) and is_exhausted:
                            # last chance: switch to fallback if we haven't yet
                            using_fallback = True
                            current_model = COMPACTOR_FALLBACK_MODEL
                            fallback_remaining = COMPACTOR_FALLBACK_MODEL_ATTEMPTS
                            logger.warning(
                                "[CHUNK] quota-timeout on primary: switching to fallback model=%s doc=%s idx=%d/%d",
                                current_model, doc_name or "unknown", i + 1, total,
                                exc_info=True,
                            )
                            t_retry_start = time.monotonic()
                            # also reset backoff so fallback waits big too
                            backoff = float(globals().get("COMPACTOR_429_INITIAL_BACKOFF", 10.0))
                            continue

                        # Otherwise: hard stop -> raw extract
                        text_out = _raw_fallback("quota_timeout_total_exceeded", e)
                        break

                    # Quota handling (429/499/resource_exhausted and cancelled respectively)
                    if is_quota_related:
                        # Primary model: retry up to max attempts; if exhausted -> switch to fallback
                        if not using_fallback:
                            if attempts_429 < COMPACTOR_429_MAX_ATTEMPTS:
                                # Big exponential backoff + jitter, capped by remaining_total
                                # Big exponential backoff was done because our previous exponential backoff was small (8s), and was crashing the app when the retries finish.
                                sleep_base = min(COMPACTOR_429_MAX_BACKOFF, backoff)
                                sleep_base = min(sleep_base, max(0.1, remaining_total))

                                sleep_for = sleep_base * (0.75 + random.random() * 0.5)
                                sleep_for = min(sleep_for, max(0.1, remaining_total))    # clamp AFTER jitter

                                logger.warning(
                                    "[CHUNK] quota-retry: doc=%s idx=%d/%d model=%s code=%s attempt=%d sleep=%.1fs remaining_total=%.1fs exhausted=%s",
                                    doc_name or "unknown", i + 1, total, current_model, code,
                                    attempts_429 + 1, sleep_for, remaining_total, is_exhausted,
                                    exc_info=True,
                                )
                                await asyncio.sleep(sleep_for)

                                backoff = min(COMPACTOR_429_MAX_BACKOFF, max(5.0, backoff * 2))
                                attempts_429 += 1
                                continue

                            # primary retries exhausted -> if resource exhausted, switch to fallback
                            if is_exhausted:
                                using_fallback = True
                                current_model = COMPACTOR_FALLBACK_MODEL
                                fallback_remaining = COMPACTOR_FALLBACK_MODEL_ATTEMPTS
                                logger.warning(
                                    "[CHUNK] primary retries exhausted: switching to fallback model=%s doc=%s idx=%d/%d",
                                    current_model, doc_name or "unknown", i + 1, total,
                                    exc_info=True,
                                )
                                backoff = float(globals().get("COMPACTOR_429_INITIAL_BACKOFF", 10.0))
                                continue

                            # not exhausted -> give up
                            text_out = _raw_fallback("quota_give_up_primary_attempts", e)
                            break

                        # Fallback model: also use BIG backoff, limited attempts
                        else:
                            if fallback_remaining > 0:
                                sleep_base = min(COMPACTOR_429_MAX_BACKOFF, backoff)
                                sleep_base = min(sleep_base, max(0.1, remaining_total))
                                sleep_for = sleep_base * (0.75 + random.random() * 0.5)  # jitter
                                sleep_for = min(sleep_for, max(0.1, remaining_total))    # clamp AFTER jitter

                                logger.warning(
                                    "[CHUNK] fallback quota-retry: doc=%s idx=%d/%d model=%s remaining=%d sleep=%.1fs remaining_total=%.1fs",
                                    doc_name or "unknown", i + 1, total, current_model,
                                    fallback_remaining, sleep_for, remaining_total,
                                    exc_info=True,
                                )
                                await asyncio.sleep(sleep_for)

                                backoff = min(COMPACTOR_429_MAX_BACKOFF, max(5.0, backoff * 2))
                                fallback_remaining -= 1
                                continue

                            text_out = _raw_fallback("quota_give_up_fallback_attempts", e)
                            break

                    # Non-quota client error: give up -> raw extract
                    text_out = _raw_fallback("client_error_give_up", e)
                    break

                except gerrors.ServerError as e:
                    # Server retries
                    if attempts_5xx < COMPACTOR_5XX_MAX_ATTEMPTS:
                        sleep_for = min(COMPACTOR_5XX_MAX_BACKOFF, backoff) * (0.5 + random.random())
                        logger.warning(
                            "[CHUNK] server-retry: doc=%s idx=%d/%d model=%s attempt=%d sleep=%.1fs",
                            doc_name or "unknown", i + 1, total, current_model, attempts_5xx + 1, sleep_for,
                            exc_info=True,
                        )
                        await asyncio.sleep(sleep_for)
                        backoff = min(COMPACTOR_5XX_MAX_BACKOFF, backoff * 2)
                        attempts_5xx += 1
                        continue

                    text_out = _raw_fallback("server_error_give_up", e)
                    break

                except Exception as e:
                    # Truly unexpected => raw extract, never crash
                    text_out = _raw_fallback("unexpected_exception", e)
                    break

            out[i] = text_out

            # Audit output (best-effort; never crash)
            if doc_key:
                try:
                    from utils.llm.compactor_cache import save_block_audit_output
                    await asyncio.to_thread(
                        save_block_audit_output,
                        doc_key=doc_key,
                        block_key=block_key,
                        doc_name=doc_name,
                        source_key=doc_source_key,
                        model=current_model,  # record actual model used (could be fallback)
                        request_config=req_cfg,
                        response_text=text_out,
                        response_obj=resp_obj,
                        duration_s=dur,
                    )
                except Exception:
                    logger.debug("[AUDIT] save_block_audit_output failed (non-fatal)", exc_info=True)

        # Progress
        async with lock:
            completed += 1
            pct = 100.0 * completed / max(1, total)
            logger.debug(
                "[DOC] progress: doc=%s %d/%d (%.1f%%) last_chunk=%d dur=%.2fs",
                doc_name or "unknown", completed, total, pct, i + 1, dur
            )

        logger.debug(
            "[CHUNK] done:  doc=%s idx=%d/%d summary_len_chars=%d",
            doc_name or "unknown", i + 1, total, len(out[i] or "")
        )
        return i

    # Launch tasks
    tasks = [asyncio.create_task(_one(i, c)) for i, c in enumerate(chunks)]
    for task in asyncio.as_completed(tasks):
        await task

    doc_dur = time.perf_counter() - start_doc
    logger.debug(
        "[DOC] summarize finished: doc=%s chunks=%d total_dur=%.1fs",
        doc_name or "unknown", total, doc_dur
    )
    return out

def _compress_one_document_if_needed(
    *,
    name: str | None,
    inner_parts: list[Part],
    client: genai.Client,
    model_for_summary: str,
    model_for_counting: str,
    limiter,
    count_tokens,
    settings: CompressionSettings,
    source_key: str | None = None,
    allow_legacy_cache_read: bool = True,
    force_summarize: bool = False,
    eval_criteria_text: str | None = None,
) -> list[Part]:
    """Return either the original inner_parts or a compressed summary Part (wrapped by caller with tags).

    Args:
        force_summarize: If True, summarize even if the document is under the chunk_target_tokens threshold.
                         This is used by the budget-aware planner to force summarization of small docs.
    """
    logger = get_logger()
    # Small-doc fast path (no compression, no cache write)
    # UNLESS force_summarize is True (budget-aware planner decided we need to compress this)
    toks = _count_parts_safe_mixed(
        model=model_for_counting, parts=inner_parts,
        count_tokens=count_tokens, chars_per_token=settings.chars_per_token
    )
    if toks <= settings.chunk_target_tokens and not force_summarize:
        return inner_parts

    # alias (by source_key)
    alias_key: str | None = make_doccache_key_from_source_key(source_key) if source_key else None
    legacy_key: str = _hash_parts_list(inner_parts)
    legacy_key8 = legacy_key[:8]
    cache_key: str = _doc_cache_key(name, inner_parts, settings)
    cache_key8 = cache_key[:8]
    alias_key8 = (alias_key or "")[:8]
    doc_key_for_audit: str = cache_key
    # In-memory hits - check alias, then content
    with _DOC_MEMO_LOCK:
        cached_mem = _DOC_COMPRESS_MEMO.get(cache_key)
        if cached_mem is not None:
            _DOC_COMPRESS_MEMO.move_to_end(cache_key)
            logger.debug("[DOCCACHE] hit mem(cache) name=%s cache=%s legacy=%s",
             name or "unknown", cache_key8, legacy_key8)
            # For large docs, also append to aggregated (idempotent)
            if alias_key:
                from utils.llm.compactor_cache import append_doc_aggregated_summary_parts
                append_doc_aggregated_summary_parts(alias_key=alias_key, part_key=legacy_key, name=name, parts=list(cached_mem))
            return list(cached_mem)

    # Persistent hits load alias, then content
    persisted = load_doc_summary_from_store(cache_key)
    if persisted:
        logger.debug("[DOCCACHE] hit store(cache) name=%s key=%s legacy=%s",
                     name or "unknown", cache_key8, legacy_key8)
        with _DOC_MEMO_LOCK:
            _DOC_COMPRESS_MEMO[cache_key] = list(persisted)
            _DOC_COMPRESS_MEMO.move_to_end(cache_key)
            if len(_DOC_COMPRESS_MEMO) > _DOC_COMPRESS_MEMO_MAX:
                _DOC_COMPRESS_MEMO.popitem(last=False)
        # For large docs, also append to aggregated (idempotent)
        if alias_key:
            from utils.llm.compactor_cache import append_doc_aggregated_summary_parts
            append_doc_aggregated_summary_parts(alias_key=alias_key, part_key=legacy_key, name=name, parts=list(persisted))
        return list(persisted)

    # Optional back-compat read
    if allow_legacy_cache_read:
        persisted_legacy = load_doc_summary_from_store(legacy_key)
        if persisted_legacy:
            logger.debug("[DOCCACHE] hit store(legacy) name=%s legacy=%s -> using for cache=%s",
                         name or "unknown", legacy_key8, cache_key8)
            # Memoize under the *settings-aware* key for this run
            with _DOC_MEMO_LOCK:
                _DOC_COMPRESS_MEMO[cache_key] = list(persisted_legacy)
                _DOC_COMPRESS_MEMO.move_to_end(cache_key)
                if len(_DOC_COMPRESS_MEMO) > _DOC_COMPRESS_MEMO_MAX:
                    _DOC_COMPRESS_MEMO.popitem(last=False)
            if alias_key:
                from utils.llm.compactor_cache import append_doc_aggregated_summary_parts
                append_doc_aggregated_summary_parts(alias_key=alias_key, part_key=legacy_key, name=name, parts=list(persisted_legacy))
            return list(persisted_legacy)



    if (
        source_key
        and _is_tech_rfp_rfp_source(source_key)
        and os.getenv("COMPACTOR_WAIT_ON_MISS", "1").lower() in ("1", "true")
        and allow_legacy_cache_read
    ):

        logger.debug(
                "[DOCCACHE] wait start (eligible): name=%s legacy=%s cache=%s timeout=%ss",
                name or "unknown",
                legacy_key8,
                cache_key8,
                os.getenv("COMPACTOR_WAIT_TIMEOUT_SEC", "3600"),
            )
        waited = _wait_for_existing_summary(content_key=legacy_key)
        if waited:
            logger.debug("[DOCCACHE] late hit after wait: name=%s legacy=%s cache=%s",
                         name or "unknown", legacy_key8, cache_key8)
            with _DOC_MEMO_LOCK:
                _DOC_COMPRESS_MEMO[cache_key] = list(waited)
                _DOC_COMPRESS_MEMO.move_to_end(cache_key)
                if len(_DOC_COMPRESS_MEMO) > _DOC_COMPRESS_MEMO_MAX:
                    _DOC_COMPRESS_MEMO.popitem(last=False)
            if alias_key:
                from utils.llm.compactor_cache import append_doc_aggregated_summary_parts
                append_doc_aggregated_summary_parts(
                    alias_key=alias_key, part_key=legacy_key, name=name, parts=list(waited)
                )
            return list(waited)

    # In-flight guard: coalesce on CONTENT key
    with _DOC_MEMO_LOCK:
        ev = _DOC_INFLIGHT.get(cache_key)
        if ev is None:
            ev = threading.Event()
            _DOC_INFLIGHT[cache_key] = ev
            leader = True
        else:
            leader = False

    if not leader:
        logger.debug("[DOCCACHE] follower waiting cache_key=%s name=%s", cache_key8, name or "unknown")
        ev.wait()
        # after leader finishes, try mem again (either alias or content)
        with _DOC_MEMO_LOCK:
            cached_mem = _DOC_COMPRESS_MEMO.get(cache_key)
            if cached_mem is not None:
                logger.debug("[DOCCACHE] follower resume mem(content) key=%s", cache_key8)
                return list(cached_mem)
        logger.debug("[DOCCACHE] follower no-cache-after-wait key=%s -> compressing", cache_key8)

    # Heavy path chunk + summarize
    true_total = _true_count_range(
        model=model_for_counting, parts=inner_parts, i=0, j=len(inner_parts),
        count_tokens=count_tokens
    )
    logger.debug("[DOC] true_tokens: name=%s total=%d target_chunk=%d",
                 name or "unknown", true_total, settings.chunk_target_tokens)

    blocks = _chunk_parts_by_true_count(
        parts=inner_parts,
        target_tokens=settings.chunk_target_tokens,
        model_for_counting=model_for_counting,
        count_tokens=count_tokens,
    )
    logger.debug("[DOC] chunking: name=%s chunks=%d target≈%d",
                 name or "unknown", len(blocks), settings.chunk_target_tokens)

    sizes = [
        _count_parts_safe_mixed(
            model=model_for_counting, parts=b,
            count_tokens=count_tokens, chars_per_token=settings.chars_per_token
        ) for b in blocks
    ]
    if sizes:
        logger.debug(
            "[DOC] chunks: doc=%s count=%d min=%d p50=%d p90=%d max=%d",
            name or "unknown",
            len(blocks),
            min(sizes),
            sorted(sizes)[len(sizes)//2],
            sorted(sizes)[max(0, int(len(sizes)*0.9)-1)],
            max(sizes),
        )
    else:
        logger.debug("[DOC] chunks: doc=%s count=0", name or "unknown")

    summaries = _run_coro_blocking(
    _summarize_mixed_chunks_parallel(
            client=client,
            model_for_summary=settings.summary_model,
            model_for_counting=model_for_counting,
            chunks=blocks,
            limiter=limiter,
            count_tokens=count_tokens,
            settings=settings,
            doc_name=name,
            doc_key=doc_key_for_audit,
            doc_source_key=source_key,
            eval_criteria_text=eval_criteria_text,
        )
    )
    summaries = [s for s in summaries if s.strip()]

    label = settings.header_label
    body = f"{label} — {name or 'document'}\n\n" + "\n\n---\n\n".join(summaries)
    result = [Part.from_text(text=body)]

    # Store under BOTH keys (persist) + memoize + release waiters
    try:
        try:
            save_doc_summary_to_store(cache_key, result, source_key=source_key)
            logger.debug("[DOCCACHE] store(cache) key=%s legacy=%s", cache_key8, legacy_key8)
        except Exception:
            logger.debug("[DOCCACHE] store(content) failed (non-fatal)", exc_info=True)

        if alias_key:
            try:
                from utils.llm.compactor_cache import append_doc_aggregated_summary_parts
                append_doc_aggregated_summary_parts(
                    alias_key=alias_key, part_key=legacy_key, name=name, parts=result
                )
                logger.debug("[DOCCACHE] aggregated-append(alias) key=%s", alias_key8)
            except Exception:
                logger.debug("[DOCCACHE] aggregated-append(alias) failed (non-fatal)", exc_info=True)

        with _DOC_MEMO_LOCK:
            _DOC_COMPRESS_MEMO[cache_key] = list(result)
            _DOC_COMPRESS_MEMO.move_to_end(cache_key)
            if len(_DOC_COMPRESS_MEMO) > _DOC_COMPRESS_MEMO_MAX:
                _DOC_COMPRESS_MEMO.popitem(last=False)
    finally:
        with _DOC_MEMO_LOCK:
            ev = _DOC_INFLIGHT.pop(cache_key, None)
        if ev:
            ev.set()

    return result

def _compress_context_document_aware(
    *,
    context_parts: list[Part] | None = None,
    model_for_counting: str,
    get_client,
    count_tokens,
    limiter,
    settings: CompressionSettings,
    name_to_source_key: dict[str, str] | None = None,
    allow_legacy_cache_read: bool = True,
    target_tokens: int | None = None,
    eval_criteria_text: str | None = None,
    _pre_parsed: tuple[list[DocNode], list[Part]] | None = None,
) -> list[Part]:
    """
    SMART PARALLEL COMPACTOR: Detects <Start of File: name> ... </End of File: name> regions.

    algorithm:
    1. Single-pass parsing to extract all documents (CPU only - should be fast)
    2. Parallel preflight: cache check + token count for all docs concurrently
    3. Budget-aware planning: decide which docs need summarization based on aggregate budget
    4. Parallel summarization: only for docs that actually need it
    5. Reassemble with tags preserved

    handles the "many small files" edge case by considering aggregate budget, not just per-document size thresholds.
    TODO: need to introduce API counting tokens in a smarter way. Current algorithm is not very accurate.
    TODO: need to test this with a large number of large files.
    TODO: need to test this with a mix of large and small files.
    TODO: need to test this with a large number of small files.
    """
    client = get_client()
    logger = get_logger()
    t_start = time.perf_counter()

    budget_target = (
        target_tokens if target_tokens is not None else settings.target_total_tokens
    )

    if _pre_parsed is not None:
        docs, non_doc_parts = _pre_parsed
        parse_dur = 0.0
        n_docs = len(docs)
        for d in docs:
            d.tokens = 0
            d.cached_summary = None
            d.cache_key = None
            d.needs_summary = False
            d.force_summarize = False
            d.is_batched = False
        n_unterminated = sum(1 for d in docs if d.close_part is None)
        logger.debug(
            "[SMART] phase1 parse: REUSED docs=%d unterminated=%d non_doc_parts=%d",
            n_docs,
            n_unterminated,
            len(non_doc_parts),
        )
    else:
        t_parse = time.perf_counter()
        docs, non_doc_parts = _parse_documents_from_context(
            context_parts or [], name_to_source_key
        )
        parse_dur = time.perf_counter() - t_parse

        n_docs = len(docs)
        n_unterminated = sum(1 for d in docs if d.close_part is None)
        logger.debug(
            "[SMART] phase1 parse: docs=%d unterminated=%d non_doc_parts=%d dur=%.3fs",
            n_docs,
            n_unterminated,
            len(non_doc_parts),
            parse_dur,
        )

    if not docs:
        # No documents found, return original parts
        return list(context_parts or [])

    # PHASE 2: Parallel preflight (cache check + token count)
    t_preflight = time.perf_counter()
    docs = _preflight_documents_parallel(docs, model_for_counting, count_tokens, settings)
    preflight_dur = time.perf_counter() - t_preflight

    total_tokens = sum(d.tokens for d in docs)
    cache_hits = sum(1 for d in docs if d.cached_summary)
    logger.debug(
        "[SMART] phase2 preflight: docs=%d total_tokens=%d cache_hits=%d dur=%.2fs",
        n_docs, total_tokens, cache_hits, preflight_dur
    )

    # PHASE 3: Budget-aware summarization planning
    t_plan = time.perf_counter()
    docs_to_summarize, batches = _compute_summarization_plan(docs, budget_target, settings)
    plan_dur = time.perf_counter() - t_plan

    n_individual = len(docs_to_summarize)
    n_batches = len(batches)
    n_batched_docs = sum(len(b.docs) for b in batches)
    logger.debug(
        "[SMART] phase3 plan: individual=%d batches=%d batched_docs=%d dur=%.3fs",
        n_individual, n_batches, n_batched_docs, plan_dur
    )

    # PHASE 4: Parallel summarization
    t_summarize = time.perf_counter()
    doc_summaries: Dict[str, List[Part]] = {}

    # 4a. Summarize individual large documents in parallel
    if docs_to_summarize:
        def _summarize_single_doc(doc: DocNode) -> Tuple[str, List[Part]]:
            inner = _compress_one_document_if_needed(
                name=doc.name,
                inner_parts=doc.content,
                client=client,
                model_for_summary=settings.summary_model,
                model_for_counting=model_for_counting,
                limiter=limiter,
                count_tokens=count_tokens,
                settings=settings,
                source_key=doc.source_key,
                allow_legacy_cache_read=allow_legacy_cache_read,
                force_summarize=doc.force_summarize,
                eval_criteria_text=eval_criteria_text,
            )
            return (doc.name, inner)

        # Run individual doc summarization in parallel
        max_workers = min(settings.max_parallel_requests, len(docs_to_summarize))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                _submit_with_context(executor, _summarize_single_doc, doc): doc.name
                for doc in docs_to_summarize
            }
            for future in as_completed(futures):
                try:
                    name, summary = future.result()
                    doc_summaries[name] = summary
                    logger.debug("[SMART] summarized individual doc: %s", name)
                except Exception as e:
                    doc_name = futures[future]
                    logger.warning("[SMART] failed to summarize %s: %s", doc_name, e)

    # 4b. Summarize batches of small documents
    for batch in batches:
        try:
            batch_results = _summarize_batch(
                batch=batch,
                client=client,
                model_for_summary=settings.summary_model,
                model_for_counting=model_for_counting,
                limiter=limiter,
                count_tokens=count_tokens,
                settings=settings,
            )
            doc_summaries.update(batch_results)
        except Exception as e:
            logger.warning("[SMART] failed to summarize batch %s: %s", batch.batch_name, e)

    summarize_dur = time.perf_counter() - t_summarize
    logger.debug(
        "[SMART] phase4 summarize: summaries=%d dur=%.2fs",
        len(doc_summaries), summarize_dur
    )

    # PHASE 5: Reassemble with tags preserved
    t_reassemble = time.perf_counter()
    out: list[Part] = list(non_doc_parts)

    for doc in docs:
        # Determine which content to use
        if doc.name in doc_summaries:
            inner = doc_summaries[doc.name]
        elif doc.cached_summary:
            inner = doc.cached_summary
        else:
            inner = doc.content

        # Rebuild with tags
        out.append(doc.open_part)
        out.extend(inner)
        if doc.close_part:
            out.append(doc.close_part)

    reassemble_dur = time.perf_counter() - t_reassemble

    # PHASE 6: Self-correction check (FAST heuristic, no API)
    # If we're still significantly over budget, log warning for outer passes
    final_tokens_heuristic = _estimate_tokens_heuristic(out, settings.chars_per_token)

    # Check if we're still over budget by a significant margin (>20%)
    if final_tokens_heuristic > budget_target * 1.2:
        # We're still significantly over - log for debugging
        # The outer function (compress_if_needed_document_aware) will handle
        # this with tighter settings in pass 2/3
        logger.warning(
            "[SMART] still over budget after pass: heuristic=%d target=%d (%.1f%% over). "
            "Outer function will tighten settings for next pass.",
            final_tokens_heuristic, budget_target,
            (final_tokens_heuristic - budget_target) / budget_target * 100
        )

    total_dur = time.perf_counter() - t_start

    logger.debug(
        "[SMART] complete: docs=%d tokens=%d->%d (Δ=%+d) dur=%.2fs "
        "(parse=%.3fs preflight=%.2fs plan=%.3fs summarize=%.2fs reassemble=%.3fs)",
        n_docs, total_tokens, final_tokens_heuristic, final_tokens_heuristic - total_tokens, total_dur,
        parse_dur, preflight_dur, plan_dur, summarize_dur, reassemble_dur
    )

    return out


def _compress_context_document_aware_legacy(
    *,
    context_parts: list[Part],
    model_for_counting: str,
    get_client,
    count_tokens,
    limiter,
    settings: CompressionSettings,
    name_to_source_key: dict[str, str] | None = None,
    allow_legacy_cache_read: bool = True,
) -> list[Part]:
    """
    LEGACY: Serial document-aware compression. Kept for fallback/comparison.
    Detects <Start of File: name> ... </End of File: name> regions.
    Compresses each document independently (if needed), and re-emits the SAME tags.
    Preserves nesting. If a document is unterminated, we flush what we saw (no synthetic close).
    """
    client = get_client()
    logger = get_logger()
    out: list[Part] = []
    stack: list[dict] = []
    doc_count = 0
    open_unclosed = 0

    for p in context_parts:
        o = _match_open_tag(p)
        if o:
            name, open_part = o
            doc_count += 1
            stack.append({"name": name, "open": open_part, "content": []})
            continue

        c = _match_close_tag(p)
        if c and stack:
            close_name, close_part = c
            if stack[-1]["name"] == close_name:
                node = stack.pop()
                nm = _norm_name(node["name"])
                sk = (name_to_source_key or {}).get(nm)
                t_before = _count_parts_safe_mixed(
                    model=model_for_counting, parts=node["content"],
                    count_tokens=count_tokens, chars_per_token=settings.chars_per_token,
                )
                is_rfp = bool(sk and _is_tech_rfp_rfp_source(sk))
                needs_summary = t_before > settings.chunk_target_tokens
                logger.debug(
                    "[DOCCACHE] wait check: name=%s sk=%s is_rfp=%s needs_summary=%s toks=%d>%d",
                    node["name"],
                    (sk or "")[:80],
                    is_rfp,
                    needs_summary,
                    t_before,
                    settings.chunk_target_tokens,
                )

                logger.debug("[DOC] begin: name=%s tokens=%d parts=%d",
                            node["name"], t_before, len(node["content"]))

                inner = _compress_one_document_if_needed(
                    name=node["name"],
                    inner_parts=node["content"],
                    client=client,
                    model_for_summary=settings.summary_model,
                    model_for_counting=model_for_counting,
                    limiter=limiter,
                    count_tokens=count_tokens,
                    settings=settings,
                    source_key=sk,
                    allow_legacy_cache_read=allow_legacy_cache_read,
                )
                t_after = _count_parts_safe_mixed(
                    model=model_for_counting, parts=inner,
                    count_tokens=count_tokens, chars_per_token=settings.chars_per_token,
                )
                logger.debug("[DOC] done:  name=%s tokens_before=%d tokens_after≈%d Δ=%+d",
                            node["name"], t_before, t_after, t_after - t_before)

                rebuilt = [node["open"]] + inner + [close_part]  # preserve original tags
                if stack:
                    stack[-1]["content"].extend(rebuilt)
                else:
                    out.extend(rebuilt)
                continue

        # normal part
        if stack:
            stack[-1]["content"].append(p)
        else:
            out.append(p)

    # Unterminated docs flush as-is witout synthesizing a close tag
    while stack:
        node = stack.pop()
        open_unclosed += 1
        frags = [node["open"]] + node["content"]
        out.extend(frags)

    logger.debug("[DOC] scan summary: docs=%d unterminated=%d", doc_count, open_unclosed)

    return out

def compress_if_needed_document_aware(
    *,
    prefix_parts: List[Part],
    context_parts: List[Part],
    suffix_parts: List[Part],
    model_for_counting: str,
    get_client,
    count_tokens,
    limiter,
    settings: CompressionSettings,
    name_to_source_key: dict[str, str] | None = None,
    eval_criteria_text: str | None = None,
) -> List[Part]:
    """
    Document-aware compressor for the CONTEXT bucket (between BEGIN/END markers).
    - Preserves <filename> ... </filename> tags.
    - Summarizes text + images inside each document if the doc exceeds the per-doc target.
    - Leaves prefix/suffix (instructions) untouched.
    """
    logger = get_logger()
    t_total_start = time.perf_counter()

    def _pack(ctx: List[Part]) -> List[Part]:
        return list(prefix_parts) + list(ctx) + list(suffix_parts)

    total_tokens = _count_parts_safe_mixed(
        model=model_for_counting,
        parts=_pack(context_parts),
        count_tokens=count_tokens,
        chars_per_token=settings.chars_per_token,
    )
    per_doc_ceiling = getattr(settings, "per_doc_ceiling_tokens", 0)
    max_passes = max(1, int(getattr(settings, "max_compress_passes", 10)))

    logger.debug(
        "[CTX] start: tokens=%d target=%d hard=%d per_doc_ceiling=%d max_passes=%d",
        total_tokens,
        settings.target_total_tokens,
        settings.hard_limit_tokens,
        per_doc_ceiling,
        max_passes,
    )
    if total_tokens <= settings.target_total_tokens:
        logger.debug(
            "Doc-aware: total=%d <= target=%d; skipping compression.",
            total_tokens,
            settings.target_total_tokens,
        )
        return _pack(context_parts)

    docs, _non_doc_parts = _parse_documents_from_context(context_parts, name_to_source_key)
    if not docs:
        logger.debug("[CTX] no documents found; returning as-is.")
        return _pack(context_parts)
    pre_parsed = (docs, _non_doc_parts)

    overshoot_ratio = total_tokens / max(1, settings.target_total_tokens)
    current_settings = settings
    if overshoot_ratio > 5.0:
        current_settings = _auto_tighten_settings(
            current_settings,
            settings,
            1,
            overshoot_ratio,
        )
        logger.debug(
            "[CTX] pre-tightened for %.1fx overshoot: chunk=%d output=%d ceiling=%d",
            overshoot_ratio,
            current_settings.chunk_target_tokens,
            current_settings.summary_max_output_tokens,
            current_settings.per_doc_ceiling_tokens,
        )

    prev_tokens = total_tokens
    best_packed: List[Part] | None = None
    best_ctx: List[Part] = list(context_parts)
    best_tokens = total_tokens
    last_ctx_out: List[Part] = list(context_parts)
    pass_num = 0

    for pass_num in range(1, max_passes + 1):
        if pass_num > 1:
            overshoot_ratio = prev_tokens / max(1, settings.target_total_tokens)
            current_settings = _auto_tighten_settings(
                current_settings,
                settings,
                pass_num,
                overshoot_ratio,
            )

        pass_target = (
            settings.hard_limit_tokens
            if pass_num == max_passes
            else settings.target_total_tokens
        )

        t_pass = time.perf_counter()
        ctx_out = _compress_context_document_aware(
            context_parts=list(last_ctx_out),
            model_for_counting=model_for_counting,
            get_client=get_client,
            count_tokens=count_tokens,
            limiter=limiter,
            settings=current_settings,
            name_to_source_key=name_to_source_key,
            allow_legacy_cache_read=(pass_num == 1),
            target_tokens=pass_target,
            eval_criteria_text=eval_criteria_text,
            _pre_parsed=pre_parsed,
        )
        last_ctx_out = ctx_out
        packed = _pack(ctx_out)

        after = _count_parts_safe_mixed(
            model=model_for_counting,
            parts=packed,
            count_tokens=count_tokens,
            chars_per_token=settings.chars_per_token,
        )
        pass_dur = time.perf_counter() - t_pass
        logger.debug(
            "[CTX] pass=%d done: tokens=%d (Δ=%+d from start) dur=%.1fs "
            "ratio=%.3f settings=[chunk=%d out=%d ceiling=%d]",
            pass_num,
            after,
            after - total_tokens,
            pass_dur,
            after / max(1, total_tokens),
            current_settings.chunk_target_tokens,
            current_settings.summary_max_output_tokens,
            current_settings.per_doc_ceiling_tokens,
        )

        if after < best_tokens:
            best_packed = packed
            best_ctx = list(ctx_out)
            best_tokens = after

        if after <= settings.target_total_tokens:
            logger.debug("[CTX] success after pass=%d", pass_num)
            return packed

        if pass_num >= 3 and prev_tokens > 0:
            reduction = 1.0 - (after / max(1, prev_tokens))
            if reduction < 0.03:
                logger.warning(
                    "[CTX] stall detected at pass=%d (%.1f%% reduction from prev); "
                    "stopping dynamic loop.",
                    pass_num,
                    reduction * 100,
                )
                break

        prev_tokens = after

    result = best_packed if best_packed is not None else _pack(last_ctx_out)
    result_ctx = best_ctx if best_packed is not None else list(last_ctx_out)
    final_tokens = best_tokens
    total_dur = time.perf_counter() - t_total_start

    if final_tokens <= settings.hard_limit_tokens:
        logger.warning(
            "[CTX] FINAL OUTCOME=HARD_LIMIT: accepted after %d pass(es) "
            "(tokens=%d under hard=%d, over target=%d) dur=%.1fs",
            pass_num,
            final_tokens,
            settings.hard_limit_tokens,
            settings.target_total_tokens,
            total_dur,
        )
        return result

    logger.warning(
        "[CTX] still over hard limit (%d > %d) after %d passes; "
        "running emergency pass with minimum settings.",
        final_tokens,
        settings.hard_limit_tokens,
        pass_num,
    )
    emergency = _dc_replace(
        settings,
        chunk_target_tokens=30_000,
        summary_max_output_tokens=3_000,
        per_doc_ceiling_tokens=max(10_000, per_doc_ceiling) if per_doc_ceiling > 0 else 10_000,
    )
    t_emergency = time.perf_counter()
    ctx_emergency = _compress_context_document_aware(
        context_parts=list(result_ctx),
        model_for_counting=model_for_counting,
        get_client=get_client,
        count_tokens=count_tokens,
        limiter=limiter,
        settings=emergency,
        name_to_source_key=name_to_source_key,
        allow_legacy_cache_read=False,
        target_tokens=settings.hard_limit_tokens,
        eval_criteria_text=eval_criteria_text,
        _pre_parsed=pre_parsed,
    )
    packed_emergency = _pack(ctx_emergency)
    after_emergency = _count_parts_safe_mixed(
        model=model_for_counting,
        parts=packed_emergency,
        count_tokens=count_tokens,
        chars_per_token=settings.chars_per_token,
    )
    logger.debug(
        "[CTX] emergency pass done: tokens=%d dur=%.1fs",
        after_emergency,
        time.perf_counter() - t_emergency,
    )

    if after_emergency < final_tokens:
        result = packed_emergency
        result_ctx = list(ctx_emergency)
        final_tokens = after_emergency

    if final_tokens <= settings.hard_limit_tokens:
        logger.warning(
            "[CTX] FINAL OUTCOME=EMERGENCY_OK: tokens=%d under hard=%d dur=%.1fs",
            final_tokens,
            settings.hard_limit_tokens,
            time.perf_counter() - t_total_start,
        )
        return result

    logger.critical(
        "[CTX] FINAL OUTCOME=TRUNCATION: tokens=%d still > hard=%d after all "
        "dynamic passes + emergency; applying tag-preserving hard cap.",
        final_tokens,
        settings.hard_limit_tokens,
    )
    return _truncate_docs_preserving_tags(
        prefix_parts=list(prefix_parts),
        context_parts=list(result_ctx),
        suffix_parts=list(suffix_parts),
        model_for_counting=model_for_counting,
        count_tokens=count_tokens,
        chars_per_token=settings.chars_per_token,
        hard_limit=settings.hard_limit_tokens,
    )

def _truncate_docs_preserving_tags(
    *,
    prefix_parts: List[Part],
    context_parts: List[Part],
    suffix_parts: List[Part],
    model_for_counting: str,
    count_tokens,
    chars_per_token: int,
    hard_limit: int,
) -> List[Part]:
    logger = get_logger()

    def _tok(ps: List[Part]) -> int:
        try:
            return _count_tokens_strict(model_for_counting, ps, count_tokens)
        except Exception:
            # Fallback heuristic: text-only + media heuristic
            total_chars = 0
            media = 0
            for p in ps:
                t = getattr(p, "text", None)
                if isinstance(t, str):
                    total_chars += len(t)
                else:
                    media += 1
            return max(1, total_chars // max(1, chars_per_token)) + media * 512

    # Parse context into top-level segments: either raw Part or a doc node with open/content/close.
    # Nested docs remain inside their parent doc node's content (we treat only top-level docs as replaceable units).
    segments: list[object] = []
    stack: List[dict] = []

    for p in context_parts:
        o = _match_open_tag(p)
        if o:
            name, open_part = o
            stack.append({"name": name, "open": open_part, "content": []})
            continue

        c = _match_close_tag(p)
        if c and stack:
            close_name, close_part = c
            if stack[-1]["name"] == close_name:
                node = stack.pop()
                node["close"] = close_part

                if stack:
                    # nested: keep inside parent content
                    stack[-1]["content"].extend([node["open"]] + node["content"] + [close_part])
                else:
                    # top-level doc segment
                    segments.append(node)
                continue

        # normal part
        if stack:
            stack[-1]["content"].append(p)
        else:
            segments.append(p)

    # Flush any unterminated docs as raw parts (do NOT synthesize close tags)
    while stack:
        node = stack.pop()
        frags = [node["open"]] + node["content"]
        if stack:
            stack[-1]["content"].extend(frags)
        else:
            segments.extend(frags)

    def _render_segments(replace_set: set[int]) -> List[Part]:
        rendered: List[Part] = []
        doc_idx = 0
        for seg in segments:
            if isinstance(seg, dict) and "open" in seg and "content" in seg and "close" in seg:
                if doc_idx in replace_set:
                    placeholder = Part.from_text(text=f"[TRUNCATED TO FIT HARD LIMIT: {seg['name']}]")
                    rendered.extend([seg["open"], placeholder, seg["close"]])
                else:
                    rendered.extend([seg["open"]] + list(seg["content"]) + [seg["close"]])
                doc_idx += 1
            else:
                rendered.append(seg)  # raw Part
        return rendered

    # Count full assembled
    assembled_full = list(prefix_parts) + _render_segments(set()) + list(suffix_parts)
    total_full = _tok(assembled_full)
    if total_full <= hard_limit:
        return assembled_full

    # Collect doc costs (top-level only)
    doc_infos: list[dict] = []
    doc_idx = 0
    for seg in segments:
        if isinstance(seg, dict) and "open" in seg and "content" in seg and "close" in seg:
            full_parts = [seg["open"]] + list(seg["content"]) + [seg["close"]]
            placeholder_parts = [seg["open"], Part.from_text(text=f"[TRUNCATED TO FIT HARD LIMIT: {seg['name']}]"), seg["close"]]
            t_full = _tok(full_parts)
            t_ph = _tok(placeholder_parts)
            savings = max(0, t_full - t_ph)
            doc_infos.append(
                {"idx": doc_idx, "name": seg["name"], "t_full": t_full, "t_ph": t_ph, "savings": savings}
            )
            doc_idx += 1

    # Greedy: replace biggest savings first
    doc_infos.sort(key=lambda d: (d["savings"], d["t_full"]), reverse=True)

    replace_set: set[int] = set()
    est_total = total_full
    for d in doc_infos:
        if est_total <= hard_limit:
            break
        if d["savings"] <= 0:
            continue
        replace_set.add(d["idx"])
        est_total -= d["savings"]

    rendered_ctx = _render_segments(replace_set)
    final_parts = list(prefix_parts) + rendered_ctx + list(suffix_parts)
    final_total = _tok(final_parts)

    logger.warning(
        "[TRUNC] hard-cap applied: total_full=%d hard=%d replaced_docs=%d final_total=%d",
        total_full, hard_limit, len(replace_set), final_total
    )
    if replace_set:
        for d in doc_infos:
            if d["idx"] in replace_set:
                logger.warning(
                    "[TRUNC] placeholder inserted: doc=%s idx=%d saved≈%d tokens",
                    d["name"], d["idx"], d["savings"]
                )

    return final_parts
