"""Centralised Gemini helper utilities.

This module exposes a centralized interface for calling Google's Gemini
(Vertex-AI) models in production environments. It abstracts client creation,
rate-limiting, token counting, and error handling into a single shared utility.

## Key Features

-  Client is initialized using `HttpOptions(api_version="v1")` to route all
traffic through Vertex's REST v1 API.

- Uses `threading.local()` for sync contexts and `contextvars` for async tasks
to ensure isolated `genai.Client` instances per execution path.

- Unified entrypoints for sending prompt parts to Gemini, with shared logic for
retries, token counting, and rate-limiting.

- Implements a per-minute sliding window limiter (`RateLimiter`) for concurrent
access control based on token and request budgets.

- Applies Tenacity-backed exponential-jitter retries with automatic bailout on
persistent failure types.

-  `count_tokens()` wraps `models.count_tokens()` with fallback to `-1`
on failure.

- `to_parts()` coerces strings, bytes, or `Part` instances into a unified list
of `Part` objects for Gemini calls.



- Uses `threading.local()` for sync contexts and `contextvars` for async tasks
to ensure isolated `genai.Client` instances per execution path.
- Unified entrypoints for sending prompt parts to Gemini, with shared logic for
retries, token counting, and rate-limiting.
- Implements a per-minute sliding window limiter (`RateLimiter`) for concurrent
access control based on token and request budgets.
- Applies Tenacity-backed exponential-jitter retries with automatic bailout on
persistent failure types.
-  `count_tokens()` wraps `models.count_tokens()` with fallback to `-1`
on failure.
- `to_parts()` coerces strings, bytes, or `Part` instances into a unified list
of `Part` objects for Gemini calls.


Import pattern for tools:
```python
from utils.llm.LLM import Part, call_llm_async, call_llm_sync, count_tokens
```
"""

from __future__ import annotations

import time
import httpx
import asyncio
import threading
from collections import deque
from dataclasses import replace as _dc_replace
from typing import Any, Iterable, List, Sequence, Tuple

import random
import tenacity
from google import genai
from google.genai import types
from google.genai.types import Part
from google.genai import errors as gerrors


from typing import Optional, Mapping
from utils.core.log import get_logger
from utils.llm.context_cache import cache_enabled, prepare_cached_call_cfg


__all__ = [
    "Part",
    "RateLimiter",
    "get_client",
    "count_tokens",
    "to_parts",
    "call_llm_sync",
    "call_llm_async",
]

ENABLE_SMART_COMPRESSION = True
from utils.llm.context_compactor import (
    CompressionSettings,
    compress_with_memo,
    _split_prompt_context_suffix,
)

COMPRESSION_SETTINGS = CompressionSettings(
    enabled=ENABLE_SMART_COMPRESSION,
    hard_limit_tokens=978_576,
    target_total_tokens=930_000,
    prompt_reserve_tokens=20_000,
    chunk_target_tokens=400_000,
    summary_model="gemini-2.5-flash",
    summary_temperature=0.2,
    summary_max_output_tokens=30_000,
    max_rounds=3,
    header_label="CONTEXT",
    truncation_seed=None,
    chars_per_token=4,
)

# Configuration

MODEL_DEFAULT = "gemini-2.5-flash"
REQUESTS_PER_MINUTE = 200
TOKENS_PER_MINUTE = 10_000_000
_RETRIABLE_CLIENT_CODES = {429, 499}
# Thread- and task-local Client cache
_CLIENT = None
_LOCK = threading.Lock()


def _make_preview(parts, max_chars: int = 120) -> str:
    try:
        for p in to_parts(parts):
            t = getattr(p, "text", None)
            if isinstance(t, str) and t.strip():
                s = " ".join(t.split())  # collapse whitespace
                return (s[:max_chars] + "...") if len(s) > max_chars else s
    except Exception:
        pass
    return ""


def _debug_merge(meta: Optional[Mapping[str, str]], fallback_preview: str) -> dict:
    meta = dict(meta or {})
    meta.setdefault("caller", "unknown")
    meta.setdefault("preview", fallback_preview or "")
    return meta


class EmptyLLMResponseError(Exception):
    """Raised when LLM returns empty output (e.g. max tokens hit)."""
    pass


def _is_retriable(exc: Exception) -> bool:
    """Return True only for transient faults we want to retry."""
    if isinstance(exc, EmptyLLMResponseError):
        return True
    if isinstance(exc, gerrors.ServerError):  # 5xx
        return True
    if isinstance(exc, gerrors.ClientError):  # 4xx
        return getattr(exc, "code", None) in _RETRIABLE_CLIENT_CODES  # rate-limit and 499
    return False

_retry_policy = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retriable),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)

def _wrap_sdk_call(fn, *args, _log_model=None, _debug_meta: dict | None = None, **kwargs):
    """Run an SDK call, classify errors, and emit structured logs."""
    logger = get_logger()
    t0 = time.perf_counter()
    meta = _debug_meta or {}
    caller = meta.get("caller", "unknown")
    preview = meta.get("preview", "")
    callee = getattr(fn, "__name__", repr(fn))

    try:
        resp = fn(*args, **kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = getattr(resp, "usage_metadata", None)

        prompt_tok = getattr(usage, "prompt_token_count", -1) if usage else -1
        total_tok = getattr(usage, "total_token_count", -1) if usage else -1
        finish = (getattr(resp, "finish_reason", None) or "FINISH").upper()

        base_msg = (
            f"LLM Call OK | model={_log_model} | latency={latency_ms}ms | "
            f"prompt_tokens={prompt_tok} | total_tokens={total_tok}"
        )

        if finish.upper() != "FINISH":  # safety-stop, max-tokens, etc.
            logger.warning(base_msg + f" | finish_reason={finish}")
        else:
            logger.debug(base_msg)

        # TODO: FOR FUTURE USE
        # prompt_token_count = getattr(usage, "prompt_token_count", -1)
        # candidate_token_count = getattr(usage, "candidates_token_count", -1)
        # total_token_count = getattr(usage, "total_token_count", -1)
        # # traffic_type = getattr(usage, "traffic_type", None)

        # # Extract modality breakdown
        # prompt_details = getattr(usage, "prompt_tokens_details", []) or []
        # prompt_token_breakdown = ", ".join(
        #     f"{mod.modality.name}={mod.token_count}" for mod in prompt_details
        # )

        # candidate_details = getattr(usage, "candidates_tokens_details", []) or []
        # candidate_token_breakdown = ", ".join(
        #     f"{mod.modality.name}={mod.token_count}" for mod in candidate_details
        # )

        #     logger.debug(
        #         f"LLM call finished | model={_log_model} | latency_ms={latency_ms} | "
        #         f"prompt_tokens={prompt_token_count} [{prompt_token_breakdown}] | "
        #         f"candidates_tokens={candidate_token_count} [{candidate_token_breakdown}] | "
        #         f"total_tokens={total_token_count} |  "
        #         f"finish_reason={getattr(resp, 'finish_reason', 'Success')}"
        #     )
        # else:
        #     logger.debug(f"LLM call finished - no usage metadata found")

        return resp

    except gerrors.ClientError as e:
        code = getattr(e, "code", None)
        logger.error("LLM ClientError | caller=%s | callee=%s | model=%s | code=%s | err=%s | preview='%s'",
                     caller, callee, _log_model, code, e, preview, exc_info=True,)
        if code == 429:
            raise
        raise
    except gerrors.ServerError as e:
        logger.error("LLM ServerError | caller=%s | callee=%s | model=%s | err=%s | preview='%s'",
                     caller, callee, _log_model, e, preview, exc_info=True,)
        raise
    except Exception as e:
        logger.exception("LLM UnexpectedError | caller=%s | callee=%s | model=%s | err=%s | preview='%s'",
                         caller, callee, _log_model, e, preview)
        raise

def _create_client() -> genai.Client:
    from utils.llm.gcp_credentials import ensure_gcp_credentials_from_vault
    ensure_gcp_credentials_from_vault()
    http_options = types.HttpOptions(
        api_version="v1",
        client_args={
            "http2": False,
            "limits": httpx.Limits(
                max_keepalive_connections=100,
                max_connections=500,
                keepalive_expiry=60.0,
            ),
            "timeout": httpx.Timeout(30.0),
        },
    )
    return genai.Client(
        vertexai=True,
        project="gen-lang-client-0502418869",
        location="us-central1",
        http_options=http_options,
    )


def get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                _CLIENT = _create_client()
    return _CLIENT


# Rate limiter
class RateLimiter:
    """Simple token/request bucket for minute-long windows (thread- and loop-safe)."""

    def __init__(
        self, req_pm: int = REQUESTS_PER_MINUTE, tok_pm: int = TOKENS_PER_MINUTE
    ):
        self.req_pm = req_pm
        self.tok_pm = tok_pm
        self._mtx = threading.Lock()
        self._req: deque[float] = deque()
        self._tok: deque[tuple[float, int]] = deque()

    def acquire_sync(self, tokens: int = 0) -> None:
        """Blocking acquire for use from sync code or threads (no event loop)."""
        tokens = max(0, int(tokens))
        while True:
            now = time.time()
            window_start = now - 60.0

            with self._mtx:
                while self._req and self._req[0] < window_start:
                    self._req.popleft()
                while self._tok and self._tok[0][0] < window_start:
                    self._tok.popleft()

                used_tokens = sum(t for _, t in self._tok)
                can_req = len(self._req) < self.req_pm
                can_tok = (used_tokens + tokens) <= self.tok_pm

                if can_req and can_tok:
                    self._req.append(now)
                    self._tok.append((now, tokens))
                    return

                next_req = (self._req[0] + 60.0 - now) if self._req else 0.05
                next_tok = (self._tok[0][0] + 60.0 - now) if self._tok else 0.05
                sleep_for = max(
                    0.001,
                    (
                        min(x for x in (next_req, next_tok) if x > 0)
                        if (self._req or self._tok)
                        else 0.05
                    ),
                )

            time.sleep(sleep_for)

    async def acquire(self, tokens: int = 0):
        tokens = max(0, int(tokens))
        while True:
            now = time.time()
            window_start = now - 60.0

            with self._mtx:
                # evict old entries
                while self._req and self._req[0] < window_start:
                    self._req.popleft()
                while self._tok and self._tok[0][0] < window_start:
                    self._tok.popleft()

                used_tokens = sum(t for _, t in self._tok)
                can_req = len(self._req) < self.req_pm
                can_tok = (used_tokens + tokens) <= self.tok_pm

                if can_req and can_tok:
                    # consume budget and return
                    self._req.append(now)
                    self._tok.append((now, tokens))
                    return

                # compute next time a slot likely frees up
                next_req = (self._req[0] + 60.0 - now) if self._req else 0.05
                next_tok = (self._tok[0][0] + 60.0 - now) if self._tok else 0.05
                sleep_for = max(
                    0.001,
                    (
                        min(x for x in (next_req, next_tok) if x > 0)
                        if (self._req or self._tok)
                        else 0.05
                    ),
                )

            await asyncio.sleep(sleep_for)

_GLOBAL_LIMITER = RateLimiter()

def get_global_limiter() -> RateLimiter:
    return _GLOBAL_LIMITER


# Misc helpers
# TODO: check this function again
def to_parts(content: Sequence[Part | str | bytes]) -> List[Part]:
    "makes content gemini safe by converting to parts"
    parts: List[Part] = []
    for item in content:
        if isinstance(item, Part):
            parts.append(item)
        elif isinstance(item, str):
            parts.append(Part.from_text(text=item))
        elif isinstance(item, (bytes, bytearray)):
            parts.append(
                Part.from_bytes(data=bytes(item), mime_type="application/octet-stream")
            )
        else:
            raise TypeError(f"Unsupported Part type: {type(item)}")
    return parts

def count_tokens(model: str, parts: Iterable[Part]) -> int:
    """
    Counts tokens for text and media parts using Gemini model token counter.
    Returns -1 on failure.
    """
    logger = get_logger()
    try:
        parts = list(parts)
        if not all(isinstance(p, Part) for p in parts):
            raise TypeError("count_tokens expects only Part objects.")

        client = get_client()
        token_info = client.models.count_tokens(model=model, contents=parts)
        return token_info.total_tokens

    except Exception as e:
        logger.exception(f"Token counting failed: {e}")
        return -1

# Sync call helper
@_retry_policy
def call_llm_sync(
    prompt_parts: Sequence[Part],
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    cfg: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    debug_caller: str | None = None,
    debug_prompt_hint: str | None = None,
    name_to_source_key: dict[str, str] | None = None,
) -> str:
    """
    Synchronously calls a large language model (LLM) with the provided prompt
    parts and configuration.
        prompt_parts (Sequence[Part]): The sequence of prompt parts to send to the LLM.
        model (str | None, optional): The model name or identifier to use.
                                If None, uses the default model.
        system_instruction (str | None, optional): Optional system instruction
                                to guide the LLM's behavior.
        cfg (dict[str, Any] | None, optional): Configuration dictionary for the
                                LLM, such as temperature and max output tokens.
                                Defaults to {"temperature": 0.0, "max_output_tokens": 8192}.
        limiter (RateLimiter | None, optional): Optional rate limiter to control
                                token usage. If None, uses the global limiter.
        str: The generated text response from the LLM.
    """
    model = model or MODEL_DEFAULT
    cfg = cfg or {"temperature": 0.0, "max_output_tokens": 8192}
    limiter = limiter or _GLOBAL_LIMITER
    prompt_parts = to_parts(prompt_parts)
    prompt_parts = compress_with_memo(
        parts=prompt_parts,
        model_for_counting=model,
        limiter=limiter,
        settings=COMPRESSION_SETTINGS,
        name_to_source_key=name_to_source_key,
    )
    tokens = count_tokens(model, prompt_parts)
    if tokens < 0:
        tokens = COMPRESSION_SETTINGS.target_total_tokens

    # Safety net: if real count still exceeds hard limit, re-compress once
    # with emergency-tightened settings (different settings -> different memo key).
    if tokens > COMPRESSION_SETTINGS.hard_limit_tokens:
        _sync_logger = get_logger()
        _sync_logger.warning(
            "[LLM-SYNC] post-compress still over hard limit: %d > %d; "
            "re-compressing with emergency settings.",
            tokens,
            COMPRESSION_SETTINGS.hard_limit_tokens,
        )
        _emergency = _dc_replace(
            COMPRESSION_SETTINGS,
            max_compress_passes=5,
            chunk_target_tokens=50_000,
            summary_max_output_tokens=5_000,
            per_doc_ceiling_tokens=10_000,
        )
        prompt_parts = compress_with_memo(
            parts=prompt_parts,
            model_for_counting=model,
            limiter=limiter,
            settings=_emergency,
            name_to_source_key=name_to_source_key,
        )
        tokens = count_tokens(model, prompt_parts)
        if tokens < 0:
            tokens = COMPRESSION_SETTINGS.target_total_tokens
        _sync_logger.debug(
            "[LLM-SYNC] post-emergency tokens=%d hard=%d",
            tokens,
            COMPRESSION_SETTINGS.hard_limit_tokens,
        )

    limiter.acquire_sync(tokens)

    client = get_client()
    chat = client.chats.create(
        model=model, config={"system_instruction": system_instruction or ""}
    )

    preview = debug_prompt_hint or _make_preview(prompt_parts)
    meta = _debug_merge({"caller": debug_caller, "preview": preview}, preview)

    resp = _wrap_sdk_call(
        chat.send_message, prompt_parts, config=cfg, _log_model=model, _debug_meta=meta
    )
    return resp.text


# Async call helper
async def call_llm_async(
    prompt_parts: Sequence[Part],
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    cfg: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    debug_caller: str | None = None,
    debug_prompt_hint: str | None = None,
    name_to_source_key: dict[str, str] | None = None,
) -> str:
    """
    Asynchronously calls a language model (LLM) with the given prompt parts and
    configuration.
    Args:
        prompt_parts (Sequence[Part]): The parts of the prompt to send to the LLM.
        model (str | None, optional): The model name to use. Defaults to MODEL_DEFAULT
                                if not specified.
        system_instruction (str | None, optional): Optional system instruction
                                to guide the LLM's behavior.
        cfg (dict[str, Any] | None, optional): Configuration dictionary for the
                                LLM (e.g., temperature, max_output_tokens).
        limiter (RateLimiter | None, optional): Optional rate limiter to control
                                token usage. Defaults to _GLOBAL_LIMITER.
    Returns:
        str: The generated response text from the LLM.
    Raises:
        ValueError: If token estimation fails before acquiring from the limiter.
    Notes:
        - This function uses a retry policy for robustness.
        - The LLM call is executed in a separate thread to avoid blocking the event loop.
    """
    logger = get_logger()
    model = model or MODEL_DEFAULT
    base_cfg = cfg or {"temperature": 0.0, "max_output_tokens": 2048}
    limiter = limiter or _GLOBAL_LIMITER

    prompt_parts = to_parts(prompt_parts)

    def _compress():
        return compress_with_memo(
            parts=prompt_parts,
            model_for_counting=model,
            limiter=limiter,
            settings=COMPRESSION_SETTINGS,
            name_to_source_key=name_to_source_key,
        )

    prompt_parts = await asyncio.to_thread(_compress)

    tokens = count_tokens(model, prompt_parts)
    if tokens < 0:
        tokens = COMPRESSION_SETTINGS.target_total_tokens

    # Safety net: if real count still exceeds hard limit, re-compress once
    # with emergency-tightened settings (different settings -> different memo key).
    if tokens > COMPRESSION_SETTINGS.hard_limit_tokens:
        logger.warning(
            "[LLM-ASYNC] post-compress still over hard limit: %d > %d; "
            "re-compressing with emergency settings.",
            tokens,
            COMPRESSION_SETTINGS.hard_limit_tokens,
        )
        _emergency = _dc_replace(
            COMPRESSION_SETTINGS,
            max_compress_passes=5,
            chunk_target_tokens=50_000,
            summary_max_output_tokens=5_000,
            per_doc_ceiling_tokens=10_000,
        )

        def _emergency_compress():
            return compress_with_memo(
                parts=prompt_parts,
                model_for_counting=model,
                limiter=limiter,
                settings=_emergency,
                name_to_source_key=name_to_source_key,
            )

        prompt_parts = await asyncio.to_thread(_emergency_compress)
        tokens = count_tokens(model, prompt_parts)
        if tokens < 0:
            tokens = COMPRESSION_SETTINGS.target_total_tokens
        logger.debug(
            "[LLM-ASYNC] post-emergency tokens=%d hard=%d",
            tokens,
            COMPRESSION_SETTINGS.hard_limit_tokens,
        )

    await limiter.acquire(tokens)

    attempt_state = {"attempt": 0}
    preview = debug_prompt_hint or _make_preview(prompt_parts)
    meta = _debug_merge({"caller": debug_caller, "preview": preview}, preview)

    @_retry_policy
    def _blocking_with_retry() -> str:
        attempt_state["attempt"] += 1
        attempt = attempt_state["attempt"]

        this_cfg = dict(base_cfg)
        if attempt > 1:
            this_cfg["temperature"] = round(random.uniform(0.2, 0.7), 2)
            this_cfg["seed"] = random.randint(1, 10_000)

        client = get_client()
        chat = client.chats.create(
            model=model, config={"system_instruction": system_instruction or ""}
        )
        resp = _wrap_sdk_call(
            chat.send_message,
            prompt_parts,
            config=this_cfg,
            _log_model=model,
            _debug_meta=meta,
        )
        result = getattr(resp, "text", None) or getattr(resp, "content", None)

        if not result or not result.strip():
            logger.error(f"[call_llm_async] Attempt {attempt}: cfg={this_cfg}")
            raise EmptyLLMResponseError("LLM returned empty text")

        return result

    try:
        return await asyncio.to_thread(_blocking_with_retry)
    except EmptyLLMResponseError:
        import uuid
        from pathlib import Path

        # All retries failed - dump image
        for part in prompt_parts:
            if part.mime_type == "application/octet-stream":
                dump_dir = Path("llm_debug_failed_images")
                dump_dir.mkdir(exist_ok=True)
                img_path = dump_dir / f"failed_image_{uuid.uuid4().hex[:8]}.png"
                img_path.write_bytes(part.data)
                print(f"❌ Saved failed image to {img_path}")
                break
        raise


def _strip_context_for_cached_call(
    parts: Sequence[Part],
) -> Tuple[List[Part], int, int, int]:
    """
    Remove the heavy context (RFP/CONTRACTOR buckets) when using cached_content.
    Returns (stripped_parts, tok_prefix, tok_context, tok_suffix) — token counts are heuristic (len chars // 4)
    for logging only; we do exact counting below if available.
    """
    logger = get_logger()
    pfx, ctx, sfx = _split_prompt_context_suffix(list(parts))

    # Heuristic counts just for visibility logs (exact count comes later):
    def _est(ps: Sequence[Part]) -> int:
        t = 0
        for p in ps:
            s = getattr(p, "text", None)
            if isinstance(s, str):
                t += len(s)
        return max(1, t // 4)

    tp, tc, ts = _est(pfx), _est(ctx), _est(sfx)
    if ctx:
        logger.debug(
            "[CACHE-STRIP] removing context parts: est_tokens context≈%d (pfx≈%d sfx≈%d)",
            tc,
            tp,
            ts,
        )
    else:
        logger.debug(
            "[CACHE-STRIP] no explicit context markers found; sending full parts as-is (pfx≈%d sfx≈%d)",
            tp,
            ts,
        )
    return (pfx + sfx, tp, tc, ts)


async def call_llm_async_cache(
    prompt_parts: Sequence[Part],
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    cfg: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    cached_content: str | None = None,
    debug_caller: str | None = None,
    debug_prompt_hint: str | None = None,
    name_to_source_key: dict[str, str] | None = None,
) -> str:
    """
    If cached_content is provided (and caching is enabled), call models.generate_content
    with the cached context. Otherwise, fall back to the non-cached chat route
    implemented in call_llm_async().
    """
    model = model or MODEL_DEFAULT
    cfg = cfg or {"temperature": 0.0, "max_output_tokens": 8192}
    limiter = limiter or _GLOBAL_LIMITER
    logger = get_logger()

    use_cache = bool(cached_content) and cache_enabled()
    parts = to_parts(prompt_parts)
    preview = debug_prompt_hint or _make_preview(parts)
    meta = _debug_merge({"caller": debug_caller, "preview": preview}, preview)

    if use_cache:
        stripped_parts, tp, tc, ts = _strip_context_for_cached_call(parts)
        try:
            tokens = count_tokens(model, stripped_parts)
        except Exception:
            tokens = max(1, (tp + ts))

        logger.debug(
            "LLM (cached-pre) model=%s cache=%s tokens_send=%s (pfx≈%d, stripped_ctx≈%d, sfx≈%d)",
            model,
            cached_content,
            tokens,
            tp,
            tc,
            ts,
        )

        if tokens < 0:
            tokens = COMPRESSION_SETTINGS.target_total_tokens
        await limiter.acquire(tokens)

        @_retry_policy
        def _blocking_cached() -> str:
            client = get_client()
            gen_cfg = prepare_cached_call_cfg(dict(cfg), cached_content)
            has_markers = any(
                isinstance(getattr(p, "text", None), str)
                and (
                    "### BEGIN RFP-REFERENCE" in p.text
                    or "### BEGIN CONTRACTOR-SUBMISSION" in p.text
                )
                for p in stripped_parts
            )
            if has_markers:
                logger.warning(
                    "[CACHE-STRIP] markers detected post-strip; removing all context again."
                )
                stripped, _, _, _ = _strip_context_for_cached_call(stripped_parts)
                send_parts = stripped
            else:
                send_parts = stripped_parts

            logger.debug(
                "LLM (cached) model=%s cache=%s sending_parts=%d",
                model,
                cached_content,
                len(send_parts),
            )
            resp = _wrap_sdk_call(
                client.models.generate_content,
                model=model,
                contents=send_parts,
                config=gen_cfg,
                _log_model=model,
                _debug_meta=meta,
            )
            result = resp.text
            if not result:
                raise RuntimeError(
                    f"LLM returned empty text (cached path): Result - {resp}"
                )
            return result

        return await asyncio.to_thread(_blocking_cached)

    return await call_llm_async(
        prompt_parts=parts,
        model=model,
        system_instruction=system_instruction,
        cfg=cfg,
        limiter=limiter,
        debug_caller=debug_caller,
        debug_prompt_hint=debug_prompt_hint,
        name_to_source_key=name_to_source_key,
    )


def main() -> bool:
    from utils.core.log import pid_tool_logger, set_logger
    import asyncio

    package_id = "system_check"
    logger = pid_tool_logger(package_id, "llm_test")
    set_logger(logger)

    print("LLM TEST START")

    try:
        # Sync test
        prompt = ["Say hello."]
        sync_response = call_llm_sync(prompt)
        if not isinstance(sync_response, str) or not sync_response.strip():
            raise ValueError("Invalid sync LLM response.")

        # Async test
        async def run_async_test():
            async_prompt = ["Say hello asynchronously."]
            async_response = await call_llm_async(async_prompt)

            if not isinstance(async_response, str) or not async_response.strip():
                raise ValueError("Invalid async LLM response.")
            return async_response

        asyncio.run(run_async_test())

        print("LLM TEST OK")
        return True

    except Exception as e:
        print("LLM TEST ERROR")
        raise


if __name__ == "__main__":
    main()
