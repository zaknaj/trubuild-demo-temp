from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.core.log import get_logger
from utils.vault import secrets
from typing import Optional, List
from google.genai.types import (
    Content,
    Part,
    CreateCachedContentConfig,
    UpdateCachedContentConfig,
    GenerateContentConfig,
)

# Config from Vault or ephemeral env (no .env)
GCS_BUCKET = secrets.get("GCS_BUCKET", default="") or ""
_use_cache = (secrets.get("USE_CONTEXT_CACHE", default="") or "").strip().lower()
USE_CONTEXT_CACHE = _use_cache in ("true", "1", "yes")

CONTEXT_CACHE_TTL_SECONDS = 21600
MAX_INLINE_REQUEST_BYTES = 10 * 1024 * 1024  # 10MB - limit of cache creation
# Gemini count_tokens request payload limit (500MB); keep headroom for JSON/base64 overhead.
COUNT_TOKENS_PAYLOAD_LIMIT = 400 * 1024 * 1024

def _vertex_env():
    project = "gen-lang-client-0502418869"
    location = "us-central1"
    return project, location


def _as_resource(model: str) -> str:
    project, location = _vertex_env()
    return (
        model
        if model.startswith("projects/")
        else (
            f"projects/{project}/locations/{location}/publishers/google/models/{model}"
        )
    )


def _inline_size_bytes(p) -> int:
    try:
        inline = getattr(p, "inline_data", None)
        data = getattr(inline, "data", None)
        return len(data) if data else 0
    except Exception:
        return 0


def _inline_offenders(parts: List[Part]) -> list[tuple[int, int]]:
    """
    Return offenders as [(index, size_bytes)] for any inline part > MAX_INLINE_REQUEST_BYTES.
    """
    offenders: list[tuple[int, int]] = []
    for i, p in enumerate(parts or []):
        sz = _inline_size_bytes(p)
        if sz > MAX_INLINE_REQUEST_BYTES:
            offenders.append((i, sz))
    return offenders


def _text_size_bytes(p: Part) -> int:
    try:
        t = getattr(p, "text", None)
        if t is None:
            return 0
        # UTF-8 size
        return len(t.encode("utf-8"))
    except Exception:
        return 0


def _total_inline_payload_bytes(parts: List[Part]) -> int:
    """Sum of all inline_data bytes + inline text bytes (URIs don't count)."""
    total = 0
    for p in parts or []:
        inline = getattr(p, "inline_data", None)
        if inline and inline.data:
            total += len(inline.data)
        total += _text_size_bytes(p)
    return total


def _inline_encoded_bytes(p: Part) -> int:
    """Estimated request payload size for inline_data after base64 expansion."""
    n = _inline_size_bytes(p)
    if n == 0:
        return 0
    return (n + 2) // 3 * 4


def _part_payload_bytes(p: Part) -> int:
    """Estimated payload bytes for one part (text + base64 inline data)."""
    return _text_size_bytes(p) + _inline_encoded_bytes(p)


def _contents_to_parts(contents: List[Content]) -> List[Part]:
    """Flatten contents into a single list of parts."""
    parts: List[Part] = []
    for c in contents or []:
        parts.extend(getattr(c, "parts", []) or [])
    return parts


def _total_request_payload_bytes(parts: List[Part]) -> int:
    """Estimated request payload size (text + base64 inline data)."""
    return sum(_part_payload_bytes(p) for p in (parts or []))


def _chunk_parts_by_payload(parts: List[Part], limit: int) -> List[List[Part]]:
    """Split parts into chunks under payload limit."""
    if not parts:
        return []
    chunks: List[List[Part]] = []
    current: List[Part] = []
    current_size = 0
    for p in parts:
        sz = _part_payload_bytes(p)
        if current_size + sz > limit and current:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(p)
        current_size += sz
    if current:
        chunks.append(current)
    return chunks


def _short_model_name(m: str) -> str:
    return m.split("/models/")[-1] if "models/" in m else m


def _min_cache_tokens(model: str) -> int:
    m = _short_model_name(model).lower()
    return 2048 if ("pro" in m and "flash" not in m) else 1024


def _count_tokens_for_contents(client, model: str, contents: List[Content]) -> int:
    try:
        info = client.models.count_tokens(model=model, contents=contents)
        return getattr(info, "total_tokens", -1)
    except Exception:
        parts: List[Part] = []
        for c in contents or []:
            parts.extend(getattr(c, "parts", []) or [])
        info = client.models.count_tokens(model=model, contents=parts)
        return getattr(info, "total_tokens", -1)


def cache_enabled() -> bool:
    return USE_CONTEXT_CACHE


def get_client_cache():
    from utils.llm.LLM import get_client

    return get_client()


def make_composite_contents(parts: List[Part]) -> List[Content]:
    return [Content(role="user", parts=parts)]


def create_cache(
    *,
    model: str,
    contents: List[Content],
    display: str,
    ttl_seconds: int | None = None,
    system_instruction: str | None = None,
) -> Optional[str]:

    logger = get_logger()

    if not USE_CONTEXT_CACHE:
        logger.debug("disabled by flag; skipping create")
        return None

    ttl = ttl_seconds if ttl_seconds is not None else CONTEXT_CACHE_TTL_SECONDS
    try:
        client = get_client_cache()
        model = _as_resource(model)

        parts = contents[0].parts if contents else []
        offenders = _inline_offenders(parts)

        # if too big or too small, skip cache
        if offenders:
            logger.warning(
                "unexpected inline >10MB found during cache-create; "
                "this should have been converted to gs:// earlier. Skipping cache. Offenders: %s",
                ", ".join(f"idx={i} size={sz}B" for i, sz in offenders),
            )
            return None  # proceed uncached

        total_tokens = _count_tokens_for_contents(client, model, contents)
        need = _min_cache_tokens(model)
        if 0 <= total_tokens < need:
            logger.debug(
                f"skip: cached content has {total_tokens} tokens (<{need}) for model '{_short_model_name(model)}'."
            )
            return None

        cache = client.caches.create(
            model=model,
            config=CreateCachedContentConfig(
                contents=contents,
                display_name=display,
                ttl=f"{ttl}s",
                system_instruction=system_instruction or None,
            ),
        )
        um = getattr(cache, "usage_metadata", None)
        if um:
            logger.debug(
                f"cache created OK model={model} name={cache.name} ttl={ttl}s "
                f"text={getattr(um,'text_count',None)} image={getattr(um,'image_count',None)} "
                f"total_tokens={getattr(um,'total_token_count',None)}"
            )
        else:
            logger.debug(
                f"cache created OK model={model} name={cache.name} ttl={ttl}s"
            )

        logger.debug(
            f"cache created OK model={model} name={cache.name} ttl={ttl}s display='{display}'"
        )
        return cache.name

    except Exception as e:
        logger.exception(
            f"create FAILED model={model} display='{display}': {e}"
        )
        raise

def delete_cache(name: Optional[str]) -> None:
    logger = get_logger()
    if not name:
        return
    try:
        client = get_client_cache()
        client.caches.delete(name=name)
        logger.debug(f"cache deleted OK name={name}")
    except Exception as e:
        logger.warning(f"cache delete FAILED name={name}: {e}")


def bump_ttl(name: Optional[str], ttl_seconds: int) -> None:
    logger = get_logger()
    if not name:
        return
    try:
        client = get_client_cache()
        updated = client.caches.update(
            name=name, config=UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
        )
        logger.debug(
            f"cache ttl bumped OK name={name} new_expire={updated.expire_time}"
        )
    except Exception as e:
        logger.warning(f"cache ttl bump FAILED name={name}: {e}")


def prepare_cached_call_cfg(
    cfg_dict: dict, cached_content: Optional[str]
) -> GenerateContentConfig:
    """Return a GenerateContentConfig with cached_content if provided."""
    if cached_content and USE_CONTEXT_CACHE:
        return GenerateContentConfig(cached_content=cached_content, **cfg_dict)
    return GenerateContentConfig(**cfg_dict)



def count_cached_tokens(model: str, contents: list[Content]) -> int:
    client = get_client_cache()
    model = _as_resource(model)
    parts = _contents_to_parts(contents)
    if not parts:
        return 0

    total_bytes = _total_request_payload_bytes(parts)
    if total_bytes <= COUNT_TOKENS_PAYLOAD_LIMIT:
        return _count_tokens_for_contents(client, model, contents)

    chunks = _chunk_parts_by_payload(parts, COUNT_TOKENS_PAYLOAD_LIMIT)
    if len(chunks) <= 1:
        return _count_tokens_for_contents(
            client, model, [Content(role="user", parts=chunks[0])]
        )

    # Preserve logger ContextVars while counting in worker threads.
    def _submit_with_context(executor, fn, *args, **kwargs):
        ctx = contextvars.copy_context()
        return executor.submit(ctx.run, fn, *args, **kwargs)

    def _count_one(chunk_parts: List[Part]) -> int:
        return _count_tokens_for_contents(
            client, model, [Content(role="user", parts=chunk_parts)]
        )

    total = 0
    max_workers = min(4, len(chunks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [_submit_with_context(executor, _count_one, chunk) for chunk in chunks]
        for future in as_completed(futures):
            n = future.result()
            if n < 0:
                return -1
            total += n
    return total

