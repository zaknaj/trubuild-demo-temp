from __future__ import annotations
import hashlib, re, json
from pathlib import Path
from google.genai.types import Part
from typing import Any, Dict, Optional, List, Tuple

from utils.storage.bucket import upload_file, get_file
from utils.core.log import get_logger


_ctx_package_id: str | None = None
_ctx_company_id: str | None = None


def _norm_name(n: str | None) -> str:
    return Path((n or "").strip()).name.casefold()


def _normalize_source_key(sk: str) -> str:
    from urllib.parse import unquote

    s = (sk or "").strip()
    s = unquote(s)

    # unify separators & spacing (before lowercasing to preserve unquote)
    s = s.replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    # strip scheme/bucket if present (e.g., s3://bucket/path -> path)
    if "://" in s:
        s = s.split("://", 1)[1]
        s = s.split("/", 1)[1] if "/" in s else ""

    # drop leading "company_id/<project-or-effective-project>/" if present
    package_id, company_id, _ = _ctx()
    if company_id:
        pref = f"{(company_id or '').strip().lower()}/"
        if s.startswith(pref):
            parts = s.split("/")
            if len(parts) >= 3:
                # remove company_id and the next segment (project/effective-project)
                s = "/".join(parts[2:])

    return s


def make_doccache_key_from_source_key(source_key: str) -> str:
    base = _normalize_source_key(source_key or "")
    return hashlib.sha256(base.encode("utf-8", "ignore")).hexdigest()


def set_compactor_context(*, package_id: str, company_id: str | None) -> None:
    """To be called at the start of each request/pipeline so loads/saves work."""
    global _ctx_package_id, _ctx_company_id
    _ctx_package_id = package_id
    _ctx_company_id = company_id


def _ctx() -> Tuple[str | None, str | None]:
    return _ctx_package_id, _ctx_company_id


def _iso_utc_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "ignore")).hexdigest()


def _blocks_root(doc_key: str) -> str:
    from utils.document.docingest import DOCSTORE_DIR

    return f"{DOCSTORE_DIR}compressed/blocks/{doc_key}"


def _block_dir(doc_key: str, block_key: str) -> str:
    return f"{_blocks_root(doc_key)}/{block_key}"


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _audit_part_dict(p: Part) -> Dict[str, Any]:
    d: Dict[str, Any] = {"kind": "unknown"}

    t = _safe_getattr(p, "text", None)
    if isinstance(t, str):
        d["kind"] = "text"
        d["text"] = t
        d["chars"] = len(t)
        d["sha256"] = _sha256_text(t)
        return d

    fd = _safe_getattr(p, "file_data", None)
    if fd and _safe_getattr(fd, "file_uri", None):
        d["kind"] = "uri"
        d["uri"] = _safe_getattr(fd, "file_uri", None)
        d["mime"] = _safe_getattr(fd, "mime_type", None) or _safe_getattr(
            p, "mime_type", None
        ) or "application/octet-stream"
        return d

    inline = _safe_getattr(p, "inline_data", None)
    if inline and _safe_getattr(inline, "data", None):
        data = _safe_getattr(inline, "data", b"") or b""
        if not isinstance(data, (bytes, bytearray)):
            data = str(data).encode("utf-8", "ignore")
        d["kind"] = "inline"
        d["mime"] = _safe_getattr(inline, "mime_type", None) or _safe_getattr(
            p, "mime_type", None
        ) or "application/octet-stream"
        d["bytes"] = len(data)
        d["sha256"] = _sha256_bytes(bytes(data))
        return d

    mt = _safe_getattr(p, "mime_type", None)
    if mt:
        d["mime"] = mt
    return d


def _audit_parts_list(parts: List[Part]) -> List[Dict[str, Any]]:
    return [_audit_part_dict(p) for p in (parts or [])]


def _audit_usage_metadata(resp: Any) -> Dict[str, Any]:
    um = _safe_getattr(resp, "usage_metadata", None) or _safe_getattr(
        resp, "usageMetadata", None
    )
    if not um:
        return {}

    out: Dict[str, Any] = {}
    for k in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
    ):
        v = _safe_getattr(um, k, None)
        if v is not None:
            out[k] = v

    for k in (
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
        "cachedContentTokenCount",
    ):
        v = _safe_getattr(um, k, None)
        if v is not None and k not in out:
            out[k] = v
    return out


def _audit_candidates(resp: Any) -> List[Dict[str, Any]]:
    cands = _safe_getattr(resp, "candidates", None) or []
    out: List[Dict[str, Any]] = []
    try:
        for c in cands:
            out.append(
                {
                    "finish_reason": _safe_getattr(c, "finish_reason", None)
                    or _safe_getattr(c, "finishReason", None),
                    "safety_ratings": _safe_getattr(c, "safety_ratings", None)
                    or _safe_getattr(c, "safetyRatings", None),
                }
            )
    except Exception:
        return []
    return out


def save_block_audit_input(
    *,
    doc_key: str,
    block_key: str,
    doc_name: Optional[str],
    source_key: Optional[str],
    model: str,
    system_instruction: str,
    head_text: str,
    block_parts: List[Part],
) -> None:
    package_id, company_id = _ctx()
    if not package_id:
        return

    payload = {
        "type": "compactor_block_input",
        "created_at": _iso_utc_now(),
        "doc_key": doc_key,
        "block_key": block_key,
        "doc_name": doc_name,
        "source_key": source_key,
        "model": model,
        "system_instruction": system_instruction,
        "head_text": head_text,
        "parts": _audit_parts_list(block_parts),
    }
    path = f"{_block_dir(doc_key, block_key)}/{block_key}_input.json"
    upload_file(
        package_id=package_id,
        file_path=path,
        json_string=json.dumps(payload, ensure_ascii=False),
        company_id=company_id,
    )


def save_block_audit_output(
    *,
    doc_key: str,
    block_key: str,
    doc_name: Optional[str],
    source_key: Optional[str],
    model: str,
    request_config: Dict[str, Any],
    response_text: str,
    response_obj: Any,
    duration_s: float,
    error: Optional[str] = None,
) -> None:
    package_id, company_id = _ctx()
    if not package_id:
        return

    payload: Dict[str, Any] = {
        "type": "compactor_block_output",
        "created_at": _iso_utc_now(),
        "doc_key": doc_key,
        "block_key": block_key,
        "doc_name": doc_name,
        "source_key": source_key,
        "model": model,
        "duration_s": float(duration_s),
        "request_config": request_config,
        "text": response_text or "",
        "usage_metadata": _audit_usage_metadata(response_obj),
        "candidates": _audit_candidates(response_obj),
    }
    if error:
        payload["error"] = error

    path = f"{_block_dir(doc_key, block_key)}/{block_key}_output.json"
    upload_file(
        package_id=package_id,
        file_path=path,
        json_string=json.dumps(payload, ensure_ascii=False),
        company_id=company_id,
    )


def _parts_to_text(parts: List[Part]) -> str:
    """Concatenate text Parts; mark non-text Parts so chat sees context."""
    out: list[str] = []
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            out.append(t)
        else:
            mt = getattr(p, "mime_type", "") or "binary"
            out.append(f"\n[media:{mt}]\n")
    return "\n".join(out).strip()


def _text_to_parts(s: str) -> List[Part]:
    """
    Summaries we produce are text-only; reconstruct as a single text Part.
    If you ever add richer structures, you can extend this.
    """
    return [Part.from_text(text=s or "")]


def _alias_path(alias_key: str) -> str:
    from utils.document.docingest import DOCSTORE_DIR

    return f"{DOCSTORE_DIR}compressed/by_source/{alias_key}.json"


def _looks_like_alias_hash(value: str | None) -> bool:
    s = (value or "").strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", s))


def load_doc_summary_for_source(source_key: str) -> Optional[List[Part]]:
    """
    look up per-source alias pointer, then load the content_key JSON.
    Returns Parts or None.
    """
    package_id, company_id = _ctx()
    logger = get_logger()
    if not package_id or not source_key:
        logger.debug(
            "[DOCCACHE] alias lookup SKIP: ctx_missing project=%s company=%s src=%s",
            package_id,
            company_id,
            source_key,
        )
        return None

    alias = (
        source_key.strip().lower()
        if _looks_like_alias_hash(source_key)
        else make_doccache_key_from_source_key(source_key)
    )
    pointer_path = _alias_path(alias)
    try:
        logger.debug(
            "[DOCCACHE] alias lookup: project=%s company=%s alias=%s path=%s src=%s",
            package_id,
            company_id,
            alias[:8],
            pointer_path,
            source_key,
        )
        obj = get_file(
            package_id=package_id, file_path=pointer_path, company_id=company_id
        )
        if not obj:
            logger.debug("[DOCCACHE] alias MISS: alias=%s", alias[:8])
            return None

        # pointer structure: {"content_key": "...", "updated_at": "...", "source_key": "..."}
        if isinstance(obj, (bytes, bytearray)):
            logger.debug(
                "[DOCCACHE] alias pointer load OK (bytes): alias=%s size=%d",
                alias[:8],
                len(obj),
            )
            obj = json.loads(obj.decode("utf-8", "ignore"))
        elif isinstance(obj, str):
            logger.debug(
                "[DOCCACHE] alias pointer load OK (str): alias=%s size=%d",
                alias[:8],
                len(obj),
            )
            obj = json.loads(obj)
        elif isinstance(obj, dict):
            logger.debug(
                "[DOCCACHE] alias pointer load OK (dict): alias=%s keys=%s",
                alias[:8],
                list(obj.keys()),
            )
        else:
            logger.debug(
                "[DOCCACHE] alias pointer unexpected type=%s alias=%s",
                type(obj).__name__,
                alias[:8],
            )
            return None

        ck = (obj or {}).get("content_key")
        if not ck:
            logger.debug(
                "[DOCCACHE] alias malformed (no content_key): alias=%s", alias[:8]
            )
            return None

        logger.debug(
            "[DOCCACHE] alias pointer OK: alias=%s -> content_key=%s", alias[:8], ck[:8]
        )

        parts = load_doc_summary_from_store(ck)
        logger.debug(
            "[DOCCACHE] alias HIT: alias=%s -> key=%s parts=%d",
            alias[:8],
            ck[:8],
            len(parts or []),
        )
        return parts
    except Exception as e:
        logger.warning("[DOCCACHE] alias load error: %s", e)
        return None


def load_doc_summary_from_store(key: str) -> Optional[List[Part]]:
    from utils.document.docingest import DOCSTORE_DIR

    package_id, company_id = _ctx()
    if not package_id:
        return None

    json_path = f"{DOCSTORE_DIR}compressed/{key}.json"
    try:
        obj = get_file(
            package_id=package_id, file_path=json_path, company_id=company_id
        )
        logger = get_logger()
        # Normalize to text string (decoded) first
        raw: Optional[str] = None
        if isinstance(obj, (bytes, bytearray)):
            raw = obj.decode("utf-8", "ignore")
        elif isinstance(obj, str):
            raw = obj
        elif isinstance(obj, dict):
            text = obj.get("text", "")
            if text:
                logger.debug(
                    "[DOCCACHE] load hit(JSON-dict): key=%s bytes=%d",
                    key[:8],
                    len(text),
                )
                return _text_to_parts(text)
            logger.debug("[DOCCACHE] load miss (dict but no text): key=%s", key[:8])
            return None

        if raw is None or not raw.strip():
            logger.debug("[DOCCACHE] load miss: key=%s", key[:8])
            return None

        # Try JSON first
        try:
            data = json.loads(raw)
            if (
                isinstance(data, dict)
                and isinstance(data.get("text"), str)
                and data["text"]
            ):
                logger.debug(
                    "[DOCCACHE] load hit(JSON-str->dict): key=%s bytes=%d",
                    key[:8],
                    len(data["text"]),
                )
                return _text_to_parts(data["text"])
        except Exception:
            pass  # not JSON -> treat as plain text

        # Plain text fallback
        logger.debug(
            "[DOCCACHE] load hit(plain-text): key=%s bytes=%d", key[:8], len(raw)
        )
        return _text_to_parts(raw)

    except Exception as e:
        get_logger().debug("[DOCCACHE] load error key=%s err=%s", key[:8], e)
        return None


def save_doc_summary_to_store(
    key: str, parts: List[Part], *, source_key: str | None = None
) -> None:
    """
    Save the content JSON and (if source_key provided) a by_source pointer.
    """
    import time
    from utils.document.docingest import DOCSTORE_DIR

    package_id, company_id = _ctx()
    logger = get_logger()
    if not package_id:
        return

    text = _parts_to_text(parts)

    # write content
    json_path = f"{DOCSTORE_DIR}compressed/{key}.json"
    upload_file(
        package_id=package_id,
        company_id=company_id,
        location=json_path,
        json_string=json.dumps({"text": text}),
    )
    logger.debug(
        "[DOCCACHE] save ok: key=%s bytes=%d path=%s", key[:8], len(text), json_path
    )

    # write alias pointer (optional but recommended)
    if source_key:
        alias = make_doccache_key_from_source_key(source_key)
        pointer_path = _alias_path(alias)
        pointer = {
            "content_key": key,
            "source_key": source_key,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        upload_file(
            package_id=package_id,
            company_id=company_id,
            location=pointer_path,
            json_string=json.dumps(pointer),
        )
        logger.debug(
            "[DOCCACHE] alias save ok: alias=%s -> key=%s path=%s",
            alias[:8],
            key[:8],
            pointer_path,
        )

    else:
        logger.debug("[DOCCACHE] alias SKIP (no source_key) key=%s", key[:8])


def _docagg_path(key: str) -> str:
    from utils.document.docingest import DOCSTORE_DIR

    return f"{DOCSTORE_DIR}compressed/aggregated/{key}.txt"


def _docagg_manifest_path(key: str) -> str:
    from utils.document.docingest import DOCSTORE_DIR

    return f"{DOCSTORE_DIR}compressed/aggregated/{key}.meta.json"


def load_doc_aggregated_summary(key: str) -> Optional[str]:
    package_id, company_id = _ctx()
    if not package_id:
        return None
    try:
        obj = get_file(
            package_id=package_id, file_path=_docagg_path(key), company_id=company_id
        )
        if obj is None:
            return None
        if isinstance(obj, (bytes, bytearray)):
            s = obj.decode("utf-8", "ignore")
            return s if s.strip() else None
        if isinstance(obj, str):
            return obj if obj.strip() else None
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False)
        return str(obj)
    except Exception:
        return None


def _coerce_json_obj(x: Any) -> Dict[str, Any]:
    """Normalize get_file output into a dict."""
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            return json.loads(x.decode("utf-8", "ignore"))
        except Exception:
            return {}
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def _safe_json_load(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def append_doc_aggregated_summary_parts(
    *, alias_key: str, part_key: str, name: str | None, parts: List[Part]
) -> None:
    """
    Append the block summary 'parts' into a single aggregated file for this source doc.
    Idempotent by 'part_key' (content hash) so the same block won't be appended twice.
    """
    package_id, company_id = _ctx()
    if not package_id:
        return

    # load manifest of which parts have been appended
    manifest_path = _docagg_manifest_path(alias_key)
    manifest_raw = get_file(
        package_id=package_id, file_path=manifest_path, company_id=company_id
    )
    manifest = _coerce_json_obj(manifest_raw) or {}
    seen = set(manifest.get("parts", []))
    if part_key in seen:
        return  # already appended

    # load existing aggregated text (if any)
    agg_path = _docagg_path(alias_key)
    existing = get_file(
        package_id=package_id, file_path=agg_path, company_id=company_id
    )

    # turn parts into plain text
    body = _parts_to_text(parts).strip()
    if not body:
        return

    header = f"AGGREGATED — {name or 'document'}\n\n"
    new_text = existing or header
    if existing:
        new_text += "\n\n---\n\n"
    new_text += body

    # write aggregated file
    upload_file(
        package_id=package_id,
        company_id=company_id,
        location=agg_path,
        json_string=new_text,
    )

    # update manifest
    seen.add(part_key)
    manifest = {"parts": sorted(seen)}
    upload_file(
        package_id=package_id,
        company_id=company_id,
        location=manifest_path,
        json_string=json.dumps(manifest),
    )
