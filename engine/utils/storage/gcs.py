from __future__ import annotations

from utils.vault import secrets
from utils.core.log import get_logger
import io, time, uuid, os, json, hashlib, re
from typing import Optional, List, Tuple, Iterable, Dict, Any, Callable

from google.genai.types import Part
from utils.storage.bucket import get_file, upload_file
from utils.document.docingest import _as_json_dict, MANIFEST_PATH

from google.cloud import storage
from google.cloud.storage.blob import Blob
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config from Vault or ephemeral env (no .env)
GCS_BUCKET = secrets.get("GCS_BUCKET", default="") or ""

# Spill media + text into GCS to keep inline body < 10MB
INLINE_BODY_BUDGET = 8 * 1024 * 1024  # keep headroom under 10MB
TEXT_PART_LIMIT = 7 * 1024 * 1024  # single text/plain Part cap
MEDIA_PART_LIMIT = 7 * 1024 * 1024  # single binary Part cap
DEFAULT_INLINE_BUDGET = 8 * 1024 * 1024  # leave headroom under 10MB

# Where to persist contractor contexts (manifest + spilled files meta)
CONTRACTOR_CTX_ROOT = "tech_rfp_ctx"
CONTRACTOR_CTX_VERSION = 1

_SENT_RE = re.compile(r"(?<=[\.\!\?])\s+")  # simple sentence splitter

_CLIENT: storage.Client | None = None


def _client() -> storage.Client:
    """Singleton storage client to avoid re-opening ADC/FDs."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = storage.Client("gen-lang-client-0502418869")
    return _CLIENT


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    """Parse gs://bucket/object into (bucket, object)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    without = uri[len("gs://") :]
    bucket, _, name = without.partition("/")
    if not bucket or not name:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket, name


def upload_bytes_return_gs_uri(
    data: bytes,
    *,
    content_type: str,
    prefix: str = "tmp",
    bucket: Optional[str] = None,
    kms_key_name: Optional[str] = None,
    cache_control: Optional[str] = "no-cache",
) -> str:
    """
    Upload `data` as an object and return its gs:// URI.
    - Uses Standard storage by default (avoid early deletion fees).
    - Sets content-type explicitly so PDFs/images aren't treated as text.
    """
    logger = get_logger()
    if not data:
        raise ValueError("upload_bytes_return_gs_uri: data is empty")
    bkt_name = bucket or GCS_BUCKET
    if not bkt_name:
        raise ValueError("No bucket provided and GCS_BUCKET not set")
    try:
        client = _client()
        bkt = client.bucket(bkt_name)
        # object key: tmp/<prefix>/<ts>_<uuid>
        object_name = f"{prefix}/{int(time.time()*1000)}_{uuid.uuid4().hex}"
        blob: Blob = bkt.blob(object_name)
        if kms_key_name:
            blob.kms_key_name = kms_key_name
        if cache_control:
            blob.cache_control = cache_control

        # Upload from memory; set content-type
        bio = io.BytesIO(data)
        blob.upload_from_file(bio, content_type=content_type, rewind=True)
        return f"gs://{bkt_name}/{object_name}"

    except Exception as e:
        logger.exception(f"Failed uploading parts to GCS for cache: {e}")
        raise


def upload_many_return_gs_uris(
    items: Iterable[Tuple[str, bytes, str]],
    *,
    bucket: Optional[str] = None,
    cache_control: Optional[str] = "no-cache",
    workers: int = 16,
    progress_log_every: int = 50,
    progress_log_secs: float = 10.0,
) -> Dict[str, str]:
    """
    Upload many objects in parallel.
    items: iterable of (object_name, data_bytes, content_type)
    Returns: {object_name: gs://...}
    """
    logger = get_logger()
    bkt_name = bucket or GCS_BUCKET
    if not bkt_name:
        raise ValueError("No bucket provided and GCS_BUCKET not set")

    client = _client()
    bkt = client.bucket(bkt_name)

    items_list = list(items)
    total = len(items_list)
    if total == 0:
        logger.debug("[UPLOAD] nothing to upload")
        return {}

    logger.debug(
        "[UPLOAD] start: total=%d bucket=gs://%s workers=%d", total, bkt_name, workers
    )

    def _one(name: str, data: bytes, ctype: str) -> Tuple[str, str, float]:
        t0 = time.perf_counter()
        blob = bkt.blob(name)
        if cache_control:
            blob.cache_control = cache_control
        blob.upload_from_string(data, content_type=ctype)
        dt = time.perf_counter() - t0
        return name, f"gs://{bkt_name}/{name}", dt

    out: Dict[str, str] = {}
    submitted_at: Dict[str, float] = {}
    sizes: Dict[str, int] = {}

    t_start = time.perf_counter()
    last_progress = t_start
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for n, d, c in items_list:
            sizes[n] = len(d)
            submitted_at[n] = time.perf_counter()
            # logger.debug("[UPLOAD] queued name=%s size=%dB ctype=%s", n, len(d), c)
            futs.append(ex.submit(_one, n, d, c))

        for f in as_completed(futs):
            try:
                name, uri, dt = f.result()
                out[name] = uri
                completed += 1
                # logger.debug("[UPLOAD] done   name=%s size=%dB dur=%.2fs uri=%s",
                #              name, sizes.get(name, -1), dt, uri)
            finally:
                now = time.perf_counter()
                if (completed % progress_log_every == 0) or (
                    now - last_progress >= progress_log_secs
                ):
                    pend = total - completed
                    logger.debug(
                        "[UPLOAD] progress: %d/%d (pending=%d) elapsed=%.1fs",
                        completed,
                        total,
                        pend,
                        now - t_start,
                    )

                    if pend:
                        # figure out which ones still pending by set difference
                        done_names = set(out.keys())
                        pending_names = [
                            n for (n, _, _) in items_list if n not in done_names
                        ]
                        logger.debug("[UPLOAD] pending sample: %s", pending_names[:5])
                    last_progress = now

    logger.debug(
        "[UPLOAD] finished: ok=%d/%d total_dur=%.1fs",
        len(out),
        total,
        time.perf_counter() - t_start,
    )
    return out


def delete_gs_prefix(
    prefix: str,
    *,
    bucket: Optional[str] = None,
    batch_size: int = 1000,
) -> int:
    """
    Bulk delete all objects with the given prefix using Storage batch API.
    Returns: number of deleted objects.
    """
    logger = get_logger()
    bkt_name = bucket or GCS_BUCKET
    if not bkt_name:
        raise ValueError("No bucket provided and GCS_BUCKET not set")

    client = _client()
    bkt = client.bucket(bkt_name)

    deleted = 0
    batch: list[Blob] = []
    for blob in client.list_blobs(bkt, prefix=prefix):
        batch.append(blob)
        if len(batch) >= batch_size:
            with client.batch():
                for bb in batch:
                    bb.delete()
            deleted += len(batch)
            batch.clear()

    if batch:
        with client.batch():
            for bb in batch:
                bb.delete()
        deleted += len(batch)

    logger.debug(f"Deleted {deleted} objects under gs://{bkt_name}/{prefix}")
    return deleted


def _text_size_bytes(p) -> int:
    t = getattr(p, "text", None)
    return len(t.encode("utf-8")) if isinstance(t, str) else 0


def preflight_inline_sizes(parts: List[Part]) -> Dict[str, int]:
    """Compute inline byte sizes before any token counting."""
    totals = {"inline_text": 0, "inline_media": 0, "inline_total": 0, "max_part": 0}
    for p in parts or []:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            sz = len(t.encode("utf-8"))
            totals["inline_text"] += sz
            totals["max_part"] = max(totals["max_part"], sz)
        else:
            inline = getattr(p, "inline_data", None)
            if inline and inline.data:
                sz = len(inline.data)
                totals["inline_media"] += sz
                totals["max_part"] = max(totals["max_part"], sz)
    totals["inline_total"] = totals["inline_text"] + totals["inline_media"]
    return totals


def chunk_text_by_budget(
    s: str,
    *,
    max_bytes: int,
    max_tokens: int,
    count_tokens: Optional[Callable[[str, List[Part]], int]] = None,
    model_for_counting: str = "gemini-2.5-pro",
    chars_per_token: float = 4.0,
    sample_bytes: int = 256_000,
    verify_final: bool = True,
    verify_split_halves: bool = True,
) -> List[str]:
    """
    Chunk UTF-8 text into pieces that satisfy BOTH:
      - byte length <= max_bytes
      - token length <= max_tokens

    Strategy:
      - ONE precise token-count on a sample to calibrate chars/token (if count_tokens is provided)
      - Greedy accumulation by paragraphs, then by sentences if needed, using the calibrated heuristic
      - Falls back to raw UTF-8 slicing for pathological long spans
      - Optional light verification pass at the end (O(#chunks))
    """
    logger = get_logger()
    if not s:
        logger.debug("[CHUNK] empty input")
        return []

    # One-time calibration: estimate effective chars/token from a small sample
    eff_cpt = max(1.0, float(chars_per_token))
    if count_tokens is not None and sample_bytes > 0:
        sample = s[:sample_bytes]
        try:
            t0 = time.perf_counter()
            tok = max(
                1, count_tokens(model_for_counting, [Part.from_text(text=sample)])
            )
            dur = time.perf_counter() - t0
            eff_cpt = max(1.0, len(sample) / float(tok))
            logger.debug(
                "[CHUNK] calibration: bytes=%d tok=%d cpt≈%.3f dur=%.2fs",
                len(sample),
                tok,
                eff_cpt,
                dur,
            )
        except Exception as e:
            logger.debug(
                "[CHUNK] calibration failed; using fallback chars/token=%.3f (%s)",
                eff_cpt,
                e,
            )

    # Helpers
    def _tok_est(txt: str) -> int:
        if not txt:
            return 0
        # heuristic based on the one-time calibration
        return max(1, int(len(txt) / eff_cpt))

    def _fits(buf: List[str], nxt: str) -> bool:
        candidate = "".join(buf) + nxt
        if len(candidate.encode("utf-8")) > max_bytes:
            return False
        return _tok_est(candidate) <= max_tokens

    def _emit_raw_slice(txt: str, out: List[str]) -> None:
        """Last resort: UTF-8 byte slicing under max_bytes; token bound best-effort."""
        b = txt.encode("utf-8")
        start = 0
        while start < len(b):
            end = min(start + max_bytes, len(b))
            piece = b[start:end].decode("utf-8", errors="ignore")
            if not piece:
                # advance at least one byte
                piece = b[start : start + 1].decode("utf-8", errors="ignore")
                end = start + 1
            out.append(piece)
            start = end

    def _split_sentences(para: str) -> List[str]:
        # keep whitespace delimiters in output by re-adding space
        sentences = _SENT_RE.split(para)
        rebuilt = []
        for i, snt in enumerate(sentences):
            if i < len(sentences) - 1 and not snt.endswith("\n"):
                rebuilt.append(snt + " ")
            else:
                rebuilt.append(snt)
        return rebuilt

    # Greedy assembly using the calibrated heuristic
    out: List[str] = []
    cur: List[str] = []
    cur_bytes = 0

    paras = s.splitlines(keepends=True)
    logger.debug(
        "[CHUNK] begin: paras=%d max_bytes=%d max_tokens=%d eff_cpt≈%.3f",
        len(paras),
        max_bytes,
        max_tokens,
        eff_cpt,
    )

    for pi, para in enumerate(paras):
        p_bytes = len(para.encode("utf-8"))
        p_tokens_est = _tok_est(para)

        # If a single paragraph violates either limit, split by sentences first
        if p_bytes > max_bytes or p_tokens_est > max_tokens:
            if cur:
                out.append("".join(cur))
                cur, cur_bytes = [], 0

            sentences = _split_sentences(para) if para.strip() else [para]
            buf: List[str] = []
            for snt in sentences:
                snt_bytes = len(snt.encode("utf-8"))
                snt_tokens_est = _tok_est(snt)

                if snt_bytes > max_bytes or snt_tokens_est > max_tokens:
                    # Even a sentence is too large → raw slice by bytes
                    if buf:
                        out.append("".join(buf))
                        buf = []
                    _emit_raw_slice(snt, out)
                    continue

                if not buf:
                    buf = [snt]
                elif (len(("".join(buf) + snt).encode("utf-8")) <= max_bytes) and (
                    _tok_est("".join(buf) + snt) <= max_tokens
                ):
                    buf.append(snt)
                else:
                    out.append("".join(buf))
                    buf = [snt]
            if buf:
                out.append("".join(buf))
            continue

        # Paragraph fits by itself; try to append to current chunk
        if not cur:
            cur = [para]
            cur_bytes = p_bytes
        elif (
            cur_bytes + p_bytes <= max_bytes
            and _tok_est("".join(cur) + para) <= max_tokens
        ):
            cur.append(para)
            cur_bytes += p_bytes
        else:
            out.append("".join(cur))
            cur = [para]
            cur_bytes = p_bytes

        # Occasional progress breadcrumb
        if (pi + 1) % 1000 == 0:
            logger.debug(
                "[CHUNK] progress: processed_paras=%d chunks_so_far=%d",
                pi + 1,
                len(out) + (1 if cur else 0),
            )

    if cur:
        out.append("".join(cur))

    logger.debug("[CHUNK] heuristic pass: produced chunks=%d", len(out))

    # Light final verification (token-accurate), if requested
    if verify_final and count_tokens is not None:
        verified: List[str] = []
        overshoot_cnt = 0
        t0 = time.perf_counter()

        for ci, ch in enumerate(out):
            # quick byte check first (should already hold)
            if len(ch.encode("utf-8")) <= max_bytes:
                try:
                    tks = count_tokens(model_for_counting, [Part.from_text(text=ch)])
                except Exception:
                    # if counting fails, accept heuristic result
                    verified.append(ch)
                    continue

                if tks <= max_tokens:
                    verified.append(ch)
                else:
                    # rare overshoot: split once into halves (keeps verification light)
                    overshoot_cnt += 1
                    if verify_split_halves and len(ch) > 1:
                        mid = len(ch) // 2
                        a, b = ch[:mid], ch[mid:]
                        # re-check halves (single extra count each)
                        try:
                            if (
                                count_tokens(
                                    model_for_counting, [Part.from_text(text=a)]
                                )
                                > max_tokens
                            ):
                                _emit_raw_slice(a, verified)  # pathological; fallback
                            else:
                                verified.append(a)
                        except Exception:
                            verified.append(a)
                        try:
                            if (
                                count_tokens(
                                    model_for_counting, [Part.from_text(text=b)]
                                )
                                > max_tokens
                            ):
                                _emit_raw_slice(b, verified)
                            else:
                                verified.append(b)
                        except Exception:
                            verified.append(b)
                    else:
                        _emit_raw_slice(ch, verified)
            else:
                # Shouldn't happen; just slice by bytes
                _emit_raw_slice(ch, verified)

            # occasional progress for verification
            if (ci + 1) % 100 == 0:
                logger.debug("[CHUNK] verify progress: %d/%d", ci + 1, len(out))

        dt = time.perf_counter() - t0
        logger.debug(
            "[CHUNK] verify pass: chunks_in=%d chunks_out=%d overshoots=%d dur=%.2fs",
            len(out),
            len(verified),
            overshoot_cnt,
            dt,
        )
        out = verified
    else:
        logger.debug(
            "[CHUNK] verify pass skipped (verify_final=%s, count_tokens=%s)",
            str(verify_final),
            "yes" if count_tokens else "no",
        )

    logger.debug("[CHUNK] done: final_chunks=%d", len(out))
    return out


def _stable_ctx_prefix(package_id: str, company_id: str, contractor: str) -> str:
    """
    Storage path (not gs://). Use with (package_id, company_id) IO helpers.
    Files under this path get uploaded AND also mirrored to gs:// by upload_many_return_gs_uris.
    """
    safe_contractor = contractor.replace("/", "_")
    return f"{CONTRACTOR_CTX_ROOT}/{package_id}/{company_id}/{safe_contractor}"


def _ctx_manifest_path(prefix_path: str) -> str:
    return f"{prefix_path}/manifest.json"


def _fingerprint_sources_from_manifest(
    package_id: str, company_id: str
) -> Dict[str, str]:
    """
    Load docstore manifest (data/docstore/manifest.json) and return source_key -> processed filename.
    Uses content-fingerprint naming to detect changes cheaply.
    """
    try:
        raw = get_file(package_id, MANIFEST_PATH, company_id)
        m = _as_json_dict(raw)
        # values look like "<original>__<fingerprint>.json"
        return dict(m or {})
    except Exception:
        return {}


def _sources_fingerprint_for_contractor(
    contractor_metas: List[Dict[str, Any]],
    docstore_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Build a stable fingerprint map limited to the files that matter for this contractor
    using the processed filenames from docstore manifest.
    """
    out: Dict[str, str] = {}
    for meta in contractor_metas:
        sk = meta.get("source_key")
        if not sk:
            continue
        proc = docstore_map.get(sk, "")
        out[sk] = proc
    return out


def _combined_fingerprint(src_map: Dict[str, str]) -> str:
    """
    Hash the ordered (key, value) list for stability.
    """
    items = sorted((k, v) for k, v in (src_map or {}).items())
    h = hashlib.blake2b(digest_size=16)
    for k, v in items:
        h.update(k.encode("utf-8"))
        h.update(b"=")
        h.update((v or "").encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _manifest_to_ordered_parts(manifest: Dict[str, Any]) -> List[Part]:
    """
    Rebuild ordered Parts (URIs + tiny inline glue) from a saved manifest.

    NOTE: inline_glue items are already represented in the `ordered` list as
    `text_inline` nodes at their original positions, so reconstruction should
    iterate `ordered` only to preserve ordering and avoid duplicate text.
    """
    parts: List[Part] = []

    for node in manifest.get("ordered", []):
        if node.get("type") == "uri":
            parts.append(Part.from_uri(file_uri=node["uri"], mime_type=node["mime"]))
        elif node.get("type") == "text_uri":
            parts.append(Part.from_uri(file_uri=node["uri"], mime_type="text/plain"))
        elif node.get("type") == "text_inline":
            parts.append(Part.from_text(text=node.get("text", "")))
    return parts


def spill_media_and_text_to_gcs(
    parts: List[Part],
    *,
    prefix: str,
    force_spill_all_text: bool = False,
    inline_budget: int = INLINE_BODY_BUDGET,
    bucket: Optional[str] = None,
    max_text_part_bytes: int = TEXT_PART_LIMIT,
    max_text_part_tokens: int = 300_000,
    model_for_counting: str = "gemini-2.5-pro",
    count_tokens: Optional[Callable[[str, List[Part]], int]] = None,
) -> Tuple[List[Part], str, Dict[str, Any]]:
    """
    Tag-preserving spill:
      - Keeps <Start of File: name> and </End of File: name> inline.
      - Spills only the body text between those tags to GCS as text/plain URIs (chunked).
      - Leaves media as URIs in-place; respects RFP/CONTRACTOR section markers.
    """
    import re

    logger = get_logger()

    OPEN_RE = re.compile(r"^\s*<Start of File:\s*(.+?)>\s*$")
    CLOSE_RE = re.compile(r"^\s*</End of File:\s*(.+?)>\s*$")

    upload_items: List[Tuple[str, bytes, str]] = []  # (object_name, payload, mime)
    text_uploads: List[Tuple[str, bytes, str]] = []  # chunked text files
    ordered_nodes: List[Dict[str, Any]] = []  # manifest nodes in final order
    inline_glue: List[Dict[str, Any]] = []  # small inline texts we keep

    # Section state (RFP/CONTRACTOR buckets)
    in_rfp = False
    in_contr = False

    # Per-document state
    in_doc = False
    cur_doc_name: Optional[str] = None
    cur_doc_buf: List[str] = []  # text accumulated inside a doc (pre-spill)
    cur_doc_interspersed_nodes: List[Dict[str, Any]] = (
        []
    )  # media/uris occurring inside doc body

    inline_total = 0
    max_inline_part = 0
    glue_threshold = 8_192  # small inline allowance

    def _safe_name(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())[:200]

    def _name_for(
        kind: str, idx: int, payload: bytes, ext: str, doc_name: Optional[str] = None
    ) -> str:
        h = hashlib.blake2b(payload, digest_size=10).hexdigest()
        base = f"{prefix}/{kind}"
        if doc_name:
            base += f"/{_safe_name(doc_name)}"
        return f"{base}/{idx:04d}_{h}.{ext}"

    def _emit_text_chunks(doc_name: str, buf: List[str]) -> List[Dict[str, Any]]:
        if not buf:
            return []
        full = "\n".join(buf)
        chunks = chunk_text_by_budget(
            full,
            max_bytes=max_text_part_bytes,
            max_tokens=max_text_part_tokens,
            count_tokens=count_tokens,
            model_for_counting=model_for_counting,
        )
        nodes: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            payload = chunk.encode("utf-8")
            # store under kind="txt" and doc subfolder
            obj = _name_for("txt", idx, payload, "txt", doc_name=doc_name)
            text_uploads.append((obj, payload, "text/plain"))
            get_logger().debug(
                "[SPILL] %s chunk %04d size=%dB", doc_name, idx, len(payload)
            )
            nodes.append({"type": "text_uri", "uri_name": obj})
        return nodes

    def _flush_open_doc(doc_name: str):
        """Append the *open* tag inline."""
        tag = f"<Start of File: {doc_name}>\n"
        ordered_nodes.append({"type": "text_inline", "text": tag})

    def _flush_close_doc(doc_name: str):
        """Append the *close* tag inline."""
        tag = f"</End of File: {doc_name}>\n"
        ordered_nodes.append({"type": "text_inline", "text": tag})

    def _finalize_current_doc():
        """When closing a doc or at EOF: spill buffered text, insert interspersed nodes, add close tag if available."""
        nonlocal in_doc, cur_doc_name, cur_doc_buf, cur_doc_interspersed_nodes
        if not in_doc or not cur_doc_name:
            return
        # text chunks
        text_nodes = _emit_text_chunks(cur_doc_name, cur_doc_buf)
        # insert text chunks + interspersed nodes in the order they appeared:
        # we buffered text contiguously, media/uri nodes were captured in-order into cur_doc_interspersed_nodes
        # emit as: [text_nodes and media nodes interleaved if needed]. Since we didn't split text at media,
        # simplest is: text first, then interspersed nodes (preserves relative order of media among themselves).
        for n in text_nodes:
            ordered_nodes.append(n)
        for n in cur_doc_interspersed_nodes:
            ordered_nodes.append(n)
        # reset state (close tag is appended by caller if we actually saw one)
        in_doc = False
        cur_doc_name = None
        cur_doc_buf = []
        cur_doc_interspersed_nodes = []

    # Walk input parts
    for p in parts or []:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            st = t.strip()

            # Section markers (kept inline)
            if st.startswith("### BEGIN RFP-REFERENCE"):
                in_rfp, in_contr = True, False
                ordered_nodes.append({"type": "text_inline", "text": t})
                continue
            if st.startswith("### END RFP-REFERENCE"):
                in_rfp = False
                ordered_nodes.append({"type": "text_inline", "text": t})
                continue
            if st.startswith("### BEGIN CONTRACTOR-SUBMISSION"):
                in_contr, in_rfp = True, False
                ordered_nodes.append({"type": "text_inline", "text": t})
                continue
            if st.startswith("### END CONTRACTOR-SUBMISSION"):
                in_contr = False
                ordered_nodes.append({"type": "text_inline", "text": t})
                continue

            # Doc tags
            m_open = OPEN_RE.match(st)
            if m_open:
                # If a doc was already open but never closed, finalize it without a close tag
                if in_doc:
                    _finalize_current_doc()
                cur_doc_name = m_open.group(1)
                in_doc = True
                cur_doc_buf = []
                cur_doc_interspersed_nodes = []
                _flush_open_doc(cur_doc_name)
                continue

            m_close = CLOSE_RE.match(st)
            if m_close and in_doc:
                close_name = m_close.group(1)
                # finalize buffered text/media for the doc
                _finalize_current_doc()
                # append the *actual* close tag
                _flush_close_doc(close_name)
                continue

            # Normal text
            if in_doc:
                # Accumulate body text for this doc; will spill on close/EOF
                cur_doc_buf.append(t)
            else:
                # Outside any doc: keep as tiny inline glue, otherwise treat as contractor blob text
                b = t.encode("utf-8")
                if len(b) <= glue_threshold and inline_total + len(b) <= inline_budget:
                    ordered_nodes.append({"type": "text_inline", "text": t})
                    inline_glue.append({"text": t})
                    inline_total += len(b)
                    max_inline_part = max(max_inline_part, len(b))
                else:
                    # Spill as independent text chunk with a synthetic name; still outside docs.
                    payload = t.encode("utf-8")
                    obj = _name_for("loose", 1, payload, "txt")
                    upload_items.append((obj, payload, "text/plain"))
                    ordered_nodes.append({"type": "text_uri", "uri_name": obj})
            continue

        # Binary/media or pre-existing URIs
        inline = getattr(p, "inline_data", None)
        if inline and inline.data:
            data = inline.data
            obj = _name_for(
                "bin", 0, data, "bin", doc_name=(cur_doc_name if in_doc else None)
            )
            upload_items.append(
                (obj, data, inline.mime_type or "application/octet-stream")
            )
            node = {
                "type": "uri",
                "uri_name": obj,
                "mime": inline.mime_type or "application/octet-stream",
            }
            if in_doc:
                cur_doc_interspersed_nodes.append(node)
            else:
                ordered_nodes.append(node)
            continue

        fd = getattr(p, "file_data", None)
        if fd and getattr(fd, "file_uri", None):
            node = {
                "type": "uri_passthrough",
                "uri": fd.file_uri,
                "mime": getattr(fd, "mime_type", "application/octet-stream"),
            }
            if in_doc:
                cur_doc_interspersed_nodes.append(node)
            else:
                ordered_nodes.append(node)
            continue

        # Rare branch: unrecognized part — try to keep small textual bit inline
        t2 = getattr(p, "text", None)
        if isinstance(t2, str):
            b2 = t2.encode("utf-8")
            if len(b2) <= glue_threshold and inline_total + len(b2) <= inline_budget:
                ordered_nodes.append({"type": "text_inline", "text": t2})
                inline_glue.append({"text": t2})
                inline_total += len(b2)
                max_inline_part = max(max_inline_part, len(b2))
            else:
                if in_doc:
                    cur_doc_buf.append(t2)
                else:
                    payload = t2.encode("utf-8")
                    obj = _name_for("loose", 1, payload, "txt")
                    upload_items.append((obj, payload, "text/plain"))
                    ordered_nodes.append({"type": "text_uri", "uri_name": obj})

    # EOF: if a doc is still open, flush its content (no synthetic close)
    if in_doc:
        _finalize_current_doc()

    # If nothing heavy to spill and inline fits budget, short-circuit
    need_spill = (
        force_spill_all_text
        or bool(upload_items)
        or (inline_total > inline_budget)
        or (max_inline_part > TEXT_PART_LIMIT)
    )
    if not need_spill:
        out_parts: List[Part] = []
        for node in ordered_nodes:
            t = node.get("type")
            if t == "text_inline":
                out_parts.append(Part.from_text(text=node["text"]))
            elif t == "uri_passthrough":
                out_parts.append(
                    Part.from_uri(
                        file_uri=node["uri"],
                        mime_type=node.get("mime", "application/octet-stream"),
                    )
                )
        manifest = {
            "version": CONTRACTOR_CTX_VERSION,
            "ordered": [
                (
                    {"type": "text_inline", "text": n["text"]}
                    if n["type"] == "text_inline"
                    else {
                        "type": "uri",
                        "uri": n["uri"],
                        "mime": n.get("mime", "application/octet-stream"),
                    }
                )
                for n in ordered_nodes
                if n["type"] in ("text_inline", "uri_passthrough")
            ],
            "inline_glue": inline_glue,
        }
        return out_parts, prefix, manifest

    # Perform uploads (media + text)
    all_uploads = upload_items + text_uploads
    name_to_uri = upload_many_return_gs_uris(all_uploads, bucket=bucket, workers=16)

    # Build final Parts + manifest
    out_parts: List[Part] = []
    manifest_ordered: List[Dict[str, Any]] = []
    for node in ordered_nodes:
        t = node.get("type")
        if t == "text_inline":
            txt = node.get("text", "")
            out_parts.append(Part.from_text(text=txt))
            manifest_ordered.append({"type": "text_inline", "text": txt})
        elif t == "uri":
            uri = name_to_uri[node["uri_name"]]
            mime = node["mime"]
            out_parts.append(Part.from_uri(file_uri=uri, mime_type=mime))
            manifest_ordered.append({"type": "uri", "uri": uri, "mime": mime})
        elif t == "uri_passthrough":
            uri = node["uri"]
            mime = node.get("mime", "application/octet-stream")
            out_parts.append(Part.from_uri(file_uri=uri, mime_type=mime))
            manifest_ordered.append({"type": "uri", "uri": uri, "mime": mime})
        elif t == "text_uri":
            uri = name_to_uri[node["uri_name"]]
            out_parts.append(Part.from_uri(file_uri=uri, mime_type="text/plain"))
            manifest_ordered.append({"type": "text_uri", "uri": uri})

    manifest = {
        "version": CONTRACTOR_CTX_VERSION,
        "ordered": manifest_ordered,
        "inline_glue": inline_glue,
    }
    return out_parts, prefix, manifest


def load_or_build_contractor_context(
    *,
    package_id: str,
    company_id: str,
    contractor_name: str,
    rfp_parts: List[Part],
    proposal_parts: List[Part],
    bucket: Optional[str] = None,
    force_rebuild: bool = False,
) -> Tuple[List[Part], str, Dict[str, Any]]:
    """
    Returns (parts_with_uris, used_prefix, manifest).
    Reuses an existing GCS-backed context if source fingerprints match; otherwise rebuilds + persists manifest.
    """
    from utils.llm.LLM import count_tokens
    from tools.tech_rfp.tech_rfp import build_cached_context_parts

    logger = get_logger()
    prefix_path = _stable_ctx_prefix(package_id, company_id, contractor_name)
    manifest_path = _ctx_manifest_path(prefix_path)

    # Build source fingerprint map (docstore-processed filenames are stable per content)
    docstore_map = _fingerprint_sources_from_manifest(package_id, company_id)
    # Choose only sources for this contractor + all RFP files
    # We infer meta sources by scanning the markers inside parts: simpler is to pass metas in caller; here we skip and hash parts content fallback.

    # Fallback: compute a combined hash of textual content and media lengths (cheap & stable enough)
    def _parts_fallback_hash(parts: List[Part]) -> str:
        h = hashlib.blake2b(digest_size=16)
        for p in parts or []:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                h.update(b"T")
                h.update(len(t.encode("utf-8")).to_bytes(8, "big"))
            else:
                inline = getattr(p, "inline_data", None)
                if inline and inline.data:
                    h.update(b"B")
                    h.update(len(inline.data).to_bytes(8, "big"))
                else:
                    fd = getattr(p, "file_data", None)
                    if fd and getattr(fd, "file_uri", None):
                        h.update(b"U")
                        h.update(fd.file_uri.encode("utf-8"))
        return h.hexdigest()

    src_fp_map = {
        "rfp": _parts_fallback_hash(rfp_parts),
        "proposal": _parts_fallback_hash(proposal_parts),
    }
    combined_fp = _combined_fingerprint(src_fp_map)

    # Try load manifest
    if not force_rebuild:
        try:
            raw = get_file(package_id, manifest_path, company_id)
            man = _as_json_dict(raw)
            if (
                man
                and man.get("version") == CONTRACTOR_CTX_VERSION
                and man.get("combined_fingerprint") == combined_fp
            ):
                parts_from_manifest = _manifest_to_ordered_parts(man)
                return parts_from_manifest, prefix_path, man
        except Exception:
            pass  # proceed to rebuild

    # Build combined context (rfp + contractor)
    combined = build_cached_context_parts(rfp_parts, proposal_parts)

    # Spill & chunk
    t0 = time.perf_counter()
    spilled_parts, used_prefix, manifest_core = spill_media_and_text_to_gcs(
        combined,
        prefix=prefix_path,
        force_spill_all_text=True,  # ensure everything large is referenced
        inline_budget=INLINE_BODY_BUDGET,
        bucket=bucket,
        max_text_part_bytes=TEXT_PART_LIMIT,
        count_tokens=count_tokens,
    )
    dt = time.perf_counter() - t0
    logger.debug(
        "[SPILL] load or build contractor context finished in %.1fs (prefix=%s)",
        dt,
        used_prefix,
    )

    # Persist manifest
    final_manifest = {
        **manifest_core,
        "combined_fingerprint": combined_fp,
        "contractor": contractor_name,
        "prefix": used_prefix,
        "created_at": int(time.time()),
        "tool": "TECH-RFP",
    }
    try:
        upload_file(
            package_id=package_id,
            company_id=company_id,
            location=manifest_path,
            json_string=json.dumps(final_manifest, indent=2),
        )
    except Exception:
        get_logger().exception(
            "Failed to upload contractor context manifest (non-fatal)."
        )

    return spilled_parts, used_prefix, final_manifest


def prepare_parts_for_compression(
    parts: List[Part],
    *,
    contractor_prefix: str,
    bucket: Optional[str] = None,
    project: Optional[str] = None,
) -> Tuple[List[Part], str]:
    """
    Ensures Parts are suitable for compression:
      - If inline payload is big or any single part exceeds 7MB, we spill to GCS (URIs) first.
      - Otherwise, returns parts unchanged.
    Returns (prepared_parts, used_prefix).
    """
    logger = get_logger()
    pre = preflight_inline_sizes(parts)
    needs_spill = pre["inline_total"] > INLINE_BODY_BUDGET or pre["max_part"] > max(
        TEXT_PART_LIMIT, MEDIA_PART_LIMIT
    )

    if not needs_spill:
        return parts, contractor_prefix

    t0 = time.perf_counter()
    spilled_parts, used_prefix, _manifest = spill_media_and_text_to_gcs(
        parts,
        prefix=contractor_prefix,
        force_spill_all_text=True,
        inline_budget=INLINE_BODY_BUDGET,
        bucket=bucket,
        max_text_part_bytes=TEXT_PART_LIMIT,
    )
    dt = time.perf_counter() - t0
    logger.debug(
        "[SPILL] prepare parts for compression finished in %.1fs (prefix=%s)",
        dt,
        used_prefix,
    )

    return spilled_parts, used_prefix
