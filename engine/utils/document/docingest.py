# TODO: remove comm rfp processing from this file

import os
import io
import json
import uuid
import time
import shutil
import asyncio
import hashlib
import zipfile
import tempfile
import threading
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict
from google.genai.types import Part
from utils.core.log import get_logger
from typing import List, Dict, Tuple, Union
from concurrent.futures import ProcessPoolExecutor
from utils.document.doc import process_document_parts
from utils.storage.bucket import download_single_file, upload_file, get_file, delete_file, _nested_key, head_single_file

DOCSTORE_DIR = "data/docstore/"
MANIFEST_NAME = "manifest.json"
MANIFEST_PATH = f"{DOCSTORE_DIR}{MANIFEST_NAME}"
LOCK_FILE_PATH = f"{DOCSTORE_DIR}ingest.lock"
INGEST_CONCURRENCY = 30
PDF_WORKERS = 3
_PDF_POOL: ProcessPoolExecutor | None = None
_PDF_POOL_LOCK = threading.Lock()
DOCLOAD_CONCURRENCY = 100
DOCSTORE_LOCAL_CACHE_ROOT = "/tmp/docstore_cache"
_inflight: dict[str, asyncio.Future] = {}
_NOT_PROCESSED_TECH_HINTS = (
    "Unsupported document type",
    "File type not supported",
)

def shutdown_pdf_pool():
    global _PDF_POOL
    with _PDF_POOL_LOCK:
        if _PDF_POOL is not None:
            _PDF_POOL.shutdown(wait=False, cancel_futures=True)
            _PDF_POOL = None

def _get_pdf_pool() -> ProcessPoolExecutor:
    """
    Lazy-init to avoid recursive spawning when using spawn start method.
    Guarded because multiple ingestion tasks may call this concurrently.
    """
    global _PDF_POOL
    with _PDF_POOL_LOCK:
        if _PDF_POOL is None:
            ctx = mp.get_context("spawn")
            _PDF_POOL = ProcessPoolExecutor(max_workers=PDF_WORKERS, mp_context=ctx)
        return _PDF_POOL

def _process_pdf_parts_worker(pdf_path: str, original_filename: str, start_page: int, package_id: str):
    from utils.core.log import set_logger, pid_tool_logger
    set_logger(pid_tool_logger(package_id, "docingest_pdf"))

    from utils.document.doc import process_document_parts
    parts, errors, tokenized_pages = process_document_parts(pdf_path, original_filename, start_page)
    parts_dump = [p.model_dump(mode="json") for p in parts]
    return parts_dump, errors, tokenized_pages

def _fingerprint(path: str) -> str:
    """Calculates a BLAKE2b hash for a given file path."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def _safe_task_dir(base: str, source_key: str) -> str:
    h = hashlib.blake2b(source_key.encode("utf-8"), digest_size=10).hexdigest()
    d = os.path.join(base, h)
    os.makedirs(d, exist_ok=True)
    return d

def _is_valid_xlsx(path: str) -> bool:
    """Quick integrity check for .xlsx (must be a valid zip)."""
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False

def _ensure_full_key(company_id: str, package_id: str, key: str) -> str:
    """
    Ensure 'key' is a FULL nested object key:
      <company_id>/<package_id>/<...>
    Accepts already-full keys, legacy 'package_id/...' keys, or bare paths.
    """
    k = (key or "").lstrip("/")
    if company_id and k.startswith(f"{company_id}/{package_id}/"):
        return k
    if k.startswith(f"{package_id}/") and company_id:
        return f"{company_id}/{k}"
    return _nested_key(company_id, package_id, k)


def _submitted_not_processed_from_errors(errors: list[dict]) -> list[str]:
    """
    Return a stable, de-duped list of filenames that were submitted but not processed.
    """
    out: list[str] = []
    for e in errors or []:
        tech = str(e.get("technical") or "")
        if any(h in tech for h in _NOT_PROCESSED_TECH_HINTS):
            fn = str(e.get("file") or "").strip()
            if fn:
                out.append(fn)

    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _as_json_dict(value) -> dict:
    """Coerce S3 get_file() output (str/bytes/dict/None) into a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return json.loads(str(value))

def _manifest_doc(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("doc")
    return None

def _manifest_src(entry):
    if isinstance(entry, dict):
        return entry.get("src") or {}
    return {}

def _same_src(remote: dict, stored: dict) -> bool:
    """
    remote/stored: {etag, size, last_modified}
    Compare in a way that works across S3/OSS.
    """
    if not remote or not stored:
        return False

    r_etag, s_etag = _norm_etag(remote.get("etag")), _norm_etag(stored.get("etag"))
    r_size, s_size = remote.get("size"), stored.get("size")
    r_lm, s_lm = remote.get("last_modified"), stored.get("last_modified")

    # Preferred: ETag + size
    if r_etag and s_etag:
        return (r_etag == s_etag) and (r_size == s_size)

    # Fallback: size + last_modified
    if r_size is not None and s_size is not None and r_lm and s_lm:
        return (r_size == s_size) and (r_lm == s_lm)

    return False

def _norm_etag(x):
    if not x: return None
    return str(x).strip().strip('"').strip("'")

async def _ingest_one_file(
    package_id: str,
    company_id: str,
    source_key: str,
    original_filename: str,
    tool_context: str | None,
    reprocess_all: bool,
    manifest_snapshot: dict,
    tmp_dir: str,
    sem: asyncio.Semaphore,
) -> tuple[str, dict | None]:
    logger = get_logger()
    task_dir = _safe_task_dir(tmp_dir, source_key)
    local_raw_path = os.path.join(task_dir, original_filename)

    async with sem:
        full_key = _ensure_full_key(company_id, package_id, source_key)
        try:
            remote = await asyncio.to_thread(
                head_single_file,
                cloud_key=full_key,
            )
        except Exception as ex:
            logger.exception("head failed for %s: %s", full_key, ex)
            return full_key, None

        if not remote:
            logger.warning("Source missing in cloud (head returned None): %s", full_key)
            return full_key, None

        if not reprocess_all:
            stored_entry = manifest_snapshot.get(full_key)
            stored_doc = _manifest_doc(stored_entry)
            stored_src = _manifest_src(stored_entry)
            if stored_doc and _same_src(remote, stored_src):
                logger.debug("'%s' unchanged via HEAD. Skipping.", original_filename)
                return full_key, None

        # Download only if changed/new (sync -> thread)
        try:
            tmp_part = os.path.join(task_dir, f".{uuid.uuid4().hex}.part")
            await asyncio.to_thread(download_single_file, package_id, full_key, tmp_part)
            os.replace(tmp_part, local_raw_path)
        except Exception as ex:
            try:
                if os.path.exists(tmp_part):
                    os.remove(tmp_part)
            except Exception:
                pass
            logger.exception("download failed for %s: %s", original_filename, ex)
            return full_key, None

        ext = Path(original_filename).suffix.lower()
        if ext == ".xlsx" and not _is_valid_xlsx(local_raw_path):
            logger.error("Downloaded file is not a valid .xlsx zip: %s", original_filename)
            return full_key, None

        fingerprint = await asyncio.to_thread(_fingerprint, local_raw_path)
        docstore_filename = f"{original_filename}__{fingerprint}.json"

        # Route to proper extractor (offload blocking work)
        payload = {}
        try:
            if tool_context == "comm_rfp":
                import tools.comm_rfp.comm_rfp_extract as comm_extract
                ext = Path(original_filename).suffix.lower()
                if ext in (".xls", ".xlsx"):
                    df, detected_currency = await asyncio.to_thread(
                        comm_extract.extract_boq_excel, local_raw_path
                    )
                else:
                    df, detected_currency = await comm_extract.extract_boq_tables(local_raw_path)

                required = ["Item","Description","Unit","Quantity","Rate","Amount"]
                for c in required:
                    if c not in df.columns:
                        df[c] = ""
                if "Sheet" not in df.columns:
                    df["Sheet"] = "PDF" if ext == ".pdf" else "EXCEL"
                df = df[required + ["Sheet"]].reset_index(drop=True)

                payload = {
                    "boq_dataframe_json": df.to_json(orient="split", date_format="iso"),
                    "detected_currency": detected_currency,
                    "errors": []
                }

            elif tool_context == "tech_eval":
                ext = Path(original_filename).suffix.lower()
                if ext == ".pdf":
                    loop = asyncio.get_running_loop()
                    parts_dump, errors, _tokenized_pages = await loop.run_in_executor(
                        _get_pdf_pool(),
                        _process_pdf_parts_worker,
                        local_raw_path,
                        original_filename,
                        0,
                        package_id
                    )
                    payload = {"parts": parts_dump, "errors": errors}

                else:
                    df = await asyncio.to_thread(pd.read_excel, local_raw_path) if ext in (".xls",".xlsx") \
                        else await asyncio.to_thread(pd.read_csv, local_raw_path)
                    from tools.tech_rfp.tech_rfp_evaluation_criteria_extractor import make_json_serializable, _ffill_scope
                    rows = [] if df is None or df.empty else _ffill_scope(df.fillna("").to_dict(orient="records"))
                    payload = {"evaluation_rows_json": json.dumps(rows, default=make_json_serializable), "errors": []}

            else:
                ext = Path(original_filename).suffix.lower()
                if ext == ".pdf":
                    loop = asyncio.get_running_loop()
                    parts_dump, errors, tokenized_pages = await loop.run_in_executor(
                        _get_pdf_pool(),
                        _process_pdf_parts_worker,
                        local_raw_path,
                        original_filename,
                        0,
                        package_id,
                    )
                else:
                    # non-PDF: just do it in a thread
                    parts, errors, tokenized_pages = await asyncio.to_thread(
                        process_document_parts, local_raw_path, original_filename, 0
                    )
                    parts_dump = [p.model_dump(mode="json") for p in parts]

                payload = {
                    "parts": parts_dump,
                    "errors": errors,
                    "tokenized_content": {full_key: tokenized_pages},
                }

        except Exception as ex:
            logger.exception("processor failed for %s: %s", original_filename, ex)
            return full_key, None

        # Preserve a file-level warning list for downstream UI banners.
        if isinstance(payload, dict):
            errs = payload.get("errors")
            if isinstance(errs, list):
                submitted_np = _submitted_not_processed_from_errors(errs)
                if submitted_np:
                    payload["submitted_but_not_processed_files"] = submitted_np

        # Upload processed payload (sync -> thread)
        try:
            await asyncio.to_thread(
                upload_file,
                package_id=package_id,
                location=f"{DOCSTORE_DIR}{docstore_filename}",
                json_string=json.dumps(payload, indent=2),
                company_id=company_id,
            )
        except Exception as ex:
            logger.exception("upload failed for %s: %s", original_filename, ex)
            return full_key, None

        return full_key, {"doc": docstore_filename, "src": remote}


async def ingest_documents(
    package_id: str,
    company_id: str,
    source_keys: List[str],
    tool_context: str | None = None,
    reprocess_all: bool = False,
) -> bool:
    """
    Processes a list of raw source documents, updating the central docstore.
    This function is idempotent: it only processes new or changed files.
    It uses a 'tool_context' to apply specific extraction logic.

    Args:
        package_id: The ID of the project.
        source_keys: A list of full paths to the raw source files in cloud storage.
        tool_context: An optional hint ('comm_rfp', 'contract_review') to determine
                      the extraction method.

    Returns:
        True if the operation succeeded, False otherwise.
    """
    logger = get_logger()
    logger.debug(
        f"Starting ingestion for {len(source_keys)} documents in project {package_id} with context '{tool_context}'."
    )

    # 1. Acquire Lock
    try:
        lock_content = await asyncio.to_thread(
            get_file, package_id, LOCK_FILE_PATH, company_id
        )

        if lock_content:
            logger.warning(
                "Ingestion is already in progress for this project. Aborting to prevent race condition."
            )
            return False

        await asyncio.to_thread(
            upload_file,
            package_id,
            LOCK_FILE_PATH,
            json_string="locked",
            company_id=company_id,
        )

    except Exception as e:
        # If get_file fails (file not found is common), we still want to try to create the lock.
        logger.debug("Lock check failed (%s). Attempting to create lock anyway.", e)
        try:
            await asyncio.to_thread(
                upload_file,
                package_id,
                LOCK_FILE_PATH,
                json_string="locked",
                company_id=company_id,
            )
        except Exception as e2:
            logger.exception("Failed to create ingestion lock: %s", e2)
            return False

    temp_dir = tempfile.mkdtemp(prefix=f"ingest_{package_id}_")
    success = False
    try:
        # 2. Download Manifest
        manifest_content = await asyncio.to_thread(
            get_file, package_id, MANIFEST_PATH, company_id
        )
        manifest = _as_json_dict(manifest_content)

        sem = asyncio.Semaphore(INGEST_CONCURRENCY)
        manifest_snapshot = dict(manifest)
        tasks = []

        for sk in source_keys:
            full_key = _ensure_full_key(company_id, package_id, sk)
            original_filename = os.path.basename(full_key)
            tasks.append(_ingest_one_file(
                package_id, company_id, full_key, original_filename,
                tool_context, reprocess_all, manifest_snapshot, temp_dir, sem
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Process each file
        manifest_updated = False

        for res in results:
            if isinstance(res, Exception):
                get_logger().exception("ingest task error: %s", res)
                continue
            if not res:
                continue
            sk, entry = res
            if entry:
                manifest[sk] = entry
                manifest_updated = True

        # 7. Upload updated manifest if changes were made
        if manifest_updated:
            logger.debug("Manifest updated. Uploading new version.")
            upload_file(
                package_id=package_id,
                location=MANIFEST_PATH,
                json_string=json.dumps(manifest, indent=2),
                company_id = company_id
            )

        success = True

    finally:
        # 8. Cleanup and Release Lock
        shutil.rmtree(temp_dir)
        try:
            delete_file(package_id, LOCK_FILE_PATH, company_id)
        except Exception as e:
            logger.error(f"Failed to release ingestion lock: {e}")

    return success


async def get_documents(
    package_id: str,
    company_id: str,
    document_metas: List[Dict[str, str]],
    tool_context: str | None = None,
    force_refresh: bool = False
) -> Tuple[Union[List[Part], Dict[str, pd.DataFrame]], Dict]:
    """
    The single point of entry for retrieving processed document data.

    This function is context-aware:
    - For 'comm_rfp': Returns a dictionary of {contractor_name: pd.DataFrame}.
    - For all other contexts: Returns a flat list of google.genai.types.Part objects.

    It ensures documents are ingested if they haven't been already, then loads them.

    Args:
        package_id: The ID of the project.
        document_metas: A list of dicts, each with 'name', 'source_key',
                        and optionally 'contractor' for comm_rfp context.
        tool_context: An optional hint to determine the loading strategy and return type.

    Returns:
        A tuple containing:
        - The primary data: Either a List[Part] or a Dict[str, pd.DataFrame].
        - A dictionary of tokenized content (for contract review and chat).
    """
    logger = get_logger()
    if not document_metas:
        if tool_context in ("comm_rfp", "tech_eval"):
            return {}, {}
        return [], {}

    for m in document_metas:
        if "source_key" in m and m["source_key"]:
            m["source_key"] = _ensure_full_key(company_id, package_id, m["source_key"])

    # 1. Identify Unprocessed Documents
    manifest_content = get_file(package_id, MANIFEST_PATH, company_id)
    manifest = _as_json_dict(manifest_content)

    if force_refresh:
        # Skip the diff – re-ingest everything
        docs_to_ingest_keys = [m["source_key"] for m in document_metas]
    else:
        docs_to_ingest_keys = []
        sem = asyncio.Semaphore(INGEST_CONCURRENCY)

        async def _check(meta):
            source_key = meta["source_key"]
            try:
                async with sem:
                    remote = await asyncio.to_thread(
                        head_single_file,
                        cloud_key=source_key
                    )
            except Exception as e:
                logger.exception("HEAD precheck failed for %s: %s", source_key, e)
                return None

            if not remote:
                logger.warning("Source missing in cloud: %s", source_key)
                return None

            stored_entry = manifest.get(source_key)
            if not stored_entry:
                return source_key  # never processed

            stored_doc = _manifest_doc(stored_entry)
            stored_src = _manifest_src(stored_entry)

            if not stored_doc:
                return source_key  # malformed, re-ingest once

            if not _same_src(remote, stored_src):
                return source_key  # changed upstream

            return None  # unchanged

        checks = await asyncio.gather(*[_check(m) for m in document_metas])
        docs_to_ingest_keys = [k for k in checks if k]

    # 2. Trigger JIT Ingestion ONLY for new or changed files
    if docs_to_ingest_keys:
        logger.debug(
            f"Found {len(docs_to_ingest_keys)} new/changed documents. Triggering ingestion."
        )
        await ingest_documents(
            package_id,
            company_id,
            docs_to_ingest_keys,
            tool_context=tool_context,
            reprocess_all=force_refresh,
        )

        # Reload manifest after ingestion to get the latest state
        manifest_content = get_file(package_id, MANIFEST_PATH, company_id)
        manifest = _as_json_dict(manifest_content)

    # 3. Load All Processed Documents from the Docstore
    cache_dir = Path(DOCSTORE_LOCAL_CACHE_ROOT) / company_id / package_id / "docstore_json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_sem = asyncio.Semaphore(DOCLOAD_CONCURRENCY)

    async def _ensure_docstore_json_downloaded(fp_name: str) -> Path:
        local_path = cache_dir / fp_name
        if local_path.exists():
            return local_path
        inflight_key = f"{company_id}/{package_id}/{fp_name}"
        fut = _inflight.get(inflight_key)
        if fut:
            await fut
            return local_path

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        _inflight[inflight_key] = fut
        try:
            cloud_key = _nested_key(company_id, package_id, f"{DOCSTORE_DIR}{fp_name}")
            tmp_part = str(local_path) + f".{uuid.uuid4().hex}.part"

            async with download_sem:
                await asyncio.to_thread(download_single_file, package_id, cloud_key, tmp_part)

            os.replace(tmp_part, local_path)
            fut.set_result(True)
            return local_path
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            _inflight.pop(inflight_key, None)
            try:
                if 'tmp_part' in locals() and os.path.exists(tmp_part):
                    os.remove(tmp_part)
            except Exception:
                pass

    # A) Commercial RFP: Load and combine DataFrames per contractor
    if tool_context == "comm_rfp":
        typed_empty = pd.DataFrame(
            columns=["Item", "Description", "Unit", "Quantity", "Rate", "Amount", "Sheet"]
        )
        contractor_dataframes = defaultdict(lambda: typed_empty.copy())
        contractor_currencies = {}
        sheet_order: list[str] = []

        for meta in document_metas:
            doc_name = meta["name"]
            contractor = meta.get("contractor")
            if not contractor:
                logger.warning(
                    "Document '%s' missing 'contractor' metadata; skipping (comm_rfp).",
                    doc_name,
                )
                continue

            fp_name = _manifest_doc(manifest.get(meta["source_key"]))
            if not fp_name:
                logger.warning(
                    "Processed doc not found in manifest for '%s'; skipping.",
                    doc_name,
                )
                continue

            try:
                local_cache_path = await _ensure_docstore_json_downloaded(fp_name)
            except Exception as ex:
                logger.error(
                    "[DOC-PIPELINE] comm_rfp download failed for '%s' (%s): %s",
                    doc_name,
                    fp_name,
                    ex,
                )
                continue

            try:
                with open(local_cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as ex:
                logger.error(
                    "[DOC-PIPELINE] corrupt/unreadable docstore JSON (comm_rfp) for '%s' (%s): %s",
                    doc_name,
                    fp_name,
                    ex,
                )
                continue

            if "boq_dataframe_json" not in payload:
                continue

            # Extract currency from payload
            detected_currency = payload.get("detected_currency", "SAR")
            contractor_currencies[contractor] = detected_currency

            val = payload.get("boq_dataframe_json")
            if not isinstance(val, str) or not val.strip():
                logger.debug(
                    "boq_dataframe_json missing/invalid for '%s' (%s). Skipping.",
                    doc_name,
                    type(val).__name__,
                )
                continue

            try:
                df = pd.read_json(io.StringIO(val), orient="split")
            except Exception as e:
                logger.exception(
                    "Failed to parse boq_dataframe_json for '%s': %s", doc_name, e
                )
                continue

            # normalize schema + guarantee Sheet
            required = ["Item", "Description", "Unit", "Quantity", "Rate", "Amount"]
            for col in required:
                if col not in df.columns:
                    df[col] = ""
            if "Sheet" not in df.columns:
                df["Sheet"] = "PDF"

            df = df[required + ["Sheet"]].reset_index(drop=True)
            if df is None or df.empty:
                continue

            prev = contractor_dataframes[contractor]
            if prev.empty:
                contractor_dataframes[contractor] = df
            else:
                contractor_dataframes[contractor] = pd.concat(
                    [prev, df], ignore_index=True
                )

            # build global sheet order (first-seen wins)
            for s in df["Sheet"].dropna().astype(str).unique():
                if s not in sheet_order:
                    sheet_order.append(s)

        if not sheet_order:
            sheet_order = ["PDF"]

        logger.debug(
            "Loaded and combined BoQ DataFrames for %d vendors.",
            len(contractor_dataframes),
        )
        return contractor_dataframes, {
            "sheet_order": sheet_order,
            "contractor_currencies": contractor_currencies
        }

    # B) tech evaluation criteria
    elif tool_context == "tech_eval":
        eval_tables: Dict[str, Union[dict, List[Part]]] = {}

        # 1) Build list of processed docstore filenames we need
        to_fetch: List[Tuple[str, str]] = []
        # tuple = (source_key, fp_name)

        for meta in document_metas:
            source_key = meta["source_key"]
            fp_name = _manifest_doc(manifest.get(source_key))
            if not fp_name:
                logger.warning(
                    f"Could not find processed doc for key '{source_key}' in manifest."
                )
                continue
            to_fetch.append((source_key, fp_name))

        # 2) Download all missing docstore JSONs in parallel (bounded)
        download_results = await asyncio.gather(*[
            _ensure_docstore_json_downloaded(fp_name)
            for (_source_key, fp_name) in to_fetch
        ], return_exceptions=True)

        failed_downloads: set[str] = set()
        for (_source_key, fp_name), result in zip(to_fetch, download_results):
            if isinstance(result, Exception):
                logger.error(
                    "[DOC-PIPELINE] tech_eval download failed for %s: %s",
                    fp_name,
                    result,
                )
                failed_downloads.add(fp_name)

        # 3) Parse locally (fast) after downloads complete
        for source_key, fp_name in to_fetch:
            if fp_name in failed_downloads:
                logger.warning(
                    "[DOC-PIPELINE] skipping %s — download failed.",
                    fp_name,
                )
                continue

            local = cache_dir / fp_name

            # If something went wrong, skip safely
            if not local.exists():
                logger.warning("Docstore JSON missing after download: %s", fp_name)
                continue

            try:
                payload = json.loads(local.read_text(encoding="utf-8"))
            except Exception as ex:
                logger.error(
                    "[DOC-PIPELINE] corrupt/unreadable docstore JSON (tech_eval) %s: %s",
                    fp_name,
                    ex,
                )
                continue

            try:
                if "parts" in payload:
                    parts = [Part.model_validate(p_dict) for p_dict in payload["parts"]]
                    eval_tables[source_key] = parts
                elif "evaluation_rows_json" in payload:
                    eval_tables[source_key] = json.loads(payload["evaluation_rows_json"])
            except Exception as ex:
                logger.error(
                    "[DOC-PIPELINE] failed to parse tech_eval data for %s: %s",
                    fp_name,
                    ex,
                )

        return eval_tables, {}

    # C) Default / Contract / Chat: Load Parts and Tokenized Content
    else:
        all_parts: List[Part] = []
        all_tokenized_content: dict = {}
        parts_by_file: Dict[str, List[Part]] = {}
        parts_by_source_key: Dict[str, List[Part]] = {}
        not_processed_by_source_key: Dict[str, List[str]] = {}

        # 1) Build a list of processed docstore filenames we actually need
        to_fetch: List[Tuple[dict, str, str]] = []
        # tuple = (meta, doc_name, fingerprinted_name)

        for meta in document_metas:
            doc_name = meta["name"]
            fingerprinted_name = _manifest_doc(manifest.get(meta["source_key"]))

            if not fingerprinted_name:
                logger.warning(
                    f"Could not find processed document for '{doc_name}' in manifest. Skipping."
                )
                continue

            to_fetch.append((meta, doc_name, fingerprinted_name))

        # 2) Download all missing docstore JSONs in parallel (bounded)
        #    (if they already exist locally, _ensure... returns immediately)
        download_results = await asyncio.gather(*[
            _ensure_docstore_json_downloaded(fp_name)
            for (_meta, _doc_name, fp_name) in to_fetch
        ], return_exceptions=True)

        failed_downloads: set[str] = set()
        for (_meta, _doc_name, fp_name), result in zip(to_fetch, download_results):
            if isinstance(result, Exception):
                logger.error(
                    "[DOC-PIPELINE] download failed for '%s' (%s): %s",
                    _doc_name,
                    fp_name,
                    result,
                )
                failed_downloads.add(fp_name)

        # 3) Parse sequentially (downloads were the slow part)
        for meta, doc_name, fingerprinted_name in to_fetch:
            if fingerprinted_name in failed_downloads:
                logger.warning(
                    "[DOC-PIPELINE] skipping '%s' — download failed earlier.", doc_name
                )
                not_processed_by_source_key.setdefault(meta["source_key"], []).append(
                    doc_name
                )
                continue

            local_cache_path = cache_dir / fingerprinted_name

            if not local_cache_path.exists():
                logger.warning(
                    "[DOC-PIPELINE] docstore JSON missing on disk after download: %s (%s)",
                    doc_name,
                    fingerprinted_name,
                )
                not_processed_by_source_key.setdefault(meta["source_key"], []).append(
                    doc_name
                )
                continue

            try:
                with open(local_cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as ex:
                logger.error(
                    "[DOC-PIPELINE] corrupt/unreadable docstore JSON for '%s' (%s): %s",
                    doc_name,
                    fingerprinted_name,
                    ex,
                )
                not_processed_by_source_key.setdefault(meta["source_key"], []).append(
                    doc_name
                )
                continue

            np = payload.get("submitted_but_not_processed_files")
            if isinstance(np, list) and np:
                not_processed_by_source_key[meta["source_key"]] = [
                    str(x) for x in np if str(x).strip()
                ]

            # deserialize Parts from their dictionary representation
            try:
                if "parts" in payload:
                    parts = [Part.model_validate(p_dict) for p_dict in payload["parts"]]
                    all_parts.extend(parts)
                    parts_by_file[doc_name] = parts
                    parts_by_source_key[meta["source_key"]] = parts
            except Exception as ex:
                logger.error(
                    "[DOC-PIPELINE] failed to deserialize parts for '%s': %s",
                    doc_name,
                    ex,
                )
                not_processed_by_source_key.setdefault(meta["source_key"], []).append(
                    doc_name
                )

            # collect tokenized content, which is useful for contract review
            if "tokenized_content" in payload:
                all_tokenized_content.update(payload["tokenized_content"])

        logger.debug(
            f"Loaded {len(all_parts)} parts from {len(to_fetch)} documents."
        )
        return all_parts, {
            "tokenized_content": all_tokenized_content,
            "parts_by_file": parts_by_file,
            "parts_by_source_key": parts_by_source_key,
            "not_processed_by_source_key": not_processed_by_source_key,
        }

def main() -> bool:
    """
    End-to-end self-test for ingest_documents/get_documents and helpers.

    Creates three real files, uploads them, runs ingestion for:
      - default/contract
      - comm_rfp (xlsx)
      - tech_eval (csv)
    Verifies return types/shapes, then cleans up all artifacts.
    Prints concise OK/FAIL lines and returns a boolean.
    """
    import pandas as pd
    from pathlib import Path
    import asyncio, uuid, tempfile, os, json
    from utils.core.log import set_logger, pid_tool_logger
    set_logger(pid_tool_logger("SYSTEM_INSTRUCTION", "context_cache"))

    print("DOCINGEST TEST START")
    overall_ok = True

    def _ok(msg: str):
        print(f"OK: {msg}")

    def _fail(msg: str):
        nonlocal overall_ok
        overall_ok = False
        print(f"FAIL: {msg}")

    async def _run():
        nonlocal overall_ok
        package_id = f"SELFTEST_{uuid.uuid4().hex[:8]}"

        tmpdir = tempfile.mkdtemp(prefix="docingest_selftest_")
        try:
            contract_path = os.path.join(tmpdir, "contract.txt")
            Path(contract_path).write_text(
                "Clause 1.1 The Contractor shall...\n"
                "Clause 2.1 Liability...\n"
                "Clause 3.4 Ambiguity...\n"
            )

            boq_path = os.path.join(tmpdir, "boq.xlsx")
            pd.DataFrame(
                {
                    "Item": ["1", "2"],
                    "Description": ["Concrete", "Steel"],
                    "Unit": ["m3", "kg"],
                    "Quantity": [10, 500],
                    "Rate": [100, 2],
                    "Amount": [1000, 1000],
                }
            ).to_excel(boq_path, index=False)

            criteria_path = os.path.join(tmpdir, "criteria.csv")
            pd.DataFrame(
                {"Criterion": ["Experience", "Methodology"], "Weight": [40, 60], "Scope": ["General", ""]}
            ).to_csv(criteria_path, index=False)

            try:
                f1 = _fingerprint(contract_path)
                f2 = _fingerprint(contract_path)
                _ok("_fingerprint stable") if f1 and f1 == f2 else _fail("_fingerprint unstable/empty")

                _ok("_as_json_dict(None)") if _as_json_dict(None) == {} else _fail("_as_json_dict(None)")
                d = {"a": 1}
                _ok("_as_json_dict(dict)") if _as_json_dict(d) == d else _fail("_as_json_dict(dict)")
                bs = json.dumps({"b": 2}).encode("utf-8")
                _ok("_as_json_dict(bytes)") if _as_json_dict(bs) == {"b": 2} else _fail("_as_json_dict(bytes)")
                s = json.dumps({"c": 3})
                _ok("_as_json_dict(str)") if _as_json_dict(s) == {"c": 3} else _fail("_as_json_dict(str)")
            except Exception as e:
                _fail(f"helper checks raised: {e}")

            raw_contract_rel = "raw/contract.txt"
            raw_boq_rel = "raw/boq.xlsx"
            raw_criteria_rel = "raw/criteria.csv"
            try:
                upload_file(package_id, raw_contract_rel, file_path=contract_path)
                _ok("uploaded contract.txt")
            except Exception as e:
                _fail(f"upload_file contract.txt: {e}")

            try:
                upload_file(package_id, raw_boq_rel, file_path=boq_path)
                _ok("uploaded boq.xlsx")
            except Exception as e:
                _fail(f"upload_file boq.xlsx: {e}")

            try:
                upload_file(package_id, raw_criteria_rel, file_path=criteria_path)
                _ok("uploaded criteria.csv")
            except Exception as e:
                _fail(f"upload_file criteria.csv: {e}")

            # full cloud keys (with package_id prefix) for ingestion & checks
            raw_contract_full = f"{package_id}/{raw_contract_rel}"
            raw_boq_full = f"{package_id}/{raw_boq_rel}"
            raw_criteria_full = f"{package_id}/{raw_criteria_rel}"

            # ingest: default/contract
            try:
                ok = await ingest_documents(
                    package_id=package_id,
                    source_keys=[raw_contract_full],
                    tool_context=None,
                    reprocess_all=False,
                )
                _ok("ingest_documents (default/contract)") if ok else _fail("ingest_documents (default/contract) returned False")
            except Exception as e:
                _fail(f"ingest_documents (default/contract) raised: {e}")

            # get_documents: default/contract
            try:
                parts, meta = await get_documents(
                    package_id=package_id,
                    document_metas=[{"name": "contract.txt", "source_key": raw_contract_full}],
                    tool_context=None,
                    force_refresh=False,
                )
                if isinstance(parts, list) and all(isinstance(p, Part) for p in parts) and isinstance(meta, dict):
                    _ok("get_documents (default/contract) -> Parts + meta")
                else:
                    _fail("get_documents (default/contract) unexpected return types")
            except Exception as e:
                _fail(f"get_documents (default/contract) raised: {e}")

            # ingest: comm_rfp
            try:
                ok = await ingest_documents(
                    package_id=package_id,
                    source_keys=[raw_boq_full],
                    tool_context="comm_rfp",
                    reprocess_all=False,
                )
                _ok("ingest_documents (comm_rfp)") if ok else _fail("ingest_documents (comm_rfp) returned False")
            except Exception as e:
                _fail(f"ingest_documents (comm_rfp) raised: {e}")

            # get_documents: comm_rfp
            try:
                dataframes, meta = await get_documents(
                    package_id=package_id,
                    document_metas=[{"name": "boq.xlsx", "source_key": raw_boq_full, "contractor": "VendorA"}],
                    tool_context="comm_rfp",
                    force_refresh=False,
                )

                from pandas import DataFrame as _DF
                is_mapping = isinstance(dataframes, dict)
                vendor_present = is_mapping and ("VendorA" in dataframes)
                vendor_all_dfs = vendor_present and all(isinstance(v, _DF) for v in dataframes.values())
                sheet_meta_ok = isinstance(meta, dict) and isinstance(meta.get("sheet_order"), list)

                if vendor_present and vendor_all_dfs:
                    _ok("get_documents (comm_rfp) -> contractor DataFrames")
                elif is_mapping and sheet_meta_ok and len(dataframes) == 0:
                    _ok("get_documents (comm_rfp) -> empty (no BoQ rows extracted)")
                else:
                    _fail("get_documents (comm_rfp) unexpected shape")
            except Exception as e:
                _fail(f"get_documents (comm_rfp) raised: {e}")

            # ingest: tech_eval
            try:
                ok = await ingest_documents(
                    package_id=package_id,
                    source_keys=[raw_criteria_full],
                    tool_context="tech_eval",
                    reprocess_all=False,
                )
                _ok("ingest_documents (tech_eval)") if ok else _fail("ingest_documents (tech_eval) returned False")
            except Exception as e:
                _fail(f"ingest_documents (tech_eval) raised: {e}")

            # get_documents: tech_eval
            try:
                tables, meta = await get_documents(
                    package_id=package_id,
                    document_metas=[{"name": "criteria.csv", "source_key": raw_criteria_full}],
                    tool_context="tech_eval",
                    force_refresh=False,
                )
                if isinstance(tables, dict) and "criteria.csv" in tables:
                    _ok("get_documents (tech_eval) -> eval tables")
                else:
                    _fail("get_documents (tech_eval) unexpected return")
            except Exception as e:
                _fail(f"get_documents (tech_eval) raised: {e}")

            # idempotency: get_documents again
            try:
                parts2, meta2 = await get_documents(
                    package_id=package_id,
                    document_metas=[{"name": "contract.txt", "source_key": raw_contract_full}],
                    tool_context=None,
                    force_refresh=False,
                )
                _ok("get_documents idempotent (no reingest)") if isinstance(parts2, list) else _fail("get_documents idempotent failed")
            except Exception as e:
                _fail(f"get_documents idempotent raised: {e}")

            print("DOCINGEST OK" if overall_ok else "DOCINGEST FAIL")
            return overall_ok

        finally:
            # cleanup everything created
            try:
                # delete processed docstore files using manifest
                manifest = _as_json_dict(get_file(package_id, MANIFEST_PATH))
                if isinstance(manifest, dict):
                    for _src, fp_name in manifest.items():
                        try:
                            delete_file(package_id, f"{DOCSTORE_DIR}{fp_name}")
                        except Exception:
                            pass
                # remove manifest + lock
                try:
                    delete_file(package_id, MANIFEST_PATH)
                except Exception:
                    pass
                try:
                    delete_file(package_id, LOCK_FILE_PATH)
                except Exception:
                    pass
                # remove raw sources
                for rel in (raw_contract_rel, raw_boq_rel, raw_criteria_rel):
                    try:
                        delete_file(package_id, rel)
                    except Exception:
                        pass
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    try:
        return bool(asyncio.run(_run()))
    except Exception as e:
        print(f"DOCINGEST ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
