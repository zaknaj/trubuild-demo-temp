"""
Commercial RFP queue-facing orchestration.

This module mirrors the tech_rfp job-queue pattern:
- GET: poll job status
- POST: enqueue extract/compare jobs
- worker processors call workflow functions defined here
"""

from __future__ import annotations

import os
import asyncio
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from utils.core.log import pid_tool_logger, set_logger, get_logger
from utils.db.job_queue import JobType
from utils.storage.bucket import list_files, list_subdirectories, download_single_file
from tools.comm_rfp.job_queue_integration import (
    create_comm_rfp_job,
    get_job_status_response,
)
from tools.comm_rfp.comm_rfp_parse_excel import ExcelParser
from tools.comm_rfp.comm_rfp_extract import extract_boq_parallel, build_reference_items_from_boq
from tools.comm_rfp.comm_rfp_compare import _compare_boqs


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _emit_progress(progress_callback: ProgressCallback, **payload: Any) -> None:
    if progress_callback:
        progress_callback(payload)


def _safe_slack_sub(logger, wact, text: str) -> None:
    if not wact:
        return
    try:
        wact.sub(text)
    except Exception:
        logger.debug("Slack sub failed (non-fatal).", exc_info=True)


def _safe_slack_done(logger, wact) -> None:
    if not wact:
        return
    try:
        wact.done()
    except Exception:
        logger.debug("Slack done failed (non-fatal).", exc_info=True)


def _safe_slack_error(logger, wact, text: str) -> None:
    if not wact:
        return
    try:
        wact.error(text)
    except Exception:
        logger.debug("Slack error failed (non-fatal).", exc_info=True)


def _is_excel_key(key: str) -> bool:
    lower = key.lower()
    return lower.endswith(".xlsx") or lower.endswith(".xls")


def _filename_from_key(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def _template_sort_key(key: str) -> tuple[int, str]:
    name = _filename_from_key(key).lower()
    if "template" in name:
        return (0, name)
    if "boq" in name:
        return (1, name)
    return (2, name)


def _pick_template_key(keys: list[str]) -> str:
    if not keys:
        raise ValueError("No candidate template files found")
    return sorted(keys, key=_template_sort_key)[0]


def _pick_optional_pte_key(keys: list[str]) -> Optional[str]:
    for key in keys:
        name = _filename_from_key(key).lower()
        if "pte" in name:
            return key
    return None


def _download_to_tempfile(*, package_id: str, key: str) -> str:
    suffix = os.path.splitext(key)[1] or ".xlsx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        local_path = tmp.name
    download_single_file(package_id=package_id, cloud_key=key, local_path=local_path)
    return local_path


def _sheet_to_text(sheet) -> str:
    lines = [
        "# Columns: ITEM_ID | DESCRIPTION | QTY | UNIT | RATE | AMOUNT",
        "# Note: AMOUNT should equal QTY x RATE",
        "",
    ]
    for row in sheet.rows:
        values = [str(v) if v is not None else "" for v in row.raw_values[:6]]
        row_text = " | ".join(v for v in values if v.strip())
        if row_text.strip():
            lines.append(row_text)
    return "\n".join(lines)


def _parse_excel_to_sheets(file_path: str) -> list[tuple[str, str]]:
    parser = ExcelParser(file_path)
    result = parser.extract()
    sheets_data = []
    for sheet in result.sheets:
        text = _sheet_to_text(sheet)
        if text.strip():
            sheets_data.append((sheet.name, text))
    return sheets_data


def _extract_boq_from_sheets(
    sheets_data: list[tuple[str, str]],
    reference_items_by_division: dict[str, list[dict]] | None = None,
) -> dict:
    if not sheets_data:
        raise ValueError("No sheets data provided")

    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    provider = os.environ.get("LLM_PROVIDER", "vertex").strip().lower()
    boq = extract_boq_parallel(
        sheets_data=sheets_data,
        provider=provider,
        model=model,
        verbose=True,
        chunk_threshold=28000,
        max_concurrent=10,
        reference_items_by_division=reference_items_by_division,
    )
    return boq.model_dump()


def _extract_boq_from_file(file_path: str) -> dict:
    sheets_data = _parse_excel_to_sheets(file_path)
    if not sheets_data:
        raise ValueError(f"No sheets with data found in {file_path}")
    return _extract_boq_from_sheets(sheets_data)


async def _do_extract_workflow(
    *,
    package_id: str,
    company_id: str,
    user_name: str | None = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    logger = get_logger()
    start_t = time.perf_counter()
    try:
        from utils.core.slack import SlackActivityLogger, SlackActivityMeta, fmt_dur

        wact = SlackActivityLogger(
            SlackActivityMeta(
                package_id=package_id,
                tool="COMM-RFP",
                user=user_name or "unknown",
                company=company_id,
            )
        )
        wact.start()
        _safe_slack_sub(logger, wact, "🛠️ Starting Commercial RFP extract")
    except Exception:
        wact = None
        fmt_dur = lambda s: f"{int(round(float(s)))}s"
        logger.debug("Slack logger not available, continuing without it")

    _emit_progress(
        progress_callback,
        stage="discover",
        message="Listing comm_rfp/boq files",
    )
    _safe_slack_sub(logger, wact, "📁 Discovering BOQ files in MinIO")
    boq_keys = list_files(package_id, "comm_rfp/boq/", company_id)
    excel_keys = [k for k in boq_keys if _is_excel_key(k)]
    if not excel_keys:
        _safe_slack_error(logger, wact, "No Excel files found under comm_rfp/boq/")
        raise ValueError("No Excel files found under comm_rfp/boq/")

    source_key = _pick_template_key(excel_keys)
    source_name = _filename_from_key(source_key)
    _emit_progress(
        progress_callback,
        stage="download",
        message=f"Downloading {source_name}",
        source_file=source_name,
    )
    _safe_slack_sub(logger, wact, f"⬇️ Downloading source file: {source_name}")

    local_path = _download_to_tempfile(package_id=package_id, key=source_key)
    try:
        _emit_progress(
            progress_callback,
            stage="extract",
            message=f"Extracting BOQ from {source_name}",
            source_file=source_name,
        )
        _safe_slack_sub(logger, wact, f"🧠 Extracting BOQ from {source_name}")
        boq_dict = await asyncio.to_thread(_extract_boq_from_file, local_path)
        _safe_slack_sub(
            logger,
            wact,
            f"🏁 Commercial extract completed in {fmt_dur(time.perf_counter() - start_t)}",
        )
        _safe_slack_done(logger, wact)
        return {
            "result": {
                "analysisType": "extract",
                "source_file": source_name,
                "source_key": source_key,
                "boq": boq_dict,
            }
        }
    except Exception as exc:
        _safe_slack_error(
            logger,
            wact,
            f"🔥 Commercial extract failed after {fmt_dur(time.perf_counter() - start_t)}: {exc}",
        )
        raise
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass


async def _do_compare_workflow(
    *,
    package_id: str,
    company_id: str,
    user_name: str | None = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    logger = get_logger()
    start_t = time.perf_counter()
    try:
        from utils.core.slack import SlackActivityLogger, SlackActivityMeta, fmt_dur

        wact = SlackActivityLogger(
            SlackActivityMeta(
                package_id=package_id,
                tool="COMM-RFP",
                user=user_name or "unknown",
                company=company_id,
            )
        )
        wact.start()
        _safe_slack_sub(logger, wact, "🛠️ Starting Commercial RFP compare")
    except Exception:
        wact = None
        fmt_dur = lambda s: f"{int(round(float(s)))}s"
        logger.debug("Slack logger not available, continuing without it")

    _emit_progress(
        progress_callback,
        stage="discover",
        message="Discovering template and vendor files in MinIO",
    )
    _safe_slack_sub(logger, wact, "📁 Discovering template/vendor files in MinIO")
    boq_keys = [k for k in list_files(package_id, "comm_rfp/boq/", company_id) if _is_excel_key(k)]
    if not boq_keys:
        _safe_slack_error(logger, wact, "No template candidates found under comm_rfp/boq/")
        raise ValueError("No Excel files found under comm_rfp/boq/ for template selection")

    template_key = _pick_template_key(boq_keys)
    pte_key = _pick_optional_pte_key([k for k in boq_keys if k != template_key])
    if not pte_key:
        rfp_excel_keys = [
            k for k in list_files(package_id, "comm_rfp/rfp/", company_id) if _is_excel_key(k)
        ]
        pte_key = _pick_optional_pte_key(rfp_excel_keys)

    vendor_dirs = list_subdirectories(package_id, "comm_rfp/tender/", company_id)
    if not vendor_dirs:
        _safe_slack_error(logger, wact, "No vendor directories found under comm_rfp/tender/")
        raise ValueError("No vendor directories found under comm_rfp/tender/")

    vendor_key_map: Dict[str, str] = {}
    for vendor in vendor_dirs:
        vendor_prefix = f"comm_rfp/tender/{vendor}/"
        vendor_excel = [
            k for k in list_files(package_id, vendor_prefix, company_id) if _is_excel_key(k)
        ]
        if not vendor_excel:
            continue
        vendor_key_map[vendor] = sorted(vendor_excel)[0]

    if not vendor_key_map:
        _safe_slack_error(
            logger, wact, "No vendor Excel files found under comm_rfp/tender/<contractor>/"
        )
        raise ValueError("No vendor Excel files found under comm_rfp/tender/<contractor>/")

    _emit_progress(
        progress_callback,
        stage="download",
        message="Downloading template, vendors, and optional PTE files",
        vendor_count=len(vendor_key_map),
    )
    _safe_slack_sub(
        logger,
        wact,
        f"⬇️ Downloading template, {len(vendor_key_map)} vendor file(s), and optional PTE",
    )

    tmp_paths: list[str] = []
    try:
        template_local = _download_to_tempfile(package_id=package_id, key=template_key)
        tmp_paths.append(template_local)

        pte_local: Optional[str] = None
        if pte_key:
            pte_local = _download_to_tempfile(package_id=package_id, key=pte_key)
            tmp_paths.append(pte_local)

        vendor_local_paths: Dict[str, str] = {}
        for vendor_name, key in vendor_key_map.items():
            local = _download_to_tempfile(package_id=package_id, key=key)
            vendor_local_paths[vendor_name] = local
            tmp_paths.append(local)

        _emit_progress(
            progress_callback,
            stage="extract",
            message="Extracting template BOQ",
            source_file=_filename_from_key(template_key),
        )
        _safe_slack_sub(logger, wact, "🧠 Extracting template BOQ")
        template_boq = await asyncio.to_thread(_extract_boq_from_file, template_local)
        reference_items = build_reference_items_from_boq(template_boq)

        pte_boq_dict = None
        if pte_local:
            _emit_progress(
                progress_callback,
                stage="extract",
                message="Extracting PTE BOQ",
                source_file=_filename_from_key(pte_key or ""),
            )
            _safe_slack_sub(logger, wact, "🧠 Extracting PTE BOQ")
            pte_sheets_data = await asyncio.to_thread(_parse_excel_to_sheets, pte_local)
            pte_boq_dict = await asyncio.to_thread(
                _extract_boq_from_sheets, pte_sheets_data, reference_items
            )

        _emit_progress(
            progress_callback,
            stage="extract",
            message=f"Extracting {len(vendor_local_paths)} vendor BOQs",
            vendor_count=len(vendor_local_paths),
        )
        _safe_slack_sub(
            logger, wact, f"🧠 Extracting {len(vendor_local_paths)} vendor BOQ(s)"
        )

        async def extract_vendor(vendor_name: str, local_path: str) -> tuple[str, Dict[str, Any]]:
            sheets_data = await asyncio.to_thread(_parse_excel_to_sheets, local_path)
            boq = await asyncio.to_thread(
                _extract_boq_from_sheets, sheets_data, reference_items
            )
            return vendor_name, boq

        vendor_results = await asyncio.gather(
            *[
                extract_vendor(name, local_path)
                for name, local_path in vendor_local_paths.items()
            ]
        )
        vendor_boqs = {name: boq for name, boq in vendor_results}

        _emit_progress(
            progress_callback,
            stage="compare",
            message="Comparing vendor BOQs against template",
            vendor_count=len(vendor_boqs),
        )
        _safe_slack_sub(
            logger, wact, f"📊 Comparing {len(vendor_boqs)} vendor BOQ(s) against template"
        )
        report = await asyncio.to_thread(
            _compare_boqs,
            template_boq,
            vendor_boqs,
            _filename_from_key(template_key),
            pte_boq=pte_boq_dict,
        )
        _safe_slack_sub(
            logger,
            wact,
            f"🏁 Commercial compare completed in {fmt_dur(time.perf_counter() - start_t)}",
        )
        _safe_slack_done(logger, wact)

        return {
            "result": {
                "analysisType": "compare",
                "template_file": _filename_from_key(template_key),
                "template_key": template_key,
                "pte_file": _filename_from_key(pte_key) if pte_key else None,
                "pte_key": pte_key,
                "vendors": {
                    vendor_name: {
                        "file": _filename_from_key(cloud_key),
                        "key": cloud_key,
                    }
                    for vendor_name, cloud_key in vendor_key_map.items()
                },
                "report": report.to_dict(),
            }
        }
    except Exception as exc:
        _safe_slack_error(
            logger,
            wact,
            f"🔥 Commercial compare failed after {fmt_dur(time.perf_counter() - start_t)}: {exc}",
        )
        raise
    finally:
        for path in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


async def comm_rfp_main(
    *,
    package_id: str | None = None,
    company_id: str | None = None,
    request_method: str | None = None,
    analysis_type: str | None = None,
    remote_ip: str | None = None,
    user_name: str | None = None,
    compute_reanalysis: bool = True,
    user_id: str | None = None,
    package_name: str | None = None,
    country_code: str = "USA",
) -> Dict[str, Any]:
    base_logger = pid_tool_logger(package_id, "comm_rfp")
    set_logger(
        base_logger,
        tool_name="comm_rfp_main",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()

    if not package_id:
        return {"status": "error", "error": "package_id is required"}
    if not company_id:
        return {"status": "error", "error": "company_id is required"}

    normalized_type = (analysis_type or "").strip().lower()
    type_map: Dict[str, JobType] = {
        "extract": JobType.COMM_RFP_EXTRACT,
        "compare": JobType.COMM_RFP_COMPARE,
    }
    job_type = type_map.get(normalized_type)
    if not job_type:
        return {
            "status": "error",
            "error": f"Unsupported analysis_type: {analysis_type}. Allowed: extract, compare",
        }

    if request_method == "GET":
        return get_job_status_response(
            package_id=package_id,
            company_id=company_id,
            job_type=job_type,
        )

    if request_method == "POST":
        if not compute_reanalysis:
            response = get_job_status_response(
                package_id=package_id,
                company_id=company_id,
                job_type=job_type,
            )
            if response.get("status") == "completed":
                return response

        job_id = create_comm_rfp_job(
            job_type=job_type,
            package_id=package_id,
            company_id=company_id,
            user_id=user_id,
            user_name=user_name,
            payload_fields={"analysis_type": normalized_type},
            remote_ip=remote_ip,
            country_code=country_code,
            package_name=package_name,
        )
        logger.info("Created %s job %s for package %s", job_type.value, job_id, package_id)
        return {
            "status": "in progress",
            "message": f"Commercial RFP {normalized_type} job created for {package_id}",
            "job_id": job_id,
        }

    return {"status": "error", "error": f"Unsupported request method: {request_method}"}
