"""
Job processors for Commercial RFP tools.
"""

from typing import Dict, Any

from utils.db.job_queue import (
    claim_job_run,
    update_job_run_status,
    update_job_run_progress,
    save_job_artifact,
    JobStatus,
)
from utils.core.log import pid_tool_logger, set_logger, get_logger


async def process_comm_rfp_extract_job(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    from tools.comm_rfp.comm_rfp import _do_extract_workflow

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    worker_logger = pid_tool_logger(package_id, "comm_rfp_extract")
    set_logger(
        worker_logger,
        tool_name="comm_rfp_extract_worker",
        package_id=package_id or "unknown",
        ip_address=payload.get("remote_ip", "no_ip"),
        request_type="WORKER",
    )
    log = get_logger()

    claim = claim_job_run(job_id)
    if not claim:
        log.debug(
            "Skipping job %s: already claimed or not retryable (latest_status=%s)",
            job_id,
            job.get("latest_status"),
        )
        return {"skipped": True, "reason": "not_claimable"}
    run_id = claim["run_id"]

    try:
        def progress_callback(progress: Dict[str, Any]):
            update_job_run_progress(run_id, progress)

        result = await _do_extract_workflow(
            package_id=package_id,
            company_id=company_id,
            user_name=payload.get("user_name"),
            progress_callback=progress_callback,
        )

        save_job_artifact(run_id, "result", result.get("result", result))
        update_job_run_status(run_id, JobStatus.COMPLETED)
        log.debug("Run %s completed successfully", run_id)
        return result
    except Exception as e:
        error_msg = str(e)
        log.exception("Run %s failed: %s", run_id, e)
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise


async def process_comm_rfp_compare_job(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    from tools.comm_rfp.comm_rfp import _do_compare_workflow

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    worker_logger = pid_tool_logger(package_id, "comm_rfp_compare")
    set_logger(
        worker_logger,
        tool_name="comm_rfp_compare_worker",
        package_id=package_id or "unknown",
        ip_address=payload.get("remote_ip", "no_ip"),
        request_type="WORKER",
    )
    log = get_logger()

    claim = claim_job_run(job_id)
    if not claim:
        log.debug(
            "Skipping job %s: already claimed or not retryable (latest_status=%s)",
            job_id,
            job.get("latest_status"),
        )
        return {"skipped": True, "reason": "not_claimable"}
    run_id = claim["run_id"]

    try:
        def progress_callback(progress: Dict[str, Any]):
            update_job_run_progress(run_id, progress)

        result = await _do_compare_workflow(
            package_id=package_id,
            company_id=company_id,
            user_name=payload.get("user_name"),
            progress_callback=progress_callback,
        )

        save_job_artifact(run_id, "result", result.get("result", result))
        update_job_run_status(run_id, JobStatus.COMPLETED)
        log.debug("Run %s completed successfully", run_id)
        return result
    except Exception as e:
        error_msg = str(e)
        log.exception("Run %s failed: %s", run_id, e)
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise
