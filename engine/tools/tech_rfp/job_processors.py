"""
Job Processors for Tech RFP Tools.

These functions are called by the worker to process jobs.
They work with the job queue schema (jobs -> job_runs -> job_artifacts).
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
from utils.llm.compactor_cache import set_compactor_context


async def process_tech_rfp_analysis_job(
    job_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a tech_rfp_analysis job.

    Args:
        job_id: Job UUID
        job: Job dict with payload

    Returns:
        Result dict (artifacts are saved separately)
    """
    from tools.tech_rfp.tech_rfp import _do_analysis_workflow

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    # Setup logging
    worker_logger = pid_tool_logger(package_id, "tech_rfp_analysis")
    set_logger(
        worker_logger,
        tool_name="tech_rfp_analysis_worker",
        package_id=package_id or "unknown",
        ip_address=payload.get("remote_ip", "no_ip"),
        request_type="WORKER",
    )
    log = get_logger()

    claim = claim_job_run(job_id)
    if not claim:
        latest_status = job.get("latest_status")
        log.debug(
            "Skipping job %s: already claimed or not retryable (latest_status=%s)",
            job_id,
            latest_status,
        )
        return {"skipped": True, "reason": "not_claimable"}
    run_id = claim["run_id"]
    attempt_no = claim["attempt_no"]
    log.debug("Claimed run %s for job %s (attempt %s)", run_id, job_id, attempt_no)

    try:
        set_compactor_context(package_id=package_id, company_id=company_id)

        # Update progress callback
        def progress_callback(progress: Dict[str, Any]):
            update_job_run_progress(run_id, progress)

        # Call the extracted workflow
        result = await _do_analysis_workflow(
            package_id=package_id,
            company_id=company_id,
            user_name=payload.get("user_name"),
            user_id=job.get("user_id"),
            package_name=payload.get("package_name"),
            evaluation_criteria=payload.get("evaluation_criteria"),
            progress_callback=progress_callback,
        )

        # Save artifacts (result and report are separate)
        if "result" in result:
            save_job_artifact(run_id, "result", result["result"])
        if "report" in result:
            save_job_artifact(run_id, "report", result["report"])

        # Mark run as completed
        update_job_run_status(run_id, JobStatus.COMPLETED)

        log.debug(f"Run {run_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        log.exception(f"Run {run_id} failed: {e}")
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise


async def process_tech_rfp_summary_job(
    job_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a tech_rfp_summary job.

    Args:
        job_id: Job UUID
        job: Job dict with payload

    Returns:
        Summary dict
    """
    from tools.tech_rfp.rfp_summarizer import _do_summary_workflow

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    worker_logger = pid_tool_logger(package_id, "rfp_summary")
    set_logger(
        worker_logger,
        tool_name="rfp_summary_worker",
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

        result = await _do_summary_workflow(
            package_id=package_id,
            company_id=company_id,
            rfp_variant=payload.get("rfp_variant", "tech"),
            progress_callback=progress_callback,
        )

        # Save as artifact
        save_job_artifact(run_id, "result", result)
        update_job_run_status(run_id, JobStatus.COMPLETED)

        log.debug(f"Run {run_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        log.exception(f"Run {run_id} failed: {e}")
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise


async def process_tech_rfp_evaluation_extract_job(
    job_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a tech_rfp_evaluation_extract job.

    Args:
        job_id: Job UUID
        job: Job dict with payload

    Returns:
        Evaluation criteria dict
    """
    from tools.tech_rfp.tech_rfp_evaluation_criteria_extractor import (
        _do_eval_extract_workflow,
    )

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    worker_logger = pid_tool_logger(package_id, "eval_analysis")
    set_logger(
        worker_logger,
        tool_name="eval_analysis_worker",
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

        result = await _do_eval_extract_workflow(
            package_id=package_id,
            company_id=company_id,
            progress_callback=progress_callback,
        )

        save_job_artifact(run_id, "result", result)
        update_job_run_status(run_id, JobStatus.COMPLETED)

        log.debug(f"Run {run_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        log.exception(f"Run {run_id} failed: {e}")
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise


async def process_tech_rfp_report_job(
    job_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a tech_rfp_report job.

    Args:
        job_id: Job UUID
        job: Job dict with payload

    Returns:
        Technical report dict
    """
    from tools.tech_rfp.tech_rfp_generate_report import _do_report_workflow

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    worker_logger = pid_tool_logger(package_id, "tech_report")
    set_logger(
        worker_logger,
        tool_name="tech_report_worker",
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

        result = await _do_report_workflow(
            package_id=package_id,
            company_id=company_id,
            evaluation_criteria=payload.get("evaluation_criteria"),
            progress_callback=progress_callback,
        )

        save_job_artifact(run_id, "result", result)
        update_job_run_status(run_id, JobStatus.COMPLETED)

        log.debug(f"Run {run_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        log.exception(f"Run {run_id} failed: {e}")
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise


async def process_tech_rfp_generate_eval_job(
    job_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a tech_rfp_generate_eval job.

    Args:
        job_id: Job UUID
        job: Job dict with payload

    Returns:
        Generated evaluation criteria dict
    """
    from tools.tech_rfp.tech_rfp_generate_evaluation import _do_generate_eval_workflow

    payload = job.get("payload", {})
    package_id = payload.get("package_id")
    company_id = job.get("company_id") or payload.get("company_id")

    worker_logger = pid_tool_logger(package_id, "generate_eval")
    set_logger(
        worker_logger,
        tool_name="generate_eval_worker",
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

        result = await _do_generate_eval_workflow(
            package_id=package_id,
            company_id=company_id,
            progress_callback=progress_callback,
        )

        save_job_artifact(run_id, "result", result)
        update_job_run_status(run_id, JobStatus.COMPLETED)

        log.debug(f"Run {run_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        log.exception(f"Run {run_id} failed: {e}")
        update_job_run_status(run_id, JobStatus.FAILED, error=error_msg)
        raise
