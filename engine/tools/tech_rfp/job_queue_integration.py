"""
Job Queue Integration for Tech RFP Tools.

This module provides functions to integrate tech_rfp tools with the job queue system.
"""

from typing import Dict, Any, Optional

from utils.db.job_queue import (
    create_job,
    get_job,
    get_latest_job_run,
    get_job_artifacts,
    get_jobs_by_package,
    JobStatus,
    JobType,
)
from utils.core.log import pid_tool_logger, set_logger, get_logger


def create_tech_rfp_job(
    job_type: JobType,
    package_id: str,
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    payload_fields: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """
    Create a tech_rfp job in the queue.

    Args:
        job_type: Type of tech_rfp job
        package_id: Package identifier
        company_id: Company identifier
        user_id: User identifier
        user_name: User name
        payload_fields: Additional top-level payload fields
        **kwargs: Additional top-level payload fields (remote_ip, country_code, package_name, etc.)

    Returns:
        job_id: UUID string of the created job
    """
    # Build payload with all job parameters
    payload = {
        "package_id": package_id,
        "user_name": user_name,
    }
    payload.update(kwargs)
    if payload_fields:
        payload.update(payload_fields)

    job_id = create_job(
        job_type=job_type.value,
        payload=payload,
        company_id=company_id,
        user_id=user_id,
    )

    set_logger(pid_tool_logger(package_id or "SYSTEM", "job_queue"))
    log = get_logger()
    log.debug(f"Created {job_type.value} job {job_id} for package {package_id}")
    return job_id


def get_latest_completed_artifact(
    *,
    package_id: str,
    company_id: Optional[str],
    job_type: JobType,
    artifact_type: str = "result",
) -> Optional[Dict[str, Any]]:
    """
    Return the newest completed artifact payload for a package/job_type.

    Returns:
        {
            "job": <job dict>,
            "run": <run dict>,
            "artifact": <artifact dict>,
            "data": <artifact data>,
        }
        or None if not found.
    """
    jobs = get_jobs_by_package(
        package_id=package_id,
        company_id=company_id,
        job_type=job_type.value,
        limit=200,
    )

    for job in jobs:
        run = get_latest_job_run(job["id"])
        if not run or run.get("status") != JobStatus.COMPLETED.value:
            continue

        artifacts = get_job_artifacts(run["id"], artifact_type=artifact_type)
        if not artifacts:
            continue

        artifact = next(
            (a for a in artifacts if a.get("artifact_type") == artifact_type), None
        )
        if not artifact:
            continue

        return {
            "job": job,
            "run": run,
            "artifact": artifact,
            "data": artifact.get("data"),
        }

    return None


def get_job_status_response(
    package_id: str,
    company_id: Optional[str] = None,
    job_type: Optional[JobType] = None,
) -> Dict[str, Any]:
    """
    Get job status for a package, returning the most recent job of the specified type.

    Args:
        package_id: package identifier
        company_id: Company identifier
        job_type: Job type to filter by (if None, gets most recent of any type)

    Returns:
        Response dict with status and data
    """
    # Get the most recent job for this package
    jobs = get_jobs_by_package(
        package_id=package_id,
        company_id=company_id,
        job_type=job_type.value if job_type else None,
        limit=1,
    )

    if not jobs:
        return {"status": "pending", "message": "No job found"}

    job = jobs[0]  # Most recent
    job_id = job["id"]

    # Get latest run for this job
    run = get_latest_job_run(job_id)

    if not run:
        # No runs yet - job is pending
        return {
            "status": "in progress",
            "message": "Job is pending",
            "job_id": job_id,
        }

    run_status = run.get("status")

    if run_status == JobStatus.COMPLETED.value:
        # Get artifacts
        artifacts = get_job_artifacts(run["id"])

        # Build response with all artifacts
        data = {}
        for artifact in artifacts:
            artifact_type = artifact.get("artifact_type")
            artifact_data = artifact.get("data")
            if artifact_type == "result":
                data["result"] = artifact_data
            elif artifact_type == "report":
                data["report"] = artifact_data
            else:
                # Include other artifact types
                data[artifact_type] = artifact_data

        # If there's a single "result" artifact, use it as top-level data
        if "result" in data and len(data) == 1:
            return {
                "status": "completed",
                "data": data["result"],
                "job_id": job_id,
                "run_id": run["id"],
            }

        return {
            "status": "completed",
            "data": data,
            "job_id": job_id,
            "run_id": run["id"],
        }

    elif run_status == JobStatus.FAILED.value:
        return {
            "status": "error",
            "error": run.get("error", "Job failed"),
            "job_id": job_id,
            "run_id": run["id"],
        }

    elif run_status == JobStatus.IN_PROGRESS.value:
        progress = run.get("progress", {})
        return {
            "status": "in progress",
            "progress": progress,
            "job_id": job_id,
            "run_id": run["id"],
        }

    elif run_status == JobStatus.PENDING.value:
        return {
            "status": "in progress",
            "message": "Job is pending",
            "job_id": job_id,
            "run_id": run["id"],
        }

    else:
        return {
            "status": "unknown",
            "message": f"Unknown run status: {run_status}",
            "job_id": job_id,
            "run_id": run["id"],
        }


def get_job_by_id_response(job_id: str) -> Dict[str, Any]:
    """
    Get job status by job_id.

    Args:
        job_id: Job UUID

    Returns:
        Response dict with status and data
    """
    job = get_job(job_id)

    if not job:
        return {"status": "error", "error": "Job not found"}

    # Get latest run for this job
    run = get_latest_job_run(job_id)

    if not run:
        return {
            "status": "in progress",
            "message": "Job is pending",
            "job_id": job_id,
        }

    run_status = run.get("status")

    if run_status == JobStatus.COMPLETED.value:
        artifacts = get_job_artifacts(run["id"])

        data = {}
        for artifact in artifacts:
            artifact_type = artifact.get("artifact_type")
            artifact_data = artifact.get("data")
            if artifact_type == "result":
                data["result"] = artifact_data
            elif artifact_type == "report":
                data["report"] = artifact_data
            else:
                data[artifact_type] = artifact_data

        # If there's a single "result" artifact, use it as top-level data
        if "result" in data and len(data) == 1:
            return {
                "status": "completed",
                "data": data["result"],
                "job_id": job_id,
                "run_id": run["id"],
            }

        return {
            "status": "completed",
            "data": data,
            "job_id": job_id,
            "run_id": run["id"],
        }

    elif run_status == JobStatus.FAILED.value:
        return {
            "status": "error",
            "error": run.get("error", "Job failed"),
            "job_id": job_id,
            "run_id": run["id"],
        }

    elif run_status == JobStatus.IN_PROGRESS.value:
        return {
            "status": "in progress",
            "progress": run.get("progress", {}),
            "job_id": job_id,
            "run_id": run["id"],
        }

    elif run_status == JobStatus.PENDING.value:
        return {
            "status": "in progress",
            "message": "Job is pending",
            "job_id": job_id,
            "run_id": run["id"],
        }

    else:
        return {
            "status": "unknown",
            "message": f"Unknown run status: {run_status}",
            "job_id": job_id,
            "run_id": run["id"],
        }
