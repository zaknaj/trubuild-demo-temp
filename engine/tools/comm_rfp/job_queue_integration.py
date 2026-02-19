"""
Job Queue Integration for Commercial RFP tools.
"""

from typing import Dict, Any, Optional

from utils.db.job_queue import (
    create_job,
    get_latest_job_run,
    get_job_artifacts,
    get_jobs_by_package,
    JobStatus,
    JobType,
)
from utils.core.log import pid_tool_logger, set_logger, get_logger


def create_comm_rfp_job(
    job_type: JobType,
    package_id: str,
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    payload_fields: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
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
    log.debug("Created %s job %s for package %s", job_type.value, job_id, package_id)
    return job_id


def get_latest_completed_artifact(
    *,
    package_id: str,
    company_id: Optional[str],
    job_type: JobType,
    artifact_type: str = "result",
) -> Optional[Dict[str, Any]]:
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
    jobs = get_jobs_by_package(
        package_id=package_id,
        company_id=company_id,
        job_type=job_type.value if job_type else None,
        limit=1,
    )

    if not jobs:
        return {"status": "pending", "message": "No job found"}

    job = jobs[0]
    job_id = job["id"]
    run = get_latest_job_run(job_id)

    if not run:
        return {"status": "in progress", "message": "Job is pending", "job_id": job_id}

    run_status = run.get("status")

    if run_status == JobStatus.COMPLETED.value:
        artifacts = get_job_artifacts(run["id"])

        data = {}
        for artifact in artifacts:
            artifact_type = artifact.get("artifact_type")
            artifact_data = artifact.get("data")
            data[artifact_type] = artifact_data

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

    if run_status == JobStatus.FAILED.value:
        return {
            "status": "error",
            "error": run.get("error", "Job failed"),
            "job_id": job_id,
            "run_id": run["id"],
        }

    if run_status == JobStatus.IN_PROGRESS.value:
        return {
            "status": "in progress",
            "progress": run.get("progress", {}),
            "job_id": job_id,
            "run_id": run["id"],
        }

    if run_status == JobStatus.PENDING.value:
        return {
            "status": "in progress",
            "message": "Job is pending",
            "job_id": job_id,
            "run_id": run["id"],
        }

    return {
        "status": "unknown",
        "message": f"Unknown run status: {run_status}",
        "job_id": job_id,
        "run_id": run["id"],
    }
