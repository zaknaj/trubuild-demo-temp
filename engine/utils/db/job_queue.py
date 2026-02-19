"""
Job Queue Management for TruBuild Engine.

Provides functions to create, update, and retrieve jobs from the job queue.
Works with both PostgreSQL and mock database backends.
schema
- jobs: Immutable job definitions (id, type, payload JSONB, company_id, user_id, created_at)
- job_runs: Execution attempts (id, job_id, attempt_no, status, progress, error, timestamps)
- job_artifacts: Outputs from runs (id, run_id, artifact_type, data JSONB, created_at)
"""

import json
import uuid
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.db.connection import get_db_connection, DB_TYPE, _mock_db
from utils.core.log import pid_tool_logger, set_logger, get_logger


class JobStatus(str, Enum):
    """Job run status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enumeration for tech_rfp tools."""

    TECH_RFP_ANALYSIS = "tech_rfp_analysis"
    TECH_RFP_SUMMARY = "tech_rfp_summary"
    TECH_RFP_EVALUATION_EXTRACT = "tech_rfp_evaluation_extract"
    TECH_RFP_REPORT = "tech_rfp_report"
    TECH_RFP_GENERATE_EVAL = "tech_rfp_generate_eval"

    COMM_RFP_EXTRACT = "comm_rfp_extract"
    COMM_RFP_COMPARE = "comm_rfp_compare"

def _generate_id() -> str:
    """Generate a UUID."""
    return str(uuid.uuid4())


def _utcnow_naive() -> datetime:
    """
    Return UTC now as a naive datetime.

    Matches historical `datetime.utcnow()` behavior/output while avoiding its
    deprecation by deriving from an aware UTC timestamp first.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Jobs
def create_job(
    job_type: str,
    payload: Dict[str, Any],
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Create a new job in the queue.

    Args:
        job_type: Type of job (e.g., "tech_rfp_analysis")
        payload: Job parameters as dict (e.g., {"package_id": "...", "metadata": {...}})
        company_id: Company identifier (optional)
        user_id: User identifier (optional)

    Returns:
        job_id: UUID string of the created job
    """
    set_logger(pid_tool_logger(payload.get("package_id", "SYSTEM"), "job_queue"))
    log = get_logger()

    job_id = _generate_id()
    now = _utcnow_naive()

    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (id, type, payload, company_id, user_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (
                        job_id,
                        job_type,
                        json.dumps(payload),
                        company_id,
                        user_id,
                        now,
                    ),
                )
                conn.commit()
                log.debug(f"Created job {job_id} of type {job_type}")
                return job_id
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to create job: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        _mock_db["jobs"][job_id] = {
            "id": job_id,
            "type": job_type,
            "payload": payload,
            "company_id": company_id,
            "user_id": user_id,
            "created_at": now.isoformat(),
        }
        log.debug(f"Created mock job {job_id} of type {job_type}")
        return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a job by ID.

    Args:
        job_id: Job UUID

    Returns:
        Job dict or None if not found
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM jobs WHERE id = %s
                """,
                    (job_id,),
                )
                row = cur.fetchone()
                if row:
                    job = dict(row)
                    # Parse JSONB fields
                    if job.get("payload"):
                        job["payload"] = (
                            json.loads(job["payload"])
                            if isinstance(job["payload"], str)
                            else job["payload"]
                        )
                    return job
                return None
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get job {job_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        return _mock_db["jobs"].get(job_id)


def delete_job(job_id: str) -> bool:
    """
    Delete a job and its runs and artifacts (DB uses ON DELETE CASCADE).
    Returns True if a row was deleted, False if job_id not found.
    """
    log = get_logger()
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
                deleted = cur.rowcount
                conn.commit()
                if deleted:
                    log.info(f"Deleted job {job_id}")
                return deleted > 0
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to delete job {job_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        if job_id not in _mock_db["jobs"]:
            return False
        run_ids = [r["id"] for r in _mock_db["job_runs"].values() if r["job_id"] == job_id]
        for aid in [a for a, ar in _mock_db["job_artifacts"].items() if ar["run_id"] in run_ids]:
            del _mock_db["job_artifacts"][aid]
        for rid in run_ids:
            del _mock_db["job_runs"][rid]
        del _mock_db["jobs"][job_id]
        log.info(f"Deleted mock job {job_id}")
        return True


def get_jobs_by_package(
    package_id: str,
    company_id: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Get jobs for a specific project.

    Args:
        package_id: Project identifier (must be in payload)
        company_id: Company identifier (optional filter)
        job_type: Job type filter (optional)
        limit: Maximum number of jobs to return

    Returns:
        List of job dicts
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                conditions = ["payload->>'package_id' = %s"]
                params = [package_id]

                if company_id:
                    conditions.append("company_id = %s")
                    params.append(company_id)

                if job_type:
                    conditions.append("type = %s")
                    params.append(job_type)

                where_clause = " AND ".join(conditions)
                params.append(limit)

                cur.execute(
                    f"""
                    SELECT * FROM jobs
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s
                """,
                    params,
                )

                jobs = []
                for row in cur.fetchall():
                    job = dict(row)
                    # Parse JSONB fields
                    if job.get("payload"):
                        job["payload"] = (
                            json.loads(job["payload"])
                            if isinstance(job["payload"], str)
                            else job["payload"]
                        )
                    jobs.append(job)
                return jobs
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get jobs for project {package_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        jobs = []
        for job_id, job in _mock_db["jobs"].items():
            payload = job.get("payload", {})
            if payload.get("package_id") == package_id:
                if company_id and job.get("company_id") != company_id:
                    continue
                if job_type and job["type"] != job_type:
                    continue
                jobs.append(job)

        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs[:limit]


def list_jobs(
    job_type: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    List all jobs (any status), with latest run status. Newest first.

    Args:
        job_type: Filter by job type (optional)
        limit: Maximum number of jobs to return

    Returns:
        List of job dicts with latest_attempt and latest_status
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                base_query = """
                    WITH latest_runs AS (
                        SELECT DISTINCT ON (job_id)
                               job_id,
                               attempt_no as latest_attempt,
                               status as latest_status
                        FROM job_runs
                        ORDER BY job_id, attempt_no DESC, created_at DESC
                    )
                    SELECT j.*,
                           COALESCE(lr.latest_attempt, 0) as latest_attempt,
                           lr.latest_status
                    FROM jobs j
                    LEFT JOIN latest_runs lr ON lr.job_id = j.id
                """
                if job_type:
                    cur.execute(
                        base_query
                        + """
                        WHERE j.type = %s
                        ORDER BY j.created_at DESC
                        LIMIT %s
                        """,
                        (job_type, limit),
                    )
                else:
                    cur.execute(
                        base_query
                        + """
                        ORDER BY j.created_at DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                jobs = []
                for row in cur.fetchall():
                    job = dict(row)
                    if job.get("payload"):
                        job["payload"] = (
                            json.loads(job["payload"])
                            if isinstance(job["payload"], str)
                            else job["payload"]
                        )
                    jobs.append(job)
                return jobs
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to list jobs: {e}")
            raise
        finally:
            conn.close()
    else:
        jobs = []
        for job_id, job in _mock_db["jobs"].items():
            if job_type and job.get("type") != job_type:
                continue
            job_copy = dict(job)
            runs = [r for r in _mock_db["job_runs"].values() if r["job_id"] == job_id]
            if not runs:
                job_copy["latest_attempt"] = 0
                job_copy["latest_status"] = None
            else:
                latest = max(runs, key=lambda r: (r.get("attempt_no", 0), r.get("created_at", "")))
                job_copy["latest_attempt"] = latest.get("attempt_no", 0)
                job_copy["latest_status"] = latest.get("status")
            jobs.append(job_copy)
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs[:limit]


# Job Runs (Execution Attempts)
def create_job_run(job_id: str, attempt_no: int = 1) -> str:
    """
    Create a new run for a job.

    Args:
        job_id: Job UUID
        attempt_no: Attempt number (default 1, increment for retries)

    Returns:
        run_id: UUID string of the created run
    """
    log = get_logger()
    run_id = _generate_id()
    now = _utcnow_naive()

    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO job_runs (id, job_id, attempt_no, status, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """,
                    (run_id, job_id, attempt_no, JobStatus.PENDING.value, now),
                )
                conn.commit()
                log.debug(
                    f"Created run {run_id} for job {job_id} (attempt {attempt_no})"
                )
                return run_id
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to create run: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        _mock_db["job_runs"][run_id] = {
            "id": run_id,
            "job_id": job_id,
            "attempt_no": attempt_no,
            "status": JobStatus.PENDING.value,
            "progress": {},
            "error": None,
            "created_at": now.isoformat(),
            "started_at": None,
            "completed_at": None,
        }
        log.debug(f"Created mock run {run_id} for job {job_id}")
        return run_id


def claim_job_run(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Atomically claim a job for processing by creating a new in-progress run.

    Claiming rules:
    - No runs yet -> create attempt 1
    - Latest run failed -> create next attempt
    - Latest run pending/in_progress/completed/cancelled -> not claimable
    """
    now = _utcnow_naive()
    run_id = _generate_id()

    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Serialize claims for this job across multiple workers.
                cur.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
                if not cur.fetchone():
                    conn.rollback()
                    return None

                cur.execute(
                    """
                    SELECT attempt_no, status
                    FROM job_runs
                    WHERE job_id = %s
                    ORDER BY attempt_no DESC, created_at DESC
                    LIMIT 1
                    """,
                    (job_id,),
                )
                latest = cur.fetchone()

                if latest:
                    if latest["status"] != JobStatus.FAILED.value:
                        conn.commit()
                        return None
                    attempt_no = int(latest["attempt_no"]) + 1
                else:
                    attempt_no = 1

                cur.execute(
                    """
                    INSERT INTO job_runs (id, job_id, attempt_no, status, created_at, started_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, job_id, attempt_no, JobStatus.IN_PROGRESS.value, now, now),
                )
                conn.commit()
                return {"run_id": run_id, "attempt_no": attempt_no}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        runs = [r for r in _mock_db["job_runs"].values() if r["job_id"] == job_id]
        if runs:
            latest = max(
                runs,
                key=lambda r: (r.get("attempt_no", 0), r.get("created_at", "")),
            )
            if latest.get("status") != JobStatus.FAILED.value:
                return None
            attempt_no = int(latest.get("attempt_no", 0)) + 1
        else:
            attempt_no = 1

        _mock_db["job_runs"][run_id] = {
            "id": run_id,
            "job_id": job_id,
            "attempt_no": attempt_no,
            "status": JobStatus.IN_PROGRESS.value,
            "progress": {},
            "error": None,
            "created_at": now.isoformat(),
            "started_at": now.isoformat(),
            "completed_at": None,
        }
        return {"run_id": run_id, "attempt_no": attempt_no}


def get_job_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a run by ID.

    Args:
        run_id: Run UUID

    Returns:
        Run dict or None if not found
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM job_runs WHERE id = %s
                """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row:
                    run = dict(row)
                    # Parse JSONB fields
                    if run.get("progress"):
                        run["progress"] = (
                            json.loads(run["progress"])
                            if isinstance(run["progress"], str)
                            else run["progress"]
                        )
                    return run
                return None
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get run {run_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        return _mock_db["job_runs"].get(run_id)


def get_latest_job_run(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent run for a job.

    Args:
        job_id: Job UUID

    Returns:
        Run dict or None if no runs exist
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM job_runs
                    WHERE job_id = %s
                    ORDER BY attempt_no DESC, created_at DESC
                    LIMIT 1
                """,
                    (job_id,),
                )
                row = cur.fetchone()
                if row:
                    run = dict(row)
                    # Parse JSONB fields
                    if run.get("progress"):
                        run["progress"] = (
                            json.loads(run["progress"])
                            if isinstance(run["progress"], str)
                            else run["progress"]
                        )
                    return run
                return None
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get latest run for job {job_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        runs = [r for r in _mock_db["job_runs"].values() if r["job_id"] == job_id]
        if not runs:
            return None
        runs.sort(
            key=lambda r: (r.get("attempt_no", 0), r.get("created_at", "")),
            reverse=True,
        )
        return runs[0]


def get_job_runs(job_id: str) -> List[Dict[str, Any]]:
    """
    Get all runs for a job, ordered by attempt number.

    Args:
        job_id: Job UUID

    Returns:
        List of run dicts, ordered by attempt_no (ascending)
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM job_runs
                    WHERE job_id = %s
                    ORDER BY attempt_no ASC, created_at ASC
                """,
                    (job_id,),
                )

                runs = []
                for row in cur.fetchall():
                    run = dict(row)
                    # Parse JSONB fields
                    if run.get("progress"):
                        run["progress"] = (
                            json.loads(run["progress"])
                            if isinstance(run["progress"], str)
                            else run["progress"]
                        )
                    runs.append(run)
                return runs
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get runs for job {job_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        runs = [r for r in _mock_db["job_runs"].values() if r["job_id"] == job_id]
        runs.sort(key=lambda r: (r.get("attempt_no", 0), r.get("created_at", "")))
        return runs


def update_job_run_status(
    run_id: str,
    status: JobStatus,
    error: Optional[str] = None,
) -> bool:
    """
    Update run status.

    Args:
        run_id: Run UUID
        status: New status
        error: Error message if status is FAILED

    Returns:
        True if updated, False if run not found
    """
    now = _utcnow_naive()
    update_fields = ["status"]
    update_values = [status.value]

    if status == JobStatus.IN_PROGRESS:
        update_fields.append("started_at")
        update_values.append(now)
    elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
        update_fields.append("completed_at")
        update_values.append(now)

    if error:
        update_fields.append("error")
        update_values.append(error)

    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                set_clause = ", ".join([f"{field} = %s" for field in update_fields])
                update_values.append(run_id)
                cur.execute(
                    f"""
                    UPDATE job_runs SET {set_clause}
                    WHERE id = %s
                """,
                    update_values,
                )
                conn.commit()
                updated = cur.rowcount > 0
                if updated:
                    log = get_logger()
                    log.debug(f"Updated run {run_id} status to {status.value}")
                return updated
        except Exception as e:
            conn.rollback()
            log = get_logger()
            log.error(f"Failed to update run {run_id} status: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        if run_id not in _mock_db["job_runs"]:
            return False
        run = _mock_db["job_runs"][run_id]
        run["status"] = status.value
        if status == JobStatus.IN_PROGRESS:
            run["started_at"] = now.isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            run["completed_at"] = now.isoformat()
        if error:
            run["error"] = error
        return True


def update_job_run_progress(run_id: str, progress: Dict[str, Any]) -> bool:
    """
    Update run progress.

    Args:
        run_id: Run UUID
        progress: Progress dict (e.g., {"stage": "processing", "percent": 50})

    Returns:
        True if updated, False if run not found
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE job_runs
                    SET progress = %s
                    WHERE id = %s
                """,
                    (json.dumps(progress), run_id),
                )
                conn.commit()
                updated = cur.rowcount > 0
                return updated
        except Exception as e:
            conn.rollback()
            log = get_logger()
            log.error(f"Failed to update run {run_id} progress: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        if run_id not in _mock_db["job_runs"]:
            return False
        _mock_db["job_runs"][run_id]["progress"] = progress
        return True


# Job Artifacts (Outputs)
def save_job_artifact(
    run_id: str,
    artifact_type: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save an artifact for a run.

    Args:
        run_id: Run UUID
        artifact_type: Type of artifact (e.g., "result", "report", "log")
        data: Artifact data as dict (optional, can be None)

    Returns:
        artifact_id: UUID string of the created artifact
    """
    log = get_logger()
    artifact_id = _generate_id()
    now = _utcnow_naive()

    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO job_artifacts (id, run_id, artifact_type, data, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """,
                    (
                        artifact_id,
                        run_id,
                        artifact_type,
                        json.dumps(data) if data else None,
                        now,
                    ),
                )
                conn.commit()
                log.debug(
                    f"Saved artifact {artifact_id} of type {artifact_type} for run {run_id}"
                )
                return artifact_id
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to save artifact: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        _mock_db["job_artifacts"][artifact_id] = {
            "id": artifact_id,
            "run_id": run_id,
            "artifact_type": artifact_type,
            "data": data,
            "created_at": now.isoformat(),
        }
        log.debug(f"Saved mock artifact {artifact_id} for run {run_id}")
        return artifact_id


def get_job_artifacts(
    run_id: str,
    artifact_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get artifacts for a run.

    Args:
        run_id: Run UUID
        artifact_type: Filter by artifact type (optional)

    Returns:
        List of artifact dicts
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if artifact_type:
                    cur.execute(
                        """
                        SELECT * FROM job_artifacts
                        WHERE run_id = %s AND artifact_type = %s
                        ORDER BY created_at ASC
                    """,
                        (run_id, artifact_type),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM job_artifacts
                        WHERE run_id = %s
                        ORDER BY created_at ASC
                    """,
                        (run_id,),
                    )

                artifacts = []
                for row in cur.fetchall():
                    artifact = dict(row)
                    # Parse JSONB fields
                    if artifact.get("data"):
                        artifact["data"] = (
                            json.loads(artifact["data"])
                            if isinstance(artifact["data"], str)
                            else artifact["data"]
                        )
                    artifacts.append(artifact)
                return artifacts
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get artifacts for run {run_id}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        artifacts = []
        for artifact_id, artifact in _mock_db["job_artifacts"].items():
            if artifact["run_id"] == run_id:
                if artifact_type is None or artifact["artifact_type"] == artifact_type:
                    artifacts.append(artifact)

        artifacts.sort(key=lambda a: a.get("created_at", ""))
        return artifacts


# Worker Helpers
def get_pending_jobs(
    job_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get jobs that need processing (jobs with no runs or only failed runs).

    Args:
        job_type: Filter by job type (optional)
        limit: Maximum number of jobs to return

    Returns:
        List of job dicts with their latest run info
    """
    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                base_query = """
                    WITH latest_runs AS (
                        SELECT DISTINCT ON (job_id)
                               job_id,
                               attempt_no as latest_attempt,
                               status as latest_status
                        FROM job_runs
                        ORDER BY job_id, attempt_no DESC, created_at DESC
                    )
                    SELECT j.*,
                           COALESCE(lr.latest_attempt, 0) as latest_attempt,
                           lr.latest_status
                    FROM jobs j
                    LEFT JOIN latest_runs lr ON lr.job_id = j.id
                    WHERE (lr.latest_status IS NULL OR lr.latest_status = 'failed')
                """
                if job_type:
                    cur.execute(
                        base_query
                        + """
                        AND j.type = %s
                        ORDER BY j.created_at ASC
                        LIMIT %s
                        """,
                        (job_type, limit),
                    )
                else:
                    cur.execute(
                        base_query
                        + """
                        ORDER BY j.created_at ASC
                        LIMIT %s
                        """,
                        (limit,),
                    )

                jobs = []
                for row in cur.fetchall():
                    job = dict(row)
                    # Parse JSONB fields
                    if job.get("payload"):
                        job["payload"] = (
                            json.loads(job["payload"])
                            if isinstance(job["payload"], str)
                            else job["payload"]
                        )
                    jobs.append(job)
                return jobs
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to get pending jobs: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode
        jobs = []
        for job_id, job in _mock_db["jobs"].items():
            if job_type and job["type"] != job_type:
                continue

            # Check if job has any successful runs
            runs = [r for r in _mock_db["job_runs"].values() if r["job_id"] == job_id]
            if not runs:
                # No runs - pending
                job_copy = job.copy()
                job_copy["latest_attempt"] = 0
                job_copy["latest_status"] = None
                jobs.append(job_copy)
            else:
                # Check if all runs failed
                latest_run = max(
                    runs,
                    key=lambda r: (r.get("attempt_no", 0), r.get("created_at", "")),
                )
                if latest_run["status"] == JobStatus.FAILED.value:
                    job_copy = job.copy()
                    job_copy["latest_attempt"] = latest_run.get("attempt_no", 0)
                    job_copy["latest_status"] = latest_run["status"]
                    jobs.append(job_copy)

        jobs.sort(key=lambda j: j.get("created_at", ""))
        return jobs[:limit]
