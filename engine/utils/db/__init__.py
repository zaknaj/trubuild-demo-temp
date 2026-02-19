"""
Database utilities for TruBuild Engine.

This module provides database connection and job queue management.
Currently supports both real PostgreSQL and a mock in-memory implementation
for development/testing.
"""

from utils.db.connection import get_db_connection, init_db
from utils.db.job_queue import (
    create_job,
    delete_job,
    get_job,
    get_jobs_by_package,
    list_jobs,
    create_job_run,
    claim_job_run,
    get_job_run,
    get_latest_job_run,
    get_job_runs,
    update_job_run_status,
    update_job_run_progress,
    save_job_artifact,
    get_job_artifacts,
    get_pending_jobs,
    JobStatus,
    JobType,
)

__all__ = [
    "get_db_connection",
    "init_db",
    "create_job",
    "delete_job",
    "get_job",
    "get_jobs_by_package",
    "list_jobs",
    "create_job_run",
    "claim_job_run",
    "get_job_run",
    "get_latest_job_run",
    "get_job_runs",
    "update_job_run_status",
    "update_job_run_progress",
    "save_job_artifact",
    "get_job_artifacts",
    "get_pending_jobs",
    "JobStatus",
    "JobType",
]
