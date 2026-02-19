"""
Job Worker Process for TruBuild Engine.

This module provides a worker that polls the job queue and processes jobs.
"""

import os
import sys
import time
import signal
import asyncio
from datetime import datetime
from typing import Dict, Any, Callable, Optional

from utils.db.job_queue import (
    get_pending_jobs,
    create_job_run,
    update_job_run_status,
    save_job_artifact,
    get_latest_job_run,
    JobStatus,
    JobType,
)
from utils.db.connection import init_db, DB_TYPE
from utils.core.log import setup_logging, pid_tool_logger, set_logger, get_logger
from utils.core.warnings_config import configure_warning_filters
from utils.llm.gcp_credentials import ensure_gcp_credentials_from_vault

# Worker configuration (Vault or ephemeral env)
WORKER_POLL_INTERVAL = 5
WORKER_BATCH_SIZE = 5

# Job processor registry
_job_processors: Dict[str, Callable] = {}

def register_job_processor(job_type: str, processor: Callable):
    """
    Register a job processor function for a specific job type.

    Args:
        job_type: Job type (e.g., "tech_rfp_analysis")
        processor: Async function that takes (job_id: str, job: Dict) and returns result Dict
    """
    _job_processors[job_type] = processor
    log = get_logger()
    log.info(f"Registered processor for job type: {job_type}")


async def process_job(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single job by calling its registered processor.

    Args:
        job_id: Job UUID
        job: Job dict with payload

    Returns:
        Result dict (processors handle creating runs and saving artifacts)
    """
    log = get_logger()
    job_type = job.get("type")
    if not job_type:
        raise ValueError(f"Job {job_id} has no type")

    processor = _job_processors.get(job_type)
    if not processor:
        raise ValueError(f"No processor registered for job type: {job_type}")

    log.info(f"Processing job {job_id} of type {job_type}")

    try:
        # Call the processor (can be async or sync)
        # Processors are responsible for creating runs and saving artifacts
        if asyncio.iscoroutinefunction(processor):
            result = await processor(job_id, job)
        else:
            result = processor(job_id, job)

        return result
    except Exception as e:
        log.exception(f"Job {job_id} processing failed: {e}")
        raise


async def worker_loop(shutdown_event: asyncio.Event):
    """
    Main worker loop that polls for pending jobs and processes them.
    Exits when shutdown_event is set (e.g. by SIGINT/SIGTERM).
    """
    set_logger(pid_tool_logger("SYSTEM", "worker"))
    log = get_logger()
    log.info("Worker loop started")

    while not shutdown_event.is_set():
        try:
            # Get pending jobs (jobs with no runs or only failed runs)
            pending_jobs = get_pending_jobs(limit=WORKER_BATCH_SIZE)

            if not pending_jobs:
                # No jobs: wait in short steps so we can react to shutdown
                for _ in range(WORKER_POLL_INTERVAL):
                    if shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
                continue

            log.info(f"Found {len(pending_jobs)} pending jobs")

            # Process jobs concurrently (with limit)
            semaphore = asyncio.Semaphore(WORKER_BATCH_SIZE)

            async def process_with_semaphore(job: Dict[str, Any]):
                async with semaphore:
                    job_id = job["id"]
                    try:
                        # Process the job (processor creates run and handles everything)
                        result = await process_job(job_id, job)

                        log.info(f"Job {job_id} completed successfully")

                    except Exception as e:
                        error_msg = str(e)
                        log.error(f"Job {job_id} failed: {error_msg}")
                        # Error handling is done by the processor (updates run status)

            # Process all pending jobs
            tasks = [process_with_semaphore(job) for job in pending_jobs]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay before next poll; sleep in steps to respect shutdown
            for _ in range(1):
                if shutdown_event.is_set():
                    break
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.exception(f"Error in worker loop: {e}")
            for _ in range(WORKER_POLL_INTERVAL):
                if shutdown_event.is_set():
                    break
                await asyncio.sleep(1)

    log.info("Worker loop stopped")


def run_worker():
    """
    Run the worker as a standalone process.
    """
    setup_logging()
    configure_warning_filters()
    ensure_gcp_credentials_from_vault()
    set_logger(pid_tool_logger("SYSTEM", "worker"))
    log = get_logger()

    # Initialize database (creates tables if they don't exist)
    if DB_TYPE == "postgres":
        log.info("Initializing database...")
        init_db()

    log.info("Starting job worker process")

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        log.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run worker loop (exits when shutdown_event is set)
    try:
        asyncio.run(worker_loop(shutdown_event))
    except KeyboardInterrupt:
        log.info("Worker interrupted by user")
    except Exception as e:
        log.exception(f"Worker crashed: {e}")
        sys.exit(1)
    finally:
        # Shut down PDF process pool so spawn workers exit
        try:
            from utils.document.docingest import shutdown_pdf_pool
            shutdown_pdf_pool()
        except Exception as e:
            log.debug("shutdown_pdf_pool: %s", e)


def register_all_processors():
    """
    Register all job processors.
    Call this before starting the worker.
    """
    from tools.tech_rfp.job_processors import (
        process_tech_rfp_analysis_job,
        process_tech_rfp_summary_job,
        process_tech_rfp_evaluation_extract_job,
        process_tech_rfp_report_job,
        process_tech_rfp_generate_eval_job,
    )
    from tools.comm_rfp.job_processors import (
        process_comm_rfp_extract_job,
        process_comm_rfp_compare_job,
    )

    register_job_processor(JobType.TECH_RFP_ANALYSIS.value, process_tech_rfp_analysis_job)
    register_job_processor(JobType.TECH_RFP_SUMMARY.value, process_tech_rfp_summary_job)
    register_job_processor(JobType.TECH_RFP_EVALUATION_EXTRACT.value, process_tech_rfp_evaluation_extract_job)
    register_job_processor(JobType.TECH_RFP_REPORT.value, process_tech_rfp_report_job)
    register_job_processor(JobType.TECH_RFP_GENERATE_EVAL.value, process_tech_rfp_generate_eval_job)
    register_job_processor(JobType.COMM_RFP_EXTRACT.value, process_comm_rfp_extract_job)
    register_job_processor(JobType.COMM_RFP_COMPARE.value, process_comm_rfp_compare_job)


if __name__ == "__main__":
    # Register all processors before starting worker
    setup_logging()
    configure_warning_filters()
    set_logger(pid_tool_logger("SYSTEM", "worker"))
    register_all_processors()
    run_worker()
