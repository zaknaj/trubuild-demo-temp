"""
Job-queue-based tester for TruBuild Engine.

Runs tests by creating jobs in the queue and processing them.
Requires real DB and storage for real runs:
  - DB_TYPE=postgres and Vault postgres_url
  - MinIO credentials in Vault
  - Set Application Default Credentials for Gemini/Vertex

Usage: set company_id, package_id, JOB_TYPE_CHOICE (0..5), and all_benchmarks in main(), then:
  python -m utils.sim
"""

import os
import sys
import asyncio

# Real DB required for job-queue tests
if os.environ.get("DB_TYPE", "postgres") == "mock":
    print(
        "sim.py requires real DB and MinIO for job-queue tests. "
        "Set DB_TYPE=postgres and ensure Vault has postgres_url and MinIO credentials.",
        file=sys.stderr,
    )
    sys.exit(2)

from utils.llm.gcp_credentials import ensure_gcp_credentials_from_vault

ensure_gcp_credentials_from_vault()

from utils.db.connection import init_db, DB_TYPE
from utils.db.job_queue import (
    create_job,
    get_pending_jobs,
    get_latest_job_run,
    get_job_artifacts,
    JobType,
)
from utils.core.log import setup_logging, pid_tool_logger, set_logger, get_logger

# Import all tech_rfp job processors
from tools.tech_rfp.job_processors import (
    process_tech_rfp_analysis_job,
    process_tech_rfp_summary_job,
    process_tech_rfp_evaluation_extract_job,
    process_tech_rfp_report_job,
    process_tech_rfp_generate_eval_job,
)

# Job type -> async processor (job_id, job) -> result
JOB_PROCESSORS = {
    JobType.TECH_RFP_ANALYSIS.value: process_tech_rfp_analysis_job,
    JobType.TECH_RFP_SUMMARY.value: process_tech_rfp_summary_job,
    JobType.TECH_RFP_EVALUATION_EXTRACT.value: process_tech_rfp_evaluation_extract_job,
    JobType.TECH_RFP_REPORT.value: process_tech_rfp_report_job,
    JobType.TECH_RFP_GENERATE_EVAL.value: process_tech_rfp_generate_eval_job,
}

ALL_JOB_TYPES = list(JOB_PROCESSORS.keys())

JOB_TYPE_OPTIONS = [
    [JobType.TECH_RFP_EVALUATION_EXTRACT.value],
    [JobType.TECH_RFP_GENERATE_EVAL.value],
    [JobType.TECH_RFP_SUMMARY.value],
    [JobType.TECH_RFP_ANALYSIS.value],
    [JobType.TECH_RFP_REPORT.value],
]
JOB_TYPE_CHOICE = 1  # 0=eval_extract, 1=generate, 2=summary, 3=analysis, 4=report

# Benchmark packages: package_id -> short label for logging
BENCHMARKS = {
    "1BMCarterHones": "tech-rfp",
    "2BMColain": "tech-rfp",
    "3BMDAR": "tech-rfp",
    "4BMBigMamma": "tech-rfp",
    "5BMSeven": "tech-rfp",
    "6BMQiddiya": "tech-rfp",
    "7BMOsoolDesign": "tech-rfp",
    "8BMOsoolECI": "tech-rfp",
    "9BMOsoolDQ": "tech-rfp",
    "10BMUAE": "comm-rfp",
    "11BMGOSI371": "comm-rfp",
    "12BMGOSI372": "comm-rfp",
    "13BMGOSI373": "comm-rfp",
    "14BMGOSI374": "comm-rfp",
    "15BMGOSI375": "comm-rfp",
    "16BMGOSI374": "comm-rfp",
    "17BMGOSI373": "comm-rfp",
    "18BMGOSI371": "comm-rfp",
    "19BMGOSI372": "comm-rfp",
    "20BMGOSI375": "comm-rfp",
    "21BMOsoolSwimmingPool": "comm-rfp",
    "22BMOsoolCivilDefense": "comm-rfp",
    "23BMOsoolCivilDefense": "comm-rfp",
    "24BMHeathrow": "review",
}


def build_tooldata(endpoint: str, method: str, analysis_type: str | None = None) -> dict:
    """
    Backward-compatible helper retained from legacy sim.py.
    """
    return {
        "endpoint": endpoint,
        "method": method,
        "analysisType": analysis_type,
    }


def build_request_payload(
    endpoint: str,
    method: str,
    analysis_type: str | None = None,
    project_id: str | None = None,
    compute_reanalysis: bool | None = None,
) -> dict:
    """
    Backward-compatible payload builder retained for older local scripts.
    """
    payload = build_tooldata(endpoint=endpoint, method=method, analysis_type=analysis_type)
    if project_id is not None:
        payload["packageId"] = project_id
    if compute_reanalysis is not None:
        payload["compute_reanalysis"] = compute_reanalysis
    return payload


def _legacy_endpoint_to_job_types(endpoint: str, analysis_type: str | None) -> list[str]:
    """
    Translate old endpoint/analysis_type combinations to job-queue job types.
    """
    if endpoint == "tech-rfp":
        t = (analysis_type or "analysis").strip().lower()
        mapping = {
            "evaluation": JobType.TECH_RFP_EVALUATION_EXTRACT.value,
            "extract-eval": JobType.TECH_RFP_EVALUATION_EXTRACT.value,
            "generate-eval": JobType.TECH_RFP_GENERATE_EVAL.value,
            "summary": JobType.TECH_RFP_SUMMARY.value,
            "analysis": JobType.TECH_RFP_ANALYSIS.value,
            "report": JobType.TECH_RFP_REPORT.value,
        }
        jt = mapping.get(t)
        if not jt:
            raise ValueError(f"Unsupported tech-rfp analysis_type: {analysis_type}")
        return [jt]
    raise ValueError(
        f"Endpoint '{endpoint}' is not supported by job-queue sim wrapper. "
        "Use endpoint='tech-rfp'."
    )


def send_request(
    endpoint: str,
    analysis_type: str | None = None,
    method: str = "POST",
    project_id: str | None = None,
    compute_reanalysis: bool | None = None,
    company_id: str = "benchmarks",
) -> bool:
    """
    Legacy-compatible entrypoint: execute one mapped job through queue processors.
    """
    setup_logging()
    set_logger(pid_tool_logger("SYSTEM", "sim_legacy"))
    log = get_logger()
    package_id = project_id or "1BMCarterHones"
    _ = build_request_payload(
        endpoint=endpoint,
        method=method,
        analysis_type=analysis_type,
        project_id=package_id,
        compute_reanalysis=compute_reanalysis,
    )
    job_types = _legacy_endpoint_to_job_types(endpoint, analysis_type)
    return asyncio.run(
        _run(
            company_id=company_id,
            package_id=package_id,
            job_types=job_types,
            all_benchmarks=False,
            log=log,
        )
    )


def run_benchmark(
    bench_key: str,
    method: str,
    analysis_type: str | None = None,
    compute_reanalysis: bool | None = None,
    company_id: str = "benchmarks",
) -> bool:
    """
    Legacy benchmark helper retained for old scripts.
    """
    endpoint = BENCHMARKS.get(bench_key)
    if not endpoint:
        raise ValueError(f"Unknown benchmark key: {bench_key}")
    return send_request(
        endpoint=endpoint,
        analysis_type=analysis_type,
        method=method,
        project_id=bench_key,
        compute_reanalysis=compute_reanalysis,
        company_id=company_id,
    )


def bench_main():
    """
    Legacy bench runner: executes current JOB_TYPE_CHOICE on a default benchmark.
    """
    company_id = "benchmarks"
    package_id = "1BMCarterHones"
    job_types = JOB_TYPE_OPTIONS[JOB_TYPE_CHOICE]
    setup_logging()
    set_logger(pid_tool_logger("SYSTEM", "sim_bench"))
    log = get_logger()
    ok = asyncio.run(
        _run(
            company_id=company_id,
            package_id=package_id,
            job_types=job_types,
            all_benchmarks=False,
            log=log,
        )
    )
    return ok


def main_test_benchmarks():
    """
    Legacy compatibility hook: run selected job type for all benchmark packages.
    """
    company_id = "benchmarks"
    job_types = JOB_TYPE_OPTIONS[JOB_TYPE_CHOICE]
    setup_logging()
    set_logger(pid_tool_logger("SYSTEM", "sim_benchmarks"))
    log = get_logger()
    return asyncio.run(
        _run(
            company_id=company_id,
            package_id=None,
            job_types=job_types,
            all_benchmarks=True,
            log=log,
        )
    )


def main_test_all():
    """
    Legacy compatibility hook: run all tech-rfp job processors for one package.
    """
    company_id = "benchmarks"
    package_id = "1BMCarterHones"
    setup_logging()
    set_logger(pid_tool_logger("SYSTEM", "sim_all"))
    log = get_logger()
    return asyncio.run(
        _run(
            company_id=company_id,
            package_id=package_id,
            job_types=ALL_JOB_TYPES,
            all_benchmarks=False,
            log=log,
        )
    )


def _find_pending_job_for(
    company_id: str, package_id: str, job_type: str
) -> tuple[str | None, dict | None]:
    """
    Return (job_id, job) if there is already a pending job for this (company_id, package_id, job_type).
    Otherwise return (None, None). Used to avoid creating duplicate jobs when sim is run multiple times.
    """
    pending = get_pending_jobs(job_type=job_type, limit=50)
    for j in pending:
        if j.get("company_id") != company_id:
            continue
        p = j.get("payload") or {}
        if p.get("package_id") != package_id:
            continue
        return (j["id"], j)
    return (None, None)


async def run_one_job(
    company_id: str,
    package_id: str,
    job_type: str,
    log,
) -> bool:
    """
    Create one job (or reuse an existing pending one), process it via the registered processor,
    assert completed and result artifact. Returns True if the run completed and a result artifact exists.
    """
    if job_type not in JOB_PROCESSORS:
        log.error("Unknown job type: %s", job_type)
        return False

    # Reuse existing pending job for this (company, package, type) to avoid duplicate work
    existing_id, existing_job = _find_pending_job_for(company_id, package_id, job_type)
    if existing_id and existing_job:
        job_id = existing_id
        job = existing_job
        log.info(
            "Reusing existing pending job %s for package=%s type=%s (no duplicate created)",
            job_id,
            package_id,
            job_type,
        )
    else:
        payload = {
            "package_id": package_id,
            "user_name": None,
            "metadata": {},
        }
        job_id = create_job(
            job_type=job_type,
            payload=payload,
            company_id=company_id,
            user_id=None,
        )
        log.info("Created job %s type=%s package=%s", job_id, job_type, package_id)

        pending = get_pending_jobs(job_type=job_type, limit=50)
        job = next((j for j in pending if j["id"] == job_id), None)
    if not job:
        log.error("Job %s not found in pending list", job_id)
        return False

    processor = JOB_PROCESSORS[job_type]
    try:
        await processor(job_id, job)
    except Exception as e:
        log.exception("Processor failed: %s", e)
        return False

    run = get_latest_job_run(job_id)
    if not run:
        log.error("No run found for job %s", job_id)
        return False
    if run["status"] != "completed":
        log.error(
            "Run status is %s, expected completed. Error: %s",
            run["status"],
            run.get("error"),
        )
        return False

    artifacts = get_job_artifacts(run["id"])
    result_artifacts = [a for a in artifacts if a.get("artifact_type") == "result"]
    if not result_artifacts:
        log.error("No result artifact for run %s", run["id"])
        return False

    data = result_artifacts[0].get("data")
    log.info(
        "Job %s completed. Result keys: %s",
        job_id,
        list(data.keys()) if isinstance(data, dict) else type(data),
    )
    return True


async def run_package(
    company_id: str,
    package_id: str,
    job_types: list[str],
    log,
) -> dict[str, bool]:
    """Run the given job types for one package. Returns {job_type: ok}."""
    results = {}
    for jt in job_types:
        log.info("=== %s / %s ===", package_id, jt)
        results[jt] = await run_one_job(company_id, package_id, jt, log)
    return results


async def run_all_benchmarks(
    company_id: str,
    job_types: list[str],
    log,
) -> dict[str, dict[str, bool]]:
    """Run given job types for every benchmark package. Returns {package_id: {job_type: ok}}."""
    out = {}
    for package_id, label in BENCHMARKS.items():
        log.info("=== Benchmark %s (%s) ===", package_id, label)
        out[package_id] = await run_package(company_id, package_id, job_types, log)
    return out


async def _run(
    company_id: str,
    package_id: str | None,
    job_types: list[str],
    all_benchmarks: bool,
    log,
) -> bool:
    init_db()

    if DB_TYPE != "postgres":
        log.error("This test requires DB_TYPE=postgres (real database).")
        return False

    if all_benchmarks:
        results = await run_all_benchmarks(company_id, job_types, log)
        ok = all(
            ok for pkg_results in results.values() for ok in pkg_results.values()
        )
        for pkg, pkg_results in results.items():
            for jt, jt_ok in pkg_results.items():
                log.info("%s %s: %s", pkg, jt, "PASS" if jt_ok else "FAIL")
        return ok
    else:
        if not package_id:
            log.error("package_id must be set when all_benchmarks is False")
            return False
        results = await run_package(company_id, package_id, job_types, log)
        ok = all(results.values())
        for jt, jt_ok in results.items():
            log.info("%s: %s", jt, "PASS" if jt_ok else "FAIL")
        return ok


def main():
    # Set run configuration here
    company_id = "benchmarks"
    package_id = "1BMCarterHones"
    job_types = JOB_TYPE_OPTIONS[JOB_TYPE_CHOICE]
    all_benchmarks = False

    setup_logging()
    set_logger(pid_tool_logger("SYSTEM", "sim"))
    log = get_logger()

    try:
        ok = asyncio.run(
            _run(
                company_id=company_id,
                package_id=package_id,
                job_types=job_types,
                all_benchmarks=all_benchmarks,
                log=log,
            )
        )
        if ok:
            print("SIM OK")
        else:
            print("SIM FAIL", file=sys.stderr)
        sys.exit(0 if ok else 1)
    except Exception as e:
        log.exception("SIM FAIL: %s", e)
        print(f"SIM FAIL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
