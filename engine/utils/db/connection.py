"""
Database connection management for TruBuild Engine.

Supports both PostgreSQL (via psycopg2) and a mock in-memory implementation.
Config from Vault
"""

import os
import json
from datetime import datetime
from utils.vault import secrets
from urllib.parse import urlparse, unquote
from typing import Any, Dict, Optional, Literal
from utils.core.log import pid_tool_logger, set_logger, get_logger

# Database configuration (Vault)
DB_TYPE = "postgres"
DATABASE_URL = secrets.get("postgres_url", default="") or ""

# Mock database storage (in-memory)
_mock_db: Dict[str, Any] = {
    "jobs": {},
    "job_runs": {},
    "job_artifacts": {},
}


class MockConnection:
    """Mock database connection for development/testing."""

    def __init__(self):
        self._mock_db = _mock_db

    def execute(self, query: str, params: tuple = None) -> Any:
        """Execute a query (mock implementation)."""
        # This is a placeholder - actual operations are handled by job_queue module
        return None

    def commit(self):
        """Commit transaction (no-op for mock)."""
        pass

    def close(self):
        """Close connection (no-op for mock)."""
        pass


def _parse_postgres_url(url: str) -> Dict[str, Any]:
    """
    Parse postgresql:// or postgres:// URL into connection kwargs.
    Uses component-based parsing so the password (with %, &, etc.) is not
    interpreted as part of the DSN and does not need to be percent-encoded in Vault.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc or ""
    path = (parsed.path or "").strip("/") or "postgres"

    # userinfo is "user:password" before the last @ in netloc
    at = netloc.rfind("@")
    if at >= 0:
        userinfo = netloc[:at]
        hostport = netloc[at + 1 :]
    else:
        userinfo = ""
        hostport = netloc

    user = ""
    password = ""
    if userinfo:
        colon = userinfo.find(":")
        if colon >= 0:
            user = unquote(userinfo[:colon])
            password = unquote(userinfo[colon + 1 :])
        else:
            user = unquote(userinfo)

    host = "localhost"
    port = 5432
    if hostport:
        if ":" in hostport:
            host, port_str = hostport.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = 5432
        else:
            host = hostport

    return {
        "host": host or "localhost",
        "port": port,
        "user": user,
        "password": password,
        "dbname": path,
    }


def get_db_connection():
    """
    Get a database connection based on DB_TYPE environment variable.

    Returns:
        - psycopg2 connection if DB_TYPE="postgres"
        - MockConnection if DB_TYPE="mock" (default)
    """
    if DB_TYPE == "postgres":
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            if not DATABASE_URL:
                raise ValueError(
                    "DATABASE_URL environment variable is required when DB_TYPE=postgres"
                )

            kwargs = _parse_postgres_url(DATABASE_URL)
            conn = psycopg2.connect(
                cursor_factory=RealDictCursor,
                **kwargs,
            )
            log = get_logger()
            log.info("Connected to PostgreSQL database")
            return conn
        except ImportError:
            log = get_logger()
            log.error(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )
            raise
        except Exception as e:
            log = get_logger()
            log.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    else:
        # Mock mode (default)
        log = get_logger()
        log.debug("Using mock database connection")
        return MockConnection()


def init_db():
    """
    Initialize database tables.
    For PostgreSQL: creates tables if they don't exist.
    For mock: initializes in-memory structures.
    """
    set_logger(pid_tool_logger("SYSTEM", "db_init"))
    log = get_logger()

    if DB_TYPE == "postgres":
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Create jobs table (immutable job definitions)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        type VARCHAR(100) NOT NULL,
                        payload JSONB NOT NULL DEFAULT '{}',
                        company_id VARCHAR(255),
                        user_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create job_runs table (execution attempts)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS job_runs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        attempt_no INTEGER NOT NULL DEFAULT 1,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending',
                        progress JSONB DEFAULT '{}',
                        error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        UNIQUE(job_id, attempt_no)
                    );
                """)

                # Create job_artifacts table (outputs from runs)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS job_artifacts (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        run_id UUID NOT NULL REFERENCES job_runs(id) ON DELETE CASCADE,
                        artifact_type TEXT NOT NULL,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_company_user
                    ON jobs(company_id, user_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_type
                    ON jobs(type);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                    ON jobs(created_at DESC);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_runs_job_id
                    ON job_runs(job_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_runs_status
                    ON job_runs(status) WHERE status IN ('pending', 'in_progress');
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_runs_created_at
                    ON job_runs(created_at DESC);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_artifacts_run_id
                    ON job_artifacts(run_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_artifacts_type
                    ON job_artifacts(artifact_type);
                """)

                conn.commit()
                log.info("Database tables initialized successfully")
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()
    else:
        # Mock mode - no initialization needed
        log.debug("Mock database mode - no initialization needed")
        _mock_db["jobs"] = {}
        _mock_db["job_runs"] = {}
        _mock_db["job_artifacts"] = {}


if __name__ == "__main__":
    # Test connection
    init_db()
    conn = get_db_connection()
    print(f"Database connection successful (type: {DB_TYPE})")
    if hasattr(conn, "close"):
        conn.close()
