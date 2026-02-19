# Job Queue System

This module provides a PostgreSQL-based job queue for TruBuild Engine.

## How It Works

1. **App** inserts a job into the `jobs` table
2. **Engine Worker** polls for pending jobs and processes them
3. **App** polls for job status/results by querying the database

---

## Creating & Tracking Jobs

### Database Schema

```sql
-- Jobs table (App creates jobs here)
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    company_id VARCHAR(255),
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Job runs table (Engine creates runs when processing)
CREATE TABLE job_runs (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id),
    attempt_no INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress JSONB DEFAULT '{}',
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(job_id, attempt_no)
);

-- Job artifacts table (Engine stores results here)
CREATE TABLE job_artifacts (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES job_runs(id),
    artifact_type TEXT NOT NULL,
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Valid Job Types

| Type | Description |
|------|-------------|
| `tech_rfp_analysis` | Full technical RFP evaluation |
| `tech_rfp_summary` | RFP summarization |
| `tech_rfp_evaluation_extract` | Extract evaluation criteria from RFP |
| `tech_rfp_report` | Generate technical report |
| `tech_rfp_generate_eval` | Generate evaluation criteria |

### Creating a Job

```sql
INSERT INTO jobs (id, type, payload, company_id, user_id, created_at)
VALUES (
    gen_random_uuid(),
    'tech_rfp_analysis',
    '{
        "package_id": "proj-12345",
        "user_name": "John Doe",
        "metadata": {
            "country_code": "USA"
        }
    }',
    'company-xyz',
    'user-123',
    NOW()
)
RETURNING id;
```

#### Payload Structure

```json
{
  "package_id": "required-project-id",
  "user_name": "optional-user-name",
  "metadata": {
    "country_code": "USA",
    "package_name": "optional",
    "rfp_variant": "tech"
  }
}
```

- `package_id`: **Required**
- `metadata.rfp_variant`: For summary jobs, either "tech" or "comm"

### Checking Job Status

```sql
SELECT
    j.id as job_id,
    r.status,
    r.progress,
    r.error
FROM jobs j
LEFT JOIN job_runs r ON j.id = r.job_id
WHERE j.id = 'your-job-id'
ORDER BY r.attempt_no DESC
LIMIT 1;
```

#### Status Values

| Status | Description |
|--------|-------------|
| (no run) | Job is pending in queue |
| `pending` | Run created, not started |
| `in_progress` | Engine is processing |
| `completed` | Success - fetch artifacts |
| `failed` | Error - check `error` field |

#### Progress Field (during `in_progress`)

```json
{
  "currentContractor": "Contractor Name",
  "overallPercentageCompletion": 0.25,
  "totalNumberOfCriteriaToBeAnalyzed": 100
}
```

### Getting Results

```sql
SELECT a.artifact_type, a.data
FROM job_artifacts a
JOIN job_runs r ON a.run_id = r.id
WHERE r.job_id = 'your-job-id'
  AND r.status = 'completed';
```

Artifact types: `result`, `report`

### Polling Strategy

1. After creating job: poll every 2-5 seconds
2. During processing: poll based on progress
3. Backoff if no change (max 30s)

---

## For Engine: Running the Worker

### How the worker runs

The worker is a **separate long-running process**.

- **In a terminal:** run `python -m utils.db.worker`; it blocks and runs an asyncio loop that polls the DB every few seconds, processes pending jobs, then sleeps again.
- **In Docker:** `make dev` should spin it up

### Configuration

```bash
# .env
DB_TYPE=postgres
DATABASE_URL=postgresql://user:password@host:5432/dbname
WORKER_POLL_INTERVAL=5
WORKER_BATCH_SIZE=5
```

### Start Worker

```bash
python -m utils.db.worker
```

The worker:
- Auto-initializes database tables (if postgres)
- Registers all job processors
- Polls for pending jobs
- Processes jobs and saves artifacts

### Mock Mode (Development)

Set `DB_TYPE=mock` (or leave unset) for in-memory storage. No PostgreSQL required.

---

## Architecture

```
App                         Database                    Engine Worker
 │                             │                             │
 │  INSERT INTO jobs           │                             │
 │ ─────────────────────────>  │                             │
 │                             │                             │
 │                             │  ◄── polls get_pending_jobs │
 │                             │                             │
 │                             │      creates job_run        │
 │                             │  ◄───────────────────────── │
 │                             │                             │
 │  polls job_runs             │      updates progress       │
 │ ─────────────────────────>  │  ◄───────────────────────── │
 │                             │                             │
 │                             │      saves artifacts        │
 │                             │  ◄───────────────────────── │
 │                             │                             │
 │  fetches artifacts          │      marks completed        │
 │ ─────────────────────────►  │  ◄───────────────────────── │
```

