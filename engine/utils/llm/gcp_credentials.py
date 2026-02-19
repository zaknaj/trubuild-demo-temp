"""
GCP / Vertex AI credentials for Gemini (Application Default Credentials).

Ensures GOOGLE_APPLICATION_CREDENTIALS is set
TODO: should be called early

Credentials file is written to the repo root
"""

import json
import os
from pathlib import Path

from utils.vault import secrets

CREDS_FILENAME = "gcp-credentials.json"


def _repo_root() -> Path:
    """
    Project/repo root, location-agnostic.
    """
    resolved = Path(__file__).resolve()
    parts = resolved.parts
    try:
        idx = parts.index("engine")
        # .../trubuild-monorepo/engine/utils/llm/... -> parent of engine
        return Path(*parts[:idx])
    except ValueError:
        # No "engine" in path (e.g. Docker: /app/utils/llm/...)
        return resolved.parent.parent.parent


def _default_creds_path() -> str:
    """Path to credentials file at repo root. Prefer GOOGLE_APPLICATION_CREDENTIALS if already set and file exists."""
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    return str(_repo_root() / CREDS_FILENAME)


def ensure_gcp_credentials_from_vault() -> None:
    """
    Set Application Default Credentials for Vertex AI from Vault or existing file.
    Writes to repo root (gcp-credentials.json)
    """
    creds_path = _default_creds_path()
    if os.path.isfile(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        if "GOOGLE_CLOUD_PROJECT" not in os.environ:
            os.environ["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0502418869"
        if "VERTEX_LOCATION" not in os.environ:
            os.environ["VERTEX_LOCATION"] = "us-central1"
        return
    try:
        raw = secrets.get("gemini-access-key", default="")
        if not (raw and str(raw).strip()):
            return
        key_data = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        creds_dir = os.path.dirname(creds_path)
        if creds_dir:
            os.makedirs(creds_dir, exist_ok=True)
        with open(creds_path, "w") as f:
            f.write(key_data)
        os.chmod(creds_path, 0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        if "GOOGLE_CLOUD_PROJECT" not in os.environ:
            os.environ["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0502418869"
        if "VERTEX_LOCATION" not in os.environ:
            os.environ["VERTEX_LOCATION"] = "us-central1"
    except KeyError:
        pass
    except Exception:
        pass
