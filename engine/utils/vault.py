"""
TruBuild Vault Client
=====================
Fetches secrets from HashiCorp Vault.

Usage:
    from utils.vault import secrets

    # Get a secret (throws KeyError if not found)
    db_url = secrets.get("DATABASE_URL")

    # Force refresh from Vault
    secrets.refresh()
"""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("TruBuildBE")

# Try to import hvac, fallback gracefully if not installed
try:
    import hvac

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logger.info("hvac not installed - using environment variables only")


class VaultClient:
    """
    Vault client for fetching secrets.

    Secrets path structure: secret/trubuild/{region}/{env}
    All secrets for a region/env are stored in a single path.
    """

    def __init__(self):
        self.vault_addr = (os.getenv("VAULT_ADDR") or "").strip()
        self.region = os.getenv("TRUBUILD_REGION", "ksa")
        self.env = os.getenv("TRUBUILD_ENV", "dev")

        self._client: Optional["hvac.Client"] = None
        self._cache: Dict[str, Any] = {}

        # Track if we've attempted connection
        self._connection_attempted = False
        self._vault_available = False

    def _get_client(self) -> Optional["hvac.Client"]:
        """Lazy initialization of Vault client with authentication."""
        if not VAULT_AVAILABLE:
            return None
        if not self.vault_addr:
            return None

        if self._client is not None:
            return self._client

        if self._connection_attempted and not self._vault_available:
            return None

        self._connection_attempted = True

        try:
            self._client = hvac.Client(url=self.vault_addr)

            # Authenticate
            role_id = os.getenv("VAULT_ROLE_ID")
            secret_id = os.getenv("VAULT_SECRET_ID")
            token = os.getenv("VAULT_TOKEN")

            if role_id and secret_id:
                self._client.auth.approle.login(role_id=role_id, secret_id=secret_id)
                logger.info(
                    f"Authenticated to Vault via AppRole for {self.region}/{self.env}"
                )
                self._vault_available = True
            elif token:
                self._client.token = token
                logger.info(
                    f"Authenticated to Vault via token for {self.region}/{self.env}"
                )
                self._vault_available = True
            else:
                logger.warning("No Vault credentials configured")
                self._client = None

        except Exception as e:
            logger.warning(f"Could not connect to Vault: {e}")
            self._client = None

        return self._client

    def _secret_path(self) -> str:
        """Build the secret path: trubuild/{region}/{env}"""
        return f"trubuild/{self.region}/{self.env}"

    def _fetch_from_vault(self) -> Dict[str, Any]:
        """Fetch all secrets from Vault."""
        client = self._get_client()
        if client is None:
            return {}

        try:
            path = self._secret_path()
            response = client.secrets.kv.v2.read_secret_version(
                path=path, mount_point="secret"
            )
            secrets = response["data"]["data"]
            logger.debug(f"Fetched {len(secrets)} secrets from Vault ({path})")
            return secrets
        except Exception as e:
            logger.warning(f"Could not fetch secrets from Vault: {e}")
            return {}

    def _load_secrets(self) -> None:
        """Load secrets from Vault into cache."""
        if not self._cache:
            self._cache = self._fetch_from_vault()

    def _env_fallback(self, key: str) -> Optional[str]:
        """Try to get value from process environment (ephemeral exports)."""
        v = os.getenv(key)
        if v is not None and v != "":
            return v
        v = os.getenv(key.upper())
        if v is not None and v != "":
            return v
        v = os.getenv(key.lower())
        if v is not None and v != "":
            return v
        return None

    def get(self, key: str, default: Optional[str] = None) -> str:
        """
        Get a secret value from Vault, or from process environment if not in Vault.

        Args:
            key: Secret key (e.g., "DATABASE_URL", "POSTGRES_PASSWORD")
            default: Value to return if not found in Vault or env.

        Returns:
            Secret value

        Raises:
            KeyError: If secret not found and no default provided.
        """
        # Load from Vault if not cached
        self._load_secrets()

        # Try Vault cache (case-insensitive lookup)
        key_lower = key.lower()
        for k, v in self._cache.items():
            if k.lower() == key_lower:
                return v

        # Fall back to process environment (ephemeral exports)
        env_val = self._env_fallback(key)
        if env_val is not None:
            return env_val

        if default is not None:
            return default
        raise KeyError(f"Secret '{key}' not found (Vault or env)")

    def refresh(self) -> None:
        """Force refresh all secrets from Vault."""
        self._cache = self._fetch_from_vault()
        logger.info(f"Secrets refreshed for {self.region}/{self.env}")


# Global singleton instance
secrets = VaultClient()
