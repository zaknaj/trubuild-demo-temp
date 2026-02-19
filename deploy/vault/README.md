# TruBuild Vault - Secrets Management

HashiCorp Vault integration for managing secrets across regions and environments.

## Quick Start

```bash
cd deploy

# 1. Start Vault
make vault-up

# 2. Initialize (first time only)
make vault-init

# 3. Seed secrets for KSA dev
make vault-seed REGION=ksa ENV=dev

# 4. Get AppRole credentials for your services
make vault-approle REGION=ksa ENV=dev

# 5. Add credentials to your .env file
```

## Architecture

Single Vault instance serving all regions and environments:

```
secret/trubuild/
├── ksa/
│   ├── dev      ← All secrets for KSA development
│   ├── staging  ← All secrets for KSA staging
│   └── prod     ← All secrets for KSA production
├── uae/
│   ├── dev
│   ├── staging
│   └── prod
├── eu/
│   └── ...
└── us/
    └── ...
```

## Secrets Stored

Each region/env path contains all secrets in a flat structure:

| Key | Description |
|-----|-------------|
| `database_url` | PostgreSQL connection string |
| `postgres_user` | PostgreSQL username |
| `postgres_password` | PostgreSQL password |
| `better_auth_secret` | Auth secret key |
| `minio_root_user` | MinIO access key |
| `minio_root_password` | MinIO secret key |
| `alibaba_access_key_id` | Alibaba Cloud key |
| `alibaba_access_key_secret` | Alibaba Cloud secret |
| `aws_access_key_id` | AWS access key |
| `aws_secret_access_key` | AWS secret key |
| `google_api_key` | Google API key |
| `sendgrid_api_key` | SendGrid API key |
| `slack_token` | Slack token |

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make vault-up` | Start Vault server |
| `make vault-init` | Initialize Vault (first time) |
| `make vault-seed REGION=x ENV=y` | Seed secrets for region/env |
| `make vault-seed-ksa` | Seed all KSA environments |
| `make vault-get REGION=x ENV=y` | View secrets |
| `make vault-approle REGION=x ENV=y` | Get AppRole credentials |
| `make vault-ui` | Open Vault web UI |
| `make vault-status` | Check Vault status |
| `make vault-unseal` | Unseal Vault (after restart) |

## Application Integration

### Python (Engine)

```python
from utils.vault import secrets

# Get a secret
db_url = secrets.get("database_url")
api_key = secrets.get("google_api_key", default="")

# Use convenience properties
db_url = secrets.database_url
minio = secrets.minio_credentials

# Force refresh
secrets.refresh()
```

### TypeScript (App)

```typescript
import { secrets, initializeSecrets } from "@/lib/vault";

// Initialize at startup
await initializeSecrets();

// Use synchronously after init
const dbUrl = secrets.DATABASE_URL;
const authSecret = secrets.BETTER_AUTH_SECRET;
```

## Environment Variables

Add to your `.env`:

```bash
VAULT_ADDR=http://vault:8200
TRUBUILD_REGION=ksa
TRUBUILD_ENV=dev
VAULT_ROLE_ID=<from make vault-approle>
VAULT_SECRET_ID=<from make vault-approle>
VAULT_CACHE_TTL=300
```

Run `make env-from-vault` to pull Vault secrets into `.env` for Docker Compose. The engine can also load directly from Vault at runtime (no need to put secrets in `.env` for the engine).

## Secret Refresh

Secrets are cached with a TTL (default 5 minutes). To force refresh:

1. **Automatic**: Wait for TTL to expire
2. **Manual (Python)**: `secrets.refresh()`
3. **Manual (TypeScript)**: `await refreshSecrets()`
4. **Service restart**: `docker compose restart engine app`

## Security Notes

1. **init-keys.txt**: After initialization, backup and delete `/vault/data/init-keys.txt`
2. **AppRole credentials**: Store securely, rotate periodically
3. **Network**: Vault runs on internal Docker network only
4. **TLS**: Enable TLS for production (edit `vault.hcl`)
