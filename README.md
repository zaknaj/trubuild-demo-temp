# TruBuild

**TruBuild is an AI-powered procurement assistant for the construction industry that helps with contract analysis and RFP evaluation.**

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
  - [Quick Start](#quick-start)
  - [Running Services Individually](#running-services-individually)
- [Secret Management with Vault](#secret-management-with-vault)
- [Logging & Monitoring](#logging--monitoring)
- [Cloud Deployment](#cloud-deployment)
- [API Documentation](#api-documentation)
- [Useful Commands](#useful-commands)

---

## Architecture Overview

TruBuild follows a microservices architecture with clear separation between the App and the AI/ML engine.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL NETWORK                                │
│                                                                             │
│    Users ──────► Nginx (Reverse Proxy / TLS) ──────► App                   │
│                           :80/:443                      :3000               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INTERNAL NETWORK                                │
│                                                                             │
│    App ◄────────────────► Engine (AI/ML engine)                           │
│    :3000                        :5000                                       │
│      │                            │                                         │
│      ▼                            ▼                                         │
│  PostgreSQL                    MinIO (S3-compatible storage)               │
│    :5432                     :9000 (API) / :9001 (Console)                 │
│                                   │                                         │
│                                   ▼                                         │
│                          MinIO Backup (Daily cron)                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           SECRETS MANAGEMENT                                 │
│                                                                             │
│                    HashiCorp Vault (Internal Network or Standalone)                            │
│                           :8200                                             │
│           secret/trubuild/{region}/{env} ─► All secrets                    │
└─────────────────────────────────────────────────────────────────────────────┘
```
### Architecture Diagram

![Architecture Diagram](./ArchitecturalDiagramv2.png)

### Services

| Service | Description | Port |
|---------|-------------|------|
| **App** | TanStack Start with React 19, handles user authentication and UI | 3000 |
| **Engine** | Python Flask API for AI/ML processing using Google Gemini | 5000 |
| **Nginx** | Reverse proxy with TLS termination | 80, 443 |
| **PostgreSQL** | Primary relational database | 5432 |
| **MinIO** | S3-compatible object storage for documents and files | 9000, 9001 |
| **Vault** | HashiCorp Vault for secrets management | 8200 |
| **Loki** | Log aggregation and storage (backed by MinIO) | 3100 |
| **Promtail** | Log collection agent (Docker + host logs) | 9080 |
| **Grafana** | Log visualization and dashboards | 3000 (3001 in dev) |

---

## Tech Stack

### App

- **Framework**: [TanStack Start](https://tanstack.com/start) (React 19 + SSR)
- **Runtime**: [Bun](https://bun.sh/)
- **Database ORM**: [Drizzle ORM](https://orm.drizzle.team/)
- **Authentication**: [Better Auth](https://www.better-auth.com/)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **UI Components**: [shadcn/ui](https://ui.shadcn.com/) + [Radix UI](https://www.radix-ui.com/)
- **State Management**: [Zustand](https://zustand-demo.pmnd.rs/) + [TanStack Query](https://tanstack.com/query)

### Engine

- **Framework**: [Flask](https://flask.palletsprojects.com/)
- **AI/ML**: [Google Gemini](https://ai.google.dev/) via `google-genai` SDK
- **Document Processing**: Custom NLP pipeline for contract and RFP analysis
- **Storage**: MinIO (S3-compatible) / GCP Buckets

### Infrastructure

- **Containerization**: Docker + Docker Compose
- **Reverse Proxy**: Nginx
- **Database**: PostgreSQL 16
- **Object Storage**: MinIO
- **Secrets Management**: HashiCorp Vault
- **Cloud Providers**:
  - **KSA Region**: GCP
  - **Other Regions**: AWS, GCP (planned)

---

## Project Structure

```
trubuild-monorepo/
├── app/                          # User Facing App
│   ├── src/
│   │   ├── auth/                 # Authentication (Better Auth)
│   │   ├── components/           # React components (shadcn/ui)
│   │   ├── db/                   # Database schema and connection
│   │   ├── fn/                   # Server functions (TanStack Start)
│   │   ├── lib/                  # Utilities and Vault client
│   │   └── routes/               # File-based routing
│   ├── package.json
│   └── vite.config.ts
│
├── engine/                       # AI/ML
│   ├── api.py                    # Flask API entry point
│   ├── tools/                    # Analysis tools (chat, RFP, contracts)
│   ├── utils/                    # Utilities (LLM, storage, encryption)
│   ├── requirements.txt
│   └── TrubuildBE.yaml           # OpenAPI specification
│
├── deploy/                       # Deployment configuration
│   ├── docker/                   # Dockerfiles
│   │   ├── Dockerfile.app
│   │   ├── Dockerfile.engine
│   │   ├── Dockerfile.nginx
│   │   └── Dockerfile.minio-backup
│   ├── docker-compose.yml        # Base compose configuration
│   ├── docker-compose.dev.yml    # Development overrides
│   ├── docker-compose.staging.yml
│   ├── docker-compose.prod.yml
│   ├── nginx/                    # Nginx configuration
│   ├── minio/                    # MinIO init and backup scripts
│   ├── loki/                     # Loki configuration
│   ├── promtail/                 # Promtail configuration
│   │   └── remote/               # Standalone Promtail for remote servers
│   ├── grafana/                  # Grafana provisioning
│   │   └── provisioning/         # Datasources and dashboard JSONs
│   ├── vault/                    # Vault configuration
│   │   ├── config/               # Vault server config
│   │   ├── policies/             # Access policies
│   │   └── scripts/              # Init and seed scripts
│   ├── env.example               # Environment variables template
│   └── Makefile                  # Deployment commands
│
└── README.md
```

---

## Prerequisites

- **Docker** 24+ and **Docker Compose** v2
- **Vault CLI** (for secrets management): [Install Guide](https://developer.hashicorp.com/vault/install)
- **Bun** (for local app development): [Install Guide](https://bun.sh/)
- **Python 3.11+** (for local engine development)

---

## Local Development

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/TruBuildAI/trubuild-monorepo.git
   cd trubuild-monorepo
   ```

2. **Set up environment variables**
   ```bash
   cd deploy
   cp env.example .env
   # Edit .env with your values (API keys, passwords, etc.)
   ```

3. **Start Vault and initialize secrets**
   ```bash
   # Start Vault
   make vault-up

   # Initialize Vault (first time only - saves keys to .vault-keys)
   make vault-init

   # Seed secrets from your .env file
   make vault-seed REGION=ksa ENV=dev
   ```

4. **Start all services**
   ```bash
   # Development mode (with hot reload)
   make dev

   # Or run in background
   make dev-d
   ```

5. **Access the application**
   - **App**: http://localhost:3000
   - **Engine API**: http://localhost:5000
   - **MinIO Console**: http://localhost:9001
   - **Grafana**: http://localhost:3001
   - **Vault UI**: http://localhost:8200

### Running Services Individually

**App**
```bash
cd app
bun install
bun dev
```

**Engine**
```bash
cd engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python api.py
```

---

## Secret Management with Vault

TruBuild uses HashiCorp Vault for centralized secrets management across all environments and regions.

### Secret Path Structure

```
secret/trubuild/{region}/{env}
```

Example paths:
- `secret/trubuild/ksa/dev` - KSA development secrets
- `secret/trubuild/ksa/staging` - KSA staging secrets
- `secret/trubuild/ksa/prod` - KSA production secrets

### Vault Commands

```bash
cd deploy

# Start/stop Vault
make vault-up          # Start Vault container
make vault-down        # Stop Vault
make vault-clean       # Stop and remove all Vault data

# Initialize and manage
make vault-init        # Initialize Vault (first time only)
make vault-unseal      # Unseal Vault after restart
make vault-seal        # Seal Vault for security
make vault-status      # Check Vault status

# Manage secrets
make vault-seed REGION=ksa ENV=dev     # Seed secrets from .env
make vault-seed-ksa                    # Seed all KSA environments
make vault-get REGION=ksa ENV=dev      # View secrets

# Authentication
make vault-approle REGION=ksa ENV=dev  # Get AppRole credentials for services
```


### Vault Credentials

After running `make vault-init`, credentials are saved to `deploy/.vault-keys`:
- `VAULT_UNSEAL_KEY` - Required to unseal Vault after restart
- `VAULT_ROOT_TOKEN` - Admin token for Vault operations

> ⚠️ **Keep `.vault-keys` secure and never commit to version control!**

---

## Logging & Monitoring

TruBuild uses **Grafana + Loki + Promtail** for centralized log aggregation. Logs are stored in MinIO with a **365-day retention policy**.

### Log Sources

| Source | Type | Labels |
|--------|------|--------|
| App, Engine, Engine Worker, Nginx, PostgreSQL, MinIO | Docker container logs | `job=docker`, `service=<name>` |
| Syslog | Host system logs | `job=syslog` |
| Auth / SSH | SSH activity & authentication | `job=ssh` |
| Kernel / Firewall | iptables, UFW, kernel events | `job=kernel` |
| Engine file logs | Application log files | `job=engine` |
| Engine process logs | Worker process logs | `job=engine-process` |

### Grafana Dashboards

Five pre-provisioned dashboards are available at `http://localhost:3001` (dev) or `https://grafana.trubuild.io` (prod):

- **Log Overview** — Aggregated log volume and errors across all sources
- **App Logs (TanStack)** — Frontend application logs and HTTP requests
- **Engine & Worker Logs** — Engine API and background worker logs
- **Security & SSH Logs** — SSH activity, failed logins, firewall events
- **Infrastructure Logs** — Nginx, PostgreSQL, MinIO, Loki, Promtail, Grafana

### Remote Log Collection

For standalone servers (Vault, DB), deploy a remote Promtail agent:

```bash
cd deploy
make remote-promtail-up REMOTE_HOST=192.168.1.100 REMOTE_USER=ubuntu
make remote-promtail-down REMOTE_HOST=192.168.1.100 REMOTE_USER=ubuntu
```

### Logging Commands

```bash
cd deploy

make logs-loki         # Tail Loki logs
make logs-promtail     # Tail Promtail logs
make logs-grafana      # Tail Grafana logs
make loki-shell        # Shell into Loki container
make grafana-shell     # Shell into Grafana container
make promtail-shell    # Shell into Promtail container
```

---

## Cloud Deployment

TruBuild is designed to be deployed on VMs in any region. Currently focused on KSA with plans to expand to UAE, EU, and US regions.

### Supported Cloud Providers

| Region | Cloud Provider | Object Storage |
|--------|----------------|----------------|
| KSA | GCP / Alibaba Cloud | MinIO / GCS |
| UAE | AWS / Azure |  MinIO / GCS |
| EU | AWS / GCP |  MinIO / GCS |
| US | AWS / GCP |  MinIO / GCS |

### VM Setup

1. **Provision a VM** with:
   - Ubuntu 24.04 LTS
   - Minimum: 4 vCPU, 8GB RAM, 200GB SSD (staging)
   - Production: 16 vCPU, 16GB RAM, 1TB SSD

2. **Configure Swap Space (Required for Engine)**

   The Engine service requires up to **50 GB of swap** for processing large documents. You must configure swap on the host before starting containers:

   ```bash
   # Create a 50GB swap file
   sudo fallocate -l 50G /swapfile
   # If fallocate doesn't work (some filesystems), use:
   # sudo dd if=/dev/zero of=/swapfile bs=1G count=50

   # Set proper permissions
   sudo chmod 600 /swapfile

   # Format as swap
   sudo mkswap /swapfile

   # Enable the swap
   sudo swapon /swapfile

   # Make it permanent (survives reboot)
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

   # Verify swap is active
   free -h
   ```

   > ⚠️ **Important**: Without adequate swap, the Engine may crash or fail when processing large documents. The Docker container is configured to use all available host swap (`memswap_limit: -1`).

3. **Install Docker and Docker Compose**
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

4. **Install Vault CLI**
   ```bash
   sudo snap install vault
   ```

5. **Clone and configure**
   ```bash
   git clone https://github.com/TruBuildAI/trubuild-monorepo.git
   cd trubuild-monorepo/deploy
   ```

6. **Initialize Vault and seed secrets**
   ```bash
   make vault-up
   make vault-init
   make vault-seed REGION=ksa ENV=prod
   ```

7. **Start production services**
   ```bash
   make prod
   ```

### DNS Configuration

Configure DNS A records pointing to your VM's IP:
- `app.trubuild.io` → Production
- `staging.trubuild.io` → Staging
- `dev.trubuild.io` → Development
- `grafana.trubuild.io` → Grafana dashboards

### TLS Certificates (WIP)

For production, configure TLS certificates in Nginx:
1. Obtain certificates (Let's Encrypt recommended)
2. Update `deploy/nginx/nginx.conf` with certificate paths
3. Rebuild Nginx: `docker compose build nginx`

---


## API Documentation

The Engine API is documented using OpenAPI 3.0 specification.

- **Specification file**: `engine/TrubuildBE.yaml`
- **API Endpoints**:
  - `d-api.trubuild.io:5000` - Development
  - `s-api.trubuild.io:5000` - Staging
  - `api.trubuild.io:5000` - Production

Main endpoints:
- `GET /ping` - Health check
- `POST /chat` - AI chat interface
- `POST /contract-review` - Contract analysis
- `POST /tech-rfp` - Technical RFP analysis
- `POST /comm-rfp` - Commercial RFP analysis

---

## Useful Commands

### Docker Compose

```bash
cd deploy

# Development
make dev              # Start with hot reload
make dev-d            # Start in background

# Staging/Production
make staging          # Start staging environment
make prod             # Start production environment

# Management
make logs             # View all logs
make clean            # Remove all containers and volumes
make build            # Build all images
make push             # Push images to registry
```

### Vault

```bash
make vault-up         # Start Vault
make vault-init       # Initialize (first time)
make vault-unseal     # Unseal after restart
make vault-seed-ksa   # Seed all KSA environments
make vault-ui         # Open Vault web UI
make vault-sh         # Shell into Vault container
```

### Logging

```bash
make logs-loki        # Tail Loki logs
make logs-promtail    # Tail Promtail logs
make logs-grafana     # Tail Grafana logs
```

### Database

```bash
cd app
bun db:push           # Push schema changes
bun db:generate       # Generate migrations
bun db:studio         # Open Drizzle Studio
```

---

