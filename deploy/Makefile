# ==============================================================================
# TruBuild - Deployment Makefile
# ==============================================================================
# Quick commands for common operations
# ==============================================================================

.PHONY: help dev staging prod build push clean logs minio-init vault-up vault-init vault-seed ssl-init-staging ssl-init-prod ssl-expand ssl-renew ssl-status ssl-fix-permissions ssl-install-hook grafana-open grafana-logs logs-loki logs-promtail logs-grafana logging-status logging-restart

# Load environment variables from .env file (if it exists)
# -include .env

# Compose project name (defaults to current directory name).
COMPOSE_PROJECT_NAME ?= $(notdir $(CURDIR))
export

# Default target
help:
	@echo "TruBuild Deployment Commands:"
	@echo ""
	@echo "  make dev         - Start development environment"
	@echo "  make staging     - Start staging environment"
	@echo "  make prod        - Start production environment"
	@echo "  make build       - Build all Docker images"
	@echo "  make push        - Push images to registry"
	@echo "  make clean       - Remove containers and volumes"
	@echo "  make logs        - Tail all logs"
	@echo "  make logs-app    - Tail app logs"
	@echo "  make logs-engine - Tail engine logs"
	@echo "  make logs-engine-worker - Tail engine worker (job queue) logs"
	@echo "  make logs-minio  - Tail MinIO logs"
	@echo ""
	@echo "MinIO Commands:"
	@echo "  make minio-init       - Create default buckets"
	@echo "  make minio-shell      - Open MinIO client shell"
	@echo "  make minio-list       - List MinIO buckets"
	@echo "  make minio-backup-now  - Run backup immediately"
	@echo "  make minio-backup-list - List available backups"
	@echo "  make minio-restore     - Restore from latest backup"
	@echo ""
	@echo "Vault Commands:"
	@echo "  make vault-up         - Start Vault server"
	@echo "  make vault-down       - Stop Vault server"
	@echo "  make vault-clean      - Stop Vault and remove all data"
	@echo "  make vault-init       - Initialize Vault (first time only)"
	@echo "  make vault-seal       - Seal Vault (lock it)"
	@echo "  make vault-unseal     - Unseal Vault (unlock it)"
	@echo "  make vault-seed REGION=ksa ENV=dev  - Seed secrets from .env file"
	@echo "  make vault-seed-ksa   - Seed .env to all KSA environments"
	@echo "  make vault-get REGION=ksa ENV=dev  - View secrets"
	@echo "  make vault-approle REGION=ksa ENV=dev  - Get AppRole credentials"
	@echo "  make vault-ui         - Open Vault web UI"
	@echo "  make vault-status     - Check Vault status"
	@echo ""
	@echo "SSL Certificate Commands (Let's Encrypt):"
	@echo "  make ssl-init-staging - Get STAGING cert (for testing)"
	@echo "  make ssl-init-prod    - Get PRODUCTION cert (real)"
	@echo "  make ssl-expand EXTRA_DOMAINS='-d new.trubuild.io' - Add domains"
	@echo "  make ssl-renew        - Force certificate renewal"
	@echo "  make ssl-status       - Check certificate status"
	@echo "  make ssl-install-hook - Install renewal hook (if missing)"
	@echo ""
	@echo "Logging Commands (Grafana + Loki):"
	@echo "  make grafana-open       - Open Grafana UI in browser"
	@echo "  make logs-loki          - Tail Loki logs"
	@echo "  make logs-promtail      - Tail Promtail logs"
	@echo "  make logs-grafana       - Tail Grafana logs"
	@echo "  make logging-status     - Check status of logging stack"
	@echo "  make logging-restart    - Restart the entire logging stack"
	@echo ""

# ------------------------------------------------------------------------------
# Environment Commands
# ------------------------------------------------------------------------------

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

dev-d:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d

staging:
	docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d

prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

down:
	docker compose -f docker-compose.yml down

# ------------------------------------------------------------------------------
# Build Commands
# ------------------------------------------------------------------------------

build:
	docker compose -f docker-compose.yml build

build-app:
	docker build -f docker/Dockerfile.app -t trubuild-app:$${TAG:-latest} ..

build-engine:
	docker build -f docker/Dockerfile.engine -t trubuild-engine:$${TAG:-latest} ..

build-nginx:
	docker build -f docker/Dockerfile.nginx -t trubuild-nginx:$${TAG:-latest} ..

build-certbot:
	docker build -f docker/Dockerfile.certbot -t trubuild-certbot:$${TAG:-latest} ..

# ------------------------------------------------------------------------------
# Registry Commands
# ------------------------------------------------------------------------------

push:
	docker compose -f docker-compose.yml push

pull:
	docker compose -f docker-compose.yml pull

# ------------------------------------------------------------------------------
# Logs
# ------------------------------------------------------------------------------

logs:
	docker compose -f docker-compose.yml logs -f

logs-app:
	docker compose -f docker-compose.yml logs -f app

logs-engine:
	docker compose -f docker-compose.yml logs -f engine

logs-engine-worker:
	docker compose -f docker-compose.yml logs -f engine-worker

logs-nginx:
	docker compose -f docker-compose.yml logs -f nginx

logs-postgres:
	docker compose -f docker-compose.yml logs -f postgres

logs-minio:
	docker compose -f docker-compose.yml logs -f minio

# ------------------------------------------------------------------------------
# MinIO Object Storage
# ------------------------------------------------------------------------------

minio-init:
	@echo "Creating default MinIO bucket..."
	docker compose -f docker-compose.yml exec minio mc alias set local http://localhost:9000 $${MINIO_ROOT_USER:-trubuild} $${MINIO_ROOT_PASSWORD:-devpassword}
	docker compose -f docker-compose.yml exec minio mc mb --ignore-existing local/trubuild
	@echo "Bucket created successfully!"

minio-shell:
	docker compose -f docker-compose.yml exec minio mc alias set local http://localhost:9000 $${MINIO_ROOT_USER:-trubuild} $${MINIO_ROOT_PASSWORD:-devpassword}
	@echo "MinIO client ready. Use 'mc' commands, e.g.: mc ls local/"
	docker compose -f docker-compose.yml exec minio sh

minio-list:
	docker compose -f docker-compose.yml exec minio mc alias set local http://localhost:9000 $${MINIO_ROOT_USER:-trubuild} $${MINIO_ROOT_PASSWORD:-devpassword} > /dev/null 2>&1
	docker compose -f docker-compose.yml exec minio mc ls local/

minio-backup-now:
	@echo "Running MinIO backup manually..."
	docker compose -f docker-compose.yml exec minio-backup /backup.sh

minio-backup-logs:
	docker compose -f docker-compose.yml exec minio-backup cat /var/log/minio-backup.log

minio-backup-list:
	@echo "Listing available backups..."
	docker compose -f docker-compose.yml exec minio-backup ls -la /backups/

minio-restore:
	@echo "Restoring MinIO from latest backup..."
	docker compose -f docker-compose.yml exec minio-backup sh -c '\
		mc alias set minio http://minio:9000 $${MINIO_ROOT_USER} $${MINIO_ROOT_PASSWORD} && \
		mc mirror --overwrite /backups/latest/trubuild/ minio/trubuild'
	@echo "Restore complete!"

# ------------------------------------------------------------------------------
# Maintenance
# ------------------------------------------------------------------------------

clean:
	docker compose -f docker-compose.yml down -v --remove-orphans

prune:
	docker system prune -af

ps:
	docker compose -f docker-compose.yml ps

restart:
	docker compose -f docker-compose.yml restart

# ------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------

db-shell:
	docker compose -f docker-compose.yml exec postgres psql -U $${POSTGRES_USER:-trubuild} -d $${POSTGRES_DB:-trubuild}

db-backup:
	docker compose -f docker-compose.yml exec postgres pg_dump -U $${POSTGRES_USER:-trubuild} $${POSTGRES_DB:-trubuild} > backup_$$(date +%Y%m%d_%H%M%S).sql

# ------------------------------------------------------------------------------
# SSL Certificates (Let's Encrypt via Certbot)
# ------------------------------------------------------------------------------
# Domains: trubuild.io, www.trubuild.io, test.trubuild.io, minio.trubuild.io,
#          vault.trubuild.io, grafana.trubuild.io, postgres.trubuild.io
#
# Initial Setup: Uses --standalone (nginx stopped, one-time)
# Renewals: Uses --webroot (nginx running, automatic via certbot container)
# ------------------------------------------------------------------------------

SSL_EMAIL = ismail.alehyen@trubuild.io
SSL_DOMAINS = -d test.trubuild.io
CERTBOT_CONF_VOLUME = certbot-conf
CERTBOT_WWW_VOLUME = certbot-www

# Initialize SSL certificates (STAGING - for testing)
# Run this first to test the setup without hitting rate limits
# Note: Uses standalone mode (nginx must be stopped) for initial request only
ssl-init-staging:
	@echo "🔒 Requesting STAGING SSL certificate (for testing)..."
	@echo "   Domains: trubuild.io, www, test, minio, vault, grafana, postgres"
	@echo ""
	@echo "⚠️  Make sure all DNS A records point to this server!"
	@echo "   Stopping nginx to free port 80 (required for initial certificate only)..."
	docker compose -f docker-compose.yml -f docker-compose.prod.yml stop nginx 2>/dev/null || true
	@echo ""
	docker run -it --rm \
		-p 80:80 \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CERTBOT_WWW_VOLUME):/var/www/certbot \
		certbot/certbot certonly \
		--staging \
		--standalone \
		--preferred-challenges http \
		$(SSL_DOMAINS) \
		--email $(SSL_EMAIL) \
		--agree-tos \
		--no-eff-email
	@echo ""
	@echo "✅ STAGING certificate obtained!"
	@echo "   Note: Renewals will use webroot mode (nginx stays running)"
	@echo "   Test your setup, then run 'make ssl-init-prod' for real certificate"
	@echo "📋 Installing renewal deploy hook..."
	docker run --rm \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CURDIR)/scripts/certbot-deploy-hook.sh:/hook.sh:ro \
		alpine sh -c 'mkdir -p /etc/letsencrypt/renewal-hooks/deploy && cp /hook.sh /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh && chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh'
	@./scripts/fix-cert-permissions.sh
	@docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d nginx

# Initialize SSL certificates (PRODUCTION - real certificate)
# Only run after ssl-init-staging works successfully
# Note: Uses standalone mode (nginx must be stopped) for initial request only
ssl-init-prod:
	@echo "🔒 Requesting PRODUCTION SSL certificate..."
	@echo "   Domains: trubuild.io, www, test, minio, vault, grafana, postgres"
	@echo ""
	@echo "⚠️  This uses your real Let's Encrypt quota (5 per week per domain)!"
	@read -p "   Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "   Stopping nginx to free port 80 (required for initial certificate only)..."
	docker compose -f docker-compose.yml -f docker-compose.prod.yml stop nginx 2>/dev/null || true
	@echo ""
	docker run -it --rm \
		-p 80:80 \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CERTBOT_WWW_VOLUME):/var/www/certbot \
		certbot/certbot certonly \
		--standalone \
		--preferred-challenges http \
		$(SSL_DOMAINS) \
		--email $(SSL_EMAIL) \
		--agree-tos \
		--no-eff-email
	@echo ""
	@echo "✅ PRODUCTION certificate obtained!"
	@echo "   Note: Renewals will use webroot mode (nginx stays running)"
	@echo "📋 Installing renewal deploy hook..."
	docker run --rm \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CURDIR)/scripts/certbot-deploy-hook.sh:/hook.sh:ro \
		alpine sh -c 'mkdir -p /etc/letsencrypt/renewal-hooks/deploy && cp /hook.sh /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh && chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh'
	@./scripts/fix-cert-permissions.sh
	@docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d nginx
	@echo "🎉 SSL is now active! Visit https://trubuild.io"

# Add more domains to existing certificate
# Usage: make ssl-expand EXTRA_DOMAINS="-d newdomain.trubuild.io"
# Note: Uses webroot mode so nginx can stay running
ssl-expand:
	@test -n "$(EXTRA_DOMAINS)" || (echo "❌ EXTRA_DOMAINS required. Usage: make ssl-expand EXTRA_DOMAINS='-d new.trubuild.io'" && exit 1)
	@echo "🔒 Expanding certificate with new domains..."
	@echo "   Using webroot mode (nginx stays running)"
	docker run -it --rm \
		--network=$(COMPOSE_PROJECT_NAME)_trubuild-external \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CERTBOT_WWW_VOLUME):/var/www/certbot \
		certbot/certbot certonly \
		--webroot \
		-w /var/www/certbot \
		--expand \
		$(SSL_DOMAINS) $(EXTRA_DOMAINS) \
		--email $(SSL_EMAIL) \
		--agree-tos \
		--no-eff-email
	@echo "📋 Installing renewal deploy hook..."
	docker run --rm \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CURDIR)/scripts/certbot-deploy-hook.sh:/hook.sh:ro \
		alpine sh -c 'mkdir -p /etc/letsencrypt/renewal-hooks/deploy && cp /hook.sh /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh && chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh'
	@./scripts/fix-cert-permissions.sh
	@docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d nginx
	@echo "✅ Certificate expanded!"

# Force certificate renewal (normally automatic)
ssl-renew:
	@echo "🔄 Forcing certificate renewal..."
	docker compose -f docker-compose.yml -f docker-compose.prod.yml exec certbot \
		certbot renew --force-renewal --webroot -w /var/www/certbot
	@./scripts/fix-cert-permissions.sh
	@docker compose -f docker-compose.yml -f docker-compose.prod.yml exec nginx nginx -s reload
	@echo "✅ Certificate renewed and nginx reloaded!"

# Check certificate status
ssl-status:
	@echo "📋 Certificate Status:"
	@echo ""
	docker run --rm \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt:ro \
		certbot/certbot certificates
	@echo ""
	@echo "🔍 Certificate expiry check:"
	@echo | openssl s_client -servername trubuild.io -connect trubuild.io:443 2>/dev/null | openssl x509 -noout -dates 2>/dev/null || echo "   (Cannot connect - is nginx running with SSL?)"

# Fix certificate permissions (called automatically by other targets)
ssl-fix-permissions:
	@./scripts/fix-cert-permissions.sh

# Install deploy hook into certbot volume (called automatically by ssl-init-*)
# Run this if you already have certificates but renewals fail with "hook not found"
ssl-install-hook:
	@echo "📋 Installing renewal deploy hook..."
	docker run --rm \
		-v $(CERTBOT_CONF_VOLUME):/etc/letsencrypt \
		-v $(CURDIR)/scripts/certbot-deploy-hook.sh:/hook.sh:ro \
		alpine sh -c 'mkdir -p /etc/letsencrypt/renewal-hooks/deploy && cp /hook.sh /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh && chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh'
	@echo "✅ Deploy hook installed!"

# ------------------------------------------------------------------------------
# Vault - Secrets Management
# ------------------------------------------------------------------------------

VAULT_COMPOSE = docker compose -f vault/docker-compose.vault.yml

# Start Vault server
vault-up:
	$(VAULT_COMPOSE) up -d vault
	@echo "✅ Vault starting at http://localhost:8200"
	@echo "   Run 'make vault-init' to initialize (first time only)"

# Stop Vault
vault-down:
	$(VAULT_COMPOSE) stop vault

# Stop Vault and remove all volumes (complete cleanup)
vault-clean:
	@echo "🧹 Stopping Vault and removing all data..."
	$(VAULT_COMPOSE) down -v --remove-orphans
	@rm -f .vault-keys
	@echo "✅ Vault cleaned up completely (including .vault-keys)"

# Initialize Vault (run once after first start)
# Requires: vault CLI installed on host machine
# Keys are saved to .vault-keys (git-ignored)
vault-init:
	@echo "Waiting for Vault to be ready..."
	@sleep 3
	@./vault/scripts/init.sh

# Read Vault token from .vault-keys if not set
VAULT_TOKEN ?= $(shell grep VAULT_ROOT_TOKEN .vault-keys 2>/dev/null | sed 's/^VAULT_ROOT_TOKEN=//')
export VAULT_ADDR ?= http://localhost:8200

# Seed secrets for a region/env from .env file
# Usage: make vault-seed REGION=ksa ENV=dev
# Requires: vault CLI installed on host machine
vault-seed:
	@test -n "$(REGION)" || (echo "❌ REGION required. Usage: make vault-seed REGION=ksa ENV=dev" && exit 1)
	@test -n "$(ENV)" || (echo "❌ ENV required. Usage: make vault-seed REGION=ksa ENV=dev" && exit 1)
	@./vault/scripts/seed-secrets.sh $(REGION) $(ENV) .env

# Seed all KSA environments from .env file
vault-seed-ksa:
	@./vault/scripts/seed-secrets.sh ksa dev .env
	@./vault/scripts/seed-secrets.sh ksa staging .env
	@./vault/scripts/seed-secrets.sh ksa prod .env

# Seed all UAE environments from .env file
vault-seed-uae:
	@./vault/scripts/seed-secrets.sh uae dev .env
	@./vault/scripts/seed-secrets.sh uae staging .env
	@./vault/scripts/seed-secrets.sh uae prod .env

# View secrets for a region/env
# Usage: make vault-get REGION=ksa ENV=dev
vault-get:
	@test -n "$(REGION)" || (echo "❌ REGION required" && exit 1)
	@test -n "$(ENV)" || (echo "❌ ENV required" && exit 1)
	@VAULT_TOKEN="$(VAULT_TOKEN)" vault kv get secret/trubuild/$(REGION)/$(ENV)

# Get AppRole credentials for a region/env
# Usage: make vault-approle REGION=ksa ENV=dev
vault-approle:
	@test -n "$(REGION)" || (echo "❌ REGION required" && exit 1)
	@test -n "$(ENV)" || (echo "❌ ENV required" && exit 1)
	@echo "📋 AppRole credentials for $(REGION)-$(ENV):"
	@echo ""
	@echo "VAULT_ROLE_ID:"
	@VAULT_TOKEN="$(VAULT_TOKEN)" vault read -field=role_id auth/approle/role/$(REGION)-$(ENV)/role-id
	@echo ""
	@echo ""
	@echo "VAULT_SECRET_ID (generating new one):"
	@VAULT_TOKEN="$(VAULT_TOKEN)" vault write -field=secret_id -f auth/approle/role/$(REGION)-$(ENV)/secret-id
	@echo ""

# Open Vault UI
vault-ui:
	@echo "Opening Vault UI at http://localhost:8200"
	@xdg-open http://localhost:8200 2>/dev/null || open http://localhost:8200 2>/dev/null || echo "Open http://localhost:8200 in your browser"

# Vault shell (inside container)
vault-sh:
	$(VAULT_COMPOSE) exec vault sh

# Vault status
vault-status:
	@vault status

# Vault logs
vault-logs:
	$(VAULT_COMPOSE) logs -f vault

# Unseal Vault (if sealed after restart)
# Reads unseal key from .vault-keys
vault-unseal:
	@test -f .vault-keys || (echo "❌ .vault-keys not found. Run 'make vault-init' first" && exit 1)
	@UNSEAL_KEY=$$(grep VAULT_UNSEAL_KEY .vault-keys | sed 's/^VAULT_UNSEAL_KEY=//') && \
	vault operator unseal "$$UNSEAL_KEY"
	@echo "✅ Vault unsealed"

# Seal Vault (for security - requires unseal key to access again)
vault-seal:
	@VAULT_TOKEN="$(VAULT_TOKEN)" vault operator seal
	@echo "🔒 Vault sealed"

# ------------------------------------------------------------------------------
# Logging Stack (Grafana + Loki + Promtail)
# ------------------------------------------------------------------------------

# Open Grafana web UI
grafana-open:
	@echo "Opening Grafana UI..."
	@xdg-open http://localhost:3001 2>/dev/null || open http://localhost:3001 2>/dev/null || echo "Open http://localhost:3001 in your browser"

# Tail Loki logs
logs-loki:
	docker compose -f docker-compose.yml logs -f loki

# Tail Promtail logs
logs-promtail:
	docker compose -f docker-compose.yml logs -f promtail

# Tail Grafana logs
logs-grafana:
	docker compose -f docker-compose.yml logs -f grafana

# Check status of the logging stack
logging-status:
	@echo "📊 Logging Stack Status:"
	@echo ""
	@echo "=== Service Status ==="
	@docker compose -f docker-compose.yml ps loki promtail grafana 2>/dev/null || echo "  (services not running)"
	@echo ""
	@echo "=== Loki Ready ==="
	@curl -s http://localhost:3100/ready 2>/dev/null || echo "  Loki not reachable (may not be exposed in prod)"
	@echo ""
	@echo "=== Promtail Targets ==="
	@curl -s http://localhost:9080/targets 2>/dev/null | head -20 || echo "  Promtail not reachable (may not be exposed in prod)"
	@echo ""
	@echo "=== Grafana Health ==="
	@curl -s http://localhost:3001/api/health 2>/dev/null || echo "  Grafana not reachable (may not be exposed in prod)"
	@echo ""

# Restart the entire logging stack
logging-restart:
	@echo "🔄 Restarting logging stack..."
	docker compose -f docker-compose.yml restart loki promtail grafana
	@echo "✅ Logging stack restarted"
