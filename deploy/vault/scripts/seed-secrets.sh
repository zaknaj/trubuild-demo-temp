#!/bin/sh
# ==============================================================================
# TruBuild Vault - Seed Secrets from .env file
# ==============================================================================
# Run this on your HOST machine (not inside Docker)
# Requires: vault CLI installed (https://developer.hashicorp.com/vault/install)
# Usage: ./seed-secrets.sh <region> <env> [env_file]
# Example: ./seed-secrets.sh ksa dev .env
# Or use: make vault-seed REGION=ksa ENV=dev
# ==============================================================================
set -e

REGION="${1:?Usage: $0 <region> <env> [env_file]}"
ENV="${2:?Usage: $0 <region> <env> [env_file]}"
ENV_FILE="${3:-.env}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
SECRET_PATH="secret/trubuild/${REGION}/${ENV}"

# Check if vault CLI is installed
if ! command -v vault >/dev/null 2>&1; then
    echo "❌ Error: vault CLI not found"
    echo ""
    echo "   Install it from: https://developer.hashicorp.com/vault/install"
    echo "   Or on Ubuntu: sudo apt install vault"
    echo "   Or on macOS: brew install vault"
    exit 1
fi

echo "🌱 Seeding secrets for: ${REGION}/${ENV}"
echo "   Path: ${SECRET_PATH}"
echo "   Env file: ${ENV_FILE}"
echo "   Vault: ${VAULT_ADDR}"
echo ""

# Check for authentication
if [ -z "$VAULT_TOKEN" ]; then
    # Try to read from .vault-keys (check multiple locations)
    if [ -f "$DEPLOY_DIR/.vault-keys" ]; then
        KEYS_FILE="$DEPLOY_DIR/.vault-keys"
    elif [ -f ".vault-keys" ]; then
        KEYS_FILE=".vault-keys"
    else
        echo "❌ Error: VAULT_TOKEN not set and .vault-keys not found"
        echo ""
        echo "   Run 'make vault-init' first, or set VAULT_TOKEN manually"
        echo "   Looked in: $DEPLOY_DIR/.vault-keys and .vault-keys"
        exit 1
    fi
    
    VAULT_TOKEN=$(grep VAULT_ROOT_TOKEN "$KEYS_FILE" | sed 's/^VAULT_ROOT_TOKEN=//')
    export VAULT_TOKEN
    echo "🔑 Using token from $KEYS_FILE"
fi

# Check if Vault is reachable and authenticated
set +e
vault token lookup >/dev/null 2>&1
AUTH_RESULT=$?
set -e

if [ $AUTH_RESULT -ne 0 ]; then
    echo "❌ Error: Cannot authenticate to Vault"
    echo ""
    echo "   Check that Vault is running: make vault-status"
    echo "   Check your VAULT_TOKEN is valid"
    echo "   Token starts with: $(echo "$VAULT_TOKEN" | cut -c1-10)..."
    exit 1
fi

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: Environment file not found: ${ENV_FILE}"
    echo ""
    echo "   Create it from template: cp env.example .env"
    exit 1
fi

echo "📖 Reading secrets from ${ENV_FILE}..."

# Build the vault kv put command arguments
# Read .env file, skip comments and empty lines, convert to lowercase keys
VAULT_ARGS=""
SECRET_COUNT=0

while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    case "$line" in
        ''|'#'*) continue ;;
    esac
    
    # Skip lines without =
    case "$line" in
        *=*) ;;
        *) continue ;;
    esac
    
    # Extract key and value (handle spaces around = and = in values)
    key="${line%%=*}"
    value="${line#*=}"
    
    # Trim whitespace from key and value
    key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Skip certain non-secret variables (Vault/deployment config only)
    case "$key" in
        REGISTRY|TAG|NODE_ENV|VAULT_*)
            echo "   ⏭️  Skipping config var: $key"
            continue
            ;;
    esac
    
    # Convert key to lowercase for Vault
    key_lower=$(echo "$key" | tr '[:upper:]' '[:lower:]')
    
    # Remove surrounding quotes from value if present
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    
    # Skip empty values (Vault doesn't accept key= without value)
    if [ -z "$value" ]; then
        echo "   ⏭️  Skipping empty: $key_lower"
        continue
    fi
    
    # Add to vault args
    VAULT_ARGS="$VAULT_ARGS ${key_lower}=${value}"
    SECRET_COUNT=$((SECRET_COUNT + 1))
    echo "   ✓ $key_lower"
    
done < "$ENV_FILE"

if [ $SECRET_COUNT -eq 0 ]; then
    echo ""
    echo "⚠️  No secrets found in ${ENV_FILE}"
    exit 1
fi

echo ""
echo "📦 Writing ${SECRET_COUNT} secrets to Vault..."

# Execute vault kv put with all arguments
# shellcheck disable=SC2086
vault kv put "${SECRET_PATH}" $VAULT_ARGS

echo ""
echo "============================================================"
echo "✅ ${SECRET_COUNT} secrets seeded for ${REGION}/${ENV}"
echo "============================================================"
echo ""
echo "📋 To view secrets:"
echo "   make vault-get REGION=${REGION} ENV=${ENV}"
echo ""
echo "📋 To update a single secret:"
echo "   vault kv patch ${SECRET_PATH} key=NEW_VALUE"
echo ""
