#!/bin/sh
# ==============================================================================
# TruBuild Vault Initialization Script
# ==============================================================================
# Run this on your HOST machine (not inside Docker)
# Requires: vault CLI installed (https://developer.hashicorp.com/vault/install)
# Usage: make vault-init (saves keys to deploy/.vault-keys)
# ==============================================================================
set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"

# Check if vault CLI is installed
if ! command -v vault >/dev/null 2>&1; then
    echo "❌ Error: vault CLI not found"
    echo ""
    echo "   Install it from: https://developer.hashicorp.com/vault/install"
    echo "   Or on Ubuntu: sudo apt install vault"
    echo "   Or on macOS: brew install vault"
    exit 1
fi

echo "🔐 Initializing TruBuild Vault..."
echo "   Vault address: $VAULT_ADDR"

# Check if Vault is reachable (exit code 2 = sealed but running, which is OK)
# Temporarily disable exit-on-error for this check
set +e
vault status >/dev/null 2>&1
STATUS_CODE=$?
set -e

if [ $STATUS_CODE -ne 0 ] && [ $STATUS_CODE -ne 2 ]; then
    echo "❌ Error: Cannot connect to Vault at $VAULT_ADDR"
    echo ""
    echo "   Make sure Vault is running: make vault-up"
    exit 1
fi

# Check if already initialized
if vault status 2>/dev/null | grep -q "Initialized.*true"; then
    echo "✅ Vault already initialized"
    
    # Check if sealed
    if vault status 2>/dev/null | grep -q "Sealed.*true"; then
        echo "🔓 Vault is sealed."
        echo ""
        echo "To unseal, run: make vault-unseal"
        echo "Your unseal key should be in: deploy/.vault-keys"
        exit 1
    fi
    
    echo ""
    echo "Vault is ready. To authenticate, set VAULT_TOKEN from your .vault-keys file."
    exit 0
fi

echo "📦 Initializing Vault for the first time..."

# Initialize and capture output
INIT_OUTPUT=$(vault operator init -key-shares=1 -key-threshold=1 -format=json)

# Extract keys from JSON using grep/sed (works without jq)
# Get the line after "unseal_keys_b64": [ and extract the key
UNSEAL_KEY=$(echo "$INIT_OUTPUT" | grep -A1 '"unseal_keys_b64"' | tail -1 | tr -d ' ",' | tr -d '[]')
# Get the root_token value
ROOT_TOKEN=$(echo "$INIT_OUTPUT" | grep '"root_token"' | sed 's/.*"root_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

# Check if we got the keys
if [ -z "$UNSEAL_KEY" ] || [ -z "$ROOT_TOKEN" ]; then
    echo "❌ Failed to extract keys from Vault output"
    echo "UNSEAL_KEY: '$UNSEAL_KEY'"
    echo "ROOT_TOKEN: '$ROOT_TOKEN'"
    echo "Raw output:"
    echo "$INIT_OUTPUT"
    exit 1
fi

# Unseal Vault
echo "🔓 Unsealing Vault..."
vault operator unseal "$UNSEAL_KEY" > /dev/null

# Save keys to file
KEYS_FILE="$DEPLOY_DIR/.vault-keys"
echo "VAULT_UNSEAL_KEY=$UNSEAL_KEY" > "$KEYS_FILE"
echo "VAULT_ROOT_TOKEN=$ROOT_TOKEN" >> "$KEYS_FILE"
chmod 600 "$KEYS_FILE"

echo ""
echo "============================================================"
echo "🔐 VAULT INITIALIZATION COMPLETE"
echo "============================================================"
echo ""
echo "⚠️  Credentials saved to: $KEYS_FILE"
echo ""
echo "VAULT_UNSEAL_KEY=$UNSEAL_KEY"
echo "VAULT_ROOT_TOKEN=$ROOT_TOKEN"
echo ""
echo "============================================================"

# Now configure Vault with the root token
export VAULT_TOKEN="$ROOT_TOKEN"

# Enable KV v2 secrets engine
echo "🗄️  Enabling KV secrets engine..."
vault secrets enable -path=secret -version=2 kv 2>/dev/null || echo "   KV engine already enabled"

# Enable AppRole auth
echo "🔑 Enabling AppRole auth..."
vault auth enable approle 2>/dev/null || echo "   AppRole already enabled"

# Create admin policy
echo "📋 Creating policies..."
vault policy write admin "$SCRIPT_DIR/../policies/admin.hcl"

# Define regions and environments
REGIONS="ksa"
ENVS="dev staging prod"

# Create policies and AppRoles for each region/env
for REGION in $REGIONS; do
    for ENV in $ENVS; do
        POLICY_NAME="trubuild-${REGION}-${ENV}"
        ROLE_NAME="${REGION}-${ENV}"
        
        echo "   Creating policy and role: ${ROLE_NAME}"
        
        # Create policy from template
        sed "s/REGION/${REGION}/g; s/ENV/${ENV}/g" "$SCRIPT_DIR/../policies/service.hcl" | \
            vault policy write "${POLICY_NAME}" -
        
        # Create AppRole
        vault write "auth/approle/role/${ROLE_NAME}" \
            token_policies="${POLICY_NAME}" \
            token_ttl=1h \
            token_max_ttl=4h \
            secret_id_ttl=0 \
            secret_id_num_uses=0
    done
done

echo ""
echo "============================================================"
echo "✅ Vault configuration complete!"
echo "============================================================"
echo ""
echo "📝 AppRoles created:"
for REGION in $REGIONS; do
    for ENV in $ENVS; do
        echo "   • ${REGION}-${ENV}"
    done
done
echo ""
echo "📋 Next steps:"
echo "   1. Create .env file: cp env.example .env"
echo "   2. Edit .env with your values"
echo "   3. Seed secrets: make vault-seed REGION=ksa ENV=dev"
echo ""
