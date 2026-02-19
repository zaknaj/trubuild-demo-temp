# ==============================================================================
# TruBuild Vault - Service Policy Template
# ==============================================================================
# Read-only access to secrets for a specific region/environment
# Replace REGION and ENV when creating the policy
# ==============================================================================

# Read secrets for this region/environment
path "secret/data/trubuild/REGION/ENV" {
  capabilities = ["read"]
}

# List secrets (for debugging)
path "secret/metadata/trubuild/REGION/ENV" {
  capabilities = ["list", "read"]
}
