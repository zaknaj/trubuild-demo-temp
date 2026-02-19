# ==============================================================================
# TruBuild Vault - Admin Policy
# ==============================================================================
# Full access to all secrets and system configuration
# ==============================================================================

# Full access to all TruBuild secrets
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# System configuration
path "sys/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Auth configuration
path "auth/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
