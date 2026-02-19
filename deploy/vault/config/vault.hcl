# ==============================================================================
# TruBuild Vault Configuration - Single Instance
# ==============================================================================
# Serves all regions: ksa, uae, eu, us
# Serves all environments: dev, staging, prod
# ==============================================================================

ui = true

# Listener - TLS disabled for internal Docker network
# Enable TLS if exposing externally
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = true
  # For production with TLS, uncomment:
  # tls_cert_file = "/vault/tls/cert.pem"
  # tls_key_file  = "/vault/tls/key.pem"
}

# File storage - simple and sufficient for single instance
storage "file" {
  path = "/vault/data"
}

# Disable mlock for Docker (enable on bare metal VM)
disable_mlock = true

# API address for client redirects
api_addr = "http://vault:8200"

# Logging
log_level = "info"
