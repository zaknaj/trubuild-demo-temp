#!/bin/sh
# ==============================================================================
# Certbot Deploy Hook - Reload Nginx After Certificate Renewal
# ==============================================================================
# This script is called by certbot after successfully renewing a certificate.
# It reloads nginx to pick up the new certificate without downtime.
#
# Requirements:
# - Docker CLI installed in certbot container
# - Docker socket mounted at /var/run/docker.sock
# ==============================================================================

echo "🔄 Certificate renewed! Fixing permissions and reloading nginx..."

# Fix directory permissions (readable by all)
find /etc/letsencrypt/live/ -type d -exec chmod 755 {} \; 2>/dev/null || true
find /etc/letsencrypt/archive/ -type d -exec chmod 755 {} \; 2>/dev/null || true

# Fix certificate permissions (readable by all - these are public)
find /etc/letsencrypt/live/ -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true
find /etc/letsencrypt/archive/ -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true

# SECURITY: Keep private keys restricted (owner + nginx group)
# Certbot writes as root; ensure nginx group can read (group: nginx or gid 101)
if getent group nginx >/dev/null 2>&1; then
    find /etc/letsencrypt/live/ -type f -name "privkey*.pem" -exec chgrp nginx {} \; 2>/dev/null || true
    find /etc/letsencrypt/archive/ -type f -name "privkey*.pem" -exec chgrp nginx {} \; 2>/dev/null || true
else
    find /etc/letsencrypt/live/ -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
    find /etc/letsencrypt/archive/ -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
fi
find /etc/letsencrypt/live/ -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true
find /etc/letsencrypt/archive/ -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true

echo "✅ Permissions fixed!"

# Reload nginx via Docker socket
if [ -S /var/run/docker.sock ]; then
    echo "🔄 Reloading nginx..."
    
    # Find nginx container by label or name pattern
    NGINX_CONTAINER=$(docker ps --filter "label=service=nginx" --format "{{.Names}}" 2>/dev/null | head -n1)
    
    # Fallback: try common naming patterns
    if [ -z "$NGINX_CONTAINER" ]; then
        NGINX_CONTAINER=$(docker ps --filter "name=nginx" --format "{{.Names}}" 2>/dev/null | head -n1)
    fi
    
    if [ -n "$NGINX_CONTAINER" ]; then
        if docker exec "$NGINX_CONTAINER" nginx -s reload 2>/dev/null; then
            echo "✅ Nginx reloaded successfully!"
        else
            echo "⚠️  Failed to reload nginx. Container: $NGINX_CONTAINER"
            echo "   Nginx will use new certs on next restart."
        fi
    else
        echo "⚠️  Could not find nginx container."
        echo "   Nginx will use new certs on next restart."
    fi
else
    echo "⚠️  Docker socket not available."
    echo "   Nginx will use new certs on next restart."
fi

echo "✅ Deploy hook completed!"
