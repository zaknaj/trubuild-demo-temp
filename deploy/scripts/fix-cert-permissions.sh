#!/bin/bash
# ==============================================================================
# Fix Let's Encrypt Certificate Permissions
# ==============================================================================
# This script fixes permissions on Let's Encrypt certificates so that
# non-root processes (like nginx) can read them, while keeping private
# keys secure.
#
# Run after:
# - Initial certificate generation
# - Certificate renewal
# ==============================================================================

set -e

CERT_DIR="$(pwd)/certbot-conf"
CERTBOT_CONF_VOLUME="${CERTBOT_CONF_VOLUME:-certbot-conf}"

fix_permissions() {
    local target_dir="$1"
    local group_name="$2"

    echo "🔧 Fixing certificate permissions in $target_dir..."

    # Fix directory permissions (allow read+execute for all)
    find "$target_dir" -type d -exec chmod 755 {} \; 2>/dev/null || true

    # Fix certificate file permissions (readable by all - these are public)
    # Includes: fullchain.pem, cert.pem, chain.pem, README
    find "$target_dir" -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true

    # SECURITY: Keep private keys restricted (owner + nginx group)
    if [ -n "$group_name" ]; then
        find "$target_dir" -type f -name "privkey*.pem" -exec chgrp "$group_name" {} \; 2>/dev/null || true
    else
        find "$target_dir" -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
    fi
    find "$target_dir" -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true

    # Specifically fix the live and archive directories
    if [ -d "$target_dir/live" ]; then
        chmod 755 "$target_dir/live" 2>/dev/null || true
        # Fix subdirectories
        find "$target_dir/live" -type d -exec chmod 755 {} \; 2>/dev/null || true
        # Public certs readable
        find "$target_dir/live" -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true
        # Private keys restricted (owner + nginx group)
        if [ -n "$group_name" ]; then
            find "$target_dir/live" -type f -name "privkey*.pem" -exec chgrp "$group_name" {} \; 2>/dev/null || true
        else
            find "$target_dir/live" -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
        fi
        find "$target_dir/live" -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true
    fi

    if [ -d "$target_dir/archive" ]; then
        chmod 755 "$target_dir/archive" 2>/dev/null || true
        # Fix subdirectories
        find "$target_dir/archive" -type d -exec chmod 755 {} \; 2>/dev/null || true
        # Public certs readable
        find "$target_dir/archive" -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true
        # Private keys restricted (owner + nginx group)
        if [ -n "$group_name" ]; then
            find "$target_dir/archive" -type f -name "privkey*.pem" -exec chgrp "$group_name" {} \; 2>/dev/null || true
        else
            find "$target_dir/archive" -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
        fi
        find "$target_dir/archive" -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true
    fi
}

# Check if running from deploy directory or system letsencrypt path
if [ ! -d "$CERT_DIR" ] && [ -d "/etc/letsencrypt" ]; then
    CERT_DIR="/etc/letsencrypt"
fi

if [ -d "$CERT_DIR" ]; then
    if getent group nginx >/dev/null 2>&1; then
        fix_permissions "$CERT_DIR" "nginx"
    else
        fix_permissions "$CERT_DIR" ""
    fi
    echo "✅ Certificate permissions fixed!"
    echo "   - Directories: 755 (readable by all)"
    echo "   - Certificates: 644 (readable by all)"
    echo "   - Private keys: 640 (owner + nginx group)"
    exit 0
fi

# Fall back to Docker volume if present (named volume not visible on host)
if docker volume inspect "$CERTBOT_CONF_VOLUME" >/dev/null 2>&1; then
    echo "🔧 Fixing certificate permissions in Docker volume: $CERTBOT_CONF_VOLUME..."
    docker run --rm \
        -v "${CERTBOT_CONF_VOLUME}:/etc/letsencrypt" \
        alpine:3.19 \
        sh -c '
            set -e
            CERT_DIR="/etc/letsencrypt"
            find "$CERT_DIR" -type d -exec chmod 755 {} \; 2>/dev/null || true
            find "$CERT_DIR" -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true
            find "$CERT_DIR" -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
            find "$CERT_DIR" -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true
            if [ -d "$CERT_DIR/live" ]; then
                chmod 755 "$CERT_DIR/live" 2>/dev/null || true
                find "$CERT_DIR/live" -type d -exec chmod 755 {} \; 2>/dev/null || true
                find "$CERT_DIR/live" -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true
                find "$CERT_DIR/live" -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
                find "$CERT_DIR/live" -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true
            fi
            if [ -d "$CERT_DIR/archive" ]; then
                chmod 755 "$CERT_DIR/archive" 2>/dev/null || true
                find "$CERT_DIR/archive" -type d -exec chmod 755 {} \; 2>/dev/null || true
                find "$CERT_DIR/archive" -type f ! -name "privkey*.pem" -exec chmod 644 {} \; 2>/dev/null || true
                find "$CERT_DIR/archive" -type f -name "privkey*.pem" -exec chgrp 101 {} \; 2>/dev/null || true
                find "$CERT_DIR/archive" -type f -name "privkey*.pem" -exec chmod 640 {} \; 2>/dev/null || true
            fi
        '
    echo "✅ Certificate permissions fixed in Docker volume!"
    echo "   - Directories: 755 (readable by all)"
    echo "   - Certificates: 644 (readable by all)"
    echo "   - Private keys: 640 (owner + nginx group)"
    exit 0
fi

echo "⚠️  Certificate directory not found. Skipping permission fix."
exit 0
