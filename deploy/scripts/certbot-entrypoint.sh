#!/bin/sh
# ==============================================================================
# Certbot Entrypoint - Ensure deploy hook exists under mounted volume
# ==============================================================================

set -e

HOOK_SRC="/usr/local/bin/certbot-deploy-hook.sh"
HOOK_DEST="/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh"

mkdir -p /etc/letsencrypt/renewal-hooks/deploy

if [ -f "$HOOK_SRC" ]; then
    cp "$HOOK_SRC" "$HOOK_DEST"
    chmod +x "$HOOK_DEST"
fi

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

trap exit TERM
while :; do
    certbot renew --webroot -w /var/www/certbot --quiet
    sleep 12h &
    wait $!
done
