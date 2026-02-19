#!/bin/sh
# ==============================================================================
# MinIO Bucket Initialization Script
# ==============================================================================
# This script creates the default buckets when MinIO starts and configures
# lifecycle policies for log retention.
# It's designed to be idempotent - safe to run multiple times
# ==============================================================================

set -e

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until mc alias set local http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} > /dev/null 2>&1; do
    sleep 1
done

echo "MinIO is ready. Creating buckets..."

# Create the trubuild bucket (--ignore-existing prevents errors if bucket exists)
mc mb --ignore-existing local/trubuild

# Create the logs bucket for Loki
mc mb --ignore-existing local/trubuild-logs

# --------------------------------------------------------------------------
# Lifecycle Policy - 365-day retention on logs bucket
# --------------------------------------------------------------------------
# Apply a lifecycle rule that automatically expires objects after 365 days.
# This ensures logs are stored for exactly 12 months and then cleaned up.
# --------------------------------------------------------------------------
echo "Configuring 365-day lifecycle policy on trubuild-logs bucket..."

cat > /tmp/lifecycle-logs.json <<'EOF'
{
    "Rules": [
        {
            "ID": "expire-logs-after-365-days",
            "Status": "Enabled",
            "Expiration": {
                "Days": 365
            },
            "Filter": {
                "Prefix": ""
            }
        }
    ]
}
EOF

mc ilm import local/trubuild-logs < /tmp/lifecycle-logs.json
rm -f /tmp/lifecycle-logs.json

echo "Lifecycle policy applied. Verifying..."
mc ilm ls local/trubuild-logs

# Set bucket policies (optional - uncomment if needed)
# mc policy set download local/trubuild

echo ""
echo "MinIO initialization complete!"
echo "Buckets created:"
mc ls local/
