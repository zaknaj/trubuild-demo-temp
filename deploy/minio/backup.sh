#!/bin/sh
# ==============================================================================
# MinIO Daily Backup Script
# ==============================================================================
# This script mirrors the MinIO trubuild bucket to a backup volume
# Designed to run via cron at midnight daily
# ==============================================================================

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y-%m-%d_%H:%M)
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

echo "[$(date)] Starting MinIO backup..."

# Set up MinIO client alias
mc alias set minio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create dated backup directory
mkdir -p ${BACKUP_DIR}/${DATE}

# Mirror trubuild bucket to backup directory
echo "[$(date)] Mirroring trubuild bucket..."
mc mirror --overwrite minio/trubuild ${BACKUP_DIR}/${DATE}/trubuild/ 2>/dev/null || echo "trubuild bucket empty or not found"

# Create a "latest" symlink for easy access
rm -f ${BACKUP_DIR}/latest
ln -sf ${BACKUP_DIR}/${DATE} ${BACKUP_DIR}/latest

# Clean up old backups (older than RETENTION_DAYS)
echo "[$(date)] Cleaning up backups older than ${RETENTION_DAYS} days..."
find ${BACKUP_DIR} -maxdepth 1 -type d -name "20*" -mtime +${RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null || true

# Show backup summary
echo "[$(date)] Backup complete!"
echo "Backup location: ${BACKUP_DIR}/${DATE}"
echo "Current backups:"
ls -la ${BACKUP_DIR}/ | grep -E "^d.*20[0-9]{6}$" || echo "No dated backups found"
du -sh ${BACKUP_DIR}/${DATE} 2>/dev/null || echo "Backup size: 0"
