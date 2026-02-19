## [2.5.0] ALPHA - 2026-01-20 - Cycle 1

### Added
- MinIO storage support (S3-compatible, self-hosted)
- New `minio_setup.sh` script for MinIO configuration
- Environment variables for MinIO: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_SECURE`

### Removed
- Alibaba OSS integration and all related code
- Amazon S3 direct integration (now uses MinIO with S3-compatible API)
- `alibaba_setup.sh` and `amazon_setup.sh` scripts
- Asymmetric RSA/AES payload encryption (`utils/core/encrypt.py`)
- `prv.pem` and `pub.pem` keypair generation from setup
- `Encryption` environment variable
- `provider` parameter from all storage functions
- Redundant code: `utils/RAG.py`, backup scripts, lifetime scripts, email integrations

### Changed
- `bucket.py` - Refactored to use MinIO endpoint with boto3 S3-compatible client
- `api.py` - Simplified to use plaintext JSON requests/responses (no encryption)
- `docingest.py` - Removed provider parameters from all functions
- `backup_system_logs.py` - Now uses MinIO client from bucket.py
- `setup.sh` - Removed encryption keypair generation, updated to call `minio_setup.sh`
- `service.sh` - Updated descriptions from S3 to MinIO
- All tool files (`tech_rfp`, `comm_rfp`, `contract`, `chat`) - Removed `get_provider` calls and `provider` parameters
- `compactor_cache.py` and `gcs.py` - Removed provider context
- `sim.py` - Simplified test client for plaintext requests
- `check.py` - Removed encrypt_main from test suite
- `README.md` - Updated documentation for MinIO, removed encryption references
- Folder restructuring in `engine/tools`

### Fixed
- Cleaned up unused imports and dead code paths

### Improved
- Simplified storage API by removing multi-provider abstraction
- Reduced codebase complexity (~1200 lines removed)
- Clearer configuration with MinIO-specific environment variables

### To Do
- Update Readme.md
- restructure utils/
- remove redundant code like utils/RAG.py, encryption code, backup, lifetime, email
- update requirements.txt
