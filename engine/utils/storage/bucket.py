"""
MinIO File Storage and Retrieval for Project Artifacts

This module provides a unified interface for interacting with MinIO object storage
using the S3-compatible API via boto3.

Key Operations:
---------------
- Create/Upload:
    - upload_file: Uploads a single file or JSON string to MinIO
    - upload_project: Uploads an entire local project directory recursively
    - upload_directory: Uploads any specified local directory

- Read/Download:
    - download_project: Downloads all files in a project
    - download_sub_directory: Downloads a specific subdirectory
    - download_single_file: Downloads an individual file to a specified path
    - get_file: Retrieves and parses a single cloud-stored JSON file

- Update:
    - upload_file / upload_project handles overwrites

- Delete:
    - delete_file: Deletes a file from MinIO storage

"""
import os
import json
import fitz
import boto3
import shutil
import tempfile
from urllib.parse import quote
from utils.vault import secrets
from utils.core.log import get_logger
from os.path import basename, splitext
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from typing import Any, Dict, Sequence, Union, Optional, Literal, Callable, Awaitable


# MinIO config from Vault/env. Only these keys are used:
#   minio_bucket, minio_endpoint, minio_root_password, minio_root_user, minio_secure
MINIO_ACCESS_KEY = (secrets.get("minio_root_user", default="") or "").strip()
MINIO_SECRET_KEY = (secrets.get("minio_root_password", default="") or "").strip()
_minio_endpoint = (secrets.get("minio_endpoint", default="") or "").strip().rstrip("/")
MINIO_BUCKET = (secrets.get("minio_bucket", default="") or "").strip()
MINIO_SECURE = (secrets.get("minio_secure", default="") or "").strip().lower() == "true"
MINIO_ENDPOINT = _minio_endpoint if _minio_endpoint else "http://minio:9000"


# Use MINIO_BUCKET as the default bucket
bucket_name = MINIO_BUCKET

EXCLUDED_FILES = {".DS_Store", "._.DS_Store", "Thumbs.db", ".git", ".gitignore"}

# Throughput tuning (pooling + retries + multipart concurrency)
S3_BOTOCORE_CONFIG = BotoConfig(
    max_pool_connections=128,
    retries={"max_attempts": 10, "mode": "adaptive"},
)

S3_TRANSFER_CONFIG = TransferConfig(
    max_concurrency=32,
    multipart_threshold=4 * 1024 * 1024,  # 4MB
    multipart_chunksize=8 * 1024 * 1024,  # 8MB
    use_threads=True,
)

# Initialize boto3 S3 client and resource with MinIO endpoint
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    use_ssl=MINIO_SECURE,
    config=S3_BOTOCORE_CONFIG,
)

s3 = boto3.resource(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    use_ssl=MINIO_SECURE,
    config=S3_BOTOCORE_CONFIG,
)


def _nested_key(company_id: str, package_id: str, path: str) -> str:
    """company/project + normalized relative path"""
    return f"{company_id}/{package_id}/{path.lstrip('/')}"


def _strip_nested_prefix(key: str, company_id: str, package_id: str) -> str:
    """
    Convert 'company/project/relpath' -> 'relpath' for local save paths.
    Falls back gracefully if prefix not present.
    """
    prefix = f"{company_id}/{package_id}/"
    if key.startswith(prefix):
        return key[len(prefix) :].lstrip("/")
    # backward safety if someone passes 'project/...'
    legacy = f"{package_id}/"
    if key.startswith(legacy):
        return key[len(legacy) :].lstrip("/")
    return key.lstrip("/")


def get_deleted_data(company_id: str) -> tuple[list[str], list[str]]:
    """
    Retrieve deleted data metadata from MinIO storage.

    Fetches the {company_id}/company-metadata/deleted_data.json file
    and parses it to extract lists of deleted projects and packages.

    Args:
        company_id: Unique company identifier

    Returns:
        Tuple of (projects, packages) where:
        - projects: List of project identifiers (strings)
        - packages: List of package identifiers (strings)

    Examples:
        >>> get_deleted_data("company123")
        (['project1', 'project2'], ['cmh4evn9x000158kcn1628maf'])
    """
    logger = get_logger()

    # Construct the file path for deleted_data.json
    file_path = "company-metadata/deleted_data.json"

    try:
        # Construct the full cloud key
        cloud_key = f"{company_id}/{file_path}"

        # Get S3 object
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(cloud_key)

        # Download and parse JSON
        response = obj.get()
        content = response["Body"].read().decode("utf-8")
        data = json.loads(content)

        # Parse the JSON structure
        projects = data.get("projects", [])
        packages = data.get("packages", [])

        logger.debug(
            f"Retrieved deleted data for company {company_id}: "
            f"{len(projects)} projects, {len(packages)} packages"
        )

        return (projects, packages)

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or error_code == "NoSuchKey":
            logger.warning(
                f"No deleted_data.json found for company {company_id}, "
                "returning empty lists"
            )
            return ([], [])
        else:
            logger.error(f"S3 ClientError in get_deleted_data: {str(e)}")
            raise
    except Exception as e:
        logger.error(
            f"Error retrieving deleted data for company {company_id}: {str(e)}"
        )
        raise


def download_project(package_id: str, company_id: str):
    """
    Download entire project from MinIO storage.

    Downloads all project files while:
    - Filtering excluded files (.DS_Store, etc.)
    - Skipping empty files
    - Validating PDF files
    - Preserving folder structure
    - Processing documents for caching

    Args:
        package_id: Unique project identifier
        company_id: Organization id

    Returns:
        str: Path to temporary directory containing RFP files

    Raises:
        Exception: If no files were successfully downloaded
    """
    logger = get_logger()

    # Setup directories
    permanent_dir, temp_dir = _setup_download_directories(package_id)

    # Get configuration
    config = _get_download_config()

    # Download files from all subdirectories
    downloaded_files = []
    for subdir in config["subdirectories"]:
        files = _download_subdirectory(
            package_id, subdir, permanent_dir, temp_dir, config, company_id
        )
        downloaded_files.extend(files)

    logger.debug(
        f"Downloaded {len(downloaded_files)} files "
        f"(rfp -> temp, others -> ../{package_id})"
    )
    if not downloaded_files:
        raise Exception("No files were successfully downloaded and validated")

    return temp_dir


def _setup_download_directories(package_id: str) -> tuple[str, str]:
    """
    Setup permanent and temporary directories for downloads.

    Args:
        package_id: Unique project identifier

    Returns:
        Tuple of (permanent_dir, temp_dir) paths
    """
    logger = get_logger()
    permanent_dir = os.path.abspath(os.path.join("..", package_id))
    os.makedirs(permanent_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Permanent: {permanent_dir}, Temporary: {temp_dir}")
    return permanent_dir, temp_dir


def _get_download_config() -> dict:
    """
    Get configuration for download operations.

    Returns:
        Dictionary with bucket_name, excluded_files, and subdirectories
    """
    return {
        "bucket_name": MINIO_BUCKET,
        "excluded_files": {
            ".DS_Store",
            "._.DS_Store",
            "Thumbs.db",
            ".git",
            ".gitignore",
        },
        "subdirectories": [
            "rfp/",
            "proposal/",
            "evaluation/",
            "contracts/",
            "documents/",
            "data/",
        ],
    }


def _download_subdirectory(
    package_id: str,
    subdir: str,
    permanent_dir: str,
    temp_dir: str,
    config: dict,
    company_id: str,
) -> list:
    """
    Download all files from a subdirectory.

    Args:
        package_id: Unique project identifier
        subdir: Subdirectory to download from
        permanent_dir: Path to permanent storage directory
        temp_dir: Path to temporary storage directory
        config: Download configuration dictionary
        company_id: Unique organization identifier

    Returns:
        List of successfully downloaded file paths
    """
    downloaded_files = []
    prefix = _nested_key(company_id, package_id, subdir).replace("\\", "/")

    # Get file listing
    file_list = _get_s3_file_list(config["bucket_name"], prefix)

    # Process each file
    for file_info in file_list:
        result = _process_cloud_file(
            file_info, package_id, subdir, permanent_dir, temp_dir, config, company_id
        )
        if result:
            downloaded_files.append(result)

    return downloaded_files


def _get_s3_file_list(bucket_name: str, prefix: str) -> list:
    """
    Get list of files from S3/MinIO bucket.

    Args:
        bucket_name: Bucket name
        prefix: Prefix to filter files

    Returns:
        List of file info dictionaries with 'Key' and 'Size'
    """
    file_list = []
    paginator = s3.meta.client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in page_iterator:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            file_list.append({"Key": obj["Key"], "Size": obj.get("Size", 0)})

    return file_list


def _process_cloud_file(
    file_info: dict,
    package_id: str,
    subdir: str,
    permanent_dir: str,
    temp_dir: str,
    config: dict,
    company_id: str,
) -> str | None:
    """
    Process and download a single cloud file.

    Args:
        file_info: Dictionary with 'Key' and 'Size' of the file
        package_id: Unique project identifier
        subdir: Subdirectory the file belongs to
        permanent_dir: Path to permanent storage directory
        temp_dir: Path to temporary storage directory
        config: Download configuration dictionary
        company_id: Unique organization identifier

    Returns:
        Local file path if successful, None otherwise
    """
    logger = get_logger()
    file_key = file_info["Key"]
    file_size = file_info["Size"]

    # Skip empty files
    if file_size == 0:
        logger.warning(f"Skipping empty file: {file_key}")
        return None

    # Check if file should be excluded
    file_name = os.path.basename(file_key)
    if _should_exclude_file(file_name, config["excluded_files"]):
        logger.debug(f"Skipping hidden/excluded file: {file_key}")
        return None

    # Determine local path
    local_path = _get_local_file_path(
        file_key, company_id, package_id, subdir, permanent_dir, temp_dir
    )

    return None


def _should_exclude_file(file_name: str, excluded_files: set) -> bool:
    """
    Check if file should be excluded from download.

    Args:
        file_name: Name of the file
        excluded_files: Set of excluded file names

    Returns:
        True if file should be excluded, False otherwise
    """
    return file_name.startswith(".") or file_name in excluded_files


def _get_local_file_path(
    file_key: str,
    company_id: str,
    package_id: str,
    subdir: str,
    permanent_dir: str,
    temp_dir: str,
) -> str:
    """
    Determine local path for downloaded file.

    Args:
        file_key: Cloud storage key of the file
        company_id: Unique organization identifier
        package_id: Unique project identifier
        subdir: Subdirectory the file belongs to
        permanent_dir: Path to permanent storage directory
        temp_dir: Path to temporary storage directory

    Returns:
        Local file path
    """
    relative_path = _strip_nested_prefix(file_key, company_id, package_id)

    # Handle special case for tender files
    if relative_path.startswith("rfp/tender/"):
        path_parts = relative_path.split("/")
        if len(path_parts) > 3:
            tender_folder = path_parts[2]
            file_name = path_parts[-1]
            relative_path = f"rfp/tender/{tender_folder}/{file_name}"

    # Determine download directory
    download_dir = temp_dir if subdir.startswith("rfp/") else permanent_dir
    return os.path.join(download_dir, relative_path)


def _download_and_validate_file(
    cloud_path: str,
    local_path: str,
    file_size: int,
    file_name: str,
    bucket_name: str,
) -> bool:
    """
    Download and validate a single file.

    Args:
        cloud_path: Path to file in cloud storage (S3 key)
        local_path: Local destination path
        file_size: Expected file size
        file_name: Name of the file
        bucket_name: Storage bucket name

    Returns:
        True if download and validation successful, False otherwise
    """
    logger = get_logger()
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file from MinIO
        s3.Bucket(bucket_name).download_file(cloud_path, local_path, Config=S3_TRANSFER_CONFIG)

        # Validate download
        if not os.path.exists(local_path):
            logger.error(f"File not found after download: {local_path}")
            return False

        # Check file size
        local_size = os.path.getsize(local_path)
        if local_size != file_size:
            logger.error(
                f"File size mismatch for {cloud_path}! "
                f"Expected: {file_size}, Got: {local_size}"
            )
            return False

        # Validate PDF files
        if local_path.lower().endswith(".pdf"):
            if not _validate_pdf_file(local_path, file_name):
                return False

        return True

    except Exception as e:
        logger.error(f"Failed to download {cloud_path}: {e}")
        return False


def _validate_pdf_file(file_path: str, file_name: str) -> bool:
    """
    Validate that a PDF file can be opened.

    Args:
        file_path: Path to PDF file
        file_name: Name of the file for logging

    Returns:
        True if valid PDF, False otherwise
    """
    logger = get_logger()
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_document = fitz.open(pdf_file)
            logger.debug(f"Validated PDF {file_name}")
        return True
    except Exception as pdf_error:
        logger.error(f"PDF validation failed for {file_name}: {pdf_error}")
        return False


def upload_file(
    package_id: str,
    location: str,
    json_string=None,
    file_path=None,
    company_id: str = None,
):
    """
    Upload JSON file or string to MinIO storage.

    Args:
        package_id: Unique project identifier
        location: Destination path within project directory
        json_string: JSON content as string (optional)
        file_path: Path to local JSON file (optional)
        company_id: Unique organization identifier

    Returns:
        Result of upload operation

    Raises:
        ValueError: If neither json_string nor file_path provided
    """
    logger = get_logger()
    logger.debug("Uploading file...")

    if file_path is not None:
        upload_path = file_path
    elif json_string is not None:
        # write the JSON to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(json_string)
            upload_path = tmp.name
    else:
        raise ValueError(
            "upload_file missing both upload path and json string (must have one)"
        )

    s3_key = _nested_key(company_id, package_id, location)
    try:
        s3_client.upload_file(upload_path, bucket_name, s3_key, Config=S3_TRANSFER_CONFIG)
        logger.debug(f"Uploaded {upload_path} to minio://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload file to MinIO: {e}")
        raise


def download_sub_directory(package_id: str, sub_dir: str, company_id: str):
    """
    Download subdirectory from MinIO storage.

    Args:
        package_id: Unique project identifier
        sub_dir: Subdirectory path within project
        company_id: Unique organization identifier

    Returns:
        str: Path to local directory containing downloaded files
    """
    logger = get_logger()
    s3_path = _nested_key(company_id, package_id, sub_dir)
    return _s3_download(s3_path, recursive=True)


def get_file(package_id: str, file_path: str, company_id: str):
    """
    Retrieve and parse JSON file from MinIO storage.

    Args:
        package_id: Unique project identifier
        file_path: Path to file within project directory
        company_id: Unique organization identifier

    Returns:
        dict: Parsed JSON content or None if file not found
    """
    logger = get_logger()

    s3_key = _nested_key(company_id, package_id, file_path)
    tmp_dir = None
    try:
        tmp_dir = _s3_download(s3_key, recursive=False)
        if not tmp_dir:
            return None
        return _read_json_file(tmp_dir)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            logger.debug(f"{s3_key} not found yet; returning None")
            return None
        raise
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _read_json_file(tmp_path) -> dict | None:
    """
    Read JSON from either:
      - a directory containing a .json file (first one wins), or
      - a direct file path to a JSON file.
    Returns dict (parsed JSON) or None if not found / unreadable.
    """
    logger = get_logger()

    # Handle None early to avoid TypeError in os.path.join
    if not tmp_path:
        logger.debug("_read_json_file called with None; returning None")
        return None

    path = str(tmp_path)

    try:
        # If direct file path
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except UnicodeDecodeError:
                # fall back for BOM or odd encodings
                with open(path, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            except Exception as e:
                logger.exception("Failed to read JSON file %r: %s", path, e)
                return None

        # If directory path
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.lower().endswith(".json"):
                    fpath = os.path.join(path, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except UnicodeDecodeError:
                        with open(fpath, "r", encoding="utf-8-sig") as f:
                            return json.load(f)
                    except Exception as e:
                        logger.exception("Failed to read JSON file %r: %s", fpath, e)
                        # try next file if there is one
                        continue
            logger.debug("No JSON files found in %r; returning None", path)
            return None

        # Path doesn't exist or is something else
        logger.debug("_read_json_file got non-existent path %r; returning None", path)
        return None

    except Exception as e:
        logger.exception("Error in _read_json_file(%r): %s", path, e)
        return None


def delete_file(package_id: str, file_path: str, company_id: str):
    """
    Delete file from MinIO storage.

    Args:
        package_id: Unique project identifier
        file_path: Path to file within project directory
        company_id: Unique organization identifier

    Returns:
        True if deletion successful, False otherwise
    """
    logger = get_logger()
    s3_key = _nested_key(company_id, package_id, file_path)
    try:
        s3.Object(bucket_name, s3_key).delete()
        logger.debug(f"Deleted minio://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file from MinIO: {e}")
        return False


def upload_project(package_id: str, company_id: str = None):
    """
    Upload entire local project directory to MinIO storage.

    Recursively uploads all files in ../{package_id} while preserving
    the folder structure.

    Args:
        package_id: Unique project identifier
        company_id: Unique organization identifier
    """
    logger = get_logger()

    root_path = f"../{package_id}"
    for root, _, files in os.walk(root_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, start=root_path)
            try:
                upload_file(
                    package_id,
                    relative_path,
                    file_path=local_file_path,
                    company_id=company_id,
                )
            except Exception as e:
                logger.error(
                    f"Failed to upload file '{relative_path}' for project: {e} "
                )
    logger.debug(f"Successfully uploaded project to MinIO")


def upload_directory(package_id: str, directory_path: str, company_id: str = None):
    """
    Upload local directory to MinIO storage.

    Recursively uploads all files while preserving folder structure.

    Args:
        package_id: Unique project identifier
        directory_path: Local directory path to upload
        company_id: Unique organization identifier
    """
    logger = get_logger()

    if not os.path.exists(directory_path):
        logger.warning(f"Directory '{directory_path}' not found for upload. Skipping.")
        return

    for root, _, files in os.walk(directory_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, start=f"../{package_id}")
            try:
                upload_file(
                    package_id,
                    relative_path,
                    file_path=local_file_path,
                    company_id=company_id,
                )
            except Exception as e:
                logger.error(
                    f"Failed to upload file '{relative_path}' for project "
                    f"'{package_id}': {e}"
                )
    logger.debug(
        f"Uploaded directory '{directory_path}' for project '{package_id}' to MinIO."
    )


def s3_download_project(package_id: str, company_id: str):
    """
    Download entire project from MinIO.

    Args:
        package_id: Project identifier
        company_id: Company identifier

    Returns:
        str: Path to temporary directory containing downloaded files

    Raises:
        Exception: If download fails
    """
    logger = get_logger()
    prefix = _nested_key(company_id, package_id, "")
    tmp_dir = tempfile.mkdtemp()
    try:
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=prefix):
            rel_key = obj.key[len(prefix) :]
            if not rel_key or rel_key.endswith("/"):
                continue
            target = os.path.join(tmp_dir, rel_key)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            bucket.download_file(obj.key, target, Config=S3_TRANSFER_CONFIG)
        return tmp_dir
    except Exception as e:
        logger.error(f"Failed to download project from MinIO: {e}")
        raise


def _s3_download(s3_path: str, recursive=True):
    """
    Download from MinIO.

    Args:
        s3_path: S3 key or prefix
        recursive: Whether to download recursively

    Returns:
        str: Path to temporary directory containing downloaded files
    """
    logger = get_logger()
    tmp_dir = tempfile.mkdtemp()

    try:
        bucket = s3.Bucket(bucket_name)
        if recursive:
            prefix = s3_path if s3_path.endswith("/") else s3_path + "/"
            for obj in bucket.objects.filter(Prefix=prefix):
                rel_key = obj.key[len(prefix) :]
                # skip the root-folder marker and any "directory" placeholders
                if not rel_key or rel_key.endswith("/"):
                    continue
                target = os.path.join(tmp_dir, obj.key[len(prefix) :])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                bucket.download_file(obj.key, target, Config=S3_TRANSFER_CONFIG)
        else:
            target = os.path.join(tmp_dir, os.path.basename(s3_path))
            bucket.download_file(s3_path, target, Config=S3_TRANSFER_CONFIG)
        logger.debug(f"Downloaded minio://{bucket_name}/{s3_path} to {tmp_dir}")
        return tmp_dir

    except Exception as e:
        logger.debug(
            f"_s3_download could not find this file in the cloud "
            f"(may still be processing): {s3_path} {e}"
        )
        return None


# Helper function to flatten tender subdirectories
def _flatten_tender_subdirectories(tender_base_dir: str):
    """
    Flatten tender directory structure.

    Moves files from sub-subdirectories directly under contractor folders.
    Example: rfp/tender/contractorA/subfolder/file.pdf ->
             rfp/tender/contractorA/file.pdf

    Args:
        tender_base_dir: Path to tender directory

    Note:
        Removes empty subdirectories after flattening
    """
    logger = get_logger()
    if not os.path.isdir(tender_base_dir):
        logger.debug(
            f"Tender directory '{tender_base_dir}' not found for flattening. "
            f"Skipping."
        )
        return

    logger.debug(f"Starting flattening process for directory: {tender_base_dir}")
    for contractor_name in os.listdir(tender_base_dir):
        contractor_path = os.path.join(tender_base_dir, contractor_name)
        if not os.path.isdir(contractor_path):
            continue

        logger.debug(
            f"Processing contractor directory for flattening: {contractor_path}"
        )
        files_to_move = []
        for dirpath, _, filenames in os.walk(contractor_path):
            # Only process files in subdirectories of the contractor_path
            if dirpath == contractor_path:
                continue

            for filename in filenames:
                source_path = os.path.join(dirpath, filename)
                target_path = os.path.join(contractor_path, filename)

                # Should not happen with dirpath != contractor_path
                if os.path.abspath(source_path) == os.path.abspath(target_path):
                    continue

                if os.path.exists(target_path):
                    logger.warning(
                        f"File '{filename}' already exists in "
                        f"'{contractor_path}'. Overwriting from "
                        f"'{source_path}'."
                    )

                files_to_move.append((source_path, target_path))

        for source, target in files_to_move:
            try:
                shutil.move(source, target)
                logger.debug(f"Moved '{source}' to '{target}' for flattening.")
            except Exception as e:
                logger.error(
                    f"Error moving file {source} to {target} for flattening: {e}"
                )

        # Clean up empty subdirectories
        for dirpath, dirnames, filenames_in_dir in os.walk(
            contractor_path, topdown=False
        ):
            # Don't attempt to remove the main contractor folder
            if dirpath == contractor_path:
                continue
            # Check if directory is empty
            if not dirnames and not filenames_in_dir:
                # Double check it's truly empty with os.listdir
                if not os.listdir(dirpath):
                    try:
                        os.rmdir(dirpath)
                        logger.debug(f"Removed empty directory: {dirpath}")
                    except OSError as e:
                        logger.error(f"Error removing empty directory {dirpath}: {e}")


def download_single_file(package_id: str, cloud_key: str, local_path: str):
    """
    Download single file from MinIO to specific local path.

    Args:
        package_id: Project identifier
        cloud_key: Full key, including package_id
        local_path: Full local destination path

    Returns:
        str: The local_path on success

    Raises:
        FileNotFoundError: If file not found in cloud
        IOError: If download fails
        Exception: On unexpected errors
    """
    logger = get_logger()

    logger.debug(
        f"Attempting to download single file. "
        f"Project: {package_id}, Key: {cloud_key}"
    )

    # ensure the target local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.Bucket(bucket_name).download_file(cloud_key, local_path, Config=S3_TRANSFER_CONFIG)
        logger.debug(f"Successfully downloaded from MinIO to {local_path}")

    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            logger.error(
                f"File not found in MinIO storage: "
                f"minio://{bucket_name}/{cloud_key}"
            )
            raise FileNotFoundError(f"File not found in MinIO: {cloud_key}") from e
        logger.error(f"Boto3 ClientError during download: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during download: {e}")
        raise

    return local_path


def list_files(package_id: str, subdir: str, company_id: str) -> list[str]:
    """
    List all files in a MinIO directory.

    Args:
        package_id: Unique project identifier
        subdir: Subdirectory path within the project (e.g., "contracts/")
        company_id: Unique company identifier

    Returns:
        List of keys
    """
    prefix = _nested_key(company_id, package_id, subdir).replace("\\", "/")
    raw_list = _get_s3_file_list(bucket_name, prefix)
    return [item["Key"] for item in raw_list]


def list_subdirectories(package_id: str, subdir: str, company_id: str) -> list[str]:
    """
    List all 'subdirectories' (common prefixes) in a MinIO directory.

    Args:
        package_id: Unique project identifier.
        subdir: Parent directory path within the project (e.g., "tech_rfp/tender/").
        company_id: Unique company identifier

    Returns:
        Sorted list of subdirectory names (e.g., ["ContractorA", "ContractorB"]).
    """
    if not bucket_name:
        raise RuntimeError("MINIO_BUCKET environment variable is not set")

    prefix = (
        _nested_key(company_id, package_id, subdir).replace("\\", "/").rstrip("/") + "/"
    )

    subdirs = _get_s3_subdirectories(bucket_name, prefix)

    # Unique + sorted for deterministic behavior
    return sorted(set(subdirs))


def _get_s3_subdirectories(bucket_name: str, prefix: str) -> list[str]:
    """Gets subdirectories from S3/MinIO using CommonPrefixes."""
    subdir_list: list[str] = []
    paginator = s3.meta.client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter="/",
    )

    for page in page_iterator:
        for cp in page.get("CommonPrefixes", []):
            full_path = cp.get("Prefix", "")
            subdir_name = full_path.replace(prefix, "", 1).strip("/")
            if subdir_name:
                subdir_list.append(subdir_name)

    return subdir_list


def _stem(p: str) -> str:
    return splitext(basename(p))[0]


async def serve_precomputed_or_recompute(
    *,
    package_id: str,
    company_id: str,
    compute_reanalysis: bool,
    start_compute: (
        Callable[[], Awaitable[Dict[str, Any]]] | Callable[[], Dict[str, Any]]
    ),
    # one or many files to read when serving cache:
    result_path: Union[str, Sequence[str]],
    # optional labels for multi-file results (same length as result_path list)
    names: Optional[Sequence[str]] = None,
    # where to put the payload: "data" (default) or None to merge top-level if dict
    envelope_key: Optional[str] = "data",
    logger: Any = None,
    miss_policy: Literal["error", "compute"] = "error",
) -> Dict[str, Any]:
    """
    If compute_reanalysis=False:
      - Load all 'result_path' files.
      - If single path: payload is the file's content.
      - If multiple paths: payload is a dict; keys come from 'names' or file name stems.
      - On success: return {"status":"done", envelope_key: payload, "cache": True}
      - On miss: return {"status":"error","error":"precomputed_not_found"} or start compute (miss_policy="compute")

    If compute_reanalysis=True:
      - Call start_compute() and return its result.
    """
    from inspect import iscoroutinefunction
    from typing import Awaitable

    # Normalize paths to a list
    paths = [result_path] if isinstance(result_path, str) else list(result_path)

    if not compute_reanalysis:
        try:
            loaded: list[Any] = []
            for p in paths:
                obj = get_file(
                    package_id=package_id, file_path=p, company_id=company_id
                )
                if not obj:
                    loaded = []  # mark miss
                    break
                loaded.append(obj)

        except Exception as e:
            if logger:
                logger.exception(f"cache fetch failed: {e}")
            return {
                "status": "error",
                "error": "precomputed_fetch_failed",
                "details": str(e),
            }

        if loaded:
            # Build payload
            if len(loaded) == 1:
                payload: Any = loaded[0]
            else:
                # label by names[] or filename stems
                if names and len(names) == len(loaded):
                    payload = {names[i]: loaded[i] for i in range(len(loaded))}
                else:
                    payload = {_stem(paths[i]): loaded[i] for i in range(len(loaded))}

            # place payload into response
            if envelope_key is None and isinstance(payload, dict):
                return {"status": "done", **payload, "cache": True}
            else:
                key = envelope_key or "data"
                return {"status": "done", key: payload, "cache": True}

        # computed cache miss
        if miss_policy == "compute":
            # fall through to compute
            pass
        else:
            return {"status": "error", "error": "precomputed_not_found"}

    # compute path (or miss fallback)
    out = start_compute()
    if isinstance(out, Awaitable):
        out = await out
    return out


def list_company_packages(company_id: str) -> list[str]:
    """
    List all package directories under a company.

    Args:
        company_id: Unique company identifier

    Returns:
        List of package IDs (directory names) under the company

    Example:
        For structure: bucket/{company_id}/{package_id}/...
        Returns: ["pkg1", "pkg2", "pkg3"]

    Note:
        Excludes the "company-metadata" folder which is at the same level
    """
    logger = get_logger()

    # Folders to exclude from package listing
    EXCLUDED_FOLDERS = {"company-metadata"}

    # The prefix is just the company_id (no package_id since we're listing projects)
    prefix = f"{company_id}/"

    logger.debug(f"Listing packages under company {company_id} with prefix: {prefix}")

    packages = _get_s3_company_packages(bucket_name, prefix, EXCLUDED_FOLDERS)

    logger.debug(
        f"Found {len(packages)} packages under company {company_id}: {packages}"
    )
    return sorted(set(packages))


def _get_s3_company_packages(
    bucket_name: str, prefix: str, excluded: set[str]
) -> list[str]:
    """Get package directories from S3/MinIO using CommonPrefixes at company level."""
    package_list: list[str] = []
    paginator = s3.meta.client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter="/",
    )

    for page in page_iterator:
        for cp in page.get("CommonPrefixes", []):
            full_path = cp.get("Prefix", "")
            # Extract package name: "company_id/package_id/" -> "package_id"
            package_name = full_path.replace(prefix, "", 1).strip("/")
            if package_name and package_name not in excluded:
                package_list.append(package_name)

    return package_list


def build_public_url_for_key(source_key: str) -> str:
    """
    Build a stable, shareable URL for a given MinIO object key.

    `source_key` must be the full key in the bucket, i.e.:
        company_id/package_id/tech_rfp/.../file.pdf

    Args:
        source_key: Full object key in the bucket

    Returns:
        URL pointing at that object in MinIO
    """
    if not bucket_name:
        raise RuntimeError("MINIO_BUCKET environment variable is not set")

    encoded_key = quote(source_key.lstrip("/"), safe="/")

    # Build MinIO URL from endpoint
    endpoint = MINIO_ENDPOINT.rstrip("/")
    return f"{endpoint}/{bucket_name}/{encoded_key}"

def get_text_or_json_file(
    package_id: str,
    file_path: str,
    company_id: str,
) -> Optional[Any]:
    """
    Retrieve an object and return:
      - dict/list if it is JSON
      - str if it is plain text (e.g., .txt)
      - None if object not found

    """
    logger = get_logger()

    def _read_text_or_json(local_fp: str) -> Optional[Any]:
        try:
            with open(local_fp, "rb") as f:
                b = f.read()
            if not b:
                return None
            s = b.decode("utf-8-sig", "ignore").strip()
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                return s
        except Exception as e:
            logger.exception("Failed reading %r: %s", local_fp, e)
            return None

    s3_key = _nested_key(company_id, package_id, file_path)
    tmp_dir = None
    try:
        tmp_dir = _s3_download(s3_key, recursive=False)
        if not tmp_dir:
            return None

        # _s3_download(non-recursive) downloads basename(s3_key) into tmp_dir
        local_fp = os.path.join(tmp_dir, os.path.basename(s3_key))
        if not os.path.isfile(local_fp):
            # fallback: first regular file
            files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            files = [f for f in files if os.path.isfile(f)]
            if not files:
                return None
            local_fp = files[0]

        return _read_text_or_json(local_fp)

    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            logger.debug(f"{s3_key} not found yet; returning None")
            return None
        raise
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def head_single_file(*, cloud_key: str) -> Optional[Dict[str, Any]]:
    """
    return object metadata WITHOUT downloading the file.

    - On success: dict with at least {key, provider, size, etag, last_modified}
    - If not found: None
    """
    logger = get_logger()

    try:
        resp = s3_client.head_object(Bucket=bucket_name, Key=cloud_key)

        etag = (resp.get("ETag") or "").strip('"')
        lm = resp.get("LastModified")  # datetime in boto3
        size = resp.get("ContentLength")

        return {
            "provider": "minio",
            "key": cloud_key,
            "size": size,
            "etag": etag,
            "last_modified": lm.isoformat() if hasattr(lm, "isoformat") else lm,
            # Optional extras
            "content_type": resp.get("ContentType"),
            "metadata": resp.get("Metadata") or {},
            "storage_class": resp.get("StorageClass"),
        }

    except ClientError as e:
        code = (e.response.get("Error", {}) or {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        logger.exception("MinIO head_object failed for %s: %s", cloud_key, e)
        raise

def main() -> bool:
    """
    MinIO self-test for this bucket module.

    What it does:
      0) Quick sanity: MINIO_BUCKET set
      1) Upload JSON by string and by file (upload_file)
      2) Upload a small local directory (upload_directory)
      3) Upload the whole ../{package_id} tree (upload_project)
      4) Verify download_sub_directory('data') actually contains expected files
      5) download_single_file (requires FULL cloud key, incl. package_id)
      6) get_file() roundtrip check
      7) delete_file() for specific objects and verify 404 miss via get_file
      8) s3_download_project() for whole-project download
      9) list_subdirectories() check (expects 'extra' from the prepared tree)

    Prints OK/FAIL and returns a boolean.
    """
    import os, json, tempfile, shutil, time, random
    from utils.core.log import pid_tool_logger, set_logger, get_logger

    print("MINIO BUCKET TEST START")

    if not bucket_name:
        print("BUCKET FAIL: MINIO_BUCKET env var is not set")
        return False

    # Prepare unique project namespace on disk
    package_id = f"system_check_{int(time.time())}_{random.randint(1000,9999)}"
    company_id = f"system_check_{int(time.time())}_{random.randint(1,10)}"
    base_proj_dir = os.path.join("..", package_id)
    os.makedirs(base_proj_dir, exist_ok=True)

    # Logging context
    set_logger(pid_tool_logger("SYSTEM_CHECK", "bucket"))
    log = get_logger()

    attempted = 0
    succeeded = 0
    tmp_things_to_cleanup: list[str] = []

    try:
        # Common, predictable content layout under ../{package_id}
        # - a nested directory used by upload_directory()
        # - a small "test_upload" subtree used by upload_project()
        # - a "documents" subtree to verify download_project() pulls non-RFP into permanent dir
        dir_for_upload_directory = os.path.join(
            base_proj_dir, "dir_for_upload_directory", "data", "extra"
        )
        os.makedirs(dir_for_upload_directory, exist_ok=True)
        with open(
            os.path.join(dir_for_upload_directory, "dummy.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"dummy": True}, f)

        # Exclusion sentinel (should never be surfaced by download paths)
        with open(os.path.join(base_proj_dir, ".DS_Store"), "w", encoding="utf-8") as f:
            f.write("ignore me")

        test_upload_root = os.path.join(base_proj_dir, "test_upload")
        os.makedirs(test_upload_root, exist_ok=True)
        with open(
            os.path.join(test_upload_root, "sample.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"message": "project sample"}, f)

        documents_dir = os.path.join(base_proj_dir, "documents")
        os.makedirs(documents_dir, exist_ok=True)
        with open(os.path.join(documents_dir, "note.json"), "w", encoding="utf-8") as f:
            json.dump({"hello": "world"}, f)

        # RFP tree used by download_project to route into temp dir
        rfp_dir = os.path.join(base_proj_dir, "rfp", "tender", "ContractorA", "nested")
        os.makedirs(rfp_dir, exist_ok=True)
        with open(os.path.join(rfp_dir, "rfp_doc.txt"), "w", encoding="utf-8") as f:
            f.write("hello rfp")

        attempted += 1
        log.debug("=== MinIO test start ===")

        # Known objects
        test_json_data = {"status": "ok", "source": "selftest:minio"}
        loc1 = "data/test_file.json"
        loc2 = "data/from_file.json"

        # 1) Upload JSON via string
        upload_file(
            package_id=package_id,
            location=loc1,
            json_string=json.dumps(test_json_data),
            company_id=company_id,
        )

        # 2) Upload JSON via temp file
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump({"upload": "from_file"}, tf)
            temp_file_path = tf.name
        try:
            upload_file(
                package_id=package_id,
                location=loc2,
                file_path=temp_file_path,
                company_id=company_id,
            )
            upload_file(
                package_id=package_id,
                location="data/extra/marker.json",
                json_string=json.dumps({"ok": True}),
                company_id=company_id,
            )
        finally:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

        # 3) Upload a directory living under ../{package_id}
        upload_directory(
            package_id,
            os.path.join(base_proj_dir, "dir_for_upload_directory"),
            company_id=company_id,
        )

        # 4) Upload the entire ../{package_id} tree
        upload_project(package_id, company_id=company_id)

        # 5) Download a subdirectory (verify presence of expected files)
        dl_subdir_path = download_sub_directory(package_id, "data", company_id)
        if dl_subdir_path:
            tmp_things_to_cleanup.append(dl_subdir_path)
            expected = {"test_file.json", "from_file.json"}
            found = set()
            for root, _, files in os.walk(dl_subdir_path):
                for fn in files:
                    if fn.endswith(".json"):
                        found.add(fn)
            assert expected.issubset(
                found
            ), f"download_sub_directory missing: {expected - found}"

        # 6) Download a single file — NOTE: pass FULL cloud key (incl. company_id/package_id)
        single_dl = tempfile.NamedTemporaryFile(delete=False)
        single_dl.close()
        tmp_things_to_cleanup.append(single_dl.name)
        full_key = _nested_key(company_id, package_id, loc1)
        download_single_file(
            package_id=package_id,
            cloud_key=full_key,
            local_path=single_dl.name,
        )
        # quick verify single file is readable JSON
        with open(single_dl.name, "r", encoding="utf-8") as f:
            js = json.load(f)
            assert js.get("status") == "ok", "single-file JSON content mismatch"

        # 7) get_file roundtrip check
        got = get_file(package_id, file_path=loc1, company_id=company_id)
        assert (
            isinstance(got, dict) and got.get("status") == "ok"
        ), "get_file roundtrip failed"

        # 8) Delete a specific object then verify missing
        delete_file(package_id, loc1, company_id=company_id)
        post = get_file(package_id, file_path=loc1, company_id=company_id)
        assert post is None, f"delete_file did not remove {loc1}"

        # 9) Provider-specific whole-project download
        proj_dl = s3_download_project(package_id, company_id=company_id)
        assert proj_dl and os.path.isdir(proj_dl), "project download failed"
        tmp_things_to_cleanup.append(proj_dl)

        # Check that a known uploaded file is present in that provider project download
        donor = os.path.join(proj_dl, "test_upload", "sample.json")
        assert os.path.exists(donor), "project download missing test_upload/sample.json"

        # 10) list_subdirectories(): should find 'extra' under data/
        subs = list_subdirectories(package_id, "data", company_id=company_id)
        assert "extra" in subs, f"list_subdirectories missing 'extra'"

        # All checks passed
        succeeded += 1
        log.debug("=== MinIO test OK ===")

        if succeeded == 0 and attempted > 0:
            print("BUCKET FAIL: no tests succeeded")
            return False

        print("MINIO BUCKET OK")
        return True

    except AssertionError as e:
        log.exception(f"Bucket self-test assertion failed: {e}")
        print("BUCKET ERROR")
        return False
    except Exception as e:
        log.exception(f"Bucket self-test error: {e}")
        print("BUCKET ERROR")
        return False
    finally:
        # CLOUD CLEANUP: purge every key under the project prefix
        try:
            keys = list_files(package_id, "", company_id=company_id)
            for full_key in keys:
                # full_key looks like '{company_id}/{package_id}/path/to/file'
                prefix = _nested_key(company_id, package_id, "")
                rel = (
                    full_key[len(prefix) :] if full_key.startswith(prefix) else full_key
                )
                try:
                    delete_file(package_id, rel, company_id=company_id)
                except Exception:
                    pass
        except Exception:
            pass

        # LOCAL CLEANUP
        try:
            shutil.rmtree(base_proj_dir, ignore_errors=True)
        except Exception:
            pass
        for p in tmp_things_to_cleanup:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()
