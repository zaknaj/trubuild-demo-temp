import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  GetObjectCommand,
} from "@aws-sdk/client-s3"
import { getSignedUrl } from "@aws-sdk/s3-request-presigner"

const minioEndpoint = process.env.minio_endpoint
if (!minioEndpoint) {
  throw new Error("Missing required environment variable: minio_endpoint")
}
const minioRootUser = process.env.minio_root_user
if (!minioRootUser) {
  throw new Error("Missing required environment variable: minio_root_user")
}
const minioRootPassword = process.env.minio_root_password
if (!minioRootPassword) {
  throw new Error("Missing required environment variable: minio_root_password")
}
const bucketName = process.env.minio_bucket
if (!bucketName) {
  throw new Error("Missing required environment variable: minio_bucket")
}

// S3 client singleton
const s3 = new S3Client({
  endpoint: minioEndpoint,
  region: "auto",
  credentials: {
    accessKeyId: minioRootUser,
    secretAccessKey: minioRootPassword,
  },
  forcePathStyle: true, // Required for most S3-compatible services
})

const BUCKET_NAME = bucketName

/**
 * Generate a presigned URL for uploading a file
 */
export async function getPresignedUploadUrl(
  key: string,
  contentType: string,
  expiresIn = 3600 // 1 hour default
): Promise<string> {
  const command = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
    ContentType: contentType,
  })

  return getSignedUrl(s3, command, { expiresIn })
}

/**
 * Generate a presigned URL for downloading a file
 */
export async function getPresignedDownloadUrl(
  key: string,
  expiresIn = 3600 // 1 hour default
): Promise<string> {
  const command = new GetObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
  })

  return getSignedUrl(s3, command, { expiresIn })
}

/**
 * Delete an object from S3
 */
export async function deleteObject(key: string): Promise<void> {
  const command = new DeleteObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
  })

  await s3.send(command)
}

/**
 * Write JSON content to an object key.
 */
export async function putJsonObject(
  key: string,
  value: unknown
): Promise<void> {
  const command = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
    Body: JSON.stringify(value),
    ContentType: "application/json",
  })

  await s3.send(command)
}

/**
 * Generate a unique S3 key for a document
 */
export function generateDocumentKey(
  entityId: string, // packageId or assetId
  category: string,
  filename: string
): string {
  const uuid = crypto.randomUUID()
  const sanitized = sanitizeFilename(filename)
  return `documents/${entityId}/${category}/${uuid}/${sanitized}`
}

function sanitizeFilename(filename: string): string {
  // Remove special characters but keep extension-friendly symbols
  return filename.replace(/[^a-zA-Z0-9._-]/g, "_")
}

function normalizePathPrefix(pathPrefix: string): string {
  return pathPrefix.replace(/^\/+|\/+$/g, "")
}

export function generateDocumentKeyWithPrefix(
  pathPrefix: string,
  filename: string,
  options?: {
    includeUuidParent?: boolean
  }
): string {
  const uuid = crypto.randomUUID()
  const sanitized = sanitizeFilename(filename)
  const normalizedPrefix = normalizePathPrefix(pathPrefix)
  if (options?.includeUuidParent === false) {
    return `${normalizedPrefix}/${sanitized}`
  }
  return `${normalizedPrefix}/${uuid}/${sanitized}`
}

export { s3, BUCKET_NAME }
