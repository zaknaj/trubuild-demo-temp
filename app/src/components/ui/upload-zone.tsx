import { useState, useRef } from "react"
import { Upload, X, FileText, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "./button"
import {
  getDocumentUploadUrlFn,
  createDocumentFn,
  deleteDocumentFn,
  type DocumentCategory,
  type DocumentStoragePathMode,
} from "@/fn/documents"
import { toast } from "sonner"

export interface UploadedFile {
  id: string // Database ID (set after upload confirmation)
  name: string
  key: string // S3 object key
}

interface UploadZoneProps {
  files: UploadedFile[]
  onFilesChange: (files: UploadedFile[]) => void
  // Upload context - at least one of packageId or assetId must be provided
  packageId?: string
  assetId?: string
  category: DocumentCategory
  storagePathMode?: DocumentStoragePathMode
  contractorId?: string
  vendorName?: string
  // Callback when uploading state changes (for blocking parent actions)
  onUploadingChange?: (isUploading: boolean) => void
  // UI options
  multiple?: boolean
  accept?: string
  className?: string
  compact?: boolean
}

interface UploadingFile {
  tempId: string
  name: string
  progress: number
}

export function UploadZone({
  files,
  onFilesChange,
  packageId,
  assetId,
  category,
  storagePathMode = "default",
  contractorId,
  vendorName,
  onUploadingChange,
  multiple = false,
  accept = ".pdf,.doc,.docx",
  className,
  compact = false,
}: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState<UploadingFile[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Notify parent when uploading state changes
  const updateUploading = (
    updater: (prev: UploadingFile[]) => UploadingFile[]
  ) => {
    setUploading((prev) => {
      const next = updater(prev)
      const wasUploading = prev.length > 0
      const isNowUploading = next.length > 0
      if (wasUploading !== isNowUploading) {
        onUploadingChange?.(isNowUploading)
      }
      return next
    })
  }

  const uploadFile = async (file: File) => {
    const tempId = crypto.randomUUID()

    // Add to uploading state
    updateUploading((prev) => [
      ...prev,
      { tempId, name: file.name, progress: 0 },
    ])

    try {
      // 1. Get presigned upload URL
      const { uploadUrl, key } = await getDocumentUploadUrlFn({
        data: {
          filename: file.name,
          contentType: file.type || "application/octet-stream",
          category,
          packageId,
          assetId,
          contractorId,
          vendorName,
          storagePathMode,
        },
      })

      // 2. Upload directly to S3
      const xhr = new XMLHttpRequest()

      await new Promise<void>((resolve, reject) => {
        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const progress = Math.round((e.loaded / e.total) * 100)
            setUploading((prev) =>
              prev.map((u) => (u.tempId === tempId ? { ...u, progress } : u))
            )
          }
        })

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve()
          } else {
            reject(new Error(`Upload failed with status ${xhr.status}`))
          }
        })

        xhr.addEventListener("error", () => reject(new Error("Upload failed")))

        xhr.open("PUT", uploadUrl)
        xhr.setRequestHeader(
          "Content-Type",
          file.type || "application/octet-stream"
        )
        xhr.send(file)
      })

      // 3. Create document record in database
      const document = await createDocumentFn({
        data: {
          name: file.name,
          key,
          contentType: file.type || "application/octet-stream",
          size: file.size,
          category,
          packageId,
          assetId,
          contractorId,
        },
      })

      // 4. Add to files list (replace existing if same id - e.g. re-uploaded same filename)
      const uploadedFile: UploadedFile = {
        id: document.id,
        name: document.name,
        key: document.key,
      }

      if (multiple) {
        const existingIndex = files.findIndex((f) => f.id === document.id)
        const newFiles =
          existingIndex >= 0
            ? files.map((f, i) => (i === existingIndex ? uploadedFile : f))
            : [...files, uploadedFile]
        onFilesChange(newFiles)
      } else {
        onFilesChange([uploadedFile])
      }

      toast.success(`Uploaded ${file.name}`)
    } catch (error) {
      console.error("Upload failed:", error)
      toast.error(`Failed to upload ${file.name}`)
    } finally {
      // Remove from uploading state
      updateUploading((prev) => prev.filter((u) => u.tempId !== tempId))
    }
  }

  const handleFileSelect = async (selectedFiles: FileList | null) => {
    if (!selectedFiles || selectedFiles.length === 0) return

    const filesToUpload = multiple
      ? Array.from(selectedFiles)
      : [selectedFiles[0]]

    // Upload files in parallel
    await Promise.all(filesToUpload.map(uploadFile))
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleRemoveFile = async (fileId: string) => {
    try {
      await deleteDocumentFn({ data: { documentId: fileId } })
      onFilesChange(files.filter((f) => f.id !== fileId))
    } catch (error) {
      console.error("Delete failed:", error)
      toast.error("Failed to delete file")
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    handleFileSelect(e.dataTransfer.files)
  }

  const isUploading = uploading.length > 0

  // If single file mode and we have a file, show the file instead of zone
  if (!multiple && files.length > 0) {
    return (
      <div className={cn("space-y-2", className)}>
        <FileItem
          file={files[0]}
          onRemove={() => handleRemoveFile(files[0].id)}
        />
      </div>
    )
  }

  return (
    <div className={cn("space-y-2", className)}>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={(e) => handleFileSelect(e.target.files)}
        className="hidden"
      />

      {/* Upload zone */}
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors",
          compact ? "p-4" : "p-8",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50",
          isUploading && "pointer-events-none opacity-50"
        )}
      >
        {isUploading ? (
          <Loader2
            className={cn(
              "mx-auto text-muted-foreground animate-spin",
              compact ? "size-6" : "size-8"
            )}
          />
        ) : (
          <Upload
            className={cn(
              "mx-auto text-muted-foreground",
              compact ? "size-6" : "size-8"
            )}
          />
        )}
        <p className={cn("font-medium", compact ? "mt-1 text-sm" : "mt-2")}>
          {isUploading ? "Uploading..." : "Click or drag and drop to upload"}
        </p>
        <p className="text-sm text-muted-foreground">
          {accept.replace(/\./g, "").toUpperCase().replace(/,/g, ", ")}
        </p>
      </div>

      {/* Uploading files */}
      {uploading.length > 0 && (
        <div className="space-y-1">
          {uploading.map((file) => (
            <UploadingFileItem key={file.tempId} file={file} />
          ))}
        </div>
      )}

      {/* File list for multiple mode */}
      {multiple && files.length > 0 && (
        <div className="space-y-1">
          {files.map((file) => (
            <FileItem
              key={file.id}
              file={file}
              onRemove={() => handleRemoveFile(file.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function FileItem({
  file,
  onRemove,
}: {
  file: UploadedFile
  onRemove: () => void
}) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-md bg-muted/50 border">
      <FileText className="size-4 text-muted-foreground shrink-0" />
      <span className="text-sm truncate flex-1">{file.name}</span>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="size-6 p-0 hover:bg-destructive/10 hover:text-destructive"
        onClick={(e) => {
          e.stopPropagation()
          onRemove()
        }}
      >
        <X className="size-3.5" />
      </Button>
    </div>
  )
}

function UploadingFileItem({ file }: { file: UploadingFile }) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-md bg-muted/50 border">
      <Loader2 className="size-4 text-muted-foreground shrink-0 animate-spin" />
      <span className="text-sm truncate flex-1">{file.name}</span>
      <span className="text-xs text-muted-foreground">{file.progress}%</span>
    </div>
  )
}
