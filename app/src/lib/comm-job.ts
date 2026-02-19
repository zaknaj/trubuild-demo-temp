import { z } from "zod"

const commRfpJobStatusSchema = z.enum([
  "pending",
  "in_progress",
  "completed",
  "failed",
  "cancelled",
])

export const commRfpJobProgressSchema = z
  .object({
    stage: z.string().optional(),
    message: z.string().optional(),
    source_file: z.string().optional(),
    vendor_count: z.number().optional(),
  })
  .passthrough()

export const commRfpReportSchema = z
  .object({
    vendors: z.array(z.string()),
    summary: z.record(z.string(), z.unknown()),
    divisions: z.array(z.record(z.string(), z.unknown())),
    warnings: z
      .array(
        z.object({
          category: z.string(),
          vendor: z.string().optional(),
          division: z.string().optional(),
          grouping: z.string().optional(),
          item_id: z.string().optional(),
          item_description: z.string().optional(),
          message: z.string().optional(),
        })
      )
      .optional(),
  })
  .passthrough()

export const commRfpCompareResultSchema = z
  .object({
    analysisType: z.literal("compare"),
    template_file: z.string(),
    template_key: z.string(),
    pte_file: z.string().nullable(),
    pte_key: z.string().nullable(),
    vendors: z.record(
      z.string(),
      z.object({
        file: z.string(),
        key: z.string(),
      })
    ),
    report: commRfpReportSchema,
  })
  .passthrough()

export const createCommRfpCompareJobInputSchema = z.object({
  packageId: z.string().uuid(),
  assetId: z.string().uuid(),
  analysisType: z.literal("compare").default("compare"),
})

export const commRfpJobStatusResponseSchema = z.object({
  status: commRfpJobStatusSchema,
  runId: z.string().uuid().nullable(),
  error: z.string().nullable(),
  progress: commRfpJobProgressSchema.nullable(),
})

export function extractCommReportFromArtifact(raw: unknown): Record<string, unknown> {
  if (raw && typeof raw === "object") {
    const obj = raw as Record<string, unknown>

    // Common wrapper used by some job artifact writers.
    if (obj.result) {
      return extractCommReportFromArtifact(obj.result)
    }

    // Alternate envelope key naming from Python-side payloads.
    if (obj.analysis_type === "compare" && obj.report) {
      return extractCommReportFromArtifact(obj.report)
    }
  }

  const compareEnvelope = commRfpCompareResultSchema.safeParse(raw)
  if (compareEnvelope.success) {
    return compareEnvelope.data.report
  }

  const directReport = commRfpReportSchema.safeParse(raw)
  if (directReport.success) {
    return directReport.data
  }

  throw new Error("Commercial result artifact has unsupported shape")
}

export function getCommJobStatusText(status: string, progress?: unknown): string {
  const parsedProgress = commRfpJobProgressSchema.safeParse(progress)
  if (parsedProgress.success && parsedProgress.data.message) {
    return parsedProgress.data.message
  }

  switch (status) {
    case "pending":
      return "Queued for processing"
    case "in_progress":
      return "Processing commercial comparison"
    case "completed":
      return "Completed"
    case "failed":
      return "Comparison failed"
    case "cancelled":
      return "Comparison cancelled"
    default:
      return "Processing commercial comparison"
  }
}
