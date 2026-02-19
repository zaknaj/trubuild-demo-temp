import { db } from "@/db"
import { jobs, jobRuns, jobArtifacts, pkg, asset } from "@/db/schema"
import { createServerFn } from "@tanstack/react-start"
import { and, desc, eq } from "drizzle-orm"
import { z } from "zod"
import { getAuthContext, requirePackageAccess } from "@/auth/auth-guards"
import { putJsonObject } from "@/lib/s3"
import {
  commRfpJobStatusResponseSchema,
  createCommRfpCompareJobInputSchema,
  extractCommReportFromArtifact,
} from "@/lib/comm-job"
import type {
  TechRfpEvaluationCriteriaType,
  TechRfpResult,
  TechRfpResultArtifact,
} from "@/lib/tech_rfp"
import type { CommRfpJobStatus } from "@/lib/types"

const TECH_RFP_EVALUATION_EXTRACT = "tech_rfp_evaluation_extract" as const
const TECH_RFP_GENERATE_EVAL = "tech_rfp_generate_eval" as const
const TECH_RFP_ANALYSIS = "tech_rfp_analysis" as const
const COMM_RFP_COMPARE = "comm_rfp_compare" as const

const jobStatusSchema = z.enum([
  "pending",
  "in_progress",
  "completed",
  "failed",
  "cancelled",
])

const evaluationCriteriaSchema: z.ZodType<TechRfpEvaluationCriteriaType> =
  z.record(
    z.string(),
    z.record(
      z.string(),
      z.object({
        weight: z.number(),
        description: z.string(),
      })
    )
  )

export const createTechRfpEvaluationExtractJobFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      packageId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [newJob] = await db
      .insert(jobs)
      .values({
        type: TECH_RFP_EVALUATION_EXTRACT,
        payload: { package_id: data.packageId },
        companyId: ctx.activeOrgId,
        userId: ctx.userId,
      })
      .returning({ id: jobs.id })

    return { jobId: newJob.id }
  })

export const createTechRfpAnalysisJobFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      evaluationCriteria: evaluationCriteriaSchema,
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [packageRow] = await db
      .select({ name: pkg.name })
      .from(pkg)
      .where(eq(pkg.id, data.packageId))
      .limit(1)

    if (!packageRow) {
      throw new Error("Package not found")
    }

    const [newJob] = await db
      .insert(jobs)
      .values({
        type: TECH_RFP_ANALYSIS,
        payload: {
          package_id: data.packageId,
          evaluation_criteria: data.evaluationCriteria,
          user_name: ctx.userEmail,
          metadata: {
            package_name: packageRow.name,
          },
        },
        companyId: ctx.activeOrgId,
        userId: ctx.userId,
      })
      .returning({ id: jobs.id })

    return { jobId: newJob.id }
  })

export const createTechRfpGenerateEvalJobFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      packageId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [newJob] = await db
      .insert(jobs)
      .values({
        type: TECH_RFP_GENERATE_EVAL,
        payload: { package_id: data.packageId },
        companyId: ctx.activeOrgId,
        userId: ctx.userId,
      })
      .returning({ id: jobs.id })

    return { jobId: newJob.id }
  })

export const writeTechRfpEvaluationJsonFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      evaluationCriteria: evaluationCriteriaSchema,
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const key = `${ctx.activeOrgId}/${data.packageId}/data/evaluation.json`
    await putJsonObject(key, data.evaluationCriteria)

    return { key }
  })

export const getTechRfpEvaluationExtractJobStatusFn = createServerFn()
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        type: jobs.type,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, TECH_RFP_EVALUATION_EXTRACT),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [latestRun] = await db
      .select({
        runId: jobRuns.id,
        status: jobRuns.status,
      })
      .from(jobRuns)
      .where(eq(jobRuns.jobId, data.jobId))
      .orderBy(desc(jobRuns.attemptNo), desc(jobRuns.createdAt))
      .limit(1)

    if (!latestRun) {
      return {
        status: "pending" as const,
        runId: null,
      }
    }

    return {
      status: jobStatusSchema.parse(latestRun.status),
      runId: latestRun.runId,
    }
  })

export const getTechRfpEvaluationExtractResultFn = createServerFn()
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Verify job exists and belongs to the user's org
    const [job] = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, TECH_RFP_EVALUATION_EXTRACT),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    // Get the latest completed run's "result" artifact
    const [artifact] = await db
      .select({
        data: jobArtifacts.data,
      })
      .from(jobArtifacts)
      .innerJoin(jobRuns, eq(jobArtifacts.runId, jobRuns.id))
      .where(
        and(
          eq(jobRuns.jobId, data.jobId),
          eq(jobRuns.status, "completed"),
          eq(jobArtifacts.artifactType, "result")
        )
      )
      .orderBy(desc(jobRuns.attemptNo))
      .limit(1)

    if (!artifact) {
      throw new Error("No result artifact found for this job")
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return { data: artifact.data as any }
  })

export const getTechRfpAnalysisJobStatusFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        type: jobs.type,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, TECH_RFP_ANALYSIS),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [latestRun] = await db
      .select({
        runId: jobRuns.id,
        status: jobRuns.status,
        error: jobRuns.error,
      })
      .from(jobRuns)
      .where(eq(jobRuns.jobId, data.jobId))
      .orderBy(desc(jobRuns.attemptNo), desc(jobRuns.createdAt))
      .limit(1)

    if (!latestRun) {
      return {
        status: "pending" as const,
        runId: null,
        error: null,
      }
    }

    return {
      status: jobStatusSchema.parse(latestRun.status),
      runId: latestRun.runId,
      error: latestRun.error,
    }
  })

export const getTechRfpGenerateEvalJobStatusFn = createServerFn()
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        type: jobs.type,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, TECH_RFP_GENERATE_EVAL),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [latestRun] = await db
      .select({
        runId: jobRuns.id,
        status: jobRuns.status,
      })
      .from(jobRuns)
      .where(eq(jobRuns.jobId, data.jobId))
      .orderBy(desc(jobRuns.attemptNo), desc(jobRuns.createdAt))
      .limit(1)

    if (!latestRun) {
      return {
        status: "pending" as const,
        runId: null,
      }
    }

    return {
      status: jobStatusSchema.parse(latestRun.status),
      runId: latestRun.runId,
    }
  })

export const getTechRfpGenerateEvalResultFn = createServerFn()
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, TECH_RFP_GENERATE_EVAL),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [artifact] = await db
      .select({
        data: jobArtifacts.data,
      })
      .from(jobArtifacts)
      .innerJoin(jobRuns, eq(jobArtifacts.runId, jobRuns.id))
      .where(
        and(
          eq(jobRuns.jobId, data.jobId),
          eq(jobRuns.status, "completed"),
          eq(jobArtifacts.artifactType, "result")
        )
      )
      .orderBy(desc(jobRuns.attemptNo))
      .limit(1)

    if (!artifact) {
      throw new Error("No result artifact found for this job")
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return { data: artifact.data as any }
  })

export const getTechRfpAnalysisResultFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, TECH_RFP_ANALYSIS),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [resultArtifact] = await db
      .select({
        data: jobArtifacts.data,
      })
      .from(jobArtifacts)
      .innerJoin(jobRuns, eq(jobArtifacts.runId, jobRuns.id))
      .where(
        and(
          eq(jobRuns.jobId, data.jobId),
          eq(jobRuns.status, "completed"),
          eq(jobArtifacts.artifactType, "result")
        )
      )
      .orderBy(desc(jobRuns.attemptNo))
      .limit(1)

    const [reportArtifact] = resultArtifact
      ? [undefined]
      : await db
          .select({
            data: jobArtifacts.data,
          })
          .from(jobArtifacts)
          .innerJoin(jobRuns, eq(jobArtifacts.runId, jobRuns.id))
          .where(
            and(
              eq(jobRuns.jobId, data.jobId),
              eq(jobRuns.status, "completed"),
              eq(jobArtifacts.artifactType, "report")
            )
          )
          .orderBy(desc(jobRuns.attemptNo))
          .limit(1)

    const artifact = resultArtifact ?? reportArtifact
    if (!artifact) {
      throw new Error('No "result" or "report" artifact found for this job')
    }

    const artifactData = artifact.data as TechRfpResultArtifact | TechRfpResult

    return { data: artifactData }
  })

export const createCommRfpCompareJobFn = createServerFn({ method: "POST" })
  .inputValidator(createCommRfpCompareJobInputSchema)
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)
    const [assetRecord] = await db
      .select({
        id: asset.id,
      })
      .from(asset)
      .where(and(eq(asset.id, data.assetId), eq(asset.packageId, data.packageId)))
      .limit(1)

    if (!assetRecord) {
      throw new Error("Asset does not belong to package")
    }

    const [newJob] = await db
      .insert(jobs)
      .values({
        type: COMM_RFP_COMPARE,
        payload: {
          package_id: data.packageId,
          asset_id: data.assetId,
          analysis_type: data.analysisType,
          user_name: ctx.userEmail,
        },
        companyId: ctx.activeOrgId,
        userId: ctx.userId,
      })
      .returning({ id: jobs.id })

    return { jobId: newJob.id }
  })

export const getLatestCommRfpCompareResultForAssetFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      assetId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [assetRecord] = await db
      .select({
        id: asset.id,
      })
      .from(asset)
      .where(and(eq(asset.id, data.assetId), eq(asset.packageId, data.packageId)))
      .limit(1)
    if (!assetRecord) {
      throw new Error("Asset does not belong to package")
    }

    const candidateJobs = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(eq(jobs.type, COMM_RFP_COMPARE), eq(jobs.companyId, ctx.activeOrgId))
      )
      .orderBy(desc(jobs.createdAt))
      .limit(200)

    const matchingJobIds = candidateJobs
      .map((job) => {
        const payload = job.payload as Record<string, unknown>
        const packageId = payload.package_id
        const assetId = payload.asset_id
        const matchesPackage = packageId === data.packageId
        const matchesAsset = typeof assetId === "string" && assetId === data.assetId
        const legacyPackageMatch = assetId == null && matchesPackage
        return matchesAsset || legacyPackageMatch ? job.id : null
      })
      .filter((id): id is string => Boolean(id))

    for (const jobId of matchingJobIds) {
      const [artifact] = await db
        .select({
          data: jobArtifacts.data,
        })
        .from(jobArtifacts)
        .innerJoin(jobRuns, eq(jobArtifacts.runId, jobRuns.id))
        .where(
          and(
            eq(jobRuns.jobId, jobId),
            eq(jobRuns.status, "completed"),
            eq(jobArtifacts.artifactType, "result")
          )
        )
        .orderBy(desc(jobRuns.attemptNo), desc(jobArtifacts.createdAt))
        .limit(1)

      if (artifact?.data) {
        return {
          jobId,
          data: extractCommReportFromArtifact(artifact.data),
        }
      }
    }

    throw new Error("No completed commercial result artifact found for this asset")
  })

export const getLatestCommRfpCompareJobStatusForAssetFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      assetId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [assetRecord] = await db
      .select({
        id: asset.id,
      })
      .from(asset)
      .where(and(eq(asset.id, data.assetId), eq(asset.packageId, data.packageId)))
      .limit(1)
    if (!assetRecord) {
      throw new Error("Asset does not belong to package")
    }

    const candidateJobs = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(eq(jobs.type, COMM_RFP_COMPARE), eq(jobs.companyId, ctx.activeOrgId))
      )
      .orderBy(desc(jobs.createdAt))
      .limit(200)

    const matchingJobIds = candidateJobs
      .map((job) => {
        const payload = job.payload as Record<string, unknown>
        const packageId = payload.package_id
        const assetId = payload.asset_id
        const matchesPackage = packageId === data.packageId
        const matchesAsset = typeof assetId === "string" && assetId === data.assetId
        const legacyPackageMatch = assetId == null && matchesPackage
        return matchesAsset || legacyPackageMatch ? job.id : null
      })
      .filter((id): id is string => Boolean(id))

    for (const jobId of matchingJobIds) {
      const [latestRun] = await db
        .select({
          runId: jobRuns.id,
          status: jobRuns.status,
          error: jobRuns.error,
          progress: jobRuns.progress,
        })
        .from(jobRuns)
        .where(eq(jobRuns.jobId, jobId))
        .orderBy(desc(jobRuns.attemptNo), desc(jobRuns.createdAt))
        .limit(1)

      if (!latestRun) continue

      const parsed = commRfpJobStatusResponseSchema.parse({
        status: jobStatusSchema.parse(latestRun.status),
        runId: latestRun.runId,
        error: latestRun.error,
        progress: (latestRun.progress as Record<string, {}> | null) ?? null,
      })

      return {
        found: true as const,
        jobId,
        ...parsed,
      }
    }

    return { found: false as const }
  })

export const getCommRfpCompareJobStatusFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, COMM_RFP_COMPARE),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [latestRun] = await db
      .select({
        runId: jobRuns.id,
        status: jobRuns.status,
        error: jobRuns.error,
        progress: jobRuns.progress,
      })
      .from(jobRuns)
      .where(eq(jobRuns.jobId, data.jobId))
      .orderBy(desc(jobRuns.attemptNo), desc(jobRuns.createdAt))
      .limit(1)

    const statusResult: CommRfpJobStatus = latestRun
      ? {
          status: jobStatusSchema.parse(latestRun.status),
          runId: latestRun.runId,
          error: latestRun.error,
          progress: (latestRun.progress as Record<string, {}> | null) ?? null,
        }
      : {
          status: "pending",
          runId: null,
          error: null,
          progress: null,
        }

    return commRfpJobStatusResponseSchema.parse(statusResult) as {
      status: "pending" | "in_progress" | "completed" | "failed" | "cancelled"
      runId: string | null
      error: string | null
      progress: Record<string, {}> | null
    }
  })

export const getCommRfpCompareResultFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      jobId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [job] = await db
      .select({
        id: jobs.id,
        payload: jobs.payload,
      })
      .from(jobs)
      .where(
        and(
          eq(jobs.id, data.jobId),
          eq(jobs.type, COMM_RFP_COMPARE),
          eq(jobs.companyId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!job) {
      throw new Error("Job not found")
    }

    const packageId = z
      .object({ package_id: z.string().uuid() })
      .parse(job.payload).package_id

    await requirePackageAccess(ctx, packageId)

    const [artifact] = await db
      .select({
        data: jobArtifacts.data,
      })
      .from(jobArtifacts)
      .innerJoin(jobRuns, eq(jobArtifacts.runId, jobRuns.id))
      .where(
        and(
          eq(jobRuns.jobId, data.jobId),
          eq(jobRuns.status, "completed"),
          eq(jobArtifacts.artifactType, "result")
        )
      )
      .orderBy(desc(jobRuns.attemptNo))
      .limit(1)

    if (!artifact?.data) {
      throw new Error("No result artifact found for this job")
    }

    return {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      data: extractCommReportFromArtifact(artifact.data) as any,
    }
  })
