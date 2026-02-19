import { db } from "@/db"
import {
  technicalEvaluation,
  commercialEvaluation,
  packageContractor,
} from "@/db/schema"
import { createServerFn } from "@tanstack/react-start"
import { desc, eq } from "drizzle-orm"
import { z } from "zod"
import {
  getAuthContext,
  requirePackageTechnicalAccess,
  requirePackageCommercialAccess,
} from "../auth/auth-guards"
import { ERRORS } from "@/lib/errors"
import { generateMockBOQData } from "@/lib/mock-boq-data"
import { generateMockPTCs } from "@/lib/mock-ptc-data"
import type { ContractorPTCs } from "@/lib/types"
import type { TechRfpTechnicalEvaluationContractorPtcs } from "@/lib/tech_rfp"

// ============================================================================
// Technical Evaluations
// ============================================================================

const lineReferenceSchema = z
  .object({
    fileName: z.string().min(1),
    startLine: z.number().int().min(1),
    endLine: z.number().int().min(1).optional(),
  })
  .refine(
    (value) => value.endLine === undefined || value.endLine >= value.startLine,
    {
      message: "endLine must be greater than or equal to startLine",
      path: ["endLine"],
    }
  )

const evidenceFileSchema = z.object({
  id: z.uuid(),
  name: z.string().min(1),
  fakeUrl: z.string().min(1).optional(),
})

const evidenceSchema = z.object({
  id: z.uuid(),
  text: z.string().trim().min(1),
  source: z.enum(["auto", "manual"]),
  files: z.array(evidenceFileSchema),
  lineReference: lineReferenceSchema.optional(),
})

const scoreDataSchema = z.object({
  score: z.number().min(0).max(100),
  comment: z.string().optional(),
  approved: z.boolean(),
  evidence: z.array(evidenceSchema),
})

const technicalPtcRefTypeSchema = z.enum(["N/A", "MISSING_INFO", "INCOMPLETE"])

const technicalPtcItemSchema = z.object({
  criterion: z.string().trim().min(1),
  queryDescription: z.string().trim().min(1),
  refType: technicalPtcRefTypeSchema,
  status: z.enum(["pending", "closed"]).optional(),
  vendorResponse: z.string().optional(),
})

const technicalContractorPtcsSchema = z.object({
  contractorId: z.uuid(),
  contractorName: z.string().trim().min(1),
  ptcs: z.array(technicalPtcItemSchema),
})

const technicalEvaluationDataSchema = z
  .object({
    status: z
      .enum(["setup", "analyzing", "ready", "review_complete"])
      .optional(),
    setupStep: z.union([z.literal(1), z.literal(2), z.literal(3)]).optional(),
    documentsUploaded: z.boolean().optional(),
    analysis: z
      .object({
        jobId: z.uuid(),
        status: z.enum(["queued", "running", "completed", "failed"]),
        error: z.string().optional(),
      })
      .optional(),
    criteria: z
      .object({
        scopes: z.array(z.unknown()),
      })
      .optional(),
    proposalsUploaded: z.array(z.uuid()).optional(),
    scores: z
      .record(z.string(), z.record(z.string(), scoreDataSchema))
      .optional(),
    ptcs: z.array(technicalContractorPtcsSchema).optional(),
  })

export const createTechnicalEvaluationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      data: z.any().optional().default({}),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageTechnicalAccess(ctx, data.packageId)
    const validatedData = technicalEvaluationDataSchema.parse(data.data ?? {})

    // Get the next round number
    const existing = await db
      .select({ roundNumber: technicalEvaluation.roundNumber })
      .from(technicalEvaluation)
      .where(eq(technicalEvaluation.packageId, data.packageId))
      .orderBy(desc(technicalEvaluation.roundNumber))
      .limit(1)

    const nextRoundNumber = existing.length > 0 ? existing[0].roundNumber + 1 : 1

    const [newEval] = await db
      .insert(technicalEvaluation)
      .values({
        packageId: data.packageId,
        roundNumber: nextRoundNumber,
        roundName: `Round ${nextRoundNumber}`,
        data: validatedData,
      })
      .returning()

    return newEval
  })

export const listTechnicalEvaluationsFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageTechnicalAccess(ctx, data.packageId)

    const evaluations = await db
      .select()
      .from(technicalEvaluation)
      .where(eq(technicalEvaluation.packageId, data.packageId))
      .orderBy(desc(technicalEvaluation.roundNumber))

    return evaluations
  })

export const getTechnicalEvaluationFn = createServerFn()
  .inputValidator(z.object({ evaluationId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [evaluation] = await db
      .select()
      .from(technicalEvaluation)
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .limit(1)

    if (!evaluation) throw new Error(ERRORS.NOT_FOUND("Technical Evaluation"))
    await requirePackageTechnicalAccess(ctx, evaluation.packageId)

    return evaluation
  })

export const updateTechnicalEvaluationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      evaluationId: z.uuid(),
      data: z.any(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    const validatedData = technicalEvaluationDataSchema.parse(data.data)

    // Get existing to check access
    const [existing] = await db
      .select()
      .from(technicalEvaluation)
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .limit(1)

    if (!existing) throw new Error(ERRORS.NOT_FOUND("Technical Evaluation"))
    await requirePackageTechnicalAccess(ctx, existing.packageId)

    const [updated] = await db
      .update(technicalEvaluation)
      .set({
        data: validatedData,
        updatedAt: new Date(),
      })
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .returning()

    return updated
  })

export const runTechnicalEvaluationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      evaluationId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Get the evaluation
    const [evaluation] = await db
      .select()
      .from(technicalEvaluation)
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .limit(1)

    if (!evaluation) throw new Error(ERRORS.NOT_FOUND("Technical Evaluation"))
    await requirePackageTechnicalAccess(ctx, evaluation.packageId)

    // Preserve existing data. Technical PTCs are generated from engine analysis artifacts.
    const existingData = (evaluation.data as Record<string, unknown>) || {}
    const updatedData = {
      ...existingData,
    }

    // Update evaluation with data
    const [updated] = await db
      .update(technicalEvaluation)
      .set({
        data: updatedData,
        updatedAt: new Date(),
      })
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .returning()

    return updated
  })

export const updateTechnicalEvaluationPTCsFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      evaluationId: z.uuid(),
      ptcs: z.array(technicalContractorPtcsSchema),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Get the evaluation
    const [evaluation] = await db
      .select()
      .from(technicalEvaluation)
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .limit(1)

    if (!evaluation) throw new Error(ERRORS.NOT_FOUND("Technical Evaluation"))
    await requirePackageTechnicalAccess(ctx, evaluation.packageId)

    // Update only the PTCs in the evaluation data
    const existingData = (evaluation.data as Record<string, unknown>) || {}
    const updatedData = {
      ...existingData,
      ptcs: data.ptcs as TechRfpTechnicalEvaluationContractorPtcs[],
    }

    const [updated] = await db
      .update(technicalEvaluation)
      .set({
        data: updatedData,
        updatedAt: new Date(),
      })
      .where(eq(technicalEvaluation.id, data.evaluationId))
      .returning()

    return updated
  })

// ============================================================================
// Commercial Evaluations
// ============================================================================

export const createCommercialEvaluationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      assetId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Get package from asset to check access
    const [assetRecord] = await db.query.asset.findMany({
      where: (asset, { eq }) => eq(asset.id, data.assetId),
      limit: 1,
    })

    if (!assetRecord) throw new Error(ERRORS.NOT_FOUND("Asset"))
    await requirePackageCommercialAccess(ctx, assetRecord.packageId)

    // Get the next round number
    const existing = await db
      .select({ roundNumber: commercialEvaluation.roundNumber })
      .from(commercialEvaluation)
      .where(eq(commercialEvaluation.assetId, data.assetId))
      .orderBy(desc(commercialEvaluation.roundNumber))
      .limit(1)

    const nextRoundNumber = existing.length > 0 ? existing[0].roundNumber + 1 : 1

    // Create evaluation without data (setup not started)
    const [newEval] = await db
      .insert(commercialEvaluation)
      .values({
        assetId: data.assetId,
        roundNumber: nextRoundNumber,
        roundName: `Round ${nextRoundNumber}`,
        data: {},
      })
      .returning()

    return newEval
  })

export const runCommercialEvaluationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      evaluationId: z.uuid(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Get the evaluation
    const [evaluation] = await db
      .select()
      .from(commercialEvaluation)
      .where(eq(commercialEvaluation.id, data.evaluationId))
      .limit(1)

    if (!evaluation) throw new Error(ERRORS.NOT_FOUND("Commercial Evaluation"))

    // Get package from asset to check access
    const [assetRecord] = await db.query.asset.findMany({
      where: (asset, { eq }) => eq(asset.id, evaluation.assetId),
      limit: 1,
    })

    if (!assetRecord) throw new Error(ERRORS.NOT_FOUND("Asset"))
    await requirePackageCommercialAccess(ctx, assetRecord.packageId)

    // Fetch the package's contractors
    const contractors = await db
      .select()
      .from(packageContractor)
      .where(eq(packageContractor.packageId, assetRecord.packageId))

    if (contractors.length === 0) {
      throw new Error("No contractors found for this package")
    }

    // Generate mock BOQ data
    const contractorList = contractors.map((c) => ({ id: c.id, name: c.name }))
    const mockBOQData = generateMockBOQData(contractorList)

    // Generate mock PTCs
    const mockPTCs = generateMockPTCs(contractorList)

    // Update evaluation with data
    const [updated] = await db
      .update(commercialEvaluation)
      .set({
        data: { ...mockBOQData, ptcs: mockPTCs },
        updatedAt: new Date(),
      })
      .where(eq(commercialEvaluation.id, data.evaluationId))
      .returning()

    return updated
  })

export const updateCommercialEvaluationPTCsFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      evaluationId: z.uuid(),
      ptcs: z.any(), // ContractorPTCs[]
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Get the evaluation
    const [evaluation] = await db
      .select()
      .from(commercialEvaluation)
      .where(eq(commercialEvaluation.id, data.evaluationId))
      .limit(1)

    if (!evaluation) throw new Error(ERRORS.NOT_FOUND("Commercial Evaluation"))

    // Get package from asset to check access
    const [assetRecord] = await db.query.asset.findMany({
      where: (asset, { eq }) => eq(asset.id, evaluation.assetId),
      limit: 1,
    })

    if (!assetRecord) throw new Error(ERRORS.NOT_FOUND("Asset"))
    await requirePackageCommercialAccess(ctx, assetRecord.packageId)

    // Update only the PTCs in the evaluation data
    const existingData = (evaluation.data as Record<string, unknown>) || {}
    const updatedData = {
      ...existingData,
      ptcs: data.ptcs as ContractorPTCs[],
    }

    const [updated] = await db
      .update(commercialEvaluation)
      .set({
        data: updatedData,
        updatedAt: new Date(),
      })
      .where(eq(commercialEvaluation.id, data.evaluationId))
      .returning()

    return updated
  })

export const updateCommercialEvaluationDataFn = createServerFn({
  method: "POST",
})
  .inputValidator(
    z.object({
      evaluationId: z.uuid(),
      data: z.any(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    const [evaluation] = await db
      .select()
      .from(commercialEvaluation)
      .where(eq(commercialEvaluation.id, data.evaluationId))
      .limit(1)

    if (!evaluation) throw new Error(ERRORS.NOT_FOUND("Commercial Evaluation"))

    const [assetRecord] = await db.query.asset.findMany({
      where: (asset, { eq }) => eq(asset.id, evaluation.assetId),
      limit: 1,
    })
    if (!assetRecord) throw new Error(ERRORS.NOT_FOUND("Asset"))
    await requirePackageCommercialAccess(ctx, assetRecord.packageId)

    const [updated] = await db
      .update(commercialEvaluation)
      .set({
        data: data.data as Record<string, unknown>,
        updatedAt: new Date(),
      })
      .where(eq(commercialEvaluation.id, data.evaluationId))
      .returning()

    return updated
  })

export const listCommercialEvaluationsFn = createServerFn()
  .inputValidator(z.object({ assetId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Get package from asset to check access
    const [assetRecord] = await db.query.asset.findMany({
      where: (asset, { eq }) => eq(asset.id, data.assetId),
      limit: 1,
    })

    if (!assetRecord) throw new Error(ERRORS.NOT_FOUND("Asset"))
    await requirePackageCommercialAccess(ctx, assetRecord.packageId)

    const evaluations = await db
      .select()
      .from(commercialEvaluation)
      .where(eq(commercialEvaluation.assetId, data.assetId))
      .orderBy(desc(commercialEvaluation.roundNumber))

    return evaluations
  })

export const hasAnyCommercialEvaluationFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageCommercialAccess(ctx, data.packageId)

    // Check if any asset in this package has a commercial evaluation
    const result = await db.query.pkg.findFirst({
      where: (pkg, { eq }) => eq(pkg.id, data.packageId),
      with: {
        assets: {
          with: {
            commercialEvaluations: {
              limit: 1,
            },
          },
        },
      },
    })

    if (!result) return { hasEvaluation: false }

    const hasEvaluation = result.assets.some(
      (asset) => asset.commercialEvaluations.length > 0
    )

    return { hasEvaluation }
  })

export const getPackageCommercialSummaryFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageCommercialAccess(ctx, data.packageId)

    // Get all assets with their latest commercial evaluation
    const result = await db.query.pkg.findFirst({
      where: (pkg, { eq }) => eq(pkg.id, data.packageId),
      with: {
        assets: {
          with: {
            commercialEvaluations: {
              orderBy: (ce, { desc }) => [desc(ce.roundNumber)],
              limit: 1,
            },
          },
        },
      },
    })

    if (!result) return { assets: [] }

    return {
      assets: result.assets.map((asset) => ({
        id: asset.id,
        name: asset.name,
        evaluation: asset.commercialEvaluations[0]?.data ?? null,
      })),
    }
  })
