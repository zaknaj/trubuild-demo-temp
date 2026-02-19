import { db } from "@/db"
import { asset, pkg, proj, packageMember, packageContractor } from "@/db/schema"
import { createServerFn } from "@tanstack/react-start"
import { and, desc, eq, isNotNull, or } from "drizzle-orm"
import { z } from "zod"
import {
  getAuthContext,
  getOrgRole,
  requirePackageAccess,
  requirePackageFullAccess,
  requireProjectFullAccess,
} from "../auth/auth-guards"
import { ERRORS } from "@/lib/errors"

// ============================================================================
// Packages
// ============================================================================

export const createPackageFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      projectId: z.uuid(),
      name: z.string().trim().min(1),
      currency: z.string().optional(),
      technicalWeight: z.number().int().min(0).max(100).default(50),
      commercialWeight: z.number().int().min(0).max(100).default(50),
      contractors: z
        .array(z.string().trim().min(1))
        .min(2, "At least 2 contractors are required"),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_CREATE_PACKAGE
    )

    // Use transaction to ensure atomicity
    return await db.transaction(async (tx) => {
      const [newPackage] = await tx
        .insert(pkg)
        .values({
          name: data.name,
          currency: data.currency,
          projectId: data.projectId,
          technicalWeight: data.technicalWeight,
          commercialWeight: data.commercialWeight,
        })
        .returning({
          id: pkg.id,
          name: pkg.name,
          currency: pkg.currency,
          projectId: pkg.projectId,
        })

      await tx.insert(packageMember).values({
        packageId: newPackage.id,
        userId: ctx.userId,
        email: ctx.userEmail,
        role: "package_lead",
      })

      // Create contractors
      if (data.contractors.length > 0) {
        await tx.insert(packageContractor).values(
          data.contractors.map((name) => ({
            name,
            packageId: newPackage.id,
          }))
        )
      }

      return newPackage
    })
  })

export const getPackageWithAssetsFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [pkgRecord] = await db
      .select({
        id: pkg.id,
        name: pkg.name,
        currency: pkg.currency,
        // stage: pkg.stage,
        // ragStatus: pkg.ragStatus,
        projectId: pkg.projectId,
        awardedContractorId: pkg.awardedContractorId,
      })
      .from(pkg)
      .innerJoin(proj, eq(pkg.projectId, proj.id))
      .where(
        and(
          eq(pkg.id, data.packageId),
          eq(proj.organizationId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!pkgRecord) throw new Error(ERRORS.NOT_FOUND("Package"))

    const [projectRecord] = await db
      .select({
        id: proj.id,
        name: proj.name,
        country: proj.country,
        userId: proj.userId,
        organizationId: proj.organizationId,
      })
      .from(proj)
      .where(eq(proj.id, pkgRecord.projectId))
      .limit(1)

    if (!projectRecord) throw new Error(ERRORS.NOT_FOUND("Parent project"))

    const assetsList = await db
      .select({
        id: asset.id,
        name: asset.name,
        packageId: asset.packageId,
      })
      .from(asset)
      .where(eq(asset.packageId, data.packageId))
      .orderBy(desc(asset.createdAt))

    return {
      package: pkgRecord,
      project: projectRecord,
      assets: assetsList,
    }
  })

// ============================================================================
// Assets
// ============================================================================

export const createAssetFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({ packageId: z.uuid(), name: z.string().trim().min(1) })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [newAsset] = await db
      .insert(asset)
      .values({
        name: data.name,
        packageId: data.packageId,
      })
      .returning({
        id: asset.id,
        name: asset.name,
        packageId: asset.packageId,
      })

    return newAsset
  })

export const getAssetDetailFn = createServerFn()
  .inputValidator(z.object({ assetId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()

    // Single query with joins instead of 3 sequential queries
    const [result] = await db
      .select({
        assetId: asset.id,
        assetName: asset.name,
        packageId: pkg.id,
        packageName: pkg.name,
        projectId: proj.id,
        projectName: proj.name,
      })
      .from(asset)
      .innerJoin(pkg, eq(asset.packageId, pkg.id))
      .innerJoin(proj, eq(pkg.projectId, proj.id))
      .where(eq(asset.id, data.assetId))
      .limit(1)

    if (!result) throw new Error(ERRORS.NOT_FOUND("Asset"))

    await requirePackageAccess(ctx, result.packageId)

    return {
      asset: {
        id: result.assetId,
        name: result.assetName,
        packageId: result.packageId,
      },
      package: {
        id: result.packageId,
        name: result.packageName,
        projectId: result.projectId,
      },
      project: {
        id: result.projectId,
        name: result.projectName,
      },
    }
  })

export const renamePackageFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({ packageId: z.uuid(), name: z.string().trim().min(1) })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_RENAME("package")
    )

    const [updated] = await db
      .update(pkg)
      .set({ name: data.name })
      .where(eq(pkg.id, data.packageId))
      .returning({ id: pkg.id, name: pkg.name })

    return updated
  })

export const updatePackageCurrencyFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ packageId: z.uuid(), currency: z.string() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_RENAME("package")
    )

    const [updated] = await db
      .update(pkg)
      .set({ currency: data.currency })
      .where(eq(pkg.id, data.packageId))
      .returning({ id: pkg.id, currency: pkg.currency })

    return updated
  })

// export const updatePackageStageFn = createServerFn({ method: "POST" })
//   .inputValidator(
//     z.object({
//       packageId: z.uuid(),
//       stage: z.string().nullable(),
//     })
//   )
//   .handler(async ({ data }) => {
//     const ctx = await getAuthContext()
//     await requirePackageFullAccess(
//       ctx,
//       data.packageId,
//       ERRORS.NO_PERMISSION_RENAME("package")
//     )

//     const [updated] = await db
//       .update(pkg)
//       .set({ stage: data.stage })
//       .where(eq(pkg.id, data.packageId))
//       .returning({ id: pkg.id, stage: pkg.stage })

//     return updated
//   })

// export const updatePackageRagStatusFn = createServerFn({ method: "POST" })
//   .inputValidator(
//     z.object({
//       packageId: z.uuid(),
//       ragStatus: z.string().nullable(),
//     })
//   )
//   .handler(async ({ data }) => {
//     const ctx = await getAuthContext()
//     await requirePackageFullAccess(
//       ctx,
//       data.packageId,
//       ERRORS.NO_PERMISSION_RENAME("package")
//     )

//     const [updated] = await db
//       .update(pkg)
//       .set({ ragStatus: data.ragStatus })
//       .where(eq(pkg.id, data.packageId))
//       .returning({ id: pkg.id, ragStatus: pkg.ragStatus })

//     return updated
//   })

export const archivePackageFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_ARCHIVE("package")
    )

    await db
      .update(pkg)
      .set({ archivedAt: new Date() })
      .where(eq(pkg.id, data.packageId))

    return { success: true }
  })

export const restorePackageFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_RESTORE("package")
    )

    await db
      .update(pkg)
      .set({ archivedAt: null })
      .where(eq(pkg.id, data.packageId))

    return { success: true }
  })

export const listArchivedPackagesFn = createServerFn().handler(async () => {
  const ctx = await getAuthContext()
  const orgRole = await getOrgRole(ctx.userId, ctx.activeOrgId)

  if (orgRole === "owner") {
    const packages = await db
      .select({
        id: pkg.id,
        name: pkg.name,
        projectId: pkg.projectId,
        projectName: proj.name,
        archivedAt: pkg.archivedAt,
      })
      .from(pkg)
      .innerJoin(proj, eq(pkg.projectId, proj.id))
      .where(
        and(eq(proj.organizationId, ctx.activeOrgId), isNotNull(pkg.archivedAt))
      )
      .orderBy(desc(pkg.archivedAt))
    return packages
  }

  const packages = await db
    .select({
      id: pkg.id,
      name: pkg.name,
      projectId: pkg.projectId,
      projectName: proj.name,
      archivedAt: pkg.archivedAt,
    })
    .from(pkg)
    .innerJoin(proj, eq(pkg.projectId, proj.id))
    .leftJoin(
      packageMember,
      and(
        eq(packageMember.packageId, pkg.id),
        eq(packageMember.userId, ctx.userId)
      )
    )
    .where(
      and(
        eq(proj.organizationId, ctx.activeOrgId),
        isNotNull(pkg.archivedAt),
        or(eq(proj.userId, ctx.userId), eq(packageMember.userId, ctx.userId))
      )
    )
    .groupBy(pkg.id, proj.name)
    .orderBy(desc(pkg.archivedAt))

  return packages
})

// ============================================================================
// Package Contractors
// ============================================================================

export const createPackageContractorFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({ packageId: z.uuid(), name: z.string().trim().min(1) })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const [newContractor] = await db
      .insert(packageContractor)
      .values({
        name: data.name,
        packageId: data.packageId,
      })
      .returning({
        id: packageContractor.id,
        name: packageContractor.name,
        packageId: packageContractor.packageId,
      })

    return newContractor
  })

export const listPackageContractorsFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const contractors = await db
      .select({
        id: packageContractor.id,
        name: packageContractor.name,
        packageId: packageContractor.packageId,
        createdAt: packageContractor.createdAt,
      })
      .from(packageContractor)
      .where(eq(packageContractor.packageId, data.packageId))
      .orderBy(desc(packageContractor.createdAt))

    return contractors
  })

export const awardPackageFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      contractorId: z.uuid(),
      comments: z.string().optional(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_RENAME("package")
    )

    const [updated] = await db
      .update(pkg)
      .set({
        awardedContractorId: data.contractorId,
        awardComments: data.comments || null,
      })
      .where(eq(pkg.id, data.packageId))
      .returning({ id: pkg.id, awardedContractorId: pkg.awardedContractorId })

    return updated
  })
