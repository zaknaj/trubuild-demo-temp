import { db } from "@/db"
import {
  asset,
  pkg,
  proj,
  projectMember,
  packageMember,
  packageContractor,
  user,
} from "@/db/schema"
import { createServerFn } from "@tanstack/react-start"
import {
  and,
  count,
  desc,
  eq,
  inArray,
  isNotNull,
  isNull,
  or,
} from "drizzle-orm"
import { z } from "zod"
import {
  getAuthContext,
  getOrgRole,
  requireCanCreateProject,
  requireProjectAccess,
  requireProjectFullAccess,
} from "../auth/auth-guards"
import { ERRORS } from "@/lib/errors"

// ============================================================================
// Projects
// ============================================================================

export const listProjectsFn = createServerFn().handler(async () => {
  const ctx = await getAuthContext()
  const orgRole = await getOrgRole(ctx.userId, ctx.activeOrgId)

  let baseProjects: {
    id: string
    name: string
    country: string | null
    userId: string
    organizationId: string
  }[]

  if (orgRole === "owner") {
    baseProjects = await db
      .select({
        id: proj.id,
        name: proj.name,
        country: proj.country,
        userId: proj.userId,
        organizationId: proj.organizationId,
      })
      .from(proj)
      .where(
        and(eq(proj.organizationId, ctx.activeOrgId), isNull(proj.archivedAt))
      )
      .orderBy(desc(proj.createdAt))
  } else {
    // For non-owners, show projects where user is:
    // 1. The project creator
    // 2. A project member
    // 3. A member of any package within the project
    baseProjects = await db
      .select({
        id: proj.id,
        name: proj.name,
        country: proj.country,
        userId: proj.userId,
        organizationId: proj.organizationId,
      })
      .from(proj)
      .leftJoin(
        projectMember,
        and(
          eq(projectMember.projectId, proj.id),
          eq(projectMember.userId, ctx.userId)
        )
      )
      .leftJoin(pkg, eq(pkg.projectId, proj.id))
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
          isNull(proj.archivedAt),
          or(
            eq(proj.userId, ctx.userId),
            eq(projectMember.userId, ctx.userId),
            eq(packageMember.userId, ctx.userId)
          )
        )
      )
      .groupBy(proj.id)
      .orderBy(desc(proj.createdAt))
  }

  if (baseProjects.length === 0) {
    return []
  }

  const projectIds = baseProjects.map((p) => p.id)

  // Get packages for all projects (for sidebar expansion)
  const allPackages = await db
    .select({
      id: pkg.id,
      name: pkg.name,
      projectId: pkg.projectId,
      awardedContractorId: pkg.awardedContractorId,
    })
    .from(pkg)
    .where(and(inArray(pkg.projectId, projectIds), isNull(pkg.archivedAt)))
    .orderBy(desc(pkg.createdAt))

  const packagesMap = new Map<
    string,
    { id: string; name: string; awardedContractorId: string | null }[]
  >()
  for (const p of allPackages) {
    const existing = packagesMap.get(p.projectId) || []
    existing.push({
      id: p.id,
      name: p.name,
      awardedContractorId: p.awardedContractorId,
    })
    packagesMap.set(p.projectId, existing)
  }

  // Get team members per project (with user info)
  const teamMembers = await db
    .select({
      projectId: projectMember.projectId,
      email: projectMember.email,
      role: projectMember.role,
      userName: user.name,
      userImage: user.image,
    })
    .from(projectMember)
    .leftJoin(user, eq(projectMember.userId, user.id))
    .where(inArray(projectMember.projectId, projectIds))

  const teamMembersMap = new Map<
    string,
    { email: string; role: string; name: string | null; image: string | null }[]
  >()
  for (const tm of teamMembers) {
    const existing = teamMembersMap.get(tm.projectId) || []
    existing.push({
      email: tm.email,
      role: tm.role,
      name: tm.userName,
      image: tm.userImage,
    })
    teamMembersMap.set(tm.projectId, existing)
  }

  // Combine data
  return baseProjects.map((project) => {
    const projectPackages = packagesMap.get(project.id) ?? []
    const awardedCount = projectPackages.filter(
      (p) => p.awardedContractorId !== null
    ).length
    return {
      ...project,
      packages: projectPackages.map(({ id, name }) => ({ id, name })),
      packageCount: projectPackages.length,
      awardedPackageCount: awardedCount,
      teamMembers: teamMembersMap.get(project.id) ?? [],
    }
  })
})

export const createProjectFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({ name: z.string().trim().min(1), country: z.string().optional() })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireCanCreateProject(ctx)

    // Use transaction to ensure atomicity
    return await db.transaction(async (tx) => {
      const [project] = await tx
        .insert(proj)
        .values({
          name: data.name,
          country: data.country,
          userId: ctx.userId,
          organizationId: ctx.activeOrgId,
        })
        .returning({
          id: proj.id,
          name: proj.name,
          country: proj.country,
          userId: proj.userId,
          organizationId: proj.organizationId,
        })

      await tx.insert(projectMember).values({
        projectId: project.id,
        userId: ctx.userId,
        email: ctx.userEmail,
        role: "project_lead",
      })

      return project
    })
  })

export const getProjectWithPackagesFn = createServerFn()
  .inputValidator(z.object({ projectId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    const accessInfo = await requireProjectAccess(ctx, data.projectId)

    // Check if user has commercial access at project level
    const canViewCommercial =
      accessInfo.access === "full" || accessInfo.access === "commercial"

    const [project] = await db
      .select({
        id: proj.id,
        name: proj.name,
        country: proj.country,
        userId: proj.userId,
        organizationId: proj.organizationId,
      })
      .from(proj)
      .where(
        and(
          eq(proj.id, data.projectId),
          eq(proj.organizationId, ctx.activeOrgId)
        )
      )
      .limit(1)

    if (!project) throw new Error(ERRORS.NOT_FOUND("Project"))

    let packages: {
      id: string
      name: string
      currency: string | null
      // stage: string | null
      // ragStatus: string | null
      projectId: string
      awardedContractorId: string | null
    }[]

    // If user has project-level access, show all packages
    // Otherwise, only show packages they are a member of
    if (accessInfo.hasProjectLevelAccess) {
      packages = await db
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
        .where(and(eq(pkg.projectId, data.projectId), isNull(pkg.archivedAt)))
        .orderBy(desc(pkg.createdAt))
    } else {
      // Only show packages user is a member of
      packages = await db
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
        .innerJoin(
          packageMember,
          and(
            eq(packageMember.packageId, pkg.id),
            eq(packageMember.userId, ctx.userId)
          )
        )
        .where(and(eq(pkg.projectId, data.projectId), isNull(pkg.archivedAt)))
        .orderBy(desc(pkg.createdAt))
    }

    // Get asset counts per package
    const packageIds = packages.map((p) => p.id)
    let assetCountMap = new Map<string, number>()

    if (packageIds.length > 0) {
      const assetCounts = await db
        .select({
          packageId: asset.packageId,
          count: count(),
        })
        .from(asset)
        .where(inArray(asset.packageId, packageIds))
        .groupBy(asset.packageId)

      assetCountMap = new Map(assetCounts.map((a) => [a.packageId, a.count]))
    }

    // Get awarded contractor names only if user has commercial access
    let contractorNameMap = new Map<string, string>()
    if (canViewCommercial) {
      const awardedContractorIds = packages
        .map((p) => p.awardedContractorId)
        .filter((id): id is string => id !== null)

      if (awardedContractorIds.length > 0) {
        const contractors = await db
          .select({
            id: packageContractor.id,
            name: packageContractor.name,
          })
          .from(packageContractor)
          .where(inArray(packageContractor.id, awardedContractorIds))

        contractorNameMap = new Map(contractors.map((c) => [c.id, c.name]))
      }
    }

    return {
      project,
      packages: packages.map((p) => ({
        ...p,
        assetCount: assetCountMap.get(p.id) ?? 0,
        // Only return awarded contractor info if user has commercial access
        awardedContractorId: canViewCommercial ? p.awardedContractorId : null,
        awardedContractorName:
          canViewCommercial && p.awardedContractorId
            ? (contractorNameMap.get(p.awardedContractorId) ?? null)
            : null,
      })),
    }
  })

export const renameProjectFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({ projectId: z.uuid(), name: z.string().trim().min(1) })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_RENAME("project")
    )

    const [updated] = await db
      .update(proj)
      .set({ name: data.name })
      .where(eq(proj.id, data.projectId))
      .returning({ id: proj.id, name: proj.name })

    return updated
  })

export const updateProjectCountryFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ projectId: z.uuid(), country: z.string() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_RENAME("project")
    )

    const [updated] = await db
      .update(proj)
      .set({ country: data.country })
      .where(eq(proj.id, data.projectId))
      .returning({ id: proj.id, country: proj.country })

    return updated
  })

export const archiveProjectFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ projectId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_ARCHIVE("project")
    )

    await db
      .update(proj)
      .set({ archivedAt: new Date() })
      .where(eq(proj.id, data.projectId))

    return { success: true }
  })

export const restoreProjectFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ projectId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_RESTORE("project")
    )

    await db
      .update(proj)
      .set({ archivedAt: null })
      .where(eq(proj.id, data.projectId))

    return { success: true }
  })

export const listArchivedProjectsFn = createServerFn().handler(async () => {
  const ctx = await getAuthContext()
  const orgRole = await getOrgRole(ctx.userId, ctx.activeOrgId)

  if (orgRole === "owner") {
    const projects = await db
      .select({
        id: proj.id,
        name: proj.name,
        userId: proj.userId,
        organizationId: proj.organizationId,
        archivedAt: proj.archivedAt,
      })
      .from(proj)
      .where(
        and(
          eq(proj.organizationId, ctx.activeOrgId),
          isNotNull(proj.archivedAt)
        )
      )
      .orderBy(desc(proj.archivedAt))
    return projects
  }

  // For non-owners, show archived projects where user is:
  // 1. The project creator
  // 2. A project member
  // 3. A member of any package within the project
  const projects = await db
    .select({
      id: proj.id,
      name: proj.name,
      userId: proj.userId,
      organizationId: proj.organizationId,
      archivedAt: proj.archivedAt,
    })
    .from(proj)
    .leftJoin(
      projectMember,
      and(
        eq(projectMember.projectId, proj.id),
        eq(projectMember.userId, ctx.userId)
      )
    )
    .leftJoin(pkg, eq(pkg.projectId, proj.id))
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
        isNotNull(proj.archivedAt),
        or(
          eq(proj.userId, ctx.userId),
          eq(projectMember.userId, ctx.userId),
          eq(packageMember.userId, ctx.userId)
        )
      )
    )
    .groupBy(proj.id)
    .orderBy(desc(proj.archivedAt))

  return projects
})
