import { auth } from "@/auth/auth"
import { getRequestHeaders } from "@tanstack/react-start/server"
import { ERRORS } from "@/lib/errors"
import { db } from "@/db"
import { member } from "@/db/schema"
import { and, eq } from "drizzle-orm"
import { getPackageAccess, getProjectAccess } from "@/lib/permissions"

export type AuthContext = {
  userId: string
  activeOrgId: string
  userEmail: string
}

export type AuthContextNoOrg = {
  userId: string
  userEmail: string
  activeOrgId: string | null
}

/**
 * Get authenticated context
 * @param requireOrg - If true (default), requires an active organization
 */
export async function getAuthContext<T extends boolean = true>(
  requireOrg?: T
): Promise<T extends false ? AuthContextNoOrg : AuthContext> {
  const headers = getRequestHeaders()
  const session = await auth.api.getSession({ headers })

  const userId = session?.user?.id
  const userEmail = session?.user?.email
  const activeOrgId = session?.session?.activeOrganizationId ?? null

  if (!userId || !userEmail) {
    throw new Error(ERRORS.MUST_BE_LOGGED_IN)
  }

  if (requireOrg !== false && !activeOrgId) {
    throw new Error(ERRORS.MUST_BE_LOGGED_IN_WITH_ORG)
  }

  return { userId, userEmail, activeOrgId } as T extends false
    ? AuthContextNoOrg
    : AuthContext
}

/**
 * Get org role for a user in an organization
 */
export async function getOrgRole(
  userId: string,
  orgId: string
): Promise<string | null> {
  const [orgMember] = await db
    .select({ role: member.role })
    .from(member)
    .where(and(eq(member.userId, userId), eq(member.organizationId, orgId)))
    .limit(1)
  return orgMember?.role ?? null
}

/**
 * Require org owner role - for org-level admin operations (edit org, invite members)
 */
export async function requireOrgOwner(ctx: AuthContext) {
  const role = await getOrgRole(ctx.userId, ctx.activeOrgId)
  if (role !== "owner") {
    throw new Error(ERRORS.NO_PERMISSION_ADMIN)
  }
  return { role }
}

/**
 * Require owner or admin role - for creating projects
 */
export async function requireCanCreateProject(ctx: AuthContext) {
  const role = await getOrgRole(ctx.userId, ctx.activeOrgId)
  if (role !== "owner" && role !== "admin") {
    throw new Error(ERRORS.NO_PERMISSION_CREATE_PROJECT)
  }
  return { role }
}

/**
 * Require any (non-none) access to a project.
 */
export async function requireProjectAccess(
  ctx: AuthContext,
  projectId: string
) {
  const accessInfo = await getProjectAccess(
    ctx.userId,
    projectId,
    ctx.activeOrgId
  )
  if (accessInfo.access === "none") throw new Error(ERRORS.NO_ACCESS("project"))
  return accessInfo
}

/**
 * Require full access to a project.
 */
export async function requireProjectFullAccess(
  ctx: AuthContext,
  projectId: string,
  errorMessage?: string
) {
  const accessInfo = await getProjectAccess(
    ctx.userId,
    projectId,
    ctx.activeOrgId
  )
  if (accessInfo.access !== "full")
    throw new Error(errorMessage ?? ERRORS.NO_PERMISSION_INVITE("project"))
  return accessInfo
}

/**
 * Require any (non-none) access to a package.
 */
export async function requirePackageAccess(
  ctx: AuthContext,
  packageId: string
) {
  const accessInfo = await getPackageAccess(
    ctx.userId,
    packageId,
    ctx.activeOrgId
  )
  if (accessInfo.access === "none") throw new Error(ERRORS.NO_ACCESS("package"))
  return accessInfo
}

/**
 * Require full access to a package.
 */
export async function requirePackageFullAccess(
  ctx: AuthContext,
  packageId: string,
  errorMessage?: string
) {
  const accessInfo = await getPackageAccess(
    ctx.userId,
    packageId,
    ctx.activeOrgId
  )
  if (accessInfo.access !== "full")
    throw new Error(errorMessage ?? ERRORS.NO_PERMISSION_INVITE("package"))
  return accessInfo
}

/**
 * Require technical access to a package.
 * Allowed: full, technical
 */
export async function requirePackageTechnicalAccess(
  ctx: AuthContext,
  packageId: string
) {
  const accessInfo = await getPackageAccess(
    ctx.userId,
    packageId,
    ctx.activeOrgId
  )
  if (accessInfo.access !== "full" && accessInfo.access !== "technical") {
    throw new Error(ERRORS.NO_TECHNICAL_ACCESS)
  }
  return accessInfo
}

/**
 * Require commercial access to a package.
 * Allowed: full, commercial
 */
export async function requirePackageCommercialAccess(
  ctx: AuthContext,
  packageId: string
) {
  const accessInfo = await getPackageAccess(
    ctx.userId,
    packageId,
    ctx.activeOrgId
  )
  if (accessInfo.access !== "full" && accessInfo.access !== "commercial") {
    throw new Error(ERRORS.NO_COMMERCIAL_ACCESS)
  }
  return accessInfo
}
