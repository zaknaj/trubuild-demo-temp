import { auth } from "@/auth/auth"
import { db } from "@/db"
import {
  member,
  user,
  projectMember,
  packageMember,
  invitation,
} from "@/db/schema"
import { createServerFn } from "@tanstack/react-start"
import { getRequestHeaders } from "@tanstack/react-start/server"
import { and, eq } from "drizzle-orm"
import { z } from "zod"
import { getProjectAccess, getPackageAccess } from "@/lib/permissions"
import {
  getAuthContext,
  getOrgRole,
  requireOrgOwner,
  requireProjectAccess,
  requireProjectFullAccess,
  requirePackageAccess,
  requirePackageFullAccess,
} from "../auth/auth-guards"
import { ERRORS } from "@/lib/errors"

// ============================================================================
// Organization Members & Invitations
// ============================================================================

export const getCurrentUserOrgRoleFn = createServerFn().handler(async () => {
  const ctx = await getAuthContext()
  const role = await getOrgRole(ctx.userId, ctx.activeOrgId)
  return { role }
})

export const getOrgMembersFn = createServerFn().handler(async () => {
  const ctx = await getAuthContext()

  // Verify user is a member of this organization
  const [userMembership] = await db
    .select({ id: member.id })
    .from(member)
    .where(
      and(
        eq(member.userId, ctx.userId),
        eq(member.organizationId, ctx.activeOrgId)
      )
    )
    .limit(1)

  if (!userMembership) {
    throw new Error(ERRORS.NO_ACCESS("organization"))
  }

  const members = await db
    .select({
      id: member.id,
      role: member.role,
      userId: member.userId,
      userName: user.name,
      email: user.email,
      userImage: user.image,
    })
    .from(member)
    .innerJoin(user, eq(member.userId, user.id))
    .where(eq(member.organizationId, ctx.activeOrgId))

  return members
})

export const getOrgPendingInvitesFn = createServerFn().handler(async () => {
  const ctx = await getAuthContext()

  // Verify user is a member of this organization
  const [userMembership] = await db
    .select({ id: member.id })
    .from(member)
    .where(
      and(
        eq(member.userId, ctx.userId),
        eq(member.organizationId, ctx.activeOrgId)
      )
    )
    .limit(1)

  if (!userMembership) {
    throw new Error(ERRORS.NO_ACCESS("organization"))
  }

  const pending = await db
    .select({
      id: invitation.id,
      email: invitation.email,
      role: invitation.role,
      createdAt: invitation.createdAt,
    })
    .from(invitation)
    .where(
      and(
        eq(invitation.organizationId, ctx.activeOrgId),
        eq(invitation.status, "pending")
      )
    )

  return pending
})

export const inviteMemberFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      email: z.email(),
      role: z.enum(["owner", "admin", "member"]),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireOrgOwner(ctx)
    const headers = getRequestHeaders()

    await auth.api.createInvitation({
      headers,
      body: {
        email: data.email,
        role: data.role,
        organizationId: ctx.activeOrgId,
      },
    })
    return { success: true, email: data.email, role: data.role }
  })

export const removeOrgMemberFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      userId: z.string(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireOrgOwner(ctx)

    // Prevent removing yourself
    if (data.userId === ctx.userId) {
      throw new Error("You cannot remove yourself from the organization.")
    }

    // Check if this is the last owner
    const owners = await db
      .select({ userId: member.userId })
      .from(member)
      .where(
        and(
          eq(member.organizationId, ctx.activeOrgId),
          eq(member.role, "owner")
        )
      )

    const memberToRemove = owners.find((o) => o.userId === data.userId)
    if (memberToRemove && owners.length === 1) {
      throw new Error("Cannot remove the last owner from the organization.")
    }

    await db
      .delete(member)
      .where(
        and(
          eq(member.userId, data.userId),
          eq(member.organizationId, ctx.activeOrgId)
        )
      )

    return { success: true }
  })

export const cancelOrgInvitationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      invitationId: z.string(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireOrgOwner(ctx)

    await db
      .delete(invitation)
      .where(
        and(
          eq(invitation.id, data.invitationId),
          eq(invitation.organizationId, ctx.activeOrgId),
          eq(invitation.status, "pending")
        )
      )

    return { success: true }
  })

// ============================================================================
// Project Members & Invitations
// ============================================================================

export const getProjectMembersFn = createServerFn()
  .inputValidator(z.object({ projectId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectAccess(ctx, data.projectId)

    const members = await db
      .select({
        role: projectMember.role,
        userId: projectMember.userId,
        email: projectMember.email,
        userName: user.name,
        userImage: user.image,
      })
      .from(projectMember)
      .leftJoin(user, eq(projectMember.userId, user.id))
      .where(eq(projectMember.projectId, data.projectId))

    return members.map((m) => ({
      ...m,
      id: `${data.projectId}-${m.email}`,
    }))
  })

export const getProjectAccessFn = createServerFn()
  .inputValidator(z.object({ projectId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    const {
      access,
      orgRole,
      projectRole,
      packageRole,
      isCreator,
      hasProjectLevelAccess,
    } = await getProjectAccess(ctx.userId, data.projectId, ctx.activeOrgId)
    return {
      access,
      orgRole,
      projectRole,
      packageRole,
      isCreator,
      hasProjectLevelAccess,
    }
  })

export const addProjectMemberFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      projectId: z.uuid(),
      email: z.email(),
      role: z.enum(["project_lead", "commercial_lead", "technical_lead"]),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_INVITE("project")
    )

    // Check if user already exists
    const [existingUser] = await db
      .select({ id: user.id })
      .from(user)
      .where(eq(user.email, data.email))
      .limit(1)

    await db
      .insert(projectMember)
      .values({
        projectId: data.projectId,
        email: data.email,
        userId: existingUser?.id ?? null,
        role: data.role,
      })
      .onConflictDoUpdate({
        target: [projectMember.projectId, projectMember.email],
        set: { role: data.role, userId: existingUser?.id ?? null },
      })

    return { success: true }
  })

export const removeProjectMemberFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      projectId: z.uuid(),
      email: z.email(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireProjectFullAccess(
      ctx,
      data.projectId,
      ERRORS.NO_PERMISSION_REMOVE("project")
    )

    await db
      .delete(projectMember)
      .where(
        and(
          eq(projectMember.projectId, data.projectId),
          eq(projectMember.email, data.email)
        )
      )

    return { success: true }
  })

// ============================================================================
// Package Members & Invitations
// ============================================================================

export const getPackageMembersFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageAccess(ctx, data.packageId)

    const members = await db
      .select({
        role: packageMember.role,
        userId: packageMember.userId,
        email: packageMember.email,
        userName: user.name,
        userImage: user.image,
      })
      .from(packageMember)
      .leftJoin(user, eq(packageMember.userId, user.id))
      .where(eq(packageMember.packageId, data.packageId))

    return members.map((m) => ({
      ...m,
      id: `${data.packageId}-${m.email}`,
    }))
  })

export const getPackageAccessFn = createServerFn()
  .inputValidator(z.object({ packageId: z.uuid() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    const { access, orgRole, projectRole, packageRole, isProjectCreator } =
      await getPackageAccess(ctx.userId, data.packageId, ctx.activeOrgId)
    return { access, orgRole, projectRole, packageRole, isProjectCreator }
  })

export const addPackageMemberFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      email: z.email(),
      role: z.enum(["package_lead", "commercial_team", "technical_team"]),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_INVITE("package")
    )

    // Check if user already exists
    const [existingUser] = await db
      .select({ id: user.id })
      .from(user)
      .where(eq(user.email, data.email))
      .limit(1)

    await db
      .insert(packageMember)
      .values({
        packageId: data.packageId,
        email: data.email,
        userId: existingUser?.id ?? null,
        role: data.role,
      })
      .onConflictDoUpdate({
        target: [packageMember.packageId, packageMember.email],
        set: { role: data.role, userId: existingUser?.id ?? null },
      })

    return { success: true }
  })

export const removePackageMemberFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      packageId: z.uuid(),
      email: z.email(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requirePackageFullAccess(
      ctx,
      data.packageId,
      ERRORS.NO_PERMISSION_REMOVE("package")
    )

    await db
      .delete(packageMember)
      .where(
        and(
          eq(packageMember.packageId, data.packageId),
          eq(packageMember.email, data.email)
        )
      )

    return { success: true }
  })
