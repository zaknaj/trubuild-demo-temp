import { auth } from "@/auth/auth"
import { db } from "@/db"
import {
  pkg,
  proj,
  member,
  projectMember,
  packageMember,
  invitation,
  organization,
} from "@/db/schema"
import { createServerFn } from "@tanstack/react-start"
import { getRequestHeaders } from "@tanstack/react-start/server"
import { and, eq, isNull, isNotNull } from "drizzle-orm"
import { z } from "zod"
import { getAuthContext, requireOrgOwner } from "../auth/auth-guards"
import { getWorkosAuthUrl } from "@/lib/workos"

// ============================================================================
// Session & Organization
// ============================================================================

export const getSessionFn = createServerFn().handler(async () => {
  const headers = getRequestHeaders()
  return auth.api.getSession({ headers })
})

export type AuthBootstrap = {
  session: Awaited<ReturnType<typeof auth.api.getSession>>
  orgs: Awaited<ReturnType<typeof auth.api.listOrganizations>>
}

async function getAuthBootstrap(headers: Headers): Promise<AuthBootstrap> {
  const session = await auth.api.getSession({ headers })
  if (!session) return { session: null, orgs: [] }
  const orgs = await auth.api.listOrganizations({ headers })
  return { session, orgs }
}

/** Fetch auth-related bootstrap data (session + orgs) in one call */
export const getAuthBootstrapFn = createServerFn().handler(async () => {
  const headers = getRequestHeaders()
  return getAuthBootstrap(headers)
})

/** Call after login to link pending project/package memberships */
export const linkPendingMembershipsFn = createServerFn({
  method: "POST",
}).handler(async () => {
  const ctx = await getAuthContext(false)
  await linkPendingMemberships(ctx.userId, ctx.userEmail)
})

/** Link pending project/package memberships and add user to orgs */
async function linkPendingMemberships(userId: string, email: string) {
  // Use transaction to ensure all membership operations are atomic
  await db.transaction(async (tx) => {
    // Accept pending organization invitations
    const pendingInvitations = await tx
      .select({
        id: invitation.id,
        organizationId: invitation.organizationId,
        role: invitation.role,
      })
      .from(invitation)
      .where(and(eq(invitation.email, email), eq(invitation.status, "pending")))

    for (const inv of pendingInvitations) {
      // Add user as member to the organization
      await tx
        .insert(member)
        .values({
          id: globalThis.crypto.randomUUID(),
          userId,
          organizationId: inv.organizationId,
          role: inv.role ?? "member",
          createdAt: new Date(),
        })
        .onConflictDoNothing()

      // Mark invitation as accepted
      await tx
        .update(invitation)
        .set({ status: "accepted" })
        .where(eq(invitation.id, inv.id))
    }

    // Link pending project memberships
    await tx
      .update(projectMember)
      .set({ userId })
      .where(and(eq(projectMember.email, email), isNull(projectMember.userId)))

    // Link pending package memberships
    await tx
      .update(packageMember)
      .set({ userId })
      .where(and(eq(packageMember.email, email), isNull(packageMember.userId)))

    // Find all orgs from linked projects
    const projectOrgs = await tx
      .selectDistinct({ organizationId: proj.organizationId })
      .from(projectMember)
      .innerJoin(proj, eq(projectMember.projectId, proj.id))
      .where(eq(projectMember.userId, userId))

    // Find all orgs from linked packages
    const packageOrgs = await tx
      .selectDistinct({ organizationId: proj.organizationId })
      .from(packageMember)
      .innerJoin(pkg, eq(packageMember.packageId, pkg.id))
      .innerJoin(proj, eq(pkg.projectId, proj.id))
      .where(eq(packageMember.userId, userId))

    // Add user as member to each org they have project/package access to
    const orgIds = new Set([
      ...projectOrgs.map((o) => o.organizationId),
      ...packageOrgs.map((o) => o.organizationId),
    ])

    for (const organizationId of orgIds) {
      await tx
        .insert(member)
        .values({
          id: globalThis.crypto.randomUUID(),
          userId,
          organizationId,
          role: "member",
          createdAt: new Date(),
        })
        .onConflictDoNothing()
    }
  })
}

export const getOrgsFn = createServerFn().handler(async () => {
  const headers = getRequestHeaders()
  return (await getAuthBootstrap(headers)).orgs
})

export const setActiveOrgFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ organizationId: z.string() }))
  .handler(async ({ data }) => {
    const headers = getRequestHeaders()
    const session = await auth.api.getSession({ headers })

    if (!session?.user?.id) {
      throw new Error("Must be logged in")
    }

    // Verify user is a member of the organization before allowing switch
    const [orgMembership] = await db
      .select({ role: member.role })
      .from(member)
      .where(
        and(
          eq(member.userId, session.user.id),
          eq(member.organizationId, data.organizationId)
        )
      )
      .limit(1)

    if (!orgMembership) {
      throw new Error("You are not a member of this organization")
    }

    await auth.api.setActiveOrganization({
      headers,
      body: { organizationId: data.organizationId },
    })
  })

// Sets org creator role to "owner" - only allowed if no other owners exist
export const setOrgCreatorAsOwnerFn = createServerFn({ method: "POST" })
  .inputValidator(z.object({ organizationId: z.string() }))
  .handler(async ({ data }) => {
    const ctx = await getAuthContext(false)

    // Verify user is a member of this organization
    const [userMembership] = await db
      .select({ role: member.role })
      .from(member)
      .where(
        and(
          eq(member.userId, ctx.userId),
          eq(member.organizationId, data.organizationId)
        )
      )
      .limit(1)

    if (!userMembership) {
      throw new Error("You are not a member of this organization")
    }

    // If already owner, nothing to do
    if (userMembership.role === "owner") {
      return { success: true }
    }

    // Only allow promotion to owner if there are no existing owners
    // This prevents any member from making themselves owner
    const [existingOwner] = await db
      .select({ userId: member.userId })
      .from(member)
      .where(
        and(
          eq(member.organizationId, data.organizationId),
          eq(member.role, "owner")
        )
      )
      .limit(1)

    if (existingOwner) {
      throw new Error("Organization already has an owner")
    }

    await db
      .update(member)
      .set({ role: "owner" })
      .where(
        and(
          eq(member.userId, ctx.userId),
          eq(member.organizationId, data.organizationId)
        )
      )
    return { success: true }
  })

export const updateOrganizationFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      name: z.string().trim().min(1).optional(),
      logo: z.string().optional(),
      metadata: z.object({ country: z.string().optional() }).optional(),
    })
  )
  .handler(async ({ data }) => {
    const ctx = await getAuthContext()
    await requireOrgOwner(ctx)
    const headers = getRequestHeaders()

    const updateData: {
      name?: string
      logo?: string
      metadata?: { country?: string }
    } = {}
    if (data.name) updateData.name = data.name
    if (data.logo) updateData.logo = data.logo
    if (data.metadata) updateData.metadata = data.metadata

    if (Object.keys(updateData).length === 0) {
      return { success: true }
    }

    await auth.api.updateOrganization({
      headers,
      body: {
        organizationId: ctx.activeOrgId,
        data: updateData,
      },
    })

    return { success: true }
  })

export const updateProfileFn = createServerFn({ method: "POST" })
  .inputValidator(
    z.object({
      name: z.string().trim().min(1).optional(),
      image: z.string().optional(),
    })
  )
  .handler(async ({ data }) => {
    // Explicit auth check
    await getAuthContext(false)
    const headers = getRequestHeaders()

    const updateData: { name?: string; image?: string } = {}
    if (data.name) updateData.name = data.name
    if (data.image) updateData.image = data.image

    if (Object.keys(updateData).length === 0) {
      return { success: true }
    }

    await auth.api.updateUser({
      headers,
      body: updateData,
    })

    return { success: true }
  })

// ============================================================================
// SSO (WorkOS)
// ============================================================================

/** Check if an email domain has WorkOS SSO configured */
export const checkSSOForEmailFn = createServerFn()
  .inputValidator(z.object({ email: z.string().email() }))
  .handler(async ({ data }) => {
    // Only check SSO in enterprise mode
    if ((process.env.AUTH_MODE ?? "dev") !== "enterprise") {
      return { hasSSO: false as const }
    }

    const domain = data.email.split("@")[1]

    // Find an organization with a WorkOS connection that uses this domain
    // For now, we store the domain in the org's metadata or match by slug
    // In a more sophisticated setup, you'd have a separate domain mapping table
    const [org] = await db
      .select({
        workosConnectionId: organization.workosConnectionId,
        id: organization.id,
      })
      .from(organization)
      .where(isNotNull(organization.workosConnectionId))
      .limit(1)

    // TODO: In production, match domain to organization more precisely
    // For now, if any org has WorkOS configured and we're in enterprise mode,
    // we'll use it. You could also store allowed domains in org metadata.

    if (org?.workosConnectionId) {
      return {
        hasSSO: true as const,
        connectionId: org.workosConnectionId,
        organizationId: org.id,
      }
    }

    return { hasSSO: false as const }
  })

/** Get WorkOS authorization URL for SSO redirect */
export const getWorkosAuthUrlFn = createServerFn()
  .inputValidator(z.object({ connectionId: z.string() }))
  .handler(async ({ data }) => {
    const betterAuthUrl = process.env.BETTER_AUTH_URL
    if (!betterAuthUrl) {
      throw new Error("Missing required environment variable: BETTER_AUTH_URL")
    }
    const redirectUri = `${betterAuthUrl}/api/workos/callback`
    const url = getWorkosAuthUrl(data.connectionId, redirectUri)
    return { url }
  })
