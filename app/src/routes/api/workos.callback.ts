import { createFileRoute, redirect } from "@tanstack/react-router"
import { db } from "@/db"
import { user, account, session, member, organization } from "@/db/schema"
import { eq, and } from "drizzle-orm"
import { getWorkosProfile } from "@/lib/workos"

export const Route = createFileRoute("/api/workos/callback")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        const url = new URL(request.url)
        const code = url.searchParams.get("code")

        if (!code) {
          return new Response("Missing authorization code", { status: 400 })
        }

        try {
          // Exchange code for profile
          const { profile } = await getWorkosProfile(code)

          // Find or create user
          let [existingUser] = await db
            .select()
            .from(user)
            .where(eq(user.email, profile.email))
            .limit(1)

          const userId = existingUser?.id ?? crypto.randomUUID()

          if (!existingUser) {
            // Create new user
            await db.insert(user).values({
              id: userId,
              email: profile.email,
              name:
                profile.firstName && profile.lastName
                  ? `${profile.firstName} ${profile.lastName}`
                  : profile.email.split("@")[0],
              emailVerified: true, // SSO users are verified
              createdAt: new Date(),
              updatedAt: new Date(),
            })
          }

          // Create or update WorkOS account link
          const workosAccountId = `workos:${profile.id}`
          const [existingAccount] = await db
            .select()
            .from(account)
            .where(
              and(eq(account.userId, userId), eq(account.providerId, "workos"))
            )
            .limit(1)

          if (!existingAccount) {
            await db.insert(account).values({
              id: crypto.randomUUID(),
              userId,
              providerId: "workos",
              accountId: profile.id,
              createdAt: new Date(),
              updatedAt: new Date(),
            })
          }

          // Find the organization this SSO connection belongs to
          const [org] = await db
            .select()
            .from(organization)
            .where(eq(organization.workosOrgId, profile.organizationId ?? ""))
            .limit(1)

          // Add user to organization if found and not already a member
          if (org) {
            const [existingMember] = await db
              .select()
              .from(member)
              .where(
                and(
                  eq(member.userId, userId),
                  eq(member.organizationId, org.id)
                )
              )
              .limit(1)

            if (!existingMember) {
              await db.insert(member).values({
                id: crypto.randomUUID(),
                userId,
                organizationId: org.id,
                role: "member",
                createdAt: new Date(),
              })
            }
          }

          // Create session
          const sessionToken = crypto.randomUUID()
          const sessionId = crypto.randomUUID()
          const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days

          await db.insert(session).values({
            id: sessionId,
            userId,
            token: sessionToken,
            expiresAt,
            createdAt: new Date(),
            updatedAt: new Date(),
            activeOrganizationId: org?.id ?? null,
          })

          // Set session cookie and redirect
          const response = new Response(null, {
            status: 302,
            headers: {
              Location: "/",
              "Set-Cookie": `better-auth.session_token=${sessionToken}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${30 * 24 * 60 * 60}`,
            },
          })

          return response
        } catch (error) {
          console.error("WorkOS callback error:", error)
          return new Response("Authentication failed", { status: 500 })
        }
      },
    },
  },
})
