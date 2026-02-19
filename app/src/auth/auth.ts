import { betterAuth } from "better-auth"
import { drizzleAdapter } from "better-auth/adapters/drizzle"
import { tanstackStartCookies } from "better-auth/tanstack-start"
import { organization, magicLink } from "better-auth/plugins"
import { Resend } from "resend"

import { db } from "@/db"
import * as schema from "@/db/schema"

const authMode = process.env.AUTH_MODE ?? "dev"
const isDev = authMode === "dev"

// Lazy initialization of Resend client
let _resend: Resend | null = null
function getResend() {
  if (!_resend) {
    const apiKey = process.env.RESEND_API_KEY
    if (!apiKey) {
      throw new Error("Missing required environment variable: RESEND_API_KEY")
    }
    _resend = new Resend(apiKey)
  }
  return _resend
}

export const auth = betterAuth({
  secret: process.env.BETTER_AUTH_SECRET,
  database: drizzleAdapter(db, {
    provider: "pg",
    schema,
  }),
  emailAndPassword: {
    enabled: isDev, // Only for dev mode
  },
  plugins: [
    organization(),
    // Magic link for production/enterprise modes
    ...(isDev
      ? []
      : [
          magicLink({
            sendMagicLink: async ({ email, url }) => {
              const resend = getResend()
              await resend.emails.send({
                from: "TruBuild <noreply@trubuild.com>",
                to: email,
                subject: "Sign in to TruBuild",
                html: `
                  <h2>Sign in to TruBuild</h2>
                  <p>Click the link below to sign in:</p>
                  <a href="${url}" style="display:inline-block;padding:12px 24px;background:#2563eb;color:white;text-decoration:none;border-radius:6px;">Sign in</a>
                  <p style="margin-top:16px;color:#666;">This link expires in 5 minutes.</p>
                `,
              })
            },
          }),
        ]),
    tanstackStartCookies(), // Always last!
  ],
})
