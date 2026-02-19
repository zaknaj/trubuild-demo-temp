import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import { createMiddleware, createServerFn } from "@tanstack/react-start"
import { useState } from "react"
import {
  getAuthBootstrapFn,
  getOrgsFn,
  linkPendingMembershipsFn,
  setActiveOrgFn,
  checkSSOForEmailFn,
  getWorkosAuthUrlFn,
} from "@/fn/auth"
import { authClient } from "@/auth/auth-client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"

const redirectIfAuthed = createMiddleware().server(async ({ next }) => {
  const { session, orgs } = await getAuthBootstrapFn()
  if (session && orgs.length > 0) throw redirect({ to: "/" })
  if (session && orgs.length === 0) throw redirect({ to: "/create-org" })
  return next()
})

// Get auth mode from server
const getAuthModeFn = createServerFn().handler(async () => {
  return process.env.AUTH_MODE ?? "dev"
})

const DEFAULT_PASSWORD = "dev-password-123"

export const Route = createFileRoute("/login")({
  component: LoginPage,
  server: { middleware: [redirectIfAuthed] },
  loader: async () => {
    const authMode = await getAuthModeFn()
    return { authMode }
  },
})

type LoginStep = "email" | "checking" | "magic-sent"

function LoginPage() {
  const navigate = useNavigate()
  const { authMode } = Route.useLoaderData()
  const [email, setEmail] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [step, setStep] = useState<LoginStep>("email")

  // Dev mode: auto-login with hardcoded password
  const handleDevLogin = async () => {
    const credentials = { email, password: DEFAULT_PASSWORD }

    const signIn = await authClient.signIn.email(credentials)
    if (signIn.data) {
      await linkPendingMembershipsFn()
      const orgs = await getOrgsFn()
      if (orgs.length === 0) {
        navigate({ to: "/create-org" })
        return
      }
      await setActiveOrgFn({ data: { organizationId: orgs[0].id } })
      navigate({ to: "/" })
      return
    }

    const signUp = await authClient.signUp.email({
      ...credentials,
      name: email.split("@")[0],
    })
    if (signUp.data) {
      await linkPendingMembershipsFn()
      const orgs = await getOrgsFn()
      if (orgs.length === 0) {
        navigate({ to: "/create-org" })
        return
      }
      await setActiveOrgFn({ data: { organizationId: orgs[0].id } })
      navigate({ to: "/" })
      return
    }

    setIsLoading(false)
  }

  // Production/Enterprise mode: magic link or SSO
  const handleProductionLogin = async () => {
    setStep("checking")

    // Check for SSO (only in enterprise mode)
    if (authMode === "enterprise") {
      const ssoCheck = await checkSSOForEmailFn({ data: { email } })
      if (ssoCheck.hasSSO) {
        // Redirect to WorkOS SSO
        const { url } = await getWorkosAuthUrlFn({
          data: { connectionId: ssoCheck.connectionId },
        })
        window.location.href = url
        return
      }
    }

    // No SSO, send magic link
    const result = await authClient.signIn.magicLink({
      email,
      callbackURL: "/",
      name: email.split("@")[0],
    })

    if (result.error) {
      console.error("Magic link error:", result.error)
      setStep("email")
      setIsLoading(false)
      return
    }

    setStep("magic-sent")
    setIsLoading(false)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    if (authMode === "dev") {
      await handleDevLogin()
    } else {
      await handleProductionLogin()
    }
  }

  // Magic link sent screen
  if (step === "magic-sent") {
    return (
      <div className="flex h-screen flex-col items-center justify-between main-gradient text-white">
        <div className="flex gap-3 h-40 items-center flex-col justify-end">
          <img src="/logo-white.png" alt="Logo" className="w-15 h-15" />
          <h1 className="text-2xl font-medium">TruBuild</h1>
        </div>
        <div className="flex flex-col gap-6 w-80 pb-20 text-center">
          <div className="bg-white/10 rounded-lg p-6">
            <h2 className="font-medium text-lg mb-2">Check your email</h2>
            <p className="text-white/70 text-sm mb-4">
              We sent a sign-in link to
            </p>
            <p className="font-medium">{email}</p>
            <p className="text-white/50 text-xs mt-4">
              The link expires in 5 minutes
            </p>
          </div>
          <Button
            variant="ghost"
            className="text-white/70 hover:text-white"
            onClick={() => {
              setStep("email")
              setEmail("")
            }}
          >
            Use a different email
          </Button>
        </div>
        <div className="text-12 text-white/50 h-40 flex items-end justify-center pb-10">
          Copyright © 2026 TruBuild. All rights reserved.
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col items-center justify-between main-gradient text-white">
      <div className="flex gap-3 h-40 items-center flex-col justify-end">
        <img src="/logo-white.png" alt="Logo" className="w-15 h-15" />
        <h1 className="text-2xl font-medium">TruBuild</h1>
      </div>
      <div className="flex flex-col gap-6 w-70 pb-20">
        <h2 className="font-medium text-center text-lg">Sign in to continue</h2>
        <form onSubmit={handleSubmit} className="flex flex-col items-center">
          <Input
            type="email"
            placeholder="name@company.com"
            autoFocus
            className="text-white placeholder:text-white/40 border-white/30 border-[0.5px] outline-red-500 bg-white/20 w-full text-center
              focus-visible:border-white/50 focus-visible:ring-white/20"
            value={email}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setEmail(e.target.value)
            }
            disabled={isLoading}
          />
          <Button
            type="submit"
            variant="primary"
            disabled={isLoading || !email.trim()}
            className="w-full px-3 mt-3"
          >
            {isLoading ? (
              <Spinner />
            ) : authMode === "dev" ? (
              "Sign in"
            ) : (
              "Continue"
            )}
          </Button>
        </form>
        {authMode !== "dev" && (
          <p className="text-white/50 text-xs text-center">
            We'll send you a magic link to sign in
          </p>
        )}
      </div>
      <div className="text-12 text-white/50 h-40 flex items-end justify-center pb-10">
        Copyright © 2026 TruBuild. All rights reserved.
      </div>
    </div>
  )
}
