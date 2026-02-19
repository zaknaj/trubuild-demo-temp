import { createAuthClient } from "better-auth/react"
import { organizationClient, magicLinkClient } from "better-auth/client/plugins"

export const authClient = createAuthClient({
  plugins: [organizationClient(), magicLinkClient()],
})
