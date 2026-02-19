import { WorkOS } from "@workos-inc/node"

const workosApiKey = process.env.WORKOS_API_KEY
if (!workosApiKey) {
  throw new Error("Missing required environment variable: WORKOS_API_KEY")
}
const workosClientId = process.env.WORKOS_CLIENT_ID
if (!workosClientId) {
  throw new Error("Missing required environment variable: WORKOS_CLIENT_ID")
}

export const workos = new WorkOS(workosApiKey)

export function getWorkosAuthUrl(connectionId: string, redirectUri: string) {
  return workos.sso.getAuthorizationUrl({
    connection: connectionId,
    clientId: workosClientId,
    redirectUri,
  })
}

export async function getWorkosProfile(code: string) {
  return workos.sso.getProfileAndToken({
    code,
    clientId: workosClientId,
  })
}
