import {
  createFileRoute,
  Outlet,
  redirect,
  useRouterState,
} from "@tanstack/react-router"
import { Sidebar } from "../components/Sidebar"
import { Suspense } from "react"
import { Spinner } from "@/components/ui/spinner"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import {
  authBootstrapQueryOptions,
  orgsQueryOptions,
  sessionQueryOptions,
} from "@/lib/query-options"

export const Route = createFileRoute("/_app")({
  beforeLoad: async ({ context }) => {
    const { queryClient } = context
    const { session, orgs } = await queryClient.ensureQueryData(
      authBootstrapQueryOptions
    )
    // Prime the individual caches only if they don't already have data.
    // This prevents re-renders from link hover preloading.
    if (!queryClient.getQueryData(sessionQueryOptions.queryKey)) {
      queryClient.setQueryData(sessionQueryOptions.queryKey, session)
    }
    if (!queryClient.getQueryData(orgsQueryOptions.queryKey)) {
      queryClient.setQueryData(orgsQueryOptions.queryKey, orgs)
    }
    if (!session) {
      throw redirect({ to: "/login" })
    }
    if (orgs.length === 0) {
      throw redirect({ to: "/create-org" })
    }
  },
  component: RouteComponent,
})

function RouteComponent() {
  const isFocusPage = useRouterState({
    select: (state) =>
      state.matches.some((match) => {
        const focusRoutePrefixes = [
          "/_app/settings",
          "/_app/new-project",
          "/_app/new-package",
          "/_app/new-asset",
          "/_app/new-tech-evaluation",
          "/_app/new-commercial-evaluation",
          "/_app/award-package",
        ]
        return focusRoutePrefixes.some((prefix) => match.routeId.startsWith(prefix))
      }),
  })

  return (
    <div className="w-screen h-screen overflow-hidden text-sm flex bg-[#F9F9F9]">
      <ErrorBoundary>
        {!isFocusPage && <Sidebar />}
      </ErrorBoundary>
      <div className="flex-1">
        <ErrorBoundary>
          <div>
            <ErrorBoundary>
              <Suspense
                fallback={
                  <div className="p-6 flex items-center justify-center size-full">
                    <Spinner className="size-6 stroke-1" />
                  </div>
                }
              >
                <Outlet />
              </Suspense>
            </ErrorBoundary>
          </div>
        </ErrorBoundary>
      </div>
    </div>
  )
}
