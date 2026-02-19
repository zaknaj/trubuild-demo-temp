import {
  createFileRoute,
  Outlet,
  redirect,
  useLocation,
} from "@tanstack/react-router"
import { Sidebar } from "@/components/Sidebar"
import { Chat } from "@/components/Chat"
import { Suspense, useEffect, useRef } from "react"
import { Spinner } from "@/components/ui/spinner"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import useStore from "@/lib/store"
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
  const location = useLocation()
  const setNavbarOpen = useStore((s) => s.setNavbarOpen)
  const prevPathRef = useRef<string | null>(null)

  // Auto-collapse/expand main sidebar when navigating between route types
  useEffect(() => {
    const pathname = location.pathname
    const prevPath = prevPathRef.current
    prevPathRef.current = pathname

    // Skip initial render - Sidebar handles its own initial state
    if (prevPath === null) return

    const isOnProjectOrPackage =
      pathname.startsWith("/project/") || pathname.startsWith("/package/")
    const wasOnProjectOrPackage =
      prevPath.startsWith("/project/") || prevPath.startsWith("/package/")
    const isOnTopLevel =
      pathname === "/" ||
      pathname === "/all-projects" ||
      pathname === "/vendor-db"
    const wasOnTopLevel =
      prevPath === "/" ||
      prevPath === "/all-projects" ||
      prevPath === "/vendor-db"

    // Collapse when entering project/package from top-level
    if (isOnProjectOrPackage && wasOnTopLevel) {
      setNavbarOpen(false)
    }
    // Expand when returning to top-level from project/package
    else if (isOnTopLevel && wasOnProjectOrPackage) {
      setNavbarOpen(true)
    }
  }, [location.pathname, setNavbarOpen])

  return (
    <div className="w-screen overflow-hidden h-screen text-sm main-gradient flex  p-4">
      {/* <Navbar />
      <div className="bg-white h-full overflow-auto max-h-full rounded-xl m-3 mt-0 ring-white/20 ring-4">
        <Suspense fallback={<div className="p-6">Loading...</div>}> 
          <Outlet />
        </Suspense>
      </div> */}

      <div className="flex flex-1 bg-white/15 ring-inset ring-[0.5px] rounded-xl ring-white/20 shadow-[0_0_12px_0_rgba(0,0,0,.25)]">
        <ErrorBoundary>
          <Sidebar />
        </ErrorBoundary>
        <div className=" flex-1 flex bg-white rounded-xl overflow-hidden shadow-[-4px_0_20px_0_rgba(0,0,0,.2)]">
          <div className="flex flex-col flex-1">
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
          <ErrorBoundary>
            <Chat />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  )
}
