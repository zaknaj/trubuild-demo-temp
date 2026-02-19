import { createFileRoute, Outlet, redirect } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { PackageSidebar } from "@/components/PackageSidebar"
import { PackageSettingsDialog } from "@/components/PackageSettingsDialog"
import {
  packageDetailQueryOptions,
  packageMembersQueryOptions,
  packageContractorsQueryOptions,
  packageAccessQueryOptions,
  technicalEvaluationsQueryOptions,
  packageCommercialSummaryQueryOptions,
} from "@/lib/query-options"

export const Route = createFileRoute("/_app/package/$id")({
  beforeLoad: async ({ params, context }) => {
    // Check access before loading the route
    try {
      const accessData = await context.queryClient.ensureQueryData(
        packageAccessQueryOptions(params.id)
      )
      if (accessData.access === "none") {
        throw redirect({ to: "/" })
      }
    } catch (error) {
      // If access check fails (e.g., package not found), redirect to home
      if (error instanceof Error && !("to" in error)) {
        throw redirect({ to: "/" })
      }
      throw error
    }
  },
  loader: ({ params, context }) => {
    context.queryClient.prefetchQuery(packageDetailQueryOptions(params.id))
    context.queryClient.prefetchQuery(packageMembersQueryOptions(params.id))
    context.queryClient.prefetchQuery(packageContractorsQueryOptions(params.id))
    context.queryClient.prefetchQuery(
      technicalEvaluationsQueryOptions(params.id)
    )
    context.queryClient.prefetchQuery(
      packageCommercialSummaryQueryOptions(params.id)
    )
    // Access already checked in beforeLoad, but prefetch for component use
    context.queryClient.prefetchQuery(packageAccessQueryOptions(params.id))
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const { data: accessData } = useSuspenseQuery(packageAccessQueryOptions(id))
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<
    "general" | "members" | "activity"
  >("general")

  const canEdit = accessData.access === "full"

  const openSettings = (
    tab: "general" | "members" | "activity" = "general"
  ) => {
    setSettingsTab(tab)
    setSettingsOpen(true)
  }

  const outlet = useMemo(() => <Outlet />, [id])

  return (
    <>
      <div className="flex flex-1 overflow-hidden h-full">
        <PackageSidebar packageId={id} onSettingsClick={openSettings} />
        <div className="flex-1 flex flex-col overflow-hidden rounded-xl content-shadow">
          {outlet}
        </div>
      </div>
      {canEdit && (
        <PackageSettingsDialog
          packageId={id}
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          defaultTab={settingsTab}
        />
      )}
    </>
  )
}
