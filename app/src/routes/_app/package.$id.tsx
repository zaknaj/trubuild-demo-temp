import { createFileRoute, Outlet, redirect } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import { useMemo } from "react"
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
  useSuspenseQuery(packageAccessQueryOptions(id))

  const outlet = useMemo(() => <Outlet />, [id])

  return (
    <div className="flex flex-1 h-full">
      <div className="flex-1">
        {outlet}
      </div>
    </div>
  )
}
