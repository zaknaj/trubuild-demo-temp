import { createFileRoute, Link, Outlet, redirect } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { ChevronRight } from "lucide-react"
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
  const { data: packageData } = useSuspenseQuery(packageDetailQueryOptions(id))
  const { data: packageMembers } = useSuspenseQuery(packageMembersQueryOptions(id))

  const memberCount = packageMembers.length
  const membersLabel =
    memberCount === 0
      ? "No members"
      : `${memberCount} ${memberCount === 1 ? "member" : "members"}`

  const outlet = useMemo(() => <Outlet />, [id])

  return (
    <div className="flex flex-1 h-full flex-col">
      <div className="flex items-center justify-between h-12 px-6">
        <div className="flex items-center gap-1 text-[14px] font-medium text-black min-w-0">
          <Link
            to="/project/$id"
            params={{ id: packageData.project.id }}
            className="opacity-40 hover:opacity-70 truncate"
          >
            {packageData.project.name}
          </Link>
          <ChevronRight size={12} className="opacity-40 shrink-0" />
          <Link
            to="/package/$id"
            params={{ id: packageData.package.id }}
            className="opacity-100 truncate"
          >
            {packageData.package.name}
          </Link>
        </div>
        <div className="flex items-center gap-[6px] min-w-0">
          <span className="h-6 w-6 rounded-full bg-[#E0E0E0] shrink-0" />
          <span className="text-[13px] font-medium text-black truncate">
            {membersLabel}
          </span>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        {outlet}
      </div>
    </div>
  )
}
