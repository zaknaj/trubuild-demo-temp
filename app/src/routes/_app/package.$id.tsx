import { createFileRoute, Link, Outlet, redirect } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { ChevronRight, Ellipsis } from "lucide-react"
import {
  packageDetailQueryOptions,
  packageMembersQueryOptions,
  packageAccessQueryOptions,
} from "@/lib/query-options"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Spinner } from "@/components/ui/spinner"

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
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const { data: accessData } = useQuery(packageAccessQueryOptions(id))
  const { data: packageData } = useQuery(packageDetailQueryOptions(id))
  const { data: packageMembers } = useQuery(packageMembersQueryOptions(id))

  if (!accessData || !packageData || !packageMembers) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const memberCount = packageMembers.length
  const membersLabel =
    memberCount === 0
      ? "No members"
      : `${memberCount} ${memberCount === 1 ? "member" : "members"}`

  return (
    <div className="flex flex-1 h-full flex-col">
      <div className="flex items-center justify-between h-12 px-6">
        <div className="group flex items-center gap-2 text-[14px] font-medium text-black min-w-0">
          <div className="flex items-center gap-2 min-w-0">
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
          <Tooltip>
            <TooltipTrigger asChild>
              <Link
                to="/settings"
                search={{ section: `package:${packageData.package.id}` }}
                className="h-7 w-7 rounded-[6px] flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-black/5 transition-opacity shrink-0"
              >
                <Ellipsis size={14} className="text-black/55" />
              </Link>
            </TooltipTrigger>
            <TooltipContent>Package settings</TooltipContent>
          </Tooltip>
        </div>
        <Link
          to="/settings"
          search={{ section: `package:${packageData.package.id}` }}
          className="flex items-center gap-[6px] min-w-0 hover:opacity-80 transition-opacity"
        >
          <span className="h-6 w-6 rounded-full bg-[#E0E0E0] shrink-0" />
          <span className="text-[13px] font-medium text-black truncate">
            {membersLabel}
          </span>
        </Link>
      </div>
      <div className="flex-1 min-h-0">
        <Outlet />
      </div>
    </div>
  )
}
