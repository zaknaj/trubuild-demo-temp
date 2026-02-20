import { useMemo } from "react"
import { createFileRoute, Link, useNavigate, redirect } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { BoxIcon, PlusIcon, ChevronRightIcon, Ellipsis } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  projectDetailQueryOptions,
  projectAccessQueryOptions,
  projectMembersQueryOptions,
} from "@/lib/query-options"
import { Spinner } from "@/components/ui/spinner"

type PackageWithAssetCount = {
  id: string
  name: string
  currency: string | null
  projectId: string
  assetCount: number
  awardedContractorId: string | null
  awardedContractorName: string | null
}

export const Route = createFileRoute("/_app/project/$id")({
  beforeLoad: async ({ params, context }) => {
    try {
      const accessData = await context.queryClient.ensureQueryData(
        projectAccessQueryOptions(params.id)
      )
      if (accessData.access === "none") {
        throw redirect({ to: "/" })
      }
    } catch (error) {
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
  const navigate = useNavigate()
  const { data: projectData } = useQuery(projectDetailQueryOptions(id))
  const { data: accessData } = useQuery(projectAccessQueryOptions(id))
  const { data: projectMembers } = useQuery(projectMembersQueryOptions(id))

  if (!projectData || !accessData || !projectMembers) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const { project, packages } = projectData
  const memberCount = projectMembers.length
  const membersLabel =
    memberCount === 0
      ? "No members"
      : `${memberCount} ${memberCount === 1 ? "member" : "members"}`

  const canViewCommercial =
    accessData.access === "full" || accessData.access === "commercial"

  const canCreatePackage = accessData.access === "full"

  const packagesList = useMemo(() => {
    if (packages.length === 0) {
      return (
        <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-8 text-center text-sm text-muted-foreground">
          No packages yet. Create one to get started.
        </div>
      )
    }

    return (
      <div className="border rounded-lg divide-y">
        {packages.map((pkg: PackageWithAssetCount) => {
          // Only show awarded info to users with commercial access
          const showAwarded = canViewCommercial && pkg.awardedContractorName
          return (
            <Link
              key={pkg.id}
              to="/package/$id"
              params={{ id: pkg.id }}
              className={`group flex flex-col px-3 py-2.5 hover:bg-muted/50 transition-colors ${
                showAwarded ? "bg-green-50" : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <BoxIcon size={16} className="text-muted-foreground shrink-0" />
                <div className="flex-1 min-w-0 flex items-center gap-2">
                  <span className="font-medium text-sm truncate">
                    {pkg.name}
                  </span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-xs text-muted-foreground">
                    {pkg.assetCount} {pkg.assetCount === 1 ? "asset" : "assets"}
                  </span>
                  <ChevronRightIcon
                    size={14}
                    className="text-muted-foreground/50 group-hover:text-muted-foreground transition-colors"
                  />
                </div>
              </div>
              {showAwarded && (
                <div className="ml-7 mt-1 text-xs text-green-700">
                  Awarded to {pkg.awardedContractorName}
                </div>
              )}
            </Link>
          )
        })}
      </div>
    )
  }, [packages, canViewCommercial])

  return (
    <>
      <div className="flex flex-1 h-full">
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between h-12 px-6">
            <div className="group flex items-center gap-2 text-[14px] font-medium text-black min-w-0">
              <div className="flex items-center gap-2 min-w-0">
                <Link to="/all-projects" className="opacity-40 hover:opacity-70 truncate">
                  All projects
                </Link>
                <ChevronRightIcon size={12} className="opacity-40 shrink-0" />
                <Link
                  to="/project/$id"
                  params={{ id: project.id }}
                  className="opacity-100 truncate"
                >
                  {project.name}
                </Link>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    to="/settings"
                    search={{ section: `project:${project.id}` }}
                    className="h-7 w-7 rounded-[6px] flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-black/5 transition-opacity shrink-0"
                  >
                    <Ellipsis size={14} className="text-black/55" />
                  </Link>
                </TooltipTrigger>
                <TooltipContent>Project settings</TooltipContent>
              </Tooltip>
            </div>
            <Link
              to="/settings"
              search={{ section: `project:${project.id}` }}
              className="flex items-center gap-[6px] min-w-0 hover:opacity-80 transition-opacity"
            >
              <span className="h-6 w-6 rounded-full bg-[#E0E0E0] shrink-0" />
              <span className="text-[13px] font-medium text-black truncate">
                {membersLabel}
              </span>
            </Link>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto">
            <div className="max-w-[600px] mx-auto p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Packages
                </h2>
                {canCreatePackage && (
                  <Button
                    onClick={() =>
                      navigate({
                        to: "/new-package/$projectId",
                        params: { projectId: project.id },
                      })
                    }
                    size="sm"
                    variant="outline"
                  >
                    <PlusIcon size={14} />
                    New package
                  </Button>
                )}
              </div>

              {packagesList}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
