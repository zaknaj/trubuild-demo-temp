import { Button } from "@/components/ui/button"
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router"
import {
  useQuery,
} from "@tanstack/react-query"
import {
  projectsQueryOptions,
  currentUserOrgRoleQueryOptions,
} from "@/lib/query-options"
import { SimpleHeader } from "@/components/SimpleHeader"
import type { Project } from "@/lib/types"
import { PlusIcon, FolderIcon, ChevronRightIcon } from "lucide-react"
import { Spinner } from "@/components/ui/spinner"

export const Route = createFileRoute("/_app/all-projects")({
  component: RouteComponent,
})

function RouteComponent() {
  const navigate = useNavigate()
  const { data: projects } = useQuery(projectsQueryOptions)
  const { data: userRole } = useQuery(currentUserOrgRoleQueryOptions)

  if (!projects || !userRole) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const canCreateProject =
    userRole.role === "owner" || userRole.role === "admin"

  // Only org owners/admins can see award stats (commercial data)
  // Regular members might have limited access to individual projects
  const canViewAwardStats = canCreateProject

  return (
    <>
      <SimpleHeader title="All projects" />
      <div className="flex-1 overflow-auto">
        <div className="max-w-[600px] mx-auto p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Projects
            </h2>
            {canCreateProject && (
              <Button
                onClick={() => navigate({ to: "/new-project" })}
                size="sm"
                variant="outline"
              >
                <PlusIcon size={14} />
                New project
              </Button>
            )}
          </div>

          {projects.length === 0 ? (
            <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-8 text-center text-sm text-muted-foreground">
              No projects yet. Create one to get started.
            </div>
          ) : (
            <div className="border rounded-lg divide-y">
              {projects.map((project: Project) => (
                <Link
                  key={project.id}
                  to="/project/$id"
                  params={{ id: project.id }}
                  className="group flex items-center gap-3 px-3 py-2.5 hover:bg-muted/50 transition-colors"
                >
                  <FolderIcon
                    size={16}
                    className="text-muted-foreground shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <span className="font-medium text-sm truncate">
                      {project.name}
                    </span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {canViewAwardStats ? (
                      <>
                        {project.awardedPackageCount}/{project.packageCount}{" "}
                        awarded
                        {project.packageCount > 0 && (
                          <span className="ml-1">
                            (
                            {Math.round(
                              (project.awardedPackageCount /
                                project.packageCount) *
                                100
                            )}
                            %)
                          </span>
                        )}
                      </>
                    ) : (
                      <>
                        {project.packageCount}{" "}
                        {project.packageCount === 1 ? "package" : "packages"}
                      </>
                    )}
                  </span>
                  <ChevronRightIcon
                    size={14}
                    className="text-muted-foreground/50 group-hover:text-muted-foreground transition-colors"
                  />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  )
}
