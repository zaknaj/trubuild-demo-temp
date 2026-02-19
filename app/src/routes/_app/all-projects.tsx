import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { createProjectFn } from "@/fn/projects"
import { createFileRoute, Link } from "@tanstack/react-router"
import { useState } from "react"
import {
  useSuspenseQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  projectsQueryOptions,
  currentUserOrgRoleQueryOptions,
  orgsQueryOptions,
  sessionQueryOptions,
} from "@/lib/query-options"
import { CountrySelect } from "@/components/CountrySelect"
import { SimpleHeader } from "@/components/SimpleHeader"
import { getOrgCountry } from "@/lib/utils"
import type { Project } from "@/lib/types"
import { PlusIcon, FolderIcon, ChevronRightIcon } from "lucide-react"

export const Route = createFileRoute("/_app/all-projects")({
  loader: ({ context }) => {
    context.queryClient.prefetchQuery(projectsQueryOptions)
    context.queryClient.prefetchQuery(currentUserOrgRoleQueryOptions)
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { data: projects } = useSuspenseQuery(projectsQueryOptions)
  const { data: userRole } = useSuspenseQuery(currentUserOrgRoleQueryOptions)
  const { data: orgs } = useSuspenseQuery(orgsQueryOptions)
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const queryClient = useQueryClient()
  const [sheetOpen, setSheetOpen] = useState(false)
  const [projectName, setProjectName] = useState("")

  const activeOrg = orgs?.find(
    (o) => o.id === session?.session?.activeOrganizationId
  )
  const orgCountry = getOrgCountry(activeOrg?.metadata)
  const [projectCountry, setProjectCountry] = useState(orgCountry)

  const canCreateProject =
    userRole.role === "owner" || userRole.role === "admin"

  // Only org owners/admins can see award stats (commercial data)
  // Regular members might have limited access to individual projects
  const canViewAwardStats = canCreateProject

  const createProject = useMutation({
    mutationFn: ({ name, country }: { name: string; country: string }) =>
      createProjectFn({ data: { name, country } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      closeSheet()
    },
  })

  const closeSheet = () => {
    setSheetOpen(false)
    setProjectName("")
    setProjectCountry(orgCountry)
    createProject.reset()
  }

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault()
    const name = projectName.trim()
    if (!name) return
    createProject.mutate({ name, country: projectCountry })
  }

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
                onClick={() => setSheetOpen(true)}
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

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent className="w-[350px] sm:max-w-none">
          <form onSubmit={handleCreate}>
            <SheetHeader>
              <SheetTitle>Create project</SheetTitle>
              <SheetDescription>
                Create a new project for your organization.
              </SheetDescription>
            </SheetHeader>
            <div className="flex-1 p-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="project-name">Project name</Label>
                <Input
                  id="project-name"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  disabled={createProject.isPending}
                  autoFocus
                />
              </div>
              <div className="space-y-2">
                <Label>Country</Label>
                <CountrySelect
                  value={projectCountry}
                  onValueChange={setProjectCountry}
                  disabled={createProject.isPending}
                />
              </div>
              {createProject.error ? (
                <p className="text-sm text-red-500">
                  {createProject.error instanceof Error
                    ? createProject.error.message
                    : "Unable to create project."}
                </p>
              ) : null}
            </div>
            <SheetFooter>
              <Button
                type="submit"
                disabled={createProject.isPending || !projectName.trim()}
              >
                {createProject.isPending ? "Creating..." : "Create project"}
              </Button>
            </SheetFooter>
          </form>
        </SheetContent>
      </Sheet>
    </>
  )
}
