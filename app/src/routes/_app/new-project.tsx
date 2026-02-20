import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { CountrySelect } from "@/components/CountrySelect"
import { FlowPageLayout } from "@/components/FlowPageLayout"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { createProjectFn } from "@/fn/projects"
import {
  currentUserOrgRoleQueryOptions,
  orgsQueryOptions,
  projectsQueryOptions,
  sessionQueryOptions,
} from "@/lib/query-options"
import { getOrgCountry } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

export const Route = createFileRoute("/_app/new-project")({
  component: RouteComponent,
})

function RouteComponent() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: userRole } = useQuery(currentUserOrgRoleQueryOptions)
  const { data: orgs } = useQuery(orgsQueryOptions)
  const { data: session } = useQuery(sessionQueryOptions)
  if (!userRole || !orgs || !session) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }
  const [projectName, setProjectName] = useState("")

  const activeOrg = orgs?.find((o) => o.id === session?.session?.activeOrganizationId)
  const orgCountry = getOrgCountry(activeOrg?.metadata)
  const [projectCountry, setProjectCountry] = useState(orgCountry)

  const createProject = useMutation({
    mutationFn: ({ name, country }: { name: string; country: string }) =>
      createProjectFn({ data: { name, country } }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: projectsQueryOptions.queryKey,
      })
      navigate({ to: "/all-projects" })
    },
  })

  const canCreateProject = userRole.role === "owner" || userRole.role === "admin"

  if (!canCreateProject) {
    return (
      <FlowPageLayout
        title="New project"
        backLabel="All projects"
        backTo="/all-projects"
      >
        <p className="text-sm text-black/55">
          You do not have permission to create projects.
        </p>
      </FlowPageLayout>
    )
  }

  return (
    <FlowPageLayout
      title="New project"
      backLabel="All projects"
      backTo="/all-projects"
    >
      <form
        className="space-y-6"
        onSubmit={(e) => {
          e.preventDefault()
          const name = projectName.trim()
          if (!name) return
          createProject.mutate({ name, country: projectCountry })
        }}
      >
        <div className="space-y-2">
          <Label htmlFor="project-name">Project name</Label>
          <Input
            id="project-name"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="e.g. Airport Expansion"
            className="bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
            autoFocus
            disabled={createProject.isPending}
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
        <div>
          <Button
            type="submit"
            className="w-full bg-black text-white hover:bg-black/90"
            disabled={createProject.isPending || !projectName.trim()}
          >
            {createProject.isPending ? "Creating..." : "Create project"}
          </Button>
        </div>
      </form>
    </FlowPageLayout>
  )
}
