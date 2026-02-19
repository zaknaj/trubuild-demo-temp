import { useState, useEffect, useRef } from "react"
import { useNavigate } from "@tanstack/react-router"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { cn } from "@/lib/utils"
import { Archive } from "lucide-react"
import { MemberAvatar } from "@/components/ui/member-avatar"
import { MemberListItem } from "@/components/ui/member-list-item"
import {
  useSuspenseQuery,
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  projectDetailQueryOptions,
  projectMembersQueryOptions,
  projectAccessQueryOptions,
  orgMembersQueryOptions,
  projectsQueryOptions,
  archivedProjectsQueryOptions,
} from "@/lib/query-options"
import { addProjectMemberFn, removeProjectMemberFn } from "@/fn/members"
import {
  archiveProjectFn,
  renameProjectFn,
  updateProjectCountryFn,
} from "@/fn/projects"
import { CountrySelect } from "@/components/CountrySelect"
import type { Member } from "@/lib/types"
import type { ProjectRole } from "@/lib/permissions"
import { toast } from "sonner"

interface ProjectSettingsDialogProps {
  projectId: string
  open: boolean
  onOpenChange: (open: boolean) => void
  defaultTab?: "general" | "members" | "activity"
}

export function ProjectSettingsDialog({
  projectId,
  open,
  onOpenChange,
  defaultTab = "general",
}: ProjectSettingsDialogProps) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: projectData } = useSuspenseQuery(
    projectDetailQueryOptions(projectId)
  )
  const { data: members } = useSuspenseQuery(
    projectMembersQueryOptions(projectId)
  )
  const { data: accessInfo } = useSuspenseQuery(
    projectAccessQueryOptions(projectId)
  )
  const { data: orgMembers = [] } = useQuery({
    ...orgMembersQueryOptions,
    enabled: open,
  })

  const [activeTab, setActiveTab] = useState<
    "general" | "members" | "activity"
  >(defaultTab)
  const [email, setEmail] = useState("")
  const [role, setRole] = useState<ProjectRole>("commercial_lead")
  const [removingEmail, setRemovingEmail] = useState<string | null>(null)
  const [isArchiveDialogOpen, setIsArchiveDialogOpen] = useState(false)
  const [projectName, setProjectName] = useState("")
  const [projectCountry, setProjectCountry] = useState("")
  const emailInputRef = useRef<HTMLInputElement>(null)

  const { project } = projectData
  const canInvite = accessInfo.access === "full"
  const canArchive = accessInfo.access === "full"
  const canRename = accessInfo.access === "full"

  useEffect(() => {
    setProjectName(project.name)
  }, [project.name])

  useEffect(() => {
    setProjectCountry(project.country ?? "")
  }, [project.country])

  useEffect(() => {
    if (open) {
      setActiveTab(defaultTab)
    }
  }, [open, defaultTab])

  // Filter org members who are not already project members
  const projectMemberEmails = new Set(members.map((m: Member) => m.email))
  const availableOrgMembers = orgMembers.filter(
    (m) => !projectMemberEmails.has(m.email)
  )

  const addMember = useMutation({
    mutationFn: (data: { email: string; role: ProjectRole }) =>
      addProjectMemberFn({ data: { projectId: project.id, ...data } }),
    onSuccess: () => {
      toast.success("Invitation sent")
      queryClient.invalidateQueries({
        queryKey: projectMembersQueryOptions(projectId).queryKey,
      })
      setEmail("")
      setRole("commercial_lead")
    },
  })

  const removeMember = useMutation({
    mutationFn: (memberEmail: string) =>
      removeProjectMemberFn({
        data: { projectId: project.id, email: memberEmail },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectMembersQueryOptions(projectId).queryKey,
      })
      setRemovingEmail(null)
    },
    onError: () => {
      setRemovingEmail(null)
    },
  })

  const archiveProject = useMutation({
    mutationFn: () => archiveProjectFn({ data: { projectId: project.id } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      queryClient.invalidateQueries({
        queryKey: archivedProjectsQueryOptions.queryKey,
      })
      onOpenChange(false)
      navigate({ to: "/all-projects" })
    },
  })

  const renameProject = useMutation({
    mutationFn: (name: string) =>
      renameProjectFn({ data: { projectId: project.id, name } }),
    onMutate: () => {
      toast.loading("Renaming project...", { id: "rename-project" })
    },
    onSuccess: () => {
      toast.success("Project renamed", { id: "rename-project" })
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(projectId).queryKey,
      })
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to rename project",
        { id: "rename-project" }
      )
    },
  })

  const updateCountry = useMutation({
    mutationFn: (country: string) =>
      updateProjectCountryFn({ data: { projectId: project.id, country } }),
    onMutate: () => {
      toast.loading("Updating country...", { id: "update-project-country" })
    },
    onSuccess: () => {
      toast.success("Country updated", { id: "update-project-country" })
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(projectId).queryKey,
      })
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update country",
        { id: "update-project-country" }
      )
    },
  })

  const handleRename = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedName = projectName.trim()
    if (!trimmedName || trimmedName === project.name) return
    renameProject.mutate(trimmedName)
  }

  const handleAddMember = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedEmail = email.trim()
    if (!trimmedEmail) return
    addMember.mutate({ email: trimmedEmail, role })
  }

  const handleRemove = (memberEmail: string) => {
    setRemovingEmail(memberEmail)
    removeMember.mutate(memberEmail)
  }

  const activeMembers = members.filter((m: Member) => m.userId !== null)
  const pendingMembers = members.filter((m: Member) => m.userId === null)

  return (
    <>
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent className="w-fit max-w-fit p-0 overflow-hidden">
          <SheetTitle className="sr-only">Project Settings</SheetTitle>
          <SheetDescription className="sr-only">
            Manage project settings and members
          </SheetDescription>
          <div className="flex h-full">
            {/* Sidenav */}
            <div className="w-[200px] border-r flex flex-col shrink-0">
              <h2 className="text-lg font-medium p-4">Project Settings</h2>
              <div className="flex-1 flex flex-col gap-px py-2 px-2">
                <button
                  onClick={() => setActiveTab("general")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "general" && "active"
                  )}
                >
                  General
                </button>
                <button
                  onClick={() => setActiveTab("members")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "members" && "active"
                  )}
                >
                  Members
                </button>
                <button
                  onClick={() => setActiveTab("activity")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "activity" && "active"
                  )}
                >
                  Activity
                </button>
              </div>
            </div>
            {/* Content */}
            <div className="w-[550px] overflow-y-auto p-6 min-w-0 shrink-0">
              {activeTab === "general" && (
                <div className="space-y-6">
                  <h2 className="text-lg font-medium">General</h2>

                  {/* Rename Project */}
                  {canRename && (
                    <div className="border rounded-lg p-4 space-y-3">
                      <div>
                        <p className="font-medium">Project Name</p>
                        <p className="text-sm text-muted-foreground">
                          Change the display name of this project
                        </p>
                      </div>
                      <form onSubmit={handleRename} className="flex gap-2">
                        <Input
                          value={projectName}
                          onChange={(e) => setProjectName(e.target.value)}
                          disabled={renameProject.isPending}
                          className="max-w-xs"
                        />
                        <Button
                          type="submit"
                          size="sm"
                          disabled={
                            renameProject.isPending ||
                            !projectName.trim() ||
                            projectName.trim() === project.name
                          }
                        >
                          {renameProject.isPending ? "Saving..." : "Save"}
                        </Button>
                      </form>
                    </div>
                  )}

                  {/* Country */}
                  {canRename && (
                    <div className="border rounded-lg p-4 space-y-3">
                      <div>
                        <p className="font-medium">Country</p>
                        <p className="text-sm text-muted-foreground">
                          The country where this project is located
                        </p>
                      </div>
                      <CountrySelect
                        value={projectCountry}
                        onValueChange={(newCountry) => {
                          setProjectCountry(newCountry)
                          updateCountry.mutate(newCountry)
                        }}
                        disabled={updateCountry.isPending}
                        className="max-w-xs"
                      />
                    </div>
                  )}

                  {/* Danger Zone */}
                  {canArchive && (
                    <div className="border border-destructive/30 rounded-lg p-4 bg-destructive/5">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="font-medium">Archive this project</p>
                          <p className="text-sm text-muted-foreground">
                            The project and all its packages will be hidden from
                            view. You can restore it later.
                          </p>
                        </div>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => setIsArchiveDialogOpen(true)}
                          className="gap-1.5 shrink-0"
                        >
                          <Archive className="size-4" />
                          Archive
                        </Button>
                      </div>
                      {archiveProject.error && (
                        <p className="text-sm text-red-500 mt-2">
                          {archiveProject.error instanceof Error
                            ? archiveProject.error.message
                            : "Failed to archive project"}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "members" && (
                <div className="space-y-4">
                  <h2 className="text-lg font-medium">Members</h2>
                  {canInvite && (
                    <form
                      onSubmit={handleAddMember}
                      className="flex items-end gap-2 pb-4 border-b"
                    >
                      <div className="flex-1">
                        <label
                          htmlFor="invite-email"
                          className="text-xs text-muted-foreground mb-1 block"
                        >
                          Email
                        </label>
                        <Input
                          ref={emailInputRef}
                          id="invite-email"
                          type="email"
                          placeholder="colleague@example.com"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          disabled={addMember.isPending}
                        />
                      </div>
                      <div className="w-40">
                        <label
                          htmlFor="invite-role"
                          className="text-xs text-muted-foreground mb-1 block"
                        >
                          Role
                        </label>
                        <Select
                          value={role}
                          onValueChange={(v) => setRole(v as ProjectRole)}
                          disabled={addMember.isPending}
                        >
                          <SelectTrigger id="invite-role" className="w-full">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="project_lead">
                              Project Lead
                            </SelectItem>
                            <SelectItem value="commercial_lead">
                              Commercial Lead
                            </SelectItem>
                            <SelectItem value="technical_lead">
                              Technical Lead
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <Button
                        type="submit"
                        disabled={addMember.isPending || !email.trim()}
                        size="sm"
                      >
                        {addMember.isPending ? "Inviting..." : "Invite"}
                      </Button>
                    </form>
                  )}

                  {addMember.error && (
                    <div className="text-sm text-red-500 bg-red-50 p-3 rounded-md">
                      {addMember.error instanceof Error
                        ? addMember.error.message
                        : "Failed to send invitation"}
                    </div>
                  )}

                  <div className="border rounded-lg divide-y">
                    {members.length === 0 ? (
                      <div className="p-4 text-sm text-muted-foreground text-center">
                        No members yet
                      </div>
                    ) : (
                      <>
                        {activeMembers.map((m: Member) => (
                          <MemberListItem
                            key={m.id}
                            name={m.userName}
                            email={m.email}
                            image={m.userImage}
                            role={m.role}
                            showRemoveButton={canInvite}
                            isRemoving={removingEmail === m.email}
                            onRemove={() => handleRemove(m.email)}
                          />
                        ))}
                        {pendingMembers.map((m: Member) => (
                          <MemberListItem
                            key={m.id}
                            name={m.userName}
                            email={m.email}
                            pending
                            role={m.role}
                            showRemoveButton={canInvite}
                            isRemoving={removingEmail === m.email}
                            onRemove={() => handleRemove(m.email)}
                          />
                        ))}
                      </>
                    )}
                  </div>
                  {removeMember.error && (
                    <p className="text-sm text-red-500">
                      {removeMember.error instanceof Error
                        ? removeMember.error.message
                        : "Failed to remove member"}
                    </p>
                  )}

                  {availableOrgMembers.length > 0 && canInvite && (
                    <div className="space-y-2">
                      <Label className="mb-4 mt-10">Quick Invite</Label>
                      <div className="border rounded-lg divide-y max-h-32 overflow-y-auto">
                        {availableOrgMembers.map((m) => (
                          <div
                            key={m.id}
                            className="flex items-center gap-2 w-full px-3 py-2 text-sm"
                          >
                            <MemberAvatar
                              name={m.userName}
                              email={m.email}
                              image={m.userImage}
                            />
                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate text-xs">
                                {m.userName || m.email}
                              </p>
                              {m.userName && (
                                <p className="text-xs text-muted-foreground truncate">
                                  {m.email}
                                </p>
                              )}
                            </div>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="shrink-0 h-7 text-xs"
                              onClick={() => {
                                setEmail(m.email)
                                emailInputRef.current?.scrollIntoView({
                                  behavior: "smooth",
                                  block: "center",
                                })
                                emailInputRef.current?.focus()
                              }}
                            >
                              Invite
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "activity" && (
                <div className="space-y-4">
                  <h2 className="text-lg font-medium">Activity</h2>
                  <div className="border rounded-lg p-8 text-center text-muted-foreground">
                    <p className="text-sm">Activity log coming soon</p>
                    <p className="text-xs mt-1">
                      Track changes and updates to this project
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <AlertDialog
        open={isArchiveDialogOpen}
        onOpenChange={setIsArchiveDialogOpen}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Archive project?</AlertDialogTitle>
            <AlertDialogDescription>
              This will archive "{project.name}" and all its packages. The
              project will be hidden from view but can be restored later from
              the archived section.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={archiveProject.isPending}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => archiveProject.mutate()}
              disabled={archiveProject.isPending}
            >
              {archiveProject.isPending ? "Archiving..." : "Archive project"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
