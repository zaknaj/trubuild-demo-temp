import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ArrowLeft, Building2, FolderOpen, Settings, Users } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { CountrySelect } from "@/components/CountrySelect"
import { CurrencySelect } from "@/components/CurrencySelect"
import { cn, getOrgCountry } from "@/lib/utils"
import {
  archivedPackagesQueryOptions,
  archivedProjectsQueryOptions,
  currentUserOrgRoleQueryOptions,
  orgMembersQueryOptions,
  orgPendingInvitesQueryOptions,
  orgsQueryOptions,
  packageAccessQueryOptions,
  packageDetailQueryOptions,
  packageMembersQueryOptions,
  projectAccessQueryOptions,
  projectDetailQueryOptions,
  projectMembersQueryOptions,
  projectsQueryOptions,
  sessionQueryOptions,
} from "@/lib/query-options"
import { updateOrganizationFn, updateProfileFn } from "@/fn/auth"
import {
  addPackageMemberFn,
  addProjectMemberFn,
  cancelOrgInvitationFn,
  inviteMemberFn,
  removeOrgMemberFn,
  removePackageMemberFn,
  removeProjectMemberFn,
} from "@/fn/members"
import {
  archiveProjectFn,
  renameProjectFn,
  restoreProjectFn,
  updateProjectCountryFn,
} from "@/fn/projects"
import {
  archivePackageFn,
  renamePackageFn,
  restorePackageFn,
  updatePackageCurrencyFn,
} from "@/fn/packages"
import { authClient } from "@/auth/auth-client"

type SettingsKey =
  | "general"
  | "organization"
  | "members"
  | `project:${string}`
  | `package:${string}`

type OrgMemberRole = "owner" | "admin" | "member"
type ProjectMemberRole = "project_lead" | "commercial_lead" | "technical_lead"
type PackageMemberRole = "package_lead" | "commercial_team" | "technical_team"

type ProjectNavItem = {
  id: string
  name: string
  country: string | null
  packages: Array<{ id: string; name: string }>
}

function navItemClass(active: boolean) {
  return cn(
    "w-full h-8 rounded-[6px] px-2 flex items-center text-[13px] font-medium text-left",
    active ? "bg-[#EDEDED]" : "hover:bg-[#EDEDED]"
  )
}

const settingsCardClass =
  "rounded-xl border border-black/10 bg-white px-5 py-4 space-y-4 shadow-[0_1px_0_rgba(0,0,0,0.03)]"
const settingsSectionTitleClass = "text-[12px] font-semibold tracking-wide text-black/50"

function SectionLoading() {
  return (
    <div className={cn(settingsCardClass, "min-h-[160px] flex items-center justify-center")}>
      <Spinner className="size-5 stroke-1.5 text-black/60" />
    </div>
  )
}

export const Route = createFileRoute("/_app/settings")({
  validateSearch: (search: Record<string, unknown>) => ({
    section: typeof search.section === "string" ? search.section : undefined,
  }),
  loader: async ({ context }) => {
    const queryClient = context.queryClient

    // Keep initial entry fast: load only always-needed data.
    await Promise.all([
      queryClient.ensureQueryData(sessionQueryOptions),
      queryClient.ensureQueryData(orgsQueryOptions),
      queryClient.ensureQueryData(currentUserOrgRoleQueryOptions),
      queryClient.ensureQueryData(projectsQueryOptions),
    ])
  },
  component: RouteComponent,
})

function RouteComponent() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const search = Route.useSearch()

  const { data: session } = useQuery(sessionQueryOptions)
  const { data: orgs } = useQuery(orgsQueryOptions)
  const { data: projects } = useQuery(projectsQueryOptions)
  const { data: orgRoleData } = useQuery(currentUserOrgRoleQueryOptions)
  if (!session || !orgs || !projects || !orgRoleData) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }
  const navProjects = (projects ?? []) as ProjectNavItem[]
  const [activeKey, setActiveKey] = useState<SettingsKey>("general")

  const setActiveSettingsKey = (key: SettingsKey) => {
    setActiveKey(key)
    navigate({
      to: "/settings",
      search: {
        section: key,
      },
      replace: true,
    })
  }

  const selectedProjectId =
    activeKey.startsWith("project:") ? activeKey.replace("project:", "") : null
  const selectedPackageId =
    activeKey.startsWith("package:") ? activeKey.replace("package:", "") : null

  const orgMembersQ = useQuery({
    ...orgMembersQueryOptions,
    enabled: activeKey === "members",
  })
  const orgPendingInvitesQ = useQuery({
    ...orgPendingInvitesQueryOptions,
    enabled: activeKey === "members",
  })
  const orgMembers = orgMembersQ.data ?? []
  const orgPendingInvites = orgPendingInvitesQ.data ?? []

  const archivedProjectsQ = useQuery({
    ...archivedProjectsQueryOptions,
    enabled: !!selectedProjectId,
  })
  const archivedPackagesQ = useQuery({
    ...archivedPackagesQueryOptions,
    enabled: !!selectedPackageId,
  })
  const archivedProjects = archivedProjectsQ.data ?? []
  const archivedPackages = archivedPackagesQ.data ?? []

  const activeOrg = orgs.find(
    (org) => org.id === session?.session?.activeOrganizationId
  )
  const canManageOrganization = orgRoleData.role === "owner"

  useEffect(() => {
    const section = search.section
    if (!section) return
    if (
      section === "general" ||
      section === "organization" ||
      section === "members" ||
      section.startsWith("project:") ||
      section.startsWith("package:")
    ) {
      setActiveKey(section as SettingsKey)
    }
  }, [search.section])

  useEffect(() => {
    if (
      selectedProjectId &&
      !navProjects.some((project) => project.id === selectedProjectId)
    ) {
      setActiveSettingsKey("general")
    }
  }, [selectedProjectId, navProjects])

  useEffect(() => {
    if (
      selectedPackageId &&
      !navProjects.some((project) =>
        project.packages.some((pkg) => pkg.id === selectedPackageId)
      )
    ) {
      setActiveSettingsKey("general")
    }
  }, [selectedPackageId, navProjects])

  const baseInvalidation = () => {
    queryClient.invalidateQueries({ queryKey: sessionQueryOptions.queryKey })
    queryClient.invalidateQueries({ queryKey: orgsQueryOptions.queryKey })
    queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
  }

  // General/profile
  const [profileName, setProfileName] = useState(session?.user?.name ?? "")
  useEffect(() => {
    setProfileName(session?.user?.name ?? "")
  }, [session?.user?.name])

  const updateProfile = useMutation({
    mutationFn: (name: string) => updateProfileFn({ data: { name } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sessionQueryOptions.queryKey })
      toast.success("Profile updated")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update profile"
      )
    },
  })

  const logout = useMutation({
    mutationFn: async () => {
      const result = await authClient.signOut()
      if (result.error) {
        throw new Error(result.error.message ?? "Failed to log out")
      }
    },
    onSuccess: () => {
      queryClient.clear()
      navigate({ to: "/login" })
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to log out")
    },
  })

  // Organization
  const [orgName, setOrgName] = useState(activeOrg?.name ?? "")
  const [orgCountry, setOrgCountry] = useState(
    getOrgCountry(activeOrg?.metadata ?? null)
  )
  useEffect(() => {
    setOrgName(activeOrg?.name ?? "")
    setOrgCountry(getOrgCountry(activeOrg?.metadata ?? null))
  }, [activeOrg])

  const updateOrganization = useMutation({
    mutationFn: (input: { name: string; country: string }) =>
      updateOrganizationFn({
        data: {
          name: input.name.trim(),
          metadata: { country: input.country },
        },
      }),
    onSuccess: () => {
      baseInvalidation()
      toast.success("Organization updated")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update organization"
      )
    },
  })

  // Members
  const [orgInviteEmail, setOrgInviteEmail] = useState("")
  const [orgInviteRole, setOrgInviteRole] = useState<OrgMemberRole>("member")

  const inviteOrgMember = useMutation({
    mutationFn: (input: { email: string; role: OrgMemberRole }) =>
      inviteMemberFn({ data: input }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: orgMembersQueryOptions.queryKey })
      queryClient.invalidateQueries({
        queryKey: orgPendingInvitesQueryOptions.queryKey,
      })
      setOrgInviteEmail("")
      setOrgInviteRole("member")
      toast.success("Invitation sent")
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to invite")
    },
  })

  const removeOrgMember = useMutation({
    mutationFn: (userId: string) => removeOrgMemberFn({ data: { userId } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: orgMembersQueryOptions.queryKey })
      toast.success("Member removed")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to remove member"
      )
    },
  })

  const cancelInvite = useMutation({
    mutationFn: (invitationId: string) =>
      cancelOrgInvitationFn({ data: { invitationId } }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: orgPendingInvitesQueryOptions.queryKey,
      })
      toast.success("Invite canceled")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to cancel invite"
      )
    },
  })

  // Selected project
  const selectedProjectDetailQ = useQuery({
    ...projectDetailQueryOptions(selectedProjectId ?? ""),
    enabled: !!selectedProjectId,
  })
  const selectedProjectAccessQ = useQuery({
    ...projectAccessQueryOptions(selectedProjectId ?? ""),
    enabled: !!selectedProjectId,
  })
  const selectedProjectMembersQ = useQuery({
    ...projectMembersQueryOptions(selectedProjectId ?? ""),
    enabled: !!selectedProjectId,
  })
  const selectedProjectMembers = selectedProjectMembersQ.data ?? []
  const canManageSelectedProject = selectedProjectAccessQ.data?.access === "full"
  const selectedProjectNav =
    selectedProjectId == null
      ? null
      : navProjects.find((project) => project.id === selectedProjectId) ?? null

  const [projectName, setProjectName] = useState("")
  const [projectCountry, setProjectCountry] = useState("US")
  const [projectInviteEmail, setProjectInviteEmail] = useState("")
  const [projectInviteRole, setProjectInviteRole] =
    useState<ProjectMemberRole>("project_lead")

  useEffect(() => {
    if (selectedProjectDetailQ.data?.project) {
      setProjectName(selectedProjectDetailQ.data.project.name)
      setProjectCountry(selectedProjectDetailQ.data.project.country ?? "US")
    }
  }, [selectedProjectDetailQ.data])

  const renameProject = useMutation({
    mutationFn: (name: string) =>
      renameProjectFn({ data: { projectId: selectedProjectId ?? "", name } }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(selectedProjectId ?? "").queryKey,
      })
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      toast.success("Project renamed")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to rename project"
      )
    },
  })

  const changeProjectCountry = useMutation({
    mutationFn: (country: string) =>
      updateProjectCountryFn({
        data: { projectId: selectedProjectId ?? "", country },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(selectedProjectId ?? "").queryKey,
      })
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      toast.success("Project country updated")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update country"
      )
    },
  })

  const archiveProject = useMutation({
    mutationFn: () => archiveProjectFn({ data: { projectId: selectedProjectId ?? "" } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      queryClient.invalidateQueries({
        queryKey: archivedProjectsQueryOptions.queryKey,
      })
      toast.success("Project archived")
      setActiveSettingsKey("general")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to archive project"
      )
    },
  })

  const restoreProject = useMutation({
    mutationFn: (projectId: string) => restoreProjectFn({ data: { projectId } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      queryClient.invalidateQueries({
        queryKey: archivedProjectsQueryOptions.queryKey,
      })
      toast.success("Project restored")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to restore project"
      )
    },
  })

  const addProjectMember = useMutation({
    mutationFn: (input: { email: string; role: ProjectMemberRole }) =>
      addProjectMemberFn({
        data: { projectId: selectedProjectId ?? "", ...input },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectMembersQueryOptions(selectedProjectId ?? "").queryKey,
      })
      setProjectInviteEmail("")
      setProjectInviteRole("project_lead")
      toast.success("Project member added")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to add project member"
      )
    },
  })

  const deleteProjectMember = useMutation({
    mutationFn: (email: string) =>
      removeProjectMemberFn({
        data: { projectId: selectedProjectId ?? "", email },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectMembersQueryOptions(selectedProjectId ?? "").queryKey,
      })
      toast.success("Project member removed")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to remove project member"
      )
    },
  })

  // Selected package
  const selectedPackageDetailQ = useQuery({
    ...packageDetailQueryOptions(selectedPackageId ?? ""),
    enabled: !!selectedPackageId,
  })
  const selectedPackageAccessQ = useQuery({
    ...packageAccessQueryOptions(selectedPackageId ?? ""),
    enabled: !!selectedPackageId,
  })
  const selectedPackageMembersQ = useQuery({
    ...packageMembersQueryOptions(selectedPackageId ?? ""),
    enabled: !!selectedPackageId,
  })
  const selectedPackageMembers = selectedPackageMembersQ.data ?? []
  const canManageSelectedPackage = selectedPackageAccessQ.data?.access === "full"

  let selectedPackageNav: { pkg: { id: string; name: string }; project: ProjectNavItem } | null =
    null
  if (selectedPackageId) {
    for (const project of navProjects) {
      const pkg = project.packages.find((item) => item.id === selectedPackageId)
      if (pkg) {
        selectedPackageNav = { pkg, project }
        break
      }
    }
  }

  const [packageName, setPackageName] = useState("")
  const [packageCurrency, setPackageCurrency] = useState("USD")
  const [packageInviteEmail, setPackageInviteEmail] = useState("")
  const [packageInviteRole, setPackageInviteRole] =
    useState<PackageMemberRole>("package_lead")

  useEffect(() => {
    if (selectedPackageDetailQ.data?.package) {
      setPackageName(selectedPackageDetailQ.data.package.name)
      setPackageCurrency(selectedPackageDetailQ.data.package.currency ?? "USD")
    }
  }, [selectedPackageDetailQ.data])

  const renamePackage = useMutation({
    mutationFn: (name: string) =>
      renamePackageFn({ data: { packageId: selectedPackageId ?? "", name } }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(selectedPackageId ?? "").queryKey,
      })
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      toast.success("Package renamed")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to rename package"
      )
    },
  })

  const changePackageCurrency = useMutation({
    mutationFn: (currency: string) =>
      updatePackageCurrencyFn({
        data: { packageId: selectedPackageId ?? "", currency },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(selectedPackageId ?? "").queryKey,
      })
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      toast.success("Package currency updated")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update currency"
      )
    },
  })

  const archivePackage = useMutation({
    mutationFn: () => archivePackageFn({ data: { packageId: selectedPackageId ?? "" } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      queryClient.invalidateQueries({
        queryKey: archivedPackagesQueryOptions.queryKey,
      })
      toast.success("Package archived")
      setActiveSettingsKey("general")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to archive package"
      )
    },
  })

  const restorePackage = useMutation({
    mutationFn: (packageId: string) => restorePackageFn({ data: { packageId } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      queryClient.invalidateQueries({
        queryKey: archivedPackagesQueryOptions.queryKey,
      })
      toast.success("Package restored")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to restore package"
      )
    },
  })

  const addPackageMember = useMutation({
    mutationFn: (input: { email: string; role: PackageMemberRole }) =>
      addPackageMemberFn({
        data: { packageId: selectedPackageId ?? "", ...input },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: packageMembersQueryOptions(selectedPackageId ?? "").queryKey,
      })
      setPackageInviteEmail("")
      setPackageInviteRole("package_lead")
      toast.success("Package member added")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to add package member"
      )
    },
  })

  const deletePackageMember = useMutation({
    mutationFn: (email: string) =>
      removePackageMemberFn({
        data: { packageId: selectedPackageId ?? "", email },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: packageMembersQueryOptions(selectedPackageId ?? "").queryKey,
      })
      toast.success("Package member removed")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to remove package member"
      )
    },
  })

  return (
    <div className="h-screen w-full flex justify-center bg-[#F9F9F9] px-6 overflow-hidden">
      <div className="w-full max-w-[960px] h-screen flex gap-6">
        <aside className="basis-[260px] w-[260px] min-w-[260px] max-w-[260px] h-screen shrink-0 overflow-y-auto py-6">
          <button
            className="top-back-link h-8 px-2 -ml-2 inline-flex items-center gap-1 hover:opacity-100"
            onClick={() => {
              if (window.history.length > 1) {
                window.history.back()
                return
              }
              navigate({ to: "/" })
            }}
            type="button"
          >
            <ArrowLeft size={14} />
            Go back
          </button>

          <h1 className="mt-2 text-[24px] leading-8 font-semibold text-black">Settings</h1>

          <nav className="mt-4 space-y-1">
            <button
              className={navItemClass(activeKey === "general")}
              onClick={() => setActiveSettingsKey("general")}
              type="button"
            >
              <Settings size={16} className="opacity-70" />
              <span className="ml-[6px] truncate opacity-70">General</span>
            </button>
            <button
              className={navItemClass(activeKey === "organization")}
              onClick={() => setActiveSettingsKey("organization")}
              type="button"
            >
              <Building2 size={16} className="opacity-70" />
              <span className="ml-[6px] truncate opacity-70">Organization</span>
            </button>
            <button
              className={navItemClass(activeKey === "members")}
              onClick={() => setActiveSettingsKey("members")}
              type="button"
            >
              <Users size={16} className="opacity-70" />
              <span className="ml-[6px] truncate opacity-70">Members</span>
            </button>
          </nav>

          <div className="mt-6">
            <div className="h-8 px-2 flex items-center text-[11px] font-medium opacity-40">
              Projects & packages
            </div>
            <div className="space-y-1 pr-1">
              {navProjects.map((project) => (
                <div key={project.id}>
                  <button
                    className={navItemClass(activeKey === `project:${project.id}`)}
                    onClick={() => setActiveSettingsKey(`project:${project.id}`)}
                    type="button"
                  >
                    <FolderOpen size={16} className="opacity-70" />
                    <span className="ml-[6px] truncate opacity-70">{project.name}</span>
                  </button>
                  {project.packages.map((pkg) => (
                    <button
                      key={pkg.id}
                      className={cn(navItemClass(activeKey === `package:${pkg.id}`), "pl-8")}
                      onClick={() => setActiveSettingsKey(`package:${pkg.id}`)}
                      type="button"
                    >
                      <span className="truncate opacity-70">{pkg.name}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="flex-1 overflow-auto pt-16 pb-10">
          <div className="max-w-[700px] space-y-8 pb-8">
            {activeKey === "general" && (
              <section className="space-y-4">
                <h2 className="text-[24px] leading-8 font-semibold text-black">General</h2>
                <h3 className={settingsSectionTitleClass}>PROFILE</h3>
                <div className={settingsCardClass}>
                  <div className="grid gap-4 md:grid-cols-[1fr_220px]">
                    <div className="space-y-2">
                      <Label htmlFor="profile-name">Name</Label>
                      <Input
                        id="profile-name"
                        value={profileName}
                        onChange={(e) => setProfileName(e.target.value)}
                        placeholder="Your name"
                        className="bg-white"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Email</Label>
                      <div className="h-9 rounded-md border bg-[#FAFAFA] px-3 text-sm flex items-center text-black/70">
                        {session?.user?.email ?? "Unknown"}
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-end">
                    <Button
                      size="sm"
                      onClick={() => {
                        const name = profileName.trim()
                        if (!name) return
                        updateProfile.mutate(name)
                      }}
                      disabled={updateProfile.isPending || !profileName.trim()}
                    >
                      {updateProfile.isPending ? "Saving..." : "Save changes"}
                    </Button>
                  </div>
                </div>

                <h3 className={settingsSectionTitleClass}>ACCOUNT</h3>
                <div className={settingsCardClass}>
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium text-black">Session</p>
                      <p className="text-xs text-black/55">
                        Log out from this device and return to sign in.
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => logout.mutate()}
                      disabled={logout.isPending}
                    >
                      {logout.isPending ? "Logging out..." : "Log out"}
                    </Button>
                  </div>
                </div>
              </section>
            )}

            {activeKey === "organization" && (
              <section className="space-y-4">
                <h2 className="text-[24px] leading-8 font-semibold text-black">Organization</h2>
                <h3 className={settingsSectionTitleClass}>ORGANIZATION DETAILS</h3>
                <div className={settingsCardClass}>
                  <div className="space-y-2">
                    <Label htmlFor="org-name">Organization name</Label>
                    <Input
                      id="org-name"
                      value={orgName}
                      onChange={(e) => setOrgName(e.target.value)}
                      placeholder="Organization name"
                      disabled={!canManageOrganization}
                      className="bg-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Country</Label>
                    <CountrySelect
                      value={orgCountry}
                      onValueChange={setOrgCountry}
                      disabled={!canManageOrganization}
                    />
                  </div>
                  {!canManageOrganization && (
                    <div className="rounded-md border border-[#EDEDED] bg-[#FAFAFA] px-3 py-2 text-xs text-black/55">
                      You have view-only access for organization settings.
                    </div>
                  )}
                  <div className="flex justify-end">
                    <Button
                      size="sm"
                      onClick={() =>
                        updateOrganization.mutate({
                          name: orgName,
                          country: orgCountry,
                        })
                      }
                      disabled={
                        !canManageOrganization ||
                        updateOrganization.isPending ||
                        !orgName.trim()
                      }
                    >
                      {updateOrganization.isPending ? "Saving..." : "Save changes"}
                    </Button>
                  </div>
                </div>
              </section>
            )}

            {activeKey === "members" && (
              <section className="space-y-4">
                <h2 className="text-[24px] leading-8 font-semibold text-black">Members</h2>

                {orgMembersQ.isPending || orgPendingInvitesQ.isPending ? (
                  <SectionLoading />
                ) : (
                  <>

                    {canManageOrganization && (
                      <>
                        <h3 className={settingsSectionTitleClass}>INVITE MEMBER</h3>
                        <div className={settingsCardClass}>
                          <div className="grid grid-cols-1 md:grid-cols-[1fr_150px_auto] gap-2">
                            <Input
                              value={orgInviteEmail}
                              onChange={(e) => setOrgInviteEmail(e.target.value)}
                              placeholder="name@company.com"
                            />
                            <select
                              className="h-9 rounded-md border bg-white px-3 text-sm"
                              value={orgInviteRole}
                              onChange={(e) =>
                                setOrgInviteRole(e.target.value as OrgMemberRole)
                              }
                            >
                              <option value="member">Member</option>
                              <option value="admin">Admin</option>
                              <option value="owner">Owner</option>
                            </select>
                            <Button
                              size="sm"
                              onClick={() =>
                                inviteOrgMember.mutate({
                                  email: orgInviteEmail.trim(),
                                  role: orgInviteRole,
                                })
                              }
                              disabled={inviteOrgMember.isPending || orgInviteEmail.trim().length === 0}
                            >
                              Invite
                            </Button>
                          </div>
                        </div>
                      </>
                    )}

                    <h3 className={settingsSectionTitleClass}>ACTIVE MEMBERS</h3>
                    <div className={settingsCardClass}>
                      {orgMembers.length === 0 ? (
                        <p className="text-sm text-black/55">No members found.</p>
                      ) : (
                        <div className="divide-y divide-[#F0F0F0]">
                          {orgMembers.map((member) => (
                            <div
                              key={member.id}
                              className="py-2.5 flex items-center justify-between gap-3"
                            >
                              <div className="min-w-0 flex items-center gap-3">
                                <div className="size-8 rounded-full bg-[#F3F3F3] flex items-center justify-center text-xs font-semibold text-black/60">
                                  {(member.userName ?? member.email).charAt(0).toUpperCase()}
                                </div>
                                <div className="min-w-0">
                                  <p className="text-sm font-medium truncate">
                                    {member.userName ?? member.email}
                                  </p>
                                  <p className="text-xs text-black/50 truncate">{member.email}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2 shrink-0">
                                <span className="rounded-full border border-[#E8E8E8] px-2 py-0.5 text-[11px] font-medium text-black/60">
                                  {member.role}
                                </span>
                                {canManageOrganization && member.userId !== session?.user?.id && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => {
                                      if (
                                        window.confirm(
                                          "Remove this member from the organization?"
                                        )
                                      ) {
                                        removeOrgMember.mutate(member.userId)
                                      }
                                    }}
                                    disabled={removeOrgMember.isPending}
                                  >
                                    Remove
                                  </Button>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <h3 className={settingsSectionTitleClass}>PENDING INVITES</h3>
                    <div className={settingsCardClass}>
                      {orgPendingInvites.length === 0 ? (
                        <p className="text-sm text-black/55">No pending invites.</p>
                      ) : (
                        <div className="divide-y divide-[#F0F0F0]">
                          {orgPendingInvites.map((invite) => (
                            <div
                              key={invite.id}
                              className="py-2.5 flex items-center justify-between gap-3"
                            >
                              <div className="min-w-0">
                                <p className="text-sm font-medium truncate">{invite.email}</p>
                                <p className="text-xs text-black/50">{invite.role}</p>
                              </div>
                              {canManageOrganization && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    if (window.confirm("Cancel this pending invite?")) {
                                      cancelInvite.mutate(invite.id)
                                    }
                                  }}
                                  disabled={cancelInvite.isPending}
                                >
                                  Cancel
                                </Button>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </section>
            )}

            {selectedProjectId && (
              <section className="space-y-4">
                <h2 className="text-[24px] leading-8 font-semibold text-black">
                  {selectedProjectNav?.name ?? "Project"}
                </h2>

                {selectedProjectDetailQ.isPending ||
                selectedProjectAccessQ.isPending ||
                selectedProjectMembersQ.isPending ||
                archivedProjectsQ.isPending ? (
                  <SectionLoading />
                ) : (
                  <>
                    <h3 className={settingsSectionTitleClass}>PROJECT DETAILS</h3>
                    <div className={settingsCardClass}>
                      <div className="space-y-2">
                        <Label htmlFor="project-name">Project name</Label>
                        <Input
                          id="project-name"
                          value={projectName}
                          onChange={(e) => setProjectName(e.target.value)}
                          disabled={!canManageSelectedProject}
                          className="bg-white"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Country</Label>
                        <CountrySelect
                          value={projectCountry}
                          onValueChange={setProjectCountry}
                          disabled={!canManageSelectedProject}
                        />
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            if (
                              canManageSelectedProject &&
                              window.confirm(
                                "Archive this project? It will be moved to archived projects."
                              )
                            ) {
                              archiveProject.mutate()
                            }
                          }}
                          disabled={!canManageSelectedProject || archiveProject.isPending}
                        >
                          Archive project
                        </Button>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => changeProjectCountry.mutate(projectCountry)}
                            disabled={!canManageSelectedProject || changeProjectCountry.isPending}
                          >
                            Save country
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => renameProject.mutate(projectName.trim())}
                            disabled={
                              !canManageSelectedProject ||
                              renameProject.isPending ||
                              !projectName.trim()
                            }
                          >
                            Save name
                          </Button>
                        </div>
                      </div>
                    </div>

                    <h3 className={settingsSectionTitleClass}>PROJECT MEMBERS</h3>
                    <div className={settingsCardClass}>
                      {canManageSelectedProject && (
                        <div className="grid grid-cols-1 md:grid-cols-[1fr_170px_auto] gap-2">
                          <Input
                            value={projectInviteEmail}
                            onChange={(e) => setProjectInviteEmail(e.target.value)}
                            placeholder="name@company.com"
                          />
                          <select
                            className="h-9 rounded-md border bg-white px-3 text-sm"
                            value={projectInviteRole}
                            onChange={(e) =>
                              setProjectInviteRole(e.target.value as ProjectMemberRole)
                            }
                          >
                            <option value="project_lead">Project lead</option>
                            <option value="commercial_lead">Commercial lead</option>
                            <option value="technical_lead">Technical lead</option>
                          </select>
                          <Button
                            size="sm"
                            onClick={() =>
                              addProjectMember.mutate({
                                email: projectInviteEmail.trim(),
                                role: projectInviteRole,
                              })
                            }
                            disabled={addProjectMember.isPending || projectInviteEmail.trim().length === 0}
                          >
                            Add member
                          </Button>
                        </div>
                      )}

                      {selectedProjectMembers.length === 0 ? (
                        <p className="text-sm text-black/55">No members found.</p>
                      ) : (
                        <div className="divide-y divide-[#F0F0F0]">
                          {selectedProjectMembers.map((member) => (
                            <div
                              key={member.id}
                              className="py-2.5 flex items-center justify-between gap-3"
                            >
                              <div className="min-w-0">
                                <p className="text-sm font-medium truncate">
                                  {member.userName ?? member.email}
                                </p>
                                <p className="text-xs text-black/50 truncate">
                                  {member.email} - {member.role}
                                </p>
                              </div>
                              {canManageSelectedProject && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    if (
                                      window.confirm(
                                        "Remove this member from the selected project?"
                                      )
                                    ) {
                                      deleteProjectMember.mutate(member.email)
                                    }
                                  }}
                                  disabled={deleteProjectMember.isPending}
                                >
                                  Remove
                                </Button>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <h3 className={settingsSectionTitleClass}>ARCHIVED PROJECTS</h3>
                    <div className={settingsCardClass}>
                      {archivedProjects.length === 0 ? (
                        <p className="text-sm text-black/55">No archived projects.</p>
                      ) : (
                        <div className="divide-y divide-[#F0F0F0]">
                          {archivedProjects.map((project) => (
                            <div
                              key={project.id}
                              className="py-2.5 flex items-center justify-between gap-3"
                            >
                              <p className="text-sm font-medium truncate">{project.name}</p>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => restoreProject.mutate(project.id)}
                                disabled={restoreProject.isPending}
                              >
                                Restore
                              </Button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </section>
            )}

            {selectedPackageId && (
              <section className="space-y-4">
                <h2 className="text-[24px] leading-8 font-semibold text-black">
                  {selectedPackageNav?.pkg.name ?? "Package"}
                </h2>

                {selectedPackageDetailQ.isPending ||
                selectedPackageAccessQ.isPending ||
                selectedPackageMembersQ.isPending ||
                archivedPackagesQ.isPending ? (
                  <SectionLoading />
                ) : (
                  <>
                    <h3 className={settingsSectionTitleClass}>PACKAGE DETAILS</h3>
                    <div className={settingsCardClass}>
                      <div className="space-y-2">
                        <Label htmlFor="package-name">Package name</Label>
                        <Input
                          id="package-name"
                          value={packageName}
                          onChange={(e) => setPackageName(e.target.value)}
                          disabled={!canManageSelectedPackage}
                          className="bg-white"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Currency</Label>
                        <CurrencySelect
                          value={packageCurrency}
                          onValueChange={setPackageCurrency}
                          disabled={!canManageSelectedPackage}
                        />
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            if (
                              canManageSelectedPackage &&
                              window.confirm(
                                "Archive this package? It will be moved to archived packages."
                              )
                            ) {
                              archivePackage.mutate()
                            }
                          }}
                          disabled={!canManageSelectedPackage || archivePackage.isPending}
                        >
                          Archive package
                        </Button>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => changePackageCurrency.mutate(packageCurrency)}
                            disabled={!canManageSelectedPackage || changePackageCurrency.isPending}
                          >
                            Save currency
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => renamePackage.mutate(packageName.trim())}
                            disabled={
                              !canManageSelectedPackage ||
                              renamePackage.isPending ||
                              !packageName.trim()
                            }
                          >
                            Save name
                          </Button>
                        </div>
                      </div>
                    </div>

                    <h3 className={settingsSectionTitleClass}>PACKAGE MEMBERS</h3>
                    <div className={settingsCardClass}>
                      {canManageSelectedPackage && (
                        <div className="grid grid-cols-1 md:grid-cols-[1fr_170px_auto] gap-2">
                          <Input
                            value={packageInviteEmail}
                            onChange={(e) => setPackageInviteEmail(e.target.value)}
                            placeholder="name@company.com"
                          />
                          <select
                            className="h-9 rounded-md border bg-white px-3 text-sm"
                            value={packageInviteRole}
                            onChange={(e) =>
                              setPackageInviteRole(e.target.value as PackageMemberRole)
                            }
                          >
                            <option value="package_lead">Package lead</option>
                            <option value="commercial_team">Commercial team</option>
                            <option value="technical_team">Technical team</option>
                          </select>
                          <Button
                            size="sm"
                            onClick={() =>
                              addPackageMember.mutate({
                                email: packageInviteEmail.trim(),
                                role: packageInviteRole,
                              })
                            }
                            disabled={addPackageMember.isPending || packageInviteEmail.trim().length === 0}
                          >
                            Add member
                          </Button>
                        </div>
                      )}

                      {selectedPackageMembers.length === 0 ? (
                        <p className="text-sm text-black/55">No members found.</p>
                      ) : (
                        <div className="divide-y divide-[#F0F0F0]">
                          {selectedPackageMembers.map((member) => (
                            <div
                              key={member.id}
                              className="py-2.5 flex items-center justify-between gap-3"
                            >
                              <div className="min-w-0">
                                <p className="text-sm font-medium truncate">
                                  {member.userName ?? member.email}
                                </p>
                                <p className="text-xs text-black/50 truncate">
                                  {member.email} - {member.role}
                                </p>
                              </div>
                              {canManageSelectedPackage && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    if (
                                      window.confirm(
                                        "Remove this member from the selected package?"
                                      )
                                    ) {
                                      deletePackageMember.mutate(member.email)
                                    }
                                  }}
                                  disabled={deletePackageMember.isPending}
                                >
                                  Remove
                                </Button>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <h3 className={settingsSectionTitleClass}>ARCHIVED PACKAGES</h3>
                    <div className={settingsCardClass}>
                      {archivedPackages.length === 0 ? (
                        <p className="text-sm text-black/55">No archived packages.</p>
                      ) : (
                        <div className="divide-y divide-[#F0F0F0]">
                          {archivedPackages.map((pkg) => (
                            <div
                              key={pkg.id}
                              className="py-2.5 flex items-center justify-between gap-3"
                            >
                              <div className="min-w-0">
                                <p className="text-sm font-medium truncate">{pkg.name}</p>
                                <p className="text-xs text-black/50 truncate">{pkg.projectName}</p>
                              </div>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => restorePackage.mutate(pkg.id)}
                                disabled={restorePackage.isPending}
                              >
                                Restore
                              </Button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </section>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
