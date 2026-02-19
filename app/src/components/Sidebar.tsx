import { Link, useLocation, useNavigate } from "@tanstack/react-router"
import {
  useQuery,
  useMutation,
  useQueryClient,
  useSuspenseQuery,
} from "@tanstack/react-query"
import { useState, useRef, useCallback, memo, useLayoutEffect } from "react"
import useStore, { SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH } from "@/lib/store"
import { cn, getOrgCountry } from "@/lib/utils"
import {
  PanelRight,
  Settings,
  Folder,
  Package,
  Plus,
  LogOutIcon,
  SettingsIcon,
  LayoutDashboard,
  FolderKanban,
  Building2,
} from "lucide-react"
import {
  projectsQueryOptions,
  orgsQueryOptions,
  sessionQueryOptions,
  currentUserOrgRoleQueryOptions,
} from "@/lib/query-options"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { setActiveOrgFn } from "@/fn/auth"
import { createProjectFn } from "@/fn/projects"
import { authClient } from "@/auth/auth-client"
import { SettingsDialog } from "@/components/SettingsDialog"
import { CountrySelect } from "@/components/CountrySelect"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

// ============================================================================
// Nav Link
// ============================================================================

function NavLink({
  to,
  icon: Icon,
  children,
}: {
  to: string
  icon: React.ComponentType<{ size?: number; className?: string }>
  children: React.ReactNode
}) {
  const location = useLocation()
  const isActive = location.pathname === to

  return (
    <Link
      to={to}
      className={cn(
        "nav-item nav-item-dark gap-2 min-w-0",
        isActive && "active"
      )}
    >
      <Icon size={14} className="shrink-0" />
      <span className="truncate">{children}</span>
    </Link>
  )
}

// ============================================================================
// Collapsed Nav Icon
// ============================================================================

function CollapsedNavIcon({
  to,
  icon: Icon,
  tooltip,
}: {
  to: string
  icon: React.ComponentType<{ size?: number; className?: string }>
  tooltip: string
}) {
  const location = useLocation()
  const isActive = location.pathname === to

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Link
          to={to}
          className={cn(
            "size-7 flex items-center justify-center rounded-md transition-colors",
            isActive
              ? "bg-white/20 shadow-[inset_0_0_0_0.5px_rgba(255,255,255,0.2)]"
              : "hover:bg-black/20"
          )}
        >
          <Icon size={14} />
        </Link>
      </TooltipTrigger>
      <TooltipContent side="right">{tooltip}</TooltipContent>
    </Tooltip>
  )
}

// ============================================================================
// Route Detection Hook
// ============================================================================

function useCurrentRoute() {
  const location = useLocation()
  const pathname = location.pathname

  const projectMatch = pathname.match(/^\/project\/([^/]+)/)
  const packageMatch = pathname.match(/^\/package\/([^/]+)/)

  return {
    projectId: projectMatch?.[1] ?? null,
    packageId: packageMatch?.[1] ?? null,
    isOnProject: !!projectMatch,
    isOnPackage: !!packageMatch,
  }
}

// ============================================================================
// Package Item (memoized)
// ============================================================================

const PackageItem = memo(function PackageItem({
  pkg,
  isActive,
}: {
  pkg: { id: string; name: string }
  isActive: boolean
}) {
  return (
    <Link
      to="/package/$id"
      params={{ id: pkg.id }}
      className={cn("nav-item nav-item-dark gap-2", isActive && "active")}
    >
      <Package size={14} className="shrink-0 opacity-60" />
      <span className="truncate">{pkg.name}</span>
    </Link>
  )
})

// ============================================================================
// Project Item with Packages (memoized)
// ============================================================================

const ProjectItem = memo(function ProjectItem({
  project,
  isActive,
  activePackageId,
}: {
  project: {
    id: string
    name: string
    packages: { id: string; name: string }[]
  }
  isActive: boolean
  activePackageId: string | null
}) {
  return (
    <>
      <Link
        to="/project/$id"
        params={{ id: project.id }}
        className={cn("nav-item nav-item-dark gap-2", isActive && "active")}
      >
        <Folder size={14} className="shrink-0 opacity-60" />
        <span className="truncate">{project.name}</span>
      </Link>
      {project.packages.length > 0 && (
        <div className="pl-4 flex flex-col gap-px">
          {project.packages.map((pkg) => (
            <PackageItem
              key={pkg.id}
              pkg={pkg}
              isActive={activePackageId === pkg.id}
            />
          ))}
        </div>
      )}
    </>
  )
})

// ============================================================================
// Project Tree
// ============================================================================

function ProjectTree() {
  const route = useCurrentRoute()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [projectName, setProjectName] = useState("")

  // Always fetch the projects list
  const { data: projects = [] } = useQuery(projectsQueryOptions)
  const { data: orgs } = useSuspenseQuery(orgsQueryOptions)
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const { data: orgRoleData } = useQuery(currentUserOrgRoleQueryOptions)

  const canCreateProject =
    orgRoleData?.role === "owner" || orgRoleData?.role === "admin"

  const activeOrg = orgs?.find(
    (o) => o.id === session?.session?.activeOrganizationId
  )
  const orgCountry = getOrgCountry(activeOrg?.metadata)
  const [projectCountry, setProjectCountry] = useState(orgCountry)

  const createProject = useMutation({
    mutationFn: ({ name, country }: { name: string; country: string }) =>
      createProjectFn({ data: { name, country } }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: projectsQueryOptions.queryKey })
      closeDrawer()
      navigate({ to: "/project/$id", params: { id: data.id } })
    },
  })

  const closeDrawer = () => {
    setIsDrawerOpen(false)
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

  // Determine active states for highlighting
  const activeProjectId = route.isOnProject ? route.projectId : null
  const activePackageId = route.isOnPackage ? route.packageId : null

  return (
    <>
      <div className="px-2 py-2 min-w-0">
        <div className="flex items-center justify-between px-2 py-1 mb-1 min-w-0">
          <span className="text-xs font-medium text-white/50 tracking-wide truncate">
            Projects
          </span>
          {canCreateProject && (
            <button
              onClick={() => setIsDrawerOpen(true)}
              className="p-1 hover:bg-white/10 rounded transition-colors shrink-0"
              title="New project"
            >
              <Plus size={14} />
            </button>
          )}
        </div>
        {projects.length === 0 ? (
          <div className="px-2 py-2 text-sm text-white/50">No projects yet</div>
        ) : (
          <div className="flex flex-col gap-px">
            {projects.map((project) => (
              <ProjectItem
                key={project.id}
                project={project}
                isActive={activeProjectId === project.id}
                activePackageId={activePackageId}
              />
            ))}
          </div>
        )}
      </div>

      <Sheet
        open={isDrawerOpen}
        onOpenChange={(open) => !open && closeDrawer()}
      >
        <SheetContent className="min-w-[500px] sm:max-w-none">
          <form className="space-y-6" onSubmit={handleCreate}>
            <SheetHeader>
              <SheetTitle>Create Project</SheetTitle>
              <SheetDescription>
                Create a new project for your organization.
              </SheetDescription>
            </SheetHeader>
            <div className="px-2 space-y-4">
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
              ) : (
                <p className="text-xs text-muted-foreground">
                  Projects are created inside your active organization.
                </p>
              )}
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

// ============================================================================
// Collapsed Project Item (memoized)
// ============================================================================

const CollapsedProjectItem = memo(function CollapsedProjectItem({
  project,
  isActive,
  activePackageId,
}: {
  project: {
    id: string
    name: string
    packages: { id: string; name: string }[]
  }
  isActive: boolean
  activePackageId: string | null
}) {
  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Link
            to="/project/$id"
            params={{ id: project.id }}
            className={cn(
              "size-7 flex items-center justify-center rounded-md transition-colors",
              isActive
                ? "bg-white/20 shadow-[inset_0_0_0_0.5px_rgba(255,255,255,0.2)]"
                : "hover:bg-black/20"
            )}
          >
            <Folder size={14} />
          </Link>
        </TooltipTrigger>
        <TooltipContent side="right">{project.name}</TooltipContent>
      </Tooltip>
      {project.packages.map((pkg) => (
        <Tooltip key={pkg.id}>
          <TooltipTrigger asChild>
            <Link
              to="/package/$id"
              params={{ id: pkg.id }}
              className={cn(
                "size-7 flex items-center justify-center rounded-md transition-colors",
                activePackageId === pkg.id
                  ? "bg-white/20 shadow-[inset_0_0_0_0.5px_rgba(255,255,255,0.2)]"
                  : "hover:bg-black/20"
              )}
            >
              <Package size={14} />
            </Link>
          </TooltipTrigger>
          <TooltipContent side="right">{pkg.name}</TooltipContent>
        </Tooltip>
      ))}
    </>
  )
})

// ============================================================================
// Collapsed Project Tree
// ============================================================================

function CollapsedProjectTree() {
  const route = useCurrentRoute()

  const { data: projects = [] } = useQuery(projectsQueryOptions)

  const activeProjectId = route.isOnProject ? route.projectId : null
  const activePackageId = route.isOnPackage ? route.packageId : null

  if (projects.length === 0) return null

  return (
    <div className="flex flex-col items-center gap-1">
      {projects.map((project) => (
        <CollapsedProjectItem
          key={project.id}
          project={project}
          isActive={activeProjectId === project.id}
          activePackageId={activePackageId}
        />
      ))}
    </div>
  )
}

// ============================================================================
// Collapsed Account Button
// ============================================================================

function CollapsedAccountButton({
  onSettingsClick,
}: {
  onSettingsClick: () => void
}) {
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const user = session?.user
  const queryClient = useQueryClient()

  const handleLogout = async () => {
    await authClient.signOut()
    queryClient.clear()
    window.location.href = "/login"
  }

  const getInitials = () => {
    if (user?.name) {
      const parts = user.name.split(" ")
      if (parts.length >= 2) {
        return (parts[0].charAt(0) + parts[1].charAt(0)).toUpperCase()
      }
      return user.name.charAt(0).toUpperCase()
    }
    return user?.email?.charAt(0).toUpperCase() ?? "?"
  }

  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <button
              className="size-7 flex items-center justify-center rounded-md transition-colors hover:bg-black/20"
              aria-label="Account menu"
            >
              {user?.image ? (
                <img
                  src={user.image}
                  alt={user.name || "User avatar"}
                  className="size-6 rounded-full object-cover"
                />
              ) : (
                <div className="size-6 rounded-full bg-white/20 flex items-center justify-center text-[10px] font-medium">
                  {getInitials()}
                </div>
              )}
            </button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">
          {user?.name || user?.email}
        </TooltipContent>
      </Tooltip>
      <DropdownMenuContent
        align="start"
        side="right"
        className="min-w-40 w-fit text-13"
      >
        <DropdownMenuLabel className="whitespace-nowrap">
          {user?.email}
        </DropdownMenuLabel>
        <DropdownMenuItem onSelect={onSettingsClick}>
          <SettingsIcon size="10" />
          Settings
        </DropdownMenuItem>
        <DropdownMenuItem variant="destructive" onSelect={handleLogout}>
          <LogOutIcon size="10" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// ============================================================================
// Sidebar Org Button
// ============================================================================

function SidebarOrgButton({
  onSettingsClick,
}: {
  onSettingsClick: () => void
}) {
  const [open, setOpen] = useState(false)
  const { data: orgs } = useSuspenseQuery(orgsQueryOptions)
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const activeOrgId = session?.session?.activeOrganizationId
  const activeOrganization = orgs.find((org) => org.id === activeOrgId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const switchOrg = useMutation({
    mutationFn: async (organizationId: string) => {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      await setActiveOrgFn({ data: { organizationId } })
      return organizationId
    },
    onSuccess: () => {
      queryClient.removeQueries({
        predicate: (query) => {
          const key = query.queryKey[0]
          return key !== "session" && key !== "organizations"
        },
      })
      queryClient.invalidateQueries({
        queryKey: sessionQueryOptions.queryKey,
      })
      setOpen(false)
      navigate({ to: "/" })
    },
  })

  const isSwitching = switchOrg.isPending
  const switchingOrgId = switchOrg.variables

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-2 hover:bg-white/10 rounded px-2 py-1.5 transition-colors text-left flex-1 min-w-0">
          {activeOrganization?.logo && (
            <img
              src={activeOrganization.logo}
              alt={`${activeOrganization.name} logo`}
              className="size-6 rounded-full object-cover shrink-0"
            />
          )}
          <span className="font-medium truncate first-letter:uppercase">
            {activeOrganization?.name}
          </span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="min-w-56 w-fit text-13" align="start">
        <DropdownMenuLabel>My organizations</DropdownMenuLabel>
        {orgs.map((org) => {
          const isCurrentOrg = org.id === activeOrgId
          const isSwitchingToThis = isSwitching && switchingOrgId === org.id
          const isDisabled = isCurrentOrg || isSwitching

          return (
            <DropdownMenuItem
              key={org.id}
              className={cn(
                "whitespace-nowrap",
                isDisabled && "pointer-events-none opacity-50"
              )}
              onSelect={(e) => {
                e.preventDefault()
                if (!isDisabled) {
                  switchOrg.mutate(org.id)
                }
              }}
            >
              {isSwitchingToThis ? (
                <Spinner className="size-6 stroke-1 opacity-50" />
              ) : org.logo ? (
                <img
                  src={org.logo}
                  alt={`${org.name} logo`}
                  className="size-6 rounded-full object-cover"
                />
              ) : (
                <div className="size-6 rounded-full bg-black/10" />
              )}
              {org.name}
            </DropdownMenuItem>
          )
        })}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          className={cn(isSwitching && "pointer-events-none opacity-50")}
          asChild
        >
          <Link to="/create-org">Create an organization</Link>
        </DropdownMenuItem>
        <DropdownMenuItem
          className={cn(isSwitching && "pointer-events-none opacity-50")}
          onSelect={onSettingsClick}
        >
          Settings
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// ============================================================================
// Sidebar Account Button
// ============================================================================

function SidebarAccountButton({
  onSettingsClick,
}: {
  onSettingsClick: () => void
}) {
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const user = session?.user
  const queryClient = useQueryClient()

  const handleLogout = async () => {
    await authClient.signOut()
    queryClient.clear()
    window.location.href = "/login"
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-2 hover:bg-white/10 rounded px-2 py-1.5 transition-colors text-left w-full min-w-0">
          {user?.image ? (
            <img
              src={user.image}
              alt={user.name || "User avatar"}
              className="size-6 rounded-full object-cover shrink-0"
            />
          ) : (
            <div className="size-6 rounded-full bg-white/20 flex items-center justify-center text-xs font-medium uppercase shrink-0">
              {user?.name?.charAt(0) || user?.email?.charAt(0)}
            </div>
          )}
          {user?.name ? (
            <span className="text-sm truncate capitalize">{user.name}</span>
          ) : (
            <span className="text-sm truncate">{user?.email}</span>
          )}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-40 w-fit text-13">
        <DropdownMenuLabel className="whitespace-nowrap">
          {user?.email}
        </DropdownMenuLabel>
        <DropdownMenuItem onSelect={onSettingsClick}>
          <SettingsIcon size="10" />
          Settings
        </DropdownMenuItem>
        <DropdownMenuItem variant="destructive" onSelect={handleLogout}>
          <LogOutIcon size="10" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// ============================================================================
// Resize Handle
// ============================================================================

function ResizeHandle({
  sidebarRef,
  onResizeEnd,
}: {
  sidebarRef: React.RefObject<HTMLDivElement | null>
  onResizeEnd: (finalWidth: number) => void
}) {
  const handleMouseDown = useCallback(() => {
    const sidebar = sidebarRef.current
    if (!sidebar) return

    let currentWidth = sidebar.offsetWidth

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.min(
        SIDEBAR_MAX_WIDTH,
        Math.max(SIDEBAR_MIN_WIDTH, currentWidth + e.movementX)
      )
      currentWidth = newWidth
      sidebar.style.width = `${newWidth}px`
    }

    const handleMouseUp = () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
      onResizeEnd(currentWidth)
    }

    document.addEventListener("mousemove", handleMouseMove)
    document.addEventListener("mouseup", handleMouseUp)
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }, [sidebarRef, onResizeEnd])

  return (
    <div
      className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-white/20 transition-colors z-10"
      onMouseDown={handleMouseDown}
    />
  )
}

// ============================================================================
// Sidebar
// ============================================================================

export const Sidebar = () => {
  const location = useLocation()
  // Use selectors to avoid re-renders when unrelated store values change
  const navbarOpen = useStore((s) => s.navbarOpen)
  const setNavbarOpen = useStore((s) => s.setNavbarOpen)
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const setSidebarWidth = useStore((s) => s.setSidebarWidth)

  const [settingsOpen, setSettingsOpen] = useState(false)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const [isHydrated, setIsHydrated] = useState(false)

  // Derive initial collapsed state from URL (works during SSR too)
  const isOnProjectOrPackage =
    location.pathname.startsWith("/project/") ||
    location.pathname.startsWith("/package/")

  // Before hydration, use URL-derived state. After hydration, use store state.
  const shouldBeCollapsed = !isHydrated ? isOnProjectOrPackage : !navbarOpen

  // On mount, sync store state with URL-derived state
  useLayoutEffect(() => {
    setNavbarOpen(!isOnProjectOrPackage)
    setIsHydrated(true)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleResizeEnd = useCallback(
    (finalWidth: number) => {
      setSidebarWidth(finalWidth)
    },
    [setSidebarWidth]
  )

  // Collapsed state
  if (shouldBeCollapsed) {
    return (
      <>
        <div className="w-12 text-white flex flex-col py-3">
          {/* Expand button */}
          <div className="flex justify-center mb-3 px-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className="size-7 flex items-center justify-center rounded-md transition-colors hover:bg-black/20"
                  onClick={() => setNavbarOpen(true)}
                >
                  <PanelRight size={14} />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right">Expand sidebar</TooltipContent>
            </Tooltip>
          </div>

          {/* Main nav icons */}
          <div className="flex flex-col items-center gap-1 px-1">
            <CollapsedNavIcon
              to="/"
              icon={LayoutDashboard}
              tooltip="Overview"
            />
            <CollapsedNavIcon
              to="/all-projects"
              icon={FolderKanban}
              tooltip="All Projects"
            />
            <CollapsedNavIcon
              to="/vendor-db"
              icon={Building2}
              tooltip="Vendor Database"
            />
          </div>

          {/* Separator */}
          <div className="border-t border-white/10 my-3 mx-2" />

          {/* Projects */}
          <div className="flex-1 overflow-auto px-1">
            <CollapsedProjectTree />
          </div>

          {/* Separator */}
          <div className="border-t border-white/10 my-3 mx-2" />

          {/* User */}
          <div className="flex justify-center">
            <CollapsedAccountButton
              onSettingsClick={() => setSettingsOpen(true)}
            />
          </div>
        </div>

        <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
      </>
    )
  }

  return (
    <>
      <div
        ref={sidebarRef}
        className="overflow-hidden text-white relative flex flex-col"
        style={{ width: sidebarWidth, minWidth: SIDEBAR_MIN_WIDTH }}
      >
        <ResizeHandle sidebarRef={sidebarRef} onResizeEnd={handleResizeEnd} />

        <div className="p-3 min-w-0">
          <div className="flex items-center justify-between shrink-0 gap-1 min-w-0">
            <SidebarOrgButton onSettingsClick={() => setSettingsOpen(true)} />
            <div className="flex items-center gap-1 shrink-0">
              <button
                className="p-1.5 hover:bg-white/10 rounded transition-colors"
                onClick={() => setSettingsOpen(true)}
              >
                <Settings size={16} />
              </button>
              <button
                className="p-1.5 hover:bg-white/10 rounded transition-colors"
                onClick={() => setNavbarOpen(false)}
              >
                <PanelRight size={16} />
              </button>
            </div>
          </div>

          <nav className="mt-3 flex flex-col gap-px">
            <NavLink to="/" icon={LayoutDashboard}>
              Overview
            </NavLink>
            <NavLink to="/all-projects" icon={FolderKanban}>
              All Projects
            </NavLink>
            <NavLink to="/vendor-db" icon={Building2}>
              Vendor Database
            </NavLink>
          </nav>
        </div>

        <div className="border-t border-white/10" />

        <div className="flex-1 overflow-auto min-w-0">
          <ProjectTree />
        </div>

        <div className="border-t border-white/10" />

        <div className="p-3 min-w-0">
          <SidebarAccountButton onSettingsClick={() => setSettingsOpen(true)} />
        </div>
      </div>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  )
}
