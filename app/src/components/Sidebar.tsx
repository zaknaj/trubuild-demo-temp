import { Link, useLocation } from "@tanstack/react-router"
import { useQuery, useSuspenseQuery } from "@tanstack/react-query"
import type { ComponentType } from "react"
import useStore from "@/lib/store"
import {
  BookUser,
  FolderOpen,
  Folders,
  LayoutDashboard,
  PanelLeft,
  Settings,
} from "lucide-react"
import { projectsQueryOptions, orgsQueryOptions, sessionQueryOptions } from "@/lib/query-options"
import { cn } from "@/lib/utils"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

type SidebarItemProps = {
  to: string
  label: string
  icon: ComponentType<{ size?: number; className?: string }>
  isActive: boolean
}

function SidebarItem({ to, label, icon: Icon, isActive }: SidebarItemProps) {
  return (
    <Link
      to={to}
      className={cn(
        "h-8 rounded-[6px] px-2 flex items-center text-[13px] font-medium",
        isActive ? "bg-[#EDEDED]" : "hover:bg-[#EDEDED]"
      )}
    >
      <Icon
        size={16}
        className={cn("shrink-0", isActive ? "opacity-100" : "opacity-70")}
      />
      <span className={cn("ml-[6px] truncate", isActive ? "opacity-100" : "opacity-70")}>
        {label}
      </span>
    </Link>
  )
}

type PackageRowProps = {
  pkg: { id: string; name: string }
  isActive: boolean
}

function PackageRow({ pkg, isActive }: PackageRowProps) {
  return (
    <Link
      to="/package/$id"
      params={{ id: pkg.id }}
      className={cn(
        "h-8 rounded-[6px] px-2 flex items-center text-[13px] font-medium",
        isActive ? "bg-[#EDEDED]" : "hover:bg-[#EDEDED]"
      )}
    >
      <span className="w-4 mr-[6px] shrink-0" />
      <span className={cn("truncate", isActive ? "opacity-100" : "opacity-70")}>
        {pkg.name}
      </span>
    </Link>
  )
}

type NoPackageRowProps = {
  isActive?: boolean
}

function NoPackageRow({ isActive = false }: NoPackageRowProps) {
  return (
    <div className="h-8 rounded-[6px] px-2 flex items-center text-[13px] font-medium">
      <span className="w-4 mr-[6px] shrink-0" />
      <span className={cn("truncate", isActive ? "opacity-100" : "opacity-70")}>
        No package
      </span>
    </div>
  )
}

export const Sidebar = () => {
  const location = useLocation()
  const navbarOpen = useStore((s) => s.navbarOpen)
  const setNavbarOpen = useStore((s) => s.setNavbarOpen)

  const { data: projects = [] } = useQuery(projectsQueryOptions)
  const { data: orgs = [] } = useSuspenseQuery(orgsQueryOptions)
  const { data: session } = useSuspenseQuery(sessionQueryOptions)

  const activeOrgId = session?.session?.activeOrganizationId
  const activeOrg = orgs.find((org) => org.id === activeOrgId)

  const pathname = location.pathname
  const activeProjectId = pathname.match(/^\/project\/([^/]+)/)?.[1] ?? null
  const activePackageId = pathname.match(/^\/package\/([^/]+)/)?.[1] ?? null

  if (!navbarOpen) {
    return (
      <aside
        className="w-11 bg-[#F9F9F9] border-r border-[#D4D4D4] text-black shrink-0"
        style={{ borderRightWidth: "0.5px" }}
      >
        <div className="h-full flex items-start justify-center pt-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="h-8 w-8 rounded-[6px] flex items-center justify-center hover:bg-[#EDEDED] opacity-40 hover:opacity-100"
                onClick={() => setNavbarOpen(true)}
                type="button"
              >
                <PanelLeft size={16} />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">Expand panel</TooltipContent>
          </Tooltip>
        </div>
      </aside>
    )
  }

  return (
    <aside
      className="w-[260px] bg-[#F9F9F9] border-r border-[#D4D4D4] text-black flex flex-col shrink-0"
      style={{ borderRightWidth: "0.5px" }}
    >
      <div className="px-2 pt-2">
        <div className="h-8 px-2 flex items-center justify-between">
          <div className="flex items-center min-w-0">
            <span className="h-6 w-6 rounded-full bg-[#E0E0E0] shrink-0" />
            <span className="ml-[6px] text-[14px] font-medium truncate opacity-100">
              {activeOrg?.name ?? "Company"}
            </span>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="h-8 w-8 rounded-[6px] flex items-center justify-center hover:bg-[#EDEDED] opacity-40 hover:opacity-100"
                onClick={() => setNavbarOpen(false)}
                type="button"
              >
                <PanelLeft size={16} />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">Collapse panel</TooltipContent>
          </Tooltip>
        </div>

        <nav className="py-6">
          <SidebarItem
            to="/"
            label="Overview"
            icon={LayoutDashboard}
            isActive={pathname === "/"}
          />
          <SidebarItem
            to="/all-projects"
            label="All projects"
            icon={Folders}
            isActive={pathname === "/all-projects"}
          />
          <SidebarItem
            to="/vendor-db"
            label="Vendor database"
            icon={BookUser}
            isActive={pathname === "/vendor-db"}
          />
          <SidebarItem
            to="/settings"
            label="Settings"
            icon={Settings}
            isActive={pathname === "/settings"}
          />
        </nav>
      </div>

      <div className="px-2 flex-1 overflow-auto">
        <div className="h-8 px-2 flex items-center text-[11px] font-medium opacity-40">
          Projects
        </div>
        {projects.map((project) => {
          const isProjectActive = activeProjectId === project.id
          return (
            <div key={project.id}>
              <Link
                to="/project/$id"
                params={{ id: project.id }}
                className={cn(
                  "h-8 rounded-[6px] px-2 flex items-center text-[13px] font-medium",
                  isProjectActive ? "bg-[#EDEDED]" : "hover:bg-[#EDEDED]"
                )}
              >
                <FolderOpen
                  size={16}
                  className={cn(
                    "shrink-0",
                    isProjectActive ? "opacity-100" : "opacity-70"
                  )}
                />
                <span
                  className={cn(
                    "ml-[6px] truncate",
                    isProjectActive ? "opacity-100" : "opacity-70"
                  )}
                >
                  {project.name}
                </span>
              </Link>

              {project.packages.length > 0 ? (
                project.packages.map((pkg) => (
                  <PackageRow
                    key={pkg.id}
                    pkg={pkg}
                    isActive={activePackageId === pkg.id}
                  />
                ))
              ) : (
                <NoPackageRow />
              )}
            </div>
          )
        })}
      </div>

      <div className="px-2 pb-2">
        <div className="h-8 px-2 flex items-center">
          <span className="h-6 w-6 rounded-full bg-[#E0E0E0] shrink-0" />
          <span className="ml-[6px] text-[14px] font-medium truncate opacity-100">
            {session?.user?.name ?? "User"}
          </span>
        </div>
      </div>
    </aside>
  )
}
