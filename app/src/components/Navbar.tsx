import { Link, useLocation } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"

import { OrgNavButton } from "./OrgNavButton"
import { AccountNavButton } from "./AccountNavButton"
import { FolderOpen, PanelsTopLeft } from "lucide-react"
import {
  projectDetailQueryOptions,
  packageDetailQueryOptions,
} from "@/lib/query-options"

function ProjectsBreadcrumb() {
  const location = useLocation()
  const pathname = location.pathname

  // Extract IDs from pathname
  const projectMatch = pathname.match(/^\/project\/([^/]+)/)
  const packageMatch = pathname.match(/^\/package\/([^/]+)/)

  const projectId = projectMatch ? projectMatch[1] : null
  const packageId = packageMatch ? packageMatch[1] : null

  const isOnAllProjects = pathname === "/all-projects"
  const isOnProject = !!projectId
  const isOnPackage = !!packageId

  // Fetch project data when on project route
  const { data: projectData } = useQuery({
    ...projectDetailQueryOptions(projectId ?? ""),
    enabled: isOnProject,
  })

  // Fetch package data when on package route (includes project info)
  const { data: packageData } = useQuery({
    ...packageDetailQueryOptions(packageId ?? ""),
    enabled: isOnPackage,
  })

  // Determine what to show - note the nested structure from the API
  const projectName = isOnPackage
    ? packageData?.project?.name
    : projectData?.project?.name
  const packageName = packageData?.package?.name
  const linkProjectId = isOnPackage ? packageData?.project?.id : projectId

  // Breadcrumb is "active" when on any project-related route
  const isActive = isOnAllProjects || isOnProject || isOnPackage

  // Non-active parts have reduced opacity
  const inactiveClass = "opacity-60 hover:opacity-100"

  return (
    <div
      className={`rounded py-1.5 text-13 font-medium flex items-center gap-1.5 px-2.5 hover:bg-black/10 ${isActive ? "text-white" : "text-white/50"}`}
    >
      {/* On all-projects: (folder) All projects */}
      {isOnAllProjects && (
        <Link to="/all-projects" className="flex items-center gap-1.5">
          <FolderOpen size="15" />
          All projects
        </Link>
      )}

      {/* On project: (folder) All projects / Project Name */}
      {isOnProject && (
        <>
          <Link
            to="/all-projects"
            className={`flex items-center gap-1.5 ${inactiveClass}`}
          >
            <FolderOpen size="15" />
            All projects
          </Link>
          <span className="text-white/50">/</span>
          <Link
            to="/project/$id"
            params={{ id: linkProjectId! }}
            className="text-white"
          >
            {projectName}
          </Link>
        </>
      )}

      {/* On package: (folder) Project Name / Package Name */}
      {isOnPackage && (
        <>
          <Link
            to="/project/$id"
            params={{ id: linkProjectId! }}
            className={`flex items-center gap-1.5 ${inactiveClass}`}
          >
            <FolderOpen size="15" />
            {projectName}
          </Link>
          <span className="text-white/50">/</span>
          <Link
            to="/package/$id"
            params={{ id: packageId! }}
            className="text-white"
          >
            {packageName}
          </Link>
        </>
      )}

      {/* Not on any project-related route: just show inactive link */}
      {!isActive && (
        <Link to="/all-projects" className="flex items-center gap-1.5">
          <FolderOpen size="15" />
          All projects
        </Link>
      )}
    </div>
  )
}

export const Navbar = () => {
  return (
    <div className="h-15 shrink-0 flex items-center  px-3 text-white justify-between">
      <div className="flex-1 flex gap-3 items-center">
        <img src="/logo-app.svg" alt="logo" className="w-7 h-7" />
        <div className="w-[1.5px] h-4 opacity-20 bg-white"></div>
        <OrgNavButton />
      </div>
      <div className="w-fit flex items-center gap-2 justify-center">
        <Link
          to="/"
          className="rounded py-1.5 text-13 font-medium flex items-center gap-1.5 px-2.5 border-[0.5px] border-transparent hover:bg-black/10 text-white/50"
          activeProps={{
            className: "text-white/100  opacity-100",
          }}
        >
          <PanelsTopLeft size="15" /> Overview
        </Link>
        <ProjectsBreadcrumb />
      </div>
      <div className="flex-1 flex justify-end">
        <AccountNavButton />
      </div>
    </div>
  )
}
