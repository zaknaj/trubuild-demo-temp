import { useState } from "react"
import { Link } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import { FolderKanban, ChevronRight, Settings, Plus } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"
import {
  projectDetailQueryOptions,
  projectMembersQueryOptions,
  projectAccessQueryOptions,
  sessionQueryOptions,
} from "@/lib/query-options"

// Member types
interface Member {
  email: string
  userId: string | null
  userName: string | null
  userImage: string | null
  role: string
}

const PROJECT_ROLE_CONFIG = {
  project_lead: { label: "Project manager", order: 1 },
  commercial_lead: { label: "Commercial manager", order: 2 },
  technical_lead: { label: "Technical manager", order: 3 },
} as const

interface ProjectSidebarProps {
  projectId: string
  onSettingsClick: (tab?: "general" | "members" | "activity") => void
}

export function ProjectSidebar({
  projectId,
  onSettingsClick,
}: ProjectSidebarProps) {
  const [isHovered, setIsHovered] = useState(false)

  const { data: projectData } = useSuspenseQuery(
    projectDetailQueryOptions(projectId)
  )
  const { data: members } = useSuspenseQuery(
    projectMembersQueryOptions(projectId)
  )
  const { data: accessData } = useSuspenseQuery(
    projectAccessQueryOptions(projectId)
  )
  const { data: session } = useSuspenseQuery(sessionQueryOptions)

  const canEdit = accessData.access === "full"

  // Get active members
  const activeMembers = members.filter((m: Member) => m.userId !== null)

  // Group and sort members
  const sortedMembers = [...activeMembers].sort((a: Member, b: Member) => {
    const aOrder =
      PROJECT_ROLE_CONFIG[a.role as keyof typeof PROJECT_ROLE_CONFIG]?.order ??
      99
    const bOrder =
      PROJECT_ROLE_CONFIG[b.role as keyof typeof PROJECT_ROLE_CONFIG]?.order ??
      99
    return aOrder - bOrder
  })

  return (
    <aside className="w-72 bg-white overflow-auto flex flex-col">
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 h-12 shrink-0"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Link
              to="/all-projects"
              className="flex items-center justify-center size-6 hover:bg-muted rounded transition-colors"
            >
              <FolderKanban className="size-4 text-muted-foreground" />
            </Link>
          </TooltipTrigger>
          <TooltipContent side="bottom" align="start">
            All projects
          </TooltipContent>
        </Tooltip>
        <ChevronRight className="size-3 text-muted-foreground/50" />
        <span className="font-semibold text-sm truncate flex-1">
          {projectData.project.name}
        </span>
        {isHovered && canEdit && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => onSettingsClick("general")}
                className="flex items-center justify-center size-6 hover:bg-muted rounded transition-colors"
              >
                <Settings className="size-4 text-muted-foreground" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Project settings</TooltipContent>
          </Tooltip>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto px-4 py-4 space-y-6">
        {/* Members Section */}
        <div className="space-y-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => onSettingsClick("members")}
                className="text-xs font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground hover:underline cursor-pointer transition-colors"
              >
                Members
              </button>
            </TooltipTrigger>
            <TooltipContent side="left">Manage members</TooltipContent>
          </Tooltip>

          {sortedMembers.length > 0 ? (
            <div className="space-y-2">
              {sortedMembers.map((member: Member) => (
                <div key={member.email} className="flex items-center gap-2">
                  {member.userImage ? (
                    <img
                      src={member.userImage}
                      alt={member.userName || member.email}
                      className="size-8 rounded-full object-cover"
                    />
                  ) : (
                    <div className="size-8 rounded-full bg-muted flex items-center justify-center text-xs font-medium uppercase text-muted-foreground">
                      {member.userName?.charAt(0) || member.email.charAt(0)}
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm truncate">
                        {member.userName || member.email}
                      </span>
                      {session?.user?.id === member.userId && (
                        <Badge
                          variant="secondary"
                          className="text-[10px] h-4 px-1.5 bg-blue-100 text-blue-700 hover:bg-blue-100"
                        >
                          You
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {PROJECT_ROLE_CONFIG[
                        member.role as keyof typeof PROJECT_ROLE_CONFIG
                      ]?.label || member.role}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No members</p>
          )}

          {canEdit && (
            <button
              onClick={() => onSettingsClick("members")}
              className="flex items-center gap-1.5 text-sm text-primary hover:underline"
            >
              <Plus className="size-3" />
              Invite a member
            </button>
          )}
        </div>
      </div>
    </aside>
  )
}
