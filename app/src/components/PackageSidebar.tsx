import { useState } from "react"
import { Link, useLocation, useNavigate } from "@tanstack/react-router"
import useStore from "@/lib/store"
import { useSuspenseQuery, useQuery } from "@tanstack/react-query"
import { Folder, ChevronRight, Settings, Plus } from "lucide-react"
import { cn, formatCurrency } from "@/lib/utils"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"
import {
  packageDetailQueryOptions,
  packageMembersQueryOptions,
  packageContractorsQueryOptions,
  packageAccessQueryOptions,
  sessionQueryOptions,
  technicalEvaluationsQueryOptions,
  technicalEvaluationDetailQueryOptions,
  packageCommercialSummaryQueryOptions,
} from "@/lib/query-options"
import type { TechnicalEvaluationData } from "@/components/TechSetupWizard"
import type { CommercialEvaluationData } from "@/lib/types"

const navLinkClass = "nav-item nav-item-light"
const navLinkActiveClass = "active"

// Member types
interface Member {
  email: string
  userId: string | null
  userName: string | null
  userImage: string | null
  role: string
}

const PACKAGE_ROLE_CONFIG = {
  package_lead: { label: "Package lead", order: 1 },
  commercial_team: { label: "Commercial member", order: 2 },
  technical_team: { label: "Technical member", order: 3 },
} as const

// Helper: Calculate weighted technical score for a contractor
function calculateWeightedScore(
  contractorId: string,
  evalData: TechnicalEvaluationData
): number {
  const allBreakdowns =
    evalData.criteria?.scopes?.flatMap((s) => s.breakdowns) ?? []
  const contractorScores = evalData.scores?.[contractorId] ?? {}

  let totalScore = 0
  for (const breakdown of allBreakdowns) {
    const score = contractorScores[breakdown.id]?.score ?? 0
    totalScore += (score * breakdown.weight) / 100
  }
  return totalScore
}

// Helper: Get ranked technical contractors
function getTechnicalRankings(
  evalData: TechnicalEvaluationData | null,
  contractors: Array<{ id: string; name: string }>
) {
  if (!evalData || !evalData.scores) return []

  const proposalsUploaded = evalData.proposalsUploaded ?? []
  const evaluated = contractors.filter((c) => proposalsUploaded.includes(c.id))

  return evaluated
    .map((c) => ({
      id: c.id,
      name: c.name,
      score: calculateWeightedScore(c.id, evalData),
    }))
    .sort((a, b) => b.score - a.score)
}

// Helper: Get ranked commercial contractors
function getCommercialRankings(
  commercialSummary: {
    assets: Array<{ evaluation: CommercialEvaluationData | null }>
  } | null
) {
  if (!commercialSummary || commercialSummary.assets.length === 0) return []

  const contractorTotals: Record<string, { name: string; total: number }> = {}

  for (const asset of commercialSummary.assets) {
    const evalData = asset.evaluation as CommercialEvaluationData | null
    if (!evalData?.contractors) continue

    for (const contractor of evalData.contractors) {
      if (!contractorTotals[contractor.contractorId]) {
        contractorTotals[contractor.contractorId] = {
          name: contractor.contractorName,
          total: 0,
        }
      }
      contractorTotals[contractor.contractorId].total += contractor.totalAmount
    }
  }

  return Object.entries(contractorTotals)
    .map(([id, data]) => ({
      id,
      name: data.name,
      total: data.total,
    }))
    .sort((a, b) => a.total - b.total)
}

interface PackageSidebarProps {
  packageId: string
  onSettingsClick: (tab?: "general" | "members" | "activity") => void
}

export function PackageSidebar({
  packageId,
  onSettingsClick,
}: PackageSidebarProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const setCreateAssetSheetOpen = useStore((s) => s.setCreateAssetSheetOpen)
  const [isHovered, setIsHovered] = useState(false)

  const { data: packageData } = useSuspenseQuery(
    packageDetailQueryOptions(packageId)
  )
  const { data: members } = useSuspenseQuery(
    packageMembersQueryOptions(packageId)
  )
  const { data: contractors } = useSuspenseQuery(
    packageContractorsQueryOptions(packageId)
  )
  const { data: accessData } = useSuspenseQuery(
    packageAccessQueryOptions(packageId)
  )
  const { data: session } = useSuspenseQuery(sessionQueryOptions)

  const canEdit = accessData.access === "full"
  const canViewTechnical =
    accessData.access === "full" || accessData.access === "technical"
  const canViewCommercial =
    accessData.access === "full" || accessData.access === "commercial"

  // Only fetch technical data if user has technical access
  const { data: technicalEvals } = useQuery({
    ...technicalEvaluationsQueryOptions(packageId),
    enabled: canViewTechnical,
  }) as {
    data: Array<{ id: string; roundName: string; data: unknown }> | undefined
  }

  // Only fetch commercial data if user has commercial access
  const { data: commercialSummary } = useQuery({
    ...packageCommercialSummaryQueryOptions(packageId),
    enabled: canViewCommercial,
  })

  // Get latest technical evaluation for rankings (only if can view)
  const latestTechEval =
    canViewTechnical && technicalEvals && technicalEvals.length > 0
      ? technicalEvals[0]
      : null
  const { data: techEvalDetail } = useQuery({
    ...technicalEvaluationDetailQueryOptions(latestTechEval?.id ?? ""),
    enabled: !!latestTechEval && canViewTechnical,
  })
  const techEvalData = canViewTechnical
    ? ((techEvalDetail as { data?: TechnicalEvaluationData } | undefined)
        ?.data ?? null)
    : null

  // Get rankings (only if access permits)
  const technicalRankings = canViewTechnical
    ? getTechnicalRankings(techEvalData, contractors)
    : []
  const commercialRankings = canViewCommercial
    ? getCommercialRankings(
        commercialSummary as {
          assets: Array<{ evaluation: CommercialEvaluationData | null }>
        } | null
      )
    : []

  // Check if we're in the commercial section (to show assets list)
  const isInCommSection = location.pathname.includes(
    `/package/${packageId}/comm`
  )

  // Get active members
  const activeMembers = members.filter((m: Member) => m.userId !== null)

  // Group and sort members
  const sortedMembers = [...activeMembers].sort((a: Member, b: Member) => {
    const aOrder =
      PACKAGE_ROLE_CONFIG[a.role as keyof typeof PACKAGE_ROLE_CONFIG]?.order ??
      99
    const bOrder =
      PACKAGE_ROLE_CONFIG[b.role as keyof typeof PACKAGE_ROLE_CONFIG]?.order ??
      99
    return aOrder - bOrder
  })

  // Merge rankings to get contractor scores and totals
  const contractorData = contractors.map((c) => {
    const techRank = technicalRankings.findIndex((r) => r.id === c.id)
    const commRank = commercialRankings.findIndex((r) => r.id === c.id)
    const techScore = technicalRankings.find((r) => r.id === c.id)?.score
    const commTotal = commercialRankings.find((r) => r.id === c.id)?.total

    // Calculate combined rank based on available data
    let rank: number | null = null
    if (techRank >= 0 && commRank >= 0) {
      // Average of ranks if both available
      rank = Math.round((techRank + commRank) / 2)
    } else if (techRank >= 0) {
      rank = techRank
    } else if (commRank >= 0) {
      rank = commRank
    }

    return {
      ...c,
      rank,
      techScore,
      commTotal,
    }
  })

  // Sort by rank (null ranks at end)
  const sortedContractors = [...contractorData].sort((a, b) => {
    if (a.rank === null && b.rank === null) return 0
    if (a.rank === null) return 1
    if (b.rank === null) return -1
    return a.rank - b.rank
  })

  const getRankColor = (rank: number | null) => {
    if (rank === null) return "text-muted-foreground"
    if (rank === 0) return "text-orange-500"
    if (rank === 1) return "text-blue-500"
    if (rank === 2) return "text-amber-700"
    return "text-muted-foreground"
  }

  return (
    <aside className="w-72 overflow-auto flex flex-col bg-[#FCFCFC]">
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 h-12 shrink-0"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Link
              to="/project/$id"
              params={{ id: packageData.project.id }}
              className="flex items-center justify-center size-6 hover:bg-muted rounded transition-colors"
            >
              <Folder className="size-4 text-muted-foreground" />
            </Link>
          </TooltipTrigger>
          <TooltipContent side="bottom" align="start">
            <span className="opacity-70">Project: </span>
            {packageData.project.name}
          </TooltipContent>
        </Tooltip>
        <ChevronRight className="size-3 text-muted-foreground/50" />
        <span className="font-semibold text-sm truncate flex-1">
          {packageData.package.name}
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
            <TooltipContent>Package settings</TooltipContent>
          </Tooltip>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto px-4 py-4 space-y-6">
        {/* Navigation Links */}
        <nav className="flex flex-col gap-px">
          <Link
            to="/package/$id"
            params={{ id: packageId }}
            activeOptions={{ exact: true }}
            className={navLinkClass}
            activeProps={{ className: `${navLinkClass} ${navLinkActiveClass}` }}
          >
            Package summary
          </Link>
          {canViewTechnical && (
            <Link
              to="/package/$id/tech"
              params={{ id: packageId }}
              className={navLinkClass}
              activeProps={{
                className: `${navLinkClass} ${navLinkActiveClass}`,
              }}
            >
              Technical Evaluation
            </Link>
          )}
          {canViewCommercial && (
            <>
              <Link
                to="/package/$id/comm"
                params={{ id: packageId }}
                activeOptions={{ exact: true }}
                className={navLinkClass}
                activeProps={{
                  className: `${navLinkClass} ${navLinkActiveClass}`,
                }}
              >
                Commercial Evaluation
              </Link>
              {/* Show assets and add button when in commercial section */}
              {isInCommSection && (
                <div className="ml-4 flex flex-col gap-px border-l border-black/10 pl-2">
                  {packageData.assets.map(
                    (asset: { id: string; name: string }) => (
                      <Link
                        key={asset.id}
                        to="/package/$id/comm/$assetId"
                        params={{ id: packageId, assetId: asset.id }}
                        className={cn(navLinkClass, "text-sm py-1.5")}
                        activeProps={{
                          className: `${navLinkClass} ${navLinkActiveClass} text-sm py-1.5`,
                        }}
                      >
                        {asset.name}
                      </Link>
                    )
                  )}
                  {canEdit && (
                    <button
                      onClick={() => {
                        // Navigate to comm page if not already there, then open sheet
                        if (!location.pathname.endsWith("/comm")) {
                          navigate({
                            to: "/package/$id/comm",
                            params: { id: packageId },
                          })
                        }
                        setCreateAssetSheetOpen(true)
                      }}
                      className="flex items-center gap-1.5 text-sm text-primary hover:underline py-1.5 px-2"
                    >
                      <Plus className="size-3" />
                      Add a new asset
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </nav>

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
                      {PACKAGE_ROLE_CONFIG[
                        member.role as keyof typeof PACKAGE_ROLE_CONFIG
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

        {/* Contractors Section */}
        <div className="space-y-3">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Contractors
          </span>

          {sortedContractors.length > 0 ? (
            <div className="space-y-2">
              {sortedContractors.map((contractor) => (
                <div key={contractor.id} className="flex items-start gap-2">
                  {contractor.rank !== null && (
                    <span
                      className={cn(
                        "text-sm font-medium",
                        getRankColor(contractor.rank)
                      )}
                    >
                      #{contractor.rank + 1}
                    </span>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {contractor.name}
                    </p>
                    {(canViewTechnical || canViewCommercial) && (
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        {canViewTechnical &&
                          (contractor.techScore !== undefined ? (
                            <span>{Math.round(contractor.techScore)}%</span>
                          ) : (
                            <span className="w-8 h-2 bg-muted rounded" />
                          ))}
                        {canViewTechnical && canViewCommercial && (
                          <span>·</span>
                        )}
                        {canViewCommercial &&
                          (contractor.commTotal !== undefined ? (
                            <span>
                              {formatCurrency(
                                contractor.commTotal,
                                packageData.package.currency ?? undefined
                              )}
                            </span>
                          ) : (
                            <span className="w-16 h-2 bg-muted rounded" />
                          ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No contractors</p>
          )}
        </div>

        {/* Activity Section */}
        <div className="space-y-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => onSettingsClick("activity")}
                className="text-xs font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground hover:underline cursor-pointer transition-colors"
              >
                Activity
              </button>
            </TooltipTrigger>
            <TooltipContent side="left">All activity</TooltipContent>
          </Tooltip>
          {/* Empty for now */}
        </div>
      </div>
    </aside>
  )
}
