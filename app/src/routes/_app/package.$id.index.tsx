import { createFileRoute, useNavigate } from "@tanstack/react-router"
import {
  useSuspenseQuery,
  useMutation,
  useQueryClient,
  useQuery,
} from "@tanstack/react-query"
import { useState, useMemo } from "react"
import { toast } from "sonner"
import {
  ClipboardList,
  FileText,
  ArrowRight,
  Award,
  Trophy,
  CheckCircle2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
} from "@/components/ui/sheet"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  technicalEvaluationsQueryOptions,
  hasCommercialEvaluationQueryOptions,
  packageDetailQueryOptions,
  packageContractorsQueryOptions,
  packageCommercialSummaryQueryOptions,
  technicalEvaluationDetailQueryOptions,
  packageAccessQueryOptions,
  projectDetailQueryOptions,
} from "@/lib/query-options"
import { awardPackageFn } from "@/fn/packages"
import type { TechnicalEvaluationData } from "@/components/TechSetupWizard"
import type { CommercialEvaluationData } from "@/lib/types"
import { formatCurrency } from "@/lib/utils"
import { AIChatButton } from "@/components/AIChatButton"

export const Route = createFileRoute("/_app/package/$id/")({
  loader: ({ params, context }) => {
    context.queryClient.prefetchQuery(
      technicalEvaluationsQueryOptions(params.id)
    )
    context.queryClient.prefetchQuery(
      hasCommercialEvaluationQueryOptions(params.id)
    )
    context.queryClient.prefetchQuery(packageDetailQueryOptions(params.id))
    context.queryClient.prefetchQuery(packageContractorsQueryOptions(params.id))
    context.queryClient.prefetchQuery(
      packageCommercialSummaryQueryOptions(params.id)
    )
    context.queryClient.prefetchQuery(packageAccessQueryOptions(params.id))
  },
  component: RouteComponent,
})

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

// Helper: Get ranked commercial contractors (aggregated across all assets)
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

function RouteComponent() {
  const { id } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [awardSheetOpen, setAwardSheetOpen] = useState(false)
  const [selectedContractorId, setSelectedContractorId] = useState<string>("")
  const [awardComments, setAwardComments] = useState<string>("")

  const { data: packageData } = useSuspenseQuery(packageDetailQueryOptions(id))
  const { data: contractors } = useSuspenseQuery(
    packageContractorsQueryOptions(id)
  )
  const { data: accessData } = useSuspenseQuery(packageAccessQueryOptions(id))

  const canViewTechnical =
    accessData.access === "full" || accessData.access === "technical"
  const canViewCommercial =
    accessData.access === "full" || accessData.access === "commercial"

  // Only fetch technical data if user has technical access
  const { data: technicalEvals } = useQuery({
    ...technicalEvaluationsQueryOptions(id),
    enabled: canViewTechnical,
  })

  // Only fetch commercial data if user has commercial access
  const { data: commercialEvalData } = useQuery({
    ...hasCommercialEvaluationQueryOptions(id),
    enabled: canViewCommercial,
  })
  const { data: commercialSummary } = useQuery({
    ...packageCommercialSummaryQueryOptions(id),
    enabled: canViewCommercial,
  })

  // Get the latest technical evaluation details (if any and can view)
  const latestTechEval =
    canViewTechnical && technicalEvals && technicalEvals.length > 0
      ? technicalEvals[0]
      : null
  const { data: techEvalDetail } = useQuery({
    ...technicalEvaluationDetailQueryOptions(latestTechEval?.id ?? ""),
    enabled: !!latestTechEval && canViewTechnical,
  })

  const awardPackage = useMutation({
    mutationFn: (data: { contractorId: string; comments?: string }) =>
      awardPackageFn({
        data: {
          packageId: id,
          contractorId: data.contractorId,
          comments: data.comments,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(id).queryKey,
      })
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(packageData.project.id).queryKey,
      })
      setAwardSheetOpen(false)
      setSelectedContractorId("")
      setAwardComments("")
      toast.success("Package awarded successfully")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to award package"
      )
    },
  })

  const hasTechEval =
    canViewTechnical && technicalEvals && technicalEvals.length > 0
  const hasCommercialEval =
    canViewCommercial && commercialEvalData?.hasEvaluation

  // Calculate rankings (only if access permits)
  const techEvalData = canViewTechnical
    ? (techEvalDetail?.data as TechnicalEvaluationData | null)
    : null
  const technicalRankings = useMemo(
    () =>
      canViewTechnical ? getTechnicalRankings(techEvalData, contractors) : [],
    [techEvalData, contractors, canViewTechnical]
  )
  const commercialRankings = useMemo(
    () =>
      canViewCommercial ? getCommercialRankings(commercialSummary ?? null) : [],
    [commercialSummary, canViewCommercial]
  )

  // Get awarded contractor info
  const awardedContractor = packageData.package.awardedContractorId
    ? contractors.find((c) => c.id === packageData.package.awardedContractorId)
    : null

  // Check if latest tech eval review is complete
  const isTechReviewComplete = techEvalData?.status === "review_complete"

  // Get contractors that appear in BOTH technical and commercial evaluations
  const eligibleContractors = useMemo(() => {
    const techContractorIds = new Set(technicalRankings.map((r) => r.id))
    const commContractorIds = new Set(commercialRankings.map((r) => r.id))

    return contractors.filter(
      (c) => techContractorIds.has(c.id) && commContractorIds.has(c.id)
    )
  }, [contractors, technicalRankings, commercialRankings])

  // Determine if award button should be disabled and why
  const canAward = eligibleContractors.length > 0 && isTechReviewComplete
  const awardDisabledReason = !isTechReviewComplete
    ? "Complete the score review on the latest technical evaluation first"
    : eligibleContractors.length === 0
      ? "No contractors have completed both technical and commercial evaluations"
      : null

  // For award sheet: get selected contractor's info
  const selectedContractor = contractors.find(
    (c) => c.id === selectedContractorId
  )
  const selectedTechRank =
    technicalRankings.findIndex((r) => r.id === selectedContractorId) + 1
  const selectedTechScore = technicalRankings.find(
    (r) => r.id === selectedContractorId
  )?.score
  const selectedCommRank =
    commercialRankings.findIndex((r) => r.id === selectedContractorId) + 1
  const selectedCommTotal = commercialRankings.find(
    (r) => r.id === selectedContractorId
  )?.total

  return (
    <>
      {/* Header */}
      <div className="flex items-center justify-between border-b-[0.5px] border-black/15 h-12 px-6">
        <div className="flex items-center gap-4">
          <span className="text-primary font-semibold text-16">
            Package summary
          </span>
          {!awardedContractor && contractors.length > 0 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <Button
                      size="sm"
                      onClick={() => {
                        if (technicalRankings.length > 0) {
                          const highestScorer = technicalRankings[0]
                          if (
                            eligibleContractors.some(
                              (c) => c.id === highestScorer.id
                            )
                          ) {
                            setSelectedContractorId(highestScorer.id)
                          }
                        }
                        setAwardSheetOpen(true)
                      }}
                      disabled={!canAward}
                    >
                      <Award className="mr-1.5 h-3.5 w-3.5" />
                      Award Package
                    </Button>
                  </span>
                </TooltipTrigger>
                {awardDisabledReason && (
                  <TooltipContent>
                    <p>{awardDisabledReason}</p>
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <AIChatButton />
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* Awarded Banner */}
        {awardedContractor && (
          <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="p-2 bg-green-100 rounded-full">
              <Trophy className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="font-medium text-green-800">Package Awarded</p>
              <p className="text-sm text-green-700">
                Awarded to{" "}
                <span className="font-semibold">{awardedContractor.name}</span>
              </p>
            </div>
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-2">
          {/* Technical Evaluation Card */}
          {canViewTechnical && (
            <div className="border rounded-lg p-6 flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <ClipboardList className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-medium">Technical Evaluation</h3>
                  <p className="text-sm text-muted-foreground">
                    Evaluate contractors on technical criteria
                  </p>
                </div>
              </div>

              <Button
                variant={hasTechEval ? "outline" : "default"}
                className="mt-auto"
                onClick={() =>
                  navigate({ to: "/package/$id/tech", params: { id } })
                }
              >
                {hasTechEval
                  ? "View Evaluations"
                  : "Start Technical Evaluation"}
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Commercial Evaluation Card */}
          {canViewCommercial && (
            <div className="border rounded-lg p-6 flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <FileText className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <h3 className="font-medium">Commercial Evaluation</h3>
                  <p className="text-sm text-muted-foreground">
                    Evaluate contractors on commercial criteria per asset
                  </p>
                </div>
              </div>

              <Button
                variant={hasCommercialEval ? "outline" : "default"}
                className="mt-auto"
                onClick={() =>
                  navigate({
                    to: "/package/$id/comm",
                    params: { id },
                  })
                }
              >
                {hasCommercialEval
                  ? "View Evaluations"
                  : "Start Commercial Evaluation"}
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {/* Evaluation Summaries */}
        {((canViewTechnical && technicalRankings.length > 0) ||
          (canViewCommercial && commercialRankings.length > 0)) && (
          <div className="grid gap-4 md:grid-cols-2">
            {/* Technical Summary */}
            {canViewTechnical &&
              technicalRankings.length > 0 &&
              latestTechEval && (
                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium text-sm">Technical Rankings</h4>
                    <span className="text-xs text-muted-foreground">
                      {latestTechEval.roundName}
                    </span>
                  </div>
                  <div className="space-y-2">
                    {technicalRankings.map((contractor, index) => (
                      <div
                        key={contractor.id}
                        className="flex items-center justify-between text-sm py-1.5 px-2 rounded bg-muted/50"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-muted-foreground w-5">
                            #{index + 1}
                          </span>
                          <span>{contractor.name}</span>
                        </div>
                        <span className="font-medium">
                          {contractor.score.toFixed(1)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            {/* Commercial Summary */}
            {canViewCommercial && commercialRankings.length > 0 && (
              <div className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-sm">Commercial Rankings</h4>
                  <span className="text-xs text-muted-foreground">
                    Latest Round
                  </span>
                </div>
                <div className="space-y-2">
                  {commercialRankings.map((contractor, index) => (
                    <div
                      key={contractor.id}
                      className="flex items-center justify-between text-sm py-1.5 px-2 rounded bg-muted/50"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-muted-foreground w-5">
                          #{index + 1}
                        </span>
                        <span>{contractor.name}</span>
                      </div>
                      <span className="font-medium">
                        {formatCurrency(
                          contractor.total,
                          packageData.package.currency
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Award Package Sheet */}
      <Sheet open={awardSheetOpen} onOpenChange={setAwardSheetOpen}>
        <SheetContent className="flex flex-col">
          <SheetHeader className="px-6">
            <SheetTitle>Award Package</SheetTitle>
            <SheetDescription>
              Select a contractor to award this package to.
            </SheetDescription>
          </SheetHeader>

          <div className="flex-1 py-6 px-6 space-y-6 overflow-auto">
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Contractor</label>
              <Select
                value={selectedContractorId}
                onValueChange={setSelectedContractorId}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Choose a contractor" />
                </SelectTrigger>
                <SelectContent>
                  {eligibleContractors.map((contractor) => (
                    <SelectItem key={contractor.id} value={contractor.id}>
                      {contractor.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Only contractors with both technical and commercial evaluations
                are shown.
              </p>
            </div>

            {selectedContractor && (
              <div className="space-y-4">
                <h4 className="font-medium text-sm">Evaluation Summary</h4>

                {/* Technical Summary */}
                {selectedTechRank > 0 && (
                  <div className="p-4 border rounded-lg space-y-2">
                    <div className="flex items-center gap-2">
                      <ClipboardList className="h-4 w-4 text-blue-600" />
                      <span className="font-medium text-sm">
                        Technical Evaluation
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Rank</span>
                      <span className="font-medium">
                        #{selectedTechRank} of {technicalRankings.length}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Score</span>
                      <span className="font-medium">
                        {selectedTechScore?.toFixed(1)}
                      </span>
                    </div>
                  </div>
                )}

                {/* Commercial Summary */}
                {selectedCommRank > 0 && (
                  <div className="p-4 border rounded-lg space-y-2">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-green-600" />
                      <span className="font-medium text-sm">
                        Commercial Evaluation
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Rank</span>
                      <span className="font-medium">
                        #{selectedCommRank} of {commercialRankings.length}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Total Bid</span>
                      <span className="font-medium">
                        {formatCurrency(
                          selectedCommTotal ?? 0,
                          packageData.package.currency
                        )}
                      </span>
                    </div>
                  </div>
                )}

                {selectedTechRank === 0 && selectedCommRank === 0 && (
                  <p className="text-sm text-muted-foreground">
                    No evaluation data available for this contractor.
                  </p>
                )}
              </div>
            )}

            {/* Award Comments */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Comments</label>
              <textarea
                className="w-full min-h-[100px] px-3 py-2 text-sm border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="Add any comments or justification for this award decision..."
                value={awardComments}
                onChange={(e) => setAwardComments(e.target.value)}
              />
            </div>
          </div>

          <SheetFooter className="px-6">
            <Button
              variant="outline"
              onClick={() => {
                setAwardSheetOpen(false)
                setSelectedContractorId("")
                setAwardComments("")
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={() =>
                awardPackage.mutate({
                  contractorId: selectedContractorId,
                  comments: awardComments,
                })
              }
              disabled={!selectedContractorId || awardPackage.isPending}
            >
              <CheckCircle2 className="mr-2 h-4 w-4" />
              {awardPackage.isPending ? "Awarding..." : "Award"}
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </>
  )
}
