import { useEffect, useMemo, useState } from "react"
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import { toast } from "sonner"
import { ClipboardList, FileText } from "lucide-react"
import { FlowPageLayout } from "@/components/FlowPageLayout"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { awardPackageFn } from "@/fn/packages"
import {
  packageAccessQueryOptions,
  packageCommercialSummaryQueryOptions,
  packageContractorsQueryOptions,
  packageDetailQueryOptions,
  projectDetailQueryOptions,
  technicalEvaluationDetailQueryOptions,
  technicalEvaluationsQueryOptions,
} from "@/lib/query-options"
import type { TechnicalEvaluationData } from "@/components/TechSetupWizard"
import type { CommercialEvaluationData } from "@/lib/types"
import { formatCurrency } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

function calculateWeightedScore(
  contractorId: string,
  evalData: TechnicalEvaluationData
): number {
  const allBreakdowns = evalData.criteria?.scopes?.flatMap((scope) => scope.breakdowns) ?? []
  const contractorScores = evalData.scores?.[contractorId] ?? {}

  let totalScore = 0
  for (const breakdown of allBreakdowns) {
    const score = contractorScores[breakdown.id]?.score ?? 0
    totalScore += (score * breakdown.weight) / 100
  }
  return totalScore
}

function getTechnicalRankings(
  evalData: TechnicalEvaluationData | null,
  contractors: Array<{ id: string; name: string }>
) {
  if (!evalData || !evalData.scores) return []

  const proposalsUploaded = evalData.proposalsUploaded ?? []
  const evaluated = contractors.filter((contractor) =>
    proposalsUploaded.includes(contractor.id)
  )

  return evaluated
    .map((contractor) => ({
      id: contractor.id,
      name: contractor.name,
      score: calculateWeightedScore(contractor.id, evalData),
    }))
    .sort((a, b) => b.score - a.score)
}

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

export const Route = createFileRoute("/_app/award-package/$packageId")({
  beforeLoad: async ({ params, context }) => {
    const accessData = await context.queryClient.ensureQueryData(
      packageAccessQueryOptions(params.packageId)
    )
    if (accessData.access !== "full") {
      throw redirect({ to: "/package/$id", params: { id: params.packageId } })
    }
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { packageId } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [selectedContractorId, setSelectedContractorId] = useState("")
  const [awardComments, setAwardComments] = useState("")

  const { data: packageData } = useQuery(packageDetailQueryOptions(packageId))
  const { data: contractors } = useQuery(packageContractorsQueryOptions(packageId))
  const { data: technicalEvals } = useQuery(
    technicalEvaluationsQueryOptions(packageId)
  )
  const { data: commercialSummary } = useQuery(
    packageCommercialSummaryQueryOptions(packageId)
  )

  if (!packageData || !contractors || !technicalEvals || !commercialSummary) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const latestTechEval = technicalEvals.length > 0 ? technicalEvals[0] : null
  const { data: techEvalDetail } = useQuery({
    ...technicalEvaluationDetailQueryOptions(latestTechEval?.id ?? ""),
    enabled: !!latestTechEval,
  })

  const techEvalData = (techEvalDetail?.data as TechnicalEvaluationData | null) ?? null
  const technicalRankings = useMemo(
    () => getTechnicalRankings(techEvalData, contractors),
    [techEvalData, contractors]
  )
  const commercialRankings = useMemo(
    () => getCommercialRankings(commercialSummary ?? null),
    [commercialSummary]
  )

  const isTechReviewComplete = techEvalData?.status === "review_complete"
  const eligibleContractors = useMemo(() => {
    const techIds = new Set(technicalRankings.map((item) => item.id))
    const commIds = new Set(commercialRankings.map((item) => item.id))
    return contractors.filter((contractor) => techIds.has(contractor.id) && commIds.has(contractor.id))
  }, [commercialRankings, contractors, technicalRankings])

  useEffect(() => {
    if (selectedContractorId || technicalRankings.length === 0) return
    const topTechContractor = technicalRankings[0]
    if (eligibleContractors.some((item) => item.id === topTechContractor.id)) {
      setSelectedContractorId(topTechContractor.id)
    }
  }, [eligibleContractors, selectedContractorId, technicalRankings])

  const selectedContractor = contractors.find((contractor) => contractor.id === selectedContractorId)
  const selectedTechRank = technicalRankings.findIndex((item) => item.id === selectedContractorId) + 1
  const selectedTechScore = technicalRankings.find((item) => item.id === selectedContractorId)?.score
  const selectedCommRank = commercialRankings.findIndex((item) => item.id === selectedContractorId) + 1
  const selectedCommTotal = commercialRankings.find((item) => item.id === selectedContractorId)?.total

  const canSubmit = selectedContractorId.length > 0 && isTechReviewComplete

  const awardPackage = useMutation({
    mutationFn: (data: { contractorId: string; comments?: string }) =>
      awardPackageFn({
        data: {
          packageId,
          contractorId: data.contractorId,
          comments: data.comments,
        },
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(packageId).queryKey,
      })
      await queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(packageData.project.id).queryKey,
      })
      toast.success("Package awarded successfully")
      navigate({ to: "/package/$id", params: { id: packageId } })
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to award package")
    },
  })

  return (
    <FlowPageLayout
      title="Award package"
      context={<span>Package: {packageData.package.name}</span>}
      backLabel={packageData.package.name}
      onBack={() => navigate({ to: "/package/$id", params: { id: packageId } })}
    >
      <div className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Select contractor</label>
          <Select value={selectedContractorId} onValueChange={setSelectedContractorId}>
            <SelectTrigger className="bg-white border-black/15 focus:border-black/35">
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
          {!isTechReviewComplete ? (
            <p className="text-xs text-amber-700">
              Complete technical score review on the latest round before awarding.
            </p>
          ) : null}
        </div>

        {selectedContractor ? (
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-black/75">Evaluation summary</h3>
            {selectedTechRank > 0 ? (
              <div className="rounded-lg border border-black/10 px-4 py-3 space-y-2">
                <div className="inline-flex items-center gap-2 text-sm font-medium">
                  <ClipboardList className="size-4 text-blue-600" />
                  Technical
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-black/50">Rank</span>
                  <span>#{selectedTechRank}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-black/50">Score</span>
                  <span>{selectedTechScore?.toFixed(1)}</span>
                </div>
              </div>
            ) : null}
            {selectedCommRank > 0 ? (
              <div className="rounded-lg border border-black/10 px-4 py-3 space-y-2">
                <div className="inline-flex items-center gap-2 text-sm font-medium">
                  <FileText className="size-4 text-green-600" />
                  Commercial
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-black/50">Rank</span>
                  <span>#{selectedCommRank}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-black/50">Total bid</span>
                  <span>
                    {formatCurrency(selectedCommTotal ?? 0, packageData.package.currency)}
                  </span>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        <div className="space-y-2">
          <label className="text-sm font-medium">Comments</label>
          <Textarea
            className="min-h-[110px] bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
            placeholder="Add any context for this award decision..."
            value={awardComments}
            onChange={(e) => setAwardComments(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Button
            type="button"
            variant="outline"
            className="w-full"
            onClick={() => navigate({ to: "/package/$id", params: { id: packageId } })}
            disabled={awardPackage.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            className="w-full bg-black text-white hover:bg-black/90"
            onClick={() =>
              awardPackage.mutate({
                contractorId: selectedContractorId,
                comments: awardComments,
              })
            }
            disabled={!canSubmit || awardPackage.isPending}
          >
            {awardPackage.isPending ? "Awarding..." : "Award package"}
          </Button>
        </div>
      </div>
    </FlowPageLayout>
  )
}
