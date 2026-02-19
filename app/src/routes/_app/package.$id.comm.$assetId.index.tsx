import { useState, useMemo, Fragment, useEffect, useRef } from "react"
import { createFileRoute } from "@tanstack/react-router"
import {
  useSuspenseQuery,
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  assetDetailQueryOptions,
  commercialEvaluationsQueryOptions,
  packageContractorsQueryOptions,
} from "@/lib/query-options"
import {
  createCommercialEvaluationFn,
  updateCommercialEvaluationDataFn,
} from "@/fn/evaluations"
import {
  createCommRfpCompareJobFn,
  getCommRfpCompareJobStatusFn,
  getCommRfpCompareResultFn,
  getLatestCommRfpCompareJobStatusForAssetFn,
  getLatestCommRfpCompareResultForAssetFn,
} from "@/fn/jobs"
import {
  normalizeContractorBids,
  DEFAULT_NORMALIZATION_SETTINGS,
} from "@/lib/mock-boq-data"
import useStore from "@/lib/store"
import type {
  CommercialEvaluationData,
  CommRfpJobStatus,
  ContractorBid,
  NormalizationSettings,
  CustomOverrides,
  BOQLineItem,
  ContractorPTCs,
  PTCItem,
  PTCCategory,
} from "@/lib/types"
import {
  extractCommReportFromArtifact,
  getCommJobStatusText,
} from "@/lib/comm-job"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
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
import { UploadZone, type UploadedFile } from "@/components/ui/upload-zone"
import { StepTitle } from "@/components/ui/step-title"
import {
  BarChart3,
  UserIcon,
  Link as LinkIcon,
  ChevronRight,
  Settings2,
  AlertTriangle,
  RotateCcw,
  Loader2,
} from "lucide-react"
import { toast } from "sonner"
import { cn, formatCurrency } from "@/lib/utils"
import {
  VendorComparison,
  type ComparisonReport,
} from "@/components/VendorComparison"

const JOB_POLL_INTERVAL_MS = 3000
const JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000

type CommercialEvaluation = {
  id: string
  assetId: string
  roundNumber: number
  roundName: string
  data: CommercialEvaluationData | Record<string, never>
  createdAt: Date
  updatedAt: Date
}

const WARNING_TO_PTC_CATEGORY: Record<string, PTCCategory> = {
  missing_item: "exclusions",
  outlier_rate: "pricing_anomalies",
  total_mismatch: "arithmetic_checks",
}

const RATE_VARIANCE_THRESHOLD = 200

function generatePTCsFromReport(
  report: Record<string, unknown>
): ContractorPTCs[] {
  const vendors = (report.vendors as string[]) ?? []
  if (vendors.length === 0) return []

  const ptcsByVendor = new Map<string, PTCItem[]>()
  for (const v of vendors) ptcsByVendor.set(v, [])

  const warnings =
    (report.warnings as Array<{
      category: string
      vendor?: string
      division?: string
      grouping?: string
      item_id?: string
      item_description?: string
      message?: string
    }>) ?? []
  for (const w of warnings) {
    const vendor = w.vendor
    if (!vendor || !ptcsByVendor.has(vendor)) continue
    const category = WARNING_TO_PTC_CATEGORY[w.category]
    if (!category) continue

    const refParts = [w.division, w.grouping, w.item_id]
      .filter(Boolean)
      .join(" / ")

    ptcsByVendor.get(vendor)!.push({
      id: crypto.randomUUID(),
      referenceSection: refParts,
      queryDescription: w.message ?? `${w.category}: ${w.item_description ?? ""}`,
      vendorResponse: "",
      status: "pending",
      category,
    })
  }

  const divisions =
    (report.divisions as Array<{
      division_name?: string
      groupings?: Array<{
        grouping_name?: string
        line_items?: Array<{
          item_id?: string
          item_description?: string
          vendor_rates?: Record<string, number | null>
          rate_variance_percent?: number
          highest_bidder?: string
          lowest_bidder?: string
        }>
      }>
    }>) ?? []

  const warnedItems = new Set(
    warnings
      .filter((w) => w.item_id && w.vendor)
      .map((w) => `${w.vendor}::${w.item_id}`)
  )

  for (const div of divisions) {
    for (const grp of div.groupings ?? []) {
      for (const item of grp.line_items ?? []) {
        const variance = item.rate_variance_percent ?? 0
        if (variance < RATE_VARIANCE_THRESHOLD) continue

        for (const vendor of vendors) {
          const rate = item.vendor_rates?.[vendor]
          if (rate == null || rate <= 0) continue
          const key = `${vendor}::${item.item_id}`
          if (warnedItems.has(key)) continue
          warnedItems.add(key)

          const isHighest = item.highest_bidder === vendor
          const isLowest = item.lowest_bidder === vendor
          if (!isHighest && !isLowest) continue

          const ref = [div.division_name, grp.grouping_name, item.item_id]
            .filter(Boolean)
            .join(" / ")
          ptcsByVendor.get(vendor)!.push({
            id: crypto.randomUUID(),
            referenceSection: ref,
            queryDescription: `${isHighest ? "Highest" : "Lowest"} rate for "${(item.item_description ?? "").slice(0, 80)}" — ${variance.toFixed(0)}% variance across vendors. Please clarify pricing basis.`,
            vendorResponse: "",
            status: "pending",
            category: "deviations",
          })
        }
      }
    }
  }

  return vendors.map((vendor) => ({
    contractorId: vendor,
    contractorName: vendor,
    ptcs: ptcsByVendor.get(vendor) ?? [],
  }))
}

export const Route = createFileRoute("/_app/package/$id/comm/$assetId/")({
  // Data prefetching handled by parent layout route
  component: RouteComponent,
})

function RouteComponent() {
  const { id, assetId } = Route.useParams()
  const queryClient = useQueryClient()

  const { data: assetData } = useSuspenseQuery(assetDetailQueryOptions(assetId))
  const { data: evaluations } = useSuspenseQuery(
    commercialEvaluationsQueryOptions(assetId)
  )
  const { data: contractors } = useSuspenseQuery(
    packageContractorsQueryOptions(id)
  )

  const [isSetupOpen, setIsSetupOpen] = useState(false)
  const [templateFiles, setTemplateFiles] = useState<UploadedFile[]>([])
  const [pteFiles, setPteFiles] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<
    Record<string, UploadedFile[]>
  >({})

  const evaluationsList = evaluations as CommercialEvaluation[]

  // Get round from Zustand store
  const selectedRoundId = useStore((s) => s.selectedCommRound[assetId])
  const setCommRound = useStore((s) => s.setCommRound)

  // Get current round
  const currentRound = selectedRoundId
    ? evaluationsList.find((e) => e.id === selectedRoundId) ?? evaluationsList[0]
    : evaluationsList[0]

  const [jobStatus, setJobStatus] = useState<CommRfpJobStatus["status"] | null>(
    null
  )
  const [jobStatusText, setJobStatusText] = useState<string | null>(null)
  const [isCreatingEvaluation, setIsCreatingEvaluation] = useState(false)
  const [isRecoveringInProgress, setIsRecoveringInProgress] = useState(false)
  const vendorCount = Object.values(vendorFiles).filter(
    (arr) => arr.length > 0
  ).length
  const canRunQueue = templateFiles.length > 0 && vendorCount >= 2

  const createAndRunEvaluation = useMutation({
    mutationFn: async (newEval: CommercialEvaluation) => {
      const { jobId } = await createCommRfpCompareJobFn({
        data: {
          packageId: id,
          assetId,
          analysisType: "compare",
        },
      })

      await updateCommercialEvaluationDataFn({
        data: {
          evaluationId: newEval.id,
          data: {
            status: "analyzing",
            analysis: {
              jobId,
              status: "pending",
            },
          },
        },
      })

      const startedAt = Date.now()
      setJobStatus("pending")
      setJobStatusText("Queued for processing")

      while (true) {
        if (Date.now() - startedAt > JOB_POLL_TIMEOUT_MS) {
          throw new Error(
            "Commercial comparison timed out. Please try again or contact support."
          )
        }

        try {
          // Prefer artifact-first reconciliation to avoid getting stuck if
          // status updates lag behind completed result persistence.
          const readyResult = await getCommRfpCompareResultFn({
            data: { jobId },
          })
          const readyReport = extractCommReportFromArtifact(
            (readyResult as { data: unknown }).data
          )
          const readyPtcs = generatePTCsFromReport(readyReport)
          if (readyPtcs.length > 0) {
            readyReport.ptcs = readyPtcs
          }
          await updateCommercialEvaluationDataFn({
            data: { evaluationId: newEval.id, data: readyReport },
          })
          return { ...newEval, data: readyReport } as CommercialEvaluation
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          if (
            !message.includes("No result artifact found") &&
            !message.includes('No "result" or "report" artifact found')
          ) {
            throw error
          }
          // Artifact not available yet, continue with status polling.
        }

        const statusResult = (await getCommRfpCompareJobStatusFn({
          data: { jobId },
        })) as CommRfpJobStatus
        setJobStatus(statusResult.status)
        setJobStatusText(
          getCommJobStatusText(statusResult.status, statusResult.progress)
        )

        if (statusResult.status === "completed") {
          const result = await getCommRfpCompareResultFn({
            data: { jobId },
          })
          const reportData = extractCommReportFromArtifact(
            (result as { data: unknown }).data
          )
          const ptcs = generatePTCsFromReport(reportData)
          if (ptcs.length > 0) {
            reportData.ptcs = ptcs
          }
          await updateCommercialEvaluationDataFn({
            data: { evaluationId: newEval.id, data: reportData },
          })
          return { ...newEval, data: reportData } as CommercialEvaluation
        }

        if (
          statusResult.status === "failed" ||
          statusResult.status === "cancelled"
        ) {
          try {
            await updateCommercialEvaluationDataFn({
              data: {
                evaluationId: newEval.id,
                data: {
                  status: "failed",
                  analysis: {
                    jobId,
                    status: statusResult.status,
                    error: statusResult.error ?? "Commercial comparison failed.",
                  },
                },
              },
            })
          } catch {}
          throw new Error(
            statusResult.error ?? "Commercial comparison failed in worker."
          )
        }

        await new Promise((resolve) =>
          setTimeout(resolve, JOB_POLL_INTERVAL_MS)
        )
      }
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
      })
      setIsSetupOpen(false)
      setTemplateFiles([])
      setPteFiles([])
      setVendorFiles({})
      setJobStatus(null)
      setJobStatusText(null)
    },
    onError: (error) => {
      setJobStatus("failed")
      toast.error(
        error instanceof Error ? error.message : "Failed to create evaluation"
      )
    },
  })

  const handleRunEvaluation = async () => {
    if (!canRunQueue) {
      toast.error(
        "Upload one template and at least two vendor files before running evaluation."
      )
      return
    }

    setIsCreatingEvaluation(true)
    try {
      const newEval = (await createCommercialEvaluationFn({
        data: { assetId },
      })) as CommercialEvaluation
      setCommRound(assetId, newEval.id)
      await updateCommercialEvaluationDataFn({
        data: {
          evaluationId: newEval.id,
          data: {
            status: "analyzing",
            analysis: {
              status: "pending",
            },
          },
        },
      })
      await queryClient.invalidateQueries({
        queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
      })
      toast.success(`${newEval.roundName} created`)
      setIsSetupOpen(false)
      createAndRunEvaluation.mutate(newEval)
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to create commercial evaluation"
      )
    } finally {
      setIsCreatingEvaluation(false)
    }
  }

  const handleOpenSetup = () => {
    setIsSetupOpen(true)
  }

  const recoverMissingData = useMutation({
    mutationFn: async () => {
      if (!currentRound) {
        throw new Error("No round selected")
      }

      const recovered = await getLatestCommRfpCompareResultForAssetFn({
        data: {
          packageId: id,
          assetId,
        },
      })
      const report = extractCommReportFromArtifact(
        (recovered as { data: unknown }).data
      )
      const ptcs = generatePTCsFromReport(report)
      if (ptcs.length > 0) {
        report.ptcs = ptcs
      }

      await updateCommercialEvaluationDataFn({
        data: { evaluationId: currentRound.id, data: report },
      })
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
      })
      toast.success("Recovered evaluation data from job artifact")
    },
    onError: (error) => {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to recover evaluation data"
      )
    },
  })

  const hasEvaluations = evaluationsList.length > 0
  const data = currentRound?.data as Record<string, unknown> | undefined
  const persistedEvaluationStatus = useMemo(() => {
    if (!data || typeof data !== "object") return null
    const status = (data as Record<string, unknown>).status
    return typeof status === "string" ? status : null
  }, [data])
  const analysisState = useMemo(() => {
    if (!data || typeof data !== "object") return null
    const record = data as {
      analysis?: { jobId?: unknown; status?: unknown; error?: unknown }
      status?: unknown
    }
    if (!record.analysis || typeof record.analysis !== "object") return null
    const jobId =
      typeof record.analysis.jobId === "string" ? record.analysis.jobId : null
    if (!jobId) return null
    const status =
      typeof record.analysis.status === "string" ? record.analysis.status : "pending"
    const error =
      typeof record.analysis.error === "string" ? record.analysis.error : undefined
    return { jobId, status, error, evaluationStatus: record.status }
  }, [data])
  const resolvedReportData = useMemo(() => {
    if (!data) return null
    try {
      return extractCommReportFromArtifact(data)
    } catch {
      return null
    }
  }, [data])
  const hasReportData = Boolean(
    resolvedReportData &&
      "vendors" in resolvedReportData &&
      "summary" in resolvedReportData &&
      "divisions" in resolvedReportData
  )
  const hasLegacyData = data && "boq" in data && "contractors" in data
  const hasData = hasReportData || hasLegacyData
  const attemptedAutoRecoveryRef = useRef(false)
  const latestAssetJobStatusQuery = useQuery({
    queryKey: ["comm-rfp", "latest-status", id, assetId] as const,
    queryFn: () =>
      getLatestCommRfpCompareJobStatusForAssetFn({
        data: { packageId: id, assetId },
      }),
    enabled: Boolean(
      hasEvaluations && currentRound && !hasData && !analysisState?.jobId
    ),
    refetchInterval: (query) => {
      const result = query.state.data as
        | {
            found: true
            status: CommRfpJobStatus["status"]
          }
        | { found: false }
        | undefined
      if (!result || !result.found) return false
      return result.status === "pending" || result.status === "in_progress"
        ? JOB_POLL_INTERVAL_MS
        : false
    },
  })
  const latestAssetJobStatus = latestAssetJobStatusQuery.data as
    | {
        found: true
        status: CommRfpJobStatus["status"]
        progress: CommRfpJobStatus["progress"]
      }
    | { found: false }
    | undefined

  // No contractors case
  if (contractors.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-12 text-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <UserIcon className="size-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No contractors</h3>
        <p className="text-muted-foreground mb-6 max-w-sm">
          Add contractors to this package before running a commercial
          evaluation.
        </p>
        <Button variant="outline" asChild>
          <a href={`/package/${id}/contractors?addContractor=true`}>
            <LinkIcon className="size-4 mr-2" />
            Go to Contractors
          </a>
        </Button>
      </div>
    )
  }

  const isExtracting =
    createAndRunEvaluation.isPending ||
    isCreatingEvaluation ||
    isRecoveringInProgress ||
    (!hasData &&
      (persistedEvaluationStatus === "analyzing" ||
        (latestAssetJobStatus?.found &&
          latestAssetJobStatus.status !== "completed" &&
          latestAssetJobStatus.status !== "failed" &&
          latestAssetJobStatus.status !== "cancelled")))

  useEffect(() => {
    if (!hasEvaluations || !currentRound || hasData || !analysisState?.jobId) {
      return
    }
    if (
      analysisState.status === "failed" ||
      analysisState.status === "cancelled"
    ) {
      setJobStatus(analysisState.status as CommRfpJobStatus["status"])
      setJobStatusText(
        analysisState.error ?? "Commercial comparison failed. Please rerun."
      )
      return
    }

    let cancelled = false
    setIsRecoveringInProgress(true)
    setJobStatus("pending")
    setJobStatusText("Recovering commercial comparison status...")

    const poll = async () => {
      try {
        try {
          const readyResult = await getCommRfpCompareResultFn({
            data: { jobId: analysisState.jobId },
          })
          if (cancelled) return
          const readyReport = extractCommReportFromArtifact(
            (readyResult as { data: unknown }).data
          )
          const readyPtcs = generatePTCsFromReport(readyReport)
          if (readyPtcs.length > 0) {
            readyReport.ptcs = readyPtcs
          }
          await updateCommercialEvaluationDataFn({
            data: { evaluationId: currentRound.id, data: readyReport },
          })
          await queryClient.invalidateQueries({
            queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
          })
          setIsRecoveringInProgress(false)
          return
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          if (
            !message.includes("No result artifact found") &&
            !message.includes('No "result" or "report" artifact found')
          ) {
            throw error
          }
        }

        const statusResult = (await getCommRfpCompareJobStatusFn({
          data: { jobId: analysisState.jobId },
        })) as CommRfpJobStatus
        if (cancelled) return
        setJobStatus(statusResult.status)
        setJobStatusText(
          getCommJobStatusText(statusResult.status, statusResult.progress)
        )

        if (
          statusResult.status === "failed" ||
          statusResult.status === "cancelled"
        ) {
          setIsRecoveringInProgress(false)
          return
        }

        window.setTimeout(() => {
          if (!cancelled) void poll()
        }, JOB_POLL_INTERVAL_MS)
      } catch {
        if (cancelled) return
        setIsRecoveringInProgress(false)
      }
    }

    void poll()

    return () => {
      cancelled = true
    }
  }, [
    analysisState,
    assetId,
    currentRound,
    hasData,
    hasEvaluations,
    queryClient,
  ])

  useEffect(() => {
    if (!hasEvaluations || !currentRound || hasData) return
    if (persistedEvaluationStatus === "analyzing") return
    if (analysisState?.jobId) return
    if (latestAssetJobStatus?.found && latestAssetJobStatus.status !== "completed")
      return
    if (recoverMissingData.isPending) return
    if (attemptedAutoRecoveryRef.current) return

    attemptedAutoRecoveryRef.current = true
    recoverMissingData.mutate()
  }, [
    analysisState?.jobId,
    hasEvaluations,
    currentRound,
    hasData,
    latestAssetJobStatus,
    persistedEvaluationStatus,
    recoverMissingData.isPending,
    recoverMissingData,
  ])

  // Extraction in progress
  if (isExtracting) {
    return (
      <ExtractionProgressView
        assetName={assetData.asset.name}
          status={
            jobStatus ??
            (analysisState?.status as CommRfpJobStatus["status"] | null) ??
            (latestAssetJobStatus?.found ? latestAssetJobStatus.status : null) ??
            (persistedEvaluationStatus === "analyzing" ? "pending" : null)
          }
          statusText={
            jobStatusText ??
            (latestAssetJobStatus?.found
              ? getCommJobStatusText(
                  latestAssetJobStatus.status,
                  latestAssetJobStatus.progress ?? null
                )
              : null) ??
            (persistedEvaluationStatus === "analyzing"
              ? "Processing in progress..."
              : null)
          }
      />
    )
  }

  // No evaluations yet - show setup sheet
  if (!hasEvaluations) {
    return (
      <>
        <div className="flex flex-col items-center justify-center h-full p-12 text-center">
          <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
            <BarChart3 className="size-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">
            No commercial evaluation yet
          </h3>
          <p className="text-muted-foreground mb-6 max-w-sm">
            Start a commercial evaluation for{" "}
            <strong>{assetData.asset.name}</strong> to compare contractor
            proposals.
          </p>
          <Button onClick={handleOpenSetup}>Start Commercial Evaluation</Button>
        </div>

        <CommercialSetupSheet
          open={isSetupOpen}
          onOpenChange={setIsSetupOpen}
          packageId={id}
          assetName={assetData.asset.name}
          contractors={contractors}
          templateFiles={templateFiles}
          onTemplateFilesChange={setTemplateFiles}
          pteFiles={pteFiles}
          onPteFilesChange={setPteFiles}
          vendorFiles={vendorFiles}
          onVendorFilesChange={(vendorId, files) =>
            setVendorFiles((prev) => ({ ...prev, [vendorId]: files }))
          }
          onRunEvaluation={() => {
            handleRunEvaluation()
          }}
          isPending={createAndRunEvaluation.isPending || isCreatingEvaluation}
        />
      </>
    )
  }

  if (hasData) {
    if (hasReportData) {
      return (
        <VendorComparison
          report={resolvedReportData as unknown as ComparisonReport}
        />
      )
    }
    return (
      <BOQTable
        evaluationData={currentRound!.data as CommercialEvaluationData}
      />
    )
  }

  // Fallback for rounds without persisted data
  return (
    <div className="flex flex-col items-center justify-center h-full p-12 text-center">
      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
        <AlertTriangle className="size-8 text-amber-600" />
      </div>
      <h3 className="text-lg font-semibold mb-2">No evaluation data found</h3>
      <p className="text-muted-foreground mb-6 max-w-sm">
        This round has no persisted comparison report yet. You can run the
        evaluation again for this asset.
      </p>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          onClick={() => recoverMissingData.mutate()}
          disabled={recoverMissingData.isPending}
        >
          {recoverMissingData.isPending
            ? "Recovering..."
            : "Recover From Job Artifact"}
        </Button>
        <Button onClick={handleOpenSetup}>Run Evaluation</Button>
      </div>
    </div>
  )
}

// ============================================================================
// Extraction Progress View (shown on main page during extraction)
// ============================================================================

function ExtractionProgressView({
  assetName,
  status,
  statusText,
}: {
  assetName: string
  status: CommRfpJobStatus["status"] | null
  statusText: string | null
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-12 text-center">
      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
        <Loader2 className="size-8 text-primary animate-spin" />
      </div>
      <h3 className="text-lg font-semibold mb-2">
        {status === "pending" ? "Queued" : "Running Commercial Comparison"}
      </h3>
      <p className="text-muted-foreground">{assetName}</p>
      <p className="text-sm text-muted-foreground mt-2">
        {statusText ?? "Processing commercial comparison"}
      </p>
    </div>
  )
}

// ============================================================================
// Commercial Setup Sheet
// ============================================================================

function CommercialSetupSheet({
  open,
  onOpenChange,
  packageId,
  assetName,
  contractors,
  templateFiles,
  onTemplateFilesChange,
  pteFiles,
  onPteFilesChange,
  vendorFiles,
  onVendorFilesChange,
  onRunEvaluation,
  isPending,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  packageId: string
  assetName: string
  contractors: Array<{ id: string; name: string }>
  templateFiles: UploadedFile[]
  onTemplateFilesChange: (files: UploadedFile[]) => void
  pteFiles: UploadedFile[]
  onPteFilesChange: (files: UploadedFile[]) => void
  vendorFiles: Record<string, UploadedFile[]>
  onVendorFilesChange: (vendorId: string, files: UploadedFile[]) => void
  onRunEvaluation: () => void
  isPending: boolean
}) {
  const [uploadingKeys, setUploadingKeys] = useState<Set<string>>(new Set())
  const isBoqDone = templateFiles.length > 0
  const vendorsWithFiles = Object.values(vendorFiles).filter(
    (arr) => arr.length > 0
  ).length
  const canRunEvaluation = isBoqDone && vendorsWithFiles >= 2
  const isUploading = uploadingKeys.size > 0

  const updateUploadState = (key: string, uploading: boolean) => {
    setUploadingKeys((prev) => {
      const next = new Set(prev)
      if (uploading) {
        next.add(key)
      } else {
        next.delete(key)
      }
      return next
    })
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Start Commercial Evaluation</SheetTitle>
          <SheetDescription>
            Upload documents and vendor proposals. At least 2 vendors must have
            files to proceed.
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 p-4 space-y-6">
          {/* Asset Name (read-only) */}
          <div className="space-y-2">
            <Label>Asset Name</Label>
            <Input value={assetName} disabled className="bg-muted" />
          </div>

          {/* BOQ File */}
          <div className="space-y-2">
            <StepTitle
              title="Bill Of Quantities (BOQ)"
              complete={isBoqDone}
              required
            />
            <UploadZone
              files={templateFiles}
              onFilesChange={onTemplateFilesChange}
              packageId={packageId}
              category="boq"
              storagePathMode="comm_rfp_boq"
              accept=".xlsx,.xls"
              onUploadingChange={(uploading) =>
                updateUploadState("boq", uploading)
              }
            />
          </div>

          {/* PTE File */}
          <div className="space-y-2">
            <StepTitle
              title="Pre-Tender Estimate (PTE)"
              complete={pteFiles.length > 0}
              description="Optional"
            />
            <UploadZone
              files={pteFiles}
              onFilesChange={onPteFilesChange}
              packageId={packageId}
              category="pte"
              storagePathMode="comm_rfp_rfp"
              accept=".xlsx,.xls"
              onUploadingChange={(uploading) =>
                updateUploadState("pte", uploading)
              }
            />
          </div>

          {/* Vendor Files */}
          <div className="space-y-3">
            <StepTitle
              title={`Vendor Proposals (${vendorsWithFiles}/${contractors.length} vendors have files)`}
              complete={canRunEvaluation}
              required
            />

            {contractors.length === 0 ? (
              <div className="text-center py-6 border rounded-lg border-dashed">
                <UserIcon className="size-8 mx-auto text-muted-foreground mb-2" />
                <p className="text-sm text-muted-foreground">
                  No contractors added to this package yet.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {contractors.map((contractor) => {
                  const files = vendorFiles[contractor.id] ?? []
                  const hasFiles = files.length > 0

                  return (
                    <div
                      key={contractor.id}
                      className={cn(
                        "rounded-lg border p-3 transition-colors",
                        hasFiles &&
                          "border-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20"
                      )}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="flex items-center justify-center w-6 h-6 rounded bg-muted">
                          <UserIcon
                            size={14}
                            className="text-muted-foreground"
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {contractor.name}
                        </span>
                      </div>
                      <UploadZone
                        files={files}
                        onFilesChange={(nextFiles) =>
                          onVendorFilesChange(contractor.id, nextFiles.slice(0, 1))
                        }
                        packageId={packageId}
                        category="vendor_proposal"
                        storagePathMode="comm_rfp_tender"
                        vendorName={contractor.name}
                        accept=".xlsx,.xls,.pdf,.doc,.docx"
                        compact
                        onUploadingChange={(uploading) =>
                          updateUploadState(`vendor:${contractor.id}`, uploading)
                        }
                      />
                    </div>
                  )
                })}

                {!canRunEvaluation && (
                  <p className="text-sm text-amber-600 dark:text-amber-500">
                    At least 2 vendors must have files to run evaluation
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <SheetFooter className="px-4 pb-4">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={onRunEvaluation}
            disabled={!canRunEvaluation || isPending || isUploading}
          >
            {isPending
              ? "Running..."
              : isUploading
                ? "Uploading..."
                : "Run Evaluation"}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}

// ============================================================================
// BOQ Table Component
// ============================================================================

function BOQTable({
  evaluationData,
}: {
  evaluationData: CommercialEvaluationData
}) {
  const [viewMode, setViewMode] = useState<"received" | "normalized">(
    "received"
  )
  const [normalizationSettings, setNormalizationSettings] =
    useState<NormalizationSettings>(DEFAULT_NORMALIZATION_SETTINGS)
  const [customOverrides, setCustomOverrides] = useState<CustomOverrides>({})
  const [isSettingsSheetOpen, setIsSettingsSheetOpen] = useState(false)
  const [selectedIssueCell, setSelectedIssueCell] = useState<{
    contractorId: string
    itemId: string
    item: BOQLineItem
  } | null>(null)
  const [expandedDivisions, setExpandedDivisions] = useState<Set<string>>(
    new Set()
  )
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set()
  )

  const { boq, contractors: rawContractors } = evaluationData
  const isNormalized = viewMode === "normalized"
  const hasCustomOverrides = Object.keys(customOverrides).length > 0

  // Compute normalized contractors if needed
  const contractors = useMemo(() => {
    if (isNormalized) {
      return normalizeContractorBids(
        boq,
        rawContractors,
        normalizationSettings,
        customOverrides
      )
    }
    return rawContractors
  }, [
    isNormalized,
    boq,
    rawContractors,
    normalizationSettings,
    customOverrides,
  ])

  // Track issue types for each cell
  const cellIssues = useMemo(() => {
    const issues: Record<string, "included" | "unpriced" | "arithmetic_error"> =
      {}
    for (const contractor of rawContractors) {
      for (const itemId of contractor.includedItems ?? []) {
        issues[`${contractor.contractorId}-${itemId}`] = "included"
      }
      for (const itemId of Object.keys(contractor.arithmeticErrors ?? {})) {
        issues[`${contractor.contractorId}-${itemId}`] = "arithmetic_error"
      }
      for (const [itemId, price] of Object.entries(contractor.prices ?? {})) {
        const key = `${contractor.contractorId}-${itemId}`
        if (price === null && !issues[key]) {
          issues[key] = "unpriced"
        }
      }
    }
    return issues
  }, [rawContractors])

  // The lowest bidder is always index 0 (contractors are pre-sorted)
  const lowestBidderId = contractors[0]?.contractorId

  const toggleDivision = (divId: string) => {
    setExpandedDivisions((prev) => {
      const next = new Set(prev)
      if (next.has(divId)) {
        next.delete(divId)
      } else {
        next.add(divId)
      }
      return next
    })
  }

  const toggleSection = (secId: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(secId)) {
        next.delete(secId)
      } else {
        next.add(secId)
      }
      return next
    })
  }

  // Calculate subtotals for divisions and sections
  const calculateSubtotal = (
    itemIds: string[],
    contractor: ContractorBid
  ): number => {
    return itemIds.reduce((sum, itemId) => {
      const price = contractor.prices[itemId]
      return sum + (price ?? 0)
    }, 0)
  }

  // Helper to get column styling for lowest bidder
  const getColumnClass = (contractorId: string, baseClass: string = "") => {
    const isLowest = contractorId === lowestBidderId
    return cn(baseClass, isLowest && "bg-emerald-50/80 dark:bg-emerald-950/30")
  }

  // Handle cell click for issues
  const handleCellClick = (
    contractorId: string,
    itemId: string,
    item: BOQLineItem
  ) => {
    const key = `${contractorId}-${itemId}`
    const issue = cellIssues[key]
    // Only open sheet for clickable issues in normalized view (not "included")
    if (isNormalized && issue && issue !== "included") {
      setSelectedIssueCell({ contractorId, itemId, item })
    }
  }

  // Handle custom override save
  const handleSaveOverride = (value: string) => {
    if (!selectedIssueCell) return
    const key = `${selectedIssueCell.contractorId}-${selectedIssueCell.itemId}`

    if (value === "" || value === undefined) {
      // Remove override if empty
      setCustomOverrides((prev) => {
        const next = { ...prev }
        delete next[key]
        return next
      })
    } else {
      const numValue = parseFloat(value)
      if (!isNaN(numValue)) {
        setCustomOverrides((prev) => ({
          ...prev,
          [key]: numValue,
        }))
      }
    }
    setSelectedIssueCell(null)
  }

  // Revert all custom overrides
  const handleRevertOverrides = () => {
    setCustomOverrides({})
    toast.success("Custom values reverted")
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-8 bg-background sticky top-0 z-10">
        <h2 className="text-base font-semibold">Asset Summary</h2>
        <div className="flex items-center gap-2">
          {/* Revert custom values button */}
          {hasCustomOverrides && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRevertOverrides}
              className="text-xs h-8"
            >
              <RotateCcw className="size-3.5 mr-1.5" />
              Revert custom values
            </Button>
          )}

          {/* Button group toggle */}
          <div className="flex items-center rounded-lg bg-muted p-1 gap-1">
            <button
              type="button"
              onClick={() => setViewMode("received")}
              className={cn(
                "h-7 text-xs font-medium rounded-md px-3 transition-colors",
                viewMode === "received"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Bids as received
            </button>
            <button
              type="button"
              onClick={() => setViewMode("normalized")}
              className={cn(
                "h-7 text-xs font-medium rounded-md px-3 transition-colors",
                viewMode === "normalized"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Normalized
            </button>
            <button
              type="button"
              onClick={() => setIsSettingsSheetOpen(true)}
              className="h-7 px-2 flex items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-background/70 transition-colors"
              title="Normalization settings"
            >
              <Settings2 className="size-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto px-4 py-4">
        <div className="w-full max-w-6xl mx-auto rounded-xl border bg-card shadow-sm overflow-hidden">
          <div className="overflow-auto p-2">
            <table className="w-full table-fixed border-collapse text-xs">
              <thead className="sticky top-0 bg-background/95 backdrop-blur-sm z-10">
                <tr>
                  <th className="text-left px-2 py-1.5 border-b font-medium text-muted-foreground w-20">
                    Code
                  </th>
                  <th className="text-left px-2 py-1.5 border-b font-medium text-muted-foreground w-[32%] whitespace-normal break-words [overflow-wrap:anywhere]">
                    Description
                  </th>
                  <th className="text-right px-2 py-1.5 border-b font-medium text-muted-foreground w-16">
                    Qty
                  </th>
                  <th className="text-left px-2 py-1.5 border-b font-medium text-muted-foreground w-12">
                    Unit
                  </th>
                  {contractors.map((contractor, idx) => {
                    const isLowest = contractor.contractorId === lowestBidderId
                    return (
                      <th
                        key={contractor.contractorId}
                        className={cn(
                          "text-right px-2 py-1.5 border-b font-medium w-[100px]",
                          isLowest && "bg-emerald-100/80 dark:bg-emerald-900/40"
                        )}
                      >
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={cn(
                              "truncate max-w-[100px] text-xs",
                              isLowest &&
                                "text-emerald-700 dark:text-emerald-400"
                            )}
                          >
                            {contractor.contractorName}
                          </span>
                          <span
                            className={cn(
                              "text-[10px] font-normal",
                              isLowest
                                ? "text-emerald-600 dark:text-emerald-500"
                                : "text-muted-foreground"
                            )}
                          >
                            {isLowest ? "#1 Lowest" : `#${idx + 1}`}
                          </span>
                        </div>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {boq.divisions.map((division) => {
                  const isDivExpanded = expandedDivisions.has(division.id)
                  const divisionItemIds = division.sections.flatMap((s) =>
                    s.lineItems.map((li) => li.id)
                  )

                  return (
                    <Fragment key={division.id}>
                      {/* Division Row */}
                      <tr
                        className="bg-accent/20 cursor-pointer hover:bg-accent/30 transition-colors"
                        onClick={() => toggleDivision(division.id)}
                      >
                        <td className="px-2 py-1.5 border-b font-semibold text-xs">
                          <div className="flex items-center gap-1">
                            <ChevronRight
                              className={cn(
                                "size-3.5 transition-transform text-muted-foreground",
                                isDivExpanded && "rotate-90"
                              )}
                            />
                            {division.code}
                          </div>
                        </td>
                        <td
                          className="px-2 py-1.5 border-b font-semibold text-xs whitespace-normal break-words [overflow-wrap:anywhere]"
                          colSpan={3}
                        >
                          {division.name}
                        </td>
                        {contractors.map((contractor) => (
                          <td
                            key={contractor.contractorId}
                            className={cn(
                              "px-2 py-1.5 border-b text-right font-semibold tabular-nums",
                              getColumnClass(contractor.contractorId)
                            )}
                          >
                            {formatCurrency(
                              calculateSubtotal(divisionItemIds, contractor)
                            )}
                          </td>
                        ))}
                      </tr>

                      {/* Sections (when division expanded) */}
                      {isDivExpanded &&
                        division.sections.map((section) => {
                          const isSecExpanded = expandedSections.has(section.id)
                          const sectionItemIds = section.lineItems.map(
                            (li) => li.id
                          )

                          return (
                            <Fragment key={section.id}>
                              {/* Section Row */}
                              <tr
                                className="bg-accent/10 cursor-pointer hover:bg-accent/20 transition-colors"
                                onClick={() => toggleSection(section.id)}
                              >
                                <td className="px-2 py-1 border-b pl-5 font-medium text-xs">
                                  <div className="flex items-center gap-1">
                                    <ChevronRight
                                      className={cn(
                                        "size-3 transition-transform text-muted-foreground",
                                        isSecExpanded && "rotate-90"
                                      )}
                                    />
                                    {section.code}
                                  </div>
                                </td>
                                <td
                                  className="px-2 py-1 border-b font-medium text-xs whitespace-normal break-words [overflow-wrap:anywhere]"
                                  colSpan={3}
                                >
                                  {section.name}
                                </td>
                                {contractors.map((contractor) => (
                                  <td
                                    key={contractor.contractorId}
                                    className={cn(
                                      "px-2 py-1 border-b text-right font-medium tabular-nums",
                                      getColumnClass(contractor.contractorId)
                                    )}
                                  >
                                    {formatCurrency(
                                      calculateSubtotal(
                                        sectionItemIds,
                                        contractor
                                      )
                                    )}
                                  </td>
                                ))}
                              </tr>

                              {/* Line Items (when section expanded) */}
                              {isSecExpanded &&
                                section.lineItems.map((item) => (
                                  <tr
                                    key={item.id}
                                    className="hover:bg-accent/30 transition-colors"
                                  >
                                    <td className="px-2 py-1 border-b pl-8 text-muted-foreground text-[11px]">
                                      {item.code}
                                    </td>
                                    <td className="px-2 py-1 border-b text-[11px] whitespace-normal break-words [overflow-wrap:anywhere]">
                                      {item.description}
                                    </td>
                                    <td className="px-2 py-1 border-b text-right tabular-nums text-[11px]">
                                      {item.quantity.toLocaleString()}
                                    </td>
                                    <td className="px-2 py-1 border-b text-muted-foreground text-[11px]">
                                      {item.unit}
                                    </td>
                                    {contractors.map((contractor) => {
                                      const rawContractor = rawContractors.find(
                                        (c) =>
                                          c.contractorId ===
                                          contractor.contractorId
                                      )!
                                      const key = `${contractor.contractorId}-${item.id}`
                                      const issue = cellIssues[key]
                                      const hasCustomOverride =
                                        customOverrides[key] !== undefined
                                      const price = contractor.prices[item.id]
                                      const isLowest =
                                        contractor.contractorId ===
                                        lowestBidderId
                                      const isClickable =
                                        isNormalized &&
                                        issue &&
                                        issue !== "included"

                                      // Determine cell styling
                                      const getCellStyles = () => {
                                        // Custom override styling (purple)
                                        if (isNormalized && hasCustomOverride) {
                                          return "bg-purple-50 dark:bg-purple-950/50 text-purple-600 dark:text-purple-400 cursor-pointer"
                                        }
                                        // Included items (gray)
                                        if (issue === "included") {
                                          return "bg-slate-100 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400"
                                        }
                                        // Arithmetic error - as received (orange/red with warning)
                                        if (
                                          issue === "arithmetic_error" &&
                                          !isNormalized
                                        ) {
                                          return "bg-orange-50 dark:bg-orange-950/50 text-orange-600 dark:text-orange-400"
                                        }
                                        // Arithmetic error - normalized (blue, clickable)
                                        if (
                                          issue === "arithmetic_error" &&
                                          isNormalized
                                        ) {
                                          return "bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 italic cursor-pointer"
                                        }
                                        // Unpriced - as received (amber)
                                        if (
                                          issue === "unpriced" &&
                                          !isNormalized
                                        ) {
                                          return "bg-amber-50 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400"
                                        }
                                        // Unpriced - normalized (blue, clickable)
                                        if (
                                          issue === "unpriced" &&
                                          isNormalized
                                        ) {
                                          return "bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 italic cursor-pointer"
                                        }
                                        // Lowest bidder highlight
                                        if (isLowest) {
                                          return "bg-emerald-50/80 dark:bg-emerald-950/30"
                                        }
                                        return ""
                                      }

                                      // Render cell content
                                      const renderCellContent = () => {
                                        // Included items always show "Included"
                                        if (issue === "included") {
                                          return (
                                            <span className="text-[10px]">
                                              Included
                                            </span>
                                          )
                                        }
                                        // Arithmetic error in "as received" view - show with warning icon
                                        if (
                                          issue === "arithmetic_error" &&
                                          !isNormalized
                                        ) {
                                          const error =
                                            rawContractor.arithmeticErrors?.[
                                              item.id
                                            ]
                                          if (error) {
                                            return (
                                              <span className="flex items-center justify-end gap-1">
                                                <AlertTriangle className="size-3" />
                                                {formatCurrency(
                                                  error.submitted
                                                )}
                                              </span>
                                            )
                                          }
                                        }
                                        // Unpriced in "as received" view
                                        if (
                                          issue === "unpriced" &&
                                          !isNormalized
                                        ) {
                                          return "—"
                                        }
                                        // Show price (normalized value, custom override, or original)
                                        return price !== null
                                          ? formatCurrency(price)
                                          : "—"
                                      }

                                      return (
                                        <td
                                          key={contractor.contractorId}
                                          className={cn(
                                            "px-2 py-1 border-b text-right tabular-nums text-[11px]",
                                            getCellStyles()
                                          )}
                                          onClick={() =>
                                            isClickable &&
                                            handleCellClick(
                                              contractor.contractorId,
                                              item.id,
                                              item
                                            )
                                          }
                                        >
                                          {renderCellContent()}
                                        </td>
                                      )
                                    })}
                                  </tr>
                                ))}
                            </Fragment>
                          )
                        })}
                    </Fragment>
                  )
                })}

                {/* Grand Total Row */}
                <tr className="bg-accent/20 font-bold">
                  <td className="px-2 py-2 border-t-2 text-xs" colSpan={4}>
                    Grand Total
                  </td>
                  {contractors.map((contractor) => {
                    const isLowest = contractor.contractorId === lowestBidderId
                    return (
                      <td
                        key={contractor.contractorId}
                        className={cn(
                          "px-2 py-2 border-t-2 text-right tabular-nums text-xs",
                          isLowest &&
                            "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-400"
                        )}
                      >
                        {formatCurrency(contractor.totalAmount)}
                      </td>
                    )
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Normalization Settings Sheet */}
      <NormalizationSettingsSheet
        open={isSettingsSheetOpen}
        onOpenChange={setIsSettingsSheetOpen}
        settings={normalizationSettings}
        onSettingsChange={setNormalizationSettings}
      />

      {/* Item Issue Sheet */}
      <ItemIssueSheet
        open={selectedIssueCell !== null}
        onOpenChange={(open) => !open && setSelectedIssueCell(null)}
        cellInfo={selectedIssueCell}
        rawContractors={rawContractors}
        contractors={contractors}
        cellIssues={cellIssues}
        customOverrides={customOverrides}
        normalizationSettings={normalizationSettings}
        onSave={handleSaveOverride}
      />
    </div>
  )
}

// ============================================================================
// Normalization Settings Sheet
// ============================================================================

function NormalizationSettingsSheet({
  open,
  onOpenChange,
  settings,
  onSettingsChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  settings: NormalizationSettings
  onSettingsChange: (settings: NormalizationSettings) => void
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-md flex flex-col">
        <SheetHeader className="px-6 pt-6">
          <SheetTitle>Normalization Settings</SheetTitle>
          <SheetDescription>
            Configure how prices are normalized for comparison.
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 px-6 py-6 space-y-6 overflow-y-auto">
          {/* Issues to Normalize */}
          <div className="space-y-4">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Issues to Normalize
            </Label>
            <div className="space-y-3 p-4 rounded-lg border bg-card">
              <div className="flex items-center space-x-3">
                <Checkbox
                  id="normalize-unpriced"
                  checked={settings.normalizeUnpriced}
                  onCheckedChange={(checked) =>
                    onSettingsChange({
                      ...settings,
                      normalizeUnpriced: !!checked,
                    })
                  }
                />
                <Label
                  htmlFor="normalize-unpriced"
                  className="text-sm font-normal cursor-pointer"
                >
                  Unpriced Items
                </Label>
              </div>
              <div className="flex items-center space-x-3">
                <Checkbox
                  id="normalize-arithmetic"
                  checked={settings.normalizeArithmeticErrors}
                  onCheckedChange={(checked) =>
                    onSettingsChange({
                      ...settings,
                      normalizeArithmeticErrors: !!checked,
                    })
                  }
                />
                <Label
                  htmlFor="normalize-arithmetic"
                  className="text-sm font-normal cursor-pointer"
                >
                  Arithmetic Errors
                </Label>
              </div>
            </div>
          </div>

          {/* Algorithm Selection */}
          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Normalization Algorithm
            </Label>
            <Select
              value={settings.algorithm}
              onValueChange={(value: "median" | "lowest") =>
                onSettingsChange({ ...settings, algorithm: value })
              }
            >
              <SelectTrigger className="w-full h-11">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="median">Median Price</SelectItem>
                <SelectItem value="lowest">Lowest Price</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {settings.algorithm === "median"
                ? "Missing or erroneous prices will be replaced with the median price from other contractors."
                : "Missing or erroneous prices will be replaced with the lowest price from other contractors."}
            </p>
          </div>
        </div>

        <SheetFooter className="px-6 pb-6 pt-4 border-t bg-muted/30">
          <Button onClick={() => onOpenChange(false)}>Done</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}

// ============================================================================
// Item Issue Sheet
// ============================================================================

function ItemIssueSheet({
  open,
  onOpenChange,
  cellInfo,
  rawContractors,
  contractors,
  cellIssues,
  customOverrides,
  normalizationSettings,
  onSave,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  cellInfo: { contractorId: string; itemId: string; item: BOQLineItem } | null
  rawContractors: ContractorBid[]
  contractors: ContractorBid[]
  cellIssues: Record<string, "included" | "unpriced" | "arithmetic_error">
  customOverrides: CustomOverrides
  normalizationSettings: NormalizationSettings
  onSave: (value: string) => void
}) {
  const [customValue, setCustomValue] = useState("")

  // Reset custom value when sheet opens
  const key = cellInfo ? `${cellInfo.contractorId}-${cellInfo.itemId}` : ""
  const existingOverride = customOverrides[key]

  // Update local state when opening
  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen && existingOverride !== undefined) {
      setCustomValue(existingOverride.toString())
    } else if (newOpen) {
      setCustomValue("")
    }
    onOpenChange(newOpen)
  }

  if (!cellInfo) return null

  const { contractorId, itemId, item } = cellInfo
  const issue = cellIssues[key]
  const rawContractor = rawContractors.find(
    (c) => c.contractorId === contractorId
  )
  const normalizedContractor = contractors.find(
    (c) => c.contractorId === contractorId
  )
  const normalizedPrice = normalizedContractor?.prices[itemId]

  const getIssueDescription = () => {
    if (issue === "unpriced") {
      return "This item was not priced by the contractor."
    }
    if (issue === "arithmetic_error" && rawContractor) {
      const error = rawContractor.arithmeticErrors?.[itemId]
      if (error) {
        return `Arithmetic error detected: Submitted ${formatCurrency(error.submitted)}, but calculated value is ${formatCurrency(error.calculated)}.`
      }
    }
    return ""
  }

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent className="sm:max-w-md flex flex-col">
        <SheetHeader className="px-6 pt-6">
          <SheetTitle>Price Override</SheetTitle>
          <SheetDescription>{rawContractor?.contractorName}</SheetDescription>
        </SheetHeader>

        <div className="flex-1 px-6 py-6 space-y-5 overflow-y-auto">
          {/* Item Details */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Item
            </Label>
            <div className="p-4 rounded-lg border bg-card">
              <p className="font-medium text-sm">{item.description}</p>
              <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                <span className="font-mono bg-muted px-1.5 py-0.5 rounded">
                  {item.code}
                </span>
                <span className="text-muted-foreground/50">|</span>
                <span>
                  {item.quantity.toLocaleString()} {item.unit}
                </span>
              </div>
            </div>
          </div>

          {/* Issue Description */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Issue
            </Label>
            <div
              className={cn(
                "p-4 rounded-lg text-sm border-l-4",
                issue === "unpriced" &&
                  "bg-amber-50 dark:bg-amber-950/30 border-amber-400 text-amber-800 dark:text-amber-200",
                issue === "arithmetic_error" &&
                  "bg-orange-50 dark:bg-orange-950/30 border-orange-400 text-orange-800 dark:text-orange-200"
              )}
            >
              {getIssueDescription()}
            </div>
          </div>

          {/* Current Normalized Price */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Normalized Price (
              {normalizationSettings.algorithm === "median"
                ? "Median"
                : "Lowest"}
              )
            </Label>
            <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800">
              <span className="text-xl font-semibold text-blue-700 dark:text-blue-300">
                {normalizedPrice != null
                  ? formatCurrency(normalizedPrice)
                  : "—"}
              </span>
            </div>
          </div>

          {/* Custom Override */}
          <div className="space-y-3 pt-2 border-t">
            <Label
              htmlFor="custom-price"
              className="text-xs font-medium text-muted-foreground uppercase tracking-wide"
            >
              Override with Custom Price
            </Label>
            <Input
              id="custom-price"
              type="number"
              placeholder="Enter custom price..."
              value={customValue}
              onChange={(e) => setCustomValue(e.target.value)}
              className="w-full h-11 text-base"
            />
            <p className="text-xs text-muted-foreground">
              Leave empty to use the normalized value above.
            </p>
          </div>
        </div>

        <SheetFooter className="px-6 pb-6 pt-4 border-t bg-muted/30">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => onSave(customValue)}>Save</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
