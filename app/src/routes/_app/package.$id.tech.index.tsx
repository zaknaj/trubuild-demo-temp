import { useState, useEffect, Fragment, useRef } from "react"
import { createFileRoute } from "@tanstack/react-router"
import {
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  technicalEvaluationDetailQueryOptions,
  technicalEvaluationsQueryOptions,
  packageContractorsQueryOptions,
} from "@/lib/query-options"
import { updateTechnicalEvaluationFn } from "@/fn/evaluations"
import {
  createTechRfpAnalysisJobFn,
  getTechRfpAnalysisJobStatusFn,
  getTechRfpAnalysisResultFn,
  writeTechRfpEvaluationJsonFn,
} from "@/fn/jobs"
import useStore from "@/lib/store"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Spinner } from "@/components/ui/spinner"
import {
  Check,
  CheckCircle2,
  Circle,
  Plus,
  Trash2,
  FileText,
  Pencil,
  X,
} from "lucide-react"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import { mapAnalysisResultToTechnicalPtcs } from "@/lib/technical-ptc"
import {
  type Breakdown,
  type TechnicalEvaluationData,
  type ScoreData,
  type Evidence,
  type EvidenceFile,
  type LineReference,
} from "@/components/TechSetupWizard"
import type {
  TechRfpContractorData,
  TechRfpEvaluationCriteriaType,
  TechRfpResult,
  TechRfpResultArtifact,
} from "@/lib/tech_rfp"

export const Route = createFileRoute("/_app/package/$id/tech/")({
  component: RouteComponent,
})

function normalizeMatchKey(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
}

function sanitizeScore(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(100, value))
}

function toTechRfpEvaluationCriteria(
  scopes: TechnicalEvaluationData["criteria"]["scopes"]
): TechRfpEvaluationCriteriaType {
  const criteria: TechRfpEvaluationCriteriaType = {}
  for (const scope of scopes) {
    const scopeName = scope.name.trim()
    if (!scopeName) continue
    criteria[scopeName] = {}
    for (const breakdown of scope.breakdowns) {
      const title = breakdown.title.trim()
      if (!title) continue
      criteria[scopeName][title] = {
        weight: breakdown.weight,
        description: breakdown.description.trim(),
      }
    }
  }
  return criteria
}

function extractTenderReport(raw: unknown): TechRfpContractorData[] {
  const artifact = raw as
    | TechRfpResultArtifact
    | TechRfpResult
    | Record<string, unknown>

  if (
    artifact &&
    typeof artifact === "object" &&
    "result" in artifact &&
    artifact.result &&
    typeof artifact.result === "object" &&
    "tenderReport" in artifact.result
  ) {
    const wrappedReport = (artifact.result as { tenderReport?: unknown })
      .tenderReport
    if (Array.isArray(wrappedReport)) {
      return wrappedReport as TechRfpContractorData[]
    }
  }

  if (
    artifact &&
    typeof artifact === "object" &&
    "tenderReport" in artifact &&
    Array.isArray((artifact as { tenderReport?: unknown }).tenderReport)
  ) {
    return (artifact as TechRfpResultArtifact).tenderReport
  }

  throw new Error("Analysis result is malformed: missing tenderReport")
}

function mapAnalysisResultToScores({
  tenderReport,
  criteria,
  contractors,
  proposalsUploaded,
}: {
  tenderReport: TechRfpContractorData[]
  criteria: TechnicalEvaluationData["criteria"]
  contractors: Array<{ id: string; name: string }>
  proposalsUploaded: string[]
}): Record<string, Record<string, ScoreData>> {
  const scores: Record<string, Record<string, ScoreData>> = {}
  const activeContractors = contractors.filter((contractor) =>
    proposalsUploaded.includes(contractor.id)
  )
  if (activeContractors.length === 0) {
    throw new Error("No active contractors found for technical evaluation")
  }
  if (tenderReport.length === 0) {
    throw new Error("Analysis result has no contractor rows")
  }

  const contractorByKey = new Map<string, { id: string; name: string }>()
  for (const contractor of activeContractors) {
    const key = normalizeMatchKey(contractor.name)
    if (contractorByKey.has(key)) {
      throw new Error(
        `Duplicate contractor names after normalization: "${contractor.name}"`
      )
    }
    contractorByKey.set(key, contractor)
    scores[contractor.id] = {}
  }

  const breakdownByKey = new Map<string, Breakdown>()
  for (const scope of criteria.scopes) {
    for (const breakdown of scope.breakdowns) {
      const key = normalizeMatchKey(breakdown.title)
      if (breakdownByKey.has(key)) {
        throw new Error(
          `Duplicate criteria names after normalization: "${breakdown.title}"`
        )
      }
      breakdownByKey.set(key, breakdown)
    }
  }

  const seenResultContractors = new Set<string>()
  for (const contractorResult of tenderReport) {
    const resultContractorKey = normalizeMatchKey(contractorResult.contractorName)
    if (!resultContractorKey) {
      throw new Error("Analysis result contains an empty contractor name")
    }
    if (seenResultContractors.has(resultContractorKey)) {
      throw new Error(
        `Analysis result has duplicate contractor entry: "${contractorResult.contractorName}"`
      )
    }
    seenResultContractors.add(resultContractorKey)

    const contractorMatch = contractorByKey.get(resultContractorKey)
    if (!contractorMatch) {
      throw new Error(
        `Result contractor "${contractorResult.contractorName}" does not match any uploaded proposal contractor`
      )
    }

    for (const scope of contractorResult.tenderEvaluation.scopes) {
      for (const criterion of scope.evaluationBreakdown) {
        const criterionKey = normalizeMatchKey(criterion.criteria)
        if (!criterionKey) {
          throw new Error(
            `Analysis result has an empty criterion for contractor "${contractorMatch.name}"`
          )
        }

        const breakdown = breakdownByKey.get(criterionKey)
        if (!breakdown) {
          throw new Error(
            `Result criterion "${criterion.criteria}" does not match any configured criteria`
          )
        }
        if (scores[contractorMatch.id][breakdown.id]) {
          throw new Error(
            `Duplicate score mapping for contractor "${contractorMatch.name}" and criterion "${breakdown.title}"`
          )
        }

        const numericScore = Number(criterion.score)
        if (!Number.isFinite(numericScore)) {
          throw new Error(
            `Invalid score for contractor "${contractorMatch.name}" criterion "${criterion.criteria}"`
          )
        }
        const clampedScore = Math.max(0, Math.min(100, numericScore))
        const criterionEvidence = Array.isArray(criterion.evidence)
          ? criterion.evidence
          : []

        scores[contractorMatch.id][breakdown.id] = {
          score: clampedScore,
          comment: criterion.clientSummary?.trim() || undefined,
          approved: false,
          evidence: criterionEvidence.map((item) => ({
            id: crypto.randomUUID(),
            text: item.text,
            source: "auto",
            files: [],
            lineReference: item.pageNumber
              ? {
                  fileName: item.source,
                  startLine: Math.max(1, Math.floor(item.pageNumber)),
                }
              : undefined,
          })),
        }
      }
    }
  }

  const missingMappings: string[] = []
  for (const contractor of activeContractors) {
    for (const scope of criteria.scopes) {
      for (const breakdown of scope.breakdowns) {
        if (!scores[contractor.id][breakdown.id]) {
          missingMappings.push(`${contractor.name} -> ${breakdown.title}`)
        }
      }
    }
  }

  if (missingMappings.length > 0) {
    const preview = missingMappings.slice(0, 3).join(", ")
    throw new Error(
      `Analysis result is incomplete. Missing score mappings: ${preview}${missingMappings.length > 3 ? "..." : ""}`
    )
  }

  return scores
}

function RouteComponent() {
  const { id: packageId } = Route.useParams()

  // Get round from Zustand store
  const evaluationId = useStore((s) => s.selectedTechRound[packageId])
  const setTechRound = useStore((s) => s.setTechRound)

  const { data: contractors } = useQuery(packageContractorsQueryOptions(packageId))
  const { data: technicalEvaluations } = useQuery(
    technicalEvaluationsQueryOptions(packageId)
  ) as { data: Array<{ id: string }> | undefined }

  if (!contractors) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  useEffect(() => {
    if (!evaluationId && technicalEvaluations && technicalEvaluations.length > 0) {
      setTechRound(packageId, technicalEvaluations[0].id)
    }
  }, [evaluationId, packageId, setTechRound, technicalEvaluations])

  if (!evaluationId) {
    if (!technicalEvaluations) {
      return (
        <div className="p-6">
          <p className="text-muted-foreground">Loading technical evaluation...</p>
        </div>
      )
    }
    return (
      <div className="p-6">
        <p className="text-muted-foreground">
          {technicalEvaluations.length === 0
            ? "Create a technical evaluation to get started"
            : "Loading selected evaluation..."}
        </p>
      </div>
    )
  }

  return (
    <EvaluationContent
      packageId={packageId}
      evaluationId={evaluationId}
      contractors={contractors}
    />
  )
}

function EvaluationContent({
  packageId,
  evaluationId,
  contractors,
}: {
  packageId: string
  evaluationId: string
  contractors: Array<{ id: string; name: string }>
}) {
  const queryClient = useQueryClient()
  const isReconcilingRef = useRef(false)
  const [isReviewOpen, setIsReviewOpen] = useState(false)
  const [initialCell, setInitialCell] = useState<{
    contractorId: string
    breakdownId: string
  } | null>(null)

  const { data: evaluation } = useQuery({
    ...technicalEvaluationDetailQueryOptions(evaluationId),
    refetchInterval: (query) => {
      const evalRecord = query.state.data as { data?: unknown } | undefined
      const data = (evalRecord?.data ?? {}) as Partial<TechnicalEvaluationData>
      return data.status === "analyzing" ? 3000 : false
    },
  }) as { data: { id: string; data: unknown } | undefined }

  const evalData = (evaluation?.data ?? {}) as Partial<TechnicalEvaluationData>
  const status = evalData.status ?? "analyzing"
  const analysis = evalData.analysis

  const retryAnalysis = useMutation({
    mutationFn: async () => {
      const criteria = evalData.criteria?.scopes
      if (!criteria || criteria.length === 0) {
        throw new Error("Cannot retry analysis without criteria")
      }

      const evaluationCriteria = toTechRfpEvaluationCriteria(criteria)
      await writeTechRfpEvaluationJsonFn({
        data: {
          packageId,
          evaluationCriteria,
        },
      })

      const { jobId } = await createTechRfpAnalysisJobFn({
        data: {
          packageId,
          evaluationCriteria,
        },
      })

      const nextData: TechnicalEvaluationData = {
        ...(evalData as TechnicalEvaluationData),
        status: "analyzing",
        analysis: {
          jobId,
          status: "queued",
        },
      }

      await updateTechnicalEvaluationFn({
        data: {
          evaluationId,
          data: nextData,
        },
      })
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["technical-evaluation", evaluationId, "detail"],
      })
      toast.success("Analysis restarted")
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to retry")
    },
  })

  useEffect(() => {
    if (!analysis?.jobId || status !== "analyzing" || isReconcilingRef.current) {
      return
    }
    if (analysis.status === "failed") {
      return
    }
    const criteria = evalData.criteria
    if (!criteria) {
      return
    }
    const proposalsUploaded =
      evalData.proposalsUploaded && evalData.proposalsUploaded.length > 0
        ? evalData.proposalsUploaded
        : contractors.map((contractor) => contractor.id)

    let cancelled = false

    const reconcileAnalysis = async () => {
      isReconcilingRef.current = true
      try {
        try {
          const result = await getTechRfpAnalysisResultFn({
            data: { jobId: analysis.jobId },
          })
          if (cancelled) return

          const tenderReport = extractTenderReport(result.data)
          const computedScores = mapAnalysisResultToScores({
            tenderReport,
            criteria,
            contractors,
            proposalsUploaded,
          })
          const computedPtcs = mapAnalysisResultToTechnicalPtcs({
            tenderReport,
            contractors,
            proposalsUploaded,
          })

          const readyData: TechnicalEvaluationData = {
            ...(evalData as TechnicalEvaluationData),
            status: "ready",
            scores: computedScores,
            ptcs: computedPtcs,
            analysis: {
              jobId: analysis.jobId,
              status: "completed",
            },
          }

          await updateTechnicalEvaluationFn({
            data: { evaluationId, data: readyData },
          })
          await queryClient.invalidateQueries({
            queryKey: ["technical-evaluation", evaluationId, "detail"],
          })
          return
        } catch {
          // Artifact may not be ready yet; fall back to status polling.
        }

        const statusResult = await getTechRfpAnalysisJobStatusFn({
          data: { jobId: analysis.jobId },
        })
        if (cancelled) return

        if (
          statusResult.status === "pending" ||
          statusResult.status === "in_progress"
        ) {
          const nextStatus =
            statusResult.status === "pending" ? "queued" : "running"
          if (analysis.status !== nextStatus || analysis.error) {
            const runningData: TechnicalEvaluationData = {
              ...(evalData as TechnicalEvaluationData),
              analysis: {
                jobId: analysis.jobId,
                status: nextStatus,
              },
            }
            await updateTechnicalEvaluationFn({
              data: { evaluationId, data: runningData },
            })
            await queryClient.invalidateQueries({
              queryKey: ["technical-evaluation", evaluationId, "detail"],
            })
          }
          return
        }

        if (
          statusResult.status === "failed" ||
          statusResult.status === "cancelled"
        ) {
          const failedData: TechnicalEvaluationData = {
            ...(evalData as TechnicalEvaluationData),
            analysis: {
              jobId: analysis.jobId,
              status: "failed",
              error: statusResult.error ?? "Technical analysis failed",
            },
          }
          await updateTechnicalEvaluationFn({
            data: { evaluationId, data: failedData },
          })
          await queryClient.invalidateQueries({
            queryKey: ["technical-evaluation", evaluationId, "detail"],
          })
          return
        }
      } catch (error) {
        if (cancelled) return
        const errorMessage =
          error instanceof Error ? error.message : "Failed to process analysis result"
        const failedData: TechnicalEvaluationData = {
          ...(evalData as TechnicalEvaluationData),
          analysis: {
            jobId: analysis.jobId,
            status: "failed",
            error: errorMessage,
          },
        }
        try {
          await updateTechnicalEvaluationFn({
            data: { evaluationId, data: failedData },
          })
          await queryClient.invalidateQueries({
            queryKey: ["technical-evaluation", evaluationId, "detail"],
          })
        } catch {}
      } finally {
        if (!cancelled) {
          isReconcilingRef.current = false
        }
      }
    }

    reconcileAnalysis()

    return () => {
      cancelled = true
    }
  }, [
    analysis?.jobId,
    analysis?.status,
    contractors,
    evalData,
    evaluationId,
    queryClient,
    status,
  ])

  const handleOpenReview = (contractorId?: string, breakdownId?: string) => {
    if (contractorId && breakdownId) {
      setInitialCell({ contractorId, breakdownId })
    } else {
      setInitialCell(null)
    }
    setIsReviewOpen(true)
  }

  if (!evaluation) {
    return <AnalyzingState />
  }

  if (status === "analyzing") {
    if (analysis?.status === "failed") {
      return (
        <AnalysisFailedState
          error={analysis.error}
          onRetry={() => retryAnalysis.mutate()}
          isRetrying={retryAnalysis.isPending}
        />
      )
    }
    return <AnalyzingState />
  }

  if (status === "ready" || status === "review_complete") {
    return (
      <>
        <ScoresTable
          evalData={evalData as TechnicalEvaluationData}
          contractors={contractors}
          onOpenReview={handleOpenReview}
        />
        <ReviewSheet
          open={isReviewOpen}
          onOpenChange={setIsReviewOpen}
          evaluationId={evaluationId}
          evalData={evalData as TechnicalEvaluationData}
          contractors={contractors}
          initialCell={initialCell}
        />
      </>
    )
  }

  // Fallback - shouldn't happen with new flow
  return <AnalyzingState />
}

// ============================================================================
// Analyzing State
// ============================================================================

function AnalyzingState() {
  return (
    <div className="flex flex-col items-center justify-center h-full p-12 text-center">
      <Spinner className="size-12 mb-4" />
      <h3 className="text-lg font-semibold mb-2">Analyzing Proposals</h3>
      <p className="text-muted-foreground max-w-sm">
        Extracting and scoring contractor proposals against evaluation
        criteria...
      </p>
    </div>
  )
}

function AnalysisFailedState({
  error,
  onRetry,
  isRetrying,
}: {
  error?: string
  onRetry: () => void
  isRetrying: boolean
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-12 text-center">
      <h3 className="text-lg font-semibold mb-2">Analysis Failed</h3>
      <p className="text-muted-foreground max-w-md mb-4">
        {error ?? "The technical evaluation job failed. Please retry analysis."}
      </p>
      <Button onClick={onRetry} disabled={isRetrying}>
        {isRetrying ? "Retrying..." : "Retry Analysis"}
      </Button>
    </div>
  )
}

// ============================================================================
// Evidence Section
// ============================================================================

function EvidenceSection({
  evidence,
  onChange,
}: {
  evidence: Evidence[]
  onChange: (evidence: Evidence[]) => void
}) {
  const [isAddingEvidence, setIsAddingEvidence] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)

  const handleAddEvidence = (newEvidence: Omit<Evidence, "id" | "source">) => {
    onChange([
      ...evidence,
      {
        ...newEvidence,
        id: crypto.randomUUID(),
        source: "manual",
      },
    ])
    setIsAddingEvidence(false)
  }

  const handleUpdateEvidence = (id: string, updates: Partial<Evidence>) => {
    onChange(evidence.map((e) => (e.id === id ? { ...e, ...updates } : e)))
    setEditingId(null)
  }

  const handleDeleteEvidence = (id: string) => {
    onChange(evidence.filter((e) => e.id !== id))
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">
          Evidence{" "}
          <span className="text-muted-foreground">({evidence.length})</span>
        </label>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsAddingEvidence(true)}
          disabled={isAddingEvidence}
        >
          <Plus className="size-4 mr-1" />
          Add
        </Button>
      </div>

      {isAddingEvidence && (
        <EvidenceForm
          onSave={handleAddEvidence}
          onCancel={() => setIsAddingEvidence(false)}
        />
      )}

      <div className="space-y-2">
        {evidence.map((item) => (
          <EvidenceItem
            key={item.id}
            evidence={item}
            isEditing={editingId === item.id}
            onEdit={() => setEditingId(item.id)}
            onUpdate={(updates) => handleUpdateEvidence(item.id, updates)}
            onDelete={() => handleDeleteEvidence(item.id)}
            onCancelEdit={() => setEditingId(null)}
          />
        ))}
        {evidence.length === 0 && !isAddingEvidence && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No evidence added yet
          </p>
        )}
      </div>
    </div>
  )
}

function getSourceFileName(source: string): string {
  const trimmed = source.trim()
  if (!trimmed) return "source"

  const withoutQuery = trimmed.split("?")[0].split("#")[0]
  const parts = withoutQuery.split("/").filter(Boolean)
  return parts[parts.length - 1] ?? trimmed
}

function EvidenceItem({
  evidence,
  isEditing,
  onEdit,
  onUpdate,
  onDelete,
  onCancelEdit,
}: {
  evidence: Evidence
  isEditing: boolean
  onEdit: () => void
  onUpdate: (updates: Partial<Evidence>) => void
  onDelete: () => void
  onCancelEdit: () => void
}) {
  if (isEditing) {
    return (
      <EvidenceForm
        initialData={evidence}
        onSave={(data) => onUpdate(data)}
        onCancel={onCancelEdit}
      />
    )
  }

  return (
    <div className="border rounded-lg p-3 space-y-2 bg-card">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm flex-1">{evidence.text}</p>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon-sm" onClick={onEdit}>
            <Pencil className="size-3" />
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={onDelete}>
            <Trash2 className="size-3" />
          </Button>
        </div>
      </div>

      {(evidence.files.length > 0 || evidence.lineReference) && (
        <div className="flex flex-wrap items-center gap-2">
          {evidence.files.map((file) => (
            <span
              key={file.id}
              className="inline-flex items-center gap-1 text-xs bg-muted px-2 py-1 rounded"
            >
              <FileText className="size-3" />
              {file.name}
            </span>
          ))}
          {evidence.lineReference && (
            <span className="text-xs text-muted-foreground">
              {getSourceFileName(evidence.lineReference.fileName)}:
              {evidence.lineReference.startLine}
              {evidence.lineReference.endLine
                ? `-${evidence.lineReference.endLine}`
                : ""}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function EvidenceForm({
  initialData,
  onSave,
  onCancel,
}: {
  initialData?: Evidence
  onSave: (data: Omit<Evidence, "id" | "source">) => void
  onCancel: () => void
}) {
  const [text, setText] = useState(initialData?.text ?? "")
  const [files, setFiles] = useState<EvidenceFile[]>(initialData?.files ?? [])
  const [lineRef, setLineRef] = useState<LineReference | undefined>(
    initialData?.lineReference
  )
  const [newFileName, setNewFileName] = useState("")

  const handleAddFile = () => {
    if (!newFileName.trim()) return
    setFiles([
      ...files,
      {
        id: crypto.randomUUID(),
        name: newFileName.trim(),
        fakeUrl: `/uploads/${newFileName.trim()}`,
      },
    ])
    setNewFileName("")
  }

  const handleRemoveFile = (id: string) => {
    setFiles(files.filter((f) => f.id !== id))
  }

  const handleSubmit = () => {
    if (!text.trim()) return
    onSave({
      text: text.trim(),
      files,
      lineReference: lineRef,
    })
  }

  return (
    <div className="border rounded-lg p-3 space-y-3 bg-muted/30">
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Describe the evidence..."
        rows={2}
        className="text-sm"
      />

      {/* Files */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">
          Attached Files
        </label>
        <div className="flex flex-wrap gap-2">
          {files.map((file) => (
            <span
              key={file.id}
              className="inline-flex items-center gap-1 text-xs bg-background px-2 py-1 rounded border"
            >
              <FileText className="size-3" />
              {file.name}
              <button
                onClick={() => handleRemoveFile(file.id)}
                className="hover:text-destructive"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
        </div>
        <div className="flex gap-2">
          <Input
            value={newFileName}
            onChange={(e) => setNewFileName(e.target.value)}
            placeholder="filename.pdf"
            className="text-sm h-8"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault()
                handleAddFile()
              }
            }}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={handleAddFile}
            disabled={!newFileName.trim()}
          >
            Add File
          </Button>
        </div>
      </div>

      {/* Line Reference */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">
          Line Reference (optional)
        </label>
        <div className="grid grid-cols-3 gap-2">
          <Input
            value={lineRef?.fileName ?? ""}
            onChange={(e) =>
              setLineRef(
                e.target.value
                  ? {
                      ...lineRef,
                      fileName: e.target.value,
                      startLine: lineRef?.startLine ?? 1,
                    }
                  : undefined
              )
            }
            placeholder="filename.pdf"
            className="text-sm h-8"
          />
          <Input
            type="number"
            value={lineRef?.startLine ?? ""}
            onChange={(e) =>
              setLineRef(
                lineRef
                  ? { ...lineRef, startLine: parseInt(e.target.value) || 1 }
                  : { fileName: "", startLine: parseInt(e.target.value) || 1 }
              )
            }
            placeholder="Start line"
            className="text-sm h-8"
          />
          <Input
            type="number"
            value={lineRef?.endLine ?? ""}
            onChange={(e) =>
              setLineRef(
                lineRef
                  ? {
                      ...lineRef,
                      endLine: e.target.value
                        ? parseInt(e.target.value)
                        : undefined,
                    }
                  : undefined
              )
            }
            placeholder="End line"
            className="text-sm h-8"
          />
        </div>
      </div>

      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button size="sm" onClick={handleSubmit} disabled={!text.trim()}>
          {initialData ? "Update" : "Add Evidence"}
        </Button>
      </div>
    </div>
  )
}

// ============================================================================
// Scores Table
// ============================================================================

function ScoresTable({
  evalData,
  contractors,
  onOpenReview,
}: {
  evalData: TechnicalEvaluationData
  contractors: Array<{ id: string; name: string }>
  onOpenReview: (contractorId?: string, breakdownId?: string) => void
}) {
  const { criteria, scores, status, proposalsUploaded } = evalData

  // Only show contractors that have proposals
  const activeContractors = contractors.filter((c) =>
    proposalsUploaded.includes(c.id)
  )

  // Calculate approval stats
  let totalScores = 0
  let approvedScores = 0
  for (const scope of criteria.scopes) {
    for (const breakdown of scope.breakdowns) {
      for (const contractor of activeContractors) {
        totalScores++
        if (scores[contractor.id]?.[breakdown.id]?.approved) {
          approvedScores++
        }
      }
    }
  }
  const approvalPercent =
    totalScores > 0 ? Math.round((approvedScores / totalScores) * 100) : 0
  const isComplete = status === "review_complete"

  const formatScore = (score: number | null): string => {
    if (score === null || !Number.isFinite(score)) return "-"
    const rounded = Math.round(score * 10) / 10
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)
  }

  const calculateScopeOverallScore = (
    scope: TechnicalEvaluationData["criteria"]["scopes"][number],
    contractorId: string
  ): number | null => {
    let weightedScoreTotal = 0
    let totalWeight = 0

    for (const breakdown of scope.breakdowns) {
      const scoreData = scores[contractorId]?.[breakdown.id]
      if (!scoreData) continue
      const weight = Number.isFinite(breakdown.weight) ? breakdown.weight : 0
      weightedScoreTotal += scoreData.score * weight
      totalWeight += weight
    }

    if (totalWeight <= 0) return null
    return weightedScoreTotal / totalWeight
  }

  const calculateVendorOverallScore = (contractorId: string): number | null => {
    let weightedScoreTotal = 0
    let totalWeight = 0

    for (const scope of criteria.scopes) {
      for (const breakdown of scope.breakdowns) {
        const scoreData = scores[contractorId]?.[breakdown.id]
        if (!scoreData) continue
        const weight = Number.isFinite(breakdown.weight) ? breakdown.weight : 0
        weightedScoreTotal += scoreData.score * weight
        totalWeight += weight
      }
    }

    if (totalWeight <= 0) return null
    return weightedScoreTotal / totalWeight
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">
              Technical Evaluation Scores
            </h2>
            <p className="text-sm text-muted-foreground">
              {approvedScores}/{totalScores} ({approvalPercent}%) scores
              approved
            </p>
          </div>
          <Button
            onClick={() => onOpenReview()}
            variant={isComplete ? "outline" : "default"}
          >
            {isComplete ? (
              <>
                <CheckCircle2 className="size-4 mr-2" />
                Edit Scores
              </>
            ) : (
              "Review Scores"
            )}
          </Button>
        </div>

        <div className="border rounded-lg overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[250px] py-4" />
                {activeContractors.map((contractor) => (
                  <TableHead
                    key={contractor.id}
                    className="text-center py-4 min-w-[190px]"
                  >
                    <div className="space-y-1">
                      <p className="text-base font-semibold leading-tight">
                        {contractor.name}
                      </p>
                      <p className="text-sm font-medium text-muted-foreground">
                        Overall: {formatScore(calculateVendorOverallScore(contractor.id))}
                      </p>
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {criteria.scopes.map((scope) => (
                <Fragment key={scope.id}>
                  <TableRow className="bg-muted/50">
                    <TableCell className="font-medium">{scope.name}</TableCell>
                    {activeContractors.map((contractor) => (
                      <TableCell
                        key={contractor.id}
                        className="text-center font-medium"
                      >
                        {formatScore(calculateScopeOverallScore(scope, contractor.id))}
                      </TableCell>
                    ))}
                  </TableRow>
                  {scope.breakdowns.map((breakdown) => (
                    <TableRow key={breakdown.id}>
                      <TableCell className="pl-6">
                        <div>
                          <p>{breakdown.title}</p>
                          <p className="text-xs text-muted-foreground">
                            {breakdown.weight}%
                          </p>
                        </div>
                      </TableCell>
                      {activeContractors.map((contractor) => {
                        const scoreData = scores[contractor.id]?.[breakdown.id]
                        return (
                          <TableCell
                            key={contractor.id}
                            className="text-center cursor-pointer hover:bg-accent"
                            onClick={() =>
                              onOpenReview(contractor.id, breakdown.id)
                            }
                          >
                            <div className="flex items-center justify-center gap-1">
                              <span>{scoreData?.score ?? "-"}</span>
                              {scoreData?.approved && (
                                <CheckCircle2 className="size-4 text-green-600" />
                              )}
                            </div>
                          </TableCell>
                        )
                      })}
                    </TableRow>
                  ))}
                </Fragment>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Review Sheet
// ============================================================================

function ReviewSheet({
  open,
  onOpenChange,
  evaluationId,
  evalData,
  contractors,
  initialCell,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  evaluationId: string
  evalData: TechnicalEvaluationData
  contractors: Array<{ id: string; name: string }>
  initialCell: { contractorId: string; breakdownId: string } | null
}) {
  const queryClient = useQueryClient()
  const { criteria, scores, proposalsUploaded } = evalData
  const latestEvalDataRef = useRef(evalData)
  const isPersistingRef = useRef(false)
  const queuedSaveRef = useRef<{
    scores: typeof scores
    status?: TechnicalEvaluationData["status"]
  } | null>(null)

  const activeContractors = contractors.filter((c) =>
    proposalsUploaded.includes(c.id)
  )

  // Build flat list of all score cells
  const allCells: Array<{
    scopeName: string
    breakdown: Breakdown
    contractor: { id: string; name: string }
  }> = []
  for (const scope of criteria.scopes) {
    for (const breakdown of scope.breakdowns) {
      for (const contractor of activeContractors) {
        allCells.push({ scopeName: scope.name, breakdown, contractor })
      }
    }
  }

  // Find first unapproved cell
  const findFirstUnapproved = () => {
    const idx = allCells.findIndex(
      (cell) => !scores[cell.contractor.id]?.[cell.breakdown.id]?.approved
    )
    return idx >= 0 ? idx : 0
  }

  // Find cell index by contractor and breakdown
  const findCellIndex = (contractorId: string, breakdownId: string) => {
    const idx = allCells.findIndex(
      (cell) =>
        cell.contractor.id === contractorId && cell.breakdown.id === breakdownId
    )
    return idx >= 0 ? idx : 0
  }

  const [currentIndex, setCurrentIndex] = useState(findFirstUnapproved)
  const [workingScores, setWorkingScores] = useState(scores)
  const [localScore, setLocalScore] = useState<number>(0)
  const [localComment, setLocalComment] = useState<string>("")
  const [localEvidence, setLocalEvidence] = useState<Evidence[]>([])
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    latestEvalDataRef.current = evalData
  }, [evalData])

  // Update currentIndex when sheet opens with initialCell
  useEffect(() => {
    if (open) {
      setWorkingScores(scores)
      if (initialCell) {
        setCurrentIndex(
          findCellIndex(initialCell.contractorId, initialCell.breakdownId)
        )
      } else {
        setCurrentIndex(findFirstUnapproved())
      }
    }
  }, [open, initialCell])

  // Update local state when current cell changes
  useEffect(() => {
    if (allCells[currentIndex]) {
      const cell = allCells[currentIndex]
      const scoreData = workingScores[cell.contractor.id]?.[cell.breakdown.id]
      setLocalScore(scoreData?.score ?? 0)
      setLocalComment(scoreData?.comment ?? "")
      setLocalEvidence(scoreData?.evidence ?? [])
    }
  }, [currentIndex, workingScores, allCells])

  const flushPersistQueue = async () => {
    if (isPersistingRef.current) return

    isPersistingRef.current = true
    setIsSaving(true)
    let wroteData = false

    while (queuedSaveRef.current) {
      const payload = queuedSaveRef.current
      queuedSaveRef.current = null

      try {
        const dataToSave: TechnicalEvaluationData = {
          ...(latestEvalDataRef.current as TechnicalEvaluationData),
          scores: payload.scores,
          ...(payload.status ? { status: payload.status } : {}),
        }

        await updateTechnicalEvaluationFn({
          data: {
            evaluationId,
            data: dataToSave,
          },
        })
        wroteData = true
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to save score")
      }
    }

    if (wroteData) {
      await queryClient.invalidateQueries({
        queryKey: ["technical-evaluation", evaluationId, "detail"],
      })
    }

    isPersistingRef.current = false
    setIsSaving(false)

    if (queuedSaveRef.current) {
      void flushPersistQueue()
    }
  }

  const queuePersist = (payload: {
    scores: typeof scores
    status?: TechnicalEvaluationData["status"]
  }) => {
    queuedSaveRef.current = payload
    void flushPersistQueue()
  }

  const currentCell = allCells[currentIndex]
  const totalCells = allCells.length
  const approvedCount = allCells.filter(
    (cell) => workingScores[cell.contractor.id]?.[cell.breakdown.id]?.approved
  ).length
  const reviewedPercent =
    totalCells > 0 ? Math.round((approvedCount / totalCells) * 100) : 0

  const handleApproveAndNext = async () => {
    if (!currentCell) return

    const nextScore = sanitizeScore(localScore)
    const newScores = { ...workingScores }
    if (!newScores[currentCell.contractor.id]) {
      newScores[currentCell.contractor.id] = {}
    }
    newScores[currentCell.contractor.id][currentCell.breakdown.id] = {
      score: nextScore,
      comment: localComment || undefined,
      approved: true,
      evidence: localEvidence,
    }

    setWorkingScores(newScores)

    // Check if all approved
    const allApproved = allCells.every((cell) => {
      if (
        cell.contractor.id === currentCell.contractor.id &&
        cell.breakdown.id === currentCell.breakdown.id
      ) {
        return true // Just approved this one
      }
      return newScores[cell.contractor.id]?.[cell.breakdown.id]?.approved
    })

    if (allApproved) {
      queuePersist({ scores: newScores, status: "review_complete" })
      toast.success("Review complete!")
      onOpenChange(false)
    } else {
      // Move to next unapproved
      const nextUnapproved = allCells.findIndex(
        (cell, idx) =>
          idx > currentIndex &&
          !newScores[cell.contractor.id]?.[cell.breakdown.id]?.approved
      )
      if (nextUnapproved >= 0) {
        setCurrentIndex(nextUnapproved)
      } else {
        // Wrap around
        const firstUnapproved = allCells.findIndex(
          (cell) =>
            !newScores[cell.contractor.id]?.[cell.breakdown.id]?.approved
        )
        if (firstUnapproved >= 0) {
          setCurrentIndex(firstUnapproved)
        }
      }
      queuePersist({ scores: newScores })
    }
  }

  const handleSaveAndNext = () => {
    if (!currentCell) return

    const nextScore = sanitizeScore(localScore)
    const newScores = { ...workingScores }
    if (!newScores[currentCell.contractor.id]) {
      newScores[currentCell.contractor.id] = {}
    }
    newScores[currentCell.contractor.id][currentCell.breakdown.id] = {
      score: nextScore,
      comment: localComment || undefined,
      approved:
        workingScores[currentCell.contractor.id]?.[currentCell.breakdown.id]
          ?.approved ?? false,
      evidence: localEvidence,
    }

    setWorkingScores(newScores)

    // Move to next
    if (currentIndex < allCells.length - 1) {
      setCurrentIndex(currentIndex + 1)
    }

    queuePersist({ scores: newScores })
  }

  if (!currentCell) return null

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-[80vw]! max-w-none! sm:max-w-none! p-0 flex flex-col gap-0 bg-background"
        showCloseButton
      >
        <SheetHeader className="shrink-0 border-b px-5 py-3 bg-background/95 backdrop-blur sticky top-0 z-10">
          <SheetTitle className="text-base font-semibold tracking-tight">
            Review Scores ({approvedCount}/{totalCells} scores reviewed,{" "}
            {reviewedPercent}%)
          </SheetTitle>
        </SheetHeader>

        <div className="flex flex-1 min-h-0">
          <div className="w-2/5 min-h-0 overflow-y-auto border-r bg-muted/20 px-4 py-3 space-y-2">
            {criteria.scopes.map((scope) => (
              <div key={scope.id} className="space-y-1 rounded-md border bg-card p-2">
                <p className="text-sm font-bold uppercase tracking-wide text-foreground">
                  {scope.name} (
                  {scope.breakdowns.reduce(
                    (total, breakdown) => total + breakdown.weight,
                    0
                  )}
                  %)
                </p>
                {scope.breakdowns.map((breakdown) => (
                  <div key={breakdown.id} className="ml-2 space-y-0.5">
                    <p className="text-xs font-medium text-muted-foreground">
                      {breakdown.title} ({breakdown.weight}%)
                    </p>
                    <div className="ml-2 space-y-0.5">
                      {activeContractors.map((contractor) => {
                        const cellIndex = findCellIndex(contractor.id, breakdown.id)
                        const isSelected = currentIndex === cellIndex
                        const scoreData = workingScores[contractor.id]?.[breakdown.id]

                        return (
                          <button
                            key={contractor.id}
                            type="button"
                            className={cn(
                              "w-full flex items-center justify-between rounded-md border border-transparent px-2 py-1 text-left text-xs transition-colors hover:bg-accent/70",
                              isSelected &&
                                "border-primary/30 bg-primary/10 font-medium text-foreground"
                            )}
                            onClick={() => setCurrentIndex(cellIndex)}
                          >
                            <span className="truncate pr-2">{contractor.name}</span>
                            <span className="inline-flex items-center gap-1 text-xs">
                              <span>{scoreData?.score ?? "-"}</span>
                              {scoreData?.approved ? (
                                <CheckCircle2 className="size-3 text-green-600" />
                              ) : (
                                <Circle className="size-3 text-muted-foreground/40" />
                              )}
                            </span>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div className="w-3/5 min-h-0 overflow-y-auto px-5 py-4">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold tracking-tight">
                {currentCell.contractor.name}: {currentCell.breakdown.title}
              </h3>

              <div>
                <label className="text-sm font-medium">Score (0-100)</label>
                <Input
                  type="number"
                  min={0}
                  max={100}
                  value={localScore}
                  onChange={(e) => {
                    if (e.target.value === "") {
                      setLocalScore(0)
                      return
                    }
                    setLocalScore(sanitizeScore(Number(e.target.value)))
                  }}
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-sm font-medium">
                  Comment (optional)
                </label>
                <Textarea
                  value={localComment}
                  onChange={(e) => setLocalComment(e.target.value)}
                  placeholder="Add notes about this score..."
                  className="mt-1"
                  rows={3}
                />
              </div>

              <EvidenceSection
                evidence={localEvidence}
                onChange={setLocalEvidence}
              />
            </div>
          </div>
        </div>

        <div className="shrink-0 border-t px-5 py-3 bg-background/95 backdrop-blur sticky bottom-0 z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>
                Score {currentIndex + 1} of {totalCells}
              </span>
              {isSaving && (
                <span className="inline-flex items-center gap-1">
                  <Spinner className="size-3" />
                  Saving...
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleSaveAndNext}>
                Save and Next
              </Button>
              <Button onClick={handleApproveAndNext}>
                <Check className="size-4 mr-1" />
                Approve and Next
              </Button>
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
