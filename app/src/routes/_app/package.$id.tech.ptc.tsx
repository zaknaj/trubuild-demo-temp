import { createFileRoute } from "@tanstack/react-router"
import {
  useSuspenseQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import { useEffect, useMemo, useState } from "react"
import { technicalEvaluationDetailQueryOptions } from "@/lib/query-options"
import { updateTechnicalEvaluationPTCsFn } from "@/fn/evaluations"
import useStore from "@/lib/store"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { FileQuestion } from "lucide-react"
import type { TechnicalEvaluationData } from "@/lib/types"
import type {
  TechRfpPtcRefType,
  TechRfpTechnicalEvaluationPtcStatus,
  TechRfpTechnicalEvaluationContractorPtcs,
} from "@/lib/tech_rfp"
import {
  buildTechnicalPtcStats,
  filterTechnicalPtcRows,
  flattenTechnicalPtcs,
  normalizeTechnicalEvaluationPtcs,
} from "@/lib/technical-ptc"
import { toast } from "sonner"

export const Route = createFileRoute("/_app/package/$id/tech/ptc")({
  component: RouteComponent,
})

function RouteComponent() {
  const { id: packageId } = Route.useParams()

  // Get round from Zustand store
  const evaluationId = useStore((s) => s.selectedTechRound[packageId])

  if (!evaluationId) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-12 text-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <FileQuestion className="size-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No Evaluation Selected</h3>
        <p className="text-muted-foreground max-w-sm">
          Select or create an evaluation round to view PTC insights.
        </p>
      </div>
    )
  }

  return <PTCContent evaluationId={evaluationId} />
}

function PTCContent({ evaluationId }: { evaluationId: string }) {
  const queryClient = useQueryClient()

  const { data: evaluation } = useSuspenseQuery(
    technicalEvaluationDetailQueryOptions(evaluationId)
  ) as { data: { data?: unknown } }

  const evalData = (evaluation.data ?? {}) as Partial<TechnicalEvaluationData>
  const persistedTechnicalPtcs = useMemo(
    () => normalizeTechnicalEvaluationPtcs(evalData.ptcs),
    [evalData.ptcs]
  )
  const [localTechnicalPtcs, setLocalTechnicalPtcs] = useState<
    TechRfpTechnicalEvaluationContractorPtcs[]
  >([])
  const [selectedContractorId, setSelectedContractorId] = useState<"all" | string>(
    "all"
  )
  const [selectedRefType, setSelectedRefType] = useState<"all" | TechRfpPtcRefType>(
    "all"
  )
  const [search, setSearch] = useState("")
  const persistedSerialized = useMemo(
    () => JSON.stringify(persistedTechnicalPtcs),
    [persistedTechnicalPtcs]
  )
  const localSerialized = useMemo(
    () => JSON.stringify(localTechnicalPtcs),
    [localTechnicalPtcs]
  )
  const isDirty = localSerialized !== persistedSerialized

  useEffect(() => {
    setLocalTechnicalPtcs(persistedTechnicalPtcs)
  }, [evaluationId])

  useEffect(() => {
    // Keep unsaved local edits if background refetches happen.
    if (!isDirty) {
      setLocalTechnicalPtcs(persistedTechnicalPtcs)
    }
  }, [isDirty, persistedTechnicalPtcs])

  const allRows = useMemo(
    () => flattenTechnicalPtcs(localTechnicalPtcs),
    [localTechnicalPtcs]
  )
  const filteredRows = useMemo(
    () =>
      filterTechnicalPtcRows(allRows, {
        contractorId: selectedContractorId,
        refType: selectedRefType,
        search,
      }),
    [allRows, search, selectedContractorId, selectedRefType]
  )
  const overallStats = useMemo(() => buildTechnicalPtcStats(allRows), [allRows])
  const filteredStats = useMemo(
    () => buildTechnicalPtcStats(filteredRows),
    [filteredRows]
  )

  const updatePTCsMutation = useMutation({
    mutationFn: (updatedPTCs: TechRfpTechnicalEvaluationContractorPtcs[]) =>
      updateTechnicalEvaluationPTCsFn({
        data: {
          evaluationId,
          ptcs: updatedPTCs,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["technical-evaluation", evaluationId, "detail"],
      })
      toast.success("PTCs saved successfully")
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to save PTCs")
    },
  })

  useEffect(() => {
    if (selectedContractorId === "all") return
    const hasContractor = localTechnicalPtcs.some(
      (contractor) => contractor.contractorId === selectedContractorId
    )
    if (!hasContractor) {
      setSelectedContractorId("all")
    }
  }, [localTechnicalPtcs, selectedContractorId])

  if (localTechnicalPtcs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-12 text-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <FileQuestion className="size-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No PTCs Available</h3>
        <p className="text-muted-foreground max-w-sm">
          PTCs will be generated when the technical evaluation analysis is run.
        </p>
      </div>
    )
  }

  const updateRow = (
    contractorId: string,
    rowIndex: number,
    updates: {
      status?: TechRfpTechnicalEvaluationPtcStatus
      vendorResponse?: string
    }
  ) => {
    setLocalTechnicalPtcs((prev) =>
      prev.map((contractor) => {
        if (contractor.contractorId !== contractorId) return contractor
        return {
          ...contractor,
          ptcs: contractor.ptcs.map((ptc, idx) =>
            idx === rowIndex ? { ...ptc, ...updates } : ptc
          ),
        }
      })
    )
  }

  const contractorOptions = overallStats.byContractor
  const refTypeOptions: Array<{ value: TechRfpPtcRefType; label: string }> = [
    { value: "MISSING_INFO", label: "Missing Info" },
    { value: "INCOMPLETE", label: "Incomplete" },
    { value: "N/A", label: "N/A" },
  ]

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-6 border-b bg-background/95 sticky top-0 z-10">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">PTC Insights</h2>
            <p className="text-sm text-muted-foreground">
              {filteredRows.length} of {overallStats.total} items shown
            </p>
          </div>
          {isDirty && (
            <Button
              onClick={() =>
                updatePTCsMutation.mutate(
                  normalizeTechnicalEvaluationPtcs(localTechnicalPtcs)
                )
              }
              disabled={updatePTCsMutation.isPending}
            >
              {updatePTCsMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6 overflow-y-auto">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <InsightCard label="Total PTCs" value={overallStats.total} />
          <InsightCard label="Pending" value={overallStats.pending} />
          <InsightCard label="Closed" value={overallStats.closed} />
          <InsightCard label="Completion" value={`${overallStats.completionPercent}%`} />
          <InsightCard label="Filtered" value={filteredRows.length} />
        </div>

        <div className="grid md:grid-cols-3 gap-3">
          <div className="border rounded-lg p-3 space-y-2">
            <p className="text-xs text-muted-foreground">By Ref Type</p>
            {refTypeOptions.map((option) => (
              <div key={option.value} className="flex items-center justify-between text-sm">
                <span>{option.label}</span>
                <Badge variant="outline">{overallStats.byRefType[option.value]}</Badge>
              </div>
            ))}
          </div>
          <div className="border rounded-lg p-3 space-y-2 md:col-span-2">
            <p className="text-xs text-muted-foreground">By Contractor</p>
            <div className="grid md:grid-cols-2 gap-2">
              {contractorOptions.map((contractor) => (
                <div
                  key={contractor.contractorId}
                  className="flex items-center justify-between text-sm border rounded px-2 py-1.5"
                >
                  <span className="truncate">{contractor.contractorName}</span>
                  <span className="text-muted-foreground">
                    {contractor.pending}/{contractor.total} pending
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Select
            value={selectedContractorId}
            onValueChange={(value) => setSelectedContractorId(value as "all" | string)}
          >
            <SelectTrigger className="w-[260px]">
              <SelectValue placeholder="Filter contractor" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All contractors</SelectItem>
              {contractorOptions.map((contractor) => (
                <SelectItem key={contractor.contractorId} value={contractor.contractorId}>
                  {contractor.contractorName}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={selectedRefType}
            onValueChange={(value) =>
              setSelectedRefType(value as "all" | TechRfpPtcRefType)
            }
          >
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="Filter ref type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All ref types</SelectItem>
              {refTypeOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Input
            className="w-[320px]"
            placeholder="Search criterion, query, contractor, response..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>

        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Contractor</TableHead>
                <TableHead className="w-[240px]">Criterion</TableHead>
                <TableHead className="w-[460px]">Query Description</TableHead>
                <TableHead className="w-[140px]">Ref Type</TableHead>
                <TableHead className="w-[220px]">Vendor Response</TableHead>
                <TableHead className="w-[120px]">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredRows.map((row) => (
                <TableRow key={row.rowId}>
                  <TableCell>{row.contractorName}</TableCell>
                  <TableCell className="whitespace-normal">{row.criterion}</TableCell>
                  <TableCell className="whitespace-normal">{row.queryDescription}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{row.refType}</Badge>
                  </TableCell>
                  <TableCell>
                    <Input
                      value={row.vendorResponse}
                      placeholder="Enter vendor response..."
                      onChange={(event) =>
                        updateRow(row.contractorId, row.index, {
                          vendorResponse: event.target.value,
                        })
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <Button
                      variant={row.status === "pending" ? "secondary" : "outline"}
                      size="sm"
                      onClick={() =>
                        updateRow(row.contractorId, row.index, {
                          status: row.status === "pending" ? "closed" : "pending",
                        })
                      }
                    >
                      {row.status === "pending" ? "Pending" : "Closed"}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {filteredRows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="h-24 text-center text-muted-foreground">
                    No PTC rows match the current filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        {overallStats.total > 0 && (
          <p className="text-xs text-muted-foreground">
            Filtered completion: {filteredStats.completionPercent}% ({filteredStats.closed}/
            {filteredStats.total})
          </p>
        )}
      </div>
    </div>
  )
}

function InsightCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="border rounded-lg px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-semibold">{value}</p>
    </div>
  )
}
