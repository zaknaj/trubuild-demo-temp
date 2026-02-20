import { createFileRoute } from "@tanstack/react-router"
import {
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import { useEffect, useMemo, useState } from "react"
import { commercialEvaluationsQueryOptions } from "@/lib/query-options"
import { updateCommercialEvaluationPTCsFn } from "@/fn/evaluations"
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
import { Spinner } from "@/components/ui/spinner"
import type {
  CommercialEvaluationData,
  ContractorPTCs,
  PTCCategory,
  PTCStatus,
} from "@/lib/types"
import { toast } from "sonner"
import {
  buildCommercialPtcStats,
  filterCommercialPtcRows,
  flattenCommercialPtcs,
  normalizeCommercialEvaluationPtcs,
} from "@/lib/comm-ptc"

type CommercialEvaluation = {
  id: string
  assetId: string
  roundNumber: number
  roundName: string
  data: CommercialEvaluationData | Record<string, never>
  createdAt: Date
  updatedAt: Date
}

export const Route = createFileRoute("/_app/package/$id/comm/$assetId/ptc")({
  component: RouteComponent,
})

function RouteComponent() {
  const { assetId } = Route.useParams()
  const queryClient = useQueryClient()

  const { data: evaluations } = useQuery(
    commercialEvaluationsQueryOptions(assetId)
  )

  if (!evaluations) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const evaluationsList = evaluations as CommercialEvaluation[]

  // Get round from Zustand store
  const selectedRoundId = useStore((s) => s.selectedCommRound[assetId])

  // Get current round
  const currentRound = selectedRoundId
    ? evaluationsList.find((e) => e.id === selectedRoundId)
    : evaluationsList[0]

  const evalData = currentRound?.data as CommercialEvaluationData | undefined
  const persistedPtcs = useMemo(
    () => normalizeCommercialEvaluationPtcs(evalData?.ptcs),
    [evalData?.ptcs]
  )
  const [localPtcs, setLocalPtcs] = useState<ContractorPTCs[]>([])
  const [selectedContractorId, setSelectedContractorId] = useState<"all" | string>(
    "all"
  )
  const [selectedCategory, setSelectedCategory] = useState<"all" | PTCCategory>("all")
  const [selectedStatus, setSelectedStatus] = useState<"all" | PTCStatus>("all")
  const [search, setSearch] = useState("")

  const persistedSerialized = useMemo(
    () => JSON.stringify(persistedPtcs),
    [persistedPtcs]
  )
  const localSerialized = useMemo(() => JSON.stringify(localPtcs), [localPtcs])
  const isDirty = localSerialized !== persistedSerialized

  useEffect(() => {
    if (!currentRound) return
    setLocalPtcs(persistedPtcs)
  }, [currentRound?.id])

  useEffect(() => {
    if (!isDirty) {
      setLocalPtcs(persistedPtcs)
    }
  }, [isDirty, persistedPtcs])

  const allRows = useMemo(() => flattenCommercialPtcs(localPtcs), [localPtcs])
  const filteredRows = useMemo(
    () =>
      filterCommercialPtcRows(allRows, {
        contractorId: selectedContractorId,
        category: selectedCategory,
        status: selectedStatus,
        search,
      }),
    [allRows, search, selectedCategory, selectedContractorId, selectedStatus]
  )
  const overallStats = useMemo(() => buildCommercialPtcStats(allRows), [allRows])
  const filteredStats = useMemo(
    () => buildCommercialPtcStats(filteredRows),
    [filteredRows]
  )

  const updatePTCsMutation = useMutation({
    mutationFn: (updatedPTCs: ContractorPTCs[]) =>
      updateCommercialEvaluationPTCsFn({
        data: {
          evaluationId: currentRound!.id,
          ptcs: updatedPTCs,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
      })
      toast.success("PTCs saved successfully")
    },
    onError: () => {
      toast.error("Failed to save PTCs")
    },
  })

  useEffect(() => {
    if (selectedContractorId === "all") return
    const hasContractor = localPtcs.some(
      (contractor) => contractor.contractorId === selectedContractorId
    )
    if (!hasContractor) {
      setSelectedContractorId("all")
    }
  }, [localPtcs, selectedContractorId])

  if (!currentRound) {
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

  if (localPtcs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-12 text-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <FileQuestion className="size-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No PTCs Available</h3>
        <p className="text-muted-foreground max-w-sm">
          PTCs will be generated when the commercial evaluation analysis is run.
        </p>
      </div>
    )
  }

  const updateRow = (
    contractorId: string,
    rowIndex: number,
    updates: {
      status?: PTCStatus
      vendorResponse?: string
    }
  ) => {
    setLocalPtcs((prev) =>
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

  const categoryOptions: Array<{ value: PTCCategory; label: string }> = [
    { value: "exclusions", label: "Exclusions" },
    { value: "deviations", label: "Deviations" },
    { value: "pricing_anomalies", label: "Pricing Anomalies" },
    { value: "arithmetic_checks", label: "Arithmetic Checks" },
  ]
  const statusOptions: Array<{ value: PTCStatus; label: string }> = [
    { value: "pending", label: "Pending" },
    { value: "closed", label: "Closed" },
  ]
  const contractorOptions = overallStats.byContractor

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
                updatePTCsMutation.mutate(normalizeCommercialEvaluationPtcs(localPtcs))
              }
              disabled={updatePTCsMutation.isPending}
            >
              {updatePTCsMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6 overflow-y-auto">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <InsightCard label="Total PTCs" value={overallStats.total} />
          <InsightCard label="Pending" value={overallStats.pending} />
          <InsightCard label="Closed" value={overallStats.closed} />
          <InsightCard label="Completion" value={`${overallStats.completionPercent}%`} />
          <InsightCard label="Filtered" value={filteredRows.length} />
          <InsightCard
            label="Filtered Completion"
            value={`${filteredStats.completionPercent}%`}
          />
        </div>

        <div className="grid md:grid-cols-3 gap-3">
          <div className="border rounded-lg p-3 space-y-2">
            <p className="text-xs text-muted-foreground">By Category</p>
            {categoryOptions.map((option) => (
              <div key={option.value} className="flex items-center justify-between text-sm">
                <span>{option.label}</span>
                <Badge variant="outline">
                  {overallStats.byCategory[option.value]}
                </Badge>
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
            value={selectedCategory}
            onValueChange={(value) => setSelectedCategory(value as "all" | PTCCategory)}
          >
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="Filter category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All categories</SelectItem>
              {categoryOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={selectedStatus}
            onValueChange={(value) => setSelectedStatus(value as "all" | PTCStatus)}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              {statusOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Input
            className="w-[320px]"
            placeholder="Search section, query, contractor, response..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>

        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Contractor</TableHead>
                <TableHead className="w-[220px]">Reference Section</TableHead>
                <TableHead className="w-[460px]">Query Description</TableHead>
                <TableHead className="w-[160px]">Category</TableHead>
                <TableHead className="w-[260px]">Vendor Response</TableHead>
                <TableHead className="w-[120px]">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredRows.map((row) => (
                <TableRow key={row.rowId}>
                  <TableCell>{row.contractorName}</TableCell>
                  <TableCell className="whitespace-normal">
                    {row.referenceSection || "General"}
                  </TableCell>
                  <TableCell className="whitespace-normal">{row.queryDescription}</TableCell>
                  <TableCell>
                    <Badge variant="outline">
                      {categoryOptions.find((c) => c.value === row.category)?.label ??
                        row.category}
                    </Badge>
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
