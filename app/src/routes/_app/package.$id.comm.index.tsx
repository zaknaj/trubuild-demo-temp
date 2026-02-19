import { useMemo } from "react"
import { createFileRoute } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import {
  packageDetailQueryOptions,
  packageCommercialSummaryQueryOptions,
} from "@/lib/query-options"
import { FileText, Plus, BarChart3, Trophy, Users, Building2 } from "lucide-react"
import { cn, formatCurrency } from "@/lib/utils"

export const Route = createFileRoute("/_app/package/$id/comm/")({
  loader: ({ params, context }) => {
    context.queryClient.prefetchQuery(
      packageCommercialSummaryQueryOptions(params.id)
    )
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const { data: packageData } = useSuspenseQuery(packageDetailQueryOptions(id))
  const { data: summaryData } = useSuspenseQuery(
    packageCommercialSummaryQueryOptions(id)
  )

  const assets = packageData?.assets ?? []

  const getLegacyContractors = (
    evaluation: unknown
  ): Array<{ contractorId: string; contractorName: string; totalAmount: number }> => {
    if (!evaluation || typeof evaluation !== "object") return []

    const contractors = (evaluation as { contractors?: unknown }).contractors
    if (!Array.isArray(contractors)) return []

    return contractors
      .map((contractor) => {
        if (!contractor || typeof contractor !== "object") return null
        const candidate = contractor as {
          contractorId?: unknown
          contractorName?: unknown
          totalAmount?: unknown
        }
        if (
          typeof candidate.contractorId !== "string" ||
          typeof candidate.totalAmount !== "number"
        ) {
          return null
        }
        return {
          contractorId: candidate.contractorId,
          contractorName:
            typeof candidate.contractorName === "string"
              ? candidate.contractorName
              : candidate.contractorId,
          totalAmount: candidate.totalAmount,
        }
      })
      .filter((contractor): contractor is NonNullable<typeof contractor> =>
        Boolean(contractor)
      )
  }

  const getReportVendorTotals = (evaluation: unknown): Map<string, number> => {
    if (!evaluation || typeof evaluation !== "object") return new Map()

    const vendors = (evaluation as { vendors?: unknown }).vendors
    const summary = (evaluation as { summary?: unknown }).summary
    if (!Array.isArray(vendors) || !summary || typeof summary !== "object") {
      return new Map()
    }

    const vendorTotals = (summary as { vendor_totals?: unknown }).vendor_totals
    if (!vendorTotals || typeof vendorTotals !== "object") return new Map()

    const totals = new Map<string, number>()
    for (const vendor of vendors) {
      if (typeof vendor !== "string") continue
      const amount = (vendorTotals as Record<string, unknown>)[vendor]
      if (typeof amount !== "number") continue
      totals.set(vendor, amount)
    }
    return totals
  }

  // Aggregate contractor data across all assets
  const { sortedContractors, assetBids, hasAnyEvaluation } = useMemo(() => {
    const contractorTotals = new Map<
      string,
      { id: string; name: string; total: number }
    >()
    const assetBidsMap = new Map<string, Map<string, number>>()

    let hasEval = false

    for (const asset of summaryData.assets) {
      if (!asset.evaluation) continue
      const bidsForAsset = new Map<string, number>()

      const legacyContractors = getLegacyContractors(asset.evaluation)
      if (legacyContractors.length > 0) {
        for (const contractor of legacyContractors) {
          bidsForAsset.set(contractor.contractorId, contractor.totalAmount)

          const existing = contractorTotals.get(contractor.contractorId)
          if (existing) {
            existing.total += contractor.totalAmount
          } else {
            contractorTotals.set(contractor.contractorId, {
              id: contractor.contractorId,
              name: contractor.contractorName,
              total: contractor.totalAmount,
            })
          }
        }
      } else {
        const reportVendorTotals = getReportVendorTotals(asset.evaluation)
        for (const [vendorName, totalAmount] of reportVendorTotals) {
          bidsForAsset.set(vendorName, totalAmount)

          const existing = contractorTotals.get(vendorName)
          if (existing) {
            existing.total += totalAmount
          } else {
            contractorTotals.set(vendorName, {
              id: vendorName,
              name: vendorName,
              total: totalAmount,
            })
          }
        }
      }

      if (bidsForAsset.size === 0) continue
      hasEval = true
      assetBidsMap.set(asset.id, bidsForAsset)
    }

    // Sort contractors by total (lowest first)
    const sorted = Array.from(contractorTotals.values()).sort(
      (a, b) => a.total - b.total
    )

    return {
      sortedContractors: sorted,
      assetBids: assetBidsMap,
      hasAnyEvaluation: hasEval,
    }
  }, [summaryData.assets])

  // Find lowest bid for each asset row
  const getLowestBidForAsset = (assetId: string): string | null => {
    const bids = assetBids.get(assetId)
    if (!bids || bids.size === 0) return null

    let lowestId: string | null = null
    let lowestAmount = Infinity

    for (const [contractorId, amount] of bids) {
      if (amount < lowestAmount) {
        lowestAmount = amount
        lowestId = contractorId
      }
    }

    return lowestId
  }

  if (assets.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-12 text-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <FileText className="size-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No assets yet</h3>
        <p className="text-muted-foreground mb-6 max-w-sm">
          Create an asset to start running commercial evaluations for this
          package.
        </p>
        <p className="text-sm text-muted-foreground">
          Use the <Plus className="inline size-3" /> <strong>New Asset</strong>{" "}
          button in the sidebar to get started.
        </p>
      </div>
    )
  }

  if (!hasAnyEvaluation) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-12 text-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <BarChart3 className="size-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No evaluations yet</h3>
        <p className="text-muted-foreground mb-6 max-w-sm">
          Run commercial evaluations on individual assets to see the package
          summary.
        </p>
      </div>
    )
  }

  // The lowest bidder is always index 0 (sorted by total)
  const lowestBidderId = sortedContractors[0]?.id
  const packageCurrency = packageData.package.currency

  const contractorRankById = useMemo(
    () =>
      new Map(
        sortedContractors.map((contractor, index) => [contractor.id, index + 1])
      ),
    [sortedContractors]
  )

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-6 bg-background sticky top-0 z-10 border-b">
        <div className="max-w-6xl mx-auto space-y-4">
          <div>
            <h2 className="text-base font-semibold">Commercial Summary</h2>
            <p className="text-xs text-muted-foreground mt-1">
              Side-by-side contractor comparison across all assets.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border bg-card px-3 py-2.5 flex items-center gap-2.5">
              <div className="p-1.5 rounded-md bg-muted">
                <Building2 className="size-3.5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground">Assets Compared</p>
                <p className="text-sm font-semibold">{summaryData.assets.length}</p>
              </div>
            </div>

            <div className="rounded-lg border bg-card px-3 py-2.5 flex items-center gap-2.5">
              <div className="p-1.5 rounded-md bg-muted">
                <Users className="size-3.5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground">Contractors</p>
                <p className="text-sm font-semibold">{sortedContractors.length}</p>
              </div>
            </div>

            <div className="rounded-lg border bg-emerald-50/60 dark:bg-emerald-950/20 px-3 py-2.5 flex items-center gap-2.5">
              <div className="p-1.5 rounded-md bg-emerald-100 dark:bg-emerald-900/40">
                <Trophy className="size-3.5 text-emerald-700 dark:text-emerald-400" />
              </div>
              <div>
                <p className="text-[11px] text-emerald-700/80 dark:text-emerald-400/80">
                  Lowest Total
                </p>
                <p className="text-sm font-semibold text-emerald-700 dark:text-emerald-400">
                  {sortedContractors[0]
                    ? formatCurrency(sortedContractors[0].total, packageCurrency)
                    : "—"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto px-4 py-4">
        <div className="w-full max-w-6xl mx-auto rounded-xl border bg-card shadow-sm overflow-hidden">
          <div className="px-3 py-2.5 border-b bg-muted/20">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <p className="text-xs font-medium text-muted-foreground">
                Contractor Ranking (by package total)
              </p>
              <div className="flex items-center gap-1.5 flex-wrap">
                {sortedContractors.slice(0, 5).map((contractor) => {
                  const isLowest = contractor.id === lowestBidderId
                  const rank = contractorRankById.get(contractor.id) ?? 0
                  return (
                    <span
                      key={contractor.id}
                      className={cn(
                        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px]",
                        isLowest
                          ? "border-emerald-300 bg-emerald-100/70 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-400"
                          : "border-border bg-background text-muted-foreground"
                      )}
                    >
                      {isLowest ? "Lowest" : `#${rank}`} {contractor.name}
                    </span>
                  )
                })}
              </div>
            </div>
          </div>
          <div className="overflow-auto p-2">
            <table className="w-full min-w-[720px] border-collapse text-xs">
              <thead className="sticky top-0 bg-background/95 backdrop-blur-sm z-10">
                <tr>
                  <th className="text-left px-2 py-1.5 border-b font-medium text-muted-foreground min-w-[160px] sticky left-0 bg-background/95 backdrop-blur-sm z-20">
                    Asset
                  </th>
                  {sortedContractors.map((contractor, idx) => {
                    const isLowest = contractor.id === lowestBidderId
                    return (
                      <th
                        key={contractor.id}
                        className={cn(
                          "text-right px-2 py-1.5 border-b font-medium min-w-[100px]",
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
                            {contractor.name}
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
                {summaryData.assets.map((asset, rowIndex) => {
                  const bids = assetBids.get(asset.id)
                  const lowestBidIdForAsset = getLowestBidForAsset(asset.id)
                  const hasEvaluation = asset.evaluation !== null
                  const winnerName = sortedContractors.find(
                    (contractor) => contractor.id === lowestBidIdForAsset
                  )?.name

                  return (
                    <tr
                      key={asset.id}
                      className={cn(
                        "hover:bg-accent/30 transition-colors",
                        rowIndex % 2 === 1 && "bg-muted/10"
                      )}
                    >
                      <td className="px-2 py-1.5 border-b font-medium sticky left-0 bg-card text-xs">
                        <div className="flex flex-col gap-0.5">
                          <span>{asset.name}</span>
                          {winnerName && (
                            <span className="text-[10px] text-muted-foreground">
                              Lowest: {winnerName}
                            </span>
                          )}
                        </div>
                      </td>
                      {sortedContractors.map((contractor) => {
                        const bid = bids?.get(contractor.id)
                        const isLowestForAsset =
                          contractor.id === lowestBidIdForAsset
                        const isLowestOverall = contractor.id === lowestBidderId

                        return (
                          <td
                            key={contractor.id}
                            className={cn(
                              "px-2 py-1.5 border-b text-right tabular-nums",
                              isLowestOverall &&
                                "bg-emerald-50/80 dark:bg-emerald-950/30",
                              isLowestForAsset &&
                                "text-emerald-700 dark:text-emerald-400 font-medium"
                            )}
                          >
                            {hasEvaluation && bid !== undefined
                              ? formatCurrency(bid, packageCurrency)
                              : "—"}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}

                {/* Total Row */}
                <tr className="bg-accent/20 font-bold">
                  <td className="px-2 py-2 border-t-2 sticky left-0 bg-accent/20 text-xs">
                    Total
                  </td>
                  {sortedContractors.map((contractor) => {
                    const isLowest = contractor.id === lowestBidderId
                    return (
                      <td
                        key={contractor.id}
                        className={cn(
                          "px-2 py-2 border-t-2 text-right tabular-nums text-xs",
                          isLowest &&
                            "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-400"
                        )}
                      >
                        {formatCurrency(contractor.total, packageCurrency)}
                      </td>
                    )
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
