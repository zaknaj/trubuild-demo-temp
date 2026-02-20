import { useMemo } from "react"
import { createFileRoute } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import {
  packageDetailQueryOptions,
  packageCommercialSummaryQueryOptions,
} from "@/lib/query-options"
import { BarChart3, FileText } from "lucide-react"
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
      <div className="h-full flex items-center justify-center px-6">
        <div className="w-full max-w-[780px] rounded-xl border border-black/10 bg-white p-10 text-center">
          <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-full bg-[#F3F3F3]">
            <FileText className="size-7 text-black/45" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No assets yet</h3>
          <p className="text-black/55 max-w-md mx-auto">
            Create a new asset from the menu to start running commercial analysis
            for this package.
          </p>
        </div>
      </div>
    )
  }

  if (!hasAnyEvaluation) {
    return (
      <div className="h-full flex items-center justify-center px-6">
        <div className="w-full max-w-[780px] rounded-xl border border-black/10 bg-white p-10 text-center">
          <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-full bg-[#F3F3F3]">
            <BarChart3 className="size-7 text-black/45" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No evaluations yet</h3>
          <p className="text-black/55 max-w-md mx-auto">
            Run commercial analysis for at least one asset to populate this
            summary.
          </p>
        </div>
      </div>
    )
  }

  // The lowest bidder is always index 0 (sorted by total)
  const lowestBidderId = sortedContractors[0]?.id
  const packageCurrency = packageData.package.currency

  return (
    <div className="h-full px-6 pb-6">
      <div className="h-full flex flex-col">
        <div className="pt-6 pb-4">
          <h2 className="text-[28px] leading-8 font-semibold text-black">Summary</h2>
        </div>

        <div className="flex-1 min-h-0 rounded-xl border border-black/10 bg-white overflow-hidden">
          <div className="h-full overflow-auto p-5">
            <table className="w-full min-w-[760px] border-collapse text-xs">
              <thead className="sticky top-0 z-10 bg-white">
                <tr>
                  <th className="text-left px-3 py-2 border-b border-black/10 font-medium text-black/50 min-w-[180px] sticky left-0 bg-white z-20">
                    Asset
                  </th>
                  {sortedContractors.map((contractor, idx) => {
                    const isLowest = contractor.id === lowestBidderId
                    return (
                      <th
                        key={contractor.id}
                        className={cn(
                          "text-right px-3 py-2 border-b border-black/10 font-medium min-w-[120px]",
                          isLowest && "bg-[#F4FAF4]"
                        )}
                      >
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={cn(
                              "truncate max-w-[120px] text-[12px]",
                              isLowest && "text-[#4A8A4D]"
                            )}
                          >
                            {contractor.name}
                          </span>
                          <span
                            className={cn(
                              "text-[10px] font-normal",
                              isLowest ? "text-[#4A8A4D]" : "text-black/40"
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
                        "transition-colors hover:bg-[#FAFAFA]",
                        rowIndex % 2 === 1 && "bg-[#FCFCFC]"
                      )}
                    >
                      <td className="px-3 py-2 border-b border-black/5 font-medium sticky left-0 bg-white text-xs">
                        <div className="flex flex-col gap-0.5">
                          <span>{asset.name}</span>
                          {winnerName && (
                            <span className="text-[10px] text-black/40">
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
                              "px-3 py-2 border-b border-black/5 text-right tabular-nums",
                              isLowestOverall && "bg-[#F4FAF4]",
                              isLowestForAsset && "text-[#4A8A4D] font-medium"
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

                <tr className="bg-[#F8F8F8] font-bold">
                  <td className="px-3 py-2 border-t-2 border-black/10 sticky left-0 bg-[#F8F8F8] text-xs">
                    Total
                  </td>
                  {sortedContractors.map((contractor) => {
                    const isLowest = contractor.id === lowestBidderId
                    return (
                      <td
                        key={contractor.id}
                        className={cn(
                          "px-3 py-2 border-t-2 border-black/10 text-right tabular-nums text-xs",
                          isLowest && "bg-[#EAF7EA] text-[#3D7D40]"
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
