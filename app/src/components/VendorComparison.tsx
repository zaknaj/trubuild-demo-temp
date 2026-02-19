import { useState, useMemo, createContext, useContext, useCallback } from "react"
import { Input } from "@/components/ui/input"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import {
  Search, ChevronRight, ChevronDown, AlertTriangle,
  ArrowDownRight, ArrowUpRight, BarChart3, Layers,
  TrendingDown, TrendingUp, CircleDot, Hash, DollarSign,
  Filter, Check, X, Users, Activity, Minus, Trophy, Medal,
  Award, Shield,
} from "lucide-react"
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Popover, PopoverContent, PopoverTrigger,
} from "@/components/ui/popover"
import {
  Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem,
} from "@/components/ui/command"
import { cn } from "@/lib/utils"

// ============================================================================
// Types
// ============================================================================

interface LineItemComparison {
  item_id: string
  item_description: string
  item_quantity: number | null
  item_unit: string | null
  sub_grouping_name: string | null
  vendor_rates: Record<string, number | null>
  vendor_amounts: Record<string, number | null>
  pte_rate?: number | null
  pte_amount?: number | null
  lowest_bidder: string | null
  highest_bidder: string | null
  lowest_rate: number | null
  highest_rate: number | null
  rate_variance_percent: number | null
  _imputed_vendors?: Set<string>
}

interface SubGroupingView {
  name: string
  items: LineItemComparison[]
  vendor_totals: Record<string, number>
  lowest_bidder: string | null
  highest_bidder: string | null
}

interface GroupingComparison {
  grouping_name: string
  vendor_totals: Record<string, number>
  pte_total?: number | null
  lowest_bidder: string | null
  highest_bidder: string | null
  line_items: LineItemComparison[]
}

interface DivisionComparison {
  division_name: string
  vendor_totals: Record<string, number>
  vendor_calculated_totals?: Record<string, number>
  pte_total?: number | null
  lowest_bidder: string | null
  highest_bidder: string | null
  groupings: GroupingComparison[]
}

interface ComparisonSummary {
  vendor_totals: Record<string, number>
  vendor_calculated_totals?: Record<string, number>
  pte_total?: number | null
  lowest_bidder: string | null
  highest_bidder: string | null
  total_line_items: number
  items_with_variance: number
  avg_variance_percent: number | null
}

export interface ComparisonReport {
  project_name: string
  template_file: string
  vendors: string[]
  has_pte?: boolean
  summary: ComparisonSummary
  divisions: DivisionComparison[]
}

type BidMode = "received" | "normalized"
type PriceDisplay = "rate" | "amount"

const SettingsCtx = createContext<{
  bidMode: BidMode
  priceDisplay: PriceDisplay
  vendors: string[]
  hasPte: boolean
}>({
  bidMode: "received",
  priceDisplay: "rate",
  vendors: [],
  hasPte: false,
})

// ============================================================================
// Helpers
// ============================================================================

function fmt(value: number | null | undefined): string {
  if (value === null || value === undefined) return "\u2014"
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function fmtCompact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`
  return fmt(value)
}

function countItems(div: DivisionComparison): number {
  return div.groupings.reduce((s, g) => s + g.line_items.length, 0)
}

function buildSubGroupings(
  grouping: GroupingComparison,
  vendors: string[]
): SubGroupingView[] {
  const map = new Map<string, LineItemComparison[]>()
  for (const it of grouping.line_items) {
    const n = it.sub_grouping_name || "GENERAL"
    if (!map.has(n)) map.set(n, [])
    map.get(n)!.push(it)
  }
  if (map.size === 1 && map.has("GENERAL")) return []
  const out: SubGroupingView[] = []
  for (const [name, items] of map) {
    const vt: Record<string, number> = {}
    for (const v of vendors)
      vt[v] = items.reduce((s, it) => s + (it.vendor_amounts[v] ?? 0), 0)
    let lo: string | null = null,
      loV = Infinity,
      hi: string | null = null,
      hiV = -Infinity
    for (const v of vendors) {
      if (vt[v] > 0 && vt[v] < loV) {
        loV = vt[v]
        lo = v
      }
      if (vt[v] > hiV) {
        hiV = vt[v]
        hi = v
      }
    }
    out.push({
      name,
      items,
      vendor_totals: vt,
      lowest_bidder: lo,
      highest_bidder: hi,
    })
  }
  return out
}

function getDisc(
  ext: number | undefined,
  calc: number | undefined
): { pct: number; diff: number } | null {
  if (ext === undefined || calc === undefined || (ext === 0 && calc === 0))
    return null
  const diff = ext - calc
  if (Math.abs(diff) < 0.5) return null
  return { pct: (diff / (ext !== 0 ? ext : calc)) * 100, diff }
}

// ── Normalize: fill unpriced items with vendor average ──
function normalizeReport(report: ComparisonReport): ComparisonReport {
  const vs = report.vendors
  function normItem(item: LineItemComparison): LineItemComparison {
    const rv = vs
      .map((v) => item.vendor_rates[v])
      .filter((r): r is number => r !== null && r > 0)
    const avgR =
      rv.length > 0 ? rv.reduce((a, b) => a + b, 0) / rv.length : null
    const av = vs
      .map((v) => item.vendor_amounts[v])
      .filter((a): a is number => a !== null && a > 0)
    const avgA =
      av.length > 0 ? av.reduce((a, b) => a + b, 0) / av.length : null
    const nr: Record<string, number | null> = {},
      na: Record<string, number | null> = {},
      imp = new Set<string>()
    for (const v of vs) {
      if (
        (item.vendor_rates[v] === null ||
          item.vendor_rates[v] === undefined) &&
        avgR !== null
      )
        imp.add(v)
      nr[v] = item.vendor_rates[v] ?? avgR
      na[v] = item.vendor_amounts[v] ?? avgA
    }
    const valid = vs
      .map((v) => ({ v, r: nr[v] }))
      .filter((x) => x.r !== null && x.r > 0) as { v: string; r: number }[]
    let lo: string | null = null,
      hi: string | null = null,
      loR: number | null = null,
      hiR: number | null = null
    for (const { v, r } of valid) {
      if (loR === null || r < loR) {
        loR = r
        lo = v
      }
      if (hiR === null || r > hiR) {
        hiR = r
        hi = v
      }
    }
    let vr: number | null = null
    if (loR !== null && hiR !== null && loR > 0)
      vr = ((hiR - loR) / loR) * 100
    return {
      ...item,
      vendor_rates: nr,
      vendor_amounts: na,
      lowest_bidder: lo,
      highest_bidder: hi,
      lowest_rate: loR,
      highest_rate: hiR,
      rate_variance_percent: vr,
      _imputed_vendors: imp.size > 0 ? imp : undefined,
    }
  }
  function normGrp(g: GroupingComparison): GroupingComparison {
    const items = g.line_items.map(normItem)
    const vt: Record<string, number> = {}
    for (const v of vs)
      vt[v] = items.reduce((s, it) => s + (it.vendor_amounts[v] ?? 0), 0)
    const vld = Object.entries(vt).filter(([, t]) => t > 0)
    return {
      ...g,
      line_items: items,
      vendor_totals: vt,
      lowest_bidder: vld.length
        ? vld.reduce((a, b) => (a[1] < b[1] ? a : b))[0]
        : null,
      highest_bidder: vld.length
        ? vld.reduce((a, b) => (a[1] > b[1] ? a : b))[0]
        : null,
    }
  }
  function normDiv(d: DivisionComparison): DivisionComparison {
    const gs = d.groupings.map(normGrp)
    const vt: Record<string, number> = {}
    for (const v of vs)
      vt[v] = gs.reduce((s, g) => s + (g.vendor_totals[v] ?? 0), 0)
    const vld = Object.entries(vt).filter(([, t]) => t > 0)
    return {
      ...d,
      groupings: gs,
      vendor_totals: vt,
      vendor_calculated_totals: vt,
      lowest_bidder: vld.length
        ? vld.reduce((a, b) => (a[1] < b[1] ? a : b))[0]
        : null,
      highest_bidder: vld.length
        ? vld.reduce((a, b) => (a[1] > b[1] ? a : b))[0]
        : null,
    }
  }
  const divs = report.divisions.map(normDiv)
  const vt: Record<string, number> = {}
  for (const v of vs)
    vt[v] = divs.reduce((s, d) => s + (d.vendor_totals[v] ?? 0), 0)
  const vld = Object.entries(vt).filter(([, t]) => t > 0)
  let ti = 0
  const varianceList: number[] = []
  for (const d of divs)
    for (const g of d.groupings)
      for (const it of g.line_items) {
        ti++
        if (
          it.rate_variance_percent !== null &&
          it.rate_variance_percent > 0
        ) {
          varianceList.push(it.rate_variance_percent)
        }
      }
  const wv = varianceList.length
  const avgVariance =
    wv > 0
      ? (() => {
          const s = [...varianceList].sort((a, b) => a - b)
          const mid = wv >> 1
          return wv % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2
        })()
      : null
  return {
    ...report,
    divisions: divs,
    summary: {
      vendor_totals: vt,
      vendor_calculated_totals: vt,
      lowest_bidder: vld.length
        ? vld.reduce((a, b) => (a[1] < b[1] ? a : b))[0]
        : null,
      highest_bidder: vld.length
        ? vld.reduce((a, b) => (a[1] > b[1] ? a : b))[0]
        : null,
      total_line_items: ti,
      items_with_variance: wv,
      avg_variance_percent: avgVariance,
    },
  }
}

// ============================================================================
// Pill Toggle
// ============================================================================
function PillToggle<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string; icon?: React.ReactNode }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="inline-flex items-center rounded-lg bg-muted/60 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-[12px] font-medium transition-all select-none",
            value === o.value
              ? "bg-card text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  )
}

// ============================================================================
// Multi-select Filter (Popover + Command)
// ============================================================================
function MultiFilter({
  label,
  options,
  selected,
  onChange,
}: {
  label: string
  options: { value: string; label: string }[]
  selected: Set<string>
  onChange: (next: Set<string>) => void
}) {
  const [open, setOpen] = useState(false)
  const count = selected.size
  const toggle = useCallback(
    (val: string) => {
      onChange(
        (() => {
          const n = new Set(selected)
          if (n.has(val)) n.delete(val)
          else n.add(val)
          return n
        })()
      )
    },
    [selected, onChange]
  )
  const clear = useCallback(() => onChange(new Set()), [onChange])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          className={cn(
            "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-[12px] font-medium transition-colors select-none",
            count > 0
              ? "border-foreground/20 bg-foreground/5 text-foreground"
              : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/20"
          )}
        >
          <Filter className="size-3" />
          {label}
          {count > 0 && (
            <span className="inline-flex items-center justify-center bg-foreground text-background text-[10px] font-bold rounded-full size-4 ml-0.5">
              {count}
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-56 p-0" align="start">
        <Command>
          <CommandInput placeholder={`Search ${label.toLowerCase()}...`} />
          <CommandList>
            <CommandEmpty>No results.</CommandEmpty>
            <CommandGroup>
              {options.map((o) => (
                <CommandItem
                  key={o.value}
                  onSelect={() => toggle(o.value)}
                  data-checked={selected.has(o.value) || undefined}
                >
                  <div
                    className={cn(
                      "flex items-center justify-center size-4 rounded border mr-1.5 shrink-0",
                      selected.has(o.value)
                        ? "bg-foreground border-foreground text-background"
                        : "border-muted-foreground/30"
                    )}
                  >
                    {selected.has(o.value) && <Check className="size-3" />}
                  </div>
                  <span className="text-[12px] truncate">{o.label}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
          {count > 0 && (
            <div className="border-t p-1">
              <button
                onClick={clear}
                className="w-full text-center text-[11px] text-muted-foreground hover:text-foreground py-1 rounded transition-colors"
              >
                Clear filters
              </button>
            </div>
          )}
        </Command>
      </PopoverContent>
    </Popover>
  )
}

// ============================================================================
// Bidder Summary Table (variance from lowest = 100%)
// ============================================================================
function RankBadge({ rank }: { rank: number }) {
  if (rank === 1)
    return (
      <span className="flex items-center justify-center size-6 rounded-full bg-emerald-100 text-emerald-700">
        <Trophy className="size-3" />
      </span>
    )
  if (rank === 2)
    return (
      <span className="flex items-center justify-center size-6 rounded-full bg-muted text-muted-foreground">
        <Medal className="size-3" />
      </span>
    )
  if (rank === 3)
    return (
      <span className="flex items-center justify-center size-6 rounded-full bg-amber-50 text-amber-600">
        <Award className="size-3" />
      </span>
    )
  return (
    <span className="flex items-center justify-center size-6 rounded-full text-[11px] font-semibold text-muted-foreground/50 tabular-nums">
      {rank}
    </span>
  )
}

function BidderSummaryTable({ report }: { report: ComparisonReport }) {
  const loVendor = report.summary.lowest_bidder
  const loTotal = loVendor ? report.summary.vendor_totals[loVendor] || 0 : 0

  const sorted = [...report.vendors]
    .map((v) => ({ name: v, total: report.summary.vendor_totals[v] || 0 }))
    .sort((a, b) => a.total - b.total)

  const pteTotal = report.summary.pte_total ?? null
  const allForMax = [...sorted.map((s) => s.total)]
  if (pteTotal) allForMax.push(pteTotal)
  const maxTotal = Math.max(...allForMax, 1)

  return (
    <div className="rounded-xl border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b flex items-center gap-2">
        <Shield className="size-4 text-muted-foreground/50" />
        <h3 className="text-[13px] font-semibold tracking-tight">
          Bidder Ranking
        </h3>
        <span className="text-[11px] text-muted-foreground">
          {sorted.length} bidders{pteTotal != null ? " + PTE benchmark" : ""}
        </span>
      </div>
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b bg-muted/20">
            <th className="text-center font-medium text-muted-foreground/70 px-3 py-2.5 w-12 text-[10px] uppercase tracking-wider">
              Rank
            </th>
            <th className="text-left font-medium text-muted-foreground/70 px-4 py-2.5 text-[10px] uppercase tracking-wider">
              Bidder
            </th>
            <th className="text-right font-medium text-muted-foreground/70 px-4 py-2.5 text-[10px] uppercase tracking-wider">
              Total Cost
            </th>
            <th className="text-right font-medium text-muted-foreground/70 px-4 py-2.5 w-20 text-[10px] uppercase tracking-wider">
              Index
            </th>
            <th className="text-right font-medium text-muted-foreground/70 px-4 py-2.5 w-24 text-[10px] uppercase tracking-wider">
              vs. Lowest
            </th>
            <th className="font-medium text-muted-foreground/70 px-4 py-2.5 w-[200px] text-[10px]" />
          </tr>
        </thead>
        <tbody>
          {pteTotal != null && (
            <tr className="border-b bg-sky-50/40">
              <td className="px-3 py-3 text-center">
                <span className="flex items-center justify-center size-6 rounded-full bg-sky-100 text-sky-600 mx-auto">
                  <BarChart3 className="size-3" />
                </span>
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-semibold text-sky-700">
                    Pre-Tender Estimate
                  </span>
                  <span className="inline-flex items-center text-[9px] font-semibold text-sky-600 bg-sky-100 rounded px-1.5 py-0.5 uppercase tracking-wider">
                    Benchmark
                  </span>
                </div>
              </td>
              <td className="px-4 py-3 text-right tabular-nums font-bold text-sky-700">
                {fmt(pteTotal)}
              </td>
              <td className="px-4 py-3 text-right tabular-nums font-semibold text-[12px] text-sky-600">
                {loTotal > 0 ? `${((pteTotal / loTotal) * 100).toFixed(1)}%` : "\u2014"}
              </td>
              <td className="px-4 py-3 text-right tabular-nums text-[12px]">
                {loTotal > 0 ? (
                  <span className="text-sky-600 font-medium">
                    {((pteTotal / loTotal) * 100 - 100) >= 0 ? "+" : ""}
                    {((pteTotal / loTotal) * 100 - 100).toFixed(1)}%
                  </span>
                ) : "\u2014"}
              </td>
              <td className="px-4 py-3">
                <div className="w-full bg-sky-100 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-sky-400 to-sky-500 transition-all duration-700 ease-out"
                    style={{ width: `${Math.min((pteTotal / maxTotal) * 100, 100)}%` }}
                  />
                </div>
              </td>
            </tr>
          )}
          {sorted.map((v, idx) => {
            const isLo = v.name === loVendor
            const isHi = v.name === report.summary.highest_bidder
            const index = loTotal > 0 ? (v.total / loTotal) * 100 : 100
            const variance = index - 100
            const barWidth = Math.min((v.total / maxTotal) * 100, 100)

            return (
              <tr
                key={v.name}
                className={cn(
                  "border-b last:border-b-0 transition-colors group",
                  isLo
                    ? "bg-emerald-50/50"
                    : "hover:bg-muted/15"
                )}
              >
                <td className="px-3 py-3 text-center">
                  <RankBadge rank={idx + 1} />
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "text-[13px] font-medium",
                        isLo && "font-semibold text-emerald-700"
                      )}
                    >
                      {v.name}
                    </span>
                    {isLo && (
                      <span className="inline-flex items-center gap-1 text-[9px] font-bold text-emerald-700 bg-emerald-100 rounded px-1.5 py-0.5 uppercase tracking-wider">
                        <ArrowDownRight className="size-2.5" />
                        Lowest
                      </span>
                    )}
                    {isHi && (
                      <span className="inline-flex items-center gap-1 text-[9px] font-bold text-destructive bg-destructive/8 rounded px-1.5 py-0.5 uppercase tracking-wider">
                        <ArrowUpRight className="size-2.5" />
                        Highest
                      </span>
                    )}
                  </div>
                </td>
                <td
                  className={cn(
                    "px-4 py-3 text-right tabular-nums font-bold",
                    isLo ? "text-emerald-700" : "text-foreground"
                  )}
                >
                  {fmt(v.total)}
                </td>
                <td
                  className={cn(
                    "px-4 py-3 text-right tabular-nums font-semibold text-[12px]",
                    isLo ? "text-emerald-600" : "text-foreground/60"
                  )}
                >
                  {index.toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-[12px]">
                  {isLo ? (
                    <span className="inline-flex items-center gap-1 text-emerald-600 font-semibold">
                      <Minus className="size-3" />
                      Baseline
                    </span>
                  ) : (
                    <span className={cn(
                      "font-semibold",
                      variance > 10 ? "text-destructive" : "text-amber-600"
                    )}>
                      +{variance.toFixed(1)}%
                    </span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <div className="w-full bg-muted/50 rounded-full h-2 overflow-hidden">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-700 ease-out",
                        isLo
                          ? "bg-gradient-to-r from-emerald-400 to-emerald-500"
                          : isHi
                          ? "bg-gradient-to-r from-destructive/40 to-destructive/60"
                          : "bg-gradient-to-r from-muted-foreground/20 to-muted-foreground/30"
                      )}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ============================================================================
// Main Component
// ============================================================================
export function VendorComparison({
  report: raw,
}: {
  report: ComparisonReport
}) {
  const [search, setSearch] = useState("")
  const [expDiv, setExpDiv] = useState<Set<string>>(new Set())
  const [expGrp, setExpGrp] = useState<Set<string>>(new Set())
  const [expSub, setExpSub] = useState<Set<string>>(new Set())
  const [bidMode, setBidMode] = useState<BidMode>("received")
  const [priceDisplay, setPriceDisplay] = useState<PriceDisplay>("rate")
  const [filterDivs, setFilterDivs] = useState<Set<string>>(new Set())
  const [filterGrps, setFilterGrps] = useState<Set<string>>(new Set())

  const normReport = useMemo(() => normalizeReport(raw), [raw])
  const report = bidMode === "normalized" ? normReport : raw

  const unpricedCount = useMemo(() => {
    let c = 0
    for (const d of raw.divisions)
      for (const g of d.groupings)
        for (const it of g.line_items)
          for (const v of raw.vendors)
            if (
              it.vendor_rates[v] === null ||
              it.vendor_rates[v] === undefined
            ) {
              c++
              break
            }
    return c
  }, [raw])

  const divisionOptions = useMemo(
    () =>
      report.divisions.map((d) => ({
        value: d.division_name,
        label: d.division_name,
      })),
    [report.divisions]
  )

  const groupingOptions = useMemo(() => {
    const divs =
      filterDivs.size > 0
        ? report.divisions.filter((d) => filterDivs.has(d.division_name))
        : report.divisions
    const seen = new Set<string>()
    const out: { value: string; label: string }[] = []
    for (const d of divs)
      for (const g of d.groupings) {
        if (!seen.has(g.grouping_name)) {
          seen.add(g.grouping_name)
          out.push({ value: g.grouping_name, label: g.grouping_name })
        }
      }
    return out
  }, [report.divisions, filterDivs])

  const activeFilterCount = filterDivs.size + filterGrps.size

  const hasPte = !!report.has_pte
  const ctx = useMemo(
    () => ({ bidMode, priceDisplay, vendors: report.vendors, hasPte }),
    [bidMode, priceDisplay, report.vendors, hasPte]
  )

  const toggleDiv = (n: string) => {
    setExpDiv((p) => {
      const x = new Set(p)
      if (x.has(n)) {
        x.delete(n)
        setExpGrp((g) => {
          const y = new Set(g)
          for (const k of g) if (k.startsWith(`${n}//`)) y.delete(k)
          return y
        })
        setExpSub((s) => {
          const y = new Set(s)
          for (const k of s) if (k.startsWith(`${n}//`)) y.delete(k)
          return y
        })
      } else x.add(n)
      return x
    })
  }
  const toggleGrp = (d: string, g: string) => {
    const k = `${d}//${g}`
    setExpGrp((p) => {
      const x = new Set(p)
      if (x.has(k)) {
        x.delete(k)
        setExpSub((s) => {
          const y = new Set(s)
          for (const j of s) if (j.startsWith(`${k}//`)) y.delete(j)
          return y
        })
      } else x.add(k)
      return x
    })
  }
  const toggleSub = (d: string, g: string, s: string) => {
    const k = `${d}//${g}//${s}`
    setExpSub((p) => {
      const x = new Set(p)
      if (x.has(k)) x.delete(k)
      else x.add(k)
      return x
    })
  }

  const filtered = useMemo(() => {
    let divs = report.divisions

    if (filterDivs.size > 0)
      divs = divs.filter((d) => filterDivs.has(d.division_name))

    if (filterGrps.size > 0) {
      divs = divs
        .map((d) => ({
          ...d,
          groupings: d.groupings.filter((g) =>
            filterGrps.has(g.grouping_name)
          ),
        }))
        .filter((d) => d.groupings.length > 0)
    }

    if (search) {
      const q = search.toLowerCase()
      divs = divs
        .map((d) => ({
          ...d,
          groupings: d.groupings
            .map((g) => ({
              ...g,
              line_items: g.line_items.filter(
                (it) =>
                  it.item_description.toLowerCase().includes(q) ||
                  it.item_id.toLowerCase().includes(q) ||
                  g.grouping_name.toLowerCase().includes(q)
              ),
            }))
            .filter((g) => g.line_items.length > 0),
        }))
        .filter((d) => d.groupings.length > 0)
    }

    return divs
  }, [report.divisions, search, filterDivs, filterGrps])

  const totalItems = filtered.reduce(
    (s, d) =>
      s + d.groupings.reduce((s2, g) => s2 + g.line_items.length, 0),
    0
  )

  const loTotal =
    report.summary.lowest_bidder
      ? report.summary.vendor_totals[report.summary.lowest_bidder] || 0
      : 0
  const hiTotal =
    report.summary.highest_bidder
      ? report.summary.vendor_totals[report.summary.highest_bidder] || 0
      : 0
  const spread = hiTotal - loTotal

  return (
    <SettingsCtx.Provider value={ctx}>
      <div className="space-y-6 px-6 py-6 overflow-auto">
        {/* ── Hero Stats ──────────────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard
            icon={<Users className="size-4" />}
            label="Bidders"
            value={String(report.vendors.length)}
            sub="compared"
            accent="default"
          />
          <StatCard
            icon={<Layers className="size-4" />}
            label="Line Items"
            value={report.summary.total_line_items.toLocaleString()}
            sub={`${report.divisions.length} divisions`}
            accent="default"
          />
          <StatCard
            icon={<Activity className="size-4" />}
            label="Avg. Variance"
            value={`${report.summary.avg_variance_percent?.toFixed(1) ?? "0"}%`}
            sub={`${report.summary.items_with_variance} items with variance`}
            accent={
              (report.summary.avg_variance_percent ?? 0) > 30
                ? "warning"
                : "default"
            }
          />
          <StatCard
            icon={<BarChart3 className="size-4" />}
            label="Cost Spread"
            value={fmtCompact(spread)}
            sub="highest \u2013 lowest"
            accent={spread > 0 ? "default" : "default"}
          />
        </div>

        {/* ── Bidder Ranking ──────────────────────────────────────── */}
        <BidderSummaryTable report={report} />

        {/* ── Toolbar ─────────────────────────────────────────────── */}
        <div className="flex flex-col gap-3">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <div className="relative flex-1 max-w-xs">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input
                placeholder="Search items..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 h-9 text-[13px]"
              />
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <MultiFilter
                label="Division"
                options={divisionOptions}
                selected={filterDivs}
                onChange={setFilterDivs}
              />
              <MultiFilter
                label="Grouping"
                options={groupingOptions}
                selected={filterGrps}
                onChange={setFilterGrps}
              />
              {activeFilterCount > 0 && (
                <button
                  onClick={() => {
                    setFilterDivs(new Set())
                    setFilterGrps(new Set())
                  }}
                  className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="size-3" />
                  Clear all
                </button>
              )}
              <div className="w-px h-5 bg-border mx-1" />
              <PillToggle<PriceDisplay>
                options={[
                  {
                    value: "rate",
                    label: "Unit Rate",
                    icon: <Hash className="size-3" />,
                  },
                  {
                    value: "amount",
                    label: "Amount",
                    icon: <DollarSign className="size-3" />,
                  },
                ]}
                value={priceDisplay}
                onChange={setPriceDisplay}
              />
              <PillToggle<BidMode>
                options={[
                  { value: "received", label: "As Received" },
                  { value: "normalized", label: "Normalized" },
                ]}
                value={bidMode}
                onChange={setBidMode}
              />
              {bidMode === "normalized" && unpricedCount > 0 && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="inline-flex items-center gap-1 text-[11px] text-violet-600 font-medium cursor-help">
                        <CircleDot className="size-3" />
                        {unpricedCount} filled
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="text-[12px] max-w-[200px]">
                        {unpricedCount} unpriced items filled with avg. rate
                        across vendors
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          </div>

          {/* Active filter tags */}
          {activeFilterCount > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-[11px] text-muted-foreground mr-0.5">
                Showing:
              </span>
              {[...filterDivs].map((d) => (
                <button
                  key={`d-${d}`}
                  onClick={() =>
                    setFilterDivs((p) => {
                      const n = new Set(p)
                      n.delete(d)
                      return n
                    })
                  }
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-muted text-[11px] font-medium text-foreground/80 hover:bg-muted/80 transition-colors group"
                >
                  {d}
                  <X className="size-3 text-muted-foreground group-hover:text-foreground" />
                </button>
              ))}
              {[...filterGrps].map((g) => (
                <button
                  key={`g-${g}`}
                  onClick={() =>
                    setFilterGrps((p) => {
                      const n = new Set(p)
                      n.delete(g)
                      return n
                    })
                  }
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-muted text-[11px] font-medium text-foreground/80 hover:bg-muted/80 transition-colors group"
                >
                  {g}
                  <X className="size-3 text-muted-foreground group-hover:text-foreground" />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── BOQ Comparison Table ────────────────────────────────── */}
        <div className="rounded-xl border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex items-center justify-center size-7 rounded-lg bg-muted/60">
                <Layers className="size-3.5 text-muted-foreground/60" />
              </div>
              <div>
                <h2 className="text-[13px] font-semibold tracking-tight">
                  Bill of Quantities
                </h2>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  {filtered.length} divisions &middot; {totalItems} items
                  &middot;
                  {priceDisplay === "rate" ? " Unit Rates" : " Total Amounts"}
                  {bidMode === "normalized" && " \u00b7 Normalized"}
                </p>
              </div>
            </div>
            {search && (
              <span className="text-[10px] font-medium text-muted-foreground bg-muted px-2 py-0.5 rounded-md uppercase tracking-wider">
                Filtered
              </span>
            )}
          </div>

          <div className="overflow-auto max-h-[calc(100vh-200px)]">
            <table className="w-full caption-bottom text-sm">
              <thead className="sticky top-0 z-10 [&_tr]:border-b">
                <tr className="border-b backdrop-blur-xl bg-card/85 shadow-[0_1px_0_0_var(--border)] supports-[backdrop-filter]:bg-card/60">
                  <th className="text-foreground h-auto px-2 text-left align-middle font-medium whitespace-nowrap min-w-[340px] pl-4 text-[10px] text-muted-foreground/70 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="text-foreground h-auto px-2 text-right align-middle font-medium whitespace-nowrap w-14 text-[10px] text-muted-foreground/70 uppercase tracking-wider">
                    Qty
                  </th>
                  <th className="text-foreground h-auto px-2 text-left align-middle font-medium whitespace-nowrap w-12 text-[10px] text-muted-foreground/70 uppercase tracking-wider">
                    Unit
                  </th>
                  {hasPte && (
                    <th className="h-auto px-2 text-right align-middle font-medium whitespace-nowrap min-w-[120px] py-2.5 bg-sky-50/60 border-l border-sky-100">
                      <div className="text-[11px] font-semibold text-sky-700">
                        PTE
                      </div>
                      <div className="text-[12px] font-bold tabular-nums mt-0.5 text-sky-600">
                        {report.summary.pte_total
                          ? fmtCompact(report.summary.pte_total)
                          : "\u2014"}
                      </div>
                      <div className="text-[10px] font-normal text-sky-500/60 mt-0.5">
                        benchmark
                      </div>
                    </th>
                  )}
                  {report.vendors.map((v) => {
                    const total = report.summary.vendor_totals[v] || 0
                    const isLo = v === report.summary.lowest_bidder
                    return (
                      <th
                        key={v}
                        className={cn(
                          "h-auto px-2 text-right align-middle font-medium whitespace-nowrap min-w-[130px] py-2.5",
                          isLo ? "bg-emerald-50/60" : ""
                        )}
                      >
                        <div
                          className={cn(
                            "text-[11px] font-semibold",
                            isLo
                              ? "text-emerald-700"
                              : "text-muted-foreground"
                          )}
                        >
                          {v}
                        </div>
                        <div
                          className={cn(
                            "text-[12px] font-bold tabular-nums mt-0.5",
                            isLo ? "text-emerald-600" : "text-foreground"
                          )}
                        >
                          {fmtCompact(total)}
                        </div>
                        <div className="text-[10px] font-normal text-muted-foreground/50 mt-0.5">
                          {priceDisplay === "rate" ? "per unit" : "total"}
                        </div>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <TableBody>
                <ProjectTotalRow report={report} />
                {filtered.map((d) => (
                  <DivisionRows
                    key={d.division_name}
                    division={d}
                    isExpanded={expDiv.has(d.division_name)}
                    onToggle={() => toggleDiv(d.division_name)}
                    itemCount={countItems(d)}
                    expandedGroupings={expGrp}
                    onToggleGrouping={(g) => toggleGrp(d.division_name, g)}
                    expandedSubGroupings={expSub}
                    onToggleSubGrouping={(g, s) =>
                      toggleSub(d.division_name, g, s)
                    }
                  />
                ))}
              </TableBody>
            </table>
          </div>
        </div>
      </div>
    </SettingsCtx.Provider>
  )
}

// ============================================================================
// Stat Card
// ============================================================================
function StatCard({
  icon,
  label,
  value,
  sub,
  accent = "default",
}: {
  icon?: React.ReactNode
  label: string
  value: string
  sub: string
  accent?: "default" | "warning"
}) {
  return (
    <div
      className={cn(
        "group relative rounded-xl border p-4 transition-all overflow-hidden",
        accent === "warning"
          ? "bg-gradient-to-br from-amber-50/80 to-amber-50/30 border-amber-200/60"
          : "bg-gradient-to-br from-card to-muted/20 hover:shadow-sm hover:border-border/80"
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">
          {label}
        </span>
        {icon && (
          <span
            className={cn(
              "flex items-center justify-center size-7 rounded-lg",
              accent === "warning"
                ? "bg-amber-100/80 text-amber-600"
                : "bg-muted/60 text-muted-foreground/60 group-hover:bg-muted/80"
            )}
          >
            {icon}
          </span>
        )}
      </div>
      <p
        className={cn(
          "text-2xl font-bold tracking-tight",
          accent === "warning" && "text-amber-700"
        )}
      >
        {value}
      </p>
      <p className="text-[11px] text-muted-foreground mt-1.5">{sub}</p>
    </div>
  )
}

// ============================================================================
// Project Total Row
// ============================================================================
function ProjectTotalRow({ report }: { report: ComparisonReport }) {
  const { bidMode, hasPte } = useContext(SettingsCtx)
  return (
    <TableRow className="bg-gradient-to-r from-muted/40 to-muted/20 font-bold border-b-2">
      <TableCell className="pl-4 py-3">
        <span className="text-[12px] font-bold uppercase tracking-wider">
          Project Total
        </span>
      </TableCell>
      <TableCell />
      <TableCell />
      {hasPte && (
        <TableCell className="text-right py-2.5 bg-sky-50/30 border-l border-sky-100">
          <span className="text-[13px] font-bold tabular-nums text-sky-600">
            {report.summary.pte_total ? fmt(report.summary.pte_total) : "\u2014"}
          </span>
        </TableCell>
      )}
      {report.vendors.map((v) => {
        const t = report.summary.vendor_totals[v] || 0
        const isLo = v === report.summary.lowest_bidder
        const disc =
          bidMode === "received"
            ? getDisc(
                report.summary.vendor_totals[v],
                report.summary.vendor_calculated_totals?.[v]
              )
            : null
        return (
          <TableCell key={v} className="text-right py-2.5">
            <div className="flex items-center justify-end gap-1">
              <span
                className={cn(
                  "text-[13px] font-bold tabular-nums",
                  isLo && "text-emerald-600"
                )}
              >
                {fmt(t)}
              </span>
              {disc && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <AlertTriangle className="size-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <div className="text-[12px] space-y-0.5">
                        <p>BOQ: {fmt(t)}</p>
                        <p>
                          Sum:{" "}
                          {fmt(
                            report.summary.vendor_calculated_totals?.[v] ?? 0
                          )}
                        </p>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          </TableCell>
        )
      })}
    </TableRow>
  )
}

// ============================================================================
// Division Rows
// ============================================================================
function DivisionRows({
  division: d,
  isExpanded,
  onToggle,
  itemCount,
  expandedGroupings,
  onToggleGrouping,
  expandedSubGroupings,
  onToggleSubGrouping,
}: {
  division: DivisionComparison
  isExpanded: boolean
  onToggle: () => void
  itemCount: number
  expandedGroupings: Set<string>
  onToggleGrouping: (g: string) => void
  expandedSubGroupings: Set<string>
  onToggleSubGrouping: (g: string, s: string) => void
}) {
  const { vendors, bidMode, hasPte } = useContext(SettingsCtx)
  return (
    <>
      <TableRow
        className={cn(
          "cursor-pointer transition-colors border-l-2",
          isExpanded
            ? "bg-muted/40 border-l-foreground/20"
            : "hover:bg-muted/20 border-l-transparent"
        )}
        onClick={onToggle}
      >
        <TableCell className="pl-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className={cn(
              "transition-transform duration-200",
              isExpanded ? "text-foreground" : "text-muted-foreground"
            )}>
              {isExpanded ? (
                <ChevronDown className="size-4" />
              ) : (
                <ChevronRight className="size-4" />
              )}
            </span>
            <span className="text-[13px] font-semibold">
              {d.division_name}
            </span>
            <span className="text-[10px] text-muted-foreground/60 tabular-nums">
              {d.groupings.length}g &middot; {itemCount}i
            </span>
          </div>
        </TableCell>
        <TableCell />
        <TableCell />
        {hasPte && (
          <TableCell className="text-right py-2.5 bg-sky-50/20 border-l border-sky-100">
            <span className="text-[13px] font-semibold tabular-nums text-sky-600">
              {d.pte_total ? fmt(d.pte_total) : "\u2014"}
            </span>
          </TableCell>
        )}
        {vendors.map((v) => {
          const t = d.vendor_totals[v] || 0
          const isLo = v === d.lowest_bidder
          const isHi = v === d.highest_bidder
          const disc =
            bidMode === "received"
              ? getDisc(d.vendor_totals[v], d.vendor_calculated_totals?.[v])
              : null
          return (
            <TableCell key={v} className="text-right py-2.5">
              <div className="flex items-center justify-end gap-1">
                <span
                  className={cn(
                    "text-[13px] font-semibold tabular-nums",
                    isLo && "text-emerald-600",
                    isHi && "text-destructive"
                  )}
                >
                  {fmt(t)}
                </span>
                {disc && (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <AlertTriangle className="size-2.5 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="bottom">
                        <div className="text-[12px]">
                          <p>BOQ: {fmt(t)}</p>
                          <p>
                            Sum:{" "}
                            {fmt(d.vendor_calculated_totals?.[v] ?? 0)}
                          </p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
              </div>
            </TableCell>
          )
        })}
      </TableRow>
      {isExpanded &&
        d.groupings.map((g) => {
          const k = `${d.division_name}//${g.grouping_name}`
          return (
            <GroupingRows
              key={k}
              divisionName={d.division_name}
              grouping={g}
              isExpanded={expandedGroupings.has(k)}
              onToggle={() => onToggleGrouping(g.grouping_name)}
              expandedSubGroupings={expandedSubGroupings}
              onToggleSubGrouping={(s) =>
                onToggleSubGrouping(g.grouping_name, s)
              }
            />
          )
        })}
    </>
  )
}

// ============================================================================
// Grouping Rows
// ============================================================================
function GroupingRows({
  divisionName,
  grouping,
  isExpanded,
  onToggle,
  expandedSubGroupings,
  onToggleSubGrouping,
}: {
  divisionName: string
  grouping: GroupingComparison
  isExpanded: boolean
  onToggle: () => void
  expandedSubGroupings: Set<string>
  onToggleSubGrouping: (s: string) => void
}) {
  const { vendors, hasPte } = useContext(SettingsCtx)
  const subs = useMemo(
    () => buildSubGroupings(grouping, vendors),
    [grouping, vendors]
  )
  return (
    <>
      <TableRow
        className={cn(
          "cursor-pointer hover:bg-muted/15 transition-colors border-l-2",
          isExpanded
            ? "border-l-muted-foreground/15"
            : "border-l-transparent"
        )}
        onClick={onToggle}
      >
        <TableCell className="pl-10 py-2">
          <div className="flex items-center gap-2">
            <span className={cn(
              "transition-transform duration-200",
              isExpanded ? "text-foreground/70" : "text-muted-foreground"
            )}>
              {isExpanded ? (
                <ChevronDown className="size-3.5" />
              ) : (
                <ChevronRight className="size-3.5" />
              )}
            </span>
            <span className="text-[12px] font-medium">
              {grouping.grouping_name}
            </span>
            <span className="text-[10px] text-muted-foreground/50 tabular-nums">
              {grouping.line_items.length}
            </span>
          </div>
        </TableCell>
        <TableCell />
        <TableCell />
        {hasPte && (
          <TableCell className="text-right text-[12px] tabular-nums py-2 text-sky-600 bg-sky-50/10 border-l border-sky-100">
            {grouping.pte_total ? fmt(grouping.pte_total) : "\u2014"}
          </TableCell>
        )}
        {vendors.map((v) => {
          const t = grouping.vendor_totals[v] || 0
          const isLo = v === grouping.lowest_bidder
          const isHi = v === grouping.highest_bidder
          return (
            <TableCell
              key={v}
              className={cn(
                "text-right text-[12px] tabular-nums py-2",
                isLo && "text-emerald-600 font-medium",
                isHi && "text-destructive"
              )}
            >
              {fmt(t)}
            </TableCell>
          )
        })}
      </TableRow>
      {isExpanded &&
        (subs.length > 0
          ? subs.map((s) => {
              const k = `${divisionName}//${grouping.grouping_name}//${s.name}`
              return (
                <SubGroupingRows
                  key={k}
                  sub={s}
                  isExpanded={expandedSubGroupings.has(k)}
                  onToggle={() => onToggleSubGrouping(s.name)}
                />
              )
            })
          : grouping.line_items.map((it, i) => (
              <LineItemRow
                key={`${grouping.grouping_name}-${it.item_id}-${i}`}
                item={it}
              />
            )))}
    </>
  )
}

// ============================================================================
// Sub-grouping Rows
// ============================================================================
function SubGroupingRows({
  sub,
  isExpanded,
  onToggle,
}: {
  sub: SubGroupingView
  isExpanded: boolean
  onToggle: () => void
}) {
  const { vendors, hasPte } = useContext(SettingsCtx)
  const pteSub = useMemo(() => {
    const t = sub.items.reduce((s, it) => s + (it.pte_amount ?? 0), 0)
    return t > 0 ? t : null
  }, [sub.items])
  return (
    <>
      <TableRow
        className="cursor-pointer hover:bg-muted/15 transition-colors"
        onClick={onToggle}
      >
        <TableCell className="pl-16 py-1.5">
          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground">
              {isExpanded ? (
                <ChevronDown className="size-3" />
              ) : (
                <ChevronRight className="size-3" />
              )}
            </span>
            <span className="text-[11px] font-medium text-muted-foreground">
              {sub.name}
            </span>
            <span className="text-[10px] text-muted-foreground/60">
              ({sub.items.length})
            </span>
          </div>
        </TableCell>
        <TableCell />
        <TableCell />
        {hasPte && (
          <TableCell className="text-right text-[11px] tabular-nums py-1.5 text-sky-600 bg-sky-50/10 border-l border-sky-100">
            {pteSub ? fmt(pteSub) : "\u2014"}
          </TableCell>
        )}
        {vendors.map((v) => {
          const t = sub.vendor_totals[v] || 0
          const isLo = v === sub.lowest_bidder
          const isHi = v === sub.highest_bidder
          return (
            <TableCell
              key={v}
              className={cn(
                "text-right text-[11px] tabular-nums py-1.5",
                isLo && "text-emerald-600 font-medium",
                isHi && "text-destructive"
              )}
            >
              {fmt(t)}
            </TableCell>
          )
        })}
      </TableRow>
      {isExpanded &&
        sub.items.map((it, i) => (
          <LineItemRow
            key={`s-${sub.name}-${it.item_id}-${i}`}
            item={it}
            indent="deep"
          />
        ))}
    </>
  )
}

// ============================================================================
// Line Item Row
// ============================================================================
function LineItemRow({
  item,
  indent = "normal",
}: {
  item: LineItemComparison
  indent?: "normal" | "deep"
}) {
  const { vendors, priceDisplay, hasPte } = useContext(SettingsCtx)
  const isImp = (v: string) => item._imputed_vendors?.has(v) ?? false

  const pteVal =
    priceDisplay === "rate" ? item.pte_rate : item.pte_amount

  return (
    <TableRow className="hover:bg-muted/10 transition-colors">
      <TableCell
        className={cn("py-1.5", indent === "deep" ? "pl-20" : "pl-16")}
      >
        <div className="flex items-start gap-2">
          <span className="font-mono text-[10px] text-muted-foreground w-10 shrink-0 pt-0.5 text-right">
            {item.item_id}
          </span>
          <span className="text-[12px] text-foreground/80 leading-relaxed">
            {item.item_description}
          </span>
        </div>
      </TableCell>
      <TableCell className="text-right text-[12px] text-muted-foreground tabular-nums py-1.5">
        {item.item_quantity ?? "\u2014"}
      </TableCell>
      <TableCell className="text-[11px] text-muted-foreground py-1.5">
        {item.item_unit}
      </TableCell>

      {hasPte && (
        <TableCell className="text-right text-[12px] tabular-nums py-1.5 text-sky-600 bg-sky-50/10 border-l border-sky-100">
          {pteVal != null ? fmt(pteVal) : "\u2014"}
        </TableCell>
      )}

      {vendors.map((v) => {
        const rawRate = item.vendor_rates[v]
        const rawAmt = item.vendor_amounts[v]
        const val = priceDisplay === "rate" ? rawRate : rawAmt
        const isLo =
          v === item.lowest_bidder &&
          item.lowest_bidder !== item.highest_bidder
        const isHi =
          v === item.highest_bidder &&
          item.lowest_bidder !== item.highest_bidder
        const imp = isImp(v)
        const unpriced =
          !imp &&
          (rawRate === null || rawRate === undefined) &&
          (rawAmt === null || rawAmt === undefined)

        return (
          <TableCell
            key={v}
            className={cn(
              "text-right text-[12px] tabular-nums py-1.5",
              imp
                ? "italic text-violet-500"
                : unpriced
                ? "text-violet-400"
                : isLo
                ? "text-emerald-600 font-medium"
                : isHi
                ? "text-destructive"
                : ""
            )}
          >
            <span className="inline-flex items-center gap-0.5">
              {unpriced ? (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="cursor-help">Unpriced</span>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p className="text-[11px]">
                        No price submitted — will be flagged as PTC
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              ) : val !== null && val !== undefined ? (
                fmt(val)
              ) : (
                "\u2014"
              )}
              {!imp && !unpriced && isLo && (
                <TrendingDown className="size-3 text-emerald-500/50" />
              )}
              {!imp && !unpriced && isHi && (
                <TrendingUp className="size-3 text-destructive/50" />
              )}
              {imp && val !== null && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CircleDot className="size-2.5 text-violet-400" />
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p className="text-[11px]">
                        Unpriced — avg. imputed
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </span>
          </TableCell>
        )
      })}
    </TableRow>
  )
}

// ============================================================================
// Data Adapter (legacy CommercialEvaluationData → ComparisonReport)
// ============================================================================

import type {
  CommercialEvaluationData,
  ContractorPTCs,
  PTCItem,
} from "@/lib/types"

export function toComparisonReport(
  data: CommercialEvaluationData
): ComparisonReport {
  const { boq, contractors } = data
  const sorted = [...contractors].sort((a, b) => a.totalAmount - b.totalAmount)
  const vs = sorted.map((c) => c.contractorName)
  const vt: Record<string, number> = {}
  for (const c of sorted) vt[c.contractorName] = c.totalAmount
  const lo = sorted[0]?.contractorName ?? null
  const hi = sorted[sorted.length - 1]?.contractorName ?? null
  let ti = 0,
    wv = 0,
    vs2 = 0

  const divs: DivisionComparison[] = boq.divisions.map((div) => {
    const dvt: Record<string, number> = {}
    for (const v of vs) dvt[v] = 0
    const gs: GroupingComparison[] = div.sections.map((sec) => {
      const gvt: Record<string, number> = {}
      for (const v of vs) gvt[v] = 0
      const items: LineItemComparison[] = sec.lineItems.map((it) => {
        ti++
        const vr: Record<string, number | null> = {}
        const va: Record<string, number | null> = {}
        let lr: number | null = null,
          hr: number | null = null,
          lv: string | null = null,
          hv: string | null = null
        for (const c of sorted) {
          const p = c.prices[it.id]
          const r =
            p !== null && p !== undefined && it.quantity > 0
              ? p / it.quantity
              : p
          vr[c.contractorName] = r ?? null
          va[c.contractorName] = p ?? null
          if (p !== null && p !== undefined) {
            gvt[c.contractorName] += p
            dvt[c.contractorName] += p
          }
          if (r !== null && r !== undefined) {
            if (lr === null || r < lr) {
              lr = r
              lv = c.contractorName
            }
            if (hr === null || r > hr) {
              hr = r
              hv = c.contractorName
            }
          }
        }
        let rv: number | null = null
        if (lr !== null && hr !== null && lr > 0) {
          rv = ((hr - lr) / lr) * 100
          if (rv > 0) {
            wv++
            vs2 += rv
          }
        }
        return {
          item_id: it.code,
          item_description: it.description,
          item_quantity: it.quantity,
          item_unit: it.unit,
          sub_grouping_name: null,
          vendor_rates: vr,
          vendor_amounts: va,
          lowest_bidder: lv,
          highest_bidder: hv,
          lowest_rate: lr,
          highest_rate: hr,
          rate_variance_percent: rv,
        }
      })
      let gl: string | null = null,
        glv = Infinity,
        gh: string | null = null,
        ghv = -Infinity
      for (const v of vs) {
        if (gvt[v] < glv) {
          glv = gvt[v]
          gl = v
        }
        if (gvt[v] > ghv) {
          ghv = gvt[v]
          gh = v
        }
      }
      return {
        grouping_name: sec.name,
        vendor_totals: gvt,
        lowest_bidder: gl,
        highest_bidder: gh,
        line_items: items,
      }
    })
    let dl: string | null = null,
      dlv = Infinity,
      dh: string | null = null,
      dhv = -Infinity
    for (const v of vs) {
      if (dvt[v] < dlv) {
        dlv = dvt[v]
        dl = v
      }
      if (dvt[v] > dhv) {
        dhv = dvt[v]
        dh = v
      }
    }
    return {
      division_name: div.name,
      vendor_totals: dvt,
      lowest_bidder: dl,
      highest_bidder: dh,
      groupings: gs,
    }
  })

  return {
    project_name: "",
    template_file: "",
    vendors: vs,
    summary: {
      vendor_totals: vt,
      lowest_bidder: lo,
      highest_bidder: hi,
      total_line_items: ti,
      items_with_variance: wv,
      avg_variance_percent: wv > 0 ? vs2 / wv : null,
    },
    divisions: divs,
  }
}

// ============================================================================
// Derive PTC entries from a ComparisonReport
// ============================================================================
export function derivePTCsFromReport(
  report: ComparisonReport
): ContractorPTCs[] {
  return report.vendors.map((vendor) => {
    const ptcs: PTCItem[] = []

    for (const div of report.divisions) {
      for (const grp of div.groupings) {
        for (const item of grp.line_items) {
          const rate = item.vendor_rates[vendor]
          const amount = item.vendor_amounts[vendor]
          const isUnpriced =
            (rate === null || rate === undefined) &&
            (amount === null || amount === undefined)

          if (isUnpriced) {
            ptcs.push({
              id: `ptc-${vendor}-${item.item_id}`,
              referenceSection: item.item_id,
              queryDescription:
                "No price submitted. Please confirm unit rate and total amount.",
              vendorResponse: "",
              status: "pending",
              category: "pricing_anomalies",
              lineItemId: item.item_id,
              lineItemDescription: item.item_description,
              lineItemQty: item.item_quantity,
              lineItemUnit: item.item_unit,
              divisionName: div.division_name,
              groupingName: grp.grouping_name,
            })
          }
        }
      }
    }

    return {
      contractorId: vendor,
      contractorName: vendor,
      ptcs,
    }
  })
}
