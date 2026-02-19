import type {
  ContractorPTCs,
  PTCCategory,
  PTCItem,
  PTCStatus,
} from "@/lib/types"

const VALID_CATEGORIES: ReadonlySet<PTCCategory> = new Set([
  "exclusions",
  "deviations",
  "pricing_anomalies",
  "arithmetic_checks",
])

function normalizeStatus(value: unknown): PTCStatus {
  return value === "closed" ? "closed" : "pending"
}

function normalizeCategory(value: unknown): PTCCategory {
  if (typeof value === "string" && VALID_CATEGORIES.has(value as PTCCategory)) {
    return value as PTCCategory
  }
  return "exclusions"
}

function normalizePtcItem(item: unknown, fallbackIndex: number): PTCItem | null {
  if (!item || typeof item !== "object") return null
  const record = item as Record<string, unknown>

  const queryDescription =
    typeof record.queryDescription === "string" ? record.queryDescription.trim() : ""
  if (!queryDescription) return null

  const id =
    typeof record.id === "string" && record.id.length > 0
      ? record.id
      : `generated-${fallbackIndex}`

  return {
    id,
    referenceSection:
      typeof record.referenceSection === "string" ? record.referenceSection : "",
    queryDescription,
    vendorResponse:
      typeof record.vendorResponse === "string" ? record.vendorResponse : "",
    status: normalizeStatus(record.status),
    category: normalizeCategory(record.category),
  }
}

export function normalizeCommercialEvaluationPtcs(input: unknown): ContractorPTCs[] {
  if (!Array.isArray(input)) return []

  return input
    .map((contractor) => {
      if (!contractor || typeof contractor !== "object") return null
      const contractorRecord = contractor as Record<string, unknown>
      const contractorId =
        typeof contractorRecord.contractorId === "string"
          ? contractorRecord.contractorId
          : ""
      const contractorName =
        typeof contractorRecord.contractorName === "string"
          ? contractorRecord.contractorName
          : ""
      if (!contractorId || !contractorName) return null

      const rawPtcs = Array.isArray(contractorRecord.ptcs) ? contractorRecord.ptcs : []
      const ptcs = rawPtcs
        .map((item, index) => normalizePtcItem(item, index))
        .filter((item): item is PTCItem => item !== null)

      return {
        contractorId,
        contractorName,
        ptcs,
      }
    })
    .filter((contractor): contractor is ContractorPTCs => contractor !== null)
}

export type CommercialPtcRow = {
  rowId: string
  contractorId: string
  contractorName: string
  index: number
  id: string
  referenceSection: string
  queryDescription: string
  vendorResponse: string
  category: PTCCategory
  status: PTCStatus
}

export function flattenCommercialPtcs(ptcs: ContractorPTCs[]): CommercialPtcRow[] {
  const rows: CommercialPtcRow[] = []
  for (const contractor of ptcs) {
    contractor.ptcs.forEach((item, index) => {
      rows.push({
        rowId: `${contractor.contractorId}:${item.id}:${index}`,
        contractorId: contractor.contractorId,
        contractorName: contractor.contractorName,
        index,
        id: item.id,
        referenceSection: item.referenceSection,
        queryDescription: item.queryDescription,
        vendorResponse: item.vendorResponse,
        category: item.category,
        status: item.status,
      })
    })
  }
  return rows
}

export type CommercialPtcStats = {
  total: number
  pending: number
  closed: number
  completionPercent: number
  byCategory: Record<PTCCategory, number>
  byContractor: Array<{
    contractorId: string
    contractorName: string
    total: number
    pending: number
    closed: number
  }>
}

export function buildCommercialPtcStats(rows: CommercialPtcRow[]): CommercialPtcStats {
  const byCategory: Record<PTCCategory, number> = {
    exclusions: 0,
    deviations: 0,
    pricing_anomalies: 0,
    arithmetic_checks: 0,
  }
  let pending = 0
  let closed = 0
  const contractorStats = new Map<
    string,
    {
      contractorId: string
      contractorName: string
      total: number
      pending: number
      closed: number
    }
  >()

  for (const row of rows) {
    byCategory[row.category] += 1
    if (row.status === "closed") {
      closed += 1
    } else {
      pending += 1
    }

    const current = contractorStats.get(row.contractorId) ?? {
      contractorId: row.contractorId,
      contractorName: row.contractorName,
      total: 0,
      pending: 0,
      closed: 0,
    }
    current.total += 1
    if (row.status === "closed") {
      current.closed += 1
    } else {
      current.pending += 1
    }
    contractorStats.set(row.contractorId, current)
  }

  const total = rows.length
  const completionPercent = total > 0 ? Math.round((closed / total) * 100) : 0

  return {
    total,
    pending,
    closed,
    completionPercent,
    byCategory,
    byContractor: Array.from(contractorStats.values()).sort(
      (a, b) => b.total - a.total
    ),
  }
}

export function filterCommercialPtcRows(
  rows: CommercialPtcRow[],
  filters: {
    contractorId: "all" | string
    category: "all" | PTCCategory
    status: "all" | PTCStatus
    search: string
  }
): CommercialPtcRow[] {
  const search = filters.search.trim().toLowerCase()

  return rows.filter((row) => {
    if (filters.contractorId !== "all" && row.contractorId !== filters.contractorId) {
      return false
    }
    if (filters.category !== "all" && row.category !== filters.category) {
      return false
    }
    if (filters.status !== "all" && row.status !== filters.status) {
      return false
    }
    if (!search) return true

    return (
      row.contractorName.toLowerCase().includes(search) ||
      row.referenceSection.toLowerCase().includes(search) ||
      row.queryDescription.toLowerCase().includes(search) ||
      row.vendorResponse.toLowerCase().includes(search)
    )
  })
}
