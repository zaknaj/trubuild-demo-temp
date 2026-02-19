import type {
  TechRfpContractorData,
  TechRfpPtc,
  TechRfpPtcRefType,
  TechRfpTechnicalEvaluationPtcStatus,
  TechRfpTechnicalEvaluationContractorPtcs,
} from "@/lib/tech_rfp"

function normalizeMatchKey(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
}

const validRefTypes: ReadonlySet<TechRfpPtcRefType> = new Set([
  "N/A",
  "MISSING_INFO",
  "INCOMPLETE",
])

function mapLegacyCategoryToRefType(category: unknown): TechRfpPtcRefType {
  if (category === "exclusions") return "MISSING_INFO"
  if (category === "deviations") return "INCOMPLETE"
  return "N/A"
}

function normalizeRefType(value: unknown, legacyCategory: unknown): TechRfpPtcRefType {
  if (typeof value === "string" && validRefTypes.has(value as TechRfpPtcRefType)) {
    return value as TechRfpPtcRefType
  }
  return mapLegacyCategoryToRefType(legacyCategory)
}

function normalizeStatus(value: unknown): TechRfpTechnicalEvaluationPtcStatus {
  return value === "closed" ? "closed" : "pending"
}

export function normalizeTechnicalEvaluationPtcs(
  input: unknown
): TechRfpTechnicalEvaluationContractorPtcs[] {
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
      const rawPtcs = Array.isArray(contractorRecord.ptcs)
        ? contractorRecord.ptcs
        : []

      if (!contractorId || !contractorName) return null

      const normalizedPtcs = rawPtcs
        .map((item) => {
          if (!item || typeof item !== "object") return null
          const record = item as Record<string, unknown>

          const criterionCandidate =
            typeof record.criterion === "string"
              ? record.criterion
              : typeof record.referenceSection === "string"
                ? record.referenceSection
                : "General"
          const criterion = criterionCandidate.trim() || "General"

          const queryDescription =
            typeof record.queryDescription === "string"
              ? record.queryDescription.trim()
              : ""
          if (!queryDescription) return null

          const refType = normalizeRefType(record.refType, record.category)
          const vendorResponse =
            typeof record.vendorResponse === "string" ? record.vendorResponse : ""

          return {
            criterion,
            queryDescription,
            refType,
            status: normalizeStatus(record.status),
            vendorResponse,
          }
        })
        .filter(
          (item): item is TechRfpTechnicalEvaluationContractorPtcs["ptcs"][number] =>
            item !== null
        )

      return {
        contractorId,
        contractorName,
        ptcs: normalizedPtcs,
      }
    })
    .filter((contractor): contractor is TechRfpTechnicalEvaluationContractorPtcs => {
      return contractor !== null
    })
}

export function mapAnalysisResultToTechnicalPtcs({
  tenderReport,
  contractors,
  proposalsUploaded,
}: {
  tenderReport: TechRfpContractorData[]
  contractors: Array<{ id: string; name: string }>
  proposalsUploaded: string[]
}): TechRfpTechnicalEvaluationContractorPtcs[] {
  const activeContractors = contractors.filter((contractor) =>
    proposalsUploaded.includes(contractor.id)
  )
  if (activeContractors.length === 0) return []

  const contractorByKey = new Map<string, { id: string; name: string }>()
  for (const contractor of activeContractors) {
    const key = normalizeMatchKey(contractor.name)
    if (!contractorByKey.has(key)) {
      contractorByKey.set(key, contractor)
    }
  }

  const contractorPtcsMap = new Map<
    string,
    TechRfpTechnicalEvaluationContractorPtcs
  >()

  for (const contractorResult of tenderReport) {
    const resultContractorKey = normalizeMatchKey(contractorResult.contractorName)
    const contractorMatch = contractorByKey.get(resultContractorKey)
    if (!contractorMatch) continue

    const existingContractorEntry = contractorPtcsMap.get(contractorMatch.id) ?? {
      contractorId: contractorMatch.id,
      contractorName: contractorMatch.name,
      ptcs: [],
    }
    const seenKeys = new Set(
      existingContractorEntry.ptcs.map(
        (item) => `${item.criterion}::${item.refType}::${item.queryDescription}`
      )
    )

    for (const scope of contractorResult.tenderEvaluation.scopes) {
      for (const criterion of scope.evaluationBreakdown) {
        if (!criterion.ptc || criterion.ptc === "NA") continue

        const ptc = criterion.ptc as TechRfpPtc
        const queryDescription = ptc.queryDescription?.trim()
        if (!queryDescription) continue
        if (queryDescription.toUpperCase() === "N/A") continue
        if (!validRefTypes.has(ptc.refType as TechRfpPtcRefType)) continue

        const criterionName = criterion.criteria?.trim()
        if (!criterionName) continue

        const dedupeKey = `${criterionName}::${ptc.refType}::${queryDescription}`
        if (seenKeys.has(dedupeKey)) continue
        seenKeys.add(dedupeKey)

        existingContractorEntry.ptcs.push({
          criterion: criterionName,
          queryDescription,
          refType: ptc.refType as TechRfpPtcRefType,
          status: "pending",
          vendorResponse: "",
        })
      }
    }

    contractorPtcsMap.set(contractorMatch.id, existingContractorEntry)
  }

  return Array.from(contractorPtcsMap.values())
}

export type TechnicalPtcRow = {
  rowId: string
  contractorId: string
  contractorName: string
  index: number
  criterion: string
  queryDescription: string
  refType: TechRfpPtcRefType
  status: TechRfpTechnicalEvaluationPtcStatus
  vendorResponse: string
}

export type TechnicalPtcStats = {
  total: number
  pending: number
  closed: number
  completionPercent: number
  byRefType: Record<TechRfpPtcRefType, number>
  byContractor: Array<{
    contractorId: string
    contractorName: string
    total: number
    pending: number
    closed: number
  }>
}

export function flattenTechnicalPtcs(
  ptcs: TechRfpTechnicalEvaluationContractorPtcs[]
): TechnicalPtcRow[] {
  const rows: TechnicalPtcRow[] = []
  for (const contractor of ptcs) {
    contractor.ptcs.forEach((item, index) => {
      rows.push({
        rowId: `${contractor.contractorId}:${index}`,
        contractorId: contractor.contractorId,
        contractorName: contractor.contractorName,
        index,
        criterion: item.criterion,
        queryDescription: item.queryDescription,
        refType: item.refType,
        status: item.status ?? "pending",
        vendorResponse: item.vendorResponse ?? "",
      })
    })
  }
  return rows
}

export function buildTechnicalPtcStats(
  rows: TechnicalPtcRow[]
): TechnicalPtcStats {
  const byRefType: Record<TechRfpPtcRefType, number> = {
    "N/A": 0,
    MISSING_INFO: 0,
    INCOMPLETE: 0,
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
    byRefType[row.refType] += 1
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
    byRefType,
    byContractor: Array.from(contractorStats.values()).sort(
      (a, b) => b.total - a.total
    ),
  }
}

export function filterTechnicalPtcRows(
  rows: TechnicalPtcRow[],
  filters: {
    contractorId: "all" | string
    refType: "all" | TechRfpPtcRefType
    search: string
  }
): TechnicalPtcRow[] {
  const search = filters.search.trim().toLowerCase()

  return rows.filter((row) => {
    if (filters.contractorId !== "all" && row.contractorId !== filters.contractorId) {
      return false
    }
    if (filters.refType !== "all" && row.refType !== filters.refType) {
      return false
    }
    if (!search) return true

    return (
      row.contractorName.toLowerCase().includes(search) ||
      row.criterion.toLowerCase().includes(search) ||
      row.queryDescription.toLowerCase().includes(search) ||
      row.vendorResponse.toLowerCase().includes(search)
    )
  })
}
