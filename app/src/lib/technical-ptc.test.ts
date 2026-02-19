import { describe, expect, it } from "vitest"
import {
  buildTechnicalPtcStats,
  filterTechnicalPtcRows,
  flattenTechnicalPtcs,
  mapAnalysisResultToTechnicalPtcs,
  normalizeTechnicalEvaluationPtcs,
} from "@/lib/technical-ptc"
import type {
  TechRfpContractorData,
  TechRfpTechnicalEvaluationContractorPtcs,
} from "@/lib/tech_rfp"

function buildContractorResult(
  contractorName: string,
  evaluationBreakdown: Array<{
    criteria: string
    ptc: unknown
  }>
): TechRfpContractorData {
  return {
    contractorName,
    tenderEvaluation: {
      totalScore: 0,
      scopes: [
        {
          scopeName: "Scope A",
          scopeTotal: 0,
          evaluationBreakdown: evaluationBreakdown.map((item) => ({
            criteria: item.criteria,
            ptc: item.ptc as never,
            grade: "N/A",
            score: 0,
            result: 0,
            evidence: [],
            clientSummary: "",
          })),
        },
      ],
    },
    raw_llm_responses: [],
  }
}

describe("mapAnalysisResultToTechnicalPtcs", () => {
  it("maps valid PTCs and filters NA values", () => {
    const tenderReport = [
      buildContractorResult("BuildCorp International", [
        {
          criteria: "Schedule",
          ptc: {
            queryDescription: "Please provide a milestone breakdown.",
            refType: "MISSING_INFO",
          },
        },
        {
          criteria: "Method Statement",
          ptc: "NA",
        },
        {
          criteria: "QA/QC",
          ptc: {
            queryDescription: "N/A",
            refType: "N/A",
          },
        },
      ]),
    ]

    const mapped = mapAnalysisResultToTechnicalPtcs({
      tenderReport,
      contractors: [
        { id: "11111111-1111-1111-1111-111111111111", name: "BuildCorp International" },
      ],
      proposalsUploaded: ["11111111-1111-1111-1111-111111111111"],
    })

    expect(mapped).toHaveLength(1)
    expect(mapped[0].ptcs).toHaveLength(1)
    expect(mapped[0].ptcs[0]).toMatchObject({
      criterion: "Schedule",
      queryDescription: "Please provide a milestone breakdown.",
      refType: "MISSING_INFO",
      status: "pending",
      vendorResponse: "",
    })
  })

  it("deduplicates repeated criterion PTC entries", () => {
    const tenderReport = [
      buildContractorResult("Alpha Build", [
        {
          criteria: "Technical Approach",
          ptc: {
            queryDescription: "Clarify the sequencing assumptions.",
            refType: "INCOMPLETE",
          },
        },
        {
          criteria: "Technical Approach",
          ptc: {
            queryDescription: "Clarify the sequencing assumptions.",
            refType: "INCOMPLETE",
          },
        },
      ]),
    ]

    const mapped = mapAnalysisResultToTechnicalPtcs({
      tenderReport,
      contractors: [{ id: "22222222-2222-2222-2222-222222222222", name: "Alpha Build" }],
      proposalsUploaded: ["22222222-2222-2222-2222-222222222222"],
    })

    expect(mapped).toHaveLength(1)
    expect(mapped[0].ptcs).toHaveLength(1)
  })
})

describe("technical ptc view model helpers", () => {
  const samplePtcs: TechRfpTechnicalEvaluationContractorPtcs[] = [
    {
      contractorId: "11111111-1111-1111-1111-111111111111",
      contractorName: "BuildCorp International",
      ptcs: [
        {
          criterion: "Schedule",
          queryDescription: "Please provide a milestone breakdown.",
          refType: "MISSING_INFO",
          status: "pending",
          vendorResponse: "",
        },
        {
          criterion: "Method Statement",
          queryDescription: "Clarify sequencing assumptions.",
          refType: "INCOMPLETE",
          status: "closed",
          vendorResponse: "Updated method statement attached.",
        },
      ],
    },
    {
      contractorId: "22222222-2222-2222-2222-222222222222",
      contractorName: "Alpha Build",
      ptcs: [
        {
          criterion: "Compliance",
          queryDescription: "Please confirm compliance with clause 8.",
          refType: "N/A",
          status: "pending",
          vendorResponse: "",
        },
      ],
    },
  ]

  it("flattens contractor grouped ptcs into stable rows", () => {
    const rows = flattenTechnicalPtcs(samplePtcs)
    expect(rows).toHaveLength(3)
    expect(rows[0]).toMatchObject({
      rowId: "11111111-1111-1111-1111-111111111111:0",
      contractorName: "BuildCorp International",
      criterion: "Schedule",
      refType: "MISSING_INFO",
    })
    expect(rows[2].rowId).toBe("22222222-2222-2222-2222-222222222222:0")
  })

  it("builds aggregate stats by status/refType/contractor", () => {
    const rows = flattenTechnicalPtcs(samplePtcs)
    const stats = buildTechnicalPtcStats(rows)

    expect(stats.total).toBe(3)
    expect(stats.pending).toBe(2)
    expect(stats.closed).toBe(1)
    expect(stats.completionPercent).toBe(33)
    expect(stats.byRefType).toEqual({
      "N/A": 1,
      MISSING_INFO: 1,
      INCOMPLETE: 1,
    })
    expect(stats.byContractor[0]).toMatchObject({
      contractorId: "11111111-1111-1111-1111-111111111111",
      total: 2,
      pending: 1,
      closed: 1,
    })
  })

  it("filters rows by contractor, refType, and search", () => {
    const rows = flattenTechnicalPtcs(samplePtcs)

    const byContractor = filterTechnicalPtcRows(rows, {
      contractorId: "11111111-1111-1111-1111-111111111111",
      refType: "all",
      search: "",
    })
    expect(byContractor).toHaveLength(2)

    const byRefType = filterTechnicalPtcRows(rows, {
      contractorId: "all",
      refType: "INCOMPLETE",
      search: "",
    })
    expect(byRefType).toHaveLength(1)
    expect(byRefType[0].criterion).toBe("Method Statement")

    const bySearch = filterTechnicalPtcRows(rows, {
      contractorId: "all",
      refType: "all",
      search: "clause 8",
    })
    expect(bySearch).toHaveLength(1)
    expect(bySearch[0].contractorName).toBe("Alpha Build")
  })
})

describe("normalizeTechnicalEvaluationPtcs", () => {
  it("normalizes legacy category based shape into technical shape", () => {
    const legacyShape = [
      {
        contractorId: "11111111-1111-1111-1111-111111111111",
        contractorName: "BuildCorp International",
        ptcs: [
          {
            id: "legacy-1",
            referenceSection: "Clause 4.2",
            queryDescription: "Please clarify missing appendix.",
            vendorResponse: "",
            status: "pending",
            category: "exclusions",
          },
        ],
      },
    ]

    const normalized = normalizeTechnicalEvaluationPtcs(legacyShape)
    expect(normalized).toEqual([
      {
        contractorId: "11111111-1111-1111-1111-111111111111",
        contractorName: "BuildCorp International",
        ptcs: [
          {
            criterion: "Clause 4.2",
            queryDescription: "Please clarify missing appendix.",
            refType: "MISSING_INFO",
            status: "pending",
            vendorResponse: "",
          },
        ],
      },
    ])
  })

  it("keeps valid technical shape unchanged and drops invalid rows", () => {
    const mixed = [
      {
        contractorId: "22222222-2222-2222-2222-222222222222",
        contractorName: "Alpha Build",
        ptcs: [
          {
            criterion: "Schedule",
            queryDescription: "Clarify sequencing assumptions.",
            refType: "INCOMPLETE",
            status: "closed",
            vendorResponse: "Response submitted.",
          },
          {
            criterion: "ShouldDrop",
            queryDescription: "",
            refType: "BAD_VALUE",
          },
        ],
      },
    ]

    const normalized = normalizeTechnicalEvaluationPtcs(mixed)
    expect(normalized).toEqual([
      {
        contractorId: "22222222-2222-2222-2222-222222222222",
        contractorName: "Alpha Build",
        ptcs: [
          {
            criterion: "Schedule",
            queryDescription: "Clarify sequencing assumptions.",
            refType: "INCOMPLETE",
            status: "closed",
            vendorResponse: "Response submitted.",
          },
        ],
      },
    ])
  })
})
