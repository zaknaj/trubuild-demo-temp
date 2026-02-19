import { describe, expect, it } from "vitest"
import {
  buildCommercialPtcStats,
  filterCommercialPtcRows,
  flattenCommercialPtcs,
  normalizeCommercialEvaluationPtcs,
} from "@/lib/comm-ptc"
import type { ContractorPTCs } from "@/lib/types"

describe("normalizeCommercialEvaluationPtcs", () => {
  it("normalizes mixed data and drops invalid rows", () => {
    const input = [
      {
        contractorId: "vendor-a",
        contractorName: "Vendor A",
        ptcs: [
          {
            id: "ptc-1",
            referenceSection: "Division 01",
            queryDescription: "Please confirm exclusion scope.",
            vendorResponse: "",
            status: "pending",
            category: "exclusions",
          },
          {
            id: "ptc-2",
            referenceSection: "Division 02",
            queryDescription: "",
            vendorResponse: "",
            status: "pending",
            category: "deviations",
          },
        ],
      },
      {
        contractorId: "vendor-b",
        contractorName: "Vendor B",
        ptcs: [
          {
            id: "ptc-3",
            referenceSection: "Division 03",
            queryDescription: "Clarify arithmetic checks",
            vendorResponse: "Attached in appendix",
            status: "closed",
            category: "arithmetic_checks",
          },
        ],
      },
    ]

    const normalized = normalizeCommercialEvaluationPtcs(input)

    expect(normalized).toEqual([
      {
        contractorId: "vendor-a",
        contractorName: "Vendor A",
        ptcs: [
          {
            id: "ptc-1",
            referenceSection: "Division 01",
            queryDescription: "Please confirm exclusion scope.",
            vendorResponse: "",
            status: "pending",
            category: "exclusions",
          },
        ],
      },
      {
        contractorId: "vendor-b",
        contractorName: "Vendor B",
        ptcs: [
          {
            id: "ptc-3",
            referenceSection: "Division 03",
            queryDescription: "Clarify arithmetic checks",
            vendorResponse: "Attached in appendix",
            status: "closed",
            category: "arithmetic_checks",
          },
        ],
      },
    ])
  })
})

describe("commercial ptc view model helpers", () => {
  const samplePtcs: ContractorPTCs[] = [
    {
      contractorId: "vendor-a",
      contractorName: "Vendor A",
      ptcs: [
        {
          id: "a-1",
          referenceSection: "Division 01",
          queryDescription: "Clarify exclusions",
          vendorResponse: "",
          status: "pending",
          category: "exclusions",
        },
        {
          id: "a-2",
          referenceSection: "Division 02",
          queryDescription: "Explain arithmetic mismatch",
          vendorResponse: "Updated sheet sent",
          status: "closed",
          category: "arithmetic_checks",
        },
      ],
    },
    {
      contractorId: "vendor-b",
      contractorName: "Vendor B",
      ptcs: [
        {
          id: "b-1",
          referenceSection: "Division 03",
          queryDescription: "Rate variance is high",
          vendorResponse: "",
          status: "pending",
          category: "pricing_anomalies",
        },
      ],
    },
  ]

  it("flattens contractor grouped rows", () => {
    const rows = flattenCommercialPtcs(samplePtcs)
    expect(rows).toHaveLength(3)
    expect(rows[0]).toMatchObject({
      contractorName: "Vendor A",
      referenceSection: "Division 01",
      category: "exclusions",
    })
  })

  it("builds aggregate stats", () => {
    const rows = flattenCommercialPtcs(samplePtcs)
    const stats = buildCommercialPtcStats(rows)
    expect(stats.total).toBe(3)
    expect(stats.pending).toBe(2)
    expect(stats.closed).toBe(1)
    expect(stats.completionPercent).toBe(33)
    expect(stats.byCategory).toEqual({
      exclusions: 1,
      deviations: 0,
      pricing_anomalies: 1,
      arithmetic_checks: 1,
    })
  })

  it("filters by contractor category status and search", () => {
    const rows = flattenCommercialPtcs(samplePtcs)

    const filtered = filterCommercialPtcRows(rows, {
      contractorId: "vendor-a",
      category: "all",
      status: "all",
      search: "",
    })
    expect(filtered).toHaveLength(2)

    const byCategory = filterCommercialPtcRows(rows, {
      contractorId: "all",
      category: "pricing_anomalies",
      status: "all",
      search: "",
    })
    expect(byCategory).toHaveLength(1)

    const byStatus = filterCommercialPtcRows(rows, {
      contractorId: "all",
      category: "all",
      status: "closed",
      search: "",
    })
    expect(byStatus).toHaveLength(1)

    const bySearch = filterCommercialPtcRows(rows, {
      contractorId: "all",
      category: "all",
      status: "all",
      search: "variance",
    })
    expect(bySearch).toHaveLength(1)
    expect(bySearch[0].contractorId).toBe("vendor-b")
  })
})
