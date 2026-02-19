import { describe, expect, it } from "vitest"

import {
  extractCommReportFromArtifact,
  getCommJobStatusText,
  commRfpCompareResultSchema,
} from "@/lib/comm-job"

describe("comm-job", () => {
  it("unwraps report from compare artifact envelope", () => {
    const artifact = {
      analysisType: "compare",
      template_file: "template.xlsx",
      template_key: "org/pkg/comm_rfp/boq/template.xlsx",
      pte_file: null,
      pte_key: null,
      vendors: {
        A: { file: "A.xlsx", key: "org/pkg/comm_rfp/tender/A/A.xlsx" },
      },
      report: {
        vendors: ["A", "B"],
        summary: {},
        divisions: [],
      },
    }

    const report = extractCommReportFromArtifact(artifact)
    expect((report.vendors as string[]).length).toBe(2)
  })

  it("accepts direct report shape", () => {
    const report = extractCommReportFromArtifact({
      vendors: ["A", "B"],
      summary: {},
      divisions: [],
    })
    expect((report.vendors as string[])[0]).toBe("A")
  })

  it("unwraps nested result wrapper shape", () => {
    const report = extractCommReportFromArtifact({
      result: {
        analysis_type: "compare",
        report: {
          vendors: ["A", "B"],
          summary: {},
          divisions: [],
        },
      },
    })
    expect((report.vendors as string[])[1]).toBe("B")
  })

  it("maps status text from progress message first", () => {
    expect(getCommJobStatusText("in_progress", { message: "Extracting template" })).toBe(
      "Extracting template"
    )
  })

  it("maps fallback status text", () => {
    expect(getCommJobStatusText("pending")).toBe("Queued for processing")
  })

  it("parses compare envelope", () => {
    const parsed = commRfpCompareResultSchema.parse({
      analysisType: "compare",
      template_file: "template.xlsx",
      template_key: "org/pkg/comm_rfp/boq/template.xlsx",
      pte_file: null,
      pte_key: null,
      vendors: {
        A: { file: "A.xlsx", key: "org/pkg/comm_rfp/tender/A/A.xlsx" },
      },
      report: {
        vendors: ["A", "B"],
        summary: {},
        divisions: [],
      },
    })
    expect(parsed.analysisType).toBe("compare")
  })
})
