import type { ContractorPTCs, PTCCategory, PTCItem } from "./types"

// Sample query descriptions for each category
const queryTemplates: Record<PTCCategory, string[]> = {
  exclusions: [
    "Please confirm whether temporary works are included in your scope",
    "Clarify if testing and commissioning costs are excluded from your bid",
    "Confirm whether site office and facilities are part of your exclusions",
    "Please specify if warranty period maintenance is excluded",
    "Clarify exclusion of permits and approvals costs",
    "Confirm whether night work premium is excluded from rates",
    "Please clarify if scaffolding is excluded from structural works",
    "Confirm exclusion of hoisting and cranage for MEP works",
  ],
  deviations: [
    "Your proposed material differs from specification - please justify",
    "Deviation noted in installation methodology - provide technical justification",
    "Alternative product proposed requires approval - submit technical data",
    "Deviation from specified brand - confirm equivalency",
    "Non-compliant delivery schedule proposed - explain constraints",
    "Deviation in testing procedures - align with project requirements",
    "Alternative fixing method proposed - provide structural calculations",
    "Proposed subcontractor not pre-qualified - submit credentials",
  ],
  pricing_anomalies: [
    "Unit rate appears significantly below market rate - confirm accuracy",
    "Pricing inconsistency between similar items - please clarify",
    "Lump sum breakdown required for verification",
    "Rate appears to exclude overhead and profit - confirm inclusion",
    "Quantity discrepancy noted - verify against BOQ",
    "Provisional sum allocation unclear - provide breakdown",
    "Mobilization cost appears excessive - justify pricing",
    "Attendances pricing missing - confirm inclusion in rates",
  ],
  arithmetic_checks: [
    "Arithmetic error identified in section total - please correct",
    "Sum of line items does not match section subtotal",
    "Percentage calculation error in preliminaries",
    "Currency conversion discrepancy noted",
    "VAT calculation appears incorrect",
    "Quantity extension error - unit rate x quantity mismatch",
    "Carry forward error to summary page",
    "Discount application error in final total",
  ],
}

// Reference section templates
const referenceSections = [
  "Section 3.1.2",
  "Section 4.2.1",
  "Section 5.3.4",
  "Section 2.1.1",
  "Section 6.4.2",
  "Section 7.1.3",
  "Section 8.2.1",
  "Section 9.1.4",
  "Section 10.3.2",
  "Section 11.2.1",
  "BOQ Item 01.100.005",
  "BOQ Item 02.200.010",
  "BOQ Item 03.150.020",
  "BOQ Item 04.300.015",
  "Schedule B - Item 12",
  "Schedule C - Item 8",
  "Appendix A - Clause 4.2",
  "Appendix B - Clause 3.1",
]

function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function generatePTCsForContractor(
  contractorId: string,
  contractorName: string
): ContractorPTCs {
  // Random chance of having no PTCs (20%)
  if (Math.random() < 0.2) {
    return { contractorId, contractorName, ptcs: [] }
  }

  const ptcs: PTCItem[] = []
  const categories: PTCCategory[] = [
    "exclusions",
    "deviations",
    "pricing_anomalies",
    "arithmetic_checks",
  ]

  // Generate random number of PTCs (1-12)
  const numPTCs = Math.floor(Math.random() * 12) + 1

  for (let i = 0; i < numPTCs; i++) {
    const category = randomChoice(categories)
    const queryDescription = randomChoice(queryTemplates[category])

    ptcs.push({
      id: crypto.randomUUID(),
      referenceSection: randomChoice(referenceSections),
      queryDescription,
      vendorResponse: "", // Start empty
      status: "pending", // All start as pending
      category,
    })
  }

  return { contractorId, contractorName, ptcs }
}

export function generateMockPTCs(
  contractors: { id: string; name: string }[]
): ContractorPTCs[] {
  return contractors.map((c) => generatePTCsForContractor(c.id, c.name))
}
