import type { listProjectsFn } from "@/fn/projects"
import type {
  getProjectMembersFn,
  getOrgMembersFn,
  getPackageMembersFn,
  getOrgPendingInvitesFn,
} from "@/fn/members"
import type { getPackageWithAssetsFn } from "@/fn/packages"
import type { TechRfpTechnicalEvaluationContractorPtcs } from "@/lib/tech_rfp"

// Inferred from server function return types - stays in sync automatically
export type Project = Awaited<ReturnType<typeof listProjectsFn>>[number]
export type Member = Awaited<ReturnType<typeof getProjectMembersFn>>[number]
export type OrgMember = Awaited<ReturnType<typeof getOrgMembersFn>>[number]
export type OrgPendingInvite = Awaited<
  ReturnType<typeof getOrgPendingInvitesFn>
>[number]
export type PackageMember = Awaited<
  ReturnType<typeof getPackageMembersFn>
>[number]
export type PackageDetail = Awaited<ReturnType<typeof getPackageWithAssetsFn>>
export type Package = PackageDetail["package"]
export type Asset = PackageDetail["assets"][number]

// BOQ / Commercial Evaluation types
export interface BOQLineItem {
  id: string
  code: string // e.g., "01.100.001"
  description: string
  quantity: number
  unit: string // e.g., "m2", "pcs", "kg"
}

export interface BOQSection {
  id: string
  code: string // e.g., "01.100"
  name: string
  lineItems: BOQLineItem[]
}

export interface BOQDivision {
  id: string
  code: string // e.g., "01", "02"
  name: string // e.g., "General Requirements"
  sections: BOQSection[]
}

export interface BOQData {
  divisions: BOQDivision[]
}

export interface ArithmeticError {
  submitted: number // The price the contractor submitted
  calculated: number // What it should be (qty × unit rate)
}

export interface ContractorBid {
  contractorId: string
  contractorName: string
  prices: Record<string, number | null> // lineItemId -> price (null = missing/unpriced)
  includedItems: string[] // item IDs marked as "Included" by this contractor
  arithmeticErrors: Record<string, ArithmeticError> // itemId -> error details
  totalAmount: number
}

// Normalization settings for BOQ comparison
export interface NormalizationSettings {
  normalizeUnpriced: boolean // default: true
  normalizeArithmeticErrors: boolean // default: true
  algorithm: "median" | "lowest" // default: 'median'
}

// Custom price overrides - key format: `${contractorId}-${itemId}`
export type CustomOverrides = Record<string, number>

export interface CommercialEvaluationData {
  boq: BOQData
  contractors: ContractorBid[]
  ptcs?: ContractorPTCs[]
}

// PTC (Post Tender Clarifications) types
export type PTCStatus = "pending" | "closed"
export type PTCCategory =
  | "exclusions"
  | "deviations"
  | "pricing_anomalies"
  | "arithmetic_checks"

export interface PTCItem {
  id: string
  referenceSection: string
  queryDescription: string
  vendorResponse: string
  status: PTCStatus
  category: PTCCategory
}

export interface ContractorPTCs {
  contractorId: string
  contractorName: string
  ptcs: PTCItem[]
}

// Technical Evaluation types
export interface TechnicalEvaluationData {
  ptcs?: TechRfpTechnicalEvaluationContractorPtcs[]
  // Other technical evaluation data can be added here
  [key: string]: unknown
}

// Commercial RFP queue contract types
export type JobRunStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "cancelled"

export interface CommRfpJobProgress {
  stage?: string
  message?: string
  source_file?: string
  vendor_count?: number
}

export interface CommRfpJobStatus {
  status: JobRunStatus
  runId: string | null
  error: string | null
  progress: CommRfpJobProgress | null
}

export interface CommRfpVendorArtifact {
  file: string
  key: string
}

export interface CommRfpCompareResultEnvelope {
  analysisType: "compare"
  template_file: string
  template_key: string
  pte_file: string | null
  pte_key: string | null
  vendors: Record<string, CommRfpVendorArtifact>
  report: Record<string, unknown>
}
