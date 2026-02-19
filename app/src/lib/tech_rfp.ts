export type TechRfpEvidenceItem = {
  text: string
  source: string
  pageNumber: number | null
}

export type TechRfpPtc = {
  queryDescription: string
  refType: TechRfpPtcRefType
}

export type TechRfpPtcRefType = "N/A" | "MISSING_INFO" | "INCOMPLETE"
export type TechRfpPtcValue = TechRfpPtc | "NA"

export type TechRfpEvaluationCriterion = {
  criteria: string
  ptc: TechRfpPtcValue
  grade: string
  score: number
  result: number
  evidence: TechRfpEvidenceItem[]
  clientSummary: string
}

export type TechRfpScope = {
  scopeName: string
  scopeTotal: number
  evaluationBreakdown: TechRfpEvaluationCriterion[]
}

export type TechRfpTenderEvaluation = {
  scopes: TechRfpScope[]
  totalScore: number
}

export type TechRfpLlmResponsePayload = {
  grade: string
  reasoning: TechRfpEvidenceItem[]
  score: number
}

export type TechRfpLlmResponse = {
  persona: string
  response: TechRfpLlmResponsePayload
}

export type TechRfpRawLlmResponse = {
  criterion: string
  llm_responses: TechRfpLlmResponse[]
}

export type TechRfpContractorData = {
  contractorName: string
  tenderEvaluation: TechRfpTenderEvaluation
  raw_llm_responses: TechRfpRawLlmResponse[]
}

export type TechRfpResult = {
  result: {
    tenderReport: TechRfpContractorData[]
  }
}

export type TechRfpResultArtifact = {
  tenderReport: TechRfpContractorData[]
}

// Engine-aligned technical PTC projection for technical_evaluation.data.ptcs
export type TechRfpTechnicalEvaluationPtcStatus = "pending" | "closed"

export type TechRfpTechnicalEvaluationPtcItem = {
  criterion: string
  queryDescription: string
  refType: TechRfpPtcRefType
  status?: TechRfpTechnicalEvaluationPtcStatus
  vendorResponse?: string
}

export type TechRfpTechnicalEvaluationContractorPtcs = {
  contractorId: string
  contractorName: string
  ptcs: TechRfpTechnicalEvaluationPtcItem[]
}

export type TechRfpTechnicalEvaluationPtcsData = {
  ptcs: TechRfpTechnicalEvaluationContractorPtcs[]
}

export type TechRfpEvaluationBreakdownType = {
  weight: number
  description: string
}

export type TechRfpEvaluationCategoryType = Record<
  string,
  TechRfpEvaluationBreakdownType
>

export type TechRfpEvaluationCriteriaType = Record<
  string,
  TechRfpEvaluationCategoryType
>

export type TechRfpJobPayloadType = {
  package_id: string
  evaluation_criteria: TechRfpEvaluationCriteriaType
  user_name: string
  metadata: {
    package_name: string
  }
}

export type TechRfpJobType = {
  id: string
  company_id: string
  user_id: string
  type: "tech_rfp_analysis"
  payload: TechRfpJobPayloadType
}

export type TechRfpJobStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "cancelled"
