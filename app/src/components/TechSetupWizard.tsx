import { useState, useEffect } from "react"
import { useNavigate } from "@tanstack/react-router"
import { useQueryClient } from "@tanstack/react-query"
import {
  createTechnicalEvaluationFn,
  updateTechnicalEvaluationFn,
} from "@/fn/evaluations"
import {
  createTechRfpAnalysisJobFn,
  createTechRfpEvaluationExtractJobFn,
  getTechRfpEvaluationExtractJobStatusFn,
  getTechRfpEvaluationExtractResultFn,
  createTechRfpGenerateEvalJobFn,
  getTechRfpGenerateEvalJobStatusFn,
  getTechRfpGenerateEvalResultFn,
  writeTechRfpEvaluationJsonFn,
} from "@/fn/jobs"
import type {
  TechRfpEvaluationCriteriaType,
  TechRfpTechnicalEvaluationContractorPtcs,
} from "@/lib/tech_rfp"
import { resolveCriteriaJob } from "@/lib/criteria-job"
import { queryKeys } from "@/lib/query-options"
import useStore from "@/lib/store"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { UploadZone, type UploadedFile } from "@/components/ui/upload-zone"
import { StepTitle } from "@/components/ui/step-title"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Trash2, UserIcon, Sparkles, Download, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

// Types - exported for use in main tech evaluation file
export interface Breakdown {
  id: string
  title: string
  description: string
  weight: number
}

export interface Scope {
  id: string
  name: string
  breakdowns: Breakdown[]
}

export interface EvidenceFile {
  id: string
  name: string
  fakeUrl: string // Simulated upload URL
}

export interface LineReference {
  fileName: string
  startLine: number
  endLine?: number // Optional - if undefined, single line
}

export interface Evidence {
  id: string
  text: string
  source: "auto" | "manual"
  files: EvidenceFile[]
  lineReference?: LineReference
}

export interface ScoreData {
  score: number
  comment?: string
  approved: boolean
  evidence: Evidence[]
}

export interface TechnicalEvaluationData {
  status: "setup" | "analyzing" | "ready" | "review_complete"
  setupStep: 1 | 2 | 3
  documentsUploaded: boolean
  analysis?: {
    jobId: string
    status: "queued" | "running" | "completed" | "failed"
    error?: string
  }
  criteria: {
    scopes: Scope[]
  }
  proposalsUploaded: string[]
  scores: Record<string, Record<string, ScoreData>>
  ptcs?: TechRfpTechnicalEvaluationContractorPtcs[]
}

// Generate fresh default criteria with new UUIDs
function generateDefaultCriteria(): Scope[] {
  return [
    {
      id: crypto.randomUUID(),
      name: "Technical Capability",
      breakdowns: [
        {
          id: crypto.randomUUID(),
          title: "Experience & Track Record",
          description: "Past project experience relevant to this scope",
          weight: 15,
        },
        {
          id: crypto.randomUUID(),
          title: "Technical Approach",
          description: "Methodology and technical solution proposed",
          weight: 20,
        },
      ],
    },
    {
      id: crypto.randomUUID(),
      name: "Resources & Team",
      breakdowns: [
        {
          id: crypto.randomUUID(),
          title: "Key Personnel",
          description: "Qualifications of proposed team members",
          weight: 15,
        },
        {
          id: crypto.randomUUID(),
          title: "Equipment & Resources",
          description: "Available equipment and resource capacity",
          weight: 10,
        },
      ],
    },
    {
      id: crypto.randomUUID(),
      name: "Project Management",
      breakdowns: [
        {
          id: crypto.randomUUID(),
          title: "Schedule & Planning",
          description: "Proposed timeline and project plan",
          weight: 20,
        },
        {
          id: crypto.randomUUID(),
          title: "Risk Management",
          description: "Risk identification and mitigation strategies",
          weight: 10,
        },
        {
          id: crypto.randomUUID(),
          title: "Quality Assurance",
          description: "QA/QC procedures and standards",
          weight: 10,
        },
      ],
    },
  ]
}

function toSafeWeight(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return 0
}

function normalizeExtractedCriteria(raw: unknown): Scope[] {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return []
  }

  const scopes: Scope[] = []

  for (const [rawScopeName, rawBreakdowns] of Object.entries(raw)) {
    if (!rawBreakdowns || typeof rawBreakdowns !== "object") {
      continue
    }

    const scopeName = rawScopeName.trim() || "Untitled Scope"
    const breakdowns: Breakdown[] = []

    for (const [rawTitle, rawDetails] of Object.entries(rawBreakdowns)) {
      const title = rawTitle.trim() || "Untitled Criteria"
      let description = ""
      let weight = 0

      if (typeof rawDetails === "number" || typeof rawDetails === "string") {
        weight = toSafeWeight(rawDetails)
      } else if (rawDetails && typeof rawDetails === "object") {
        const details = rawDetails as {
          weight?: unknown
          description?: unknown
        }
        weight = toSafeWeight(details.weight)
        if (typeof details.description === "string") {
          description = details.description.trim()
        }
      }

      breakdowns.push({
        id: crypto.randomUUID(),
        title,
        description,
        weight,
      })
    }

    if (breakdowns.length > 0) {
      scopes.push({
        id: crypto.randomUUID(),
        name: scopeName,
        breakdowns,
      })
    }
  }

  return scopes
}

function normalizeEditableCriteria(criteria: Scope[]): Scope[] {
  return criteria
    .map((scope) => ({
      ...scope,
      name: scope.name.trim(),
      breakdowns: scope.breakdowns.map((breakdown) => ({
        ...breakdown,
        title: breakdown.title.trim(),
        description: breakdown.description.trim(),
        weight: toSafeWeight(breakdown.weight),
      })),
    }))
    .filter((scope) => scope.breakdowns.length > 0)
}

function toTechRfpEvaluationCriteria(
  scopes: Scope[]
): TechRfpEvaluationCriteriaType {
  const criteria: TechRfpEvaluationCriteriaType = {}
  for (const scope of scopes) {
    const scopeName = scope.name.trim()
    if (!scopeName) continue
    criteria[scopeName] = {}
    for (const breakdown of scope.breakdowns) {
      const title = breakdown.title.trim()
      if (!title) continue
      criteria[scopeName][title] = {
        weight: toSafeWeight(breakdown.weight),
        description: breakdown.description.trim(),
      }
    }
  }
  return criteria
}

// ============================================================================
// Setup Wizard
// ============================================================================

export function TechSetupWizard({
  open,
  onOpenChange,
  packageId,
  contractors,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  packageId: string
  contractors: Array<{ id: string; name: string }>
}) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const setTechRound = useStore((s) => s.setTechRound)

  // Step state
  const [step, setStep] = useState(1)
  const [isExtracting, setIsExtracting] = useState(false)
  const [criteriaProcessingMode, setCriteriaProcessingMode] = useState<
    "extract" | "generate" | null
  >(null)
  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false)
  const [uploadingCount, setUploadingCount] = useState(0)

  // Step 1 state
  const [rfpFile, setRfpFile] = useState<UploadedFile[]>([])
  const [criteriaMode, setCriteriaMode] = useState<"upload" | "ai" | null>(null)
  const [criteriaFile, setCriteriaFile] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<
    Record<string, UploadedFile[]>
  >({})

  // Step 2 state (criteria editing)
  const [criteria, setCriteria] = useState<Scope[]>(() =>
    generateDefaultCriteria()
  )

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setStep(1)
      setIsExtracting(false)
      setCriteriaProcessingMode(null)
      setUploadingCount(0)
      setRfpFile([])
      setCriteriaMode(null)
      setCriteriaFile([])
      setVendorFiles({})
      setCriteria(generateDefaultCriteria())
    }
  }, [open])

  // Step 1 validation
  const isRfpDone = rfpFile.length > 0
  const isCriteriaDone =
    criteriaMode === "ai" ||
    (criteriaMode === "upload" && criteriaFile.length > 0)
  const vendorsWithFiles = Object.entries(vendorFiles).filter(
    ([, files]) => files.length > 0
  ).length
  const isVendorsDone = vendorsWithFiles >= 2
  const isUploading = uploadingCount > 0

  const canProceedStep1 =
    isRfpDone && isCriteriaDone && isVendorsDone && !isUploading

  // Step 2 validation
  const totalWeight = criteria.reduce(
    (sum, scope) =>
      sum + scope.breakdowns.reduce((s, b) => s + (b.weight || 0), 0),
    0
  )
  const canProceedStep2 = totalWeight === 100

  // Get contractor IDs with files
  const proposalsUploaded = Object.entries(vendorFiles)
    .filter(([, files]) => files.length > 0)
    .map(([id]) => id)

  const handleNext = async () => {
    if (step === 1) {
      if (criteriaMode === "ai") {
        setIsExtracting(true)
        setCriteriaProcessingMode("generate")
        try {
          const result = await resolveCriteriaJob({
            createJob: () =>
              createTechRfpGenerateEvalJobFn({
                data: { packageId },
              }),
            getStatus: (jobId) =>
              getTechRfpGenerateEvalJobStatusFn({
                data: { jobId },
              }),
            getResult: (jobId) =>
              getTechRfpGenerateEvalResultFn({
                data: { jobId },
              }),
            normalize: normalizeExtractedCriteria,
          })

          if (result.status === "completed") {
            setCriteria(result.scopes)
            setStep(2)
          } else if (result.status === "empty") {
            toast.error(
              "No valid criteria were generated. Please try again or upload your own criteria."
            )
          } else if (result.status === "cancelled") {
            toast.error("AI criteria generation was cancelled. Please try again.")
          } else {
            toast.error(
              "AI criteria generation failed. Please try again or upload your own criteria."
            )
          }
        } catch (error) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Failed to start AI criteria generation"
          )
        } finally {
          setIsExtracting(false)
          setCriteriaProcessingMode(null)
        }
        return
      }

      if (criteriaMode !== "upload") {
        toast.error("Please choose how you want to provide evaluation criteria.")
        return
      }

      setIsExtracting(true)
      setCriteriaProcessingMode("extract")
      try {
        const result = await resolveCriteriaJob({
          createJob: () =>
            createTechRfpEvaluationExtractJobFn({
              data: { packageId },
            }),
          getStatus: (jobId) =>
            getTechRfpEvaluationExtractJobStatusFn({
              data: { jobId },
            }),
          getResult: (jobId) =>
            getTechRfpEvaluationExtractResultFn({
              data: { jobId },
            }),
          normalize: normalizeExtractedCriteria,
        })

        if (result.status === "completed") {
          setCriteria(result.scopes)
          setStep(2)
        } else if (result.status === "empty") {
          toast.error(
            "No valid criteria found in extraction result. Please try another file."
          )
        } else if (result.status === "cancelled") {
          toast.error("Criteria extraction was cancelled. Please try again.")
        } else {
          toast.error(
            "Criteria extraction failed. Please try again or use a different file."
          )
        }
      } catch (error) {
        toast.error(
          error instanceof Error
            ? error.message
            : "Failed to start criteria extraction"
        )
      } finally {
        setIsExtracting(false)
        setCriteriaProcessingMode(null)
      }
    }
  }

  const handleStartAnalysis = async () => {
    const normalizedCriteria = normalizeEditableCriteria(criteria)
    if (normalizedCriteria.length === 0) {
      toast.error("Please add at least one scope with criteria before starting.")
      return
    }

    const normalizedTotalWeight = normalizedCriteria.reduce(
      (sum, scope) =>
        sum + scope.breakdowns.reduce((scopeSum, b) => scopeSum + b.weight, 0),
      0
    )
    if (normalizedTotalWeight !== 100) {
      toast.error("Criteria weights must total 100% before starting analysis.")
      return
    }

    let createdEvaluationId: string | null = null
    let createdEvaluationRoundName: string | null = null
    setIsStartingAnalysis(true)
    try {
      setCriteria(normalizedCriteria)
      const initialEvaluationData: TechnicalEvaluationData = {
        status: "setup",
        setupStep: 3,
        documentsUploaded: true,
        criteria: { scopes: normalizedCriteria },
        proposalsUploaded,
        scores: {},
      }

      const createdEvaluation = (await createTechnicalEvaluationFn({
        data: { packageId, data: initialEvaluationData },
      })) as { id: string; roundName: string }
      createdEvaluationId = createdEvaluation.id
      createdEvaluationRoundName = createdEvaluation.roundName
      setTechRound(packageId, createdEvaluation.id)
      await queryClient.invalidateQueries({
        queryKey: queryKeys.package.technicalEvaluations(packageId),
      })

      const evaluationCriteria = toTechRfpEvaluationCriteria(normalizedCriteria)
      await writeTechRfpEvaluationJsonFn({
        data: {
          packageId,
          evaluationCriteria,
        },
      })

      const { jobId } = await createTechRfpAnalysisJobFn({
        data: {
          packageId,
          evaluationCriteria,
        },
      })

      const evaluationData: TechnicalEvaluationData = {
        ...initialEvaluationData,
        status: "analyzing",
        analysis: {
          jobId,
          status: "queued",
        },
      }

      await updateTechnicalEvaluationFn({
        data: {
          evaluationId: createdEvaluation.id,
          data: evaluationData,
        },
      })
      await queryClient.invalidateQueries({
        queryKey: queryKeys.package.technicalEvaluations(packageId),
      })
      toast.success(`${createdEvaluation.roundName} created`)
      navigate({ to: "/package/$id/tech", params: { id: packageId } })
      onOpenChange(false)
    } catch (error) {
      if (createdEvaluationId) {
        setTechRound(packageId, createdEvaluationId)
        navigate({ to: "/package/$id/tech", params: { id: packageId } })
      }
      toast.error(
        createdEvaluationId
          ? `Round "${createdEvaluationRoundName ?? "New round"}" was created, but failed to start analysis. Please retry from the technical page.`
          : error instanceof Error
            ? error.message
            : "Failed to start technical analysis"
      )
    } finally {
      setIsStartingAnalysis(false)
    }
  }

  const handleVendorFilesChange = (vendorId: string, files: UploadedFile[]) => {
    setVendorFiles((prev) => ({ ...prev, [vendorId]: files }))
  }

  const handleUploadingChange = (isUploading: boolean) => {
    setUploadingCount((prev) => prev + (isUploading ? 1 : -1))
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        {isExtracting ? (
          // Extracting state
          <div className="flex flex-col items-center justify-center py-16 gap-4">
            <Loader2 className="size-12 animate-spin text-primary" />
            <div className="text-center">
              <p className="font-medium text-lg">
                {criteriaProcessingMode === "generate"
                  ? "Generating Evaluation Criteria"
                  : "Extracting Evaluation Criteria"}
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                {criteriaProcessingMode === "generate"
                  ? "Analyzing your RFP to build a weighted rubric..."
                  : "Analyzing your documents..."}
              </p>
            </div>
          </div>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>
                {step === 1 && "Step 1: Upload Documents"}
                {step === 2 && "Step 2: Review Evaluation Criteria"}
              </DialogTitle>
              <DialogDescription>
                {step === 1 &&
                  "Upload the required documents and vendor proposals for technical evaluation."}
                {step === 2 &&
                  "Review and adjust the evaluation criteria generated from your documents."}
              </DialogDescription>
            </DialogHeader>

            <div className="flex-1 overflow-y-auto py-4">
              {step === 1 && (
                <Step1Documents
                  packageId={packageId}
                  contractors={contractors}
                  rfpFile={rfpFile}
                  onRfpFileChange={setRfpFile}
                  criteriaMode={criteriaMode}
                  onCriteriaModeChange={setCriteriaMode}
                  criteriaFile={criteriaFile}
                  onCriteriaFileChange={setCriteriaFile}
                  vendorFiles={vendorFiles}
                  onVendorFilesChange={handleVendorFilesChange}
                  vendorsWithFiles={vendorsWithFiles}
                  onUploadingChange={handleUploadingChange}
                />
              )}
              {step === 2 && (
                <Step2Criteria
                  criteria={criteria}
                  onChange={setCriteria}
                  totalWeight={totalWeight}
                />
              )}
            </div>

            <DialogFooter>
              {step > 1 && (
                <Button variant="outline" onClick={() => setStep(step - 1)}>
                  Back
                </Button>
              )}
              {step === 1 && (
                <Button
                  onClick={handleNext}
                  disabled={!canProceedStep1 || isExtracting}
                >
                  {isUploading ? "Uploading..." : "Next"}
                </Button>
              )}
              {step === 2 && (
                <Button
                  onClick={handleStartAnalysis}
                  disabled={!canProceedStep2 || isStartingAnalysis}
                >
                  {isStartingAnalysis
                    ? "Starting..."
                    : totalWeight !== 100
                      ? `Weights: ${totalWeight}% (need 100%)`
                      : "Start Analysis"}
                </Button>
              )}
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

// ============================================================================
// Step 1: Document Upload
// ============================================================================

function Step1Documents({
  packageId,
  contractors,
  rfpFile,
  onRfpFileChange,
  criteriaMode,
  onCriteriaModeChange,
  criteriaFile,
  onCriteriaFileChange,
  vendorFiles,
  onVendorFilesChange,
  vendorsWithFiles,
  onUploadingChange,
}: {
  packageId: string
  contractors: Array<{ id: string; name: string }>
  rfpFile: UploadedFile[]
  onRfpFileChange: (files: UploadedFile[]) => void
  criteriaMode: "upload" | "ai" | null
  onCriteriaModeChange: (mode: "upload" | "ai") => void
  criteriaFile: UploadedFile[]
  onCriteriaFileChange: (files: UploadedFile[]) => void
  vendorFiles: Record<string, UploadedFile[]>
  onVendorFilesChange: (vendorId: string, files: UploadedFile[]) => void
  vendorsWithFiles: number
  onUploadingChange: (isUploading: boolean) => void
}) {
  const isRfpDone = rfpFile.length > 0
  const isCriteriaDone =
    criteriaMode === "ai" ||
    (criteriaMode === "upload" && criteriaFile.length > 0)

  return (
    <div className="space-y-8 px-1">
      {/* Section 1: RFP */}
      <div className="space-y-3">
        <StepTitle
          title="Request For Proposals (RFP)"
          complete={isRfpDone}
          required
        />
        <UploadZone
          files={rfpFile}
          onFilesChange={onRfpFileChange}
          packageId={packageId}
          category="rfp"
          storagePathMode="tech_rfp_rfp"
          onUploadingChange={onUploadingChange}
          accept=".pdf,.doc,.docx"
        />
      </div>

      {/* Section 2: Evaluation Criteria */}
      <div className="space-y-3">
        <StepTitle
          title="Evaluation Criteria"
          complete={isCriteriaDone}
          required
        />

        <RadioGroup
          value={criteriaMode ?? ""}
          onValueChange={(v) => onCriteriaModeChange(v as "upload" | "ai")}
          className="space-y-3"
        >
          {/* Upload option */}
          <div
            className={cn(
              "flex items-start gap-3 p-4 rounded-lg border transition-colors cursor-pointer",
              criteriaMode === "upload"
                ? "border-primary bg-primary/5"
                : "hover:bg-muted/50"
            )}
            onClick={() => onCriteriaModeChange("upload")}
          >
            <RadioGroupItem
              value="upload"
              id="criteria-upload"
              className="mt-1"
            />
            <div className="flex-1 space-y-3">
              <div className="flex items-center justify-between">
                <Label
                  htmlFor="criteria-upload"
                  className="cursor-pointer font-medium"
                >
                  Upload your own
                </Label>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    toast.success("Template download started")
                  }}
                  className="text-sm text-primary hover:underline flex items-center gap-1"
                >
                  <Download className="size-3.5" />
                  Download template
                </a>
              </div>
              {criteriaMode === "upload" && (
                <UploadZone
                  files={criteriaFile}
                  onFilesChange={onCriteriaFileChange}
                  packageId={packageId}
                  category="criteria"
                  storagePathMode="tech_rfp_evaluation"
                  onUploadingChange={onUploadingChange}
                  accept=".pdf,.doc,.docx,.xlsx"
                  compact
                />
              )}
            </div>
          </div>

          {/* AI option */}
          <div
            className={cn(
              "flex items-start gap-3 p-4 rounded-lg border transition-colors cursor-pointer",
              criteriaMode === "ai"
                ? "border-primary bg-primary/5"
                : "hover:bg-muted/50"
            )}
            onClick={() => onCriteriaModeChange("ai")}
          >
            <RadioGroupItem value="ai" id="criteria-ai" className="mt-1" />
            <div className="flex-1">
              <Label
                htmlFor="criteria-ai"
                className="cursor-pointer font-medium flex items-center gap-2"
              >
                <Sparkles className="size-4 text-amber-500" />
                Generate with AI
              </Label>
              <p className="text-sm text-muted-foreground mt-1">
                We'll analyze your RFP and generate evaluation criteria
                automatically
              </p>
            </div>
          </div>
        </RadioGroup>
      </div>

      {/* Section 3: Vendor Proposals */}
      <div className="space-y-3">
        <StepTitle
          title={`Vendor Proposals (${vendorsWithFiles}/${contractors.length} vendors have files)`}
          complete={vendorsWithFiles >= 2}
          required
        />

        {contractors.length === 0 ? (
          <div className="text-center py-8 border rounded-lg">
            <UserIcon className="size-10 mx-auto text-muted-foreground mb-2" />
            <p className="text-muted-foreground">
              No contractors added to this package yet.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {contractors.map((contractor) => {
              const files = vendorFiles[contractor.id] ?? []
              const hasFiles = files.length > 0

              return (
                <div
                  key={contractor.id}
                  className={cn(
                    "rounded-lg border p-4 transition-colors",
                    hasFiles &&
                      "border-emerald-500 bg-emerald-50/50"
                  )}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="flex items-center justify-center w-8 h-8 rounded-md bg-muted">
                      <UserIcon size={16} className="text-muted-foreground" />
                    </div>
                    <span className="font-medium">{contractor.name}</span>
                  </div>
                  <UploadZone
                    files={files}
                    onFilesChange={(newFiles) =>
                      onVendorFilesChange(contractor.id, newFiles)
                    }
                    packageId={packageId}
                    category="vendor_proposal"
                    storagePathMode="tech_rfp_tender"
                    contractorId={contractor.id}
                    vendorName={contractor.name}
                    onUploadingChange={onUploadingChange}
                    multiple
                    accept=".pdf,.doc,.docx,.xlsx"
                    compact
                  />
                </div>
              )
            })}

            {vendorsWithFiles < 2 && (
              <p className="text-sm text-amber-600">
                At least 2 vendors must have files to proceed
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// Step 2: Criteria Review
// ============================================================================

// Inline editable text input that looks like plain text
function InlineInput({
  value,
  onChange,
  placeholder,
  className,
  inputClassName,
}: {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  inputClassName?: string
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={cn(
        "bg-transparent border border-transparent rounded px-1.5 py-0.5 -mx-1.5 -my-0.5 outline-none transition-colors",
        "hover:border-border/50",
        "focus:border-ring focus:ring-2 focus:ring-ring/20",
        "placeholder:text-muted-foreground/50",
        inputClassName,
        className
      )}
    />
  )
}

// Weight input styled as a pill
function WeightInput({
  value,
  onChange,
}: {
  value: number
  onChange: (value: number) => void
}) {
  return (
    <div className="inline-flex items-center bg-muted/50 rounded-lg px-3 py-1.5 h-9 min-w-18 justify-center transition-colors focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2">
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value) || 0)}
        className="bg-transparent w-8 text-center outline-none text-sm font-medium [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />
      <span className="text-sm text-muted-foreground">%</span>
    </div>
  )
}

// Display-only weight total styled to match WeightInput
function WeightTotal({ value }: { value: number }) {
  return (
    <div className="inline-flex items-center bg-transparent rounded-lg px-3 py-1.5 h-9 min-w-18 justify-center">
      <span className="text-sm font-medium text-muted-foreground">
        {value}%
      </span>
    </div>
  )
}

function Step2Criteria({
  criteria,
  onChange,
  totalWeight,
}: {
  criteria: Scope[]
  onChange: (criteria: Scope[]) => void
  totalWeight: number
}) {
  const addScope = () => {
    onChange([
      ...criteria,
      {
        id: crypto.randomUUID(),
        name: "New Scope",
        breakdowns: [],
      },
    ])
  }

  const updateScope = (scopeId: string, updates: Partial<Scope>) => {
    onChange(criteria.map((s) => (s.id === scopeId ? { ...s, ...updates } : s)))
  }

  const removeScope = (scopeId: string) => {
    onChange(criteria.filter((s) => s.id !== scopeId))
  }

  const addBreakdown = (scopeId: string) => {
    onChange(
      criteria.map((s) =>
        s.id === scopeId
          ? {
              ...s,
              breakdowns: [
                ...s.breakdowns,
                {
                  id: crypto.randomUUID(),
                  title: "New Criteria",
                  description: "",
                  weight: 0,
                },
              ],
            }
          : s
      )
    )
  }

  const updateBreakdown = (
    scopeId: string,
    breakdownId: string,
    updates: Partial<Breakdown>
  ) => {
    onChange(
      criteria.map((s) =>
        s.id === scopeId
          ? {
              ...s,
              breakdowns: s.breakdowns.map((b) =>
                b.id === breakdownId ? { ...b, ...updates } : b
              ),
            }
          : s
      )
    )
  }

  const removeBreakdown = (scopeId: string, breakdownId: string) => {
    onChange(
      criteria.map((s) =>
        s.id === scopeId
          ? {
              ...s,
              breakdowns: s.breakdowns.filter((b) => b.id !== breakdownId),
            }
          : s
      )
    )
  }

  return (
    <div className="space-y-8 px-4">
      {/* Header */}
      <div className="flex items-baseline justify-between">
        <div>
          <h3 className="text-lg font-semibold">Review Evaluation Criteria</h3>
          <button className="text-sm text-muted-foreground hover:text-foreground underline underline-offset-2">
            Revert all changes
          </button>
        </div>
        <span
          className={cn(
            "text-2xl font-semibold",
            totalWeight === 100 ? "text-foreground" : "text-amber-600"
          )}
        >
          {totalWeight}%
        </span>
      </div>

      {/* Scopes */}
      <div className="space-y-8">
        {criteria.map((scope) => {
          const scopeWeight = scope.breakdowns.reduce(
            (sum, b) => sum + (b.weight || 0),
            0
          )
          return (
            <div key={scope.id} className="space-y-1">
              {/* Scope header */}
              <div className="group flex items-center justify-between gap-4">
                <div className="flex-1 flex items-center gap-2 min-w-0">
                  <InlineInput
                    value={scope.name}
                    onChange={(value) => updateScope(scope.id, { name: value })}
                    className="font-semibold text-base flex-1"
                    inputClassName="w-full"
                  />
                  <button
                    onClick={() => removeScope(scope.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-muted rounded shrink-0"
                  >
                    <Trash2 className="size-4 text-muted-foreground" />
                  </button>
                </div>
                <WeightTotal value={scopeWeight} />
              </div>

              {/* Breakdowns with tree structure */}
              <div className="relative ml-2">
                {scope.breakdowns.map((breakdown, idx) => {
                  const isLast = idx === scope.breakdowns.length - 1
                  return (
                    <div
                      key={breakdown.id}
                      className="group relative flex items-start gap-4 py-3"
                    >
                      {/* Vertical line - extends down unless this is the last item */}
                      {!isLast && (
                        <div className="absolute left-0 top-0 bottom-0 w-px bg-border" />
                      )}
                      {/* Vertical line segment for the last item - only goes to the connector */}
                      {isLast && (
                        <div className="absolute left-0 top-0 h-6 w-px bg-border" />
                      )}

                      {/* Horizontal connector */}
                      <div className="absolute left-0 top-6 w-4 h-px bg-border" />

                      {/* Content */}
                      <div className="flex-1 ml-6 space-y-0.5 min-w-0">
                        <div className="flex items-center gap-2">
                          <InlineInput
                            value={breakdown.title}
                            onChange={(value) =>
                              updateBreakdown(scope.id, breakdown.id, {
                                title: value,
                              })
                            }
                            placeholder="Title"
                            className="font-medium flex-1"
                            inputClassName="w-full"
                          />
                          <button
                            onClick={() =>
                              removeBreakdown(scope.id, breakdown.id)
                            }
                            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-muted rounded shrink-0"
                          >
                            <Trash2 className="size-3.5 text-muted-foreground" />
                          </button>
                        </div>
                        <InlineInput
                          value={breakdown.description}
                          onChange={(value) =>
                            updateBreakdown(scope.id, breakdown.id, {
                              description: value,
                            })
                          }
                          placeholder="Add description..."
                          className="text-sm text-muted-foreground w-full"
                        />
                      </div>

                      {/* Weight pill */}
                      <WeightInput
                        value={breakdown.weight}
                        onChange={(value) =>
                          updateBreakdown(scope.id, breakdown.id, {
                            weight: value,
                          })
                        }
                      />
                    </div>
                  )
                })}

                {/* Add breakdown button - no tree line */}
                <div className="flex items-center py-2">
                  <button
                    onClick={() => addBreakdown(scope.id)}
                    className="ml-6 text-sm text-muted-foreground hover:text-foreground transition-colors"
                  >
                    + Add breakdown
                  </button>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Add scope button */}
      <button
        onClick={addScope}
        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        + Add a scope
      </button>
    </div>
  )
}
