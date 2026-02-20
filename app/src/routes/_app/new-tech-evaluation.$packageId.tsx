import { useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import {
  Loader2,
  Sparkles,
  UserIcon,
  Download,
} from "lucide-react"
import { toast } from "sonner"
import { FlowPageLayout } from "@/components/FlowPageLayout"
import {
  Step2Criteria,
  type Scope,
  generateDefaultCriteria,
  normalizeEditableCriteria,
  normalizeExtractedCriteria,
  toTechRfpEvaluationCriteria,
  type TechnicalEvaluationData,
} from "@/components/TechSetupWizard"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { StepTitle } from "@/components/ui/step-title"
import { UploadZone, type UploadedFile } from "@/components/ui/upload-zone"
import {
  createTechnicalEvaluationFn,
  updateTechnicalEvaluationFn,
} from "@/fn/evaluations"
import {
  createTechRfpAnalysisJobFn,
  createTechRfpEvaluationExtractJobFn,
  createTechRfpGenerateEvalJobFn,
  getTechRfpEvaluationExtractJobStatusFn,
  getTechRfpEvaluationExtractResultFn,
  getTechRfpGenerateEvalJobStatusFn,
  getTechRfpGenerateEvalResultFn,
  writeTechRfpEvaluationJsonFn,
} from "@/fn/jobs"
import { resolveCriteriaJob } from "@/lib/criteria-job"
import {
  packageAccessQueryOptions,
  packageContractorsQueryOptions,
  packageDetailQueryOptions,
  queryKeys,
} from "@/lib/query-options"
import useStore from "@/lib/store"
import { cn } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

const steps = [
  { id: "general", label: "General information" },
  { id: "vendors", label: "Vendor files" },
  { id: "criteria", label: "Criteria review" },
] as const

export const Route = createFileRoute("/_app/new-tech-evaluation/$packageId")({
  beforeLoad: async ({ params, context }) => {
    const accessData = await context.queryClient.ensureQueryData(
      packageAccessQueryOptions(params.packageId)
    )
    if (accessData.access !== "full" && accessData.access !== "technical") {
      throw redirect({ to: "/package/$id", params: { id: params.packageId } })
    }
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { packageId } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const setTechRound = useStore((s) => s.setTechRound)
  const { data: packageData } = useQuery(packageDetailQueryOptions(packageId))
  const { data: contractors } = useQuery(packageContractorsQueryOptions(packageId))
  if (!packageData || !contractors) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const [currentStep, setCurrentStep] = useState<(typeof steps)[number]["id"]>("general")
  const [isExtracting, setIsExtracting] = useState(false)
  const [criteriaProcessingMode, setCriteriaProcessingMode] = useState<
    "extract" | "generate" | null
  >(null)
  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false)
  const [uploadingCount, setUploadingCount] = useState(0)

  const [rfpFile, setRfpFile] = useState<UploadedFile[]>([])
  const [criteriaMode, setCriteriaMode] = useState<"upload" | "ai" | null>(null)
  const [criteriaFile, setCriteriaFile] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<Record<string, UploadedFile[]>>({})
  const [criteria, setCriteria] = useState<Scope[]>(() => generateDefaultCriteria())

  const isRfpDone = rfpFile.length > 0
  const isCriteriaDone =
    criteriaMode === "ai" || (criteriaMode === "upload" && criteriaFile.length > 0)
  const vendorsWithFiles = Object.entries(vendorFiles).filter(([, files]) => files.length > 0).length
  const isVendorsDone = vendorsWithFiles >= 2
  const isUploading = uploadingCount > 0
  const canContinueGeneral = isRfpDone && isCriteriaDone && !isUploading

  const totalWeight = criteria.reduce(
    (sum, scope) => sum + scope.breakdowns.reduce((scopeSum, item) => scopeSum + (item.weight || 0), 0),
    0
  )
  const canStart = totalWeight === 100 && !isStartingAnalysis

  const proposalsUploaded = Object.entries(vendorFiles)
    .filter(([, files]) => files.length > 0)
    .map(([id]) => id)

  const handleUploadingChange = (uploading: boolean) => {
    setUploadingCount((prev) => prev + (uploading ? 1 : -1))
  }

  const goToCriteriaStep = async () => {
    if (criteriaMode === "ai") {
      setIsExtracting(true)
      setCriteriaProcessingMode("generate")
      try {
        const result = await resolveCriteriaJob({
          createJob: () => createTechRfpGenerateEvalJobFn({ data: { packageId } }),
          getStatus: (jobId) => getTechRfpGenerateEvalJobStatusFn({ data: { jobId } }),
          getResult: (jobId) => getTechRfpGenerateEvalResultFn({ data: { jobId } }),
          normalize: normalizeExtractedCriteria,
        })
        if (result.status === "completed") {
          setCriteria(result.scopes)
          setCurrentStep("criteria")
        } else if (result.status === "empty") {
          toast.error("No valid criteria were generated. Try again or upload your own criteria.")
        } else if (result.status === "cancelled") {
          toast.error("AI criteria generation was cancelled. Please try again.")
        } else {
          toast.error("AI criteria generation failed. Please try again.")
        }
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Failed to start AI criteria generation"
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
        createJob: () => createTechRfpEvaluationExtractJobFn({ data: { packageId } }),
        getStatus: (jobId) => getTechRfpEvaluationExtractJobStatusFn({ data: { jobId } }),
        getResult: (jobId) => getTechRfpEvaluationExtractResultFn({ data: { jobId } }),
        normalize: normalizeExtractedCriteria,
      })
      if (result.status === "completed") {
        setCriteria(result.scopes)
        setCurrentStep("criteria")
      } else if (result.status === "empty") {
        toast.error("No valid criteria found in extraction result. Please try another file.")
      } else if (result.status === "cancelled") {
        toast.error("Criteria extraction was cancelled. Please try again.")
      } else {
        toast.error("Criteria extraction failed. Please try again.")
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to start criteria extraction")
    } finally {
      setIsExtracting(false)
      setCriteriaProcessingMode(null)
    }
  }

  const handleStartAnalysis = async () => {
    const normalizedCriteria = normalizeEditableCriteria(criteria)
    if (normalizedCriteria.length === 0) {
      toast.error("Please add at least one scope with criteria before starting.")
      return
    }
    const normalizedTotalWeight = normalizedCriteria.reduce(
      (sum, scope) => sum + scope.breakdowns.reduce((scopeSum, item) => scopeSum + item.weight, 0),
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

  return (
    <FlowPageLayout
      title="New technical evaluation"
      context={
        <div className="space-y-0.5">
          <p>Package: {packageData.package.name}</p>
        </div>
      }
      backLabel={packageData.package.name}
      onBack={() => navigate({ to: "/package/$id/tech", params: { id: packageId } })}
      steps={steps}
      currentStepId={currentStep}
    >
      {isExtracting ? (
        <div className="py-14 flex flex-col items-center justify-center gap-3 text-center">
          <Loader2 className="size-9 animate-spin text-primary" />
          <p className="text-base font-medium">
            {criteriaProcessingMode === "generate"
              ? "Generating evaluation criteria"
              : "Extracting evaluation criteria"}
          </p>
          <p className="text-sm text-muted-foreground">
            {criteriaProcessingMode === "generate"
              ? "Analyzing your RFP and preparing weighted criteria..."
              : "Analyzing uploaded criteria document..."}
          </p>
        </div>
      ) : (
        <>
          {currentStep === "general" ? (
            <div className="space-y-8">
              <div className="space-y-3">
                <StepTitle title="Request For Proposals (RFP)" complete={isRfpDone} required />
                <UploadZone
                  files={rfpFile}
                  onFilesChange={setRfpFile}
                  packageId={packageId}
                  category="rfp"
                  storagePathMode="tech_rfp_rfp"
                  onUploadingChange={handleUploadingChange}
                  accept=".pdf,.doc,.docx"
                />
              </div>

              <div className="space-y-3">
                <StepTitle title="Evaluation Criteria Source" complete={isCriteriaDone} required />
                <RadioGroup
                  value={criteriaMode ?? ""}
                  onValueChange={(v) => setCriteriaMode(v as "upload" | "ai")}
                  className="space-y-3"
                >
                  <div
                    className={cn(
                      "flex items-start gap-3 p-4 rounded-lg border transition-colors cursor-pointer",
                      criteriaMode === "upload"
                        ? "border-primary bg-primary/5"
                        : "hover:bg-muted/50"
                    )}
                    onClick={() => setCriteriaMode("upload")}
                  >
                    <RadioGroupItem value="upload" id="new-tech-criteria-upload" className="mt-1" />
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="new-tech-criteria-upload" className="cursor-pointer font-medium">
                          Upload your own
                        </Label>
                        <button
                          className="text-sm text-primary hover:underline inline-flex items-center gap-1"
                          type="button"
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            toast.success("Template download started")
                          }}
                        >
                          <Download className="size-3.5" />
                          Download template
                        </button>
                      </div>
                      {criteriaMode === "upload" ? (
                        <UploadZone
                          files={criteriaFile}
                          onFilesChange={setCriteriaFile}
                          packageId={packageId}
                          category="criteria"
                          storagePathMode="tech_rfp_evaluation"
                          onUploadingChange={handleUploadingChange}
                          accept=".pdf,.doc,.docx,.xlsx"
                          compact
                        />
                      ) : null}
                    </div>
                  </div>

                  <div
                    className={cn(
                      "flex items-start gap-3 p-4 rounded-lg border transition-colors cursor-pointer",
                      criteriaMode === "ai"
                        ? "border-primary bg-primary/5"
                        : "hover:bg-muted/50"
                    )}
                    onClick={() => setCriteriaMode("ai")}
                  >
                    <RadioGroupItem value="ai" id="new-tech-criteria-ai" className="mt-1" />
                    <div className="flex-1">
                      <Label
                        htmlFor="new-tech-criteria-ai"
                        className="cursor-pointer font-medium flex items-center gap-2"
                      >
                        <Sparkles className="size-4 text-amber-500" />
                        Generate with AI
                      </Label>
                      <p className="text-sm text-muted-foreground mt-1">
                        We will analyze your RFP and generate a weighted evaluation rubric.
                      </p>
                    </div>
                  </div>
                </RadioGroup>
              </div>

              <div>
                <Button
                  type="button"
                  className="w-full bg-black text-white hover:bg-black/90"
                  disabled={!canContinueGeneral}
                  onClick={() => setCurrentStep("vendors")}
                >
                  {isUploading ? "Uploading..." : "Continue"}
                </Button>
              </div>
            </div>
          ) : null}

          {currentStep === "vendors" ? (
            <div className="space-y-5">
              <div className="space-y-3">
                <StepTitle
                  title={`Vendor Proposals (${vendorsWithFiles}/${contractors.length} vendors have files)`}
                  complete={isVendorsDone}
                  required
                />
                {contractors.length === 0 ? (
                  <div className="text-center py-8 border rounded-lg">
                    <UserIcon className="size-10 mx-auto text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">No contractors added to this package yet.</p>
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
                            hasFiles && "border-emerald-500 bg-emerald-50/50"
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
                              setVendorFiles((prev) => ({ ...prev, [contractor.id]: newFiles }))
                            }
                            packageId={packageId}
                            category="vendor_proposal"
                            storagePathMode="tech_rfp_tender"
                            contractorId={contractor.id}
                            vendorName={contractor.name}
                            onUploadingChange={handleUploadingChange}
                            multiple
                            accept=".pdf,.doc,.docx,.xlsx"
                            compact
                          />
                        </div>
                      )
                    })}
                    {!isVendorsDone ? (
                      <p className="text-sm text-amber-600">
                        At least 2 vendors must have files to proceed.
                      </p>
                    ) : null}
                  </div>
                )}
              </div>
              <div className="space-y-2">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  onClick={() => setCurrentStep("general")}
                >
                  Back
                </Button>
                <Button
                  type="button"
                  className="w-full bg-black text-white hover:bg-black/90"
                  disabled={!isVendorsDone || isUploading}
                  onClick={goToCriteriaStep}
                >
                  {isUploading ? "Uploading..." : "Continue"}
                </Button>
              </div>
            </div>
          ) : null}

          {currentStep === "criteria" ? (
            <div className="space-y-5">
              <Step2Criteria criteria={criteria} onChange={setCriteria} totalWeight={totalWeight} />
              <div className="space-y-2">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  onClick={() => setCurrentStep("vendors")}
                >
                  Back
                </Button>
                <Button
                  type="button"
                  className="w-full bg-black text-white hover:bg-black/90"
                  onClick={handleStartAnalysis}
                  disabled={!canStart}
                >
                  {isStartingAnalysis
                    ? "Starting..."
                    : totalWeight !== 100
                      ? `Weights: ${totalWeight}% (need 100%)`
                      : "Start analysis"}
                </Button>
              </div>
            </div>
          ) : null}
        </>
      )}
    </FlowPageLayout>
  )
}
