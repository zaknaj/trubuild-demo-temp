import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import { toast } from "sonner"
import { UserIcon } from "lucide-react"
import { FlowPageLayout } from "@/components/FlowPageLayout"
import { Button } from "@/components/ui/button"
import { StepTitle } from "@/components/ui/step-title"
import { UploadZone, type UploadedFile } from "@/components/ui/upload-zone"
import { createCommercialEvaluationFn, updateCommercialEvaluationDataFn } from "@/fn/evaluations"
import { createCommRfpCompareJobFn } from "@/fn/jobs"
import {
  assetDetailQueryOptions,
  commercialEvaluationsQueryOptions,
  packageAccessQueryOptions,
  packageContractorsQueryOptions,
} from "@/lib/query-options"
import useStore from "@/lib/store"
import { cn } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

type CommercialEvaluation = {
  id: string
}

const steps = [
  { id: "general", label: "General files" },
  { id: "vendors", label: "Vendor files" },
] as const

export const Route = createFileRoute("/_app/new-commercial-evaluation/$assetId")({
  beforeLoad: async ({ params, context }) => {
    const assetData = await context.queryClient.ensureQueryData(
      assetDetailQueryOptions(params.assetId)
    )
    const accessData = await context.queryClient.ensureQueryData(
      packageAccessQueryOptions(assetData.asset.packageId)
    )
    if (accessData.access !== "full" && accessData.access !== "commercial") {
      throw redirect({
        to: "/package/$id/comm/$assetId",
        params: { id: assetData.asset.packageId, assetId: params.assetId },
      })
    }
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { assetId } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const setCommRound = useStore((s) => s.setCommRound)
  const { data: assetData } = useQuery(assetDetailQueryOptions(assetId))
  if (!assetData) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }
  const packageId = assetData.asset.packageId
  const { data: contractors } = useQuery(packageContractorsQueryOptions(packageId))
  if (!contractors) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }

  const [currentStep, setCurrentStep] = useState<(typeof steps)[number]["id"]>("general")
  const [templateFiles, setTemplateFiles] = useState<UploadedFile[]>([])
  const [pteFiles, setPteFiles] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<Record<string, UploadedFile[]>>({})
  const [uploadingKeys, setUploadingKeys] = useState<Set<string>>(new Set())

  const createAndRun = useMutation({
    mutationFn: async () => {
      const newEval = (await createCommercialEvaluationFn({
        data: { assetId },
      })) as CommercialEvaluation
      setCommRound(assetId, newEval.id)

      await updateCommercialEvaluationDataFn({
        data: {
          evaluationId: newEval.id,
          data: {
            status: "analyzing",
            analysis: {
              status: "pending",
            },
          },
        },
      })

      const { jobId } = await createCommRfpCompareJobFn({
        data: {
          packageId,
          assetId,
          analysisType: "compare",
        },
      })

      await updateCommercialEvaluationDataFn({
        data: {
          evaluationId: newEval.id,
          data: {
            status: "analyzing",
            analysis: {
              jobId,
              status: "pending",
            },
          },
        },
      })

      return newEval
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
      })
      toast.success("Commercial evaluation started")
      navigate({
        to: "/package/$id/comm/$assetId",
        params: { id: packageId, assetId },
      })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to start commercial evaluation"
      )
    },
  })

  const updateUploadState = (key: string, uploading: boolean) => {
    setUploadingKeys((prev) => {
      const next = new Set(prev)
      if (uploading) {
        next.add(key)
      } else {
        next.delete(key)
      }
      return next
    })
  }

  const vendorsWithFiles = Object.values(vendorFiles).filter((arr) => arr.length > 0).length
  const canContinueGeneral = templateFiles.length > 0 && uploadingKeys.size === 0
  const canRunEvaluation = templateFiles.length > 0 && vendorsWithFiles >= 2

  return (
    <FlowPageLayout
      title="New commercial evaluation"
      context={
        <div className="space-y-0.5">
          <p>Package: {assetData.package.name}</p>
        </div>
      }
      backLabel={assetData.asset.name}
      onBack={() => navigate({ to: "/package/$id/comm/$assetId", params: { id: packageId, assetId } })}
      steps={steps}
      currentStepId={currentStep}
    >
      {currentStep === "general" ? (
        <div className="space-y-5">
          <div className="space-y-2">
            <StepTitle title="Bill Of Quantities (BOQ)" complete={templateFiles.length > 0} required />
            <UploadZone
              files={templateFiles}
              onFilesChange={setTemplateFiles}
              packageId={packageId}
              category="boq"
              storagePathMode="comm_rfp_boq"
              accept=".xlsx,.xls"
              onUploadingChange={(uploading) => updateUploadState("boq", uploading)}
            />
          </div>

          <div className="space-y-2">
            <StepTitle title="Pre-Tender Estimate (PTE)" complete={pteFiles.length > 0} description="Optional" />
            <UploadZone
              files={pteFiles}
              onFilesChange={setPteFiles}
              packageId={packageId}
              category="pte"
              storagePathMode="comm_rfp_rfp"
              accept=".xlsx,.xls"
              onUploadingChange={(uploading) => updateUploadState("pte", uploading)}
            />
          </div>

          <div>
            <Button
              type="button"
              className="w-full bg-black text-white hover:bg-black/90"
              disabled={!canContinueGeneral || createAndRun.isPending}
              onClick={() => setCurrentStep("vendors")}
            >
              {uploadingKeys.size > 0 ? "Uploading..." : "Continue"}
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-5">
          <div className="space-y-3">
            <StepTitle
              title={`Vendor Proposals (${vendorsWithFiles}/${contractors.length} vendors have files)`}
              complete={canRunEvaluation}
              required
            />

            {contractors.length === 0 ? (
              <div className="text-center py-6 border rounded-lg border-dashed">
                <UserIcon className="size-8 mx-auto text-muted-foreground mb-2" />
                <p className="text-sm text-muted-foreground">
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
                        "rounded-lg border p-3 transition-colors",
                        hasFiles && "border-emerald-500 bg-emerald-50/50"
                      )}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="flex items-center justify-center w-6 h-6 rounded bg-muted">
                          <UserIcon size={14} className="text-muted-foreground" />
                        </div>
                        <span className="text-sm font-medium">{contractor.name}</span>
                      </div>
                      <UploadZone
                        files={files}
                        onFilesChange={(nextFiles) =>
                          setVendorFiles((prev) => ({
                            ...prev,
                            [contractor.id]: nextFiles.slice(0, 1),
                          }))
                        }
                        packageId={packageId}
                        category="vendor_proposal"
                        storagePathMode="comm_rfp_tender"
                        vendorName={contractor.name}
                        accept=".xlsx,.xls,.pdf,.doc,.docx"
                        compact
                        onUploadingChange={(uploading) =>
                          updateUploadState(`vendor:${contractor.id}`, uploading)
                        }
                      />
                    </div>
                  )
                })}

                {!canRunEvaluation ? (
                  <p className="text-sm text-amber-600">
                    At least 2 vendors must have files to run evaluation.
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
              disabled={createAndRun.isPending}
            >
              Back
            </Button>
            <Button
              type="button"
              className="w-full bg-black text-white hover:bg-black/90"
              disabled={!canRunEvaluation || createAndRun.isPending || uploadingKeys.size > 0}
              onClick={() => createAndRun.mutate()}
            >
              {createAndRun.isPending
                ? "Starting..."
                : uploadingKeys.size > 0
                  ? "Uploading..."
                  : "Run evaluation"}
            </Button>
          </div>
        </div>
      )}
    </FlowPageLayout>
  )
}
