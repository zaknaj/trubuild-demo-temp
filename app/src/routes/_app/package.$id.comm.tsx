import { useState, useEffect } from "react"
import {
  createFileRoute,
  Outlet,
  useMatches,
  useNavigate,
  redirect,
} from "@tanstack/react-router"
import {
  useSuspenseQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  packageDetailQueryOptions,
  packageContractorsQueryOptions,
  commercialEvaluationsQueryOptions,
  packageAccessQueryOptions,
} from "@/lib/query-options"
import { createAssetFn } from "@/fn/packages"
import {
  createCommercialEvaluationFn,
  updateCommercialEvaluationDataFn,
} from "@/fn/evaluations"
import {
  createCommRfpCompareJobFn,
  getCommRfpCompareJobStatusFn,
  getCommRfpCompareResultFn,
} from "@/fn/jobs"
import useStore from "@/lib/store"
import type { CommercialEvaluationData } from "@/lib/types"
import type { CommRfpJobStatus } from "@/lib/types"
import { extractCommReportFromArtifact } from "@/lib/comm-job"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
} from "@/components/ui/sheet"
import { UploadZone, type UploadedFile } from "@/components/ui/upload-zone"
import { StepTitle } from "@/components/ui/step-title"
import { UserIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

type CommercialEvaluation = {
  id: string
  assetId: string
  roundNumber: number
  roundName: string
  data: CommercialEvaluationData
  createdAt: Date
  updatedAt: Date
}

const JOB_POLL_INTERVAL_MS = 3000
const JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000

export const Route = createFileRoute("/_app/package/$id/comm")({
  beforeLoad: async ({ params, context }) => {
    // Check commercial access before loading the route
    const accessData = await context.queryClient.ensureQueryData(
      packageAccessQueryOptions(params.id)
    )
    if (accessData.access !== "full" && accessData.access !== "commercial") {
      throw redirect({ to: "/package/$id", params: { id: params.id } })
    }
  },
  loader: ({ params, context }) => {
    context.queryClient.prefetchQuery(packageContractorsQueryOptions(params.id))
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const matches = useMatches()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: packageData } = useSuspenseQuery(packageDetailQueryOptions(id))
  const { data: contractors } = useSuspenseQuery(
    packageContractorsQueryOptions(id)
  )
  const assets = packageData?.assets ?? []

  // Get the currently selected assetId from URL
  const currentAssetId = matches
    .map((m) => (m.params as { assetId?: string }).assetId)
    .find((assetId) => assetId)

  // Get the current asset if we have one
  const currentAsset = currentAssetId
    ? assets.find((a) => a.id === currentAssetId)
    : null

  // Create asset sheet state (from store for cross-component access)
  const isSheetOpen = useStore((s) => s.createAssetSheetOpen)
  const setIsSheetOpen = useStore((s) => s.setCreateAssetSheetOpen)

  // Asset creation form state
  const [assetName, setAssetName] = useState("")
  const [boqFile, setBoqFile] = useState<UploadedFile[]>([])
  const [pteFile, setPteFile] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<
    Record<string, UploadedFile[]>
  >({})

  // Commercial setup sheet state
  const [isSetupOpen, setIsSetupOpen] = useState(false)
  const [evalBoqFile, setEvalBoqFile] = useState<UploadedFile[]>([])
  const [evalPteFile, setEvalPteFile] = useState<UploadedFile[]>([])
  const [evalVendorFiles, setEvalVendorFiles] = useState<
    Record<string, UploadedFile[]>
  >({})

  const setAssetFiles = useStore((s) => s.setAssetFiles)

  const createAsset = useMutation({
    mutationFn: (name: string) =>
      createAssetFn({ data: { packageId: id, name } }),
    onSuccess: (newAsset) => {
      toast.success("Asset created successfully")
      // Save the files to the store before resetting
      setAssetFiles(newAsset.id, {
        boqFile: [...boqFile],
        pteFile: [...pteFile],
        vendorFiles: { ...vendorFiles },
      })
      queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(id).queryKey,
      })
      setIsSheetOpen(false)
      resetForm()
      navigate({
        to: "/package/$id/comm/$assetId",
        params: { id, assetId: newAsset.id },
      })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to create asset"
      )
    },
  })

  const resetForm = () => {
    setAssetName("")
    setBoqFile([])
    setPteFile([])
    setVendorFiles({})
  }

  // Reset form when sheet opens
  useEffect(() => {
    if (isSheetOpen) {
      resetForm()
    }
  }, [isSheetOpen])

  const handleVendorFilesChange = (vendorId: string, files: UploadedFile[]) => {
    setVendorFiles((prev) => ({ ...prev, [vendorId]: files }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedName = assetName.trim()
    if (!trimmedName) {
      toast.error("Asset name is required")
      return
    }
    if (boqFile.length === 0) {
      toast.error("BOQ file is required")
      return
    }
    createAsset.mutate(trimmedName)
  }

  // Validation
  const isBoqDone = boqFile.length > 0
  const vendorsWithFiles = Object.entries(vendorFiles).filter(
    ([, files]) => files.length > 0
  ).length
  const canCreate = assetName.trim() && isBoqDone

  // Determine which view to show based on whether we're viewing an asset
  const isInAssetView = !!currentAssetId

  // For asset view, we need evaluations data
  if (isInAssetView && currentAsset) {
    return (
      <AssetView
        packageId={id}
        assetId={currentAssetId}
        assetName={currentAsset.name}
        contractors={contractors}
        isSetupOpen={isSetupOpen}
        setIsSetupOpen={setIsSetupOpen}
        evalBoqFile={evalBoqFile}
        setEvalBoqFile={setEvalBoqFile}
        evalPteFile={evalPteFile}
        setEvalPteFile={setEvalPteFile}
        evalVendorFiles={evalVendorFiles}
        setEvalVendorFiles={setEvalVendorFiles}
      />
    )
  }

  // Package summary view
  return (
    <div className="flex flex-1 flex-col overflow-hidden h-full">
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>

      {/* Create Asset Sheet */}
      <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
        <SheetContent className="sm:max-w-xl overflow-y-auto">
          <SheetHeader>
            <SheetTitle>Create New Asset</SheetTitle>
            <SheetDescription>
              Add a new asset to this package with BOQ, optional PTE, and vendor
              files.
            </SheetDescription>
          </SheetHeader>

          <form onSubmit={handleSubmit} className="flex flex-col gap-6 p-4">
            {/* Asset Name */}
            <div className="space-y-2">
              <Label htmlFor="asset-name">
                Asset Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="asset-name"
                placeholder="e.g. HVAC System"
                value={assetName}
                onChange={(e) => setAssetName(e.target.value)}
                disabled={createAsset.isPending}
              />
            </div>

            {/* BOQ File */}
            <div className="space-y-2">
              <StepTitle
                title="Bill Of Quantities (BOQ)"
                complete={isBoqDone}
                required
              />
              <UploadZone
                files={boqFile}
                onFilesChange={setBoqFile}
                packageId={id}
                category="boq"
                storagePathMode="comm_rfp_boq"
                accept=".pdf,.xlsx,.xls"
              />
            </div>

            {/* PTE File */}
            <div className="space-y-2">
              <StepTitle
                title="Pre-Tender Estimate (PTE)"
                complete={pteFile.length > 0}
                description="Optional"
              />
              <UploadZone
                files={pteFile}
                onFilesChange={setPteFile}
                packageId={id}
                category="pte"
                storagePathMode="comm_rfp_rfp"
                accept=".pdf,.xlsx,.xls"
              />
            </div>

            {/* Vendor Files */}
            <div className="space-y-3">
              <StepTitle
                title={`Vendor Files (${vendorsWithFiles}/${contractors.length} vendors)`}
                complete={vendorsWithFiles > 0}
                description="Optional - can be added when running evaluation"
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
                          hasFiles &&
                            "border-emerald-500 bg-emerald-50/50"
                        )}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <div className="flex items-center justify-center w-6 h-6 rounded bg-muted">
                            <UserIcon
                              size={14}
                              className="text-muted-foreground"
                            />
                          </div>
                          <span className="text-sm font-medium">
                            {contractor.name}
                          </span>
                        </div>
                        <UploadZone
                          files={files}
                          onFilesChange={(newFiles) =>
                            handleVendorFilesChange(contractor.id, newFiles)
                          }
                          packageId={id}
                          category="vendor_proposal"
                          contractorId={contractor.id}
                          storagePathMode="comm_rfp_tender"
                          vendorName={contractor.name}
                          multiple
                          accept=".pdf,.xlsx,.xls,.doc,.docx"
                          compact
                        />
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </form>

          <SheetFooter className="px-4 pb-4">
            <Button
              variant="outline"
              onClick={() => {
                setIsSheetOpen(false)
                resetForm()
              }}
              disabled={createAsset.isPending}
            >
              Cancel
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={createAsset.isPending || !canCreate}
            >
              {createAsset.isPending ? "Creating..." : "Create Asset"}
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  )
}

// ============================================================================
// Asset View Component
// ============================================================================

function AssetView({
  packageId,
  assetId,
  assetName,
  contractors,
  isSetupOpen,
  setIsSetupOpen,
  evalBoqFile,
  setEvalBoqFile,
  evalPteFile,
  setEvalPteFile,
  evalVendorFiles,
  setEvalVendorFiles,
}: {
  packageId: string
  assetId: string
  assetName: string
  contractors: Array<{ id: string; name: string }>
  isSetupOpen: boolean
  setIsSetupOpen: (open: boolean) => void
  evalBoqFile: UploadedFile[]
  setEvalBoqFile: React.Dispatch<React.SetStateAction<UploadedFile[]>>
  evalPteFile: UploadedFile[]
  setEvalPteFile: React.Dispatch<React.SetStateAction<UploadedFile[]>>
  evalVendorFiles: Record<string, UploadedFile[]>
  setEvalVendorFiles: React.Dispatch<
    React.SetStateAction<Record<string, UploadedFile[]>>
  >
}) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()

  // Get evaluations for this asset
  const { data: evaluations } = useSuspenseQuery(
    commercialEvaluationsQueryOptions(assetId)
  )
  const evaluationsList = evaluations as CommercialEvaluation[]

  // Get/set round from Zustand store
  const selectedRoundId = useStore((s) => s.selectedCommRound[assetId])
  const setCommRound = useStore((s) => s.setCommRound)

  // Auto-select latest round when no round is stored or stored round is invalid
  useEffect(() => {
    if (evaluationsList.length > 0) {
      const storedRoundValid =
        selectedRoundId && evaluationsList.some((e) => e.id === selectedRoundId)
      if (!storedRoundValid) {
        setCommRound(assetId, evaluationsList[0].id)
      }
    }
  }, [evaluationsList, selectedRoundId, assetId, setCommRound])

  // Create a commercial evaluation with data
  const createAndRunEvaluation = useMutation({
    mutationFn: async () => {
      const vendorFileCount = Object.values(evalVendorFiles).filter(
        (arr) => arr.length > 0
      ).length
      if (evalBoqFile.length === 0) {
        throw new Error("BOQ template is required")
      }
      if (vendorFileCount < 2) {
        throw new Error("At least 2 vendors must have files")
      }

      const newEval = (await createCommercialEvaluationFn({
        data: { assetId },
      })) as CommercialEvaluation

      const { jobId } = await createCommRfpCompareJobFn({
        data: {
          packageId,
          assetId,
          analysisType: "compare",
        },
      })

      const startedAt = Date.now()
      while (true) {
        if (Date.now() - startedAt > JOB_POLL_TIMEOUT_MS) {
          throw new Error(
            "Commercial comparison timed out. Please try again later."
          )
        }

        try {
          const readyResult = await getCommRfpCompareResultFn({
            data: { jobId },
          })
          const readyReport = extractCommReportFromArtifact(
            (readyResult as { data: unknown }).data
          )
          await updateCommercialEvaluationDataFn({
            data: {
              evaluationId: newEval.id,
              data: readyReport,
            },
          })
          return { ...newEval, data: readyReport } as unknown as CommercialEvaluation
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          if (
            !message.includes("No result artifact found") &&
            !message.includes('No "result" or "report" artifact found')
          ) {
            throw error
          }
          // Artifact not ready yet, continue status polling.
        }

        const statusResult = (await getCommRfpCompareJobStatusFn({
          data: { jobId },
        })) as CommRfpJobStatus

        if (statusResult.status === "completed") {
          const result = await getCommRfpCompareResultFn({
            data: { jobId },
          })
          const reportData = extractCommReportFromArtifact(
            (result as { data: unknown }).data
          )
          await updateCommercialEvaluationDataFn({
            data: {
              evaluationId: newEval.id,
              data: reportData,
            },
          })
          return { ...newEval, data: reportData } as unknown as CommercialEvaluation
        }

        if (
          statusResult.status === "failed" ||
          statusResult.status === "cancelled"
        ) {
          throw new Error(
            statusResult.error ?? "Commercial comparison failed in worker."
          )
        }

        await new Promise((resolve) =>
          setTimeout(resolve, JOB_POLL_INTERVAL_MS)
        )
      }
    },
    onSuccess: async (newEval) => {
      toast.success("Commercial evaluation created")
      await queryClient.invalidateQueries({
        queryKey: commercialEvaluationsQueryOptions(assetId).queryKey,
      })
      setCommRound(assetId, newEval.id)
      setIsSetupOpen(false)
      setEvalVendorFiles({})
      navigate({
        to: "/package/$id/comm/$assetId",
        params: { id: packageId, assetId },
      })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to create commercial evaluation"
      )
    },
  })

  const handleVendorFilesChange = (vendorId: string, files: UploadedFile[]) => {
    setEvalVendorFiles((prev) => ({ ...prev, [vendorId]: files }))
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden h-full">
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>

      {/* Commercial Setup Sheet */}
      <CommercialSetupSheet
        open={isSetupOpen}
        onOpenChange={setIsSetupOpen}
        packageId={packageId}
        assetName={assetName}
        boqFile={evalBoqFile}
        onBoqFileChange={setEvalBoqFile}
        pteFile={evalPteFile}
        onPteFileChange={setEvalPteFile}
        contractors={contractors}
        vendorFiles={evalVendorFiles}
        onVendorFilesChange={handleVendorFilesChange}
        onRunEvaluation={() => createAndRunEvaluation.mutate()}
        isPending={createAndRunEvaluation.isPending}
      />
    </div>
  )
}

// ============================================================================
// Commercial Setup Sheet
// ============================================================================

function CommercialSetupSheet({
  open,
  onOpenChange,
  packageId,
  assetName,
  boqFile,
  onBoqFileChange,
  pteFile,
  onPteFileChange,
  contractors,
  vendorFiles,
  onVendorFilesChange,
  onRunEvaluation,
  isPending,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  packageId: string
  assetName: string
  boqFile: UploadedFile[]
  onBoqFileChange: (files: UploadedFile[]) => void
  pteFile: UploadedFile[]
  onPteFileChange: (files: UploadedFile[]) => void
  contractors: Array<{ id: string; name: string }>
  vendorFiles: Record<string, UploadedFile[]>
  onVendorFilesChange: (vendorId: string, files: UploadedFile[]) => void
  onRunEvaluation: () => void
  isPending: boolean
}) {
  const [uploadingKeys, setUploadingKeys] = useState<Set<string>>(new Set())
  const isBoqDone = boqFile.length > 0
  const vendorsWithFiles = Object.entries(vendorFiles).filter(
    ([, files]) => files.length > 0
  ).length
  const canRunEvaluation = isBoqDone && vendorsWithFiles >= 2
  const isUploading = uploadingKeys.size > 0

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

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Run Commercial Evaluation</SheetTitle>
          <SheetDescription>
            Review documents and upload vendor proposals. At least 2 vendors
            must have files to proceed.
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 p-4 space-y-6">
          {/* Asset Name (read-only) */}
          <div className="space-y-2">
            <Label>Asset Name</Label>
            <Input value={assetName} disabled className="bg-muted" />
          </div>

          {/* BOQ File */}
          <div className="space-y-2">
            <StepTitle
              title="Bill Of Quantities (BOQ)"
              complete={isBoqDone}
              required
            />
            <UploadZone
              files={boqFile}
              onFilesChange={onBoqFileChange}
              packageId={packageId}
              category="boq"
              storagePathMode="comm_rfp_boq"
              accept=".pdf,.xlsx,.xls"
              onUploadingChange={(uploading) =>
                updateUploadState("boq", uploading)
              }
            />
          </div>

          {/* PTE File */}
          <div className="space-y-2">
            <StepTitle
              title="Pre-Tender Estimate (PTE)"
              complete={pteFile.length > 0}
              description="Optional"
            />
            <UploadZone
              files={pteFile}
              onFilesChange={onPteFileChange}
              packageId={packageId}
              category="pte"
              storagePathMode="comm_rfp_rfp"
              accept=".pdf,.xlsx,.xls"
              onUploadingChange={(uploading) =>
                updateUploadState("pte", uploading)
              }
            />
          </div>

          {/* Vendor Files */}
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
                        hasFiles &&
                          "border-emerald-500 bg-emerald-50/50"
                      )}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="flex items-center justify-center w-6 h-6 rounded bg-muted">
                          <UserIcon
                            size={14}
                            className="text-muted-foreground"
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {contractor.name}
                        </span>
                      </div>
                      <UploadZone
                        files={files}
                        onFilesChange={(newFiles) =>
                          onVendorFilesChange(contractor.id, newFiles)
                        }
                        packageId={packageId}
                        category="vendor_proposal"
                        contractorId={contractor.id}
                        storagePathMode="comm_rfp_tender"
                        vendorName={contractor.name}
                        multiple
                        accept=".pdf,.xlsx,.xls,.doc,.docx"
                        compact
                        onUploadingChange={(uploading) =>
                          updateUploadState(`vendor:${contractor.id}`, uploading)
                        }
                      />
                    </div>
                  )
                })}

                {!canRunEvaluation && (
                  <p className="text-sm text-amber-600">
                    At least 2 vendors must have files to run evaluation
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <SheetFooter className="px-4 pb-4">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={onRunEvaluation}
            disabled={!canRunEvaluation || isPending || isUploading}
          >
            {isPending
              ? "Running..."
              : isUploading
                ? "Uploading..."
                : "Run Evaluation"}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
