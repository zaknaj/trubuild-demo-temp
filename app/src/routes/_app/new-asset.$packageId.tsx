import { type FormEvent, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import { UserIcon } from "lucide-react"
import { toast } from "sonner"
import { FlowPageLayout } from "@/components/FlowPageLayout"
import { StepTitle } from "@/components/ui/step-title"
import { UploadZone, type UploadedFile } from "@/components/ui/upload-zone"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { createAssetFn } from "@/fn/packages"
import {
  packageAccessQueryOptions,
  packageContractorsQueryOptions,
  packageDetailQueryOptions,
} from "@/lib/query-options"
import { cn } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

const steps = [
  { id: "general", label: "General information" },
  { id: "vendors", label: "Vendor files" },
] as const

export const Route = createFileRoute("/_app/new-asset/$packageId")({
  beforeLoad: async ({ params, context }) => {
    const accessData = await context.queryClient.ensureQueryData(
      packageAccessQueryOptions(params.packageId)
    )
    if (accessData.access !== "full" && accessData.access !== "commercial") {
      throw redirect({ to: "/package/$id", params: { id: params.packageId } })
    }
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { packageId } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
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
  const [assetName, setAssetName] = useState("")
  const [boqFile, setBoqFile] = useState<UploadedFile[]>([])
  const [pteFile, setPteFile] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<Record<string, UploadedFile[]>>({})

  const createAsset = useMutation({
    mutationFn: (name: string) => createAssetFn({ data: { packageId, name } }),
    onSuccess: async (newAsset) => {
      toast.success("Asset created successfully")
      await queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(packageId).queryKey,
      })
      navigate({
        to: "/package/$id/comm/$assetId",
        params: { id: packageId, assetId: newAsset.id },
      })
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to create asset")
    },
  })

  const handleVendorFilesChange = (vendorId: string, files: UploadedFile[]) => {
    setVendorFiles((prev) => ({ ...prev, [vendorId]: files }))
  }

  const handleSubmit = (e: FormEvent) => {
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

  const vendorsWithFiles = Object.values(vendorFiles).filter((files) => files.length > 0).length

  return (
    <FlowPageLayout
      title="New asset"
      context={
        <div className="space-y-0.5">
          <p>Package: {packageData.package.name}</p>
        </div>
      }
      backLabel={packageData.package.name}
      onBack={() => navigate({ to: "/package/$id/comm", params: { id: packageId } })}
      steps={steps}
      currentStepId={currentStep}
    >
      <form className="space-y-6" onSubmit={handleSubmit}>
        {currentStep === "general" ? (
          <div className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="asset-name">
                Asset name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="asset-name"
                placeholder="e.g. HVAC System"
                value={assetName}
                onChange={(e) => setAssetName(e.target.value)}
                className="bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
                disabled={createAsset.isPending}
              />
            </div>
            <div className="space-y-2">
              <StepTitle title="Bill Of Quantities (BOQ)" complete={boqFile.length > 0} required />
              <UploadZone
                files={boqFile}
                onFilesChange={setBoqFile}
                packageId={packageId}
                category="boq"
                storagePathMode="comm_rfp_boq"
                accept=".pdf,.xlsx,.xls"
              />
            </div>
            <div className="space-y-2">
              <StepTitle
                title="Pre-Tender Estimate (PTE)"
                complete={pteFile.length > 0}
                description="Optional"
              />
              <UploadZone
                files={pteFile}
                onFilesChange={setPteFile}
                packageId={packageId}
                category="pte"
                storagePathMode="comm_rfp_rfp"
                accept=".pdf,.xlsx,.xls"
              />
            </div>
            <div>
              <Button
                type="button"
                className="w-full bg-black text-white hover:bg-black/90"
                onClick={() => setCurrentStep("vendors")}
                disabled={!assetName.trim() || boqFile.length === 0 || createAsset.isPending}
              >
                Continue
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-5">
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
                          onFilesChange={(newFiles) =>
                            handleVendorFilesChange(contractor.id, newFiles)
                          }
                          packageId={packageId}
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
            <div className="space-y-2">
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => setCurrentStep("general")}
                disabled={createAsset.isPending}
              >
                Back
              </Button>
              <Button
                type="submit"
                className="w-full bg-black text-white hover:bg-black/90"
                disabled={createAsset.isPending || !assetName.trim() || boqFile.length === 0}
              >
                {createAsset.isPending ? "Creating..." : "Create asset"}
              </Button>
            </div>
          </div>
        )}
      </form>
    </FlowPageLayout>
  )
}
