import { type FormEvent, useEffect, useState } from "react"
import {
  createFileRoute,
  Link,
  Outlet,
  redirect,
  useMatches,
  useNavigate,
} from "@tanstack/react-router"
import { useMutation, useQueryClient, useSuspenseQuery } from "@tanstack/react-query"
import {
  packageAccessQueryOptions,
  packageContractorsQueryOptions,
  packageDetailQueryOptions,
} from "@/lib/query-options"
import { createAssetFn } from "@/fn/packages"
import useStore from "@/lib/store"
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
import { ArrowLeft, Plus, UserIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

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

  const isSheetOpen = useStore((s) => s.createAssetSheetOpen)
  const setIsSheetOpen = useStore((s) => s.setCreateAssetSheetOpen)

  const [assetName, setAssetName] = useState("")
  const [boqFile, setBoqFile] = useState<UploadedFile[]>([])
  const [pteFile, setPteFile] = useState<UploadedFile[]>([])
  const [vendorFiles, setVendorFiles] = useState<
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

  useEffect(() => {
    if (isSheetOpen) {
      resetForm()
    }
  }, [isSheetOpen])

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

  // Validation
  const isBoqDone = boqFile.length > 0
  const vendorsWithFiles = Object.entries(vendorFiles).filter(
    ([, files]) => files.length > 0
  ).length
  const canCreate = assetName.trim() && isBoqDone

  const isInAssetView = !!currentAssetId

  const isPtcRoute = matches.some(
    (match) => match.routeId === "/_app/package/$id/comm/$assetId/ptc"
  )

  if (isInAssetView && currentAssetId && currentAsset) {
    return (
      <div className="h-full w-full bg-[#F9F9F9]">
        <div className="h-full w-full flex flex-col">
          <header className="px-8 pt-6 pb-4 border-b border-black/5 shrink-0">
            <Link
              to="/package/$id/comm"
              params={{ id }}
              className="inline-flex items-center gap-1 text-[13px] font-medium opacity-60 hover:opacity-100"
            >
              <ArrowLeft size={14} />
              Commercial Analysis
            </Link>
            <div className="mt-3 flex items-end justify-between gap-3">
              <h1 className="text-[24px] leading-8 font-semibold text-black">
                {currentAsset.name}
              </h1>
              <nav className="flex items-center gap-5">
                <Link
                  to="/package/$id/comm/$assetId"
                  params={{ id, assetId: currentAssetId }}
                  className={cn(
                    "text-[13px] font-medium pb-1 border-b",
                    !isPtcRoute
                      ? "border-black text-black"
                      : "border-transparent text-black/45 hover:text-black/70"
                  )}
                >
                  Summary
                </Link>
                <Link
                  to="/package/$id/comm/$assetId/ptc"
                  params={{ id, assetId: currentAssetId }}
                  className={cn(
                    "text-[13px] font-medium pb-1 border-b",
                    isPtcRoute
                      ? "border-black text-black"
                      : "border-transparent text-black/45 hover:text-black/70"
                  )}
                >
                  PTC Insights
                </Link>
              </nav>
            </div>
          </header>
          <div className="flex-1 min-h-0 overflow-auto">
            <Outlet />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full bg-[#F9F9F9] p-6">
      <div className="h-full w-full flex gap-6">
        <aside className="basis-[260px] w-[260px] min-w-[260px] max-w-[260px] shrink-0">
          <Link
            to="/package/$id"
            params={{ id }}
            className="h-8 px-2 -ml-2 inline-flex items-center gap-1 text-[13px] font-medium opacity-60 hover:opacity-100"
          >
            <ArrowLeft size={14} />
            Package summary
          </Link>
          <h1 className="mt-2 text-[28px] leading-8 font-semibold text-[#8D4BAF]">
            Commercial Analysis
          </h1>
          <nav className="mt-6 space-y-1">
            {assets.map((asset) => (
              <Link
                key={asset.id}
                to="/package/$id/comm/$assetId"
                params={{ id, assetId: asset.id }}
                className="w-full h-8 rounded-[6px] px-2 flex items-center text-[13px] font-medium hover:bg-[#EDEDED]"
              >
                <span className="truncate opacity-70">{asset.name}</span>
              </Link>
            ))}
            <button
              type="button"
              className="w-full h-8 rounded-[6px] px-2 flex items-center gap-1 text-[13px] font-medium text-[#7EA16B] hover:bg-[#EDEDED]"
              onClick={() => setIsSheetOpen(true)}
            >
              <Plus size={14} />
              New asset
            </button>
          </nav>
        </aside>

        <main className="flex-1 min-w-0 overflow-auto">
          <Outlet />
        </main>
      </div>

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
