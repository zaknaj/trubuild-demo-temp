import { useMemo, useState } from "react"
import {
  createFileRoute,
  Link,
  useNavigate,
  redirect,
} from "@tanstack/react-router"
import {
  useSuspenseQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import { BoxIcon, PlusIcon, ChevronRightIcon, X, UserIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ProjectSidebar } from "@/components/ProjectSidebar"
import { ProjectSettingsDialog } from "@/components/ProjectSettingsDialog"
import { createPackageFn } from "@/fn/packages"
import {
  projectDetailQueryOptions,
  projectAccessQueryOptions,
  projectMembersQueryOptions,
  projectsQueryOptions,
} from "@/lib/query-options"
import { getCurrencyForCountry } from "@/lib/countries"
import { CurrencySelect } from "@/components/CurrencySelect"
import { AIChatButton } from "@/components/AIChatButton"
import { StepTitle } from "@/components/ui/step-title"

type PackageWithAssetCount = {
  id: string
  name: string
  currency: string | null
  projectId: string
  assetCount: number
  awardedContractorId: string | null
  awardedContractorName: string | null
}

export const Route = createFileRoute("/_app/project/$id")({
  beforeLoad: async ({ params, context }) => {
    try {
      const accessData = await context.queryClient.ensureQueryData(
        projectAccessQueryOptions(params.id)
      )
      if (accessData.access === "none") {
        throw redirect({ to: "/" })
      }
    } catch (error) {
      if (error instanceof Error && !("to" in error)) {
        throw redirect({ to: "/" })
      }
      throw error
    }
  },
  loader: ({ params, context }) => {
    context.queryClient.prefetchQuery(projectDetailQueryOptions(params.id))
    context.queryClient.prefetchQuery(projectAccessQueryOptions(params.id))
    context.queryClient.prefetchQuery(projectMembersQueryOptions(params.id))
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: projectData } = useSuspenseQuery(projectDetailQueryOptions(id))
  const { data: accessData } = useSuspenseQuery(projectAccessQueryOptions(id))
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<
    "general" | "members" | "activity"
  >("general")

  const { project, packages } = projectData

  const hasProjectLevelAccess = accessData.hasProjectLevelAccess
  const canViewCommercial =
    accessData.access === "full" || accessData.access === "commercial"

  const defaultCurrency = getCurrencyForCountry(project.country ?? "") ?? "USD"

  const [drawerOpen, setDrawerOpen] = useState(false)
  const [packageName, setPackageName] = useState("")
  const [packageCurrency, setPackageCurrency] = useState(defaultCurrency)
  const [technicalWeight, setTechnicalWeight] = useState(50)
  const [contractors, setContractors] = useState<string[]>([])
  const [newContractorName, setNewContractorName] = useState("")

  const canCreatePackage = accessData.access === "full"

  const createPackage = useMutation({
    mutationFn: ({
      name,
      currency,
      technicalWeight,
      commercialWeight,
      contractors,
    }: {
      name: string
      currency: string
      technicalWeight: number
      commercialWeight: number
      contractors: string[]
    }) =>
      createPackageFn({
        data: {
          projectId: project.id,
          name,
          currency,
          technicalWeight,
          commercialWeight,
          contractors,
        },
      }),
    onSuccess: (newPackage) => {
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(project.id).queryKey,
      })
      queryClient.invalidateQueries({
        queryKey: projectsQueryOptions.queryKey,
      })
      closeDrawer()
      navigate({
        to: "/package/$id",
        params: { id: newPackage.id },
      })
    },
  })

  const closeDrawer = () => {
    setDrawerOpen(false)
    setPackageName("")
    setPackageCurrency(defaultCurrency)
    setTechnicalWeight(50)
    setContractors([])
    setNewContractorName("")
    createPackage.reset()
  }

  const handleAddContractor = () => {
    const name = newContractorName.trim()
    if (!name || contractors.includes(name)) return
    setContractors((prev) => [...prev, name])
    setNewContractorName("")
  }

  const handleRemoveContractor = (name: string) => {
    setContractors((prev) => prev.filter((c) => c !== name))
  }

  const handleCreatePackage = (e: React.FormEvent) => {
    e.preventDefault()
    const name = packageName.trim()
    if (!name || contractors.length < 2) return
    createPackage.mutate({
      name,
      currency: packageCurrency,
      technicalWeight,
      commercialWeight: 100 - technicalWeight,
      contractors,
    })
  }

  const openSettings = (
    tab: "general" | "members" | "activity" = "general"
  ) => {
    setSettingsTab(tab)
    setSettingsOpen(true)
  }

  const packagesList = useMemo(() => {
    if (packages.length === 0) {
      return (
        <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-8 text-center text-sm text-muted-foreground">
          No packages yet. Create one to get started.
        </div>
      )
    }

    return (
      <div className="border rounded-lg divide-y">
        {packages.map((pkg: PackageWithAssetCount) => {
          // Only show awarded info to users with commercial access
          const showAwarded = canViewCommercial && pkg.awardedContractorName
          return (
            <Link
              key={pkg.id}
              to="/package/$id"
              params={{ id: pkg.id }}
              className={`group flex flex-col px-3 py-2.5 hover:bg-muted/50 transition-colors ${
                showAwarded ? "bg-green-50" : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <BoxIcon size={16} className="text-muted-foreground shrink-0" />
                <div className="flex-1 min-w-0 flex items-center gap-2">
                  <span className="font-medium text-sm truncate">
                    {pkg.name}
                  </span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-xs text-muted-foreground">
                    {pkg.assetCount} {pkg.assetCount === 1 ? "asset" : "assets"}
                  </span>
                  <ChevronRightIcon
                    size={14}
                    className="text-muted-foreground/50 group-hover:text-muted-foreground transition-colors"
                  />
                </div>
              </div>
              {showAwarded && (
                <div className="ml-7 mt-1 text-xs text-green-700">
                  Awarded to {pkg.awardedContractorName}
                </div>
              )}
            </Link>
          )
        })}
      </div>
    )
  }, [packages, canViewCommercial])

  return (
    <>
      <div className="flex flex-1 overflow-hidden h-full">
        {/* Sidebar */}
        {hasProjectLevelAccess && (
          <ProjectSidebar projectId={id} onSettingsClick={openSettings} />
        )}

        {/* Main content */}
        <div className="flex-1 flex flex-col overflow-hidden content-shadow rounded-xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b-[0.5px] border-black/15 h-12 px-6">
            <span className="text-primary font-semibold text-16">
              Project overview
            </span>
            <AIChatButton />
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto">
            <div className="max-w-[600px] mx-auto p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Packages
                </h2>
                {canCreatePackage && (
                  <Button
                    onClick={() => setDrawerOpen(true)}
                    size="sm"
                    variant="outline"
                  >
                    <PlusIcon size={14} />
                    New package
                  </Button>
                )}
              </div>

              {packagesList}
            </div>
          </div>
        </div>
      </div>

      {hasProjectLevelAccess && (
        <ProjectSettingsDialog
          projectId={id}
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          defaultTab={settingsTab}
        />
      )}

      <Sheet open={drawerOpen} onOpenChange={(open) => !open && closeDrawer()}>
        <SheetContent className="min-w-[500px] sm:max-w-none overflow-y-auto">
          <form className="space-y-6" onSubmit={handleCreatePackage}>
            <SheetHeader>
              <SheetTitle>Create package</SheetTitle>
              <SheetDescription>
                Packages live inside your project and gather related assets.
              </SheetDescription>
            </SheetHeader>
            <div className="px-2 space-y-6">
              <div className="space-y-2">
                <Label htmlFor="package-name">Package name</Label>
                <Input
                  id="package-name"
                  value={packageName}
                  onChange={(e) => setPackageName(e.target.value)}
                  disabled={createPackage.isPending}
                  autoFocus
                />
              </div>
              <div className="space-y-2">
                <Label>Currency</Label>
                <CurrencySelect
                  value={packageCurrency}
                  onValueChange={setPackageCurrency}
                  disabled={createPackage.isPending}
                />
              </div>
              <div className="space-y-2">
                <Label>Score weighting</Label>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <span className="text-xs text-muted-foreground">
                      Technical
                    </span>
                    <div className="relative">
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        value={technicalWeight}
                        onChange={(e) => {
                          const val = Math.min(
                            100,
                            Math.max(0, parseInt(e.target.value) || 0)
                          )
                          setTechnicalWeight(val)
                        }}
                        disabled={createPackage.isPending}
                        className="pr-8"
                      />
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
                        %
                      </span>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <span className="text-xs text-muted-foreground">
                      Commercial
                    </span>
                    <div className="relative">
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        value={100 - technicalWeight}
                        onChange={(e) => {
                          const val = Math.min(
                            100,
                            Math.max(0, parseInt(e.target.value) || 0)
                          )
                          setTechnicalWeight(100 - val)
                        }}
                        disabled={createPackage.isPending}
                        className="pr-8"
                      />
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
                        %
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Contractors Section */}
              <div className="space-y-3">
                <StepTitle
                  title={`Contractors (${contractors.length})`}
                  complete={contractors.length >= 2}
                  required
                />

                {/* Add contractor input */}
                <div className="flex gap-2">
                  <Input
                    placeholder="e.g. BuildCorp International"
                    value={newContractorName}
                    onChange={(e) => setNewContractorName(e.target.value)}
                    disabled={createPackage.isPending}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault()
                        handleAddContractor()
                      }
                    }}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleAddContractor}
                    disabled={
                      createPackage.isPending || !newContractorName.trim()
                    }
                  >
                    <PlusIcon size={14} className="mr-1" />
                    Add
                  </Button>
                </div>

                {/* Contractor list */}
                {contractors.length > 0 ? (
                  <div className="border rounded-lg divide-y">
                    {contractors.map((name) => (
                      <div
                        key={name}
                        className="flex items-center gap-3 px-3 py-2.5"
                      >
                        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-muted">
                          <UserIcon
                            size={16}
                            className="text-muted-foreground"
                          />
                        </div>
                        <span className="font-medium text-sm flex-1">
                          {name}
                        </span>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="size-7 p-0 hover:bg-destructive/10 hover:text-destructive"
                          onClick={() => handleRemoveContractor(name)}
                          disabled={createPackage.isPending}
                        >
                          <X size={14} />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm text-muted-foreground">
                    No contractors yet. Add at least 2 to create the package.
                  </div>
                )}

                {contractors.length > 0 && contractors.length < 2 && (
                  <p className="text-sm text-amber-600">
                    Add at least {2 - contractors.length} more contractor
                    {contractors.length === 1 ? "" : "s"} to proceed
                  </p>
                )}
              </div>

              {createPackage.error && (
                <p className="text-sm text-red-500">
                  {createPackage.error instanceof Error
                    ? createPackage.error.message
                    : "Unable to create package."}
                </p>
              )}
            </div>
            <SheetFooter>
              <Button
                type="submit"
                disabled={
                  createPackage.isPending ||
                  !packageName.trim() ||
                  contractors.length < 2
                }
              >
                {createPackage.isPending ? "Creating..." : "Create package"}
              </Button>
            </SheetFooter>
          </form>
        </SheetContent>
      </Sheet>
    </>
  )
}
