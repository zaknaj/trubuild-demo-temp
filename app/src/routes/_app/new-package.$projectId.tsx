import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import { PlusIcon, UserIcon, X } from "lucide-react"
import { CurrencySelect } from "@/components/CurrencySelect"
import { FlowPageLayout } from "@/components/FlowPageLayout"
import { StepTitle } from "@/components/ui/step-title"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { createPackageFn } from "@/fn/packages"
import {
  projectAccessQueryOptions,
  projectDetailQueryOptions,
  projectsQueryOptions,
} from "@/lib/query-options"
import { getCurrencyForCountry } from "@/lib/countries"
import { Spinner } from "@/components/ui/spinner"

const steps = [
  { id: "details", label: "Package details" },
  { id: "vendors", label: "Vendors" },
] as const

export const Route = createFileRoute("/_app/new-package/$projectId")({
  beforeLoad: async ({ params, context }) => {
    const accessData = await context.queryClient.ensureQueryData(
      projectAccessQueryOptions(params.projectId)
    )
    if (accessData.access === "none") {
      throw redirect({ to: "/" })
    }
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { projectId } = Route.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: projectData } = useQuery(projectDetailQueryOptions(projectId))
  const { data: accessData } = useQuery(projectAccessQueryOptions(projectId))
  if (!projectData || !accessData) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }
  const project = projectData.project

  const defaultCurrency = getCurrencyForCountry(project.country ?? "") ?? "USD"
  const canCreatePackage = accessData.access === "full"
  const [currentStep, setCurrentStep] = useState<(typeof steps)[number]["id"]>("details")

  const [packageName, setPackageName] = useState("")
  const [packageCurrency, setPackageCurrency] = useState(defaultCurrency)
  const [technicalWeight, setTechnicalWeight] = useState(50)
  const [contractors, setContractors] = useState<string[]>([])
  const [newContractorName, setNewContractorName] = useState("")

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
          projectId,
          name,
          currency,
          technicalWeight,
          commercialWeight,
          contractors,
        },
      }),
    onSuccess: async (newPackage) => {
      await queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(projectId).queryKey,
      })
      await queryClient.invalidateQueries({
        queryKey: projectsQueryOptions.queryKey,
      })
      navigate({
        to: "/package/$id",
        params: { id: newPackage.id },
      })
    },
  })

  const handleAddContractor = () => {
    const name = newContractorName.trim()
    if (!name || contractors.includes(name)) return
    setContractors((prev) => [...prev, name])
    setNewContractorName("")
  }

  const handleRemoveContractor = (name: string) => {
    setContractors((prev) => prev.filter((contractor) => contractor !== name))
  }

  return (
    <FlowPageLayout
      title="New package"
      context={
        <div className="space-y-0.5">
          <p>Project: {project.name}</p>
        </div>
      }
      backLabel={project.name}
      onBack={() => navigate({ to: "/project/$id", params: { id: projectId } })}
      steps={steps}
      currentStepId={currentStep}
    >
      {!canCreatePackage ? (
        <p className="text-sm text-black/55">
          You do not have permission to create packages in this project.
        </p>
      ) : (
        <form
          className="space-y-6"
          onSubmit={(e) => {
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
          }}
        >
          {currentStep === "details" ? (
            <div className="space-y-5">
              <div className="space-y-2">
                <Label htmlFor="package-name">Package name</Label>
                <Input
                  id="package-name"
                  value={packageName}
                  onChange={(e) => setPackageName(e.target.value)}
                  className="bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
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
                    <span className="text-xs text-muted-foreground">Technical</span>
                    <div className="relative">
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        value={technicalWeight}
                        onChange={(e) => {
                          const value = Math.min(100, Math.max(0, parseInt(e.target.value) || 0))
                          setTechnicalWeight(value)
                        }}
                        className="pr-8 bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
                        disabled={createPackage.isPending}
                      />
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
                        %
                      </span>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <span className="text-xs text-muted-foreground">Commercial</span>
                    <div className="relative">
                      <Input
                        type="number"
                        min={0}
                        max={100}
                        value={100 - technicalWeight}
                        onChange={(e) => {
                          const value = Math.min(100, Math.max(0, parseInt(e.target.value) || 0))
                          setTechnicalWeight(100 - value)
                        }}
                        className="pr-8 bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
                        disabled={createPackage.isPending}
                      />
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
                        %
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <Button
                  type="button"
                  className="w-full bg-black text-white hover:bg-black/90"
                  onClick={() => setCurrentStep("vendors")}
                  disabled={!packageName.trim() || createPackage.isPending}
                >
                  Continue
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              <div className="space-y-3">
                <StepTitle
                  title={`Vendors (${contractors.length})`}
                  complete={contractors.length >= 2}
                  required
                />
                <div className="flex gap-2">
                  <Input
                    placeholder="e.g. BuildCorp International"
                    value={newContractorName}
                    onChange={(e) => setNewContractorName(e.target.value)}
                    className="bg-white border-black/15 focus-visible:border-black/35 focus-visible:ring-black/15"
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
                    disabled={createPackage.isPending || !newContractorName.trim()}
                  >
                    <PlusIcon size={14} className="mr-1" />
                    Add
                  </Button>
                </div>
                {contractors.length > 0 ? (
                  <div className="border rounded-lg divide-y">
                    {contractors.map((name) => (
                      <div key={name} className="flex items-center gap-3 px-3 py-2.5">
                        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-muted">
                          <UserIcon size={16} className="text-muted-foreground" />
                        </div>
                        <span className="font-medium text-sm flex-1">{name}</span>
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
                    No vendors yet. Add at least 2 to create the package.
                  </div>
                )}
                {contractors.length > 0 && contractors.length < 2 ? (
                  <p className="text-sm text-amber-600">
                    Add at least {2 - contractors.length} more vendor
                    {contractors.length === 1 ? "" : "s"} to continue.
                  </p>
                ) : null}
              </div>
              {createPackage.error ? (
                <p className="text-sm text-red-500">
                  {createPackage.error instanceof Error
                    ? createPackage.error.message
                    : "Unable to create package."}
                </p>
              ) : null}
              <div className="space-y-2">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  onClick={() => setCurrentStep("details")}
                  disabled={createPackage.isPending}
                >
                  Back
                </Button>
                <Button
                  type="submit"
                  className="w-full bg-black text-white hover:bg-black/90"
                  disabled={createPackage.isPending || !packageName.trim() || contractors.length < 2}
                >
                  {createPackage.isPending ? "Creating..." : "Create package"}
                </Button>
              </div>
            </div>
          )}
        </form>
      )}
    </FlowPageLayout>
  )
}
