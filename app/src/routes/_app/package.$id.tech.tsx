import { useState, useEffect } from "react"
import { createFileRoute, Outlet, redirect } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import {
  technicalEvaluationsQueryOptions,
  packageContractorsQueryOptions,
  packageAccessQueryOptions,
} from "@/lib/query-options"
import useStore from "@/lib/store"
import { BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { TechSetupWizard } from "@/components/TechSetupWizard"
import { PackageContentHeader } from "@/components/PackageContentHeader"

type TechnicalEvaluation = {
  id: string
  packageId: string
  roundNumber: number
  roundName: string
  data: unknown
  createdAt: Date
  updatedAt: Date
}

export const Route = createFileRoute("/_app/package/$id/tech")({
  beforeLoad: async ({ params, context }) => {
    // Check technical access before loading the route
    const accessData = await context.queryClient.ensureQueryData(
      packageAccessQueryOptions(params.id)
    )
    if (accessData.access !== "full" && accessData.access !== "technical") {
      throw redirect({ to: "/package/$id", params: { id: params.id } })
    }
  },
  loader: ({ params, context }) => {
    context.queryClient.prefetchQuery(
      technicalEvaluationsQueryOptions(params.id)
    )
    context.queryClient.prefetchQuery(packageContractorsQueryOptions(params.id))
  },
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()

  const { data: evaluations } = useSuspenseQuery(
    technicalEvaluationsQueryOptions(id)
  ) as {
    data: TechnicalEvaluation[]
  }
  const { data: contractors } = useSuspenseQuery(
    packageContractorsQueryOptions(id)
  )

  // Get/set round from Zustand store
  const selectedRoundId = useStore((s) => s.selectedTechRound[id])
  const setTechRound = useStore((s) => s.setTechRound)

  // State for setup wizard
  const [isSetupWizardOpen, setIsSetupWizardOpen] = useState(false)

  // Auto-select latest round when no round is stored or stored round is invalid
  useEffect(() => {
    if (evaluations.length > 0) {
      const storedRoundValid =
        selectedRoundId && evaluations.some((e) => e.id === selectedRoundId)
      if (!storedRoundValid) {
        setTechRound(id, evaluations[0].id)
      }
    }
  }, [evaluations, selectedRoundId, setTechRound, id])

  // Get current round - use stored, or default to latest
  const currentRound = selectedRoundId
    ? evaluations.find((e) => e.id === selectedRoundId)
    : evaluations[0]

  const hasEvaluations = evaluations.length > 0

  const handleRoundSelect = (roundId: string) => {
    setTechRound(id, roundId)
  }

  // Empty state - no evaluations yet
  if (!hasEvaluations) {
    return (
      <>
        <div className="flex flex-col items-center justify-center h-full p-12 text-center">
          <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
            <BarChart3 className="size-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">
            No technical evaluation yet
          </h3>
          <p className="text-muted-foreground mb-6 max-w-sm">
            Start a technical evaluation to analyze and compare contractor
            proposals.
          </p>
          <Button onClick={() => setIsSetupWizardOpen(true)}>
            Start Technical Evaluation
          </Button>
        </div>

        <TechSetupWizard
          open={isSetupWizardOpen}
          onOpenChange={setIsSetupWizardOpen}
          packageId={id}
          contractors={contractors}
        />
      </>
    )
  }

  const rounds = evaluations.map((e) => ({
    id: e.id,
    roundName: e.roundName,
  }))

  return (
    <div className="flex flex-1 flex-col overflow-hidden h-full">
      <PackageContentHeader
        variant="technical"
        packageId={id}
        rounds={rounds}
        currentRound={
          currentRound
            ? { id: currentRound.id, roundName: currentRound.roundName }
            : undefined
        }
        onRoundSelect={handleRoundSelect}
        onNewRound={() => setIsSetupWizardOpen(true)}
      />
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>

      <TechSetupWizard
        open={isSetupWizardOpen}
        onOpenChange={setIsSetupWizardOpen}
        packageId={id}
        contractors={contractors}
      />
    </div>
  )
}
