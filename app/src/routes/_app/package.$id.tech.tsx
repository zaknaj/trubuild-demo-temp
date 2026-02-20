import { useEffect } from "react"
import {
  createFileRoute,
  Link,
  Outlet,
  redirect,
  useNavigate,
} from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import {
  technicalEvaluationsQueryOptions,
  packageAccessQueryOptions,
} from "@/lib/query-options"
import useStore from "@/lib/store"
import { ArrowLeft, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"

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
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const navigate = useNavigate()

  const { data: evaluations } = useQuery(technicalEvaluationsQueryOptions(id))

  if (!evaluations) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }
  // Get/set round from Zustand store
  const selectedRoundId = useStore((s) => s.selectedTechRound[id])
  const setTechRound = useStore((s) => s.setTechRound)

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

  const hasEvaluations = evaluations.length > 0

  // Empty state - no evaluations yet
  if (!hasEvaluations) {
    return (
      <div className="flex flex-1 flex-col">
        <div className="px-6 pt-6">
          <Link
            to="/package/$id"
            params={{ id }}
            className="top-back-link inline-flex items-center gap-1 hover:opacity-100"
          >
            <ArrowLeft size={14} />
            Package summary
          </Link>
        </div>
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
          <Button
            onClick={() =>
              navigate({
                to: "/new-tech-evaluation/$packageId",
                params: { packageId: id },
              })
            }
          >
            Start Technical Evaluation
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden h-full">
      <div className="px-6 pt-6 shrink-0">
        <Link
          to="/package/$id"
          params={{ id }}
          className="top-back-link inline-flex items-center gap-1 hover:opacity-100"
        >
          <ArrowLeft size={14} />
          Package summary
        </Link>
      </div>
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>
    </div>
  )
}
