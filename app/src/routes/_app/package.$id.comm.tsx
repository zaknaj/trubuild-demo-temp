import {
  createFileRoute,
  Link,
  Outlet,
  redirect,
  useMatches,
  useNavigate,
} from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import {
  packageAccessQueryOptions,
  packageDetailQueryOptions,
} from "@/lib/query-options"
import { ArrowLeft, Plus } from "lucide-react"
import { cn } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

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
  component: RouteComponent,
})

function RouteComponent() {
  const { id } = Route.useParams()
  const matches = useMatches()
  const navigate = useNavigate()

  const { data: packageData } = useQuery(packageDetailQueryOptions(id))

  if (!packageData) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner className="size-6 stroke-1" />
      </div>
    )
  }
  const assets = packageData.assets

  // Get the currently selected assetId from URL
  const currentAssetId = matches
    .map((m) => (m.params as { assetId?: string }).assetId)
    .find((assetId) => assetId)

  // Get the current asset if we have one
  const currentAsset = currentAssetId
    ? assets.find((a) => a.id === currentAssetId)
    : null

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
              className="top-back-link inline-flex items-center gap-1 hover:opacity-100"
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
            className="top-back-link h-8 px-2 -ml-2 inline-flex items-center gap-1 hover:opacity-100"
          >
            <ArrowLeft size={14} />
            Package summary
          </Link>
          <h1 className="mt-2 text-[24px] leading-8 font-semibold text-[#8D4BAF]">
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
              onClick={() =>
                navigate({
                  to: "/new-asset/$packageId",
                  params: { packageId: id },
                })
              }
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
    </div>
  )
}
