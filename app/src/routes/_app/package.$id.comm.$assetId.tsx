import { createFileRoute, Outlet } from "@tanstack/react-router"
import {
  assetDetailQueryOptions,
  commercialEvaluationsQueryOptions,
  packageContractorsQueryOptions,
} from "@/lib/query-options"

export const Route = createFileRoute("/_app/package/$id/comm/$assetId")({
  loader: ({ params, context }) => {
    // Prefetch all data needed by child routes
    context.queryClient.prefetchQuery(assetDetailQueryOptions(params.assetId))
    context.queryClient.prefetchQuery(
      commercialEvaluationsQueryOptions(params.assetId)
    )
    context.queryClient.prefetchQuery(packageContractorsQueryOptions(params.id))
  },
  component: RouteComponent,
})

function RouteComponent() {
  return <Outlet />
}
