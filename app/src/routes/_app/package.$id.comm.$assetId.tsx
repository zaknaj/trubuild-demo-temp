import { createFileRoute, Outlet } from "@tanstack/react-router"

export const Route = createFileRoute("/_app/package/$id/comm/$assetId")({
  component: RouteComponent,
})

function RouteComponent() {
  return <Outlet />
}
