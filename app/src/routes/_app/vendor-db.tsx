import { createFileRoute } from "@tanstack/react-router"
import { SimpleHeader } from "@/components/SimpleHeader"

export const Route = createFileRoute("/_app/vendor-db")({
  component: RouteComponent,
})

function RouteComponent() {
  return (
    <>
      <SimpleHeader title="Vendor Database" />
      <div className="flex-1 overflow-auto p-6">
        <p className="text-sm text-muted-foreground">Coming soon.</p>
      </div>
    </>
  )
}
