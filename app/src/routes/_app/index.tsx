import { createFileRoute } from "@tanstack/react-router"
import { SimpleHeader } from "@/components/SimpleHeader"

export const Route = createFileRoute("/_app/")({
  component: RouteComponent,
})

function RouteComponent() {
  return (
    <>
      <SimpleHeader title="Overview" />
      <div className="flex-1 overflow-auto p-6">
        {/* Overview content here */}
      </div>
    </>
  )
}
