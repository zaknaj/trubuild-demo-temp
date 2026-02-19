import { createFileRoute } from "@tanstack/react-router"

export const Route = createFileRoute("/_app/package/$id/tech/docs")({
  component: RouteComponent,
})

function RouteComponent() {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-8 bg-background sticky top-0 z-10">
        <h2 className="text-base font-semibold">Documents</h2>
      </div>

      <div className="p-6">
        <p className="text-muted-foreground">No documents uploaded yet.</p>
      </div>
    </div>
  )
}
