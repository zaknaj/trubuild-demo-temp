import { QueryClient, QueryCache, MutationCache } from "@tanstack/react-query"
import { setupRouterSsrQueryIntegration } from "@tanstack/react-router-ssr-query"
import { createRouter, useRouter } from "@tanstack/react-router"
import { toast } from "sonner"
import { AlertTriangle, RefreshCw, Home } from "lucide-react"

// Import the generated route tree
import { routeTree } from "./routeTree.gen"

// Default error component with retry and home navigation
function DefaultErrorComponent({ error }: { error: Error }) {
  const router = useRouter()

  const handleRetry = () => {
    router.invalidate()
  }

  const handleGoHome = () => {
    router.navigate({ to: "/" })
  }

  // Check if it's a "not found" type error
  const isNotFound =
    error.message?.toLowerCase().includes("not found") ||
    error.message?.toLowerCase().includes("does not exist")

  return (
    <div className="flex flex-col items-center justify-center p-8 text-center min-h-[300px]">
      <AlertTriangle className="size-12 text-amber-500 mb-4" />
      <h2 className="text-lg font-semibold mb-2">
        {isNotFound ? "Not Found" : "Something went wrong"}
      </h2>
      <p className="text-sm text-muted-foreground mb-6 max-w-md">
        {error.message || "An unexpected error occurred"}
      </p>
      <div className="flex gap-3">
        <button
          onClick={handleRetry}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground transition-colors"
        >
          <RefreshCw className="size-4" />
          Try again
        </button>
        <button
          onClick={handleGoHome}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <Home className="size-4" />
          Go home
        </button>
      </div>
    </div>
  )
}

// Create a new router instance
export const getRouter = () => {
  const queryClient = new QueryClient({
    queryCache: new QueryCache({
      onError: (error) => {
        console.error("[Query Error]", error)
        toast.error(error.message || "Something went wrong")
      },
    }),
    mutationCache: new MutationCache({
      onError: (error) => {
        console.error("[Mutation Error]", error)
        toast.error(error.message || "Something went wrong")
      },
    }),
    defaultOptions: {
      queries: {
        staleTime: 5 * 1000,
        gcTime: 5 * 60 * 1000,
        refetchOnWindowFocus: false,
      },
    },
  })

  const router = createRouter({
    routeTree,
    context: { queryClient },
    scrollRestoration: true,
    defaultPreload: "intent",
    defaultPreloadDelay: 0,
    defaultErrorComponent: DefaultErrorComponent,
  })

  setupRouterSsrQueryIntegration({
    router,
    queryClient,
  })

  return router
}

declare module "@tanstack/react-router" {
  interface Register {
    router: ReturnType<typeof getRouter>
  }
}
