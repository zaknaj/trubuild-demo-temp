import { Component, type ErrorInfo, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { AlertTriangle, RefreshCw } from "lucide-react"

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onReset?: () => void
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * Error Boundary component to catch and handle React errors.
 * Use this to wrap feature components to prevent entire app crashes.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    this.props.onReset?.()
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="flex flex-col items-center justify-center p-8 text-center">
          <AlertTriangle className="size-12 text-amber-500 mb-4" />
          <h2 className="text-lg font-semibold mb-2">Something went wrong</h2>
          <p className="text-sm text-muted-foreground mb-4 max-w-md">
            {this.state.error?.message || "An unexpected error occurred"}
          </p>
          <Button onClick={this.handleReset} variant="outline" size="sm">
            <RefreshCw className="size-4 mr-2" />
            Try again
          </Button>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Minimal error fallback for smaller UI sections
 */
export function MinimalErrorFallback({
  error,
  onRetry,
}: {
  error?: Error | null
  onRetry?: () => void
}) {
  return (
    <div className="p-4 text-center text-sm text-muted-foreground">
      <p className="text-red-500 mb-2">Failed to load</p>
      {error?.message && (
        <p className="text-xs mb-2 opacity-70">{error.message}</p>
      )}
      {onRetry && (
        <button
          onClick={onRetry}
          className="text-xs underline hover:no-underline"
        >
          Retry
        </button>
      )}
    </div>
  )
}
