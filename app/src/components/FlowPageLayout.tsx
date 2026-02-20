import { type ReactNode } from "react"
import { useNavigate } from "@tanstack/react-router"
import { ArrowLeft, Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface FlowStep {
  id: string
  label: string
}

interface FlowPageLayoutProps {
  title: string
  context?: ReactNode
  backLabel: string
  backTo?: string
  onBack?: () => void
  steps?: readonly FlowStep[]
  currentStepId?: string
  children: ReactNode
}

export function FlowPageLayout({
  title,
  context,
  backLabel,
  backTo,
  onBack,
  steps,
  currentStepId,
  children,
}: FlowPageLayoutProps) {
  const navigate = useNavigate()

  const handleBackClick = () => {
    if (onBack) {
      onBack()
      return
    }
    if (window.history.length > 1) {
      window.history.back()
      return
    }
    if (backTo) {
      navigate({ to: backTo })
    }
  }

  return (
    <div className="h-screen w-full bg-[#F9F9F9] overflow-auto px-5 py-8">
      <div className="mx-auto w-full max-w-[550px]">
        <button
          className="h-8 px-2 -ml-2 inline-flex items-center gap-1 text-[13px] font-medium opacity-60 hover:opacity-100"
          onClick={handleBackClick}
          type="button"
        >
          <ArrowLeft size={14} />
          {backLabel}
        </button>

        <header className="mt-3 space-y-2.5">
          <h1 className="text-[32px] leading-[1.05] font-semibold tracking-tight text-black">
            {title}
          </h1>
          {context ? (
            <div className="text-[12px] text-black/50 leading-5">
              {context}
            </div>
          ) : null}
        </header>

        {steps && steps.length > 0 ? (
          <nav className="mt-6">
            <ol className="flex items-center gap-2.5">
            {steps.map((step, index) => {
              const currentIndex =
                currentStepId != null
                  ? steps.findIndex((item) => item.id === currentStepId)
                  : -1
              const isCurrent = step.id === currentStepId
              const isDone = currentIndex > index
              const isLast = index === steps.length - 1

              return (
                <li key={step.id} className="flex items-center gap-2.5 min-w-0 flex-1">
                  <div className="flex items-center gap-2 min-w-0">
                    <div
                      className={cn(
                        "size-6 rounded-full border flex items-center justify-center shrink-0 transition-colors",
                        isCurrent
                          ? "bg-black border-black text-white"
                          : isDone
                            ? "bg-black/10 border-black/20 text-black"
                            : "bg-transparent border-black/15 text-black/35"
                      )}
                    >
                      {isDone ? <Check size={12} /> : <span className="text-[11px]">{index + 1}</span>}
                    </div>
                    <span
                      className={cn(
                        "text-[12px] truncate transition-colors",
                        isCurrent ? "text-black font-medium" : "text-black/45"
                      )}
                    >
                      {step.label}
                    </span>
                  </div>
                  {!isLast ? <span className="h-px flex-1 bg-black/10 min-w-3" /> : null}
                </li>
              )
            })}
            </ol>
          </nav>
        ) : null}

        <main className="mt-7">
          {children}
        </main>
      </div>
    </div>
  )
}
