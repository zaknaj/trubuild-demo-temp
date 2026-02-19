import { CheckCircle2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface StepTitleProps {
  title: string
  complete?: boolean
  description?: string
  className?: string
  required?: boolean
}

export function StepTitle({
  title,
  complete = false,
  description,
  className,
  required = false,
}: StepTitleProps) {
  return (
    <div className={cn("space-y-0.5", className)}>
      <div className="flex items-center gap-2">
        <h4 className="font-semibold text-sm">
          {title}
          {required && !complete && (
            <span className="text-destructive ml-0.5">*</span>
          )}
        </h4>
        {complete && <CheckCircle2 className="size-4 text-green-600" />}
      </div>
      {description && (
        <p className="text-sm text-muted-foreground">{description}</p>
      )}
    </div>
  )
}
