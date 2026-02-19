import { cn } from "@/lib/utils"
import { Clock } from "lucide-react"

export interface MemberAvatarProps {
  name?: string | null
  email: string
  image?: string | null
  size?: "sm" | "md" | "lg"
  pending?: boolean
  className?: string
}

const sizeClasses = {
  sm: "size-6",
  md: "size-7",
  lg: "size-9",
}

const textSizeClasses = {
  sm: "text-[9px]",
  md: "text-[10px]",
  lg: "text-xs",
}

const iconSizeClasses = {
  sm: "size-3",
  md: "size-3.5",
  lg: "size-4",
}

export function MemberAvatar({
  name,
  email,
  image,
  size = "md",
  pending = false,
  className,
}: MemberAvatarProps) {
  const initials = name?.charAt(0) || email.charAt(0)

  if (pending) {
    return (
      <div
        className={cn(
          "rounded-full bg-amber-100 flex items-center justify-center",
          sizeClasses[size],
          className
        )}
      >
        <Clock className={cn("text-amber-600", iconSizeClasses[size])} />
      </div>
    )
  }

  if (image) {
    return (
      <img
        src={image}
        alt={name || email}
        className={cn(
          "rounded-full object-cover",
          sizeClasses[size],
          className
        )}
      />
    )
  }

  return (
    <div
      className={cn(
        "rounded-full bg-muted flex items-center justify-center font-medium uppercase text-muted-foreground",
        sizeClasses[size],
        textSizeClasses[size],
        className
      )}
    >
      {initials}
    </div>
  )
}
