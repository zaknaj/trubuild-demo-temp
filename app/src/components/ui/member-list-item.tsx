import { cn } from "@/lib/utils"
import { MemberAvatar, type MemberAvatarProps } from "./member-avatar"
import { Button } from "./button"
import { X, Loader2 } from "lucide-react"

export interface MemberListItemProps {
  name?: string | null
  email: string
  image?: string | null
  role?: string
  pending?: boolean
  avatarSize?: MemberAvatarProps["size"]
  showRemoveButton?: boolean
  isRemoving?: boolean
  onRemove?: () => void
  className?: string
  variant?: "default" | "pending"
}

export function MemberListItem({
  name,
  email,
  image,
  role,
  pending = false,
  avatarSize = "md",
  showRemoveButton = false,
  isRemoving = false,
  onRemove,
  className,
  variant = "default",
}: MemberListItemProps) {
  const isPending = pending || variant === "pending"

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-3 py-2",
        isPending && "bg-amber-50/50",
        className
      )}
    >
      <MemberAvatar
        name={name}
        email={email}
        image={image}
        size={avatarSize}
        pending={isPending}
      />
      <div className="flex-1 min-w-0">
        <p className="font-medium truncate text-xs">
          {isPending ? email : name || email}
        </p>
        {isPending ? (
          <p className="text-xs text-amber-600">Pending signup</p>
        ) : (
          name && (
            <p className="text-xs text-muted-foreground truncate">{email}</p>
          )
        )}
      </div>
      {role && (
        <span className="text-xs px-2 py-1 rounded-full bg-muted text-muted-foreground capitalize">
          {role.replace("_", " ")}
        </span>
      )}
      {showRemoveButton && onRemove && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onRemove}
          disabled={isRemoving}
          className="text-muted-foreground hover:text-destructive"
          aria-label={`Remove ${name || email}`}
        >
          {isRemoving ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <X className="size-4" />
          )}
        </Button>
      )}
    </div>
  )
}
