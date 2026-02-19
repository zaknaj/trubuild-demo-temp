import { useState } from "react"
import { Button } from "@/components/ui/button"
import { authClient } from "@/auth/auth-client"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useSuspenseQuery, useQueryClient } from "@tanstack/react-query"
import { sessionQueryOptions } from "@/lib/query-options"
import { SettingsIcon, LogOutIcon } from "lucide-react"
import { SettingsDialog } from "@/components/SettingsDialog"

export const AccountNavButton = () => {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const user = session?.user
  const queryClient = useQueryClient()

  const handleLogout = async () => {
    await authClient.signOut()
    queryClient.clear()
    // Hard navigate to ensure auth cookies + loader state are definitely fresh.
    window.location.href = "/login"
  }

  return (
    <>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="navGhost" className="flex items-center gap-2">
            {user?.image ? (
              <img
                src={user.image}
                alt={user.name || "User avatar"}
                className="size-6 rounded-full object-cover"
              />
            ) : (
              <div className="size-6 rounded-full bg-white/20 flex items-center justify-center text-xs font-medium uppercase">
                {user?.name?.charAt(0) || user?.email?.charAt(0)}
              </div>
            )}
            {user?.name ? (
              <span className="font-medium capitalize">{user.name}</span>
            ) : (
              <span className="text-sm">{user?.email}</span>
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-40 w-fit text-13">
          <DropdownMenuLabel className="whitespace-nowrap">
            {user?.email}
          </DropdownMenuLabel>
          <DropdownMenuItem onSelect={() => setSettingsOpen(true)}>
            <SettingsIcon size="10" />
            Settings
          </DropdownMenuItem>
          <DropdownMenuItem variant="destructive" onSelect={handleLogout}>
            <LogOutIcon size="10" />
            Log out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  )
}
