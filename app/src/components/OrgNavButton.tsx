import { useState } from "react"
import { useNavigate, Link } from "@tanstack/react-router"
import { setActiveOrgFn } from "@/fn/auth"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  useMutation,
  useSuspenseQuery,
  useQueryClient,
} from "@tanstack/react-query"
import { orgsQueryOptions, sessionQueryOptions } from "@/lib/query-options"
import { cn } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"
import { Button } from "./ui/button"
import { SettingsDialog } from "@/components/SettingsDialog"

export const OrgNavButton = () => {
  const [open, setOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const { data: orgs } = useSuspenseQuery(orgsQueryOptions)
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const activeOrgId = session?.session?.activeOrganizationId
  const activeOrganization = orgs.find((org) => org.id === activeOrgId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const switchOrg = useMutation({
    mutationFn: async (organizationId: string) => {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      await setActiveOrgFn({ data: { organizationId } })
      return organizationId
    },
    onSuccess: () => {
      // Clear all cached data except session/orgs when switching org
      queryClient.removeQueries({
        predicate: (query) => {
          const key = query.queryKey[0]
          return key !== "session" && key !== "organizations"
        },
      })
      queryClient.invalidateQueries({
        queryKey: sessionQueryOptions.queryKey,
      })
      setOpen(false)
      navigate({ to: "/" })
    },
  })

  const isSwitching = switchOrg.isPending
  const switchingOrgId = switchOrg.variables

  return (
    <>
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger asChild>
          <Button variant="navGhost" className="-ml-2 flex items-center gap-2">
            {activeOrganization?.logo && (
              <div className="size-6 rounded-full bg-white/20"></div>
            )}
            <span className="font-medium first-letter:uppercase">
              {activeOrganization?.name}
            </span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="min-w-56 w-fit text-13">
          <DropdownMenuLabel>My organizations</DropdownMenuLabel>
          {orgs.map((org) => {
            const isCurrentOrg = org.id === activeOrgId
            const isSwitchingToThis = isSwitching && switchingOrgId === org.id
            const isDisabled = isCurrentOrg || isSwitching

            return (
              <DropdownMenuItem
                key={org.id}
                className={cn(
                  "whitespace-nowrap",
                  isDisabled && "pointer-events-none opacity-50"
                )}
                onSelect={(e) => {
                  e.preventDefault()
                  if (!isDisabled) {
                    switchOrg.mutate(org.id)
                  }
                }}
              >
                {isSwitchingToThis ? (
                  <Spinner className="size-6 stroke-1 opacity-50 " />
                ) : (
                  <div className={cn("size-6 rounded-full bg-black/10")} />
                )}
                {org.name}
              </DropdownMenuItem>
            )
          })}
          <DropdownMenuSeparator />

          <DropdownMenuItem
            className={cn(isSwitching && "pointer-events-none opacity-50")}
            asChild
          >
            <Link to="/create-org">Create an organization</Link>
          </DropdownMenuItem>
          <DropdownMenuItem
            className={cn(isSwitching && "pointer-events-none opacity-50")}
            onSelect={() => setSettingsOpen(true)}
          >
            Settings
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  )
}
