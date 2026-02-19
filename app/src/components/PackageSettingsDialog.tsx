import { useState, useEffect, useRef } from "react"
import { useNavigate } from "@tanstack/react-router"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { cn } from "@/lib/utils"
import { Archive } from "lucide-react"
import { MemberAvatar } from "@/components/ui/member-avatar"
import { MemberListItem } from "@/components/ui/member-list-item"
import {
  useSuspenseQuery,
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  packageDetailQueryOptions,
  packageMembersQueryOptions,
  packageAccessQueryOptions,
  orgMembersQueryOptions,
  projectDetailQueryOptions,
  archivedPackagesQueryOptions,
} from "@/lib/query-options"
import { addPackageMemberFn, removePackageMemberFn } from "@/fn/members"
import {
  archivePackageFn,
  renamePackageFn,
  updatePackageCurrencyFn,
} from "@/fn/packages"
import { CurrencySelect } from "@/components/CurrencySelect"
import type { PackageMember } from "@/lib/types"
import type { PackageRole } from "@/lib/permissions"
import { toast } from "sonner"

interface PackageSettingsDialogProps {
  packageId: string
  open: boolean
  onOpenChange: (open: boolean) => void
  defaultTab?: "general" | "members" | "activity"
}

export function PackageSettingsDialog({
  packageId,
  open,
  onOpenChange,
  defaultTab = "general",
}: PackageSettingsDialogProps) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: packageData } = useSuspenseQuery(
    packageDetailQueryOptions(packageId)
  )
  const { data: members } = useSuspenseQuery(
    packageMembersQueryOptions(packageId)
  )
  const { data: accessInfo } = useSuspenseQuery(
    packageAccessQueryOptions(packageId)
  )
  const { data: orgMembers = [] } = useQuery({
    ...orgMembersQueryOptions,
    enabled: open,
  })

  const { package: pkg, project } = packageData

  const [activeTab, setActiveTab] = useState<
    "general" | "members" | "activity"
  >(defaultTab)
  const [email, setEmail] = useState("")
  const [role, setRole] = useState<PackageRole>("commercial_team")
  const [removingEmail, setRemovingEmail] = useState<string | null>(null)
  const [isArchiveDialogOpen, setIsArchiveDialogOpen] = useState(false)
  const [packageName, setPackageName] = useState("")
  const [packageCurrency, setPackageCurrency] = useState("")
  const emailInputRef = useRef<HTMLInputElement>(null)

  const canInvite = accessInfo.access === "full"
  const canArchive = accessInfo.access === "full"
  const canRename = accessInfo.access === "full"

  useEffect(() => {
    setPackageName(pkg.name)
  }, [pkg.name])

  useEffect(() => {
    setPackageCurrency(pkg.currency ?? "")
  }, [pkg.currency])

  useEffect(() => {
    if (open) {
      setActiveTab(defaultTab)
    }
  }, [open, defaultTab])

  // Filter org members who are not already package members
  const packageMemberEmails = new Set(
    members.map((m: PackageMember) => m.email)
  )
  const availableOrgMembers = orgMembers.filter(
    (m) => !packageMemberEmails.has(m.email)
  )

  const addMember = useMutation({
    mutationFn: (data: { email: string; role: PackageRole }) =>
      addPackageMemberFn({ data: { packageId: pkg.id, ...data } }),
    onSuccess: () => {
      toast.success("Invitation sent")
      queryClient.invalidateQueries({
        queryKey: packageMembersQueryOptions(packageId).queryKey,
      })
      setEmail("")
      setRole("commercial_team")
    },
  })

  const removeMember = useMutation({
    mutationFn: (memberEmail: string) =>
      removePackageMemberFn({
        data: { packageId: pkg.id, email: memberEmail },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: packageMembersQueryOptions(packageId).queryKey,
      })
      setRemovingEmail(null)
    },
    onError: () => {
      setRemovingEmail(null)
    },
  })

  const archivePackage = useMutation({
    mutationFn: () => archivePackageFn({ data: { packageId: pkg.id } }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(project.id).queryKey,
      })
      queryClient.invalidateQueries({
        queryKey: archivedPackagesQueryOptions.queryKey,
      })
      onOpenChange(false)
      navigate({ to: "/project/$id", params: { id: project.id } })
    },
  })

  const renamePackage = useMutation({
    mutationFn: (name: string) =>
      renamePackageFn({ data: { packageId: pkg.id, name } }),
    onMutate: () => {
      toast.loading("Renaming package...", { id: "rename-package" })
    },
    onSuccess: () => {
      toast.success("Package renamed", { id: "rename-package" })
      queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(packageId).queryKey,
      })
      queryClient.invalidateQueries({
        queryKey: projectDetailQueryOptions(project.id).queryKey,
      })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to rename package",
        { id: "rename-package" }
      )
    },
  })

  const updateCurrency = useMutation({
    mutationFn: (currency: string) =>
      updatePackageCurrencyFn({ data: { packageId: pkg.id, currency } }),
    onMutate: () => {
      toast.loading("Updating currency...", { id: "update-package-currency" })
    },
    onSuccess: () => {
      toast.success("Currency updated", { id: "update-package-currency" })
      queryClient.invalidateQueries({
        queryKey: packageDetailQueryOptions(packageId).queryKey,
      })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update currency",
        { id: "update-package-currency" }
      )
    },
  })

  const handleRename = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedName = packageName.trim()
    if (!trimmedName || trimmedName === pkg.name) return
    renamePackage.mutate(trimmedName)
  }

  const handleAddMember = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedEmail = email.trim()
    if (!trimmedEmail) return
    addMember.mutate({ email: trimmedEmail, role })
  }

  const handleRemove = (memberEmail: string) => {
    setRemovingEmail(memberEmail)
    removeMember.mutate(memberEmail)
  }

  const activeMembers = members.filter((m: PackageMember) => m.userId !== null)
  const pendingMembers = members.filter((m: PackageMember) => m.userId === null)

  return (
    <>
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent className="w-fit max-w-fit p-0 overflow-hidden">
          <SheetTitle className="sr-only">Package Settings</SheetTitle>
          <SheetDescription className="sr-only">
            Manage package settings and members
          </SheetDescription>
          <div className="flex h-full">
            {/* Sidenav */}
            <div className="w-[200px] border-r flex flex-col shrink-0">
              <h2 className="text-lg font-medium p-4">Package Settings</h2>
              <div className="flex-1 flex flex-col gap-px py-2 px-2">
                <button
                  onClick={() => setActiveTab("general")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "general" && "active"
                  )}
                >
                  General
                </button>
                <button
                  onClick={() => setActiveTab("members")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "members" && "active"
                  )}
                >
                  Members
                </button>
                <button
                  onClick={() => setActiveTab("activity")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "activity" && "active"
                  )}
                >
                  Activity
                </button>
              </div>
            </div>
            {/* Content */}
            <div className="w-[550px] overflow-y-auto p-6 min-w-0 shrink-0">
              {activeTab === "general" && (
                <div className="space-y-6">
                  <h2 className="text-lg font-medium">General</h2>

                  {/* Rename Package */}
                  {canRename && (
                    <div className="border rounded-lg p-4 space-y-3">
                      <div>
                        <p className="font-medium">Package Name</p>
                        <p className="text-sm text-muted-foreground">
                          Change the display name of this package
                        </p>
                      </div>
                      <form onSubmit={handleRename} className="flex gap-2">
                        <Input
                          value={packageName}
                          onChange={(e) => setPackageName(e.target.value)}
                          disabled={renamePackage.isPending}
                          className="max-w-xs"
                        />
                        <Button
                          type="submit"
                          size="sm"
                          disabled={
                            renamePackage.isPending ||
                            !packageName.trim() ||
                            packageName.trim() === pkg.name
                          }
                        >
                          {renamePackage.isPending ? "Saving..." : "Save"}
                        </Button>
                      </form>
                    </div>
                  )}

                  {/* Currency */}
                  {canRename && (
                    <div className="border rounded-lg p-4 space-y-3">
                      <div>
                        <p className="font-medium">Currency</p>
                        <p className="text-sm text-muted-foreground">
                          The currency used for this package
                        </p>
                      </div>
                      <CurrencySelect
                        value={packageCurrency}
                        onValueChange={(newCurrency) => {
                          setPackageCurrency(newCurrency)
                          updateCurrency.mutate(newCurrency)
                        }}
                        disabled={updateCurrency.isPending}
                        className="max-w-xs"
                      />
                    </div>
                  )}

                  {/* Danger Zone */}
                  {canArchive && (
                    <div className="border border-destructive/30 rounded-lg p-4 bg-destructive/5">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="font-medium">Archive this package</p>
                          <p className="text-sm text-muted-foreground">
                            The package will be hidden from view. You can
                            restore it later.
                          </p>
                        </div>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => setIsArchiveDialogOpen(true)}
                          className="gap-1.5 shrink-0"
                        >
                          <Archive className="size-4" />
                          Archive
                        </Button>
                      </div>
                      {archivePackage.error && (
                        <p className="text-sm text-red-500 mt-2">
                          {archivePackage.error instanceof Error
                            ? archivePackage.error.message
                            : "Failed to archive package"}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "members" && (
                <div className="space-y-4">
                  <h2 className="text-lg font-medium">Members</h2>
                  {canInvite && (
                    <form
                      onSubmit={handleAddMember}
                      className="flex items-end gap-2 pb-4 border-b"
                    >
                      <div className="flex-1">
                        <label
                          htmlFor="invite-email"
                          className="text-xs text-muted-foreground mb-1 block"
                        >
                          Email
                        </label>
                        <Input
                          ref={emailInputRef}
                          id="invite-email"
                          type="email"
                          placeholder="colleague@example.com"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          disabled={addMember.isPending}
                        />
                      </div>
                      <div className="w-40">
                        <label
                          htmlFor="invite-role"
                          className="text-xs text-muted-foreground mb-1 block"
                        >
                          Role
                        </label>
                        <Select
                          value={role}
                          onValueChange={(v) => setRole(v as PackageRole)}
                          disabled={addMember.isPending}
                        >
                          <SelectTrigger id="invite-role" className="w-full">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="package_lead">
                              Package Lead
                            </SelectItem>
                            <SelectItem value="commercial_team">
                              Commercial Team
                            </SelectItem>
                            <SelectItem value="technical_team">
                              Technical Team
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <Button
                        type="submit"
                        disabled={addMember.isPending || !email.trim()}
                        size="sm"
                      >
                        {addMember.isPending ? "Inviting..." : "Invite"}
                      </Button>
                    </form>
                  )}

                  {addMember.error && (
                    <div className="text-sm text-red-500 bg-red-50 p-3 rounded-md">
                      {addMember.error instanceof Error
                        ? addMember.error.message
                        : "Failed to send invitation"}
                    </div>
                  )}

                  <div className="border rounded-lg divide-y">
                    {members.length === 0 ? (
                      <div className="p-4 text-sm text-muted-foreground text-center">
                        No members yet
                      </div>
                    ) : (
                      <>
                        {activeMembers.map((m: PackageMember) => (
                          <MemberListItem
                            key={m.id}
                            name={m.userName}
                            email={m.email}
                            image={m.userImage}
                            role={m.role}
                            showRemoveButton={canInvite}
                            isRemoving={removingEmail === m.email}
                            onRemove={() => handleRemove(m.email)}
                          />
                        ))}
                        {pendingMembers.map((m: PackageMember) => (
                          <MemberListItem
                            key={m.id}
                            name={m.userName}
                            email={m.email}
                            pending
                            role={m.role}
                            showRemoveButton={canInvite}
                            isRemoving={removingEmail === m.email}
                            onRemove={() => handleRemove(m.email)}
                          />
                        ))}
                      </>
                    )}
                  </div>
                  {removeMember.error && (
                    <p className="text-sm text-red-500">
                      {removeMember.error instanceof Error
                        ? removeMember.error.message
                        : "Failed to remove member"}
                    </p>
                  )}

                  {availableOrgMembers.length > 0 && canInvite && (
                    <div className="space-y-2">
                      <Label className="mb-4 mt-10">Quick Invite</Label>
                      <div className="border rounded-lg divide-y max-h-32 overflow-y-auto">
                        {availableOrgMembers.map((m) => (
                          <div
                            key={m.id}
                            className="flex items-center gap-2 w-full px-3 py-2 text-sm"
                          >
                            <MemberAvatar
                              name={m.userName}
                              email={m.email}
                              image={m.userImage}
                            />
                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate text-xs">
                                {m.userName || m.email}
                              </p>
                              {m.userName && (
                                <p className="text-xs text-muted-foreground truncate">
                                  {m.email}
                                </p>
                              )}
                            </div>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="shrink-0 h-7 text-xs"
                              onClick={() => {
                                setEmail(m.email)
                                emailInputRef.current?.scrollIntoView({
                                  behavior: "smooth",
                                  block: "center",
                                })
                                emailInputRef.current?.focus()
                              }}
                            >
                              Invite
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "activity" && (
                <div className="space-y-4">
                  <h2 className="text-lg font-medium">Activity</h2>
                  <div className="border rounded-lg p-8 text-center text-muted-foreground">
                    <p className="text-sm">Activity log coming soon</p>
                    <p className="text-xs mt-1">
                      Track changes and updates to this package
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <AlertDialog
        open={isArchiveDialogOpen}
        onOpenChange={setIsArchiveDialogOpen}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Archive package?</AlertDialogTitle>
            <AlertDialogDescription>
              This will archive "{pkg.name}". The package will be hidden from
              view but can be restored later from the archived section.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={archivePackage.isPending}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => archivePackage.mutate()}
              disabled={archivePackage.isPending}
            >
              {archivePackage.isPending ? "Archiving..." : "Archive package"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
