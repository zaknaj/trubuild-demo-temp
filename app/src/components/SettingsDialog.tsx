import { updateOrganizationFn, updateProfileFn } from "@/fn/auth"
import {
  inviteMemberFn,
  removeOrgMemberFn,
  cancelOrgInvitationFn,
} from "@/fn/members"
import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import { cn, getOrgCountry } from "@/lib/utils"
import { CountrySelect } from "@/components/CountrySelect"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  useSuspenseQuery,
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  orgMembersQueryOptions,
  orgPendingInvitesQueryOptions,
  orgsQueryOptions,
  sessionQueryOptions,
} from "@/lib/query-options"
import type { OrgMember, OrgPendingInvite } from "@/lib/types"
import { toast } from "sonner"
import { Upload, Trash2 } from "lucide-react"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

const orgRoleLabels: Record<string, string> = {
  owner: "Admin",
  admin: "Project Owner",
  member: "Member",
}

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { data: session } = useSuspenseQuery(sessionQueryOptions)
  const { data: orgs } = useSuspenseQuery(orgsQueryOptions)
  // Only query org members and invites when dialog is open (requires active org)
  const { data: members = [] } = useQuery({
    ...orgMembersQueryOptions,
    enabled: open,
  })
  const { data: pendingInvites = [] } = useQuery({
    ...orgPendingInvitesQueryOptions,
    enabled: open,
  })
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<
    "profile" | "organization" | "members"
  >("profile")
  const [email, setEmail] = useState("")
  const [role, setRole] = useState<"owner" | "admin" | "member">("member")
  const profileImageInputRef = useRef<HTMLInputElement>(null)
  const orgLogoInputRef = useRef<HTMLInputElement>(null)

  const activeOrg = orgs?.find(
    (o) => o.id === session?.session?.activeOrganizationId
  )
  const [orgName, setOrgName] = useState(activeOrg?.name ?? "")
  const [profileName, setProfileName] = useState(session?.user?.name ?? "")
  const [orgCountry, setOrgCountry] = useState<string>(
    getOrgCountry(activeOrg?.metadata)
  )

  useEffect(() => {
    setOrgName(activeOrg?.name ?? "")
  }, [activeOrg?.name])

  useEffect(() => {
    setOrgCountry(getOrgCountry(activeOrg?.metadata))
  }, [activeOrg?.metadata])

  useEffect(() => {
    setProfileName(session?.user?.name ?? "")
  }, [session?.user?.name])

  const currentUserRole = members.find(
    (m: OrgMember) => m.userId === session?.user?.id
  )?.role
  const canInvite = currentUserRole === "owner"
  const canEditOrg = currentUserRole === "owner"
  const canRemoveMembers = currentUserRole === "owner"

  const updateOrgName = useMutation({
    mutationFn: (name: string) => updateOrganizationFn({ data: { name } }),
    onMutate: () => {
      toast.loading("Renaming organization...", { id: "update-org-name" })
    },
    onSuccess: () => {
      toast.success("Organization renamed", { id: "update-org-name" })
      queryClient.invalidateQueries({ queryKey: orgsQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to rename organization",
        { id: "update-org-name" }
      )
    },
  })

  const updateOrgLogo = useMutation({
    mutationFn: (logo: string) => updateOrganizationFn({ data: { logo } }),
    onMutate: () => {
      toast.loading("Uploading logo...", { id: "update-org-logo" })
    },
    onSuccess: () => {
      toast.success("Logo updated", { id: "update-org-logo" })
      queryClient.invalidateQueries({ queryKey: orgsQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update logo",
        { id: "update-org-logo" }
      )
    },
  })

  const updateOrgCountry = useMutation({
    mutationFn: (country: string) =>
      updateOrganizationFn({ data: { metadata: { country } } }),
    onMutate: () => {
      toast.loading("Updating country...", { id: "update-org-country" })
    },
    onSuccess: () => {
      toast.success("Country updated", { id: "update-org-country" })
      queryClient.invalidateQueries({ queryKey: orgsQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update country",
        { id: "update-org-country" }
      )
    },
  })

  const updateProfileName = useMutation({
    mutationFn: (name: string) => updateProfileFn({ data: { name } }),
    onMutate: () => {
      toast.loading("Updating name...", { id: "update-profile-name" })
    },
    onSuccess: () => {
      toast.success("Name updated", { id: "update-profile-name" })
      queryClient.invalidateQueries({ queryKey: sessionQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update name",
        { id: "update-profile-name" }
      )
    },
  })

  const updateProfileImage = useMutation({
    mutationFn: (image: string) => updateProfileFn({ data: { image } }),
    onMutate: () => {
      toast.loading("Uploading image...", { id: "update-profile-image" })
    },
    onSuccess: () => {
      toast.success("Profile image updated", { id: "update-profile-image" })
      queryClient.invalidateQueries({ queryKey: sessionQueryOptions.queryKey })
    },
    onError: (error) => {
      toast.error(
        error instanceof Error ? error.message : "Failed to update image",
        { id: "update-profile-image" }
      )
    },
  })

  const handleProfileImageChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0]
    if (!file) return
    const base64 = await toBase64(file)
    updateProfileImage.mutate(base64)
  }

  const handleOrgLogoChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0]
    if (!file) return
    const base64 = await toBase64(file)
    updateOrgLogo.mutate(base64)
  }

  const inviteMember = useMutation({
    mutationFn: (data: { email: string; role: typeof role }) =>
      inviteMemberFn({ data }),
    onMutate: async ({ email, role }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: orgPendingInvitesQueryOptions.queryKey,
      })

      // Snapshot previous value
      const previousInvites = queryClient.getQueryData<OrgPendingInvite[]>(
        orgPendingInvitesQueryOptions.queryKey
      )

      // Optimistically update
      const optimisticInvite: OrgPendingInvite = {
        id: `temp-${Date.now()}`,
        email,
        role,
        createdAt: new Date(),
      }

      queryClient.setQueryData<OrgPendingInvite[]>(
        orgPendingInvitesQueryOptions.queryKey,
        (old = []) => [...old, optimisticInvite]
      )

      toast.loading("Sending invitation...", { id: "invite-member" })

      return { previousInvites }
    },
    onError: (error, _, context) => {
      // Rollback on error
      if (context?.previousInvites) {
        queryClient.setQueryData(
          orgPendingInvitesQueryOptions.queryKey,
          context.previousInvites
        )
      }
      toast.error(
        error instanceof Error ? error.message : "Failed to send invitation",
        { id: "invite-member" }
      )
    },
    onSuccess: () => {
      toast.success("Invitation sent", { id: "invite-member" })
      queryClient.invalidateQueries({
        queryKey: orgPendingInvitesQueryOptions.queryKey,
      })
      queryClient.invalidateQueries({
        queryKey: orgMembersQueryOptions.queryKey,
      })
      setEmail("")
      setRole("member")
    },
  })

  const handleInvite = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedEmail = email.trim()
    if (!trimmedEmail) return
    inviteMember.mutate({ email: trimmedEmail, role })
  }

  const removeMember = useMutation({
    mutationFn: (userId: string) => removeOrgMemberFn({ data: { userId } }),
    onMutate: async (userId) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: orgMembersQueryOptions.queryKey,
      })

      // Snapshot previous value
      const previousMembers = queryClient.getQueryData<OrgMember[]>(
        orgMembersQueryOptions.queryKey
      )

      // Optimistically update
      queryClient.setQueryData<OrgMember[]>(
        orgMembersQueryOptions.queryKey,
        (old = []) => old.filter((m) => m.userId !== userId)
      )

      toast.loading("Removing member...", { id: "remove-member" })

      return { previousMembers }
    },
    onError: (error, _, context) => {
      // Rollback on error
      if (context?.previousMembers) {
        queryClient.setQueryData(
          orgMembersQueryOptions.queryKey,
          context.previousMembers
        )
      }
      toast.error(
        error instanceof Error ? error.message : "Failed to remove member",
        { id: "remove-member" }
      )
    },
    onSuccess: () => {
      toast.success("Member removed", { id: "remove-member" })
      queryClient.invalidateQueries({
        queryKey: orgMembersQueryOptions.queryKey,
      })
    },
  })

  const cancelInvitation = useMutation({
    mutationFn: (invitationId: string) =>
      cancelOrgInvitationFn({ data: { invitationId } }),
    onMutate: async (invitationId) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: orgPendingInvitesQueryOptions.queryKey,
      })

      // Snapshot previous value
      const previousInvites = queryClient.getQueryData<OrgPendingInvite[]>(
        orgPendingInvitesQueryOptions.queryKey
      )

      // Optimistically update
      queryClient.setQueryData<OrgPendingInvite[]>(
        orgPendingInvitesQueryOptions.queryKey,
        (old = []) => old.filter((inv) => inv.id !== invitationId)
      )

      toast.loading("Canceling invitation...", { id: "cancel-invitation" })

      return { previousInvites }
    },
    onError: (error, _, context) => {
      // Rollback on error
      if (context?.previousInvites) {
        queryClient.setQueryData(
          orgPendingInvitesQueryOptions.queryKey,
          context.previousInvites
        )
      }
      toast.error(
        error instanceof Error ? error.message : "Failed to cancel invitation",
        { id: "cancel-invitation" }
      )
    },
    onSuccess: () => {
      toast.success("Invitation canceled", { id: "cancel-invitation" })
      queryClient.invalidateQueries({
        queryKey: orgPendingInvitesQueryOptions.queryKey,
      })
    },
  })

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-fit max-w-fit p-0 overflow-hidden">
        <SheetTitle className="sr-only">Settings</SheetTitle>
        <SheetDescription className="sr-only">
          Manage your profile, organization, and team members
        </SheetDescription>
        <div className="flex h-full min-h-0">
          {/* Sidenav */}
          <div className="w-[200px] border-r flex flex-col shrink-0 h-full">
            <h2 className="text-lg font-medium p-4">Settings</h2>
            <div className="flex-1 flex flex-col gap-px py-2 px-2">
              <button
                onClick={() => setActiveTab("profile")}
                className={cn(
                  "nav-item nav-item-light text-left",
                  activeTab === "profile" && "active"
                )}
              >
                Profile
              </button>
              {canEditOrg && (
                <button
                  onClick={() => setActiveTab("organization")}
                  className={cn(
                    "nav-item nav-item-light text-left",
                    activeTab === "organization" && "active"
                  )}
                >
                  Organization
                </button>
              )}
              <button
                onClick={() => setActiveTab("members")}
                className={cn(
                  "nav-item nav-item-light text-left",
                  activeTab === "members" && "active"
                )}
              >
                Members
              </button>
            </div>
          </div>
          {/* Content */}
          <div className="w-[550px] overflow-y-auto p-6 min-w-0 h-full shrink-0">
            {activeTab === "profile" && (
              <div className="space-y-4">
                <h2 className="text-lg font-medium">Profile</h2>
                <div className="border rounded-lg p-4 space-y-4">
                  <div className="flex items-center gap-3">
                    <input
                      ref={profileImageInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleProfileImageChange}
                    />
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          onClick={() => profileImageInputRef.current?.click()}
                          disabled={updateProfileImage.isPending}
                          className="relative size-12 rounded-full group cursor-pointer shrink-0"
                        >
                          {session?.user?.image ? (
                            <img
                              src={session.user.image}
                              alt={session.user.name || "Profile image"}
                              className="size-12 rounded-full object-cover"
                            />
                          ) : (
                            <div className="size-12 rounded-full bg-muted flex items-center justify-center text-lg font-medium uppercase text-muted-foreground">
                              {session?.user?.name?.charAt(0) ||
                                session?.user?.email?.charAt(0)}
                            </div>
                          )}
                          <div className="absolute inset-0 rounded-full bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <Upload className="size-4 text-white" />
                          </div>
                        </button>
                      </TooltipTrigger>
                      <TooltipContent>Upload profile image</TooltipContent>
                    </Tooltip>
                    <Input
                      value={profileName}
                      onChange={(e) => setProfileName(e.target.value)}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Enter" &&
                          !updateProfileName.isPending &&
                          profileName.trim() !== session?.user?.name
                        ) {
                          updateProfileName.mutate(profileName)
                        }
                      }}
                      disabled={updateProfileName.isPending}
                      placeholder="Your name"
                      className="flex-1"
                    />
                    <Button
                      onClick={() => updateProfileName.mutate(profileName)}
                      disabled={
                        updateProfileName.isPending ||
                        profileName.trim() === session?.user?.name
                      }
                      size="sm"
                    >
                      Save
                    </Button>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {session?.user?.email}
                  </div>
                </div>
              </div>
            )}
            {activeTab === "organization" && canEditOrg && (
              <div className="space-y-4">
                <h2 className="text-lg font-medium">Organization</h2>
                <div className="border rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <input
                      ref={orgLogoInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleOrgLogoChange}
                    />
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          onClick={() => orgLogoInputRef.current?.click()}
                          disabled={updateOrgLogo.isPending}
                          className="relative size-12 rounded-lg group cursor-pointer shrink-0"
                        >
                          {activeOrg?.logo ? (
                            <img
                              src={activeOrg.logo}
                              alt={`${activeOrg.name} logo`}
                              className="size-12 rounded-lg object-cover"
                            />
                          ) : (
                            <div className="size-12 rounded-lg bg-muted flex items-center justify-center text-lg font-medium uppercase text-muted-foreground">
                              {activeOrg?.name?.charAt(0)}
                            </div>
                          )}
                          <div className="absolute inset-0 rounded-lg bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <Upload className="size-4 text-white" />
                          </div>
                        </button>
                      </TooltipTrigger>
                      <TooltipContent>Upload organization logo</TooltipContent>
                    </Tooltip>
                    <Input
                      value={orgName}
                      onChange={(e) => setOrgName(e.target.value)}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Enter" &&
                          !updateOrgName.isPending &&
                          orgName.trim() !== activeOrg?.name
                        ) {
                          updateOrgName.mutate(orgName)
                        }
                      }}
                      disabled={updateOrgName.isPending}
                      placeholder="Organization name"
                      className="flex-1"
                    />
                    <Button
                      onClick={() => updateOrgName.mutate(orgName)}
                      disabled={
                        updateOrgName.isPending ||
                        orgName.trim() === activeOrg?.name
                      }
                      size="sm"
                    >
                      Save
                    </Button>
                  </div>
                </div>
                <div className="border rounded-lg p-4">
                  <label className="text-sm font-medium mb-2 block">
                    Country
                  </label>
                  <CountrySelect
                    value={orgCountry}
                    onValueChange={(newCountry) => {
                      setOrgCountry(newCountry)
                      updateOrgCountry.mutate(newCountry)
                    }}
                    disabled={updateOrgCountry.isPending}
                  />
                </div>
              </div>
            )}
            {activeTab === "members" && (
              <div className="space-y-4">
                <h2 className="text-lg font-medium">Members</h2>
                {canInvite && (
                  <form
                    onSubmit={handleInvite}
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
                        id="invite-email"
                        type="email"
                        placeholder="colleague@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        disabled={inviteMember.isPending}
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
                        onValueChange={(v) => setRole(v as typeof role)}
                        disabled={inviteMember.isPending}
                      >
                        <SelectTrigger id="invite-role" className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="owner">Admin</SelectItem>
                          <SelectItem value="admin">Project Owner</SelectItem>
                          <SelectItem value="member">Member</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button
                      type="submit"
                      disabled={inviteMember.isPending || !email.trim()}
                      size="sm"
                    >
                      {inviteMember.isPending ? "Inviting..." : "Invite"}
                    </Button>
                  </form>
                )}
                {inviteMember.error && (
                  <div className="text-sm text-red-500 bg-red-50 p-3 rounded-md">
                    {inviteMember.error instanceof Error
                      ? inviteMember.error.message
                      : "Failed to send invitation"}
                  </div>
                )}
                <div className="border rounded-lg divide-y">
                  {members.length === 0 && pendingInvites.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground text-center">
                      No members found
                    </div>
                  ) : (
                    <>
                      {members.map((m: OrgMember) => (
                        <div
                          key={m.id}
                          className="flex items-center gap-2 px-3 py-2"
                        >
                          {m.userImage ? (
                            <img
                              src={m.userImage}
                              alt={m.userName || m.email}
                              className="size-7 rounded-full object-cover"
                            />
                          ) : (
                            <div className="size-7 rounded-full bg-muted flex items-center justify-center text-[10px] font-medium uppercase text-muted-foreground">
                              {m.userName?.charAt(0) || m.email.charAt(0)}
                            </div>
                          )}
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
                          <span className="text-xs px-2 py-1 rounded-full bg-muted text-muted-foreground">
                            {orgRoleLabels[m.role] ?? m.role}
                          </span>
                          {canRemoveMembers &&
                            m.userId !== session?.user?.id && (
                              <AlertDialog>
                                <AlertDialogTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                                    disabled={removeMember.isPending}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                  <AlertDialogHeader>
                                    <AlertDialogTitle>
                                      Remove member?
                                    </AlertDialogTitle>
                                    <AlertDialogDescription>
                                      Are you sure you want to remove{" "}
                                      <strong>{m.userName || m.email}</strong>{" "}
                                      from this organization? This action cannot
                                      be undone.
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel>
                                      Cancel
                                    </AlertDialogCancel>
                                    <AlertDialogAction
                                      onClick={() =>
                                        removeMember.mutate(m.userId)
                                      }
                                      className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                    >
                                      Remove
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                            )}
                        </div>
                      ))}
                      {pendingInvites.map((inv) => (
                        <div
                          key={inv.id}
                          className="flex items-center gap-2 px-3 py-2"
                        >
                          <div className="size-7 rounded-full bg-muted flex items-center justify-center text-[10px] font-medium uppercase text-muted-foreground">
                            {inv.email.charAt(0)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate text-xs">
                              {inv.email}
                            </p>
                            <p className="text-xs text-amber-600">
                              Pending signup
                            </p>
                          </div>
                          <span className="text-xs px-2 py-1 rounded-full bg-muted text-muted-foreground">
                            {orgRoleLabels[inv.role ?? "member"]}
                          </span>
                          {canRemoveMembers && (
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                                  disabled={cancelInvitation.isPending}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>
                                    Cancel invitation?
                                  </AlertDialogTitle>
                                  <AlertDialogDescription>
                                    Are you sure you want to cancel the
                                    invitation for <strong>{inv.email}</strong>?
                                    This action cannot be undone.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction
                                    onClick={() =>
                                      cancelInvitation.mutate(inv.id)
                                    }
                                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                  >
                                    Cancel Invitation
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          )}
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
