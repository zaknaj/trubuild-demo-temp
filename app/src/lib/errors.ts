export const ERRORS = {
  NOT_FOUND: (resource: string) => `${resource} not found.`,
  NO_ACCESS: (resource: string) => `You don't have access to this ${resource}.`,
  NO_PERMISSION_INVITE: (resource: string) =>
    `You don't have permission to add members to this ${resource}.`,
  NO_PERMISSION_REMOVE: (resource: string) =>
    `You don't have permission to remove members from this ${resource}.`,
  NO_PERMISSION_ARCHIVE: (resource: string) =>
    `You don't have permission to archive this ${resource}.`,
  NO_PERMISSION_RESTORE: (resource: string) =>
    `You don't have permission to restore this ${resource}.`,
  NO_PERMISSION_RENAME: (resource: string) =>
    `You don't have permission to rename this ${resource}.`,
  NO_PERMISSION_ADMIN: "You don't have permission to perform this action.",
  NO_PERMISSION_CREATE_PROJECT: "You don't have permission to create projects.",
  NO_PERMISSION_CREATE_PACKAGE: "You don't have permission to create packages.",
  NO_TECHNICAL_ACCESS: "You don't have access to technical evaluations.",
  NO_COMMERCIAL_ACCESS: "You don't have access to commercial evaluations.",
  MUST_BE_LOGGED_IN: "You must be logged in.",
  MUST_BE_LOGGED_IN_WITH_ORG: "You must be logged in with an active organization.",
} as const

