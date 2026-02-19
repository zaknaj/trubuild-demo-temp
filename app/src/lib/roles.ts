/**
 * Centralized role definitions for the application.
 * These types are used across the database schema, permissions, and UI.
 */

// Organization-level roles
export const ORG_ROLES = ["owner", "admin", "member"] as const
export type OrgRole = (typeof ORG_ROLES)[number]

// Project-level roles
export const PROJECT_ROLES = [
  "project_lead",
  "commercial_lead",
  "technical_lead",
] as const
export type ProjectRole = (typeof PROJECT_ROLES)[number]

// Package-level roles
export const PACKAGE_ROLES = [
  "package_lead",
  "commercial_team",
  "technical_team",
] as const
export type PackageRole = (typeof PACKAGE_ROLES)[number]

// Access levels derived from roles
export const ACCESS_LEVELS = [
  "full",
  "commercial",
  "technical",
  "none",
] as const
export type AccessLevel = (typeof ACCESS_LEVELS)[number]

/**
 * Convert a role to an access level.
 * Lead roles get full access, others get their domain-specific access.
 */
export function roleToAccess(
  role: ProjectRole | PackageRole | null
): AccessLevel {
  if (!role) return "none"
  if (role === "project_lead" || role === "package_lead") return "full"
  if (role === "commercial_lead" || role === "commercial_team")
    return "commercial"
  if (role === "technical_lead" || role === "technical_team") return "technical"
  return "none"
}

/**
 * Check if access level allows viewing technical data.
 */
export function canViewTechnical(access: AccessLevel): boolean {
  return access === "full" || access === "technical"
}

/**
 * Check if access level allows viewing commercial data.
 */
export function canViewCommercial(access: AccessLevel): boolean {
  return access === "full" || access === "commercial"
}
