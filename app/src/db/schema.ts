export * from "@/auth/auth-schema"
import { relations } from "drizzle-orm"
import {
  pgTable,
  text,
  timestamp,
  index,
  uuid,
  primaryKey,
  integer,
  jsonb,
  unique,
} from "drizzle-orm/pg-core"
import { user, organization } from "@/auth/auth-schema"

/**
 * ID Type Strategy:
 *
 * - Auth tables (user, organization, member, invitation, session, account, verification)
 *   use `text` IDs because better-auth requires text-based identifiers.
 *
 * - App tables (proj, pkg, asset, projectMember, packageMember, projectInvitation, packageInvitation)
 *   use `uuid` IDs for better type safety and database performance.
 *
 * Foreign keys reference the appropriate ID type based on the referenced table.
 */

// Common timestamp pattern
const timestamps = {
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at")
    .defaultNow()
    .$onUpdate(() => /* @__PURE__ */ new Date())
    .notNull(),
}

export const proj = pgTable(
  "proj",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    name: text("name").notNull(),
    country: text("country"),
    userId: text("user_id")
      .notNull()
      .references(() => user.id, { onDelete: "cascade" }),
    organizationId: text("organization_id")
      .notNull()
      .references(() => organization.id, { onDelete: "cascade" }),
    archivedAt: timestamp("archived_at"),
    ...timestamps,
  },
  (table) => [
    index("proj_userId_idx").on(table.userId),
    index("proj_organizationId_idx").on(table.organizationId),
    // Index for archived project filtering (isNull/isNotNull queries)
    index("proj_archivedAt_idx").on(table.archivedAt),
    // Composite index for common query pattern: org + non-archived
    index("proj_orgId_archivedAt_idx").on(
      table.organizationId,
      table.archivedAt
    ),
  ]
)

export const pkg = pgTable(
  "pkg",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    name: text("name").notNull(),
    currency: text("currency"),
    stage: text("stage"),
    ragStatus: text("rag_status"),
    technicalWeight: integer("technical_weight").default(50).notNull(),
    commercialWeight: integer("commercial_weight").default(50).notNull(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => proj.id, { onDelete: "cascade" }),
    awardedContractorId: uuid("awarded_contractor_id"),
    awardComments: text("award_comments"),
    archivedAt: timestamp("archived_at"),
    ...timestamps,
  },
  (table) => [
    index("pkg_projectId_idx").on(table.projectId),
    // Index for archived package filtering
    index("pkg_archivedAt_idx").on(table.archivedAt),
    // Composite index for common query pattern: project + non-archived
    index("pkg_projectId_archivedAt_idx").on(table.projectId, table.archivedAt),
    // Index for awarded contractor lookups
    index("pkg_awardedContractorId_idx").on(table.awardedContractorId),
  ]
)

export const asset = pgTable(
  "asset",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    name: text("name").notNull(),
    packageId: uuid("package_id")
      .notNull()
      .references(() => pkg.id, { onDelete: "cascade" }),
    ...timestamps,
  },
  (table) => [index("asset_packageId_idx").on(table.packageId)]
)

export const projectMember = pgTable(
  "project_member",
  {
    projectId: uuid("project_id")
      .notNull()
      .references(() => proj.id, { onDelete: "cascade" }),
    userId: text("user_id").references(() => user.id, { onDelete: "cascade" }), // nullable for pending members
    email: text("email").notNull(),
    role: text("role")
      .notNull()
      .$type<"project_lead" | "commercial_lead" | "technical_lead">(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.projectId, table.email] }),
    index("project_member_userId_idx").on(table.userId),
    // Index for email lookups (linking pending memberships)
    index("project_member_email_idx").on(table.email),
  ]
)

export const packageMember = pgTable(
  "package_member",
  {
    packageId: uuid("package_id")
      .notNull()
      .references(() => pkg.id, { onDelete: "cascade" }),
    userId: text("user_id").references(() => user.id, { onDelete: "cascade" }), // nullable for pending members
    email: text("email").notNull(),
    role: text("role")
      .notNull()
      .$type<"package_lead" | "commercial_team" | "technical_team">(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.packageId, table.email] }),
    index("package_member_userId_idx").on(table.userId),
    // Index for email lookups (linking pending memberships)
    index("package_member_email_idx").on(table.email),
  ]
)

export const projRelations = relations(proj, ({ one, many }) => ({
  user: one(user, {
    fields: [proj.userId],
    references: [user.id],
  }),
  organization: one(organization, {
    fields: [proj.organizationId],
    references: [organization.id],
  }),
  pkgs: many(pkg),
  members: many(projectMember),
}))

export const pkgRelations = relations(pkg, ({ one, many }) => ({
  proj: one(proj, {
    fields: [pkg.projectId],
    references: [proj.id],
  }),
  assets: many(asset),
  members: many(packageMember),
  contractors: many(packageContractor),
  technicalEvaluations: many(technicalEvaluation),
  awardedContractor: one(packageContractor, {
    fields: [pkg.awardedContractorId],
    references: [packageContractor.id],
  }),
}))

export const assetRelations = relations(asset, ({ one, many }) => ({
  pkg: one(pkg, {
    fields: [asset.packageId],
    references: [pkg.id],
  }),
  commercialEvaluations: many(commercialEvaluation),
}))

export const projectMemberRelations = relations(projectMember, ({ one }) => ({
  project: one(proj, {
    fields: [projectMember.projectId],
    references: [proj.id],
  }),
  user: one(user, {
    fields: [projectMember.userId],
    references: [user.id],
  }),
}))

export const packageMemberRelations = relations(packageMember, ({ one }) => ({
  package: one(pkg, {
    fields: [packageMember.packageId],
    references: [pkg.id],
  }),
  user: one(user, {
    fields: [packageMember.userId],
    references: [user.id],
  }),
}))

export const contractor = pgTable("contractor", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull(),
  ...timestamps,
})

export const packageContractor = pgTable(
  "package_contractor",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    packageId: uuid("package_id")
      .notNull()
      .references(() => pkg.id, { onDelete: "cascade" }),
    name: text("name").notNull(),
    ...timestamps,
  },
  (table) => [index("package_contractor_packageId_idx").on(table.packageId)]
)

export const packageContractorRelations = relations(
  packageContractor,
  ({ one }) => ({
    package: one(pkg, {
      fields: [packageContractor.packageId],
      references: [pkg.id],
    }),
  })
)

// Technical Evaluation - belongs to a package
export const technicalEvaluation = pgTable(
  "technical_evaluation",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    packageId: uuid("package_id")
      .notNull()
      .references(() => pkg.id, { onDelete: "cascade" }),
    roundNumber: integer("round_number").notNull(),
    roundName: text("round_name").notNull(),
    data: jsonb("data").default({}).notNull(),
    ...timestamps,
  },
  (table) => [index("technical_evaluation_packageId_idx").on(table.packageId)]
)

export const technicalEvaluationRelations = relations(
  technicalEvaluation,
  ({ one }) => ({
    package: one(pkg, {
      fields: [technicalEvaluation.packageId],
      references: [pkg.id],
    }),
  })
)

// Commercial Evaluation - belongs to an asset
export const commercialEvaluation = pgTable(
  "commercial_evaluation",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    assetId: uuid("asset_id")
      .notNull()
      .references(() => asset.id, { onDelete: "cascade" }),
    roundNumber: integer("round_number").notNull(),
    roundName: text("round_name").notNull(),
    data: jsonb("data").default({}).notNull(),
    ...timestamps,
  },
  (table) => [index("commercial_evaluation_assetId_idx").on(table.assetId)]
)

export const commercialEvaluationRelations = relations(
  commercialEvaluation,
  ({ one }) => ({
    asset: one(asset, {
      fields: [commercialEvaluation.assetId],
      references: [asset.id],
    }),
  })
)

// Document - uploaded files stored in S3
export const document = pgTable(
  "document",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    name: text("name").notNull(), // Original filename
    key: text("key").notNull().unique(), // S3 object key
    contentType: text("content_type"),
    size: integer("size"),

    // Polymorphic: belongs to package OR asset (one should be set)
    packageId: uuid("package_id").references(() => pkg.id, {
      onDelete: "cascade",
    }),
    assetId: uuid("asset_id").references(() => asset.id, {
      onDelete: "cascade",
    }),

    // Document categorization
    category: text("category").$type<
      "rfp" | "boq" | "pte" | "vendor_proposal" | "criteria" | "other"
    >(),
    // For vendor proposals, link to the contractor
    contractorId: uuid("contractor_id").references(() => packageContractor.id, {
      onDelete: "set null",
    }),

    uploadedBy: text("uploaded_by").references(() => user.id, {
      onDelete: "set null",
    }),
    ...timestamps,
  },
  (table) => [
    index("document_packageId_idx").on(table.packageId),
    index("document_assetId_idx").on(table.assetId),
    index("document_key_idx").on(table.key),
    index("document_contractorId_idx").on(table.contractorId),
  ]
)

export const documentRelations = relations(document, ({ one }) => ({
  package: one(pkg, {
    fields: [document.packageId],
    references: [pkg.id],
  }),
  asset: one(asset, {
    fields: [document.assetId],
    references: [asset.id],
  }),
  contractor: one(packageContractor, {
    fields: [document.contractorId],
    references: [packageContractor.id],
  }),
  uploader: one(user, {
    fields: [document.uploadedBy],
    references: [user.id],
  }),
}))

// Jobs table for background processing
export const jobs = pgTable(
  "jobs",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    type: text("type").notNull(),
    payload: jsonb("payload").default({}).notNull(),
    companyId: text("company_id"),
    userId: text("user_id"),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("jobs_type_idx").on(table.type),
    index("jobs_companyId_idx").on(table.companyId),
    index("jobs_userId_idx").on(table.userId),
  ]
)

// Job runs table (engine creates runs when processing jobs)
export const jobRuns = pgTable(
  "job_runs",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    jobId: uuid("job_id")
      .notNull()
      .references(() => jobs.id),
    attemptNo: integer("attempt_no").default(1).notNull(),
    status: text("status").default("pending").notNull(),
    progress: jsonb("progress").default({}),
    error: text("error"),
    createdAt: timestamp("created_at").defaultNow(),
    startedAt: timestamp("started_at"),
    completedAt: timestamp("completed_at"),
  },
  (table) => [
    unique("job_runs_jobId_attemptNo_unique").on(table.jobId, table.attemptNo),
    index("job_runs_jobId_idx").on(table.jobId),
    index("job_runs_status_idx").on(table.status),
  ]
)

// Job artifacts table (engine stores run results)
export const jobArtifacts = pgTable(
  "job_artifacts",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    runId: uuid("run_id")
      .notNull()
      .references(() => jobRuns.id),
    artifactType: text("artifact_type").notNull(),
    data: jsonb("data"),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => [
    index("job_artifacts_runId_idx").on(table.runId),
    index("job_artifacts_artifactType_idx").on(table.artifactType),
  ]
)
