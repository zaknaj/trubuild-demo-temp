# ==============================================================================
# TruBuild App - Multi-stage Dockerfile
# TanStack Start + Bun runtime
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependencies
# ------------------------------------------------------------------------------
FROM oven/bun:1.3.8-alpine AS deps

WORKDIR /app

# Copy only dependency files first (better caching)
COPY app/package.json app/bun.lock* ./

# Install all dependencies (including devDependencies for build)
RUN bun install

# ------------------------------------------------------------------------------
# Stage 2: Builder
# ------------------------------------------------------------------------------
FROM oven/bun:1.3.8-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source code
COPY app/ ./

# Build the application
RUN bun run build

# ------------------------------------------------------------------------------
# Stage 3: Production Runtime
# ------------------------------------------------------------------------------
FROM oven/bun:1.3.8-alpine AS runner

# Security: Labels for container identification
LABEL org.opencontainers.image.source="https://github.com/trubuild/trubuild-monorepo"
LABEL org.opencontainers.image.description="TruBuild App - TanStack Start Application"
LABEL org.opencontainers.image.licenses="Proprietary"

# Security: Create non-root user
RUN addgroup --system --gid 1001 trubuild && \
    adduser --system --uid 1001 --ingroup trubuild trubuild

WORKDIR /app

# Security: Set environment to production
ENV NODE_ENV=production
ENV BUN_ENV=production

# Copy built application from builder stage
COPY --from=builder --chown=trubuild:trubuild /app/.output ./.output
COPY --from=builder --chown=trubuild:trubuild /app/package.json ./package.json

# Copy drizzle config for migrations (if needed at runtime)
COPY --from=builder --chown=trubuild:trubuild /app/drizzle.config.ts ./drizzle.config.ts

# Install only production dependencies for migrations
COPY --from=deps /app/node_modules ./node_modules

# Security: Switch to non-root user
USER trubuild

# Expose application port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/auth/session || exit 1

# Start the application
CMD ["bun", "run", ".output/server/index.mjs"]
