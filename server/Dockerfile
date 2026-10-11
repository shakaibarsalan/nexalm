# syntax=docker/dockerfile:1
# Build context is the REPOSITORY ROOT (so db/migrations is reachable).
# Multi-arch: the Oracle host is aarch64, so nothing here may be x86-only.

# ── Build stage ──────────────────────────────────────────────
FROM node:22-alpine AS build
WORKDIR /app

COPY server/package*.json ./
RUN npm ci

COPY server/tsconfig.json ./
COPY server/src ./src
RUN npm run build

# ── Runtime stage ────────────────────────────────────────────
FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production

RUN apk add --no-cache wget

COPY server/package*.json ./
RUN npm ci --omit=dev && npm cache clean --force

COPY --from=build /app/dist ./dist
# Migrations are read at startup from MIGRATIONS_DIR.
COPY db/migrations ./db/migrations
ENV MIGRATIONS_DIR=/app/db/migrations

# The kernel loader source, so the API can start a Kaggle session on request.
COPY worker/kaggle/kaggle_worker_entry.py ./kaggle/kaggle_worker_entry.py
ENV KAGGLE_ENTRY_SOURCE=/app/kaggle/kaggle_worker_entry.py

# Drop privileges — the node image ships an unprivileged `node` user.
USER node

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/api/health || exit 1

CMD ["node", "dist/index.js"]
