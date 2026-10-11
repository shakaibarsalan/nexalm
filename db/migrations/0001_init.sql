-- ─────────────────────────────────────────────────────────────
-- 0001_init — core schema for the Local LLM chat platform
-- ─────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Models ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS models (
    id              TEXT PRIMARY KEY,
    display_name    TEXT        NOT NULL,
    engine          TEXT        NOT NULL,             -- llamacpp | vllm | transformers
    quantization    TEXT,                             -- Q4_K_M | AWQ | ...
    context_length  INTEGER     NOT NULL DEFAULT 8192,
    params          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    enabled         BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Workers ─────────────────────────────────────────────────
-- status: WORKER_STARTING | WORKER_READY | WORKER_BUSY | WORKER_ERROR | WORKER_OFFLINE
CREATE TABLE IF NOT EXISTS workers (
    id                 TEXT PRIMARY KEY,
    name               TEXT        NOT NULL,
    status             TEXT        NOT NULL DEFAULT 'WORKER_STARTING',
    engine             TEXT,
    model              TEXT,
    gpus               JSONB       NOT NULL DEFAULT '[]'::jsonb,
    gpu_count          INTEGER     NOT NULL DEFAULT 0,
    vram_total_mb      INTEGER,
    vram_free_mb       INTEGER,
    max_context        INTEGER,
    capabilities       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    active_jobs        INTEGER     NOT NULL DEFAULT 0,
    version            TEXT,
    last_error         TEXT,
    registered_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_heartbeat_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_expires_at   TIMESTAMPTZ,
    deregistered_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS workers_status_idx    ON workers (status);
CREATE INDEX IF NOT EXISTS workers_heartbeat_idx ON workers (last_heartbeat_at DESC);

-- ── Conversations ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   TEXT        NOT NULL,
    title        TEXT        NOT NULL DEFAULT 'New chat',
    settings     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS conversations_session_idx
    ON conversations (session_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversations_active_idx
    ON conversations (session_id, updated_at DESC) WHERE archived_at IS NULL;

-- ── Messages ────────────────────────────────────────────────
-- role:   user | assistant | system
-- status: complete | streaming | pending | error | cancelled
CREATE TABLE IF NOT EXISTS messages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq              INTEGER     NOT NULL,
    role             TEXT        NOT NULL CHECK (role IN ('user','assistant','system')),
    content          TEXT        NOT NULL DEFAULT '',
    status           TEXT        NOT NULL DEFAULT 'complete',
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx
    ON messages (conversation_id, seq);

-- ── Inference runs (persisted metrics) ──────────────────────
-- One row per assistant generation. Metrics live here, not in the browser.
CREATE TABLE IF NOT EXISTS inference_runs (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id             TEXT        NOT NULL UNIQUE,
    message_id         UUID        REFERENCES messages(id)      ON DELETE CASCADE,
    conversation_id    UUID        REFERENCES conversations(id) ON DELETE CASCADE,
    worker_id          TEXT,
    worker_name        TEXT,
    gpu_info           TEXT,
    model              TEXT,
    engine             TEXT,
    status             TEXT        NOT NULL DEFAULT 'queued',   -- queued|running|complete|error|cancelled
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    total_tokens       INTEGER,
    queue_ms           INTEGER,
    ttft_ms            INTEGER,
    gen_ms             INTEGER,
    total_ms           INTEGER,
    tokens_per_sec     NUMERIC(10,2),
    stop_reason        TEXT,
    params             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    error              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at         TIMESTAMPTZ,
    finished_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS inference_runs_message_idx      ON inference_runs (message_id);
CREATE INDEX IF NOT EXISTS inference_runs_conversation_idx ON inference_runs (conversation_id);
CREATE INDEX IF NOT EXISTS inference_runs_created_idx      ON inference_runs (created_at DESC);

-- ── Application settings (global defaults) ──────────────────
CREATE TABLE IF NOT EXISTS app_settings (
    key         TEXT PRIMARY KEY,
    value       JSONB       NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
