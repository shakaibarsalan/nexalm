-- ─────────────────────────────────────────────────────────────
-- 0005_users — real per-user accounts, replacing the shared passphrase
--
-- Additive on purpose. `conversations.session_id` stays so that chats created
-- under the shared-token era remain readable while their owners sign in for
-- the first time; the API adopts them onto a user id at that point.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email          TEXT        NOT NULL,
    -- NULL for accounts that only ever signed in through Google.
    password_hash  TEXT,
    display_name   TEXT        NOT NULL DEFAULT '',
    avatar_url     TEXT,
    role           TEXT        NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),
    google_sub     TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at  TIMESTAMPTZ,
    disabled_at    TIMESTAMPTZ
);

-- Case-insensitive uniqueness without depending on the citext contrib module,
-- which is not guaranteed to be installed in every Postgres image we run on.
-- The API lowercases before writing; this index is the safety net.
CREATE UNIQUE INDEX IF NOT EXISTS users_email_key
    ON users (lower(email));

-- Partial: many rows legitimately have no Google identity, and a plain UNIQUE
-- would still be fine in Postgres but the partial index stays smaller.
CREATE UNIQUE INDEX IF NOT EXISTS users_google_sub_key
    ON users (google_sub) WHERE google_sub IS NOT NULL;

-- ── Sessions ────────────────────────────────────────────────
-- Only the SHA-256 of the bearer token is stored, so a database dump cannot be
-- replayed as a login.
CREATE TABLE IF NOT EXISTS user_sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash    TEXT        NOT NULL UNIQUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL,
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_agent    TEXT,
    ip            TEXT
);

CREATE INDEX IF NOT EXISTS user_sessions_user_idx   ON user_sessions (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS user_sessions_expiry_idx ON user_sessions (expires_at);

-- ── Conversation ownership ──────────────────────────────────
-- Deleting an account takes its private chats with it; there is no other
-- reader for them once the owner is gone.
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE;

-- Rows written from now on carry a user id instead; existing rows keep theirs.
ALTER TABLE conversations
    ALTER COLUMN session_id DROP NOT NULL;

CREATE INDEX IF NOT EXISTS conversations_user_idx
    ON conversations (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversations_user_active_idx
    ON conversations (user_id, updated_at DESC) WHERE archived_at IS NULL;

-- Unadopted legacy rows are found by session_id only, and only while user_id
-- is still NULL, so keep that lookup cheap as the table grows.
CREATE INDEX IF NOT EXISTS conversations_unclaimed_idx
    ON conversations (session_id) WHERE user_id IS NULL;
