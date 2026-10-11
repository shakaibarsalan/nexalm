-- Email verification on sign-up.
--
-- An account is not created until the address is proven, so a pending sign-up
-- has to be parked somewhere. It is parked here rather than as a disabled user
-- row, which keeps unverified addresses out of the users table entirely --
-- nothing can accidentally treat one as an account, and an abandoned sign-up
-- expires instead of lingering.
--
-- The password is stored already hashed. There is no point at which this table
-- holds a password in the clear, even briefly.

CREATE TABLE IF NOT EXISTS email_verifications (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email          TEXT        NOT NULL,
    -- SHA-256 of the code, never the code. A leaked backup should not hand
    -- someone a working sign-up token.
    code_hash      TEXT        NOT NULL,
    password_hash  TEXT        NOT NULL,
    display_name   TEXT,
    attempts       INTEGER     NOT NULL DEFAULT 0,
    expires_at     TIMESTAMPTZ NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One pending sign-up per address: asking again replaces the old code rather
-- than leaving several valid at once.
CREATE UNIQUE INDEX IF NOT EXISTS email_verifications_email_key
    ON email_verifications (lower(email));

-- Every send is recorded so the hourly cap can be enforced against a table
-- rather than memory, which would reset on deploy and let the cap be bypassed
-- by anyone willing to wait for a restart.
CREATE TABLE IF NOT EXISTS email_sends (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email     TEXT        NOT NULL,
    kind      TEXT        NOT NULL,
    sent_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS email_sends_recent_idx
    ON email_sends (lower(email), sent_at DESC);

ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ;

-- Accounts that predate verification are treated as verified: they were created
-- when no proof was asked for, and locking them out retroactively would be a
-- worse answer than grandfathering them.
UPDATE users SET email_verified_at = created_at WHERE email_verified_at IS NULL;
