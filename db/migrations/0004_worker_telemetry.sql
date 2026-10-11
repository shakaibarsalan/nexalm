-- ─────────────────────────────────────────────────────────────
-- 0004_worker_telemetry — time series for the resource charts
--
-- One row per GPU per heartbeat (~10 s). Deliberately narrow and pruned to a
-- rolling window: this is for a live utilisation chart, not long-term history,
-- and an unbounded table on a shared 11 GB host is a liability.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS worker_samples (
    id           BIGSERIAL PRIMARY KEY,
    worker_id    TEXT        NOT NULL,
    sampled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    gpu_index    INTEGER     NOT NULL DEFAULT 0,
    util_pct     INTEGER,
    mem_used_mb  INTEGER,
    mem_total_mb INTEGER,
    temp_c       INTEGER,
    power_w      NUMERIC(6,1),
    active_jobs  INTEGER     NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS worker_samples_recent_idx
    ON worker_samples (worker_id, sampled_at DESC);

-- Lets the UI show how long a worker has been idle before auto-shutdown.
ALTER TABLE workers ADD COLUMN IF NOT EXISTS idle_seconds INTEGER;
