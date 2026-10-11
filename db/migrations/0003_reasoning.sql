-- ─────────────────────────────────────────────────────────────
-- 0003_reasoning — store thinking traces separately from answers
--
-- Qwen3 and similar models emit a reasoning trace before their answer, which
-- llama.cpp reports as `reasoning_content` rather than `content`. It can be
-- hundreds of tokens: the user waits for it either way, so it is kept and
-- shown collapsed rather than discarded.
-- ─────────────────────────────────────────────────────────────

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS reasoning TEXT;

ALTER TABLE inference_runs
    ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER;
