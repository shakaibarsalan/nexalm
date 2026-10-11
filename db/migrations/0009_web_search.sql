-- Sources behind an answer.
--
-- When a reply was informed by a web search, the citations have to outlive the
-- request that produced them: a page reload would otherwise leave numbered
-- references like [1] in the text pointing at nothing, which is worse than
-- having no citations at all.
--
-- Hung off inference_runs rather than messages, following the existing pattern
-- for everything that describes how a generation happened rather than what it
-- said.

ALTER TABLE inference_runs ADD COLUMN IF NOT EXISTS sources JSONB;
