-- Multiple models, chosen from the UI.
--
-- Until now a worker advertised exactly one model, fixed at process start, and
-- the model table described only that one. Both sides gain the vocabulary to
-- talk about a set: what a worker *could* load given its VRAM, and what it has
-- resident right now.
--
-- Every column is additive with a default, so a worker running the previous
-- build keeps registering successfully and simply reports one model.

ALTER TABLE workers ADD COLUMN IF NOT EXISTS platform        TEXT;
ALTER TABLE workers ADD COLUMN IF NOT EXISTS loaded_models   JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE workers ADD COLUMN IF NOT EXISTS loadable_models JSONB NOT NULL DEFAULT '[]'::jsonb;

-- The catalogue needs to know where weights come from and what they cost, so
-- the UI can grey out what will not fit rather than offering a choice that
-- fails several minutes later.
ALTER TABLE models ADD COLUMN IF NOT EXISTS repo           TEXT;
ALTER TABLE models ADD COLUMN IF NOT EXISTS family         TEXT;
ALTER TABLE models ADD COLUMN IF NOT EXISTS approx_size_gb NUMERIC(6,2);
ALTER TABLE models ADD COLUMN IF NOT EXISTS min_vram_gb    NUMERIC(6,2);
ALTER TABLE models ADD COLUMN IF NOT EXISTS sort_order     INTEGER NOT NULL DEFAULT 100;

-- Sizes below are the Q4_K_M build of each repo, read from the Hugging Face API
-- on 6 September 2026 rather than typed from memory -- a hardcoded size table
-- in this project has already drifted once after a repo was requantised. The
-- worker still resolves the exact filename at load time; these are for fitting
-- decisions and for what the UI shows before anything is downloaded.
--
-- min_vram_gb is the weights plus ~2.5 GB for the CUDA context, activations and
-- KV cache, which is the same reserve worker/provision/common.py applies.

INSERT INTO models (id, display_name, engine, quantization, context_length, params, enabled,
                    repo, family, approx_size_gb, min_vram_gb, sort_order)
VALUES
  ('gemma-3-4b', 'Gemma 3 4B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"4B","vendor":"Google"}'::jsonb, TRUE,
   'unsloth/gemma-3-4b-it-GGUF', 'gemma3', 2.5, 5.0, 10),

  ('qwen3-8b', 'Qwen3 8B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"8B","vendor":"Alibaba"}'::jsonb, TRUE,
   'unsloth/Qwen3-8B-GGUF', 'qwen3', 5.0, 7.5, 20),

  ('gemma-3-12b', 'Gemma 3 12B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"12B","vendor":"Google"}'::jsonb, TRUE,
   'unsloth/gemma-3-12b-it-GGUF', 'gemma3', 7.3, 9.8, 30),

  ('phi-4', 'Phi-4 14B', 'llamacpp', 'Q4_K_M', 16384,
   '{"total_params":"14B","vendor":"Microsoft"}'::jsonb, TRUE,
   'unsloth/phi-4-GGUF', 'phi', 8.9, 11.4, 40),

  ('qwen3-14b', 'Qwen3 14B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"14B","vendor":"Alibaba"}'::jsonb, TRUE,
   'unsloth/Qwen3-14B-GGUF', 'qwen3', 9.0, 11.5, 50),

  ('gpt-oss-20b', 'GPT-OSS 20B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"20B","active_params":"3.6B","moe":true,"vendor":"OpenAI"}'::jsonb, TRUE,
   'unsloth/gpt-oss-20b-GGUF', 'gpt-oss', 11.6, 14.1, 60),

  ('mistral-small-24b', 'Mistral Small 24B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"24B","vendor":"Mistral"}'::jsonb, TRUE,
   'unsloth/Mistral-Small-24B-Instruct-2501-GGUF', 'mistral', 14.3, 16.8, 70),

  ('gemma-3-27b', 'Gemma 3 27B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"27B","vendor":"Google"}'::jsonb, TRUE,
   'unsloth/gemma-3-27b-it-GGUF', 'gemma3', 16.5, 19.0, 80),

  ('qwen3-32b', 'Qwen3 32B', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"32B","vendor":"Alibaba"}'::jsonb, TRUE,
   'unsloth/Qwen3-32B-GGUF', 'qwen3', 19.8, 22.3, 90)
ON CONFLICT (id) DO UPDATE SET
  display_name   = EXCLUDED.display_name,
  repo           = EXCLUDED.repo,
  family         = EXCLUDED.family,
  approx_size_gb = EXCLUDED.approx_size_gb,
  min_vram_gb    = EXCLUDED.min_vram_gb,
  sort_order     = EXCLUDED.sort_order;

-- Backfill the two rows that predate these columns. The 30B-A3B pair is one
-- repo at two quantisations: the full build for a 2x T4 pair, and a reduced one
-- for a single card.
UPDATE models SET repo = 'unsloth/Qwen3-30B-A3B-GGUF', family = 'qwen3-moe',
                  approx_size_gb = 18.6, min_vram_gb = 21.1, sort_order = 5
 WHERE id = 'qwen3-30b-a3b';

UPDATE models SET repo = 'unsloth/Qwen3-30B-A3B-GGUF', family = 'qwen3-moe',
                  approx_size_gb = 11.8, min_vram_gb = 14.3, sort_order = 6
 WHERE id = 'qwen3-30b-a3b-q3';

UPDATE models SET family = 'qwen3', approx_size_gb = 0.4, min_vram_gb = 0, sort_order = 200
 WHERE id = 'qwen3-0.6b';
