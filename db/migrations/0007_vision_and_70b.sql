-- Vision models, and a 70B that genuinely fits.
--
-- Two corrections to the catalogue seeded in 0006.
--
-- The 70B was dismissed too quickly. Seven quantisations of Llama 3.3 70B fit
-- inside a 2x T4's 29.5 GB, not just the plain Q2_K that was quoted: Unsloth's
-- dynamic UD-IQ2_M is 24.3 GB and holds up considerably better than a flat Q2.
-- It is added at a reduced context, because on a 70B the KV cache competes hard
-- with weights for what little room is left. Speed is the real constraint here
-- rather than fit -- a dense 70B reads every one of those gigabytes per token --
-- so it is enabled and measurable rather than assumed.
--
-- Vision is a smaller change than it sounds. llama.cpp reads images through a
-- separate projector file (mmproj, ~0.85 GB) passed alongside the weights, so a
-- vision model is an ordinary GGUF plus one extra download.

ALTER TABLE models ADD COLUMN IF NOT EXISTS supports_vision BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE models ADD COLUMN IF NOT EXISTS mmproj_pattern  TEXT;

-- Gemma 3 is multimodal from 4B up; the text-only rows seeded in 0006 simply
-- were not told about the projector that was there all along.
UPDATE models
   SET supports_vision = TRUE,
       mmproj_pattern  = 'mmproj',
       min_vram_gb     = min_vram_gb + 0.9   -- the projector lives in VRAM too
 WHERE id IN ('gemma-3-4b', 'gemma-3-12b', 'gemma-3-27b');

INSERT INTO models (id, display_name, engine, quantization, context_length, params, enabled,
                    repo, family, approx_size_gb, min_vram_gb, sort_order,
                    supports_vision, mmproj_pattern)
VALUES
  ('qwen2.5-vl-7b', 'Qwen2.5-VL 7B (vision)', 'llamacpp', 'Q4_K_M', 32768,
   '{"total_params":"7B","vendor":"Alibaba","modality":"text+image"}'::jsonb, TRUE,
   'ggml-org/Qwen2.5-VL-7B-Instruct-GGUF', 'qwen-vl', 4.7, 8.1, 25,
   TRUE, 'mmproj'),

  -- Enabled because it fits and the question deserves a measurement rather than
  -- an assertion. Expect single-digit tokens per second: 70B is dense, so every
  -- token reads the whole 24 GB, where the 30B-A3B that manages 65 tok/s is a
  -- mixture-of-experts reading roughly 2 GB.
  ('llama-3.3-70b', 'Llama 3.3 70B (slow — dense)', 'llamacpp', 'UD-IQ2_M', 8192,
   '{"total_params":"70B","vendor":"Meta","note":"dense; expect single-digit tok/s on 2x T4"}'::jsonb,
   TRUE,
   'unsloth/Llama-3.3-70B-Instruct-GGUF', 'llama3', 24.3, 26.8, 95,
   FALSE, NULL)
ON CONFLICT (id) DO UPDATE SET
  display_name    = EXCLUDED.display_name,
  repo            = EXCLUDED.repo,
  quantization    = EXCLUDED.quantization,
  context_length  = EXCLUDED.context_length,
  approx_size_gb  = EXCLUDED.approx_size_gb,
  min_vram_gb     = EXCLUDED.min_vram_gb,
  sort_order      = EXCLUDED.sort_order,
  supports_vision = EXCLUDED.supports_vision,
  mmproj_pattern  = EXCLUDED.mmproj_pattern;
