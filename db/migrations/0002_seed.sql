-- ─────────────────────────────────────────────────────────────
-- 0002_seed — default model catalogue + global settings
-- Idempotent: safe to re-run.
-- ─────────────────────────────────────────────────────────────

INSERT INTO models (id, display_name, engine, quantization, context_length, params, enabled)
VALUES
  ('qwen3-30b-a3b',
   'Qwen3-30B-A3B (Q4_K_M)',
   'llamacpp',
   'Q4_K_M',
   32768,
   '{"family":"qwen3-moe","total_params":"30.5B","active_params":"3.3B","approx_size_gb":18.6,"min_vram_gb":24}'::jsonb,
   TRUE),
  ('qwen3-30b-a3b-q3',
   'Qwen3-30B-A3B (Q3_K_M — single-GPU fallback)',
   'llamacpp',
   'Q3_K_M',
   16384,
   '{"family":"qwen3-moe","total_params":"30.5B","active_params":"3.3B","approx_size_gb":14.7,"min_vram_gb":16}'::jsonb,
   TRUE),
  ('qwen3-0.6b',
   'Qwen3-0.6B (CPU dev worker)',
   'llamacpp',
   'Q4_K_M',
   8192,
   '{"family":"qwen3","total_params":"0.6B","approx_size_gb":0.4,"min_vram_gb":0,"dev_only":true}'::jsonb,
   TRUE)
ON CONFLICT (id) DO UPDATE
  SET display_name   = EXCLUDED.display_name,
      engine         = EXCLUDED.engine,
      quantization   = EXCLUDED.quantization,
      context_length = EXCLUDED.context_length,
      params         = EXCLUDED.params;

INSERT INTO app_settings (key, value) VALUES
  ('defaults', '{
     "model": "qwen3-30b-a3b",
     "temperature": 0.7,
     "topP": 0.9,
     "topK": 40,
     "maxTokens": 2048,
     "contextLength": 16384,
     "repeatPenalty": 1.1,
     "systemPrompt": "You are a helpful, precise assistant."
   }'::jsonb)
ON CONFLICT (key) DO NOTHING;
