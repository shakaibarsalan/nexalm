-- Image generation.
--
-- Every model row until now was a chat model, and nothing said so. Adding an
-- image model without a `kind` would let the routing substitute one for the
-- other: decideModel() picks a relative by family prefix when a conversation
-- named no model, and a flat list of resident ids has no way to tell a
-- diffusion model from a language model.
--
-- `disk_gb` matters as much as VRAM here and is new. FLUX needs about 6.4 GB of
-- VRAM with the text encoder on CPU, which fits a single T4 comfortably -- but
-- it is 12.2 GB across four files on disk, and a free Google Drive is 15 GB
-- with a chat model already in it. So the honest reason image generation is
-- Kaggle-only is storage, not VRAM, and sizing it by VRAM alone would offer it
-- on Colab and then fail halfway through a download.

ALTER TABLE models ADD COLUMN IF NOT EXISTS kind    TEXT NOT NULL DEFAULT 'chat';
ALTER TABLE models ADD COLUMN IF NOT EXISTS disk_gb NUMERIC(6,2);

-- Everything that already exists is a chat model; make that explicit rather
-- than relying on the default for rows written before this column existed.
UPDATE models SET kind = 'chat' WHERE kind IS NULL OR kind = '';

-- Weights on disk for the chat models, so one rule covers both kinds. These are
-- the single GGUF, so disk and size agree.
UPDATE models SET disk_gb = approx_size_gb WHERE disk_gb IS NULL AND approx_size_gb IS NOT NULL;

/*
 * FLUX is four files, not one:
 *   flux1-schnell-Q4_0.gguf        6.77 GB   the diffusion model
 *   t5xxl_fp8_e4m3fn.safetensors   4.89 GB   text encoder
 *   clip_l.safetensors             0.25 GB   text encoder
 *   ae.safetensors                 0.34 GB   VAE
 * The worker fetches all four; the catalogue records the total.
 *
 * schnell is the four-step distilled model, which is what makes it usable at
 * all on a T4 -- dev needs twenty-odd steps for the same picture.
 */
INSERT INTO models (id, display_name, engine, quantization, context_length, params, enabled,
                    repo, family, approx_size_gb, disk_gb, min_vram_gb, sort_order, kind)
VALUES
  ('flux-schnell', 'FLUX.1 schnell (images)', 'stablediffusion', 'Q4_0', 512,
   '{"vendor":"Black Forest Labs","steps":4,"note":"four-step distilled; dev needs ~20 for the same image","files":4}'::jsonb,
   TRUE,
   'city96/FLUX.1-schnell-gguf', 'flux', 6.77, 12.20, 9.30, 300, 'image'),

  -- Small, single-file and old enough to be unimpressive, but it fits anywhere
  -- and answers in seconds, which makes it the right thing to reach for when a
  -- 12 GB download is not worth it.
  ('sd-1-5', 'Stable Diffusion 1.5 (images)', 'stablediffusion', 'Q4_0', 512,
   '{"vendor":"Runway/StabilityAI","steps":20,"note":"dated, but small and quick","files":1}'::jsonb,
   TRUE,
   'second-state/stable-diffusion-v1-5-GGUF', 'sd', 1.57, 1.57, 4.10, 310, 'image')
ON CONFLICT (id) DO UPDATE SET
  display_name   = EXCLUDED.display_name,
  engine         = EXCLUDED.engine,
  repo           = EXCLUDED.repo,
  approx_size_gb = EXCLUDED.approx_size_gb,
  disk_gb        = EXCLUDED.disk_gb,
  min_vram_gb    = EXCLUDED.min_vram_gb,
  sort_order     = EXCLUDED.sort_order,
  kind           = EXCLUDED.kind;

/*
 * Where a generated image lives.
 *
 * Hung off the run rather than the message, following inference_runs: it
 * describes how a generation happened rather than what was said. messages.content
 * is TEXT NOT NULL and there is no attachment mechanism anywhere in the schema,
 * so the alternative was base64 in a text column, which bloats every history
 * fetch for the life of the conversation.
 */
CREATE TABLE IF NOT EXISTS generated_images (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id  UUID        REFERENCES messages(id)      ON DELETE CASCADE,
    job_id      TEXT        NOT NULL,
    path        TEXT        NOT NULL,
    mime        TEXT        NOT NULL DEFAULT 'image/png',
    width       INTEGER,
    height      INTEGER,
    seed        BIGINT,
    steps       INTEGER,
    model       TEXT,
    bytes       INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS generated_images_message_idx ON generated_images (message_id);
CREATE INDEX IF NOT EXISTS generated_images_job_idx     ON generated_images (job_id);
