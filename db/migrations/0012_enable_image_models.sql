-- Re-enable image generation, now that something can actually serve it.
--
-- 0010 shipped the catalogue rows before any engine existed, so the only thing
-- selecting them could do was hand diffusion weights to llama.cpp -- which
-- failed, and took a working chat model with it through the eviction it had
-- already performed. 0011 hid them again. This turns them back on against a
-- worker that has an sd-server binary and a dispatch path to reach it.

UPDATE models SET enabled = TRUE WHERE kind = 'image';

/*
 * Correct the disk figures to the files actually fetched.
 *
 * 0010 recorded 12.20 GB for FLUX from the obvious sources: city96 for the
 * GGUF, comfyanonymous for the encoders, black-forest-labs for the VAE. That
 * last repository is `gated: auto`, which an anonymous worker cannot download
 * from -- and it would fail *after* pulling 6.7 GB of diffusion model.
 *
 * Everything now comes from second-state/FLUX.1-schnell-GGUF, which is ungated
 * and also publishes a quantised T5. Taking the encoder at Q4_K rather than
 * fp16 is the difference between 10 GB and 17 GB on disk, for a model whose
 * output is a conditioning vector:
 *
 *   flux1-schnell-Q4_0.gguf   6.69 GB   diffusion model
 *   t5xxl-Q4_K.gguf           2.75 GB   text encoder  (CPU)
 *   clip_l.safetensors        0.25 GB   text encoder  (CPU)
 *   ae.safetensors            0.34 GB   VAE
 *                            10.03 GB
 *
 * min_vram_gb stays well below that because the two encoders are deliberately
 * kept on CPU with --clip-on-cpu, so the card holds the diffusion model and the
 * VAE and nothing else.
 */
UPDATE models
   SET disk_gb     = 10.03,
       min_vram_gb = 9.30,
       params      = params || '{"files":4,"encoders_on_cpu":true}'::jsonb
 WHERE id = 'flux-schnell';
