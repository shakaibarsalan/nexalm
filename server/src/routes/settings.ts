import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { query } from '../db/pool.js';
import { workerRegistry } from '../services/workerRegistry.js';

const defaultsSchema = z
  .object({
    model: z.string().min(1),
    temperature: z.number().min(0).max(2),
    topP: z.number().min(0).max(1),
    topK: z.number().int().min(0).max(200).optional(),
    maxTokens: z.number().int().min(1).max(32768),
    contextLength: z.number().int().min(512).max(131072),
    repeatPenalty: z.number().min(0).max(3).optional(),
    systemPrompt: z.string().max(8000).optional(),
    useGlobalContext: z.boolean().optional(),
    /** Start a Kaggle GPU when a message is queued and nothing is running. */
    autoStartWorker: z.boolean().optional(),
    /** Look things up on the web before answering. */
    webSearch: z.boolean().optional(),
    /** Minutes a GPU worker may sit with no work before it exits. 0 disables. */
    idleShutdownMinutes: z.number().int().min(0).max(720).optional(),
  })
  .strict();

export function registerSettingsRoutes(app: FastifyInstance): void {
  app.get('/api/settings', async () => {
    const { rows } = await query<{ value: unknown }>(
      `SELECT value FROM app_settings WHERE key = 'defaults'`,
    );
    const { rows: models } = await query(
      `SELECT id, display_name, engine, quantization, context_length, params, enabled
         FROM models WHERE enabled ORDER BY display_name`,
    );
    return { defaults: rows[0]?.value ?? {}, models };
  });

  app.put('/api/settings', async (req, reply) => {
    const parsed = defaultsSchema.safeParse(req.body);
    if (!parsed.success) {
      return reply.code(400).send({ error: 'invalid_settings', issues: parsed.error.issues });
    }
    const { rows } = await query<{ value: unknown }>(
      `INSERT INTO app_settings (key, value, updated_at)
       VALUES ('defaults', $1::jsonb, now())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
       RETURNING value`,
      [JSON.stringify(parsed.data)],
    );
    return { defaults: rows[0].value };
  });

  app.get('/api/models', async () => {
    const { rows } = await query<{
      id: string;
      min_vram_gb: string | null;
      approx_size_gb: string | null;
    }>(
      `SELECT id, display_name, engine, quantization, context_length, params, enabled,
              repo, family, approx_size_gb, min_vram_gb, supports_vision,
              kind, disk_gb
         FROM models ORDER BY enabled DESC, sort_order, display_name`,
    );

    // Each model is answered against the pool as it is right now: resident,
    // loadable, or larger than any connected card. Offering a choice that will
    // fail several minutes into a download is worse than not offering it.
    const loaded = new Set(workerRegistry.availableModels());
    const workers = workerRegistry.list();
    const largestVramGb = Math.max(0, ...workers.map((w) => (w.vramTotalMb ?? 0) / 1024));

    const models = rows.map((m) => {
      const needGb = Number(m.min_vram_gb ?? 0);
      const fits = workers.length === 0 ? null : needGb === 0 || needGb <= largestVramGb;
      return {
        ...m,
        approx_size_gb: m.approx_size_gb === null ? null : Number(m.approx_size_gb),
        min_vram_gb: m.min_vram_gb === null ? null : Number(m.min_vram_gb),
        loaded: loaded.has(m.id),
        fits,
        availability: loaded.has(m.id)
          ? 'ready'
          : fits === false
            ? 'too_large'
            : workers.length === 0
              ? 'no_worker'
              : 'loadable',
      };
    });

    return { models, largestVramGb: workers.length ? Number(largestVramGb.toFixed(1)) : null };
  });
}
