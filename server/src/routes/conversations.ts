import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { query } from '../db/pool.js';
import { ownedByUser, ownerParams, requireUser } from '../middleware/auth.js';

/**
 * Per-conversation overrides.
 *
 * Deliberately a subset of the global schema in routes/settings.ts: some
 * settings only make sense deployment-wide. It must nonetheless accept every
 * key the shared ChatSettings type carries, or the browser can send something
 * this rejects -- which it did, as a 500.
 */
const settingsSchema = z
  .object({
    model: z.string().optional(),
    temperature: z.number().min(0).max(2).optional(),
    topP: z.number().min(0).max(1).optional(),
    topK: z.number().int().min(0).max(200).optional(),
    maxTokens: z.number().int().min(1).max(32768).optional(),
    contextLength: z.number().int().min(512).max(131072).optional(),
    repeatPenalty: z.number().min(0).max(3).optional(),
    systemPrompt: z.string().max(8000).optional(),
    useGlobalContext: z.boolean().optional(),
    webSearch: z.boolean().optional(),
    // Accepted and ignored per conversation: both are deployment-wide. Rejecting
    // them outright turned an ordinary save into a 500, because this schema used
    // .parse() where the settings route uses safeParse.
    autoStartWorker: z.boolean().optional(),
    idleShutdownMinutes: z.number().int().min(0).max(720).optional(),
  })
  .strict();

/** Parse overrides, answering 400 rather than throwing a 500 at Fastify. */
function parseSettings(raw: unknown): { ok: true; value: z.infer<typeof settingsSchema> } | { ok: false; issues: z.ZodIssue[] } {
  const parsed = settingsSchema.safeParse(raw);
  return parsed.success ? { ok: true, value: parsed.data } : { ok: false, issues: parsed.error.issues };
}

export function registerConversationRoutes(app: FastifyInstance): void {
  /** List the signed-in user's conversations, with a lightweight metrics roll-up. */
  app.get('/api/conversations', async (req) => {
    const includeArchived = (req.query as { archived?: string }).archived === 'true';
    const { rows } = await query(
      `SELECT c.id, c.title, c.created_at, c.updated_at, c.archived_at, c.settings,
              COUNT(m.id) FILTER (WHERE m.role <> 'system')            AS message_count,
              COALESCE(SUM(r.total_tokens), 0)                         AS total_tokens,
              COALESCE(SUM(r.completion_tokens), 0)                    AS output_tokens,
              ROUND(AVG(r.tokens_per_sec)::numeric, 2)                 AS avg_tokens_per_sec
         FROM conversations c
         LEFT JOIN messages       m ON m.conversation_id = c.id
         LEFT JOIN inference_runs r ON r.conversation_id = c.id AND r.status = 'complete'
        WHERE ${ownedByUser('c', '$1', '$2')}
          AND ($3::bool OR c.archived_at IS NULL)
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT 200`,
      [...ownerParams(req), includeArchived],
    );
    return { conversations: rows };
  });

  app.post('/api/conversations', async (req, reply) => {
    const body = (req.body ?? {}) as { title?: string; settings?: unknown };
    const parsed = body.settings ? parseSettings(body.settings) : ({ ok: true, value: {} } as const);
    if (!parsed.ok) {
      return reply
        .code(400)
        .send({ error: 'invalid_settings', message: 'Unknown or invalid setting.', issues: parsed.issues });
    }
    const settings = parsed.value;
    const { rows } = await query(
      `INSERT INTO conversations (user_id, title, settings)
       VALUES ($1, $2, $3::jsonb)
       RETURNING id, title, created_at, updated_at, archived_at, settings`,
      [requireUser(req).id, body.title?.slice(0, 200) || 'New chat', JSON.stringify(settings)],
    );
    return { conversation: rows[0] };
  });

  /** Full thread: messages joined to their persisted inference metrics. */
  app.get('/api/conversations/:id', async (req, reply) => {
    const { id } = req.params as { id: string };
    const { rows: convo } = await query(
      `SELECT id, title, created_at, updated_at, archived_at, settings
         FROM conversations WHERE id = $1 AND ${ownedByUser('', '$2', '$3')}`,
      [id, ...ownerParams(req)],
    );
    if (convo.length === 0) return reply.code(404).send({ error: 'not_found' });

    const { rows: messages } = await query(
      `SELECT m.id, m.seq, m.role, m.content, m.reasoning, m.status, m.error, m.created_at,
              r.job_id, r.model, r.worker_name, r.gpu_info,
              r.prompt_tokens, r.completion_tokens, r.total_tokens,
              r.ttft_ms, r.gen_ms, r.total_ms, r.tokens_per_sec, r.stop_reason, r.sources,
              -- Images are joined here rather than fetched per message: without
              -- them, a picture would render while it streamed and then vanish
              -- on the next page load, which is the sort of bug that looks like
              -- data loss to the person it happens to.
              g.id AS image_id, g.width AS image_width, g.height AS image_height
         FROM messages m
         LEFT JOIN inference_runs r   ON r.message_id = m.id
         LEFT JOIN generated_images g ON g.message_id = m.id
        WHERE m.conversation_id = $1
        ORDER BY m.seq ASC`,
      [id],
    );

    return { conversation: convo[0], messages };
  });

  app.patch('/api/conversations/:id', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = (req.body ?? {}) as { title?: string; settings?: unknown; archived?: boolean };

    const sets: string[] = [];
    const params: unknown[] = [id, ...ownerParams(req)];

    if (typeof body.title === 'string') {
      params.push(body.title.slice(0, 200));
      sets.push(`title = $${params.length}`);
    }
    if (body.settings !== undefined) {
      const parsed = parseSettings(body.settings);
      if (!parsed.ok) {
        return reply
          .code(400)
          .send({ error: 'invalid_settings', message: 'Unknown or invalid setting.', issues: parsed.issues });
      }
      // Merged, not replaced. Assigning the whole object meant picking a model
      // silently discarded every other override on the conversation.
      params.push(JSON.stringify(parsed.value));
      sets.push(`settings = COALESCE(settings, '{}'::jsonb) || $${params.length}::jsonb`);
    }
    if (typeof body.archived === 'boolean') {
      sets.push(`archived_at = ${body.archived ? 'now()' : 'NULL'}`);
    }
    if (sets.length === 0) return reply.code(400).send({ error: 'no_fields' });

    const { rows } = await query(
      `UPDATE conversations SET ${sets.join(', ')}, updated_at = now()
        WHERE id = $1 AND ${ownedByUser('', '$2', '$3')}
        RETURNING id, title, created_at, updated_at, archived_at, settings`,
      params,
    );
    if (rows.length === 0) return reply.code(404).send({ error: 'not_found' });
    return { conversation: rows[0] };
  });

  app.delete('/api/conversations/:id', async (req, reply) => {
    const { id } = req.params as { id: string };
    const { rowCount } = await query(
      `DELETE FROM conversations WHERE id = $1 AND ${ownedByUser('', '$2', '$3')}`,
      [id, ...ownerParams(req)],
    );
    if (rowCount === 0) return reply.code(404).send({ error: 'not_found' });
    return { ok: true };
  });

  /** Aggregate metrics for the right-hand panel. */
  app.get('/api/conversations/:id/metrics', async (req, reply) => {
    const { id } = req.params as { id: string };
    const { rows: owned } = await query(
      `SELECT 1 FROM conversations WHERE id = $1 AND ${ownedByUser('', '$2', '$3')}`,
      [id, ...ownerParams(req)],
    );
    if (owned.length === 0) return reply.code(404).send({ error: 'not_found' });

    const { rows } = await query(
      `SELECT COUNT(*)                                    AS runs,
              COALESCE(SUM(prompt_tokens), 0)             AS input_tokens,
              COALESCE(SUM(completion_tokens), 0)         AS output_tokens,
              COALESCE(SUM(total_tokens), 0)              AS total_tokens,
              COALESCE(SUM(gen_ms), 0)                    AS total_gen_ms,
              ROUND(AVG(tokens_per_sec)::numeric, 2)      AS avg_tokens_per_sec,
              ROUND(AVG(ttft_ms)::numeric, 0)             AS avg_ttft_ms,
              MAX(tokens_per_sec)                         AS peak_tokens_per_sec
         FROM inference_runs
        WHERE conversation_id = $1 AND status = 'complete'`,
      [id],
    );
    return { metrics: rows[0] };
  });
}
