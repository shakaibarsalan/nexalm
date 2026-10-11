import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { config } from '../config.js';
import { query, withTransaction } from '../db/pool.js';
import { ownedByUser, ownerParams } from '../middleware/auth.js';
import { jobManager, NoWorkerAvailableError } from '../services/jobManager.js';
import { workerRegistry } from '../services/workerRegistry.js';
import { configureQueue, maybeAutoStart } from '../services/pendingQueue.js';
import {
  asContextBlock,
  search,
  searchQuery,
  type SearchResult,
} from '../services/webSearch.js';
import type { ChatTurn, GenerationParams, StreamEvent } from '../lib/types.js';

interface ResolvedSettings extends GenerationParams {
  model: string;
  contextLength: number;
  systemPrompt?: string;
  /** Let this chat see what was said in the user's other chats. */
  useGlobalContext: boolean;
  /** Start a Kaggle GPU on demand when a message queues with nothing running. */
  autoStartWorker: boolean;
  /** Search the web and put the sources in front of the model before it answers. */
  webSearch: boolean;
  /**
   * True when this conversation names a model itself.
   *
   * The distinction decides what happens when that model is not loaded. An
   * explicit choice is honoured or refused; an inherited default means "no
   * preference", and answering from whatever is actually running is help
   * rather than substitution.
   */
  modelExplicit: boolean;
}

/** Global defaults, overlaid with any per-conversation overrides. */
/**
 * Why no job can run, naming what the pool *can* serve.
 *
 * Routing no longer substitutes a different model when the requested one is
 * absent, so this message is now the only thing standing between the user and
 * an unexplained refusal. Listing the available models turns "it did not work"
 * into "switch to one of these".
 */
interface ModelDecision {
  /** The model the job should actually run. */
  model: string;
  /** Set when the answer will come from something other than what was asked. */
  substituted?: string;
  /** Set when nothing can serve it and the caller should refuse outright. */
  refuse?: string;
  /** Set when a worker was asked to load it and the turn should wait. */
  loading?: boolean;
}

/**
 * What to run, given what the pool actually has.
 *
 * This is the decision that used to be missing, and its absence is why a
 * message could sit pending forever: the default model needed 21 GB, the only
 * connected worker had 12.5 GB, and the queue waited patiently for a worker
 * that was never going to arrive.
 *
 * Three cases, deliberately different:
 *
 *  - Asked for something resident: run it.
 *  - Asked for nothing in particular: run what is loaded. The conversation
 *    inherited a global default it never chose, so honouring the default over
 *    the machine that is actually running is pedantry, not correctness.
 *  - Asked for something specific that is not loaded: load it if a worker has
 *    the memory, and refuse plainly if none does. Queueing for a model that
 *    cannot fit any connected GPU is a promise the server cannot keep.
 */
async function decideModel(settings: ResolvedSettings): Promise<ModelDecision> {
  const wanted = settings.model;
  if (workerRegistry.hasAvailable(wanted)) return { model: wanted };

  // Only ever substitute within the same kind. The family-prefix match below
  // has no idea a diffusion model is not a language model, and answering a
  // chat turn with an image generator would be a strange way to fail.
  const chatModels = await query<{ id: string }>(
    `SELECT id FROM models WHERE enabled AND kind = 'chat'`,
  );
  const chatIds = new Set(chatModels.rows.map((r) => r.id));
  const resident = workerRegistry.availableModels().filter((m) => chatIds.has(m));

  if (!settings.modelExplicit && resident.length > 0) {
    // Prefer a relative of the default -- the Q3 build of the same model beats
    // an unrelated one -- then fall back to whatever is loaded.
    const family = wanted.split('-').slice(0, 3).join('-');
    const near = resident.find((m) => m.startsWith(family));
    const chosen = near ?? resident[0];
    return { model: chosen, substituted: wanted };
  }

  const { rows } = await query<{ min_vram_gb: string | null }>(
    `SELECT min_vram_gb FROM models WHERE id = $1 AND enabled`,
    [wanted],
  );
  if (rows.length === 0) {
    return { model: wanted, refuse: describeNoWorker(wanted) };
  }

  const needGb = Number(rows[0].min_vram_gb ?? 0);
  const capable = workerRegistry.pickForLoad(needGb);
  if (capable) {
    const loaded = await query<{
      repo: string | null;
      quantization: string | null;
      context_length: number | null;
      supports_vision: boolean;
      engine: string | null;
      family: string | null;
      kind: string | null;
    }>(
      `SELECT repo, quantization, context_length, supports_vision, engine, family, kind
         FROM models WHERE id = $1`,
      [wanted],
    );
    // The user's context setting is what should bind, capped by what the model
    // and the card can actually hold. Sending only the catalogue value is why
    // the setting appeared to do nothing.
    const catalogueCtx = loaded.rows[0]?.context_length ?? settings.contextLength;

    // Same rule as the admin path: never hand an image model to a worker whose
    // build cannot serve one. It would fail inside llama.cpp, after evicting a
    // working model to make room for weights it cannot read.
    if (loaded.rows[0]?.kind === 'image' && !workerRegistry.canGenerateImages(capable.id)) {
      return {
        model: wanted,
        refuse:
          `${capable.name} is running a build from before image generation shipped. ` +
          'Restart that GPU session — it will pick up the new worker code automatically.',
      };
    }

    workerRegistry.requestLoad(capable.id, {
      modelId: wanted,
      repo: loaded.rows[0]?.repo ?? undefined,
      quantization: loaded.rows[0]?.quantization ?? undefined,
      contextLength: Math.min(settings.contextLength, catalogueCtx),
      vision: loaded.rows[0]?.supports_vision,
      engine: loaded.rows[0]?.engine ?? undefined,
      family: loaded.rows[0]?.family ?? undefined,
      kind: loaded.rows[0]?.kind ?? undefined,
    });
    return { model: wanted, loading: true };
  }

  if (workerRegistry.list().length === 0) {
    // Nothing connected at all: the turn waits, and auto-start may bring one up.
    return { model: wanted };
  }

  const largest = workerRegistry.largestVramGb();
  return {
    model: wanted,
    refuse:
      `${wanted} needs about ${needGb} GB of VRAM and the largest connected GPU has ` +
      `${largest.toFixed(1)} GB, so no worker here can run it. ` +
      (resident.length
        ? `Ready right now: ${resident.join(', ')}.`
        : 'Start a larger worker, or pick a smaller model.'),
  };
}

function describeNoWorker(model: string): string {
  const available = workerRegistry.availableModels();
  if (available.length === 0) {
    return 'No GPU worker is connected. Start one on Kaggle or Colab and try again.';
  }
  return (
    `No connected worker is running ${model}. ` +
    `Available right now: ${available.join(', ')}. ` +
    'Pick one of those in settings, or start a worker for this model.'
  );
}

async function resolveSettings(conversationSettings: Record<string, unknown>): Promise<ResolvedSettings> {
  const { rows } = await query<{ value: Record<string, unknown> }>(
    `SELECT value FROM app_settings WHERE key = 'defaults'`,
  );
  const globals = rows[0]?.value ?? {};
  const merged = { ...globals, ...conversationSettings } as Record<string, unknown>;

  return {
    model: (merged.model as string) ?? config.defaults.model,
    temperature: (merged.temperature as number) ?? config.defaults.temperature,
    topP: (merged.topP as number) ?? config.defaults.topP,
    topK: merged.topK as number | undefined,
    maxTokens: (merged.maxTokens as number) ?? config.defaults.maxTokens,
    contextLength: (merged.contextLength as number) ?? config.defaults.contextLength,
    repeatPenalty: merged.repeatPenalty as number | undefined,
    systemPrompt: merged.systemPrompt as string | undefined,
    useGlobalContext: Boolean(merged.useGlobalContext),
    autoStartWorker: Boolean(merged.autoStartWorker),
    webSearch: Boolean(merged.webSearch),
    modelExplicit: typeof conversationSettings.model === 'string',
  };
}

/**
 * Recent exchanges from the user's OTHER conversations, as a briefing block.
 *
 * Conversations are isolated by default, which is why telling the assistant
 * your name in one chat leaves a new chat unaware of it. When the setting is
 * on, this pulls the most recent turns from the same user's other chats so
 * details carry across.
 *
 * Kept to a hard character budget and to completed turns only: this competes
 * with the actual conversation for context window, and it must never grow
 * without bound as history accumulates.
 */
/**
 * How much of the window cross-chat context may occupy.
 *
 * The comment below has always promised a quarter; the code passed the whole
 * context length, and passed it as *characters* while the caller was counting
 * tokens. Turning the toggle on therefore added roughly four thousand tokens on
 * top of the trimmed history, invisibly, and nothing in fitHistory knew.
 */
/** Blank line between the system prompt, cross-chat context and search results. */
const PROMPT_SEPARATOR = '\n\n';

const CROSS_CHAT_SHARE = 0.25;
const CHARS_PER_TOKEN = 4;

async function buildCrossChatContext(
  req: FastifyRequest,
  excludeConversationId: string,
  budgetChars: number,
): Promise<string | undefined> {
  const { rows } = await query<{ title: string; role: string; content: string }>(
    `SELECT c.title, m.role, m.content
       FROM messages m
       JOIN conversations c ON c.id = m.conversation_id
      WHERE ${ownedByUser('c', '$1', '$2')}
        AND c.id <> $3
        AND c.archived_at IS NULL
        AND m.status = 'complete'
        AND m.content <> ''
      ORDER BY m.created_at DESC
      LIMIT 60`,
    [...ownerParams(req), excludeConversationId],
  );
  if (rows.length === 0) return undefined;

  // Rows arrive newest-first so the budget keeps the most recent context;
  // flip back to chronological order for readability.
  const kept: string[] = [];
  let used = 0;
  for (const row of rows) {
    const speaker = row.role === 'user' ? 'User' : 'Assistant';
    const line = `[${row.title}] ${speaker}: ${row.content.slice(0, 600)}`;
    if (used + line.length > budgetChars) break;
    kept.unshift(line);
    used += line.length;
  }
  if (kept.length === 0) return undefined;

  const preamble =
    `Context from the user's other conversations. Use it only when relevant, ` +
    `and do not mention that you were given it:`;
  return [preamble, ...kept].join('\n');
}

/**
 * Trim history to a rough token budget, keeping the most recent turns.
 * ~4 chars/token is crude but adequate; the worker enforces the real limit.
 */
function fitHistory(turns: ChatTurn[], contextLength: number, maxTokens: number): ChatTurn[] {
  const budgetChars = Math.max(1000, (contextLength - maxTokens) * 4);
  const kept: ChatTurn[] = [];
  let used = 0;
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const turn = turns[i];
    const cost = turn.content.length + 8;
    if (used + cost > budgetChars && kept.length > 0) break;
    kept.unshift(turn);
    used += cost;
  }
  return kept;
}

function sseHeaders(reply: FastifyReply): void {
  reply.raw.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  reply.raw.setHeader('Cache-Control', 'no-cache, no-transform');
  reply.raw.setHeader('Connection', 'keep-alive');
  // Belt and braces alongside `proxy_buffering off` in nginx.
  reply.raw.setHeader('X-Accel-Buffering', 'no');
  reply.raw.flushHeaders?.();
}

/**
 * Confirm the signed-in user owns the conversation a job belongs to.
 *
 * Job ids are opaque but not secret — they travel to the browser and appear in
 * URLs. Before accounts existed everyone shared one session, so an unchecked
 * job id leaked nothing; with real users it would let anyone who learns an id
 * read another person's token stream or cancel their generation.
 */
async function userOwnsJob(req: FastifyRequest, jobId: string): Promise<boolean> {
  const { rows } = await query(
    `SELECT 1
       FROM inference_runs r
       JOIN conversations c ON c.id = r.conversation_id
      WHERE r.job_id = $1 AND ${ownedByUser('c', '$2', '$3')}
      LIMIT 1`,
    [jobId, ...ownerParams(req)],
  );
  return rows.length > 0;
}

// The queue rebuilds jobs from history exactly as the retry path does, so it
// borrows these rather than growing a second copy of the same logic.
configureQueue({ resolveSettings, fitHistory });

export function registerChatRoutes(app: FastifyInstance): void {
  /**
   * Persist the user turn plus an assistant placeholder, then dispatch a job.
   * Returns immediately with ids; tokens arrive over the SSE endpoint below.
   */
  app.post('/api/conversations/:id/messages', async (req, reply) => {
    const { id: conversationId } = req.params as { id: string };
    const body = (req.body ?? {}) as { content?: string };

    const content = (body.content ?? '').trim();
    if (!content) return reply.code(400).send({ error: 'empty_message' });

    const { rows: convoRows } = await query<{ settings: Record<string, unknown>; title: string }>(
      `SELECT settings, title FROM conversations
        WHERE id = $1 AND ${ownedByUser('', '$2', '$3')}`,
      [conversationId, ...ownerParams(req)],
    );
    if (convoRows.length === 0) return reply.code(404).send({ error: 'not_found' });

    const settings = await resolveSettings(convoRows[0].settings ?? {});

    // Decide what will actually run before writing anything down. Refusing
    // here costs the user a retype; queueing something unservable costs them a
    // message that never gets answered at all, which is worse.
    const decision = await decideModel(settings);
    if (decision.refuse) {
      return reply.code(503).send({
        error: 'NO_WORKER_AVAILABLE',
        message: decision.refuse,
        availableModels: workerRegistry.availableModels(),
      });
    }
    settings.model = decision.model;

    // Trim against the window the worker actually has, not the one the user
    // typed. llama.cpp is started with a fixed --ctx-size; asking the server to
    // pack a larger prompt only means llama.cpp truncates it later, so the
    // difference was spent rather than used.
    const serving = workerRegistry.pick(settings.model);
    const workerCtx = serving?.maxContext ?? 0;
    const effectiveContext = workerCtx > 0 ? Math.min(settings.contextLength, workerCtx) : settings.contextLength;
    const contextClamped = effectiveContext < settings.contextLength;
    settings.contextLength = effectiveContext;

    const queued = !workerRegistry.hasAvailable(settings.model);

    const { userMessage, assistantMessage, history } = await withTransaction(async (client) => {
      const { rows: seqRows } = await client.query<{ next: number }>(
        `SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages WHERE conversation_id = $1`,
        [conversationId],
      );
      const nextSeq = Number(seqRows[0].next);

      const { rows: userRows } = await client.query(
        `INSERT INTO messages (conversation_id, seq, role, content, status)
         VALUES ($1, $2, 'user', $3, 'complete')
         RETURNING id, seq, role, content, status, created_at`,
        [conversationId, nextSeq, content],
      );

      const { rows: asstRows } = await client.query(
        `INSERT INTO messages (conversation_id, seq, role, content, status)
         VALUES ($1, $2, 'assistant', '', 'pending')
         RETURNING id, seq, role, content, status, created_at`,
        [conversationId, nextSeq + 1],
      );

      const { rows: priorRows } = await client.query<ChatTurn>(
        `SELECT role, content FROM messages
          WHERE conversation_id = $1 AND seq <= $2 AND role <> 'system'
                AND NOT (role = 'assistant' AND content = '')
          ORDER BY seq ASC`,
        [conversationId, nextSeq],
      );

      // First real exchange — name the conversation from the opening prompt.
      if (nextSeq === 1) {
        const title = content.replace(/\s+/g, ' ').slice(0, 60) + (content.length > 60 ? '…' : '');
        await client.query(`UPDATE conversations SET title = $2, updated_at = now() WHERE id = $1`, [
          conversationId,
          title,
        ]);
      } else {
        await client.query(`UPDATE conversations SET updated_at = now() WHERE id = $1`, [conversationId]);
      }

      return { userMessage: userRows[0], assistantMessage: asstRows[0], history: priorRows };
    });

    // A quarter of the window at most, so cross-chat context can never crowd
    // out the conversation the user is actually having.
    const crossContext = settings.useGlobalContext
      ? await buildCrossChatContext(
          req,
          conversationId,
          settings.contextLength * CROSS_CHAT_SHARE * CHARS_PER_TOKEN,
        )
      : undefined;

    // Searched before the model runs rather than by the model deciding to.
    // A local model's knowledge stops at its cut-off and it has no way to know
    // that, so asking about anything recent produces a confident answer instead
    // of an admission. A failed search degrades the reply; it never loses it.
    let sources: SearchResult[] = [];
    let searchBlock = '';
    if (settings.webSearch) {
      try {
        const q = searchQuery(content);
        sources = await search(q, 5);
        searchBlock = asContextBlock(q, sources);
      } catch (err) {
        req.log.warn({ err }, 'web search failed; answering without it');
      }
    }

    const systemPrompt =
      [settings.systemPrompt, crossContext, searchBlock]
        .filter(Boolean)
        .join(PROMPT_SEPARATOR) || undefined;
      undefined;

    // Nothing can serve this yet. The turn is already on disk, so the queue
    // will pick it up the moment a worker registers -- and if Kaggle is
    // configured and the setting allows it, one is started now.
    if (queued) {
      const startNotice = decision.loading
        ? `Loading ${settings.model} onto the GPU. First use downloads it, so this can ` +
          'take a few minutes; the answer appears here when it is ready.'
        : await maybeAutoStart(settings.autoStartWorker ?? false);
      return reply.code(202).send({
        userMessage,
        assistantMessage,
        queued: true,
        jobId: null,
        loadingModel: decision.loading ? settings.model : undefined,
        message: startNotice ?? describeNoWorker(settings.model),
        availableModels: workerRegistry.availableModels(),
      });
    }

    try {
      const job = await jobManager.create({
        conversationId,
        messageId: assistantMessage.id,
        model: settings.model,
        systemPrompt,
        messages: fitHistory(history, settings.contextLength, settings.maxTokens),
        params: {
          temperature: settings.temperature,
          topP: settings.topP,
          topK: settings.topK,
          maxTokens: settings.maxTokens,
          repeatPenalty: settings.repeatPenalty,
        },
        sources: sources.length ? sources : undefined,
      });

      await query(`UPDATE messages SET status = 'streaming' WHERE id = $1`, [assistantMessage.id]);

      return {
        userMessage,
        assistantMessage: { ...assistantMessage, status: 'streaming' },
        jobId: job.id,
        model: settings.model,
        effectiveContextLength: settings.contextLength,
        sources: sources.length ? sources : undefined,
        contextClamped: contextClamped || undefined,
        // Set when the conversation named no model and the pool had something
        // else loaded. Saying so is what separates this from substitution.
        substitutedFor: decision.substituted,
      };
    } catch (err) {
      if (err instanceof NoWorkerAvailableError) {
        // Left pending rather than failed: the worker vanished between the
        // check and the dispatch, and the queue will answer it when one is back.
        await query(`UPDATE messages SET status = 'pending', error = NULL WHERE id = $1`, [
          assistantMessage.id,
        ]);
        const startNotice = await maybeAutoStart(settings.autoStartWorker ?? false);
        return reply.code(202).send({
          userMessage,
          assistantMessage,
          queued: true,
          jobId: null,
          message: startNotice ?? describeNoWorker(settings.model),
        });
      }
      throw err;
    }
  });

  /**
   * SSE token stream for one job.
   *
   * Reconnecting replays every delta already produced, so a page reload in the
   * middle of a long generation resumes rather than losing the output.
   */
  app.get('/api/jobs/:jobId/stream', async (req, reply) => {
    const { jobId } = req.params as { jobId: string };

    if (!(await userOwnsJob(req, jobId))) {
      return reply.code(404).send({ error: 'not_found' });
    }
    sseHeaders(reply);

    const write = (ev: StreamEvent): void => {
      if (reply.raw.writableEnded) return;
      reply.raw.write(`event: ${ev.event}\ndata: ${JSON.stringify(ev.data)}\n\n`);
    };

    const unsubscribe = jobManager.subscribe(jobId, (ev) => {
      write(ev);
      if (ev.event === 'done' || ev.event === 'error') {
        setTimeout(() => reply.raw.end(), 20);
      }
    });

    if (!unsubscribe) {
      // The job finished long enough ago to have been evicted — fall back to
      // the persisted record so the client still gets a terminal event.
      const { rows } = await query(
        `SELECT r.status, r.error, r.prompt_tokens, r.completion_tokens, r.total_tokens,
                r.ttft_ms, r.gen_ms, r.total_ms, r.tokens_per_sec, r.model, r.worker_name,
                m.content
           FROM inference_runs r LEFT JOIN messages m ON m.id = r.message_id
          WHERE r.job_id = $1`,
        [jobId],
      );
      const run = rows[0];
      if (!run) {
        write({ event: 'error', data: { code: 'UNKNOWN_JOB', message: 'Job not found.' } });
      } else if (run.status === 'complete' || run.status === 'cancelled') {
        if (run.content) write({ event: 'delta', data: { text: run.content, index: 0 } });
        write({ event: 'done', data: { ...run, replayed: true } });
      } else {
        write({
          event: 'error',
          data: { code: 'INFERENCE_ERROR', message: run.error ?? 'Generation failed.' },
        });
      }
      reply.raw.end();
      return reply;
    }

    // Comment frames keep intermediaries from closing an idle connection.
    const keepAlive = setInterval(() => {
      if (!reply.raw.writableEnded) reply.raw.write(': ping\n\n');
    }, 15_000);
    keepAlive.unref?.();

    req.raw.on('close', () => {
      clearInterval(keepAlive);
      unsubscribe();
    });

    return reply;
  });

  app.post('/api/jobs/:jobId/cancel', async (req, reply) => {
    const { jobId } = req.params as { jobId: string };

    if (!(await userOwnsJob(req, jobId))) {
      return reply.code(404).send({ error: 'not_found' });
    }
    const cancelled = await jobManager.cancel(jobId);
    if (!cancelled) return reply.code(409).send({ error: 'not_cancellable' });
    return { ok: true };
  });

  /** Re-run the last assistant turn after an error, reusing the prior user prompt. */
  app.post('/api/conversations/:id/retry', async (req, reply) => {
    const { id: conversationId } = req.params as { id: string };

    const { rows: convoRows } = await query<{ settings: Record<string, unknown> }>(
      `SELECT settings FROM conversations
        WHERE id = $1 AND ${ownedByUser('', '$2', '$3')}`,
      [conversationId, ...ownerParams(req)],
    );
    if (convoRows.length === 0) return reply.code(404).send({ error: 'not_found' });

    const { rows: lastRows } = await query<{ id: string; seq: number }>(
      `SELECT id, seq FROM messages
        WHERE conversation_id = $1 AND role = 'assistant'
        ORDER BY seq DESC LIMIT 1`,
      [conversationId],
    );
    if (lastRows.length === 0) return reply.code(400).send({ error: 'nothing_to_retry' });

    const settings = await resolveSettings(convoRows[0].settings ?? {});
    if (!workerRegistry.hasAvailable(settings.model)) {
      return reply
        .code(503)
        .send({
          error: 'NO_WORKER_AVAILABLE',
          message: describeNoWorker(settings.model),
          availableModels: workerRegistry.availableModels(),
        });
    }

    const assistantId = lastRows[0].id;
    const { rows: history } = await query<ChatTurn>(
      `SELECT role, content FROM messages
        WHERE conversation_id = $1 AND seq < $2 AND role <> 'system' AND content <> ''
        ORDER BY seq ASC`,
      [conversationId, lastRows[0].seq],
    );

    await query(
      `UPDATE messages SET content = '', status = 'streaming', error = NULL, updated_at = now()
        WHERE id = $1`,
      [assistantId],
    );

    const retryContext = settings.useGlobalContext
      ? await buildCrossChatContext(
          req,
          conversationId,
          settings.contextLength * CROSS_CHAT_SHARE * CHARS_PER_TOKEN,
        )
      : undefined;
    const retrySystemPrompt =
      [settings.systemPrompt, retryContext].filter(Boolean).join('\n\n') || undefined;

    try {
      const job = await jobManager.create({
        conversationId,
        messageId: assistantId,
        model: settings.model,
        systemPrompt: retrySystemPrompt,
        messages: fitHistory(history, settings.contextLength, settings.maxTokens),
        params: {
          temperature: settings.temperature,
          topP: settings.topP,
          topK: settings.topK,
          maxTokens: settings.maxTokens,
          repeatPenalty: settings.repeatPenalty,
        },
      });
      return { assistantMessageId: assistantId, jobId: job.id };
    } catch (err) {
      if (err instanceof NoWorkerAvailableError) {
        await query(`UPDATE messages SET status = 'error', error = $2 WHERE id = $1`, [
          assistantId,
          err.message,
        ]);
        return reply.code(503).send({ error: err.code, message: err.message });
      }
      throw err;
    }
  });
}
