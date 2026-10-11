import { query } from '../db/pool.js';
import { config } from '../config.js';
import { jobManager, NoWorkerAvailableError } from './jobManager.js';
import { workerRegistry } from './workerRegistry.js';
import { getQuota, isConfigured, startSession } from './kaggleControl.js';
import type { ChatTurn } from '../lib/types.js';

/**
 * Messages waiting for a GPU.
 *
 * Sending a message with no worker connected used to return 503 in
 * milliseconds and *throw the message away* — the user then had to start a
 * worker, wait several minutes, and retype. Now the message is written down and
 * answered when a worker arrives.
 *
 * Deliberately no queue table and no stashed request payload: a pending
 * assistant message with no job id already *is* the queue, and rebuilding the
 * job from conversation history is what the retry path does anyway. That means
 * the queue survives an API restart for free, which a queue held in memory
 * would not.
 */

/** Below this much remaining quota a start would be spent without finishing. */
const MIN_START_SECONDS = 6 * 60;

/**
 * How long after an auto-start before another may fire.
 *
 * A burst of three messages must start one session, not three. The window is
 * comfortably longer than a cold start so a slow-booting session is never
 * mistaken for a failed one.
 */
const AUTO_START_COOLDOWN_MS = 10 * 60 * 1000;

let lastAutoStart = 0;
let draining = false;

interface PendingRow {
  message_id: string;
  conversation_id: string;
  seq: number;
  settings: Record<string, unknown> | null;
}

export interface QueueDeps {
  /** Injected because the settings merge lives in the chat route. */
  resolveSettings: (overrides: Record<string, unknown>) => Promise<{
    model: string;
    temperature: number;
    topP: number;
    topK?: number;
    maxTokens: number;
    contextLength: number;
    repeatPenalty?: number;
    systemPrompt?: string;
    useGlobalContext?: boolean;
    modelExplicit?: boolean;
  }>;
  fitHistory: (turns: ChatTurn[], contextLength: number, maxTokens: number) => ChatTurn[];
}

let deps: QueueDeps | null = null;

export function configureQueue(d: QueueDeps): void {
  deps = d;
}

/** Pending assistant turns, oldest first. */
async function pending(): Promise<PendingRow[]> {
  // "Waiting" is an assistant turn marked pending with no inference run against
  // it. messages has no job column -- a run row is what records the dispatch --
  // so the absence of one is exactly the state this queue exists to fix.
  const { rows } = await query<PendingRow>(
    `SELECT m.id AS message_id, m.conversation_id, m.seq, c.settings
       FROM messages m
       JOIN conversations c ON c.id = m.conversation_id
       LEFT JOIN inference_runs r ON r.message_id = m.id
      WHERE m.role = 'assistant'
        AND m.status = 'pending'
        AND r.id IS NULL
        AND m.created_at > now() - interval '24 hours'
      ORDER BY m.created_at ASC
      LIMIT 50`,
  );
  return rows;
}

export async function pendingCount(): Promise<number> {
  return (await pending()).length;
}

/**
 * Dispatch whatever the pool can now serve.
 *
 * Serialised: `workerMaxQueue` is small, and firing every waiting message at a
 * single worker the instant it registers would simply overflow it.
 */
export async function drain(): Promise<number> {
  if (!deps || draining) return 0;
  draining = true;
  let dispatched = 0;
  try {
    for (const row of await pending()) {
      const settings = await deps.resolveSettings(row.settings ?? {});

      // Same rule as the live path: a conversation that named no model takes
      // whatever is loaded. Without this a turn queued against an inherited
      // default waits for a worker that may never exist -- which is exactly
      // how four messages ended up pending with nothing to answer them.
      let model = settings.model;
      if (!workerRegistry.hasAvailable(model)) {
        // Same rule as the live path: never substitute across kinds. A queued
        // chat turn answered by an image generator would be a strange failure.
        const chat = await query<{ id: string }>(
          `SELECT id FROM models WHERE enabled AND kind = 'chat'`,
        );
        const chatIds = new Set(chat.rows.map((r) => r.id));
        const resident = workerRegistry.availableModels().filter((m) => chatIds.has(m));
        if (settings.modelExplicit || resident.length === 0) continue;
        const family = model.split('-').slice(0, 3).join('-');
        model = resident.find((m) => m.startsWith(family)) ?? resident[0];
      }

      const { rows: history } = await query<ChatTurn>(
        `SELECT role, content FROM messages
          WHERE conversation_id = $1 AND seq < $2 AND role <> 'system' AND content <> ''
          ORDER BY seq ASC`,
        [row.conversation_id, row.seq],
      );

      try {
        const job = await jobManager.create({
          conversationId: row.conversation_id,
          messageId: row.message_id,
          model,
          systemPrompt: settings.systemPrompt || undefined,
          messages: deps.fitHistory(history, settings.contextLength, settings.maxTokens),
          params: {
            temperature: settings.temperature,
            topP: settings.topP,
            topK: settings.topK,
            maxTokens: settings.maxTokens,
            repeatPenalty: settings.repeatPenalty,
          },
        });
        // jobManager.create already wrote the inference_runs row, which is what
        // takes this message out of the pending set on the next sweep.
        await query(
          `UPDATE messages SET status = 'streaming', updated_at = now() WHERE id = $1`,
          [row.message_id],
        );
        void job;
        dispatched += 1;
      } catch (err) {
        if (err instanceof NoWorkerAvailableError) break; // pool filled up; try later
        await query(
          `UPDATE messages SET status = 'error', error = $2, updated_at = now() WHERE id = $1`,
          [row.message_id, (err as Error).message],
        );
      }
    }
  } finally {
    draining = false;
  }
  return dispatched;
}

/**
 * Start a GPU session because something is waiting for one.
 *
 * Only Kaggle can be started from here — Colab has no such API — so a queued
 * message on a Colab-only deployment simply waits for the notebook to be run.
 * Every guard below exists to keep a finite weekly quota from being spent by
 * accident.
 */
export async function maybeAutoStart(enabled: boolean): Promise<string | null> {
  if (!enabled) return null;
  if (!isConfigured()) return null;
  if (workerRegistry.list().length > 0) return null;
  if (Date.now() - lastAutoStart < AUTO_START_COOLDOWN_MS) return null;

  const quota = await getQuota();
  const remaining =
    (quota.gpuSecondsLimit ?? 0) > 0
      ? (quota.gpuSecondsLimit ?? 0) - (quota.gpuSecondsUsed ?? 0)
      : null;
  if (remaining !== null && remaining < MIN_START_SECONDS) {
    return `Not starting a GPU automatically: only ${Math.max(
      0,
      Math.round(remaining / 60),
    )} minutes of quota remain.`;
  }

  // Stamped before the call, not after: a start that takes a while to answer
  // must not leave the window open for a second one.
  lastAutoStart = Date.now();
  const res = await startSession();
  return res.ok
    ? 'Starting a Kaggle GPU session; your message will be answered when it is ready.'
    : `Could not start a GPU automatically: ${res.message}`;
}

/**
 * Drain whenever the pool changes.
 *
 * `changed` fires on register and on every job finishing, which covers both
 * "a worker arrived" and "a worker freed up" without a poll.
 */
export function watchPool(): void {
  workerRegistry.on('changed', () => {
    void drain();
  });
  // One sweep at boot picks up anything queued before a restart.
  setTimeout(() => void drain(), 2_000).unref?.();
  // A slow backstop in case a 'changed' event is ever missed.
  setInterval(() => void drain(), Math.max(30_000, config.sweepIntervalMs * 6)).unref?.();
}
