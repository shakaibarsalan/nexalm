import { config } from '../config.js';
import { query } from '../db/pool.js';

/**
 * Starts and stops the Kaggle GPU session from the app.
 *
 * Uses `kernels/create-session`, NOT `kernels/push`. That distinction is the
 * whole design:
 *
 *   push          replaces the notebook source, and as a side effect resets the
 *                 accelerator to the default single GPU and drops the attached
 *                 Secrets. Starting this way silently halves VRAM.
 *   create-session runs the notebook as it already is. `machineShape` is
 *                 optional, so omitting it keeps whatever the notebook is set
 *                 to — including the 2x T4 (32 GB) pair that the API cannot
 *                 request explicitly. Secrets and source are untouched.
 *
 * Stopping goes through our own WebSocket instead (see routes/workers.ts):
 * ending the worker process ends the session, and that works identically for
 * Colab or a dedicated host. `cancel-session` is kept here as a fallback for a
 * worker that never connected.
 */

/**
 * Two different Kaggle surfaces, both needed.
 *
 * The legacy REST routes under www.kaggle.com/api/v1 serve status and quota.
 * Session control is NOT there — /kernels/create-session/{slug} 404s. It lives
 * on the newer service host as an RPC-style call, which is what the official
 * SDK actually uses. Verified against the live API.
 */
const KAGGLE_REST = 'https://www.kaggle.com/api/v1';
const KAGGLE_RPC = 'https://api.kaggle.com/v1';

/** Last session we started, so Stop can cancel it even if no worker connected. */
let lastSessionId: number | null = null;

/**
 * The session id also lives in app_settings, because holding it only in memory
 * means an API restart orphans a session the server itself started: Kaggle
 * refuses a new start with 409 ALREADY_EXISTS while the old one keeps billing
 * a finite weekly quota, and nothing left knows its id to cancel it. That
 * happened, and cost quota, before this was persisted.
 *
 * app_settings is already a generic key/value table, so this needs no
 * migration.
 */
const SESSION_KEY = 'kaggle_session';

async function rememberSession(id: number | null): Promise<void> {
  lastSessionId = id;
  try {
    if (id === null) {
      await query('DELETE FROM app_settings WHERE key = $1', [SESSION_KEY]);
    } else {
      await query(
        `INSERT INTO app_settings (key, value, updated_at)
         VALUES ($1, $2::jsonb, now())
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()`,
        [SESSION_KEY, JSON.stringify({ sessionId: id, startedAt: new Date().toISOString() })],
      );
    }
  } catch {
    // A failure to persist must not fail the start itself; the in-memory copy
    // still works for the life of this process.
  }
}

/** Recover the id of a session started before the last restart. */
export async function loadPersistedSession(): Promise<number | null> {
  if (lastSessionId !== null) return lastSessionId;
  try {
    const rows = await query<{ value: { sessionId?: number } }>(
      'SELECT value FROM app_settings WHERE key = $1',
      [SESSION_KEY],
    );
    const id = rows.rows[0]?.value?.sessionId;
    if (typeof id === 'number') lastSessionId = id;
  } catch {
    /* table not ready yet */
  }
  return lastSessionId;
}

export function getLastSessionId(): number | null {
  return lastSessionId;
}

export interface KaggleStatus {
  configured: boolean;
  status?: string;
  failureMessage?: string | null;
  sessionId?: number | null;
  error?: string;
}

export interface KaggleQuota {
  gpuSecondsUsed?: number | null;
  gpuSecondsLimit?: number | null;
  /** When the weekly allowance resets. */
  refreshesAt?: string | null;
  error?: string;
}

export function isConfigured(): boolean {
  return Boolean(config.kaggle.apiToken && config.kaggle.kernelSlug);
}

async function kaggleFetch(path: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  try {
    return await fetch(`${KAGGLE_REST}${path}`, {
      ...init,
      headers: {
        Authorization: `Bearer ${config.kaggle.apiToken}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...(init?.headers ?? {}),
      },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

/** RPC-style call on the service host, mirroring the official SDK. */
async function rpc<T>(method: string, body: Record<string, unknown>): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  try {
    const res = await fetch(`${KAGGLE_RPC}/kernels.KernelsApiService/${method}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${config.kaggle.apiToken}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`Kaggle ${method} returned ${res.status}: ${text.slice(0, 200)}`);
    }
    return (text ? JSON.parse(text) : {}) as T;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Live session state.
 *
 * Uses the RPC endpoint, not REST /kernels/status. The REST route reports the
 * status of the last saved *version* run, which stays "running" long after the
 * interactive session has ended — verified: RPC said COMPLETE and the quota
 * meter had stopped while REST still claimed running.
 */
export async function getStatus(): Promise<KaggleStatus> {
  if (!isConfigured()) return { configured: false };

  const [owner, slug] = config.kaggle.kernelSlug.split('/');
  try {
    const body = await rpc<{ status?: string; failureMessage?: string | null }>(
      'GetKernelSessionStatus',
      { userName: owner, kernelSlug: slug },
    );
    return {
      configured: true,
      status: (body.status ?? 'unknown').toLowerCase(),
      failureMessage: body.failureMessage ?? null,
      sessionId: lastSessionId,
    };
  } catch (err) {
    return { configured: true, error: (err as Error).message };
  }
}

/** Remaining GPU allowance — the reason the on/off control exists at all. */
export async function getQuota(): Promise<KaggleQuota> {
  if (!isConfigured()) return {};
  try {
    const res = await kaggleFetch('/kernels/quota');
    if (!res.ok) return { error: `Kaggle returned ${res.status}` };
    // Shape confirmed against the live API: durations are protobuf-style
    // {seconds, nanos} objects nested under gpuQuota, not flat fields.
    const body = (await res.json()) as {
      quotaRefreshTime?: string;
      gpuQuota?: {
        timeUsed?: { seconds?: number };
        totalTimeAllowed?: { seconds?: number };
      };
    };
    return {
      gpuSecondsUsed: body.gpuQuota?.timeUsed?.seconds ?? null,
      gpuSecondsLimit: body.gpuQuota?.totalTimeAllowed?.seconds ?? null,
      refreshesAt: body.quotaRefreshTime ?? null,
    };
  } catch (err) {
    return { error: (err as Error).message };
  }
}

/**
 * Run the notebook exactly as configured.
 *
 * `machineShape` is deliberately omitted so the notebook keeps its own
 * accelerator selection (2x T4), which the API has no way to request directly.
 */
export async function startSession(): Promise<{ ok: boolean; message: string }> {
  if (!isConfigured()) {
    return { ok: false, message: 'Kaggle control is not configured on the server.' };
  }

  const body: Record<string, unknown> = {
    slug: config.kaggle.kernelSlug,
    enableInternet: true,
  };
  // Only pin the accelerator if explicitly overridden; normally we inherit the
  // notebook's own choice, which is the only way to get the 2x T4 pair.
  if (config.kaggle.machineShape) body.machineShape = config.kaggle.machineShape;

  try {
    let result: { metadata?: { kernelSessionId?: number } };
    try {
      result = await rpc<{ metadata?: { kernelSessionId?: number } }>('CreateKernelSession', body);
    } catch (err) {
      // Kaggle refuses a second session for the same kernel, and getStatus is
      // not a reliable way to know one is up: it has reported "complete" while
      // a session was alive and billing. When a start is refused this way and
      // the caller already established that no worker is connected, what is
      // running is an orphan that never registered -- so cancel the session we
      // remember and try once more, instead of leaving it to burn quota until
      // someone notices. Twice today that recovery was done by hand.
      const message = (err as Error).message;
      const orphan = await loadPersistedSession();
      if (!message.includes('ALREADY_EXISTS') || !orphan) throw err;

      await rpc('CancelKernelSession', { kernelSessionId: orphan });
      await rememberSession(null);
      result = await rpc<{ metadata?: { kernelSessionId?: number } }>('CreateKernelSession', body);
    }
    await rememberSession(result.metadata?.kernelSessionId ?? null);
    return {
      ok: true,
      message:
        `Kaggle session ${lastSessionId ?? ''} starting with the notebook's own settings. ` +
        'The model takes a few minutes to load.',
    };
  } catch (err) {
    return { ok: false, message: (err as Error).message };
  }
}

/**
 * Fallback stop for a session whose worker never connected — or wedged and
 * stopped heart-beating while the session kept billing quota.
 */
export async function cancelSession(
  sessionId?: number,
): Promise<{ ok: boolean; message: string }> {
  if (!isConfigured()) return { ok: false, message: 'Kaggle control is not configured.' };
  const id = sessionId ?? (await loadPersistedSession());
  if (!id) {
    return {
      ok: false,
      message:
        'No session id is known. Kaggle only returns one when the session is started ' +
        'from here, so a session started in the notebook must be stopped there.',
    };
  }
  try {
    await rpc('CancelKernelSession', { kernelSessionId: id });
    if (id === lastSessionId) await rememberSession(null);
    return { ok: true, message: `Cancelled Kaggle session ${id}.` };
  } catch (err) {
    return { ok: false, message: (err as Error).message };
  }
}
