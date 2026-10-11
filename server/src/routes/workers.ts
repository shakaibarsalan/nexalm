import type { FastifyInstance } from 'fastify';
import { query } from '../db/pool.js';
import { requireAdmin } from '../middleware/auth.js';
import { workerRegistry } from '../services/workerRegistry.js';
import {
  cancelSession,
  getQuota,
  getStatus,
  isConfigured,
  startSession,
} from '../services/kaggleControl.js';

/**
 * A cold start costs ~3 minutes of wall clock (model download plus load) now
 * that the llama.cpp build is served prebuilt. Starting with less than double
 * that left would spend the remainder without ever answering a prompt.
 */
/**
 * Which platform a worker runs on, derived from its id and name.
 *
 * A stopgap: the worker does not declare its platform yet, and Phase 1 replaces
 * this with a field in the register frame. It decides only whether a Kaggle
 * session needs cancelling, and the cost of guessing wrong is a redundant
 * cancel call against a session that does not exist.
 */
function platformOf(worker: { id: string; name: string }): 'kaggle' | 'colab' | 'other' {
  const hay = `${worker.id} ${worker.name}`.toLowerCase();
  if (hay.includes('kaggle')) return 'kaggle';
  if (hay.includes('colab')) return 'colab';
  return 'other';
}

const MIN_START_SECONDS = 6 * 60;

/** Below this, the UI warns before the session is worth starting. */
const LOW_QUOTA_SECONDS = 45 * 60;

export function registerWorkerRoutes(app: FastifyInstance): void {
  /** Live pool state — drives the availability banner and settings panel. */
  app.get('/api/workers', async () => {
    const live = workerRegistry.list().map((w) => ({
      id: w.id,
      name: w.name,
      status: w.status,
      engine: w.engine,
      model: w.model,
      gpus: w.gpus,
      gpuCount: w.gpus.length,
      vramTotalMb: w.vramTotalMb,
      vramFreeMb: w.vramFreeMb ?? null,
      maxContext: w.maxContext ?? null,
      version: w.version ?? null,
      activeJobs: w.activeJobs.size,
      lastHeartbeatAt: new Date(w.lastHeartbeat).toISOString(),
      platform: w.platform,
      loaded: w.loaded,
      loadable: w.loadable,
      freeVramGb: w.freeVramGb ?? null,
      loading: w.loading ?? null,
    }));

    // Recently-seen workers that are no longer connected, for context in the UI.
    const { rows: recent } = await query(
      `SELECT id, name, status, engine, model, gpu_count, vram_total_mb,
              last_heartbeat_at, deregistered_at, last_error
         FROM workers
        WHERE last_heartbeat_at > now() - interval '1 hour'
        ORDER BY last_heartbeat_at DESC
        LIMIT 20`,
    );

    // Grouped by platform so the UI can show Kaggle and Colab as separate
    // things with their own state, rather than one anonymous pool.
    const platforms = (['kaggle', 'colab', 'other'] as const).map((name) => {
      const own = live.filter((w) => w.platform === name);
      return {
        platform: name,
        connected: own.length,
        workers: own.map((w) => w.id),
        loaded: [...new Set(own.flatMap((w) => w.loaded))].sort(),
        /** Only Kaggle can be started from here; Colab has no such API. */
        startable: name === 'kaggle' && isConfigured(),
      };
    });

    return {
      available: workerRegistry.hasAvailable(),
      connected: live.length,
      workers: live,
      platforms,
      loadedModels: workerRegistry.availableModels(),
      largestVramGb: workerRegistry.largestVramGb() || null,
      recent,
      /** Present only when Kaggle control is configured, so the UI can hide the buttons. */
      control: { kaggle: isConfigured() },
    };
  });

  /** Recent GPU telemetry for the utilisation chart. */
  app.get('/api/workers/samples', async (req) => {
    const { minutes } = req.query as { minutes?: string };
    const window = Math.min(Math.max(Number(minutes) || 15, 1), 120);
    const { rows } = await query(
      `SELECT worker_id, gpu_index, sampled_at, util_pct, mem_used_mb,
              mem_total_mb, temp_c, power_w, active_jobs
         FROM worker_samples
        WHERE sampled_at > now() - ($1 || ' minutes')::interval
        ORDER BY sampled_at ASC`,
      [String(window)],
    );
    return { windowMinutes: window, samples: rows };
  });

  /** Live state of the hosted session, independent of whether a worker connected. */
  app.get('/api/workers/session', async () => {
    const [status, quota] = await Promise.all([getStatus(), getQuota()]);
    return { kaggle: status, quota };
  });

  /**
   * Start the GPU session.
   *
   * Uses create-session, which runs the notebook as configured — so the 2x T4
   * pair and the attached Secrets are preserved. (A `push` would reset both.)
   */
  app.post('/api/workers/start', async (req, reply) => {
    // Starting a session spends a finite weekly GPU quota, so it is an
    // administrator action. Until now any signed-in account could do it.
    if (!requireAdmin(req, reply)) return reply;

    const body = (req.body ?? {}) as { force?: boolean; platform?: string };
    const platform = body.platform === 'colab' ? 'colab' : 'kaggle';

    // Colab has no start API at all -- a human has to open the notebook and run
    // the cells. Saying so plainly beats pretending the button did something.
    if (platform === 'colab') {
      return reply.send({
        ok: false,
        platform: 'colab',
        manual: true,
        message:
          'Colab cannot be started from here: Google provides no API for it. Open ' +
          'NexaLM_Colab_Worker.ipynb in Colab, set Runtime -> Change runtime type -> ' +
          'T4 GPU, and run the cells. It will connect on its own once the model loads.',
      });
    }

    if (!isConfigured()) {
      return reply.code(501).send({
        error: 'not_configured',
        message: 'Kaggle control is not configured on the server.',
      });
    }
    if (workerRegistry.list().length > 0) {
      return reply.code(409).send({
        error: 'already_running',
        message: 'A worker is already connected.',
      });
    }
    // Refuse rather than silently burning the last of the week's allowance on a
    // session that cannot finish loading. A cold start costs roughly three
    // minutes; below that the session would be killed mid-load and the quota
    // spent for nothing.
    const quota = await getQuota();
    const used = quota.gpuSecondsUsed ?? 0;
    const limit = quota.gpuSecondsLimit ?? 0;
    const remaining = limit > 0 ? limit - used : null;

    if (remaining !== null && remaining < MIN_START_SECONDS && !body.force) {
      return reply.code(409).send({
        error: 'quota_exhausted',
        message:
          `Only ${Math.max(0, Math.round(remaining / 60))} minutes of GPU quota remain, ` +
          'which is not enough to load the model. Use a Colab worker until the quota ' +
          `resets${quota.refreshesAt ? ` on ${quota.refreshesAt.slice(0, 10)}` : ''}.`,
        remainingSeconds: remaining,
        refreshesAt: quota.refreshesAt ?? null,
      });
    }

    const result = await startSession();
    if (!result.ok) return reply.code(502).send({ error: 'start_failed', message: result.message });
    return {
      ok: true,
      message: result.message,
      platform: 'kaggle',
      remainingSeconds: remaining,
      lowQuota: remaining !== null && remaining < LOW_QUOTA_SECONDS,
    };
  });

  /**
   * Bring a model into a worker's VRAM.
   *
   * Loading is the expensive operation in this system -- weights have to be
   * fetched and mapped -- so it is deliberate and administrator-gated rather
   * than a side effect of someone picking a model in a dropdown.
   */
  app.post('/api/workers/models/load', async (req, reply) => {
    if (!requireAdmin(req, reply)) return reply;

    const body = (req.body ?? {}) as { modelId?: string; workerId?: string };
    if (!body.modelId) {
      return reply.code(400).send({ error: 'missing_model', message: 'modelId is required.' });
    }

    const { rows } = await query<{ id: string; repo: string | null; quantization: string | null;
                                   context_length: number | null; min_vram_gb: string | null;
                                   supports_vision: boolean; engine: string | null;
                                   family: string | null; kind: string | null }>(
      `SELECT id, repo, quantization, context_length, min_vram_gb, supports_vision,
              engine, family, kind
         FROM models WHERE id = $1 AND enabled`,
      [body.modelId],
    );
    if (rows.length === 0) {
      return reply.code(404).send({ error: 'unknown_model', message: `No such model: ${body.modelId}.` });
    }
    const model = rows[0];

    const worker = body.workerId
      ? workerRegistry.get(body.workerId)
      : workerRegistry.pickForLoad(Number(model.min_vram_gb ?? 0));
    if (!worker) {
      // Say which constraint failed. "No worker" is true but useless when the
      // real reason is that the model is larger than the card.
      const need = Number(model.min_vram_gb ?? 0);
      const biggest = Math.max(0, ...workerRegistry.list().map((w) => (w.vramTotalMb ?? 0) / 1024));
      return reply.code(409).send({
        error: 'no_capable_worker',
        message:
          biggest > 0 && need > biggest
            ? `${body.modelId} needs about ${need} GB of VRAM; the largest connected worker has ` +
              `${biggest.toFixed(1)} GB.`
            : 'No connected worker can load this model right now.',
      });
    }
    // An image model on a worker that cannot serve one is the failure this
    // whole feature was fixing: llama.cpp is handed diffusion weights, fails
    // with "unknown model architecture", and the eviction it already performed
    // to make room stands. Refusing here costs nothing; being wrong costs the
    // user a loaded model.
    if (model.kind === 'image' && !workerRegistry.canGenerateImages(worker.id)) {
      return reply.code(409).send({
        error: 'worker_too_old',
        message:
          `${worker.name} is running a build from before image generation shipped, so it ` +
          'cannot load this model. Restart that GPU session — it will pick up the new ' +
          'worker code automatically — and try again.',
      });
    }

    if (worker.loaded.includes(body.modelId)) {
      return { ok: true, alreadyLoaded: true, workerId: worker.id,
               message: `${body.modelId} is already loaded on ${worker.name}.` };
    }
    if (worker.loading) {
      return reply.code(409).send({
        error: 'worker_busy',
        message: `${worker.name} is already loading ${worker.loading}.`,
      });
    }

    const sent = workerRegistry.requestLoad(worker.id, {
      modelId: model.id,
      repo: model.repo ?? undefined,
      quantization: model.quantization ?? undefined,
      contextLength: model.context_length ?? undefined,
      vision: model.supports_vision,
      engine: model.engine ?? undefined,
      family: model.family ?? undefined,
      kind: model.kind ?? undefined,
    });
    if (!sent) {
      return reply.code(502).send({ error: 'unreachable', message: 'Could not reach that worker.' });
    }
    return {
      ok: true,
      workerId: worker.id,
      message:
        `Loading ${body.modelId} on ${worker.name}. First use of a model downloads it, ` +
        'which takes a few minutes; afterwards it loads from local disk.',
    };
  });

  /** Free a model's VRAM without stopping the worker. */
  app.post('/api/workers/models/unload', async (req, reply) => {
    if (!requireAdmin(req, reply)) return reply;

    const body = (req.body ?? {}) as { modelId?: string; workerId?: string };
    if (!body.modelId) {
      return reply.code(400).send({ error: 'missing_model', message: 'modelId is required.' });
    }
    const holders = workerRegistry
      .list()
      .filter((w) => w.loaded.includes(body.modelId!) && (!body.workerId || w.id === body.workerId));
    if (holders.length === 0) {
      return reply.code(404).send({
        error: 'not_loaded',
        message: `No connected worker has ${body.modelId} loaded.`,
      });
    }
    const asked = holders.filter((w) => workerRegistry.requestUnload(w.id, body.modelId!));
    return {
      ok: asked.length > 0,
      unloaded: asked.map((w) => w.name),
      message: `Asked ${asked.length} worker(s) to unload ${body.modelId}.`,
    };
  });

  /**
   * Stop the GPU session by asking the worker to exit.
   *
   * Done over our own socket rather than Kaggle's API so it works for any
   * host — Colab, a dedicated box — and because ending the process is what
   * actually stops the quota meter.
   */
  app.post('/api/workers/stop', async (req, reply) => {
    if (!requireAdmin(req, reply)) return reply;

    const body = (req.body ?? {}) as { workerId?: string; all?: boolean };
    const connected = workerRegistry.list();

    // Choosing which worker to stop only became necessary once more than one
    // could be connected. Stopping the whole pool because the caller did not
    // say which is how a single click took down both Kaggle and Colab.
    let workers = connected;
    if (body.workerId) {
      workers = connected.filter((w) => w.id === body.workerId);
      if (workers.length === 0 && connected.length > 0) {
        return reply.code(404).send({
          error: 'no_such_worker',
          message: `No connected worker with id ${body.workerId}.`,
          connected: connected.map((w) => ({ id: w.id, name: w.name })),
        });
      }
    } else if (connected.length > 1 && !body.all) {
      return reply.code(400).send({
        error: 'ambiguous_target',
        message:
          'More than one worker is connected. Name the one to stop with workerId, ' +
          'or pass all:true to stop every one of them.',
        connected: connected.map((w) => ({ id: w.id, name: w.name, model: w.model })),
      });
    }

    if (workers.length === 0) {
      // No worker ever connected, but a session may still be burning quota.
      const status = await getStatus();
      if (String(status.status).toLowerCase() === 'running') {
        const cancelled = await cancelSession();
        return reply.code(cancelled.ok ? 200 : 502).send({
          ok: cancelled.ok,
          stopped: [],
          message: cancelled.ok
            ? 'No worker was connected; cancelled the Kaggle session directly.'
            : cancelled.message,
        });
      }
      return reply.code(409).send({ error: 'no_worker', message: 'No worker is connected.' });
    }
    const asked = workers.filter((w) =>
      workerRegistry.requestShutdown(w.id, 'stopped from the dashboard'),
    );
    if (asked.length === 0) {
      return reply
        .code(502)
        .send({ error: 'unreachable', message: 'Could not reach any worker.' });
    }

    // A successful socket write only proves the frame was sent. Confirm the
    // worker actually went away: an older build silently ignores frames it does
    // not know, and reporting "stopped" while it keeps burning quota would be
    // worse than reporting a failure.
    const deadline = Date.now() + 20_000;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 1000));
      if (asked.every((w) => workerRegistry.get(w.id) === undefined)) {
        // The worker process is gone, but that is not enough on its own: a
        // Kaggle session created through the API is INTERACTIVE and outlives
        // the script that ran in it, so the quota meter keeps running. Cancel
        // it — but only when no Kaggle worker is left, or stopping Colab would
        // tear down a Kaggle session that is still serving.
        const stoppedKaggle = asked.some((w) => platformOf(w) === 'kaggle');
        const kaggleRemains = workerRegistry.list().some((w) => platformOf(w) === 'kaggle');
        const cancelled =
          stoppedKaggle && !kaggleRemains
            ? await cancelSession()
            : { ok: true, message: '' };
        return {
          ok: true,
          stopped: asked.map((w) => w.name),
          message: cancelled.ok
            ? stoppedKaggle && !kaggleRemains
              ? `Stopped ${asked.length} worker(s) and cancelled the Kaggle session. ` +
                'GPU quota is no longer being used.'
              : `Stopped ${asked.length} worker(s).`
            : `Stopped ${asked.length} worker(s), but the Kaggle session may still be ` +
              `running: ${cancelled.message}`,
        };
      }
    }

    const stillUp = asked.filter((w) => workerRegistry.get(w.id) !== undefined);
    return reply.code(504).send({
      error: 'shutdown_not_confirmed',
      stopped: asked.filter((w) => !stillUp.includes(w)).map((w) => w.name),
      message:
        `Asked ${asked.length} worker(s) to stop, but ${stillUp.length} did not disconnect. ` +
        'A worker running a build older than this feature ignores the request — ' +
        'stop the session from the Kaggle notebook instead.',
    });
  });
}
