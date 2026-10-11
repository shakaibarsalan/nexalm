import { EventEmitter } from 'node:events';
import type { WebSocket } from 'ws';
import { config } from '../config.js';
import { query } from '../db/pool.js';
import type { WorkerOutbound, WorkerRegistration, WorkerStatus } from '../lib/types.js';

/**
 * Where a worker runs, when it does not say so itself.
 *
 * Workers built before the platform field report only an id and a name, both of
 * which have carried a platform prefix since the first Kaggle bootstrap. This
 * only decides how the worker is grouped in the UI and whether a Kaggle session
 * needs cancelling, so a wrong guess is cosmetic rather than dangerous.
 */
function inferPlatform(id: string, name: string): 'kaggle' | 'colab' | 'other' {
  const hay = `${id} ${name}`.toLowerCase();
  if (hay.includes('kaggle')) return 'kaggle';
  if (hay.includes('colab')) return 'colab';
  return 'other';
}

/**
 * Where a worker runs, preferring evidence over assertion.
 *
 * A worker's id carries its platform and is assigned by the same bootstrap that
 * sets the declaration, so the two agree in every correct case. When they
 * disagree the id is believed: a Colab worker once registered declaring itself
 * Kaggle, and the UI then showed a Kaggle host that was not running while the
 * Colab one that was serving did not appear at all.
 */
function reconcilePlatform(
  id: string,
  name: string,
  declared?: 'kaggle' | 'colab' | 'other',
): 'kaggle' | 'colab' | 'other' {
  const fromId = inferPlatform(id, name);
  if (fromId !== 'other') return fromId;
  return declared ?? 'other';
}

export interface LiveWorker {
  id: string;
  name: string;
  socket: WebSocket;
  engine: string;
  model: string;
  gpus: WorkerRegistration['gpus'];
  vramTotalMb: number;
  vramFreeMb?: number;
  maxContext?: number;
  version?: string;
  /** What this build can do. A worker that predates a feature simply omits it. */
  capabilities?: Record<string, unknown>;
  status: WorkerStatus;
  activeJobs: Set<string>;
  lastHeartbeat: number;
  /** Seconds since the worker last had work; drives the idle-shutdown display. */
  idleSeconds?: number;
  /** Where this worker runs. Absent on workers built before platforms existed. */
  platform: 'kaggle' | 'colab' | 'other';
  /** Resident in VRAM: servable immediately. */
  loaded: string[];
  /** Could be loaded, given this worker's VRAM. */
  loadable: string[];
  /** VRAM free for further weights, in GB. */
  freeVramGb?: number;
  /** Set while a load is in flight, so two loads cannot race. */
  loading?: string;
}

/**
 * Tracks connected inference workers.
 *
 * The live socket map is authoritative for dispatch; Postgres mirrors it so the
 * UI and any future replica can read worker state. A sweeper expires workers
 * whose heartbeat lease has run out, so a Kaggle session that dies without a
 * clean deregister cannot stay marked available forever.
 */
class WorkerRegistry extends EventEmitter {
  private workers = new Map<string, LiveWorker>();
  private sweeper?: NodeJS.Timeout;

  async register(socket: WebSocket, reg: WorkerRegistration): Promise<LiveWorker> {
    // A reconnecting worker reusing its id replaces the stale socket.
    const existing = this.workers.get(reg.workerId);
    if (existing && existing.socket !== socket) {
      try {
        existing.socket.close(4001, 'replaced by new connection');
      } catch {
        /* already gone */
      }
    }

    const vramTotalMb = reg.vramTotalMb ?? reg.gpus.reduce((sum, g) => sum + (g.totalMb || 0), 0);

    // A worker built before multi-model support sends neither `loaded` nor
    // `platform`. Treating its single `model` as a one-entry resident set keeps
    // it a first-class member of the pool instead of one that can never be
    // routed to, which is what makes this change safe to deploy on its own.
    // `undefined` means a worker too old to report residency, and its single
    // `model` is the honest answer. `[]` means a worker that reported and has
    // nothing loaded -- which is now the normal state at startup, since
    // preloading a model cost every session minutes before it could do
    // anything. Collapsing the two would advertise a model that is not there
    // and route jobs to it.
    const loaded = reg.loaded === undefined ? [reg.model].filter(Boolean) : reg.loaded;
    const loadable = reg.loadable?.length ? reg.loadable : loaded;

    const worker: LiveWorker = {
      id: reg.workerId,
      name: reg.name,
      socket,
      engine: reg.engine,
      model: reg.model,
      gpus: reg.gpus ?? [],
      vramTotalMb,
      maxContext: reg.maxContext,
      version: reg.version,
      capabilities: reg.capabilities,
      status: 'WORKER_READY',
      activeJobs: new Set(),
      lastHeartbeat: Date.now(),
      // The id prefix wins over a contradictory declaration. Both are set by
      // the same bootstrap, so they should agree -- and when they do not, the
      // id is the one that has been right every time: a Colab worker turned up
      // declaring itself Kaggle and the whole pool was mislabelled behind it.
      platform: reconcilePlatform(reg.workerId, reg.name, reg.platform),
      loaded,
      loadable,
      freeVramGb: reg.freeVramGb,
    };

    this.workers.set(worker.id, worker);
    await this.persist(worker);
    this.emit('changed');
    return worker;
  }

  async heartbeat(
    workerId: string,
    patch: {
      status?: WorkerStatus;
      vramFreeMb?: number;
      gpus?: LiveWorker['gpus'];
      idleSeconds?: number;
    },
  ): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) return;
    worker.lastHeartbeat = Date.now();
    if (patch.vramFreeMb !== undefined) worker.vramFreeMb = patch.vramFreeMb;
    if (patch.gpus?.length) worker.gpus = patch.gpus;
    if (patch.idleSeconds !== undefined) worker.idleSeconds = patch.idleSeconds;
    // A worker running a job is BUSY regardless of what it self-reports.
    if (patch.status && worker.activeJobs.size === 0) worker.status = patch.status;

    await query(
      `UPDATE workers
          SET last_heartbeat_at = now(),
              lease_expires_at  = now() + ($2 || ' milliseconds')::interval,
              vram_free_mb      = COALESCE($3, vram_free_mb),
              status            = $4,
              active_jobs       = $5
        WHERE id = $1`,
      [
        workerId,
        String(config.leaseTimeoutMs),
        patch.vramFreeMb ?? null,
        worker.status,
        worker.activeJobs.size,
      ],
    );

    await this.recordSample(worker);
  }

  /** One row per GPU per heartbeat, feeding the utilisation chart. */
  private async recordSample(worker: LiveWorker): Promise<void> {
    if (worker.gpus.length === 0) return;
    try {
      for (const gpu of worker.gpus) {
        await query(
          `INSERT INTO worker_samples
             (worker_id, gpu_index, util_pct, mem_used_mb, mem_total_mb, temp_c, power_w, active_jobs)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
          [
            worker.id,
            gpu.index ?? 0,
            gpu.utilPct ?? null,
            gpu.freeMb != null && gpu.totalMb != null ? gpu.totalMb - gpu.freeMb : null,
            gpu.totalMb ?? null,
            gpu.tempC ?? null,
            gpu.powerW ?? null,
            worker.activeJobs.size,
          ],
        );
      }
      // Prune opportunistically rather than on a timer — cheap, and keeps the
      // table bounded without another moving part.
      if (Math.random() < 0.02) {
        await query(`DELETE FROM worker_samples WHERE sampled_at < now() - interval '2 hours'`);
      }
    } catch (err) {
      console.error('[workers] failed to record telemetry sample', err);
    }
  }

  /** Ask a worker to exit. Ending its process ends the hosted session. */
  requestShutdown(workerId: string, reason: string): boolean {
    return this.send(workerId, { type: 'shutdown', payload: { reason } });
  }

  async setStatus(workerId: string, status: WorkerStatus, lastError?: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (worker) worker.status = status;
    await query(
      `UPDATE workers SET status = $2, last_error = COALESCE($3, last_error) WHERE id = $1`,
      [workerId, status, lastError ?? null],
    );
    this.emit('changed');
  }

  async deregister(workerId: string, reason?: string): Promise<void> {
    const worker = this.workers.get(workerId);
    this.workers.delete(workerId);
    if (worker) {
      for (const jobId of worker.activeJobs) {
        this.emit('job-orphaned', { jobId, workerId, reason: reason ?? 'worker disconnected' });
      }
    }
    await query(
      `UPDATE workers
          SET status = 'WORKER_OFFLINE', deregistered_at = now(),
              last_error = COALESCE($2, last_error)
        WHERE id = $1`,
      [workerId, reason ?? null],
    );
    this.emit('changed');
  }

  /**
   * Least-loaded worker advertising exactly this model, or undefined.
   *
   * It used to fall back to *any* worker when none advertised the requested
   * model. With a single model that was a harmless convenience. With several it
   * is a silent lie: you ask for Gemma, the job runs on Qwen, and the answer
   * comes back with no indication it came from a different model. Refusing is
   * the only honest answer, and the NO_WORKER_AVAILABLE path already handles it.
   *
   * BUSY workers stay eligible up to `workerMaxQueue`: excluding them would make
   * a single-worker deployment look offline for the whole of every generation.
   */
  pick(model?: string): LiveWorker | undefined {
    const candidates = this.available();
    if (candidates.length === 0) return undefined;

    if (!model) return candidates.sort((a, b) => a.activeJobs.size - b.activeJobs.size)[0];

    // Resident first: a worker with the weights already in VRAM answers now,
    // where one that merely *could* load them costs a minute or more. Only a
    // worker that actually has the model is ever returned.
    const resident = candidates.filter((w) => w.loaded.includes(model));
    if (resident.length === 0) return undefined;
    return resident.sort((a, b) => a.activeJobs.size - b.activeJobs.size)[0];
  }

  /** Workers that could accept a job right now, regardless of model. */
  private available(): LiveWorker[] {
    return [...this.workers.values()].filter(
      (w) =>
        (w.status === 'WORKER_READY' || w.status === 'WORKER_BUSY') &&
        w.activeJobs.size < config.workerMaxQueue,
    );
  }

  /**
   * Models the connected workers can actually serve.
   *
   * So that refusing a job can say what *is* available instead of only what is
   * not — otherwise the honest failure above is less useful than the dishonest
   * success it replaces.
   */
  availableModels(): string[] {
    return [...new Set(this.available().flatMap((w) => w.loaded))].sort();
  }

  /** Largest VRAM among connected workers, in GB — what the catalogue fits against. */
  largestVramGb(): number {
    return Math.max(0, ...this.available().map((w) => (w.vramTotalMb ?? 0) / 1024));
  }

  /**
   * A worker with room for a model of this size, most free VRAM first.
   *
   * Sized against total VRAM rather than what is free, because a worker can
   * evict a resident model to make room. Free VRAM only decides which of two
   * capable workers is asked, since the one that need not evict is faster.
   */
  pickForLoad(minVramGb: number): LiveWorker | undefined {
    const able = this.available().filter(
      (w) => !w.loading && (w.vramTotalMb ?? 0) / 1024 >= minVramGb,
    );
    if (able.length === 0) return undefined;
    return able.sort((a, b) => (b.freeVramGb ?? 0) - (a.freeVramGb ?? 0))[0];
  }

  /** Record what a worker reports about its resident set. */
  async setModelState(
    workerId: string,
    state: { loaded: string[]; loadable?: string[]; freeVramGb?: number; busyWith?: string },
  ): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) return;
    worker.loaded = state.loaded;
    if (state.loadable?.length) worker.loadable = state.loadable;
    if (state.freeVramGb !== undefined) worker.freeVramGb = state.freeVramGb;
    worker.loading = state.busyWith;
    // Keep `model` meaningful for anything still reading a single value.
    if (state.loaded.length > 0 && !state.loaded.includes(worker.model)) {
      worker.model = state.loaded[0];
    }
    await this.persist(worker);
    this.emit('changed');
  }

  /**
   * Whether this worker's build can generate images.
   *
   * A session that started before the diffusion engine shipped is still
   * connected and still answering chat perfectly well, and nothing else
   * distinguishes it. Sending it an image model would put diffusion weights
   * into llama.cpp, which fails *after* evicting a working model to make room
   * -- the user loses what they had and gains nothing.
   */
  canGenerateImages(workerId: string): boolean {
    return Boolean(this.workers.get(workerId)?.capabilities?.imageGeneration);
  }

  /** Ask a worker to bring a model into VRAM. Returns false if it cannot be asked. */
  requestLoad(
    workerId: string,
    payload: {
      modelId: string;
      repo?: string;
      quantization?: string;
      contextLength?: number;
      vision?: boolean;
      /**
       * Which runtime serves this model. The column has existed since the
       * first migration and nothing ever sent it, so every load built a
       * llama.cpp engine -- harmless while every model was a language model,
       * and the reason a diffusion model was handed to llama.cpp.
       */
      engine?: string;
      family?: string;
      kind?: string;
    },
  ): boolean {
    const worker = this.workers.get(workerId);
    if (!worker) return false;
    // Marked before the frame is acknowledged so two callers racing on the same
    // worker cannot both start a multi-gigabyte load.
    worker.loading = payload.modelId;

    // A worker built before model switching ignores frames it does not know,
    // and would then look permanently busy and never accept another load. The
    // window is generous because a first load genuinely can take many minutes
    // -- this releases a stuck flag, it does not cancel a real load.
    setTimeout(
      () => {
        const w = this.workers.get(workerId);
        if (w && w.loading === payload.modelId && !w.loaded.includes(payload.modelId)) {
          w.loading = undefined;
          this.emit('changed');
        }
      },
      20 * 60 * 1000,
    ).unref?.();

    return this.send(workerId, { type: 'model.load', payload });
  }

  requestUnload(workerId: string, modelId: string): boolean {
    return this.send(workerId, { type: 'model.unload', payload: { modelId } });
  }

  get(workerId: string): LiveWorker | undefined {
    return this.workers.get(workerId);
  }

  list(): LiveWorker[] {
    return [...this.workers.values()];
  }

  /** True when at least one worker could accept a job right now. */
  hasAvailable(model?: string): boolean {
    return this.pick(model) !== undefined;
  }

  send(workerId: string, frame: WorkerOutbound): boolean {
    const worker = this.workers.get(workerId);
    if (!worker || worker.socket.readyState !== 1 /* OPEN */) return false;
    worker.socket.send(JSON.stringify(frame));
    return true;
  }

  markJobStarted(workerId: string, jobId: string): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;
    worker.activeJobs.add(jobId);
    worker.status = 'WORKER_BUSY';
    void query(`UPDATE workers SET status = 'WORKER_BUSY', active_jobs = $2 WHERE id = $1`, [
      workerId,
      worker.activeJobs.size,
    ]).catch(() => undefined);
    this.emit('changed');
  }

  markJobFinished(workerId: string, jobId: string): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;
    worker.activeJobs.delete(jobId);
    worker.status = worker.activeJobs.size > 0 ? 'WORKER_BUSY' : 'WORKER_READY';
    void query(`UPDATE workers SET status = $2, active_jobs = $3 WHERE id = $1`, [
      workerId,
      worker.status,
      worker.activeJobs.size,
    ]).catch(() => undefined);
    this.emit('changed');
  }

  private async persist(w: LiveWorker): Promise<void> {
    await query(
      `INSERT INTO workers (id, name, status, engine, model, gpus, gpu_count,
                            vram_total_mb, vram_free_mb, max_context, version,
                            platform, loaded_models, loadable_models,
                            registered_at, last_heartbeat_at, lease_expires_at, deregistered_at)
       VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,$11,$12,$13::jsonb,$14::jsonb,
               now(), now(), now() + ($15 || ' milliseconds')::interval, NULL)
       ON CONFLICT (id) DO UPDATE SET
         name = EXCLUDED.name, status = EXCLUDED.status, engine = EXCLUDED.engine,
         model = EXCLUDED.model, gpus = EXCLUDED.gpus, gpu_count = EXCLUDED.gpu_count,
         vram_total_mb = EXCLUDED.vram_total_mb, max_context = EXCLUDED.max_context,
         version = EXCLUDED.version, platform = EXCLUDED.platform,
         loaded_models = EXCLUDED.loaded_models, loadable_models = EXCLUDED.loadable_models,
         last_heartbeat_at = now(),
         lease_expires_at = EXCLUDED.lease_expires_at, deregistered_at = NULL`,
      [
        w.id,
        w.name,
        w.status,
        w.engine,
        w.model,
        JSON.stringify(w.gpus),
        w.gpus.length,
        w.vramTotalMb,
        w.vramFreeMb ?? null,
        w.maxContext ?? null,
        w.version ?? null,
        w.platform,
        JSON.stringify(w.loaded ?? []),
        JSON.stringify(w.loadable ?? []),
        String(config.leaseTimeoutMs),
      ],
    );
  }

  /** Expire workers that stopped heart-beating. */
  startSweeper(): void {
    if (this.sweeper) return;
    this.sweeper = setInterval(() => {
      const cutoff = Date.now() - config.leaseTimeoutMs;
      for (const worker of this.workers.values()) {
        if (worker.lastHeartbeat < cutoff) {
          console.warn(`[workers] lease expired for ${worker.id}; marking offline`);
          try {
            worker.socket.close(4008, 'heartbeat lease expired');
          } catch {
            /* ignore */
          }
          void this.deregister(worker.id, 'heartbeat lease expired');
        }
      }
      // Also clean rows left behind by a crashed API process.
      void query(
        `UPDATE workers SET status = 'WORKER_OFFLINE'
          WHERE status <> 'WORKER_OFFLINE'
            AND last_heartbeat_at < now() - ($1 || ' milliseconds')::interval`,
        [String(config.leaseTimeoutMs)],
      ).catch(() => undefined);
    }, config.sweepIntervalMs);
    this.sweeper.unref?.();
  }

  stopSweeper(): void {
    if (this.sweeper) clearInterval(this.sweeper);
    this.sweeper = undefined;
  }
}

export const workerRegistry = new WorkerRegistry();
