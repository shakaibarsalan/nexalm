import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';
import { config } from '../config.js';
import { query } from '../db/pool.js';
import { workerRegistry } from './workerRegistry.js';
import type { ChatTurn, GenerationParams, StreamEvent } from '../lib/types.js';

export type JobState = 'queued' | 'assigned' | 'running' | 'complete' | 'error' | 'cancelled';

interface Job {
  id: string;
  conversationId: string;
  messageId: string;
  model: string;
  params: GenerationParams;
  state: JobState;
  workerId?: string;
  workerName?: string;
  /** Ordered deltas, replayed to a client that reconnects mid-stream. */
  chunks: string[];
  text: string;
  /** Thinking trace, kept separate from the answer. */
  reasoningChunks: string[];
  reasoning: string;
  subscribers: Set<(ev: StreamEvent) => void>;
  createdAt: number;
  assignedAt?: number;
  firstTokenAt?: number;
  finishedAt?: number;
  ttftMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  tokensPerSec?: number;
  stopReason?: string;
  error?: string;
  timeout?: NodeJS.Timeout;
}

export class NoWorkerAvailableError extends Error {
  code = 'NO_WORKER_AVAILABLE';
  constructor() {
    super('No GPU worker is currently available.');
  }
}

/**
 * Owns the lifecycle of a single generation: dispatch to a worker, fan tokens
 * out to SSE subscribers, and persist metrics when it ends.
 *
 * Finished jobs are kept briefly so a client that reconnects just after
 * completion still receives the terminal event instead of hanging.
 */
class JobManager extends EventEmitter {
  private jobs = new Map<string, Job>();
  private readonly retainAfterFinishMs = 60_000;

  constructor() {
    super();
    workerRegistry.on('job-orphaned', ({ jobId, reason }: { jobId: string; reason: string }) => {
      void this.fail(jobId, `Worker disconnected: ${reason}`);
    });
  }

  async create(input: {
    /** Citations behind this answer, kept so [1] still resolves after a reload. */
    sources?: unknown;
    conversationId: string;
    messageId: string;
    model: string;
    systemPrompt?: string;
    messages: ChatTurn[];
    params: GenerationParams;
  }): Promise<Job> {
    const worker = workerRegistry.pick(input.model);
    if (!worker) throw new NoWorkerAvailableError();

    const jobId = randomUUID();
    const job: Job = {
      id: jobId,
      conversationId: input.conversationId,
      messageId: input.messageId,
      model: input.model,
      params: input.params,
      state: 'queued',
      chunks: [],
      text: '',
      reasoningChunks: [],
      reasoning: '',
      subscribers: new Set(),
      createdAt: Date.now(),
    };
    this.jobs.set(jobId, job);

    const gpuInfo = worker.gpus.map((g) => `${g.name} (${Math.round(g.totalMb / 1024)}GB)`).join(', ');

    await query(
      `INSERT INTO inference_runs
         (job_id, message_id, conversation_id, worker_id, worker_name, gpu_info,
          model, engine, status, params, sources, created_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'queued',$9::jsonb,$10::jsonb, now())`,
      [
        jobId,
        input.messageId,
        input.conversationId,
        worker.id,
        worker.name,
        gpuInfo || null,
        input.model,
        worker.engine,
        JSON.stringify(input.params),
        input.sources ? JSON.stringify(input.sources) : null,
      ],
    );

    const sent = workerRegistry.send(worker.id, {
      type: 'job.assign',
      payload: {
        jobId,
        model: input.model,
        systemPrompt: input.systemPrompt,
        messages: input.messages,
        params: input.params,
      },
    });

    if (!sent) {
      this.jobs.delete(jobId);
      await query(
        `UPDATE inference_runs SET status='error', error='worker socket closed before dispatch',
             finished_at = now() WHERE job_id = $1`,
        [jobId],
      );
      throw new NoWorkerAvailableError();
    }

    job.workerId = worker.id;
    job.workerName = worker.name;
    job.state = 'assigned';
    job.assignedAt = Date.now();
    workerRegistry.markJobStarted(worker.id, jobId);

    job.timeout = setTimeout(() => {
      void this.fail(jobId, `Generation timed out after ${config.jobTimeoutMs}ms`);
    }, config.jobTimeoutMs);
    job.timeout.unref?.();

    this.publish(job, {
      event: 'status',
      data: { state: 'assigned', workerId: worker.id, workerName: worker.name },
    });

    return job;
  }

  get(jobId: string): Job | undefined {
    return this.jobs.get(jobId);
  }

  /** Subscribe an SSE client; buffered chunks are replayed first. */
  subscribe(jobId: string, onEvent: (ev: StreamEvent) => void): (() => void) | undefined {
    const job = this.jobs.get(jobId);
    if (!job) return undefined;

    onEvent({
      event: 'status',
      data: { state: job.state, workerId: job.workerId, workerName: job.workerName },
    });
    job.reasoningChunks.forEach((text, index) =>
      onEvent({ event: 'reasoning', data: { text, index } }),
    );
    job.chunks.forEach((text, index) => onEvent({ event: 'delta', data: { text, index } }));

    if (job.state === 'complete') {
      onEvent({ event: 'done', data: this.metricsOf(job) });
      return () => undefined;
    }
    if (job.state === 'error' || job.state === 'cancelled') {
      onEvent({
        event: 'error',
        data: { code: job.state.toUpperCase(), message: job.error ?? 'Generation stopped' },
      });
      return () => undefined;
    }

    job.subscribers.add(onEvent);
    return () => job.subscribers.delete(onEvent);
  }

  onFirstToken(jobId: string, ttftMs: number): void {
    const job = this.jobs.get(jobId);
    if (!job) return;
    job.state = 'running';
    job.firstTokenAt = Date.now();
    // Trust our own clock: the worker's ttft excludes network transit.
    job.ttftMs = ttftMs || job.firstTokenAt - (job.assignedAt ?? job.createdAt);
    this.publish(job, { event: 'first_token', data: { ttftMs: job.ttftMs } });
    void query(`UPDATE inference_runs SET status='running', started_at=now(), ttft_ms=$2 WHERE job_id=$1`, [
      jobId,
      job.ttftMs,
    ]).catch(() => undefined);
  }

  /**
   * A thinking token. Surfaced rather than dropped: a model like Qwen3 can
   * spend hundreds of tokens here before its first answer token, and the user
   * is already waiting for them.
   */
  onReasoning(jobId: string, text: string): void {
    const job = this.jobs.get(jobId);
    if (!job || !text) return;
    if (job.state === 'assigned') job.state = 'running';
    const index = job.reasoningChunks.length;
    job.reasoningChunks.push(text);
    job.reasoning += text;
    this.publish(job, { event: 'reasoning', data: { text, index } });
  }

  onDelta(jobId: string, text: string): void {
    const job = this.jobs.get(jobId);
    if (!job || !text) return;
    if (job.state === 'assigned') job.state = 'running';
    const index = job.chunks.length;
    job.chunks.push(text);
    job.text += text;
    this.publish(job, { event: 'delta', data: { text, index } });
  }

  /**
   * Diffusion progress. Carries elapsed time and the worker's queue position,
   * which is all sd-server actually reports -- it has no per-step counter, so
   * a percentage here would be invented rather than measured.
   */
  onProgress(
    jobId: string,
    payload: { status: string; elapsedMs: number; steps?: number; queuePosition?: number | null },
  ): void {
    const job = this.jobs.get(jobId);
    if (!job) return;
    if (job.state === 'assigned') job.state = 'running';
    this.publish(job, { event: 'progress', data: payload });
  }

  async onDone(
    jobId: string,
    payload: {
      promptTokens?: number;
      completionTokens?: number;
      genMs?: number;
      tokensPerSec?: number;
      stopReason?: string;
      imageId?: string;
    },
  ): Promise<void> {
    const job = this.jobs.get(jobId);
    if (!job || this.isTerminal(job)) return;

    job.state = 'complete';
    job.finishedAt = Date.now();
    job.promptTokens = payload.promptTokens;
    job.completionTokens = payload.completionTokens;
    job.stopReason = payload.stopReason;

    const genMs = payload.genMs ?? job.finishedAt - (job.firstTokenAt ?? job.createdAt);
    job.tokensPerSec =
      payload.tokensPerSec ??
      (payload.completionTokens && genMs > 0
        ? Number(((payload.completionTokens / genMs) * 1000).toFixed(2))
        : undefined);

    this.clearTimeout(job);
    if (job.workerId) workerRegistry.markJobFinished(job.workerId, jobId);

    await this.persistFinal(job, 'complete', genMs);
    await query(
      `UPDATE messages SET content = $2, reasoning = $3, status = 'complete', updated_at = now()
        WHERE id = $1`,
      [job.messageId, job.text, job.reasoning || null],
    );
    await query(`UPDATE conversations SET updated_at = now() WHERE id = $1`, [job.conversationId]);

    if (payload.imageId) {
      this.publish(job, { event: 'image', data: { id: payload.imageId } });
    }
    this.publish(job, { event: 'done', data: this.metricsOf(job) });
    this.scheduleCleanup(jobId);
  }

  async fail(jobId: string, message: string): Promise<void> {
    const job = this.jobs.get(jobId);
    if (!job || this.isTerminal(job)) return;

    job.state = 'error';
    job.error = message;
    job.finishedAt = Date.now();
    this.clearTimeout(job);
    if (job.workerId) workerRegistry.markJobFinished(job.workerId, jobId);

    await this.persistFinal(job, 'error', undefined, message);
    // Keep whatever text arrived before the failure — partial output is useful.
    await query(
      `UPDATE messages SET content = $2, reasoning = $3, status = 'error', error = $4,
              updated_at = now() WHERE id = $1`,
      [job.messageId, job.text, job.reasoning || null, message],
    );

    this.publish(job, { event: 'error', data: { code: 'INFERENCE_ERROR', message } });
    this.scheduleCleanup(jobId);
  }

  async cancel(jobId: string): Promise<boolean> {
    const job = this.jobs.get(jobId);
    if (!job || this.isTerminal(job)) return false;

    if (job.workerId) {
      workerRegistry.send(job.workerId, { type: 'job.cancel', payload: { jobId } });
      workerRegistry.markJobFinished(job.workerId, jobId);
    }

    job.state = 'cancelled';
    job.finishedAt = Date.now();
    this.clearTimeout(job);

    const genMs = job.finishedAt - (job.firstTokenAt ?? job.createdAt);
    await this.persistFinal(job, 'cancelled', genMs);
    await query(
      `UPDATE messages SET content = $2, reasoning = $3, status = 'cancelled', updated_at = now()
        WHERE id = $1`,
      [job.messageId, job.text, job.reasoning || null],
    );

    this.publish(job, { event: 'done', data: { ...this.metricsOf(job), cancelled: true } });
    this.scheduleCleanup(jobId);
    return true;
  }

  private metricsOf(job: Job): Record<string, unknown> {
    const genMs = (job.finishedAt ?? Date.now()) - (job.firstTokenAt ?? job.createdAt);
    return {
      jobId: job.id,
      messageId: job.messageId,
      state: job.state,
      model: job.model,
      workerId: job.workerId,
      workerName: job.workerName,
      promptTokens: job.promptTokens ?? null,
      completionTokens: job.completionTokens ?? null,
      totalTokens:
        job.promptTokens != null && job.completionTokens != null
          ? job.promptTokens + job.completionTokens
          : null,
      ttftMs: job.ttftMs ?? null,
      genMs,
      totalMs: (job.finishedAt ?? Date.now()) - job.createdAt,
      tokensPerSec: job.tokensPerSec ?? null,
      stopReason: job.stopReason ?? null,
      reasoningChars: job.reasoning.length || null,
    };
  }

  private async persistFinal(
    job: Job,
    status: JobState,
    genMs?: number,
    error?: string,
  ): Promise<void> {
    const totalMs = (job.finishedAt ?? Date.now()) - job.createdAt;
    const queueMs = (job.assignedAt ?? job.createdAt) - job.createdAt;
    const total =
      job.promptTokens != null && job.completionTokens != null
        ? job.promptTokens + job.completionTokens
        : null;

    await query(
      `UPDATE inference_runs SET
         status = $2, prompt_tokens = $3, completion_tokens = $4, total_tokens = $5,
         queue_ms = $6, ttft_ms = COALESCE($7, ttft_ms), gen_ms = $8, total_ms = $9,
         tokens_per_sec = $10, stop_reason = $11, error = $12, finished_at = now()
       WHERE job_id = $1`,
      [
        job.id,
        status,
        job.promptTokens ?? null,
        job.completionTokens ?? null,
        total,
        queueMs,
        job.ttftMs ?? null,
        genMs ?? null,
        totalMs,
        job.tokensPerSec ?? null,
        job.stopReason ?? null,
        error ?? null,
      ],
    ).catch((err) => console.error('[jobs] failed to persist metrics', err));
  }

  private publish(job: Job, ev: StreamEvent): void {
    for (const fn of job.subscribers) {
      try {
        fn(ev);
      } catch {
        /* a dead subscriber must not break the others */
      }
    }
  }

  private isTerminal(job: Job): boolean {
    return job.state === 'complete' || job.state === 'error' || job.state === 'cancelled';
  }

  private clearTimeout(job: Job): void {
    if (job.timeout) clearTimeout(job.timeout);
    job.timeout = undefined;
  }

  private scheduleCleanup(jobId: string): void {
    const t = setTimeout(() => {
      const job = this.jobs.get(jobId);
      job?.subscribers.clear();
      this.jobs.delete(jobId);
    }, this.retainAfterFinishMs);
    t.unref?.();
  }
}

export const jobManager = new JobManager();
