/** Shared protocol + domain types. */

export type WorkerStatus =
  | 'WORKER_STARTING'
  | 'WORKER_READY'
  | 'WORKER_BUSY'
  | 'WORKER_ERROR'
  | 'WORKER_OFFLINE';

export interface GpuInfo {
  index: number;
  name: string;
  totalMb: number;
  freeMb?: number;
  utilPct?: number | null;
  memUtilPct?: number | null;
  tempC?: number | null;
  powerW?: number | null;
}

/** Payload a worker sends on its `register` frame. */
export interface WorkerRegistration {
  workerId: string;
  name: string;
  engine: string;
  /** The active model. Kept for workers built before `loaded` existed. */
  model: string;
  gpus: GpuInfo[];
  vramTotalMb?: number;
  maxContext?: number;
  version?: string;
  capabilities?: Record<string, unknown>;
  /** Where this worker runs, so the UI can offer Kaggle and Colab separately. */
  platform?: 'kaggle' | 'colab' | 'other';
  /** Model ids resident in VRAM right now — servable with no load delay. */
  loaded?: string[];
  /** Model ids this worker could load, given its VRAM. */
  loadable?: string[];
  /** VRAM still free for weights, in GB, after what is already resident. */
  freeVramGb?: number;
}

export interface GenerationParams {
  temperature: number;
  topP: number;
  topK?: number;
  maxTokens: number;
  repeatPenalty?: number;
  stop?: string[];
}

export interface ChatTurn {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/** ── Worker → Server frames ───────────────────────────────── */
export type WorkerInbound =
  | { type: 'register'; payload: WorkerRegistration }
  | {
      type: 'heartbeat';
      payload: {
        status: WorkerStatus;
        vramFreeMb?: number;
        activeJobs?: number;
        gpus?: GpuInfo[];
        idleSeconds?: number;
      };
    }
  | { type: 'job.accepted'; payload: { jobId: string } }
  | {
      type: 'model.state';
      payload: { loaded: string[]; loadable?: string[]; freeVramGb?: number; busyWith?: string };
    }
  | { type: 'model.error'; payload: { modelId: string; message: string } }
  | { type: 'job.first_token'; payload: { jobId: string; ttftMs: number } }
  | { type: 'job.delta'; payload: { jobId: string; text: string } }
  | { type: 'job.reasoning'; payload: { jobId: string; text: string } }
  | {
      /**
       * Image generation has no first token and no per-token stream, and
       * sd-server exposes no step counter -- only a status and a queue
       * position. Sending this as a `job.delta` would concatenate it into the
       * message body, so it is a frame of its own.
       */
      type: 'job.progress';
      payload: {
        jobId: string;
        status: string;
        elapsedMs: number;
        steps?: number;
        queuePosition?: number | null;
      };
    }
  | {
      type: 'job.done';
      payload: {
        jobId: string;
        promptTokens?: number;
        completionTokens?: number;
        genMs?: number;
        tokensPerSec?: number;
        stopReason?: string;
        /** Set when the turn produced a picture rather than text. */
        imageId?: string;
      };
    }
  | { type: 'job.error'; payload: { jobId: string; message: string; retryable?: boolean } }
  | { type: 'deregister'; payload: { reason?: string } }
  | { type: 'pong'; payload?: Record<string, never> };

/** ── Server → Worker frames ───────────────────────────────── */
export type WorkerOutbound =
  | {
      type: 'registered';
      payload: {
        ok: true;
        serverTime: string;
        heartbeatIntervalMs: number;
        /** Minutes with no work before the worker exits; 0 disables it. */
        idleShutdownMinutes?: number;
      };
    }
  | {
      type: 'job.assign';
      payload: {
        jobId: string;
        model: string;
        systemPrompt?: string;
        messages: ChatTurn[];
        params: GenerationParams;
      };
    }
  | { type: 'job.cancel'; payload: { jobId: string } }
  | {
      type: 'model.load';
      payload: {
        modelId: string;
        repo?: string;
        quantization?: string;
        contextLength?: number;
        /** Also fetch the mmproj projector so the model can read images. */
        vision?: boolean;
      };
    }
  | { type: 'model.unload'; payload: { modelId: string } }
  | { type: 'ping'; payload: Record<string, never> }
  | { type: 'shutdown'; payload: { reason?: string } }
  | { type: 'error'; payload: { message: string } };

/** ── Browser-facing SSE events ────────────────────────────── */
export type StreamEvent =
  | { event: 'status'; data: { state: string; workerId?: string; workerName?: string; queueMs?: number } }
  | { event: 'first_token'; data: { ttftMs: number } }
  | { event: 'delta'; data: { text: string; index: number } }
  | { event: 'reasoning'; data: { text: string; index: number } }
  | {
      event: 'progress';
      data: { status: string; elapsedMs: number; steps?: number; queuePosition?: number | null };
    }
  | { event: 'image'; data: { id: string; width?: number; height?: number } }
  | { event: 'done'; data: Record<string, unknown> }
  | { event: 'error'; data: { code: string; message: string } };
