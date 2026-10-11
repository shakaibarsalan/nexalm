export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
  settings: ChatSettings;
  message_count?: number;
  total_tokens?: number;
  output_tokens?: number;
  avg_tokens_per_sec?: number | null;
}

export type MessageStatus = 'complete' | 'streaming' | 'pending' | 'error' | 'cancelled';

export interface WebSource {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

export interface Message {
  id: string;
  seq: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  /** Thinking trace, shown collapsed. Null for models that do not emit one. */
  reasoning?: string | null;
  status: MessageStatus;
  error: string | null;
  created_at: string;
  /** Joined from inference_runs — present on assistant turns only. */
  job_id?: string | null;
  model?: string | null;
  worker_name?: string | null;
  gpu_info?: string | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  ttft_ms?: number | null;
  gen_ms?: number | null;
  total_ms?: number | null;
  tokens_per_sec?: number | null;
  stop_reason?: string | null;
  /** Sources the answer was given, when web search was on. */
  sources?: WebSource[] | null;
  /** Set when this turn produced a picture instead of text. */
  image_id?: string | null;
  image_width?: number | null;
  image_height?: number | null;
}

export interface ChatSettings {
  model?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  contextLength?: number;
  repeatPenalty?: number;
  systemPrompt?: string;
  /** Let each chat see recent turns from the user's other chats. */
  useGlobalContext?: boolean;
  /** Start a Kaggle GPU automatically when a message has nowhere to run. */
  autoStartWorker?: boolean;
  /** Minutes a GPU worker may sit idle before it exits. 0 keeps it running. */
  idleShutdownMinutes?: number;
  /** Look things up on the web before answering. */
  webSearch?: boolean;
}

export interface ModelInfo {
  id: string;
  display_name: string;
  engine: string;
  quantization: string | null;
  context_length: number;
  params: Record<string, unknown>;
  enabled: boolean;
  repo?: string | null;
  family?: string | null;
  approx_size_gb?: number | null;
  min_vram_gb?: number | null;
  /** Resident in VRAM right now: answers with no load delay. */
  loaded?: boolean;
  /** Null when no worker is connected and fit cannot be judged. */
  fits?: boolean | null;
  availability?: 'ready' | 'loadable' | 'too_large' | 'no_worker';
  supports_vision?: boolean;
  /** 'chat' or 'image' — an image model answers a prompt with a picture. */
  kind?: 'chat' | 'image';
  /** Weights on disk. FLUX is four files, so this is not the same as size. */
  disk_gb?: number | null;
}

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
}

export interface WorkerInfo {
  id: string;
  name: string;
  status: WorkerStatus;
  engine: string;
  model: string;
  gpus: GpuInfo[];
  gpuCount: number;
  vramTotalMb: number;
  vramFreeMb: number | null;
  idleSeconds?: number | null;
  maxContext: number | null;
  version: string | null;
  activeJobs: number;
  lastHeartbeatAt: string;
  platform?: 'kaggle' | 'colab' | 'other';
  loaded?: string[];
  freeVramGb?: number | null;
  /** Model id currently being brought into VRAM, if any. */
  loading?: string | null;
}

export interface PlatformState {
  platform: 'kaggle' | 'colab' | 'other';
  connected: number;
  workers: string[];
  loaded: string[];
  /** Only Kaggle can be started from the app; Colab has no such API. */
  startable: boolean;
}

export interface WorkerPool {
  available: boolean;
  connected: number;
  workers: WorkerInfo[];
  platforms?: PlatformState[];
  loadedModels?: string[];
  largestVramGb?: number | null;
  recent: unknown[];
  control?: { kaggle: boolean };
}

/** One GPU reading, sampled per heartbeat. */
export interface WorkerSample {
  worker_id: string;
  gpu_index: number;
  sampled_at: string;
  util_pct: number | null;
  mem_used_mb: number | null;
  mem_total_mb: number | null;
  temp_c: number | null;
  power_w: number | null;
  active_jobs: number;
}

export interface SessionStatus {
  kaggle: {
    configured: boolean;
    status?: string;
    failureMessage?: string | null;
    sessionId?: number | null;
    error?: string;
  };
  quota?: {
    gpuSecondsUsed?: number | null;
    gpuSecondsLimit?: number | null;
    refreshesAt?: string | null;
    error?: string;
  };
}

export interface ConversationMetrics {
  runs: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  total_gen_ms: number;
  avg_tokens_per_sec: number | null;
  avg_ttft_ms: number | null;
  peak_tokens_per_sec: number | null;
}

/** Live metrics for the turn currently generating. */
export interface LiveMetrics {
  ttftMs?: number;
  genMs?: number;
  totalMs?: number;
  promptTokens?: number | null;
  completionTokens?: number | null;
  totalTokens?: number | null;
  tokensPerSec?: number | null;
  workerName?: string;
  model?: string;
  stopReason?: string | null;
  reasoningChars?: number | null;
}

export interface AuthUser {
  id: string;
  email: string;
  displayName: string | null;
  avatarUrl: string | null;
  role: 'admin' | 'user';
}

export interface AuthStatus {
  authenticated: boolean;
  user: AuthUser | null;
  googleEnabled: boolean;
  signupEnabled: boolean;
  requiresAccessToken: boolean;
  unlocked: boolean;
  /** No accounts exist yet — the first signup becomes the admin. */
  setupRequired: boolean;
  /** Sign-up sends a code to the address before creating the account. */
  emailVerification?: boolean;
}

/** Row shape from the admin users endpoint. */
export interface AdminUser {
  id: string;
  email: string;
  displayName: string | null;
  avatarUrl: string | null;
  role: 'admin' | 'user';
  googleLinked: boolean;
  createdAt: string;
  lastLoginAt: string | null;
  disabledAt: string | null;
  conversationCount: number;
  activeSessions: number;
}
