/**
 * Demo mode: the whole API, faked inside the browser.
 *
 * Built only with `vite build --mode demo`. It replaces `fetch` for `/api/*`
 * and `EventSource` for the token stream, so the real UI runs unchanged with
 * no server, no GPU and no keys. Replies are canned and the timings are
 * simulated (paced at roughly the speed measured on a Kaggle T4 pair).
 * Nothing leaves the page, and a reload starts fresh.
 */
import type {
  AdminUser,
  AuthUser,
  ChatSettings,
  Conversation,
  Message,
  ModelInfo,
  WorkerInfo,
  WorkerPool,
  WorkerSample,
} from '../lib/types';
import { CANNED, pickReply, type Canned } from './content';

// ── Catalogue: the same rows the migrations seed ───────────────────────────

const MODELS: ModelInfo[] = [
  m('qwen3-30b-a3b', 'Qwen3-30B-A3B (Q4_K_M)', 'Q4_K_M', 32768, 18.6, 21.1, 'qwen3-moe'),
  m('qwen3-30b-a3b-q3', 'Qwen3-30B-A3B (Q3_K_M, single GPU)', 'Q3_K_M', 16384, 11.8, 14.3, 'qwen3-moe'),
  m('gemma-3-4b', 'Gemma 3 4B', 'Q4_K_M', 32768, 2.5, 5.9, 'gemma3', true),
  m('qwen3-8b', 'Qwen3 8B', 'Q4_K_M', 32768, 5.0, 7.5, 'qwen3'),
  m('qwen2.5-vl-7b', 'Qwen2.5-VL 7B (vision)', 'Q4_K_M', 32768, 4.7, 8.1, 'qwen-vl', true),
  m('gemma-3-12b', 'Gemma 3 12B', 'Q4_K_M', 32768, 7.3, 10.7, 'gemma3', true),
  m('phi-4', 'Phi-4 14B', 'Q4_K_M', 16384, 8.9, 11.4, 'phi'),
  m('qwen3-14b', 'Qwen3 14B', 'Q4_K_M', 32768, 9.0, 11.5, 'qwen3'),
  m('gpt-oss-20b', 'GPT-OSS 20B', 'Q4_K_M', 32768, 11.6, 14.1, 'gpt-oss'),
  m('mistral-small-24b', 'Mistral Small 24B', 'Q4_K_M', 32768, 14.3, 16.8, 'mistral'),
  m('gemma-3-27b', 'Gemma 3 27B', 'Q4_K_M', 32768, 16.5, 19.9, 'gemma3', true),
  m('qwen3-32b', 'Qwen3 32B', 'Q4_K_M', 32768, 19.8, 22.3, 'qwen3'),
  m('llama-3.3-70b', 'Llama 3.3 70B (slow, dense)', 'UD-IQ2_M', 8192, 24.3, 26.8, 'llama3'),
  {
    ...m('flux-schnell', 'FLUX.1 schnell (images)', 'Q4_0', 512, 6.77, 9.3, 'flux'),
    engine: 'stablediffusion',
    kind: 'image',
    disk_gb: 10.03,
  },
  {
    ...m('sd-1-5', 'Stable Diffusion 1.5 (images)', 'Q4_0', 512, 1.57, 4.1, 'sd'),
    engine: 'stablediffusion',
    kind: 'image',
    disk_gb: 1.57,
  },
];

function m(
  id: string,
  display_name: string,
  quantization: string,
  context_length: number,
  approx_size_gb: number,
  min_vram_gb: number,
  family: string,
  supports_vision = false,
): ModelInfo {
  return {
    id,
    display_name,
    engine: 'llamacpp',
    quantization,
    context_length,
    params: {},
    enabled: true,
    family,
    approx_size_gb,
    min_vram_gb,
    supports_vision,
    kind: 'chat',
    disk_gb: approx_size_gb,
  };
}

const modelById = (id: string) => MODELS.find((x) => x.id === id);

// ── State ──────────────────────────────────────────────────────────────────

const USER: AuthUser = {
  id: 'demo-user',
  email: 'visitor@nexalm.demo',
  displayName: 'Demo visitor',
  avatarUrl: null,
  role: 'admin',
};

let authed = true;
let defaults: ChatSettings = {
  model: 'qwen3-30b-a3b',
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 2048,
  contextLength: 16384,
  useGlobalContext: false,
  autoStartWorker: false,
  idleShutdownMinutes: 30,
  webSearch: false,
};

interface DemoWorker {
  info: WorkerInfo;
  /** Most recently used last. */
  loaded: string[];
}

const now = () => new Date().toISOString();
const ago = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString();

const workers: DemoWorker[] = [
  {
    loaded: ['gemma-3-4b', 'qwen3-30b-a3b'],
    info: worker('kaggle-t4x2', 'kaggle', 2),
  },
  {
    loaded: ['qwen3-30b-a3b-q3'],
    info: worker('colab-t4', 'colab', 1),
  },
];

function worker(name: string, platform: 'kaggle' | 'colab', gpus: number): WorkerInfo {
  return {
    id: `demo-${name}`,
    name,
    status: 'WORKER_READY',
    engine: 'llamacpp',
    model: '',
    gpus: Array.from({ length: gpus }, (_, index) => ({ index, name: 'Tesla T4', totalMb: 15360 })),
    gpuCount: gpus,
    vramTotalMb: 15360 * gpus,
    vramFreeMb: null,
    idleSeconds: 0,
    maxContext: 16384,
    version: 'demo',
    activeJobs: 0,
    lastHeartbeatAt: now(),
    platform,
    loaded: [],
    loading: null,
  };
}

const conversations: Conversation[] = [];
const messages = new Map<string, Message[]>();
let seq = 0;
const uid = (p: string) => `${p}-${Date.now().toString(36)}-${(seq++).toString(36)}`;

/** Seed three finished chats so the first screen is not empty. */
function seed(): string {
  const seeds: Array<{ title: string; prompt: string; reply: Canned; model: string; minutes: number }> = [
    { title: 'How a GPU behind NAT gets work', prompt: CANNED.nat.prompt, reply: CANNED.nat, model: 'qwen3-30b-a3b', minutes: 4 },
    { title: 'Stream tokens from llama-server', prompt: CANNED.code.prompt, reply: CANNED.code, model: 'qwen3-30b-a3b', minutes: 60 * 26 },
    { title: 'MoE vs dense on two T4s', prompt: CANNED.moe.prompt, reply: CANNED.moe, model: 'qwen3-30b-a3b-q3', minutes: 60 * 50 },
  ];
  for (const s of seeds.reverse()) {
    const c = newConversation(s.title, s.model, ago(s.minutes));
    const ttft = 470 + Math.round(Math.random() * 60);
    const tokens = pieces(s.reply.text).length + (s.reply.reasoning ? pieces(s.reply.reasoning).length : 0);
    const tps = s.model === 'qwen3-30b-a3b-q3' ? 62.9 : 65.5;
    const genMs = Math.round((tokens / tps) * 1000);
    messages.set(c.id, [
      msg(c.id, 1, 'user', s.prompt, ago(s.minutes)),
      {
        ...msg(c.id, 2, 'assistant', s.reply.text, ago(s.minutes)),
        reasoning: s.reply.reasoning ?? null,
        model: s.model,
        worker_name: s.model === 'qwen3-30b-a3b-q3' ? 'colab-t4' : 'kaggle-t4x2',
        prompt_tokens: estimateTokens(s.prompt) + 24,
        completion_tokens: tokens,
        total_tokens: tokens + estimateTokens(s.prompt) + 24,
        ttft_ms: ttft,
        gen_ms: genMs,
        total_ms: ttft + genMs,
        tokens_per_sec: tps,
        stop_reason: 'stop',
      },
    ]);
  }
  return conversations[0].id;
}

function newConversation(title: string, model?: string, at = now()): Conversation {
  const c: Conversation = {
    id: uid('conv'),
    title,
    created_at: at,
    updated_at: at,
    archived_at: null,
    settings: model ? { model } : {},
  };
  conversations.unshift(c);
  messages.set(c.id, []);
  return c;
}

function msg(conversationId: string, n: number, role: Message['role'], content: string, at = now()): Message {
  return {
    id: uid(`${conversationId}-m`),
    seq: n,
    role,
    content,
    status: 'complete',
    error: null,
    created_at: at,
  };
}

const estimateTokens = (text: string) => Math.max(1, Math.round(text.length / 3.8));

function decorate(c: Conversation): Conversation {
  const list = messages.get(c.id) ?? [];
  const runs = list.filter((x) => x.role === 'assistant' && x.tokens_per_sec);
  return {
    ...c,
    message_count: list.length,
    total_tokens: list.reduce((a, x) => a + (x.total_tokens ?? 0), 0),
    output_tokens: list.reduce((a, x) => a + (x.completion_tokens ?? 0), 0),
    avg_tokens_per_sec: runs.length
      ? runs.reduce((a, x) => a + (x.tokens_per_sec ?? 0), 0) / runs.length
      : null,
  };
}

function metricsFor(id: string) {
  const runs = (messages.get(id) ?? []).filter((x) => x.role === 'assistant' && x.tokens_per_sec);
  const sum = (f: (x: Message) => number | null | undefined) => runs.reduce((a, x) => a + (f(x) ?? 0), 0);
  return {
    runs: runs.length,
    input_tokens: sum((x) => x.prompt_tokens),
    output_tokens: sum((x) => x.completion_tokens),
    total_tokens: sum((x) => x.total_tokens),
    total_gen_ms: sum((x) => x.gen_ms),
    avg_tokens_per_sec: runs.length ? sum((x) => x.tokens_per_sec) / runs.length : null,
    avg_ttft_ms: runs.length ? sum((x) => x.ttft_ms) / runs.length : null,
    peak_tokens_per_sec: runs.length ? Math.max(...runs.map((x) => x.tokens_per_sec ?? 0)) : null,
  };
}

// ── Workers and models ─────────────────────────────────────────────────────

function pool(): WorkerPool {
  const infos = workers.map((w) => {
    const usedGb = w.loaded.reduce((a, id) => a + (modelById(id)?.min_vram_gb ?? 0), 0);
    return {
      ...w.info,
      loaded: [...w.loaded],
      lastHeartbeatAt: now(),
      model: w.loaded[w.loaded.length - 1] ?? '',
      vramFreeMb: Math.max(0, Math.round(w.info.vramTotalMb - usedGb * 1024)),
      freeVramGb: Math.max(0, w.info.vramTotalMb / 1024 - usedGb),
      status: w.info.activeJobs > 0 ? ('WORKER_BUSY' as const) : ('WORKER_READY' as const),
    };
  });
  const loaded = [...new Set(workers.flatMap((w) => w.loaded))];
  return {
    available: true,
    connected: workers.length,
    workers: infos,
    platforms: workers.map((w) => ({
      platform: w.info.platform ?? 'other',
      connected: 1,
      workers: [w.info.id],
      loaded: [...w.loaded],
      startable: w.info.platform === 'kaggle',
    })),
    loadedModels: loaded,
    largestVramGb: 30,
    recent: [],
    control: { kaggle: true },
  };
}

function modelsView() {
  const loaded = new Set(workers.flatMap((w) => w.loaded));
  return {
    models: MODELS.map((x) => ({
      ...x,
      loaded: loaded.has(x.id),
      fits: (x.min_vram_gb ?? 0) <= 30,
      availability: loaded.has(x.id) ? ('ready' as const) : (x.min_vram_gb ?? 0) <= 30 ? ('loadable' as const) : ('too_large' as const),
    })),
    largestVramGb: 30,
  };
}

/** The worker that holds a model, preferring the bigger host. */
function workerFor(modelId: string): DemoWorker | undefined {
  return workers.find((w) => w.loaded.includes(modelId));
}

function touch(w: DemoWorker, modelId: string) {
  w.loaded = [...w.loaded.filter((x) => x !== modelId), modelId];
}

function loadModel(modelId: string, workerId?: string) {
  const model = modelById(modelId);
  const w = workers.find((x) => x.info.id === workerId) ?? workers[0];
  if (!model) return { ok: false, message: `Unknown model ${modelId}.` };
  const totalGb = w.info.vramTotalMb / 1024;
  if ((model.min_vram_gb ?? 0) > totalGb) {
    return { ok: false, message: `${model.display_name} needs about ${model.min_vram_gb} GB of VRAM; ${w.info.name} has ${totalGb.toFixed(0)} GB.` };
  }
  const evicted = makeRoom(w, modelId);
  w.info.loading = modelId;
  setTimeout(() => {
    touch(w, modelId);
    w.info.loading = null;
  }, 1800);
  const note = evicted.length ? ` Evicted ${evicted.join(', ')} to make room (least recently used).` : '';
  return {
    ok: true,
    workerId: w.info.id,
    message: `Loading ${model.display_name} on ${w.info.name}.${note} (Demo: this takes a moment here, minutes on a real GPU.)`,
  };
}

/** LRU, as the real ModelManager does: evict the least recently used idle model until it fits. */
function makeRoom(w: DemoWorker, modelId: string): string[] {
  const need = modelById(modelId)?.min_vram_gb ?? 0;
  const totalGb = w.info.vramTotalMb / 1024;
  const evicted: string[] = [];
  const used = () => w.loaded.reduce((a, id) => a + (modelById(id)?.min_vram_gb ?? 0), 0);
  while (used() + need > totalGb && w.loaded.length) {
    evicted.push(w.loaded.shift() as string);
  }
  return evicted;
}

function unloadModel(modelId: string, workerId?: string) {
  const w = workers.find((x) => x.info.id === workerId) ?? workerFor(modelId);
  if (!w || !w.loaded.includes(modelId)) return { ok: false, message: `${modelId} is not loaded.` };
  w.loaded = w.loaded.filter((x) => x !== modelId);
  return { ok: true, message: `Unloaded ${modelId} from ${w.info.name}.` };
}

/** Simulated per-GPU telemetry for the usage chart: the last 15 minutes. */
function samples(minutes: number): WorkerSample[] {
  const out: WorkerSample[] = [];
  const end = Date.now();
  for (const w of workers) {
    for (const g of w.info.gpus) {
      for (let t = minutes * 60; t >= 0; t -= 20) {
        const at = end - t * 1000;
        const wave = Math.max(0, Math.sin(at / 47_000 + g.index) * 0.5 + Math.sin(at / 13_000) * 0.25);
        const util = Math.round(Math.min(97, wave * 120));
        out.push({
          worker_id: w.info.id,
          gpu_index: g.index,
          sampled_at: new Date(at).toISOString(),
          util_pct: util,
          mem_used_mb: Math.round(g.totalMb * (w.info.platform === 'kaggle' ? 0.78 : 0.82)),
          mem_total_mb: g.totalMb,
          temp_c: 44 + Math.round(util / 5),
          power_w: 28 + Math.round(util * 0.4),
          active_jobs: util > 40 ? 1 : 0,
        });
      }
    }
  }
  return out;
}

// ── Jobs and the token stream ──────────────────────────────────────────────

interface Job {
  id: string;
  conversationId: string;
  messageId: string;
  model: string;
  worker: DemoWorker;
  reply: Canned;
  promptTokens: number;
  cancelled: boolean;
  started: boolean;
}

const jobs = new Map<string, Job>();

function startJob(conversationId: string, assistant: Message, model: string, reply: Canned, promptTokens: number): string {
  let w = workerFor(model);
  if (!w) {
    // Not resident anywhere: load it on the bigger host first, as the router would.
    w = workers[0];
    makeRoom(w, model);
  }
  touch(w, model);
  const job: Job = {
    id: uid('job'),
    conversationId,
    messageId: assistant.id,
    model,
    worker: w,
    reply,
    promptTokens,
    cancelled: false,
    started: false,
  };
  jobs.set(job.id, job);
  assistant.job_id = job.id;
  assistant.model = model;
  assistant.worker_name = w.info.name;
  return job.id;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Split into small pieces the way a tokenizer roughly would. */
function pieces(text: string): string[] {
  return text.match(/\s*\S{1,6}/g) ?? [text];
}

async function run(job: Job, es: DemoEventSource) {
  const assistant = (messages.get(job.conversationId) ?? []).find((x) => x.id === job.messageId);
  if (!assistant) return;
  const isImage = modelById(job.model)?.kind === 'image';
  const tps = job.worker.info.platform === 'colab' ? 62.9 : 65.5;
  const started = performance.now();
  job.worker.info.activeJobs += 1;

  es.emit('status', { workerName: job.worker.info.name });

  if (isImage) {
    for (let i = 0; i <= 4 && !job.cancelled; i++) {
      es.emit('progress', { status: i === 0 ? 'queued' : 'generating', elapsedMs: Math.round(performance.now() - started) });
      await sleep(900);
    }
  } else {
    await sleep(470 + Math.random() * 60);
  }
  const ttft = Math.round(performance.now() - started);
  es.emit('first_token', { ttftMs: ttft });

  let reasoning = '';
  if (job.reply.reasoning && !isImage) {
    let i = 0;
    for (const p of pieces(job.reply.reasoning)) {
      if (job.cancelled) break;
      reasoning += p;
      es.emit('reasoning', { text: p, index: i++ });
      await sleep(1000 / tps);
    }
  }

  let text = '';
  let i = 0;
  for (const p of pieces(job.reply.text)) {
    if (job.cancelled) break;
    text += p;
    es.emit('delta', { text: p, index: i++ });
    await sleep(1000 / tps);
  }
  const emitted = i + (reasoning ? pieces(reasoning).length : 0);
  // Report the simulated rate itself rather than what the browser's timers
  // managed, so the numbers stay consistent with each other.
  const completion = emitted;
  const rate = Math.round((tps + (Math.random() - 0.5)) * 10) / 10;
  const genMs = Math.round((completion / rate) * 1000);

  Object.assign(assistant, {
    content: text,
    reasoning: reasoning || null,
    status: job.cancelled ? 'cancelled' : 'complete',
    prompt_tokens: job.promptTokens,
    completion_tokens: completion,
    total_tokens: job.promptTokens + completion,
    ttft_ms: ttft,
    gen_ms: genMs,
    total_ms: ttft + genMs,
    tokens_per_sec: isImage ? null : rate,
    stop_reason: job.cancelled ? 'cancelled' : 'stop',
  } satisfies Partial<Message>);
  const c = conversations.find((x) => x.id === job.conversationId);
  if (c) c.updated_at = now();
  job.worker.info.activeJobs = Math.max(0, job.worker.info.activeJobs - 1);

  es.emit('done', {
    ttftMs: ttft,
    genMs,
    totalMs: ttft + genMs,
    promptTokens: job.promptTokens,
    completionTokens: completion,
    totalTokens: job.promptTokens + completion,
    tokensPerSec: assistant.tokens_per_sec,
    workerName: job.worker.info.name,
    model: job.model,
    stopReason: assistant.stop_reason,
  });
}

class DemoEventSource extends EventTarget {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSED = 2;
  readyState = 0;
  readonly withCredentials = false;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;

  constructor(public readonly url: string) {
    super();
    setTimeout(() => this.attach(), 0);
  }

  private attach() {
    if (this.readyState === 2) return;
    this.readyState = 1;
    const id = /\/api\/jobs\/([^/]+)\/stream/.exec(this.url)?.[1];
    const job = id ? jobs.get(id) : undefined;
    if (!job || job.started) {
      this.emit('error', { code: 'NOT_FOUND', message: 'This demo job has already finished.' });
      return;
    }
    job.started = true;
    void run(job, this);
  }

  emit(type: string, data: unknown) {
    if (this.readyState === 2) return;
    this.dispatchEvent(new MessageEvent(type, { data: JSON.stringify(data) }));
  }

  close() {
    this.readyState = 2;
  }
}

// ── Routes ─────────────────────────────────────────────────────────────────

type Body = Record<string, unknown> | undefined;
type Result = { status: number; body?: unknown };

const ok = (body: unknown): Result => ({ status: 200, body });
const notFound = (): Result => ({ status: 404, body: { error: 'not_found', message: 'Not found in the demo.' } });

function route(method: string, path: string, query: URLSearchParams, body: Body): Result {
  const parts = path.replace(/^\/api\//, '').split('/');
  const [a, b, c] = parts;

  if (a === 'health') return ok({ ok: true, demo: true });

  if (a === 'auth') {
    if (b === 'status') {
      return ok({
        authenticated: authed,
        user: authed ? USER : null,
        googleEnabled: false,
        signupEnabled: true,
        requiresAccessToken: false,
        unlocked: true,
        setupRequired: false,
        emailVerification: false,
      });
    }
    if (b === 'me') return authed ? ok({ user: USER }) : { status: 401, body: { error: 'unauthorized', message: 'Sign in first.' } };
    if (b === 'login' || b === 'signup' || b === 'verify-email') {
      authed = true;
      return ok({ user: USER, adoptedConversations: 0 });
    }
    if (b === 'logout' || b === 'logout-all') {
      authed = false;
      return ok({ ok: true });
    }
    return ok({ ok: true });
  }

  if (a === 'conversations') {
    if (!b) {
      if (method === 'POST') return ok({ conversation: decorate(newConversation('New chat')) });
      return ok({ conversations: conversations.filter((x) => !x.archived_at).map(decorate) });
    }
    const conv = conversations.find((x) => x.id === b);
    if (!conv) return notFound();
    const list = messages.get(conv.id) ?? [];

    if (!c) {
      if (method === 'GET') return ok({ conversation: decorate(conv), messages: list });
      if (method === 'DELETE') {
        conversations.splice(conversations.indexOf(conv), 1);
        messages.delete(conv.id);
        return ok({ ok: true });
      }
      if (method === 'PATCH') {
        if (typeof body?.title === 'string') conv.title = body.title;
        if (body?.settings) conv.settings = { ...conv.settings, ...(body.settings as ChatSettings) };
        if (body?.archived) conv.archived_at = now();
        conv.updated_at = now();
        return ok({ conversation: decorate(conv) });
      }
    }
    if (c === 'metrics') return ok({ metrics: metricsFor(conv.id) });

    if (c === 'messages' && method === 'POST') {
      const content = String(body?.content ?? '').trim();
      const model = conv.settings.model ?? defaults.model ?? 'qwen3-30b-a3b';
      const kind = modelById(model)?.kind ?? 'chat';
      const user = msg(conv.id, list.length + 1, 'user', content);
      const assistant: Message = { ...msg(conv.id, list.length + 2, 'assistant', ''), status: 'streaming' };
      list.push(user, assistant);
      if (conv.title === 'New chat') conv.title = content.length > 48 ? `${content.slice(0, 45)}…` : content;
      conv.updated_at = now();
      const reply = pickReply(content, modelById(model)?.display_name ?? model, kind);
      const jobId = startJob(conv.id, assistant, model, reply, estimateTokens(content) + 24);
      return ok({ userMessage: user, assistantMessage: assistant, jobId, model });
    }

    if (c === 'retry' && method === 'POST') {
      const last = [...list].reverse().find((x) => x.role === 'assistant');
      const prompt = [...list].reverse().find((x) => x.role === 'user');
      if (!last || !prompt) return notFound();
      const model = conv.settings.model ?? defaults.model ?? 'qwen3-30b-a3b';
      Object.assign(last, { content: '', reasoning: null, status: 'streaming', error: null });
      const reply = pickReply(prompt.content, modelById(model)?.display_name ?? model, modelById(model)?.kind ?? 'chat');
      const jobId = startJob(conv.id, last, model, reply, estimateTokens(prompt.content) + 24);
      return ok({ assistantMessageId: last.id, jobId });
    }
    return notFound();
  }

  if (a === 'jobs' && c === 'cancel') {
    const job = jobs.get(b);
    if (job) job.cancelled = true;
    return ok({ ok: true });
  }

  if (a === 'settings') {
    if (method === 'PUT') {
      defaults = { ...defaults, ...(body as ChatSettings) };
      return ok({ defaults });
    }
    return ok({ defaults, models: MODELS });
  }

  if (a === 'models') return ok(modelsView());

  if (a === 'workers') {
    if (!b) return ok(pool());
    if (b === 'samples') {
      const minutes = Number(query.get('minutes') ?? 15);
      return ok({ windowMinutes: minutes, samples: samples(minutes) });
    }
    if (b === 'session') {
      return ok({ kaggle: { configured: true, status: 'running', failureMessage: null, sessionId: null } });
    }
    if (b === 'models' && c === 'load') return ok(loadModel(String(body?.modelId), body?.workerId as string | undefined));
    if (b === 'models' && c === 'unload') return ok(unloadModel(String(body?.modelId), body?.workerId as string | undefined));
    if (b === 'start' || b === 'stop') {
      return ok({ ok: false, stopped: [], message: 'This is the demo build, so there is no real Kaggle session to start or stop.' });
    }
  }

  if (a === 'users') {
    if (!b) {
      const me: AdminUser = {
        id: USER.id,
        email: USER.email,
        displayName: USER.displayName,
        avatarUrl: null,
        role: 'admin',
        googleLinked: false,
        createdAt: ago(60 * 24 * 3),
        lastLoginAt: now(),
        disabledAt: null,
        conversationCount: conversations.length,
        activeSessions: 1,
      };
      return ok({ users: [me] });
    }
    return { status: 400, body: { error: 'demo', message: 'User management is read-only in the demo.' } };
  }

  return notFound();
}

// ── Install ────────────────────────────────────────────────────────────────

export function installDemo() {
  const firstId = seed();
  try {
    localStorage.setItem('llm-active-conversation', firstId);
  } catch {
    /* storage blocked: the app simply opens on an empty chat */
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const url = new URL(raw, window.location.href);
    if (url.origin !== window.location.origin || !url.pathname.startsWith('/api/')) {
      return realFetch(input, init);
    }
    await sleep(60 + Math.random() * 120);
    const method = (init?.method ?? 'GET').toUpperCase();
    let body: Body;
    try {
      body = init?.body ? (JSON.parse(String(init.body)) as Body) : undefined;
    } catch {
      body = undefined;
    }
    const res = route(method, url.pathname, url.searchParams, body);
    return new Response(res.status === 204 ? null : JSON.stringify(res.body ?? {}), {
      status: res.status,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  (window as unknown as { EventSource: unknown }).EventSource = DemoEventSource;
}
