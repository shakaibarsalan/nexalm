/** Central runtime configuration, read once from the environment. */

function str(name: string, fallback?: string): string {
  const v = process.env[name] ?? fallback;
  if (v === undefined || v === '') {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return v;
}

function num(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  if (Number.isNaN(n)) throw new Error(`Environment variable ${name} must be a number`);
  return n;
}

function bool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  return raw === 'true' || raw === '1';
}

const google = {
  clientId: process.env.GOOGLE_CLIENT_ID ?? '',
  clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? '',
  redirectUri: process.env.GOOGLE_REDIRECT_URI ?? '',
};

export const config = {
  env: process.env.NODE_ENV ?? 'development',
  port: num('PORT', 8080),
  host: '0.0.0.0',
  logLevel: process.env.LOG_LEVEL ?? 'info',
  publicOrigin: process.env.PUBLIC_ORIGIN ?? 'http://localhost:8081',

  databaseUrl: str('DATABASE_URL'),
  dbPoolMax: num('DB_POOL_MAX', 10),

  /**
   * Legacy shared passphrase. Now that accounts exist it is an optional second
   * gate in front of the login page, off unless REQUIRE_ACCESS_TOKEN=true.
   */
  appAccessToken: process.env.APP_ACCESS_TOKEN ?? '',
  requireAccessToken: bool('REQUIRE_ACCESS_TOKEN', false),
  cookieSecret: str('COOKIE_SECRET'),

  /** Open by default; the first account created becomes the admin. */
  signupEnabled: bool('SIGNUP_ENABLED', true),

  /**
   * Outbound mail. Verification is only demanded when this is configured:
   * requiring a code the server cannot send would lock everyone out.
   */
  smtp: {
    host: process.env.SMTP_HOST ?? '',
    port: Number(process.env.SMTP_PORT ?? 587),
    user: process.env.SMTP_USER ?? '',
    pass: process.env.SMTP_PASS ?? '',
    from: process.env.SMTP_FROM || process.env.SMTP_USER || '',
  },

  /**
   * Addresses on these domains skip verification.
   *
   * Not a convenience switch: `.local` is reserved and cannot receive mail at
   * all, so demanding a code from such an address would make sign-up impossible
   * rather than merely slower. It exists so the end-to-end test harness can
   * create the throwaway accounts it needs. An account made this way is an
   * ordinary member with no extra privilege.
   */
  /**
   * Web search. Brave has a free tier and is the only real web search here;
   * Wikipedia needs no key and is the fallback, so the feature degrades to
   * something useful rather than to nothing.
   */
  search: {
    // Apify's RAG Web Browser: real Google results with the pages fetched.
    // Best quality, ~13s per search, and each call spends Apify credit.
    apifyToken: process.env.APIFY_TOKEN ?? '',
    braveKey: process.env.BRAVE_API_KEY ?? '',
  },

  otpExemptDomains: (process.env.OTP_EXEMPT_DOMAINS ?? 'nexalm.local')
    .split(',')
    .map((d) => d.trim().toLowerCase())
    .filter(Boolean),
  sessionTtlDays: num('SESSION_TTL_DAYS', 30),

  /**
   * Google sign-in is opt-in: with any of these unset the routes stay
   * registered but refuse, and /api/auth/status reports it as disabled so the
   * UI can hide the button.
   */
  google,
  googleEnabled: Boolean(google.clientId && google.clientSecret && google.redirectUri),

  /** Bearer token every inference worker must present. */
  workerToken: str('WORKER_TOKEN'),

  heartbeatIntervalMs: num('WORKER_HEARTBEAT_INTERVAL_MS', 10_000),
  leaseTimeoutMs: num('WORKER_LEASE_TIMEOUT_MS', 30_000),
  sweepIntervalMs: num('WORKER_SWEEP_INTERVAL_MS', 5_000),
  /**
   * Jobs a single worker may hold at once. A busy worker must still be
   * selectable, otherwise a single-worker deployment reports "no worker"
   * for the whole duration of every generation. The engine serialises
   * internally (llama.cpp runs with --parallel 1), so this is a queue depth.
   */
  workerMaxQueue: num('WORKER_MAX_QUEUE', 2),

  defaults: {
    model: process.env.DEFAULT_MODEL ?? 'qwen3-30b-a3b',
    temperature: num('DEFAULT_TEMPERATURE', 0.7),
    topP: num('DEFAULT_TOP_P', 0.9),
    maxTokens: num('DEFAULT_MAX_TOKENS', 2048),
    contextLength: num('DEFAULT_CONTEXT_LENGTH', 16384),
  },

  jobTimeoutMs: num('JOB_TIMEOUT_MS', 600_000),

  /** Prebuilt worker artifacts (llama.cpp binaries) served to remote GPU hosts. */
  artifactsDir: process.env.ARTIFACTS_DIR ?? '/app/artifacts',

  /**
   * Generated images. Separate from artifactsDir because that mount is
   * read-only by design -- it holds build outputs the server only ever serves,
   * whereas this one is written to at runtime by the upload endpoint.
   */
  imagesDir: process.env.IMAGES_DIR ?? '/app/images',

  /**
   * Optional Kaggle control. When a token is present the UI can start and stop
   * the GPU session; without one those controls simply stay hidden.
   */
  kaggle: {
    apiToken: process.env.KAGGLE_API_TOKEN ?? '',
    kernelSlug: process.env.KAGGLE_KERNEL_SLUG ?? 'shakaibkaggle/local-llm-gpu-worker',
    kernelTitle: process.env.KAGGLE_KERNEL_TITLE ?? 'Local LLM GPU Worker',
    datasetSources: (process.env.KAGGLE_DATASET_SOURCES ?? 'shakaibkaggle/local-llm-worker-code')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
    /**
     * Empty by default on purpose: create-session inherits the notebook's own
     * accelerator, which is the only way to get the 2x T4 (32 GB) pair. Setting
     * this pins a single GPU instead.
     */
    machineShape: process.env.KAGGLE_MACHINE_SHAPE ?? '',
  },
} as const;

export const isProd = config.env === 'production';

// Fail at boot rather than locking everyone out with an unsatisfiable gate.
if (config.requireAccessToken && !config.appAccessToken) {
  throw new Error('REQUIRE_ACCESS_TOKEN=true but APP_ACCESS_TOKEN is not set');
}
