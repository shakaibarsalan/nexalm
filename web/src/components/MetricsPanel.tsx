import { useState, type ReactNode } from 'react';
import type {
  ConversationMetrics,
  LiveMetrics,
  SessionStatus,
  WorkerPool,
  WorkerSample,
} from '../lib/types';
import { ChipIcon, CloseIcon } from './Icons';
import { UsageChart } from './UsageChart';

function fmtMs(ms?: number | null): string {
  if (ms == null) return '—';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function fmtInt(n?: number | null): string {
  return n == null ? '—' : Number(n).toLocaleString();
}

function Row({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-xs" style={{ color: 'rgb(var(--text-muted))' }}>
        {label}
      </span>
      <span className={`text-xs ${mono ? 'font-mono' : ''}`} style={{ color: 'rgb(var(--text))' }}>
        {value}
      </span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b px-4 py-3" style={{ borderColor: 'rgb(var(--border))' }}>
      <h3
        className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider"
        style={{ color: 'rgb(var(--text-faint))' }}
      >
        {title}
      </h3>
      {children}
    </section>
  );
}

const STATUS_COLOR: Record<string, string> = {
  WORKER_READY: 'var(--success)',
  WORKER_BUSY: 'var(--warn)',
  WORKER_STARTING: 'var(--warn)',
  WORKER_ERROR: 'var(--danger)',
  WORKER_OFFLINE: 'var(--text-faint)',
};

interface Props {
  open: boolean;
  onClose: () => void;
  live: LiveMetrics | null;
  isStreaming: boolean;
  conversationMetrics: ConversationMetrics | null;
  pool: WorkerPool | null;
  samples: WorkerSample[];
  sampleWindow: number;
  session: SessionStatus | null;
  /**
   * The model picker, shown as a tab in this rail.
   *
   * It was briefly its own column, which put four panels side by side and left
   * each of them too narrow to read. One rail with two tabs gives both the room
   * they need and hands a whole column back to the conversation.
   */
  modelsSlot?: ReactNode;
  /** The conversation's model, so this panel reports the worker that serves it. */
  selectedModel?: string;
  workerNotice: string | null;
}

function fmtHours(seconds: number): string {
  const h = seconds / 3600;
  return h >= 1 ? `${h.toFixed(1)} h` : `${Math.round(seconds / 60)} min`;
}

function fmtDuration(seconds?: number | null): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

const PLATFORM_NAME: Record<string, string> = {
  kaggle: 'Kaggle',
  colab: 'Google Colab',
  other: 'Self-hosted',
};

const fmtGb = (mb?: number | null) => (mb ? `${(mb / 1024).toFixed(0)} GB` : '—');

export function MetricsPanel({
  open,
  onClose,
  live,
  isStreaming,
  conversationMetrics,
  pool,
  samples,
  sampleWindow,
  session,
  workerNotice,
  modelsSlot,
  selectedModel,
}: Props) {
  const [tab, setTab] = useState<'models' | 'inference'>('models');

  if (!open) return null;

  const cm = conversationMetrics;
  const hasWorker = (pool?.connected ?? 0) > 0;

  // The worker that will answer *this conversation*: the one holding the model
  // it is set to. Falling back to "whichever worker is first" is what let this
  // panel name Colab while the Models tab showed a Kaggle model as chosen --
  // two true statements that read as a contradiction.
  const holders = (pool?.workers ?? []).filter((w) =>
    selectedModel ? (w.loaded ?? []).includes(selectedModel) : true,
  );
  const liveWorker =
    holders.find((w) => w.status === 'WORKER_BUSY') ??
    holders.find((w) => w.status === 'WORKER_READY') ??
    holders[0];
  const canControl = pool?.control?.kaggle ?? false;

  // The worker only registers once the model is resident, so "session running
  // but nothing connected" is the loading window — several minutes of download
  // and load with no other outward sign. Name it rather than showing "offline".
  // Kaggle meters wall-clock, not GPU work, so a session left running drains
  // the week even while idle. Surfacing the remaining budget is what makes the
  // stop control something people actually use.
  const used = session?.quota?.gpuSecondsUsed ?? 0;
  const limit = session?.quota?.gpuSecondsLimit ?? 0;
  const quota = {
    used,
    limit,
    remaining: Math.max(0, limit - used),
    pct: limit > 0 ? Math.min(100, (used / limit) * 100) : 0,
  };
  // Thresholds are absolute hours, not fractions: the weekly grant is 30 h, so
  // a percentage would call 20% "fine" when it is only six hours.
  const quotaColor =
    quota.remaining < 60 * 60
      ? 'rgb(var(--danger))'
      : quota.remaining < 6 * 60 * 60
        ? 'rgb(var(--warn))'
        : 'rgb(var(--success))';

  const sessionRunning = String(session?.kaggle?.status ?? '').toLowerCase() === 'running';
  // Resolved against liveWorker -- the one holding this conversation's model --
  // rather than whichever worker happens to be first in the pool. Picking
  // arbitrarily is what made this section name a different model from the one
  // answering, once two hosts were connected.
  const model: { label: string; detail: string; tone: string } = liveWorker
    ? {
        label: liveWorker.status === 'WORKER_BUSY' ? 'Generating' : 'Ready',
        tone: 'var(--success)',
        detail: `${liveWorker.engine} · ${liveWorker.gpuCount || 'CPU'}${
          liveWorker.gpuCount ? '× GPU' : ''
        }`,
      }
    : sessionRunning
      ? {
          label: 'Loading model',
          detail: 'Session up — downloading weights and loading onto the GPU.',
          tone: 'var(--warn)',
        }
      : { label: 'Offline', detail: 'No GPU session is running.', tone: 'var(--text-faint)' };

  return (
    <aside
      className="fixed inset-y-0 right-0 z-40 w-96 overflow-y-auto border-l lg:static lg:z-auto"
      style={{ backgroundColor: 'rgb(var(--bg-elevated))', borderColor: 'rgb(var(--border))' }}
    >
      <div
        className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b px-3 py-2.5"
        style={{ backgroundColor: 'rgb(var(--bg-elevated))', borderColor: 'rgb(var(--border))' }}
      >
        <div
          className="flex gap-0.5 rounded-lg p-0.5"
          style={{ backgroundColor: 'rgb(var(--bg-subtle))' }}
          role="tablist"
        >
          {(['models', 'inference'] as const).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className="rounded-md px-3 py-1.5 text-xs font-medium capitalize transition-colors"
              style={
                tab === t
                  ? { backgroundColor: 'rgb(var(--bg-elevated))', color: 'rgb(var(--text))' }
                  : { color: 'rgb(var(--text-muted))' }
              }
            >
              {t}
            </button>
          ))}
        </div>
        <button onClick={onClose} className="btn btn-ghost p-1.5" aria-label="Close panel">
          <CloseIcon />
        </button>
      </div>

      {tab === 'models' && modelsSlot}
      <div hidden={tab !== 'inference'}>

      {(isStreaming || live) && (
        <Section title={isStreaming ? 'Current generation' : 'Last generation'}>
          <Row label="Tokens / sec" value={live?.tokensPerSec != null ? Number(live.tokensPerSec).toFixed(2) : '—'} />
          <Row label="Time to first token" value={fmtMs(live?.ttftMs)} />
          <Row label="Generation time" value={fmtMs(live?.genMs)} />
          <Row label="Total time" value={fmtMs(live?.totalMs)} />
          <Row label="Input tokens" value={fmtInt(live?.promptTokens)} />
          <Row label="Output tokens" value={fmtInt(live?.completionTokens)} />
          <Row label="Total tokens" value={fmtInt(live?.totalTokens)} />
          {live?.stopReason && <Row label="Stop reason" value={live.stopReason} />}
          {isStreaming && (
            <div className="mt-2 flex items-center gap-2 text-xs" style={{ color: 'rgb(var(--warn))' }}>
              <span className="h-1.5 w-1.5 animate-blink rounded-full bg-current" />
              Generating
            </div>
          )}
        </Section>
      )}

      <Section title="This conversation">
        <Row label="Generations" value={fmtInt(cm?.runs)} />
        <Row label="Input tokens" value={fmtInt(cm?.input_tokens)} />
        <Row label="Output tokens" value={fmtInt(cm?.output_tokens)} />
        <Row label="Total tokens" value={fmtInt(cm?.total_tokens)} />
        <Row label="Total generation" value={fmtMs(cm?.total_gen_ms)} />
        <Row label="Avg tokens / sec" value={cm?.avg_tokens_per_sec != null ? Number(cm.avg_tokens_per_sec).toFixed(2) : '—'} />
        <Row label="Peak tokens / sec" value={cm?.peak_tokens_per_sec != null ? Number(cm.peak_tokens_per_sec).toFixed(2) : '—'} />
        <Row label="Avg TTFT" value={fmtMs(cm?.avg_ttft_ms)} />
      </Section>

      <Section title="Answering this chat">
        {liveWorker ? (
          <>
            <div className="mb-1.5 flex items-center gap-2">
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: `rgb(${model.tone})` }}
              />
              <span className="text-sm font-medium">{model.label}</span>
            </div>
            <Row label="Host" value={PLATFORM_NAME[liveWorker.platform ?? 'other']} />
            <Row label="Worker" value={liveWorker.name} />
            <Row label="Model" value={selectedModel ?? liveWorker.model} mono />
            <Row label="Engine" value={model.detail} />
            <Row
              label="GPUs"
              value={`${liveWorker.gpuCount} × ${fmtGb(
                (liveWorker.vramTotalMb ?? 0) / Math.max(1, liveWorker.gpuCount),
              )}`}
            />
          </>
        ) : (
          <>
            <div className="mb-1.5 flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  model.label === 'Loading model' ? 'animate-blink' : ''
                }`}
                style={{ backgroundColor: `rgb(${model.tone})` }}
              />
              <span className="text-sm font-medium">{model.label}</span>
            </div>
            <p className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
              {selectedModel
                ? `No connected worker has ${selectedModel} loaded. Load it from the Models tab, or pick a model that is.`
                : model.detail}
            </p>
          </>
        )}
      </Section>

      <Section title="GPU session">
        {!canControl ? (
          <p className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
            Remote control is not configured on the server.
          </p>
        ) : (
          <>
            {/* Read-only. Starting and stopping live in the Models tab, beside
                the hosts they act on: a control here could only ever mean
                Kaggle, which is precisely what made a hand-started Colab
                worker look like this button's doing. */}
            {session?.kaggle?.status && <Row label="Kaggle session" value={session.kaggle.status} />}

            {quota.limit > 0 && (
              <div className="mt-2">
                <div className="mb-1 flex items-baseline justify-between">
                  <span className="text-xs" style={{ color: 'rgb(var(--text-muted))' }}>
                    Weekly GPU quota
                  </span>
                  <span className="font-mono text-xs" style={{ color: quotaColor }}>
                    {fmtHours(quota.remaining)} left
                  </span>
                </div>
                <div
                  className="h-1.5 w-full overflow-hidden rounded-full"
                  style={{ backgroundColor: 'rgb(var(--bg-subtle))' }}
                  role="progressbar"
                  aria-valuenow={Math.round(quota.pct)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className="h-full rounded-full transition-[width]"
                    style={{ width: `${quota.pct}%`, backgroundColor: quotaColor }}
                  />
                </div>
                <p className="mt-1 text-[10px]" style={{ color: 'rgb(var(--text-faint))' }}>
                  {fmtHours(quota.used)} of {fmtHours(quota.limit)} used
                  {session?.quota?.refreshesAt
                    ? ` · resets ${session.quota.refreshesAt.slice(0, 10)}`
                    : ''}
                </p>
              </div>
            )}
            {workerNotice && (
              <p className="mt-1.5 text-[11px] leading-4" style={{ color: 'rgb(var(--text-muted))' }}>
                {workerNotice}
              </p>
            )}
            {!hasWorker && (
              <p className="mt-1.5 text-[11px] leading-4" style={{ color: 'rgb(var(--text-faint))' }}>
                No GPU is running. Start one from the Models tab.
              </p>
            )}
          </>
        )}
      </Section>

      <Section title="Resource usage">
        <UsageChart samples={samples} windowMinutes={sampleWindow} />
      </Section>

      <Section title="Workers">
        {!pool || pool.workers.length === 0 ? (
          <p className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
            No worker connected.
          </p>
        ) : (
          <ul className="space-y-3">
            {pool.workers.map((w) => (
              <li key={w.id}>
                <div className="flex items-center gap-2">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: `rgb(${STATUS_COLOR[w.status] ?? 'var(--text-faint)'})` }}
                  />
                  <span className="truncate text-xs font-medium">{w.name}</span>
                </div>
                <div className="mt-1 pl-3.5">
                  <Row label="Status" value={w.status.replace('WORKER_', '')} />
                  <Row label="Engine" value={w.engine} />
                  <Row label="Model" value={w.model} />
                  <Row label="Active jobs" value={String(w.activeJobs)} />
                  <Row label="Idle for" value={fmtDuration(w.idleSeconds)} />
                  {w.gpus.map((g) => (
                    <div
                      key={g.index}
                      className="mt-1 flex items-center gap-1.5 font-mono text-[11px]"
                      style={{ color: 'rgb(var(--text-muted))' }}
                    >
                      <ChipIcon className="h-3 w-3 shrink-0" />
                      <span className="truncate">
                        {g.name} · {(g.totalMb / 1024).toFixed(0)} GB
                      </span>
                    </div>
                  ))}
                  {w.gpus.length === 0 && (
                    <div
                      className="mt-1 flex items-center gap-1.5 font-mono text-[11px]"
                      style={{ color: 'rgb(var(--text-muted))' }}
                    >
                      <ChipIcon className="h-3 w-3 shrink-0" />
                      CPU only
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Section>
      </div>
    </aside>
  );
}
