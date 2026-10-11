import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { AuthUser, ChatSettings, ModelInfo, WorkerInfo, WorkerPool } from '../lib/types';
import { UsersPanel } from './UsersPanel';
import { CloseIcon, MoonIcon, SunIcon } from './Icons';

function Slider({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="font-mono text-xs" style={{ color: 'rgb(var(--text-muted))' }}>
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-current"
        style={{ color: 'rgb(var(--accent))' }}
      />
      <p className="mt-1 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
        {hint}
      </p>
    </div>
  );
}

/**
 * Explanation, and the recommended value as a single click.
 *
 * The house pattern: plain sentence, then the mechanism, then what to do. The
 * chip only appears when the current value differs, so a correctly-configured
 * screen stays quiet.
 */
function FieldHint({
  text,
  recommended,
  current,
  onApply,
}: {
  text: string;
  recommended: number;
  current: number;
  onApply: () => void;
}) {
  return (
    <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <p className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
        {text}
      </p>
      {current !== recommended && (
        <button
          type="button"
          onClick={onApply}
          className="rounded px-1.5 py-0.5 text-[11px] font-medium"
          style={{ backgroundColor: 'rgb(var(--accent-soft))', color: 'rgb(var(--accent-text))' }}
        >
          Use {recommended.toLocaleString()}
        </button>
      )}
    </div>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onSaved: () => void;
  me: AuthUser | null;
  /**
   * The connected GPU pool.
   *
   * Without it this screen could only show numbers with no idea whether they
   * were achievable — which is why the two fields that depend most on the
   * hardware were also the only two with no explanation at all.
   */
  pool?: WorkerPool | null;
}

const PLATFORM_LABEL: Record<string, string> = {
  kaggle: 'Kaggle',
  colab: 'Colab',
  other: 'this host',
};

/**
 * What each setting should be for the model and GPU actually in play.
 *
 * Offered, never enforced: the box stays editable and a larger number is still
 * accepted. It exists so that "8,192" is a recommendation with a reason
 * attached rather than a number someone has to already understand.
 */
function recommendationsFor(model: ModelInfo | undefined, worker: WorkerInfo | undefined) {
  // llama.cpp is started with a fixed --ctx-size. Asking for more than the
  // worker holds does not enlarge it; the surplus is trimmed and wasted.
  const modelCtx = model?.context_length ?? 16384;
  const workerCtx = worker?.maxContext ?? 0;
  const context = workerCtx > 0 ? Math.min(modelCtx, workerCtx) : modelCtx;
  // History is trimmed to (context - maxTokens), so raising the reply length
  // directly shortens the memory. A quarter leaves three for the conversation.
  const maxTokens = Math.max(256, Math.round(context / 4 / 256) * 256);
  return { context, maxTokens };
}

export function SettingsModal({ open, onClose, theme, onToggleTheme, onSaved, me, pool }: Props) {
  const [tab, setTab] = useState<'model' | 'users'>('model');
  const [settings, setSettings] = useState<ChatSettings | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The worker that would actually serve the selected model, so the advice
  // describes the machine in play rather than the largest one anywhere.
  const activeWorker = (pool?.workers ?? []).find((w) =>
    settings?.model ? (w.loaded ?? []).includes(settings.model) : true,
  );
  const hostName = activeWorker ? PLATFORM_LABEL[activeWorker.platform ?? 'other'] : '';
  const rec = recommendationsFor(
    models.find((m) => m.id === settings?.model),
    activeWorker,
  );

  useEffect(() => {
    if (!open) return;
    setError(null);
    api
      .getSettings()
      .then((r) => {
        setSettings(r.defaults);
        setModels(r.models);
      })
      .catch((e) => setError(e.message));
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const patch = (p: Partial<ChatSettings>) => setSettings((s) => ({ ...(s ?? {}), ...p }));

  async function save() {
    if (!settings) return;
    setSaving(true);
    setError(null);
    try {
      await api.saveSettings({
        model: settings.model ?? 'qwen3-30b-a3b',
        temperature: settings.temperature ?? 0.7,
        topP: settings.topP ?? 0.9,
        topK: settings.topK,
        maxTokens: settings.maxTokens ?? 2048,
        contextLength: settings.contextLength ?? 16384,
        repeatPenalty: settings.repeatPenalty,
        systemPrompt: settings.systemPrompt,
        useGlobalContext: Boolean(settings.useGlobalContext),
        autoStartWorker: Boolean(settings.autoStartWorker),
        idleShutdownMinutes: Number(settings.idleShutdownMinutes ?? 30),
        webSearch: Boolean(settings.webSearch),
      });
      onSaved();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const selectedModel = models.find((m) => m.id === settings?.model);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-xl border shadow-2xl"
        style={{ backgroundColor: 'rgb(var(--bg-elevated))', borderColor: 'rgb(var(--border))' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="sticky top-0 flex items-center justify-between border-b px-5 py-3.5"
          style={{ backgroundColor: 'rgb(var(--bg-elevated))', borderColor: 'rgb(var(--border))' }}
        >
          <h2 className="text-base font-semibold">Settings</h2>
          <button onClick={onClose} className="btn btn-ghost p-1.5" aria-label="Close settings">
            <CloseIcon />
          </button>
        </div>

        {me?.role === 'admin' && (
          <div className="flex gap-1 border-b px-5" style={{ borderColor: 'rgb(var(--border))' }}>
            {(['model', 'users'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="relative px-3 py-2 text-sm font-medium capitalize"
                style={{ color: tab === t ? 'rgb(var(--text))' : 'rgb(var(--text-muted))' }}
              >
                {t === 'model' ? 'Model' : 'Users'}
                {tab === t && (
                  <span
                    className="absolute inset-x-2 -bottom-px h-0.5 rounded-full"
                    style={{ backgroundColor: 'rgb(var(--accent))' }}
                  />
                )}
              </button>
            ))}
          </div>
        )}

        {tab === 'users' && me?.role === 'admin' ? (
          <div className="px-5 py-5">
            <UsersPanel me={me} />
          </div>
        ) : !settings ? (
          <p className="px-5 py-8 text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
            Loading…
          </p>
        ) : (
          <div className="space-y-6 px-5 py-5">
            <div>
              <label className="mb-1 block text-sm font-medium">Model</label>
              <select
                className="input"
                value={settings.model ?? ''}
                onChange={(e) => patch({ model: e.target.value })}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name}
                  </option>
                ))}
              </select>
              {selectedModel && (
                <p className="mt-1 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                  {selectedModel.engine} · {selectedModel.quantization ?? 'unquantized'} · context up to{' '}
                  {selectedModel.context_length.toLocaleString()} tokens
                </p>
              )}
            </div>

            <Slider
              label="Temperature"
              hint="Higher is more creative; lower is more deterministic."
              value={settings.temperature ?? 0.7}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => patch({ temperature: v })}
            />

            <Slider
              label="Top-p"
              hint="Nucleus sampling — the cumulative probability cut-off."
              value={settings.topP ?? 0.9}
              min={0.05}
              max={1}
              step={0.05}
              onChange={(v) => patch({ topP: v })}
            />

            <Slider
              label="Repeat penalty"
              hint="Discourages the model from repeating itself."
              value={settings.repeatPenalty ?? 1.1}
              min={1}
              max={2}
              step={0.05}
              onChange={(v) => patch({ repeatPenalty: v })}
            />

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-sm font-medium">Longest reply</label>
                <input
                  type="number"
                  className="input"
                  min={1}
                  max={32768}
                  value={settings.maxTokens ?? 2048}
                  onChange={(e) => patch({ maxTokens: Number(e.target.value) })}
                />
                <FieldHint
                  text={
                    `The most the assistant may write in one answer. It shares the window with ` +
                    `the conversation, so a longer reply means less of the chat is remembered.`
                  }
                  recommended={rec.maxTokens}
                  current={settings.maxTokens ?? 2048}
                  onApply={() => patch({ maxTokens: rec.maxTokens })}
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">Memory window</label>
                <input
                  type="number"
                  className="input"
                  min={512}
                  max={131072}
                  step={512}
                  value={settings.contextLength ?? 16384}
                  onChange={(e) => patch({ contextLength: Number(e.target.value) })}
                />
                <FieldHint
                  text={
                    hostName
                      ? `How much of the conversation the assistant can see at once. ` +
                        `${hostName} is running ${rec.context.toLocaleString()} tokens; asking for ` +
                        `more does not enlarge it, so the extra is trimmed rather than used.`
                      : `How much of the conversation the assistant can see at once. Start a GPU ` +
                        `to see what it can actually hold.`
                  }
                  recommended={rec.context}
                  current={settings.contextLength ?? 16384}
                  onApply={() => patch({ contextLength: rec.context })}
                />
              </div>
            </div>

            <div
              className="flex items-start justify-between gap-4 rounded-lg border p-3"
              style={{ borderColor: 'rgb(var(--border))' }}
            >
              <div className="min-w-0">
                <p className="text-sm font-medium">Share context across chats</p>
                <p className="mt-0.5 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                  Chats are separate by default, so telling the assistant your name in
                  one chat leaves a new chat unaware of it. Turn this on to let every
                  chat see recent turns from your other chats. It uses part of the
                  context window, so long answers have slightly less room.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={Boolean(settings.useGlobalContext)}
                onClick={() => patch({ useGlobalContext: !settings.useGlobalContext })}
                className="relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition-colors"
                style={{
                  backgroundColor: settings.useGlobalContext
                    ? 'rgb(var(--accent))'
                    : 'rgb(var(--border))',
                }}
              >
                <span
                  className="absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform"
                  style={{
                    transform: settings.useGlobalContext ? 'translateX(22px)' : 'translateX(2px)',
                  }}
                />
              </button>
            </div>

            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium">Start a GPU automatically</p>
                <p className="mt-0.5 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                  When you send a message and no GPU is running, keep the message and
                  start a Kaggle session for it. Your weekly Kaggle quota is finite, so
                  this is off unless you turn it on; a burst of messages starts one
                  session, not one each. Colab cannot be started this way — Google
                  provides no API for it — so a message simply waits for the notebook.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={Boolean(settings.autoStartWorker)}
                onClick={() => patch({ autoStartWorker: !settings.autoStartWorker })}
                className="relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition-colors"
                style={{
                  backgroundColor: settings.autoStartWorker
                    ? 'rgb(var(--accent))'
                    : 'rgb(var(--border))',
                }}
              >
                <span
                  className="absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform"
                  style={{
                    transform: settings.autoStartWorker ? 'translateX(22px)' : 'translateX(2px)',
                  }}
                />
              </button>
            </div>

            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium">Search the web</p>
                <p className="mt-0.5 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                  A local model only knows what it was trained on, and has no way to tell
                  you where that ends — so asked about anything recent it answers
                  confidently and wrongly. Turn this on to look the question up first and
                  hand the model real sources, which it cites as [1], [2]. Fetching the
                  pages takes ten to fifteen seconds, so replies start slower; the same
                  question asked twice is answered from cache.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={Boolean(settings.webSearch)}
                onClick={() => patch({ webSearch: !settings.webSearch })}
                className="relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition-colors"
                style={{
                  backgroundColor: settings.webSearch
                    ? 'rgb(var(--accent))'
                    : 'rgb(var(--border))',
                }}
              >
                <span
                  className="absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform"
                  style={{
                    transform: settings.webSearch ? 'translateX(22px)' : 'translateX(2px)',
                  }}
                />
              </button>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">
                Shut an idle GPU down after
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={0}
                  max={720}
                  step={5}
                  className="input w-28"
                  value={settings.idleShutdownMinutes ?? 30}
                  onChange={(e) => patch({ idleShutdownMinutes: Number(e.target.value) })}
                />
                <span className="text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
                  minutes
                </span>
              </div>
              <p className="mt-1 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                A worker with nothing to do still bills GPU time, and Kaggle's weekly
                allowance is finite — so it stops itself. Raise this if a worker keeps
                disappearing between questions; set 0 to keep it running until you stop
                it. Takes effect the next time a worker connects.
              </p>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">System prompt</label>
              <textarea
                className="input min-h-[90px] resize-y"
                value={settings.systemPrompt ?? ''}
                placeholder="You are a helpful assistant."
                onChange={(e) => patch({ systemPrompt: e.target.value })}
              />
            </div>

            <div className="flex items-center justify-between border-t pt-4" style={{ borderColor: 'rgb(var(--border))' }}>
              <div>
                <p className="text-sm font-medium">Appearance</p>
                <p className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                  Currently {theme}.
                </p>
              </div>
              <button onClick={onToggleTheme} className="btn btn-ghost">
                {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
                {theme === 'dark' ? 'Light' : 'Dark'}
              </button>
            </div>

            {error && (
              <p className="text-sm" style={{ color: 'rgb(var(--danger))' }}>
                {error}
              </p>
            )}

            <div className="flex justify-end gap-2">
              <button onClick={onClose} className="btn btn-ghost">
                Cancel
              </button>
              <button onClick={save} className="btn btn-primary" disabled={saving}>
                {saving ? 'Saving…' : 'Save settings'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
