import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, ApiError } from '../lib/api';
import type { ModelInfo, WorkerInfo, WorkerPool } from '../lib/types';

/**
 * Which models each GPU host can run, and which one is answering.
 *
 * Everything is scoped to **one host at a time**. A flat list was readable while
 * only one GPU could be connected, but with Kaggle and Colab both up it could
 * not answer the question that matters — which of them holds this model, and
 * where is a Load going to happen. Picking a host at the top answers it for
 * every row below.
 *
 * Text and image models are listed separately because they are different tools:
 * one answers with words, the other with a picture, and only one of them can
 * serve a given conversation.
 */

interface Props {
  pool: WorkerPool | null;
  /** The model this conversation will use, which is the one actually in play. */
  selectedModel?: string;
  onSelectModel?: (id: string) => void;
  isAdmin: boolean;
  onStartWorker?: () => void;
  onStopWorker?: (workerId?: string) => void;
  workerBusy?: string | null;
  workerNotice?: string | null;
}

type Platform = 'kaggle' | 'colab' | 'other';

const PLATFORM_LABEL: Record<Platform, string> = {
  kaggle: 'Kaggle',
  colab: 'Colab',
  other: 'Self-hosted',
};

/** Availability is a property of a model *on a host*, never of a model alone. */
type State = 'active' | 'ready' | 'loadable' | 'too_large' | 'no_worker';

// "Selected" rather than "In use": this marks the model the conversation is set
// to, which is not the same claim as "this is what answered". Saying the latter
// contradicted the Inference tab whenever the two hosts held different models.
const STATE_LABEL: Record<State, string> = {
  active: 'Selected',
  ready: 'Loaded',
  loadable: 'Not loaded',
  too_large: 'Too large',
  no_worker: 'No worker',
};

const fmtCtx = (n?: number | null) =>
  !n ? '—' : n >= 1024 ? `${Math.round(n / 1024)}k context` : `${n} context`;

export function ModelsPanel({
  pool,
  selectedModel,
  onSelectModel,
  isAdmin,
  onStartWorker,
  onStopWorker,
  workerBusy,
  workerNotice,
}: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string>('');
  const [host, setHost] = useState<Platform | null>(null);
  // Text and image models answer with different things and are never
  // alternatives to one another, so the list shows one kind at a time. They
  // used to be stacked, which buried the image models under a dozen text ones
  // and implied a single ranked list they are not part of.
  const [kind, setKind] = useState<'chat' | 'image'>('chat');

  const refresh = useCallback(async () => {
    try {
      setModels((await api.models()).models);
    } catch {
      /* the pool poll brings us back */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, pool?.loadedModels?.join(',')]);

  const platforms = useMemo(
    () => (pool?.platforms ?? []).filter((p) => p.connected > 0 || p.startable),
    [pool?.platforms],
  );

  // Default to a host that is actually running, so the list is useful on open.
  useEffect(() => {
    if (host && platforms.some((p) => p.platform === host)) return;
    const live = platforms.find((p) => p.connected > 0) ?? platforms[0];
    if (live) setHost(live.platform as Platform);
  }, [platforms, host]);

  const hostWorkers: WorkerInfo[] = useMemo(
    () => (pool?.workers ?? []).filter((w) => (w.platform ?? 'other') === host),
    [pool?.workers, host],
  );

  const hostVramGb = Math.max(0, ...hostWorkers.map((w) => (w.vramTotalMb ?? 0) / 1024));
  const hostLoaded = useMemo(
    () => new Set(hostWorkers.flatMap((w) => w.loaded ?? [])),
    [hostWorkers],
  );

  // Roughly what each host can keep on disk. Kaggle stores weights as Datasets
  // and has room; Colab persists through a free Drive, which is 15 GB with a
  // chat model already in it. So an image model is refused there on storage
  // rather than VRAM — FLUX needs about 6.4 GB of VRAM but 12.2 GB of files.
  const hostDiskGb = host === 'colab' ? 3 : 60;
  const hostName = host ? PLATFORM_LABEL[host] : '';

  function stateOf(m: ModelInfo): State {
    if (hostWorkers.length === 0) return 'no_worker';
    if (hostLoaded.has(m.id)) return m.id === selectedModel ? 'active' : 'ready';
    const needVram = Number(m.min_vram_gb ?? 0);
    if (needVram > 0 && needVram > hostVramGb) return 'too_large';
    if (Number(m.disk_gb ?? 0) > hostDiskGb) return 'too_large';
    return 'loadable';
  }

  /** Why a model will not run here, in the terms that actually decided it. */
  function whyTooLarge(m: ModelInfo): string {
    const needVram = Number(m.min_vram_gb ?? 0);
    if (needVram > hostVramGb) {
      return `Needs about ${needVram} GB of VRAM; ${hostName} has ${hostVramGb.toFixed(0)} GB`;
    }
    return (
      `Needs ${m.disk_gb} GB of weights on disk, which will not fit ${hostName}'s storage. ` +
      `It would run on the GPU — it is the download that will not fit.`
    );
  }

  async function act(kind: 'load' | 'unload', id: string) {
    setBusy(id);
    setNotice('');
    try {
      // Named explicitly, so a Load happens on the host the list is showing
      // rather than wherever the server would otherwise pick.
      const workerId = hostWorkers[0]?.id;
      const res =
        kind === 'load' ? await api.loadModel(id, workerId) : await api.unloadModel(id, workerId);
      setNotice(res.message);
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : `Could not ${kind} the model.`);
    } finally {
      setBusy(null);
      void refresh();
    }
  }

  const enabled = models.filter((m) => m.enabled);
  const chatModels = enabled.filter((m) => (m.kind ?? 'chat') === 'chat');
  const imageModels = enabled.filter((m) => m.kind === 'image');

  const renderModels = (list: ModelInfo[]) => (
    <ul className="model-list">
      {list.map((m) => {
        const state = stateOf(m);
        return (
          <li key={m.id} className={`model ${state}`}>
            <button
              type="button"
              className="model-pick"
              onClick={() => onSelectModel?.(m.id)}
              disabled={state === 'too_large'}
              title={
                state === 'too_large'
                  ? whyTooLarge(m)
                  : `Use ${m.display_name} for this conversation`
              }
            >
              <span className="model-name">{m.display_name}</span>
              <span className="model-meta">
                {m.approx_size_gb ? `${m.approx_size_gb} GB` : '—'}
                {m.kind === 'image'
                  ? m.disk_gb && m.disk_gb !== m.approx_size_gb
                    ? ` · ${m.disk_gb} GB on disk`
                    : ''
                  : ` · ${fmtCtx(m.context_length)}`}
                {m.supports_vision ? ' · reads images' : ''}
              </span>
            </button>

            <span className={`model-state ${state}`}>{STATE_LABEL[state]}</span>

            {isAdmin && state === 'loadable' && hostWorkers.length > 0 ? (
              <button
                type="button"
                className="model-action"
                onClick={() => act('load', m.id)}
                disabled={busy !== null}
                title={`Download if needed, and load onto ${hostName}`}
              >
                {busy === m.id ? 'Loading…' : 'Load'}
              </button>
            ) : null}
            {isAdmin && (state === 'ready' || state === 'active') ? (
              <button
                type="button"
                className="model-action subtle"
                onClick={() => act('unload', m.id)}
                disabled={busy !== null}
                title={`Free its VRAM on ${hostName}`}
              >
                {busy === m.id ? 'Unloading…' : 'Unload'}
              </button>
            ) : null}
          </li>
        );
      })}
    </ul>
  );

  return (
    <section className="models-panel">
      <h3 className="panel-heading">GPU hosts</h3>
      {platforms.length === 0 ? (
        <p className="muted small">
          No GPU host is connected. Start a Kaggle session, or run the Colab notebook.
        </p>
      ) : (
        <ul className="platform-list">
          {platforms.map((p) => {
            const selected = p.platform === host;
            return (
              <li key={p.platform}>
                <button
                  type="button"
                  aria-pressed={selected}
                  onClick={() => setHost(p.platform as Platform)}
                  className={`platform ${p.connected > 0 ? 'up' : 'down'}${
                    selected ? ' selected' : ''
                  }`}
                >
                  <span className="platform-name">{PLATFORM_LABEL[p.platform as Platform]}</span>
                  <span className="platform-state">
                    {p.connected > 0
                      ? `${p.loaded.length || 'no'} model${p.loaded.length === 1 ? '' : 's'} loaded`
                      : p.startable
                        ? 'not running'
                        : 'not running · run the notebook'}
                  </span>

                  {isAdmin && (
                    <span className="platform-actions">
                      {p.connected > 0 ? (
                        <span
                          role="button"
                          tabIndex={0}
                          className="model-action subtle"
                          onClick={(e) => {
                            e.stopPropagation();
                            onStopWorker?.(p.workers[0]);
                          }}
                        >
                          {workerBusy === 'stop' ? 'Stopping…' : 'Stop'}
                        </span>
                      ) : p.startable ? (
                        <span
                          role="button"
                          tabIndex={0}
                          className="model-action"
                          onClick={(e) => {
                            e.stopPropagation();
                            onStartWorker?.();
                          }}
                        >
                          {workerBusy === 'start' ? 'Starting…' : 'Start'}
                        </span>
                      ) : (
                        <span
                          className="platform-manual"
                          title="Google provides no API to start Colab"
                        >
                          manual
                        </span>
                      )}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {hostWorkers.length > 0 ? (
        <p className="muted small">
          {hostName}: {hostWorkers[0].gpuCount} ×{' '}
          {(hostVramGb / Math.max(1, hostWorkers[0].gpuCount)).toFixed(0)} GB (
          {hostVramGb.toFixed(0)} GB total). Loading and unloading affects {hostName} only.
        </p>
      ) : (
        <p className="muted small">
          {hostName || 'This host'} is not running, so nothing can be loaded on it yet.
        </p>
      )}

      <div className="kind-switch" role="tablist" aria-label="Model kind">
        <button
          type="button"
          role="tab"
          aria-selected={kind === 'chat'}
          className={`kind-tab${kind === 'chat' ? ' selected' : ''}`}
          onClick={() => setKind('chat')}
        >
          Text
          <span className="kind-count">{chatModels.length}</span>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={kind === 'image'}
          className={`kind-tab${kind === 'image' ? ' selected' : ''}`}
          onClick={() => setKind('image')}
          disabled={imageModels.length === 0}
          title={
            imageModels.length === 0
              ? 'No image models are enabled.'
              : 'Models that answer with a picture'
          }
        >
          Image
          <span className="kind-count">{imageModels.length}</span>
        </button>
      </div>

      <p className="muted small">
        {kind === 'image'
          ? 'These answer with a picture rather than text. Their weights are large, so they load where there is room to keep them.'
          : 'These answer with text.'}
        {host ? ` Showing what ${hostName} can run.` : ''}
      </p>

      {renderModels(kind === 'image' ? imageModels : chatModels)}

      {(notice || workerNotice) && <p className="model-notice">{notice || workerNotice}</p>}
      {!isAdmin && (
        <p className="muted small">Loading and unloading models is an administrator action.</p>
      )}
    </section>
  );
}
