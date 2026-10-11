import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from './lib/api';
import type {
  AuthStatus,
  Conversation,
  ConversationMetrics,
  Message,
  SessionStatus,
  WorkerPool,
  WorkerSample,
} from './lib/types';
import { useChatStream } from './hooks/useChatStream';
import { AuthScreen } from './components/AuthScreen';
import { LogoMark } from './components/Logo';
import { Sidebar } from './components/Sidebar';
import { MessageList } from './components/MessageList';
import { Composer } from './components/Composer';
import { MetricsPanel } from './components/MetricsPanel';
import { ModelsPanel } from './components/ModelsPanel';
import { SettingsModal } from './components/SettingsModal';
import { ChartIcon, MenuIcon, MoonIcon, SunIcon } from './components/Icons';

type Theme = 'light' | 'dark';

function initialTheme(): Theme {
  const stored = localStorage.getItem('llm-theme');
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const unlocked = auth?.authenticated ?? null;
  const [theme, setTheme] = useState<Theme>(initialTheme);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(() =>
    localStorage.getItem('llm-active-conversation'),
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<ConversationMetrics | null>(null);
  const [pool, setPool] = useState<WorkerPool | null>(null);
  const [samples, setSamples] = useState<WorkerSample[]>([]);
  const [session, setSession] = useState<SessionStatus | null>(null);
  const [workerBusy, setWorkerBusy] = useState<'start' | 'stop' | null>(null);
  const [workerNotice, setWorkerNotice] = useState<string | null>(null);
  const SAMPLE_WINDOW = 15;

  const [draft, setDraft] = useState('');
  const [sendError, setSendError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const activeIdRef = useRef<string | null>(null);
  activeIdRef.current = activeId;

  // Remember which chat was open so a reload returns to it.
  useEffect(() => {
    if (activeId) localStorage.setItem('llm-active-conversation', activeId);
    else localStorage.removeItem('llm-active-conversation');
  }, [activeId]);

  // ── Theme ────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('llm-theme', theme);
  }, [theme]);

  // ── Auth ─────────────────────────────────────────────────
  const refreshAuth = useCallback(async () => {
    try {
      setAuth(await api.authStatus());
    } catch {
      // Offline or the API is down — show the sign-in screen rather than a
      // blank page, so the failure is at least legible.
      setAuth({
        authenticated: false,
        user: null,
        googleEnabled: false,
        signupEnabled: true,
        requiresAccessToken: false,
        unlocked: false,
        setupRequired: false,
      });
    }
  }, []);

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.listConversations());
    } catch {
      /* transient — the list refreshes again after the next action */
    }
  }, []);

  const loadConversation = useCallback(async (id: string): Promise<Message[]> => {
    const [{ messages: msgs }, m] = await Promise.all([
      api.getConversation(id),
      api.conversationMetrics(id).catch(() => null),
    ]);
    setMessages(msgs);
    setMetrics(m);
    return msgs;
  }, []);

  /** After a generation ends, re-read the thread so persisted metrics appear. */
  const onStreamFinished = useCallback(() => {
    const id = activeIdRef.current;
    if (!id) return;
    void loadConversation(id);
    void refreshConversations();
  }, [loadConversation, refreshConversations]);

  const stream = useChatStream(onStreamFinished);
  const startStream = stream.start;

  // ── Initial load ─────────────────────────────────────────
  useEffect(() => {
    if (!unlocked) return;
    void refreshConversations();
  }, [unlocked, refreshConversations]);

  // Opening a chat also re-attaches to a generation that is still running —
  // the server replays the deltas it has already emitted, so a reload during a
  // long answer resumes instead of stranding a half-written message.
  useEffect(() => {
    if (!unlocked) return;
    if (!activeId) {
      setMessages([]);
      setMetrics(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const msgs = await loadConversation(activeId);
        if (cancelled) return;
        const lastAssistant = [...msgs].reverse().find((m) => m.role === 'assistant');
        if (
          lastAssistant &&
          (lastAssistant.status === 'streaming' || lastAssistant.status === 'pending') &&
          lastAssistant.job_id
        ) {
          startStream(lastAssistant.job_id, lastAssistant.id);
        }
      } catch {
        // A conversation id remembered from a previous session may be gone.
        setActiveId(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId, unlocked, loadConversation, startStream]);

  // ── Worker availability polling ──────────────────────────
  useEffect(() => {
    if (!unlocked) return;
    let alive = true;
    const tick = async () => {
      try {
        const p = await api.workers();
        if (!alive) return;
        setPool(p);
        // Telemetry and session state only matter when control is configured.
        if (p.control?.kaggle) {
          const [s, sess] = await Promise.all([
            api.workerSamples(SAMPLE_WINDOW).catch(() => null),
            api.sessionStatus().catch(() => null),
          ]);
          if (!alive) return;
          if (s) setSamples(s.samples);
          if (sess) setSession(sess);
        }
      } catch {
        if (alive) setPool(null);
      }
    };
    void tick();
    const t = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [unlocked]);

  // ── Actions ──────────────────────────────────────────────
  const newChat = useCallback(async () => {
    const c = await api.createConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
    setSidebarOpen(false);
    stream.reset();
    setSendError(null);
  }, [stream]);

  const send = useCallback(async () => {
    const content = draft.trim();
    if (!content) return;
    setSendError(null);

    let conversationId = activeId;
    if (!conversationId) {
      const c = await api.createConversation();
      setConversations((prev) => [c, ...prev]);
      setActiveId(c.id);
      conversationId = c.id;
    }

    setDraft('');
    // Show the user's turn immediately rather than waiting on the round trip.
    const optimisticId = `optimistic-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: optimisticId,
        seq: prev.length + 1,
        role: 'user',
        content,
        status: 'complete',
        error: null,
        created_at: new Date().toISOString(),
      },
    ]);

    try {
      const res = await api.sendMessage(conversationId, content);
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== optimisticId),
        res.userMessage,
        res.assistantMessage,
      ]);
      if (res.queued || !res.jobId) {
        // Nothing can serve it yet, but it is saved. The pool poll below picks
        // it up when a worker arrives — the user does not retype anything.
        setWorkerNotice(res.message ?? 'Waiting for a GPU; this will be answered automatically.');
      } else {
        // Say so when the answer is coming from a different model than the
        // default named. The user did not choose one, so this is information
        // rather than an error — but it must not be silent.
        if (res.substitutedFor && res.model) {
          setWorkerNotice(`Answering with ${res.model}, the model loaded right now.`);
        }
        stream.start(res.jobId, res.assistantMessage.id);
      }
      void refreshConversations();
    } catch (err) {
      setMessages((prev) => prev.filter((m) => m.id !== optimisticId));
      setDraft(content);
      setSendError(
        err instanceof ApiError && err.code === 'NO_WORKER_AVAILABLE'
          ? 'No GPU worker is connected right now. Start a worker and try again.'
          : err instanceof ApiError
            ? err.message
            : 'Could not reach the server.',
      );
    }
  }, [draft, activeId, stream, refreshConversations]);

  const stop = useCallback(async () => {
    if (!stream.jobId) return;
    try {
      await api.cancelJob(stream.jobId);
    } catch {
      /* the job may have finished a moment earlier */
    }
  }, [stream.jobId]);

  const retry = useCallback(async () => {
    if (!activeId) return;
    setSendError(null);
    try {
      const res = await api.retry(activeId);
      stream.start(res.jobId, res.assistantMessageId);
    } catch (err) {
      setSendError(err instanceof ApiError ? err.message : 'Retry failed.');
    }
  }, [activeId, stream]);

  const signOut = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      // Drop local view state too, so the next account never sees the
      // previous one's conversation list flash before the reload.
      setConversations([]);
      setMessages([]);
      setActiveId(null);
      await refreshAuth();
    }
  }, [refreshAuth]);

  /** Pin this conversation to a model — routing matches on exactly this id. */
  const selectModel = useCallback(
    async (modelId: string) => {
      if (!activeId) return;
      try {
        // The server merges rather than replaces now, so this no longer discards
        // the conversation's other overrides.
        await api.updateConversation(activeId, { settings: { model: modelId } });
        await refreshConversations();
        setSendError('');
      } catch (err) {
        setSendError(err instanceof ApiError ? err.message : 'Could not switch model.');
      }
    },
    [activeId, refreshConversations],
  );

  // A queued turn becomes a streaming one the moment a GPU registers; reloading
  // the thread re-attaches the stream through the same path a page refresh uses.
  const hadWorkerRef = useRef(false);
  useEffect(() => {
    const now = (pool?.connected ?? 0) > 0;
    if (now && !hadWorkerRef.current && activeIdRef.current) {
      void loadConversation(activeIdRef.current).then(setMessages).catch(() => {});
      setWorkerNotice(null);
    }
    hadWorkerRef.current = now;
  }, [pool?.connected, loadConversation]);

  const startWorker = useCallback(async () => {
    setWorkerBusy('start');
    setWorkerNotice(null);
    try {
      const res = await api.startWorker();
      setWorkerNotice(`${res.message} The model takes a few minutes to load.`);
    } catch (err) {
      setWorkerNotice(err instanceof ApiError ? err.message : 'Could not start the worker.');
    } finally {
      setWorkerBusy(null);
    }
  }, []);

  // Takes the worker to stop, because more than one can be connected and the
  // server refuses an unqualified stop in that case rather than felling the
  // whole pool -- which is what one click used to do.
  const stopWorker = useCallback(async (workerId?: string) => {
    setWorkerBusy('stop');
    setWorkerNotice(null);
    try {
      const res = await api.stopWorker(workerId);
      setWorkerNotice(res.message);
    } catch (err) {
      setWorkerNotice(err instanceof ApiError ? err.message : 'Could not stop the worker.');
    } finally {
      setWorkerBusy(null);
    }
  }, []);

  const remove = useCallback(
    async (id: string) => {
      await api.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) setActiveId(null);
    },
    [activeId],
  );

  const archive = useCallback(
    async (id: string) => {
      await api.updateConversation(id, { archived: true });
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) setActiveId(null);
    },
    [activeId],
  );

  // ── Render ───────────────────────────────────────────────
  if (auth === null) {
    return (
      <div className="flex min-h-full items-center justify-center">
        <span className="text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
          Loading…
        </span>
      </div>
    );
  }

  if (!auth.authenticated) return <AuthScreen status={auth} onAuthenticated={refreshAuth} />;

  // `available` means "can accept a job right now"; a single worker is briefly
  // unavailable while it generates. Gating the UI on that would disable the
  // composer during every answer, so gate on whether any worker is connected.
  const workerConnected = (pool?.connected ?? 0) > 0;
  const activeConversation = conversations.find((c) => c.id === activeId);

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={(id) => {
          setActiveId(id);
          setSidebarOpen(false);
          stream.reset();
          setSendError(null);
        }}
        onNew={newChat}
        onDelete={remove}
        onArchive={archive}
        onOpenSettings={() => setSettingsOpen(true)}
        user={auth?.user ?? null}
        onSignOut={signOut}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header
          className="flex items-center gap-2 border-b px-3 py-2.5 md:px-4"
          style={{ borderColor: 'rgb(var(--border))' }}
        >
          <button
            onClick={() => setSidebarOpen(true)}
            className="btn btn-ghost p-1.5 md:hidden"
            aria-label="Open sidebar"
          >
            <MenuIcon />
          </button>

          <h1 className="min-w-0 flex-1 truncate text-sm font-medium">
            {activeConversation?.title ?? 'New chat'}
          </h1>

          <div className="flex items-center gap-1.5">
            <span
              className="hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] sm:flex"
              style={{ borderColor: 'rgb(var(--border))', color: 'rgb(var(--text-muted))' }}
              title={
                workerConnected
                  ? `${pool?.connected ?? 0} worker(s) connected`
                  : 'No inference worker is connected'
              }
            >
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: workerConnected ? 'rgb(var(--success))' : 'rgb(var(--danger))',
                }}
              />
              {workerConnected ? `${pool?.connected} worker${pool?.connected === 1 ? '' : 's'}` : 'No worker'}
            </span>

            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="btn btn-ghost p-1.5"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
            </button>

            <button
              onClick={() => setMetricsOpen((v) => !v)}
              className="btn btn-ghost p-1.5"
              aria-label="Toggle metrics panel"
              title="Inference metrics"
            >
              <ChartIcon />
            </button>
          </div>
        </header>

        {!workerConnected && (
          <div
            className="border-b px-4 py-2 text-center text-xs"
            style={{
              borderColor: 'rgb(var(--border))',
              backgroundColor: 'rgb(var(--bg-subtle))',
              color: 'rgb(var(--warn))',
            }}
          >
            No GPU worker is connected — start a worker to send messages.
          </div>
        )}

        {messages.length === 0 && !stream.isStreaming ? (
          <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
            <LogoMark size={52} className="mb-4" />
            <h2 className="text-lg font-semibold">How can I help?</h2>
            <p className="mt-1 max-w-sm text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
              Ask anything. Responses stream from your own GPU worker, and every generation is measured.
            </p>
          </div>
        ) : (
          <MessageList
            messages={messages}
            streamingMessageId={stream.messageId}
            streamingText={stream.text}
            streamingReasoning={stream.reasoning}
            isStreaming={stream.isStreaming}
            liveMetrics={stream.metrics}
            streamError={stream.error}
            onRetry={retry}
            streamingImageId={stream.imageId}
            streamingProgress={stream.progress}
          />
        )}

        {sendError && (
          <div className="mx-auto w-full max-w-3xl px-4 md:px-6">
            <div
              className="mb-2 rounded-lg border px-3 py-2 text-sm"
              style={{ borderColor: 'rgb(var(--danger))', color: 'rgb(var(--danger))' }}
            >
              {sendError}
            </div>
          </div>
        )}

        <Composer
          value={draft}
          onChange={setDraft}
          onSend={send}
          onStop={stop}
          isStreaming={stream.isStreaming}
          disabled={stream.isStreaming}
          placeholder={workerConnected ? 'Send a message…' : 'Waiting for a GPU worker…'}
        />
      </main>

      <MetricsPanel
        selectedModel={conversations.find((c) => c.id === activeId)?.settings?.model}
        modelsSlot={
          <ModelsPanel
            pool={pool}
            selectedModel={conversations.find((c) => c.id === activeId)?.settings?.model}
            onSelectModel={selectModel}
            isAdmin={auth?.user?.role === 'admin'}
            onStartWorker={startWorker}
            onStopWorker={stopWorker}
            workerBusy={workerBusy}
            workerNotice={workerNotice}
          />
        }
        open={metricsOpen}
        onClose={() => setMetricsOpen(false)}
        live={stream.metrics}
        isStreaming={stream.isStreaming}
        conversationMetrics={metrics}
        pool={pool}
        samples={samples}
        sampleWindow={SAMPLE_WINDOW}
        session={session}
        workerNotice={workerNotice}
      />

      <SettingsModal
        pool={pool}
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        onSaved={() => void refreshConversations()}
        me={auth?.user ?? null}
      />
    </div>
  );
}
