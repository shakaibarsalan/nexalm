import { useCallback, useEffect, useRef, useState } from 'react';
import type { LiveMetrics, Message } from '../lib/types';
import { Markdown } from './Markdown';
import { CheckIcon, CopyIcon, RetryIcon } from './Icons';

/**
 * Thinking trace, collapsed by default.
 *
 * Qwen3 spends most of its tokens here — often hundreds before the first word
 * of the answer. The user waits for them regardless, so hiding them entirely
 * makes the model look slow for no reason; showing them expanded would bury
 * the answer. Collapsed is the honest middle.
 */
function ReasoningBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;

  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs transition-colors"
        style={{ color: 'rgb(var(--text-muted))' }}
      >
        <span
          className="inline-block transition-transform"
          style={{ transform: open ? 'rotate(90deg)' : 'none' }}
        >
          &rsaquo;
        </span>
        {live ? 'Thinking…' : 'Thought process'}
        <span style={{ color: 'rgb(var(--text-faint))' }}>
          ({text.length.toLocaleString()} chars)
        </span>
      </button>

      {open && (
        <div
          className="mt-1.5 max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg border px-3 py-2
                     font-mono text-[12px] leading-5"
          style={{
            borderColor: 'rgb(var(--border))',
            backgroundColor: 'rgb(var(--bg-subtle))',
            color: 'rgb(var(--text-muted))',
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}

function fmtMs(ms?: number | null): string {
  if (ms == null) return '—';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** Compact per-message metric strip shown under an assistant turn. */
/**
 * The pages an answer was given.
 *
 * Shown because a citation the reader cannot follow is worse than none: the
 * model writes [1] whether or not the source says what it claims, and the only
 * defence is making the source one click away.
 */
function Sources({ list }: { list?: { title: string; url: string; source: string }[] | null }) {
  if (!list || list.length === 0) return null;
  const host = (u: string) => {
    try {
      return new URL(u).hostname.replace(/^www\./, '');
    } catch {
      return u;
    }
  };
  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      <span className="text-[11px]" style={{ color: 'rgb(var(--text-faint))' }}>
        Sources
      </span>
      {list.map((item, i) => (
        <a
          key={item.url}
          href={item.url}
          target="_blank"
          rel="noreferrer noopener"
          title={`${item.title} — ${item.source}`}
          className="max-w-[15rem] truncate rounded px-1.5 py-0.5 text-[11px]"
          style={{ backgroundColor: 'rgb(var(--bg-subtle))', color: 'rgb(var(--text-muted))' }}
        >
          [{i + 1}] {host(item.url)}
        </a>
      ))}
    </div>
  );
}

function MessageMetrics({ m, live }: { m: Message; live?: LiveMetrics | null }) {
  const tps = live?.tokensPerSec ?? m.tokens_per_sec;
  const out = live?.completionTokens ?? m.completion_tokens;
  const inp = live?.promptTokens ?? m.prompt_tokens;
  const ttft = live?.ttftMs ?? m.ttft_ms;
  const gen = live?.genMs ?? m.gen_ms;

  if (tps == null && out == null && ttft == null) return null;

  const items: Array<[string, string]> = [
    ['in', inp != null ? String(inp) : '—'],
    ['out', out != null ? String(out) : '—'],
    ['tok/s', tps != null ? Number(tps).toFixed(1) : '—'],
    ['ttft', fmtMs(ttft)],
    ['gen', fmtMs(gen)],
  ];

  return (
    <div
      className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]"
      style={{ color: 'rgb(var(--text-faint))' }}
    >
      {items.map(([k, v]) => (
        <span key={k}>
          <span className="opacity-70">{k}</span> {v}
        </span>
      ))}
      {(live?.workerName ?? m.worker_name) && (
        <span className="opacity-70">· {live?.workerName ?? m.worker_name}</span>
      )}
    </div>
  );
}

function CopyMessage({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="btn btn-ghost p-1.5"
      aria-label={copied ? 'Copied' : 'Copy message'}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        } catch {
          /* clipboard blocked */
        }
      }}
    >
      {copied ? <CheckIcon className="h-3.5 w-3.5" /> : <CopyIcon className="h-3.5 w-3.5" />}
    </button>
  );
}

/**
 * What to say while a picture is being drawn.
 *
 * sd-server reports a status and a queue position and nothing else -- there is
 * no per-step counter to read -- so this counts seconds and names the state
 * rather than showing a progress bar whose position would be made up.
 */
function ImageSkeleton({
  progress,
}: {
  progress: { status: string; elapsedMs: number; steps?: number };
}) {
  return (
    <div className="mt-1">
      {/* Sized like the thumbnail it becomes, so the thread does not jump when
          the picture arrives. A blinking dot said "something is happening";
          this says "a picture is happening", which is the part that was
          missing while a generation takes the better part of a minute. */}
      <div
        className="image-skeleton flex h-64 w-64 max-w-full items-center justify-center rounded-lg border"
        style={{ borderColor: 'rgb(var(--border))' }}
        role="status"
        aria-live="polite"
        aria-label={describeProgress(progress)}
      >
        <ImagePlaceholderIcon className="h-10 w-10 opacity-40" />
      </div>
      <p className="mt-1.5 text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
        {describeProgress(progress)}
      </p>
    </div>
  );
}

/** A picture that is not there yet. Deliberately plain; it is a placeholder. */
function ImagePlaceholderIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="m21 15-4.5-4.5L3 21" />
    </svg>
  );
}

function describeProgress(p: { status: string; elapsedMs: number; steps?: number }): string {
  const seconds = Math.round(p.elapsedMs / 1000);
  if (p.status === 'queued') return 'Queued on the GPU…';
  const steps = p.steps ? ` (${p.steps} steps)` : '';
  return `Drawing${steps}… ${seconds}s`;
}

/**
 * One generated picture.
 *
 * Fetched by id through the API rather than embedded in the message, so it is
 * served under the viewer's own session and a conversation's history stays
 * small no matter how many images it holds.
 */
function GeneratedImage({
  id,
  width,
  height,
}: {
  id: string;
  width?: number | null;
  height?: number | null;
}) {
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);
  const src = `/api/images/${id}`;

  // Escape closes, matching the settings dialog. Bound only while open so the
  // thread does not carry a listener per image it has ever rendered.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  if (failed) {
    return (
      <p className="mt-2 text-sm italic" style={{ color: 'rgb(var(--text-faint))' }}>
        This image is no longer available.
      </p>
    );
  }

  return (
    <>
      {/* A thumbnail, not the full picture. A 768px image filled the column and
          pushed the rest of the conversation off screen, which made a thread
          with several of them unreadable. Capped by height rather than width so
          portrait and landscape take a similar amount of the thread. */}
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mt-1 block cursor-zoom-in overflow-hidden rounded-lg border transition-opacity hover:opacity-90"
        style={{ borderColor: 'rgb(var(--border))' }}
        title="Open full size"
        aria-label="Open the generated image full size"
      >
        <img
          src={src}
          alt="Generated image"
          onError={() => setFailed(true)}
          className="block max-h-64 w-auto max-w-full object-contain"
          loading="lazy"
        />
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="Generated image"
        >
          <img
            src={src}
            alt="Generated image, full size"
            width={width ?? undefined}
            height={height ?? undefined}
            // Stops the click that opened nothing from closing it again, so a
            // click on the picture itself is not read as a click on the backdrop.
            onClick={(e) => e.stopPropagation()}
            className="max-h-[90vh] max-w-[90vw] cursor-zoom-out rounded-lg object-contain shadow-2xl"
          />
          <a
            href={src}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="absolute bottom-6 right-6 rounded-lg px-3 py-1.5 text-sm"
            style={{
              backgroundColor: 'rgb(var(--bg-elevated))',
              color: 'rgb(var(--text-muted))',
            }}
          >
            Open in a new tab
          </a>
        </div>
      )}
    </>
  );
}

interface Props {
  messages: Message[];
  streamingMessageId: string | null;
  streamingText: string;
  streamingReasoning: string;
  isStreaming: boolean;
  liveMetrics: LiveMetrics | null;
  streamError: string | null;
  onRetry: () => void;
  /** Set when the streaming turn produced a picture rather than text. */
  streamingImageId?: string | null;
  /** Coarse state of an in-flight image generation, or null for a text turn. */
  streamingProgress?: { status: string; elapsedMs: number; steps?: number } | null;
}

export function MessageList({
  messages,
  streamingMessageId,
  streamingText,
  streamingReasoning,
  isStreaming,
  liveMetrics,
  streamError,
  onRetry,
  streamingImageId = null,
  streamingProgress = null,
}: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Only auto-scroll while the user is already at the bottom, so reading
  // earlier output is not yanked away by incoming tokens.
  const onScroll = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }, []);

  useEffect(() => {
    if (pinnedRef.current) endRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [messages, streamingText]);

  return (
    <div ref={scrollerRef} onScroll={onScroll} className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-4 py-6 md:px-6">
        {messages.map((m) => {
          const isStreamTarget = m.id === streamingMessageId;
          const body = isStreamTarget ? streamingText : m.content;
          const thinking = isStreamTarget ? streamingReasoning : (m.reasoning ?? '');
          const failed = m.status === 'error' && !isStreaming;
          // The live id wins while the turn streams; afterwards the same
          // picture arrives on the message itself, so a reload shows it too.
          const imageId = (isStreamTarget ? streamingImageId : null) ?? m.image_id ?? null;

          if (m.role === 'user') {
            return (
              <div key={m.id} className="mb-6 flex justify-end animate-fade-in">
                <div
                  className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md px-4 py-2.5 text-[15px] leading-7"
                  style={{ backgroundColor: 'rgb(var(--accent-soft))', color: 'rgb(var(--text))' }}
                >
                  {m.content}
                </div>
              </div>
            );
          }

          return (
            <div key={m.id} className="group mb-8 animate-fade-in">
              <div className="mb-1.5 flex items-center gap-2">
                <div
                  className="flex h-6 w-6 items-center justify-center rounded-md text-[10px] font-bold text-white"
                  style={{ backgroundColor: 'rgb(var(--accent))' }}
                >
                  AI
                </div>
                <span className="text-xs font-medium" style={{ color: 'rgb(var(--text-muted))' }}>
                  {m.model ?? 'Assistant'}
                </span>
                {body && (
                  <span className="opacity-0 transition-opacity group-hover:opacity-100">
                    <CopyMessage text={body} />
                  </span>
                )}
              </div>

              <ReasoningBlock text={thinking} live={isStreamTarget && isStreaming && !body} />

              {imageId ? (
                <GeneratedImage id={imageId} width={m.image_width} height={m.image_height} />
              ) : body ? (
                <Markdown content={body} />
              ) : isStreamTarget && isStreaming && streamingProgress ? (
                <ImageSkeleton progress={streamingProgress} />
              ) : isStreamTarget && isStreaming ? (
                <div className="flex items-center gap-2 py-1" style={{ color: 'rgb(var(--text-muted))' }}>
                  <span className="h-2 w-2 animate-blink rounded-full bg-current" />
                  <span className="text-sm">{thinking ? 'Thinking…' : 'Generating…'}</span>
                </div>
              ) : null}

              {isStreamTarget && isStreaming && body && (
                <span className="ml-0.5 inline-block h-4 w-[2px] animate-blink align-text-bottom bg-current" />
              )}

              {(failed || (isStreamTarget && streamError)) && (
                <div
                  className="mt-3 rounded-lg border px-3 py-2 text-sm"
                  style={{ borderColor: 'rgb(var(--danger))', color: 'rgb(var(--danger))' }}
                >
                  <p>{streamError ?? m.error ?? 'Generation failed.'}</p>
                  <button className="btn btn-ghost mt-1.5 px-2 py-1 text-xs" onClick={onRetry}>
                    <RetryIcon className="h-3.5 w-3.5" />
                    Retry
                  </button>
                </div>
              )}

              {m.status === 'cancelled' && (
                <p className="mt-2 text-xs italic" style={{ color: 'rgb(var(--text-faint))' }}>
                  Generation stopped.
                </p>
              )}

              <Sources list={m.sources} />
              <MessageMetrics m={m} live={isStreamTarget ? liveMetrics : null} />
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}
