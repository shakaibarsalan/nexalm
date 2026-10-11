import { useCallback, useEffect, useRef, useState } from 'react';
import type { LiveMetrics } from '../lib/types';

export interface StreamState {
  jobId: string | null;
  messageId: string | null;
  text: string;
  reasoning: string;
  isStreaming: boolean;
  metrics: LiveMetrics | null;
  error: string | null;
  /** Set once an image turn finishes; the picture is fetched by id. */
  imageId: string | null;
  /**
   * How an image generation is going. sd-server reports a status and a queue
   * position and nothing else -- no step counter -- so this is elapsed time
   * and a state, not a percentage we would otherwise have to invent.
   */
  progress: { status: string; elapsedMs: number; steps?: number } | null;
}

const IDLE: StreamState = {
  jobId: null,
  messageId: null,
  text: '',
  reasoning: '',
  isStreaming: false,
  metrics: null,
  error: null,
  imageId: null,
  progress: null,
};

/**
 * Consumes the SSE token stream for one generation.
 *
 * The server replays every delta it has already emitted when a client
 * reconnects, so `text` is rebuilt from scratch on each attach rather than
 * appended to — that keeps a mid-stream reload from duplicating output.
 */
export function useChatStream(onFinished?: () => void) {
  const [state, setState] = useState<StreamState>(IDLE);
  const sourceRef = useRef<EventSource | null>(null);
  const finishedRef = useRef(onFinished);
  const seenRef = useRef<Set<number>>(new Set());
  const seenReasoningRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    finishedRef.current = onFinished;
  }, [onFinished]);

  const close = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  const start = useCallback(
    (jobId: string, messageId: string) => {
      close();
      seenRef.current = new Set();
      seenReasoningRef.current = new Set();
      setState({ ...IDLE, jobId, messageId, isStreaming: true });

      const es = new EventSource(`/api/jobs/${jobId}/stream`);
      sourceRef.current = es;

      es.addEventListener('status', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data);
        setState((s) => ({ ...s, metrics: { ...s.metrics, workerName: data.workerName } }));
      });

      es.addEventListener('first_token', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data);
        setState((s) => ({ ...s, metrics: { ...s.metrics, ttftMs: data.ttftMs } }));
      });

      es.addEventListener('reasoning', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data) as { text: string; index: number };
        if (seenReasoningRef.current.has(data.index)) return;
        seenReasoningRef.current.add(data.index);
        setState((s) => ({ ...s, reasoning: s.reasoning + data.text }));
      });

      es.addEventListener('delta', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data) as { text: string; index: number };
        // Guard against a replayed chunk arriving twice.
        if (seenRef.current.has(data.index)) return;
        seenRef.current.add(data.index);
        setState((s) => ({ ...s, text: s.text + data.text }));
      });

      es.addEventListener('progress', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data) as {
          status: string;
          elapsedMs: number;
          steps?: number;
        };
        setState((s) => ({ ...s, progress: data }));
      });

      es.addEventListener('image', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data) as { id: string };
        setState((s) => ({ ...s, imageId: data.id }));
      });

      es.addEventListener('done', (ev) => {
        const data = JSON.parse((ev as MessageEvent).data) as LiveMetrics;
        setState((s) => ({ ...s, isStreaming: false, metrics: { ...s.metrics, ...data } }));
        close();
        finishedRef.current?.();
      });

      es.addEventListener('error', (ev) => {
        // A native EventSource failure carries no `data`; a server-sent
        // `error` event does. Only the latter is a real inference error.
        const raw = (ev as MessageEvent).data;
        if (raw) {
          const data = JSON.parse(raw) as { code: string; message: string };
          setState((s) => ({ ...s, isStreaming: false, error: data.message }));
          close();
          finishedRef.current?.();
        } else if (es.readyState === EventSource.CLOSED) {
          setState((s) =>
            s.isStreaming ? { ...s, isStreaming: false, error: 'Connection to the server was lost.' } : s,
          );
          close();
          finishedRef.current?.();
        }
      });
    },
    [close],
  );

  const reset = useCallback(() => {
    close();
    setState(IDLE);
  }, [close]);

  useEffect(() => close, [close]);

  return { ...state, start, reset, close };
}
