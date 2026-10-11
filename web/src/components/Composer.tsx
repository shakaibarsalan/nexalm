import { useEffect, useRef } from 'react';
import { SendIcon, StopIcon } from './Icons';

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled: boolean;
  placeholder: string;
}

const MIN_HEIGHT = 36;
const MAX_HEIGHT = 224;

export function Composer({ value, onChange, onSend, onStop, isStreaming, disabled, placeholder }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with content up to a cap, then scroll internally.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const resize = () => {
      el.style.height = 'auto';
      el.style.height = `${Math.min(Math.max(el.scrollHeight, MIN_HEIGHT), MAX_HEIGHT)}px`;
    };

    resize();

    // The measurement is only meaningful once the element has its real width.
    // On first paint it can still be zero-width, which wraps the placeholder
    // over many lines, reports a huge scrollHeight, and latches the box open at
    // the cap for the rest of the session. Re-measuring on resize fixes that
    // and keeps the height right when the pane or window changes.
    const observer = new ResizeObserver(resize);
    observer.observe(el);

    // Web fonts land after first paint and change line height.
    let stale = false;
    document.fonts?.ready.then(() => {
      if (!stale) resize();
    });

    return () => {
      stale = true;
      observer.disconnect();
    };
  }, [value]);

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  }

  return (
    <div className="border-t px-4 py-3 md:px-6" style={{ borderColor: 'rgb(var(--border))' }}>
      <div className="mx-auto w-full max-w-3xl">
        <div
          className="flex items-end gap-2 rounded-2xl border p-2 transition"
          style={{ backgroundColor: 'rgb(var(--bg-elevated))', borderColor: 'rgb(var(--border))' }}
        >
          <textarea
            ref={ref}
            rows={1}
            value={value}
            disabled={disabled}
            placeholder={placeholder}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            className="max-h-56 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-6
                       outline-none placeholder:opacity-60 disabled:opacity-60"
            style={{ color: 'rgb(var(--text))' }}
          />

          {isStreaming ? (
            <button onClick={onStop} className="btn btn-ghost shrink-0 px-2.5" title="Stop generating">
              <StopIcon className="h-4 w-4" />
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={disabled || !value.trim()}
              className="btn btn-primary shrink-0 px-2.5"
              title="Send (Enter)"
            >
              <SendIcon className="h-4 w-4" />
            </button>
          )}
        </div>

        <p className="mt-1.5 text-center text-[11px]" style={{ color: 'rgb(var(--text-faint))' }}>
          Enter to send · Shift+Enter for a new line
        </p>
      </div>
    </div>
  );
}
