import type { ReactNode } from 'react';

/**
 * Wraps the app with a slim bar saying this page is the demo build. It takes
 * its own row rather than floating, so it never covers anything in the app.
 */
export function DemoFrame({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full flex-col">
      <div
        role="note"
        className="flex shrink-0 items-center justify-center gap-2 border-b px-3 py-1 text-center text-[11.5px]"
        style={{
          borderColor: 'rgb(var(--border))',
          backgroundColor: 'rgb(var(--bg-subtle))',
          color: 'rgb(var(--text-muted))',
        }}
      >
        <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: 'rgb(var(--warn))' }} />
        <span className="truncate">
          <strong style={{ color: 'rgb(var(--text))' }}>Demo build.</strong> No GPU behind this page: replies are canned
          and timings are simulated.
        </span>
        <a
          href="https://github.com/shakaibarsalan/nexalm"
          target="_blank"
          rel="noreferrer"
          className="shrink-0 font-medium"
          style={{ color: 'rgb(var(--accent-text))' }}
        >
          Run your own ↗
        </a>
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}
