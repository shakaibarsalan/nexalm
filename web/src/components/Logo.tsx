import { useId } from 'react';

/**
 * The NexaLM mark: a rounded hexagon holding an N, drawn in the brand's
 * cyan → blue → violet run.
 *
 * Inline SVG rather than the source PNG so it stays crisp at any size and
 * needs no light/dark variant — the gradient carries both grounds.
 */

/** Pointy-top hexagon, R=13.8 about (16,16); the 3px stroke stays inside 0–32. */
const HEX = 'M16 2.2 27.95 9.1v13.8L16 29.8 4.05 22.9V9.1Z';
/** The N: stems inset from the hexagon's flanks, one diagonal between them. */
const N = 'M10.4 22.2V9.8l11.2 12.4V9.8';

interface MarkProps {
  /** Rendered box in px; the artwork scales with it. */
  size?: number;
  className?: string;
  /**
   * The three chat dots live in the N's counter, so they are only a few
   * hundredths of the box wide — below ~40px they turn to mush. Off by
   * default at small sizes, on for hero lockups.
   */
  detail?: boolean;
  /** Supply only when the mark is the sole label for something. */
  title?: string;
}

export function LogoMark({ size = 28, className, detail, title }: MarkProps) {
  // The mark can appear twice on a page (header + empty state); a shared
  // gradient id would let the first instance win and blank the second.
  const gradientId = `nexalm-${useId().replace(/:/g, '')}`;
  const url = `url(#${gradientId})`;
  const showDots = detail ?? size >= 40;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#22D3EE" />
          <stop offset="48%" stopColor="#3B82F6" />
          <stop offset="100%" stopColor="#8B5CF6" />
        </linearGradient>
      </defs>

      {/* A whisper of fill keeps the mark from reading as a wire outline on white. */}
      <path d={HEX} fill={url} opacity="0.12" />
      <path d={HEX} stroke={url} strokeWidth="3" strokeLinejoin="round" />
      <path d={N} stroke={url} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />

      {showDots && (
        <g fill={url}>
          <circle cx="15.3" cy="11.9" r="0.85" />
          <circle cx="17.3" cy="11.9" r="0.85" />
          <circle cx="19.3" cy="11.9" r="0.85" />
        </g>
      )}
    </svg>
  );
}

interface WordmarkProps {
  /** Height of the mark in px; the wordmark is sized from it. */
  size?: number;
  className?: string;
  showTagline?: boolean;
}

/** Mark + "NexaLM", with "LM" carrying the gradient exactly as the brand sheet does. */
export function LogoWordmark({ size = 30, className, showTagline = false }: WordmarkProps) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className ?? ''}`}>
      <LogoMark size={size} />
      <span className="flex flex-col">
        <span
          className="font-bold leading-none tracking-tight"
          // Tied to the mark so the lockup keeps its proportions at any scale.
          style={{ fontSize: Math.round(size * 0.62) }}
        >
          Nexa<span className="brand-gradient-text">LM</span>
        </span>
        {showTagline && (
          <span
            className="mt-1.5 text-[10px] font-medium uppercase leading-none tracking-[0.18em]"
            style={{ color: 'rgb(var(--text-faint))' }}
          >
            Your AI. Your Control.
          </span>
        )}
      </span>
    </span>
  );
}
