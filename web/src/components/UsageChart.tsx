import { useMemo } from 'react';
import type { WorkerSample } from '../lib/types';

/**
 * Compact area chart for GPU telemetry.
 *
 * Hand-drawn SVG rather than a charting library: the whole panel needs three
 * small sparklines, and pulling in a chart package would cost more bundle than
 * the entire rest of the app. Also keeps it theme-aware for free, since it
 * inherits the CSS custom properties.
 */

interface SeriesProps {
  label: string;
  unit: string;
  values: Array<number | null>;
  /** Fixed upper bound; without one the axis is scaled to the data. */
  max?: number;
  color: string;
  height?: number;
}

function Sparkline({ label, unit, values, max, color, height = 44 }: SeriesProps) {
  const points = values.filter((v): v is number => v != null);

  const { path, area, latest, peak, upper } = useMemo(() => {
    if (points.length === 0) {
      return { path: '', area: '', latest: null, peak: null, upper: max ?? 100 };
    }
    const peakValue = Math.max(...points);
    // Round the auto-scaled axis up so the line never touches the top edge.
    const bound = max ?? Math.max(1, Math.ceil((peakValue * 1.15) / 10) * 10);
    const width = 100;
    const step = points.length > 1 ? width / (points.length - 1) : width;

    const coords = points.map((v, i) => {
      const x = i * step;
      const y = height - (Math.min(v, bound) / bound) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    return {
      path: `M ${coords.join(' L ')}`,
      area: `M 0,${height} L ${coords.join(' L ')} L ${width},${height} Z`,
      latest: points[points.length - 1],
      peak: peakValue,
      upper: bound,
    };
  }, [points, max, height]);

  const gradientId = `grad-${label.replace(/\W/g, '')}`;

  return (
    <div className="mb-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs" style={{ color: 'rgb(var(--text-muted))' }}>
          {label}
        </span>
        <span className="font-mono text-xs" style={{ color: 'rgb(var(--text))' }}>
          {latest != null ? `${Math.round(latest)}${unit}` : '—'}
          {peak != null && (
            <span className="ml-1.5" style={{ color: 'rgb(var(--text-faint))' }}>
              peak {Math.round(peak)}
              {unit}
            </span>
          )}
        </span>
      </div>

      {points.length < 2 ? (
        <div
          className="flex items-center justify-center rounded border text-[11px]"
          style={{ height, borderColor: 'rgb(var(--border))', color: 'rgb(var(--text-faint))' }}
        >
          collecting…
        </div>
      ) : (
        <svg
          viewBox={`0 0 100 ${height}`}
          preserveAspectRatio="none"
          className="w-full rounded border"
          style={{ height, borderColor: 'rgb(var(--border))' }}
          role="img"
          aria-label={`${label}: currently ${latest ?? 0}${unit}, peak ${peak ?? 0}${unit}`}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.35" />
              <stop offset="100%" stopColor={color} stopOpacity="0.02" />
            </linearGradient>
          </defs>
          {/* Mid-line for scale, so a flat trace still reads as "low" not "empty". */}
          <line
            x1="0"
            y1={height / 2}
            x2="100"
            y2={height / 2}
            stroke="currentColor"
            strokeWidth="0.3"
            strokeDasharray="2 2"
            style={{ color: 'rgb(var(--border))' }}
          />
          <path d={area} fill={`url(#${gradientId})`} />
          <path
            d={path}
            fill="none"
            stroke={color}
            strokeWidth="1.2"
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
          />
        </svg>
      )}
      <div className="mt-0.5 text-right text-[10px]" style={{ color: 'rgb(var(--text-faint))' }}>
        0–{Math.round(upper)}
        {unit}
      </div>
    </div>
  );
}

export function UsageChart({
  samples,
  windowMinutes,
}: {
  samples: WorkerSample[];
  windowMinutes: number;
}) {
  // Samples arrive per GPU; group so each card gets its own trace.
  const byGpu = useMemo(() => {
    const map = new Map<number, WorkerSample[]>();
    for (const s of samples) {
      const list = map.get(s.gpu_index) ?? [];
      list.push(s);
      map.set(s.gpu_index, list);
    }
    return [...map.entries()].sort((a, b) => a[0] - b[0]);
  }, [samples]);

  if (samples.length === 0) {
    return (
      <p className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
        No telemetry yet — it appears once a worker starts sending heartbeats.
      </p>
    );
  }

  return (
    <div>
      <p className="mb-2 text-[11px]" style={{ color: 'rgb(var(--text-faint))' }}>
        Last {windowMinutes} minutes
      </p>
      {byGpu.map(([index, rows]) => {
        const totalMb = rows.find((r) => r.mem_total_mb)?.mem_total_mb ?? null;
        return (
          <div key={index} className="mb-4">
            <p className="mb-1.5 text-[11px] font-semibold" style={{ color: 'rgb(var(--text-muted))' }}>
              GPU {index}
            </p>
            <Sparkline
              label="Utilisation"
              unit="%"
              max={100}
              color="rgb(96,152,255)"
              values={rows.map((r) => r.util_pct)}
            />
            <Sparkline
              label="VRAM used"
              unit=" GB"
              max={totalMb ? totalMb / 1024 : undefined}
              color="rgb(74,222,128)"
              values={rows.map((r) => (r.mem_used_mb != null ? r.mem_used_mb / 1024 : null))}
            />
            <Sparkline
              label="Temperature"
              unit="°C"
              max={100}
              color="rgb(251,191,36)"
              values={rows.map((r) => r.temp_c)}
            />
          </div>
        );
      })}
    </div>
  );
}
