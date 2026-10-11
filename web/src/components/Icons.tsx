/** Inline 24×24 stroke icons — avoids pulling in an icon package. */
type P = { className?: string };

const base = 'h-4 w-4';
const svg = (d: React.ReactNode, className?: string) => (
  <svg
    className={className ?? base}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    {d}
  </svg>
);

export const PlusIcon = ({ className }: P) => svg(<><path d="M12 5v14" /><path d="M5 12h14" /></>, className);

export const TrashIcon = ({ className }: P) =>
  svg(
    <>
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
    </>,
    className,
  );

export const ArchiveIcon = ({ className }: P) =>
  svg(
    <>
      <rect x="3" y="4" width="18" height="4" rx="1" />
      <path d="M5 8v11a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8" />
      <path d="M10 12h4" />
    </>,
    className,
  );

export const CopyIcon = ({ className }: P) =>
  svg(
    <>
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15V5a2 2 0 0 1 2-2h10" />
    </>,
    className,
  );

export const CheckIcon = ({ className }: P) => svg(<path d="M20 6L9 17l-5-5" />, className);

export const SendIcon = ({ className }: P) =>
  svg(<><path d="M12 19V5" /><path d="M5 12l7-7 7 7" /></>, className);

export const StopIcon = ({ className }: P) =>
  svg(<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />, className);

export const RetryIcon = ({ className }: P) =>
  svg(
    <>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </>,
    className,
  );

/** Sliders rather than a gear: the old gear path ran outside the 24 box, and
 *  sliders is the mark the brand sheet already uses for "customizable". */
export const SettingsIcon = ({ className }: P) =>
  svg(
    <>
      <path d="M4 7h8M18 7h2M4 12h2M12 12h8M4 17h8M18 17h2" />
      <circle cx="15" cy="7" r="2" />
      <circle cx="9" cy="12" r="2" />
      <circle cx="15" cy="17" r="2" />
    </>,
    className,
  );

export const SunIcon = ({ className }: P) =>
  svg(
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>,
    className,
  );

export const MoonIcon = ({ className }: P) =>
  svg(<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />, className);

export const ChartIcon = ({ className }: P) =>
  svg(
    <>
      <path d="M3 3v18h18" />
      <path d="M7 15l3-4 3 3 4-6" />
    </>,
    className,
  );

export const MenuIcon = ({ className }: P) =>
  svg(<><path d="M4 6h16" /><path d="M4 12h16" /><path d="M4 18h16" /></>, className);

export const CloseIcon = ({ className }: P) =>
  svg(<><path d="M18 6L6 18" /><path d="M6 6l12 12" /></>, className);

export const ChipIcon = ({ className }: P) =>
  svg(
    <>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
      <path d="M10 2v3M14 2v3M10 19v3M14 19v3M2 10h3M2 14h3M19 10h3M19 14h3" />
    </>,
    className,
  );

/* ── Brand vocabulary ─────────────────────────────────────
   The pillars from the brand sheet — private, powerful, self-hosted —
   drawn on the same 24-box grid and stroke weight as everything above. */

export const ChatIcon = ({ className }: P) =>
  svg(
    <>
      <path d="M20 15a2 2 0 0 1-2 2H8l-4 3V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z" />
      <path d="M8.5 10.5h.01M12 10.5h.01M15.5 10.5h.01" />
    </>,
    className,
  );

export const ShieldIcon = ({ className }: P) =>
  svg(<path d="M12 3l7 3v5.5c0 4.3-2.9 7.7-7 9.5-4.1-1.8-7-5.2-7-9.5V6z" />, className);

export const BoltIcon = ({ className }: P) => svg(<path d="M13.5 2L5 13.5h5.5L9.5 22 19 10.5h-5.5z" />, className);

export const ServerIcon = ({ className }: P) =>
  svg(
    <>
      <rect x="3" y="4" width="18" height="6" rx="1.5" />
      <rect x="3" y="14" width="18" height="6" rx="1.5" />
      <path d="M7 7h.01M7 17h.01" />
    </>,
    className,
  );
