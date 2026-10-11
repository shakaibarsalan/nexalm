/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'JetBrains Mono', 'Menlo', 'monospace'],
      },
      // Themed in index.css so a shadow never has to be swapped per mode.
      boxShadow: {
        soft: 'var(--shadow-1)',
        elevated: 'var(--shadow-2)',
        float: 'var(--shadow-3)',
        glow: '0 0 0 1px rgb(var(--accent) / 0.35), 0 10px 30px -12px rgb(var(--accent) / 0.55)',
      },
      backgroundImage: {
        brand: 'linear-gradient(135deg, rgb(var(--brand-1)), rgb(var(--brand-2)) 48%, rgb(var(--brand-3)))',
      },
      // Motion is restrained on purpose: short, eased-out, no overshoot.
      keyframes: {
        'fade-in': { '0%': { opacity: '0', transform: 'translateY(4px)' }, '100%': { opacity: '1', transform: 'none' } },
        'slide-up': { '0%': { opacity: '0', transform: 'translateY(10px)' }, '100%': { opacity: '1', transform: 'none' } },
        'scale-in': { '0%': { opacity: '0', transform: 'scale(0.97)' }, '100%': { opacity: '1', transform: 'none' } },
        blink: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0' } },
        // For skeletons: a pass of light across a placeholder, not a pulse.
        shimmer: { '0%': { backgroundPosition: '-160% 0' }, '100%': { backgroundPosition: '260% 0' } },
        // Breathing halo for "worker live" style indicators.
        'pulse-glow': {
          '0%,100%': { boxShadow: '0 0 0 0 rgb(var(--accent) / 0.45)' },
          '70%': { boxShadow: '0 0 0 6px rgb(var(--accent) / 0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.18s cubic-bezier(0.2, 0, 0, 1)',
        'slide-up': 'slide-up 0.24s cubic-bezier(0.2, 0, 0, 1)',
        'scale-in': 'scale-in 0.16s cubic-bezier(0.2, 0, 0, 1)',
        blink: 'blink 1.1s step-end infinite',
        shimmer: 'shimmer 1.6s linear infinite',
        'pulse-glow': 'pulse-glow 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      transitionTimingFunction: {
        emphasis: 'cubic-bezier(0.2, 0, 0, 1)',
      },
    },
  },
  plugins: [],
};
