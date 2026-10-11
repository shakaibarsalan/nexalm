import { useEffect, useState } from 'react';
import { api, ApiError } from '../lib/api';
import type { AuthStatus } from '../lib/types';
import { LogoWordmark } from './Logo';
import { BoltIcon, ChatIcon, ServerIcon, ShieldIcon } from './Icons';

/** The brand pillars, used as the marketing rail beside the form. */
const PILLARS = [
  { Icon: ChatIcon, title: 'Chat', body: 'A familiar interface over your own model.' },
  { Icon: ShieldIcon, title: 'Private', body: 'Conversations stay on your infrastructure.' },
  { Icon: BoltIcon, title: 'Fast', body: 'Streaming responses, measured every turn.' },
  { Icon: ServerIcon, title: 'Self-hosted', body: 'Your GPU, your rules, no per-token bill.' },
];

interface Props {
  status: AuthStatus;
  onAuthenticated: () => void;
}

/**
 * The same rules the server enforces, shown live rather than after the fact.
 *
 * Duplicated deliberately: the server is the authority and rejects anything
 * that slips past, but a form that only says "too weak" once you submit makes
 * people guess which part was wrong, and they usually guess the length.
 */
const PASSWORD_RULES: { id: string; label: string; test: (v: string) => boolean }[] = [
  { id: 'length', label: 'At least 10 characters', test: (v) => v.length >= 10 },
  { id: 'lower', label: 'A lowercase letter', test: (v) => /[a-z]/.test(v) },
  { id: 'upper', label: 'An uppercase letter', test: (v) => /[A-Z]/.test(v) },
  { id: 'digit', label: 'A number', test: (v) => /\d/.test(v) },
  { id: 'symbol', label: 'A symbol', test: (v) => /[^A-Za-z0-9]/.test(v) },
];

export function AuthScreen({ status, onAuthenticated }: Props) {
  // The very first visit has no accounts at all, so lead with signup rather
  // than a login form nobody can satisfy yet.
  const [mode, setMode] = useState<'login' | 'signup'>(status.setupRequired ? 'signup' : 'login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Set when sign-up returns 202: the account does not exist yet and the screen
  // switches to asking for the code rather than navigating away.
  const [awaitingCode, setAwaitingCode] = useState<string | null>(null);
  const [code, setCode] = useState('');
  const [notice, setNotice] = useState<string | null>(null);

  // OAuth failures come back as a query parameter on the redirect.
  useEffect(() => {
    const code = new URLSearchParams(window.location.search).get('auth_error');
    if (!code) return;
    const reasons: Record<string, string> = {
      access_denied: 'Google sign-in was cancelled.',
      signup_disabled: 'That Google account has no access here, and registration is closed.',
      account_disabled: 'That account has been disabled.',
      state_mismatch: 'Google sign-in expired — please try again.',
    };
    setError(reasons[code] ?? `Google sign-in failed (${code}).`);
    window.history.replaceState({}, '', window.location.pathname);
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === 'signup') {
        if (password !== confirm) {
          setError('The two passwords do not match.');
          return;
        }
        const res = await api.signup(email, password, displayName || email.split('@')[0]);
        if (res.verificationRequired) {
          setAwaitingCode(res.email ?? email);
          setNotice(res.message ?? `We sent a 6-digit code to ${email}.`);
          return;
        }
      } else {
        await api.login(email, password);
      }
      onAuthenticated();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach the server.');
    } finally {
      setBusy(false);
    }
  }

  async function submitCode(e: React.FormEvent) {
    e.preventDefault();
    if (!awaitingCode) return;
    setBusy(true);
    setError(null);
    try {
      await api.verifyEmail(awaitingCode, code.trim());
      onAuthenticated();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach the server.');
    } finally {
      setBusy(false);
    }
  }

  async function resend() {
    if (!awaitingCode) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.resendCode(awaitingCode);
      setNotice(`A new code is on its way. ${3 - res.sentThisHour} sends left this hour.`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach the server.');
    } finally {
      setBusy(false);
    }
  }

  const isSignup = mode === 'signup';
  const ruleState = PASSWORD_RULES.map((r) => ({ ...r, ok: r.test(password) }));
  const passwordReady = ruleState.every((r) => r.ok);
  const confirmMismatch = confirm.length > 0 && confirm !== password;
  const canSubmit = !isSignup || (passwordReady && confirm === password && confirm.length > 0);

  return (
    <div className="flex min-h-full">
      {/* Brand rail. Hidden below lg so the form never competes for space. */}
      <aside
        className="relative hidden w-[46%] flex-col justify-between overflow-hidden p-10 lg:flex"
        style={{ backgroundColor: 'rgb(var(--bg-elevated))' }}
      >
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage:
              'radial-gradient(38rem 24rem at 15% 0%, rgb(var(--brand-1) / 0.14), transparent 65%),' +
              'radial-gradient(34rem 22rem at 90% 100%, rgb(var(--brand-3) / 0.16), transparent 65%)',
          }}
        />
        <div className="relative">
          <LogoWordmark size={40} showTagline />
        </div>

        <ul className="relative grid gap-5">
          {PILLARS.map(({ Icon, title, body }) => (
            <li key={title} className="flex gap-3">
              <span
                className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
                style={{ backgroundColor: 'rgb(var(--accent-soft))', color: 'rgb(var(--accent-text))' }}
              >
                <Icon className="h-4 w-4" />
              </span>
              <span>
                <span className="block text-sm font-semibold">{title}</span>
                <span className="block text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
                  {body}
                </span>
              </span>
            </li>
          ))}
        </ul>

        <p className="relative text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
          PRIVATE · POWERFUL · LIMITLESS
        </p>
      </aside>

      <main className="flex flex-1 items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm animate-fade-in">
          <div className="mb-8 lg:hidden">
            <LogoWordmark size={34} />
          </div>

          {awaitingCode ? (
            <>
              <h1 className="text-xl font-semibold">Check your email</h1>
              <p className="mt-1 text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
                We sent a 6-digit code to <strong>{awaitingCode}</strong>. Enter it to finish
                creating your account.
              </p>

              <form onSubmit={submitCode} className="mt-6">
                <label className="block">
                  <span className="mb-1 block text-sm font-medium">Verification code</span>
                  <input
                    className="input text-center text-lg tracking-[0.4em]"
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    maxLength={6}
                    placeholder="000000"
                    value={code}
                    onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    autoFocus
                  />
                </label>

                {notice && !error && (
                  <p className="mt-3 text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
                    {notice}
                  </p>
                )}
                {error && (
                  <p className="mt-3 text-sm" style={{ color: 'rgb(var(--danger))' }}>
                    {error}
                  </p>
                )}

                <button
                  type="submit"
                  className="btn btn-primary mt-5 w-full"
                  disabled={busy || code.length !== 6}
                >
                  {busy ? 'Checking…' : 'Verify and continue'}
                </button>
              </form>

              <div className="mt-5 flex items-center justify-between text-sm">
                <button type="button" onClick={resend} disabled={busy}
                  className="font-medium" style={{ color: 'rgb(var(--accent-text))' }}>
                  Send another code
                </button>
                <button type="button" style={{ color: 'rgb(var(--text-muted))' }}
                  onClick={() => { setAwaitingCode(null); setCode(''); setNotice(null); setError(null); }}>
                  Use a different address
                </button>
              </div>
            </>
          ) : (
            <>
          <h1 className="text-xl font-semibold">
            {status.setupRequired
              ? 'Create the first account'
              : isSignup
                ? 'Create your account'
                : 'Welcome back'}
          </h1>
          <p className="mt-1 text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
            {status.setupRequired
              ? 'This account becomes the administrator.'
              : isSignup
                ? 'Your chats stay private to your account.'
                : 'Sign in to continue your conversations.'}
          </p>

          {status.googleEnabled && (
            <>
              <a
                href="/api/auth/google"
                className="btn mt-6 w-full border"
                style={{
                  borderColor: 'rgb(var(--border))',
                  backgroundColor: 'rgb(var(--bg-elevated))',
                  color: 'rgb(var(--text))',
                }}
              >
                <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
                  <path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5a5.6 5.6 0 0 1-2.4 3.7v3h3.9c2.3-2.1 3.5-5.2 3.5-8.9z" />
                  <path fill="#34A853" d="M12 24c3.2 0 5.9-1.1 7.9-2.9l-3.9-3c-1.1.7-2.4 1.2-4 1.2-3.1 0-5.7-2.1-6.6-4.9H1.4v3.1A12 12 0 0 0 12 24z" />
                  <path fill="#FBBC05" d="M5.4 14.4a7.2 7.2 0 0 1 0-4.6V6.7H1.4a12 12 0 0 0 0 10.8z" />
                  <path fill="#EA4335" d="M12 4.8c1.8 0 3.4.6 4.6 1.8l3.4-3.4A12 12 0 0 0 1.4 6.7l4 3.1C6.3 6.9 8.9 4.8 12 4.8z" />
                </svg>
                Continue with Google
              </a>

              <div className="my-5 flex items-center gap-3">
                <span className="h-px flex-1" style={{ backgroundColor: 'rgb(var(--border))' }} />
                <span className="text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                  or
                </span>
                <span className="h-px flex-1" style={{ backgroundColor: 'rgb(var(--border))' }} />
              </div>
            </>
          )}

          <form onSubmit={submit} className={status.googleEnabled ? '' : 'mt-6'}>
            {isSignup && (
              <label className="mb-3 block">
                <span className="mb-1 block text-sm font-medium">Name</span>
                <input
                  className="input"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="Shakaib"
                  autoComplete="name"
                />
              </label>
            )}

            <label className="mb-3 block">
              <span className="mb-1 block text-sm font-medium">Email</span>
              <input
                className="input"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                autoFocus
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-sm font-medium">Password</span>
              <input
                className="input"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={isSignup ? 'At least 10 characters' : '••••••••'}
                autoComplete={isSignup ? 'new-password' : 'current-password'}
              />
            </label>

            {isSignup && password.length > 0 && (
              <ul className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1">
                {ruleState.map((r) => (
                  <li
                    key={r.id}
                    className="flex items-center gap-1.5 text-xs"
                    style={{ color: r.ok ? 'rgb(var(--success))' : 'rgb(var(--text-faint))' }}
                  >
                    <span aria-hidden="true" className="w-3 text-center">
                      {r.ok ? '✓' : '·'}
                    </span>
                    {r.label}
                  </li>
                ))}
              </ul>
            )}

            {isSignup && (
              <label className="mt-3 block">
                <span className="mb-1 block text-sm font-medium">Retype password</span>
                <input
                  className="input"
                  type="password"
                  required
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="new-password"
                  aria-invalid={confirmMismatch}
                />
                {confirmMismatch && (
                  <span className="mt-1 block text-xs" style={{ color: 'rgb(var(--danger))' }}>
                    These do not match yet.
                  </span>
                )}
              </label>
            )}

            {error && (
              <p className="mt-3 text-sm" style={{ color: 'rgb(var(--danger))' }}>
                {error}
              </p>
            )}

            <button
              type="submit"
              className="btn btn-primary mt-5 w-full"
              disabled={busy || !canSubmit}
            >
              {busy ? 'Please wait…' : isSignup ? 'Create account' : 'Sign in'}
            </button>
          </form>
            </>
          )}

          {!awaitingCode && !status.setupRequired && (status.signupEnabled || !isSignup) && (
            <p className="mt-5 text-center text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
              {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
              <button
                className="font-medium underline underline-offset-2"
                style={{ color: 'rgb(var(--accent-text))' }}
                onClick={() => {
                  setMode(isSignup ? 'login' : 'signup');
                  setError(null);
                }}
              >
                {isSignup ? 'Sign in' : 'Sign up'}
              </button>
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
