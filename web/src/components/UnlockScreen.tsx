import { useState } from 'react';
import { api, ApiError } from '../lib/api';

export function UnlockScreen({ onUnlocked }: { onUnlocked: () => void }) {
  const [token, setToken] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.unlock(token);
      onUnlocked();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach the server.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center px-6">
      <form onSubmit={submit} className="w-full max-w-sm animate-fade-in">
        <div className="mb-8 text-center">
          <div
            className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl text-lg font-bold text-white"
            style={{ backgroundColor: 'rgb(var(--accent))' }}
          >
            LL
          </div>
          <h1 className="text-xl font-semibold">Local LLM</h1>
          <p className="mt-1 text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
            Enter the access token to continue.
          </p>
        </div>

        <input
          type="password"
          className="input"
          placeholder="Access token"
          value={token}
          autoFocus
          autoComplete="current-password"
          onChange={(e) => setToken(e.target.value)}
        />

        {error && (
          <p className="mt-3 text-sm" style={{ color: 'rgb(var(--danger))' }}>
            {error}
          </p>
        )}

        <button type="submit" className="btn btn-primary mt-4 w-full" disabled={busy || !token}>
          {busy ? 'Unlocking…' : 'Unlock'}
        </button>
      </form>
    </div>
  );
}
