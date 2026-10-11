import { useCallback, useEffect, useState } from 'react';
import { api, ApiError } from '../lib/api';
import type { AdminUser, AuthUser } from '../lib/types';

function fmtDate(iso: string | null): string {
  if (!iso) return 'never';
  const d = new Date(iso);
  const days = Math.floor((Date.now() - d.getTime()) / 86_400_000);
  if (days === 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 30) return `${days}d ago`;
  return d.toLocaleDateString();
}

/**
 * Admin-only account management.
 *
 * The server already refuses self-escalation, self-disable and removing the
 * last admin; the UI mirrors those rules by disabling the controls so an admin
 * is not invited to attempt something that will only come back as an error.
 */
export function UsersPanel({ me }: { me: AuthUser }) {
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setUsers(await api.listUsers());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not load users.');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const act = useCallback(
    async (id: string, fn: () => Promise<unknown>) => {
      setBusyId(id);
      setError(null);
      try {
        await fn();
        await load();
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Action failed.');
      } finally {
        setBusyId(null);
        setConfirmDelete(null);
      }
    },
    [load],
  );

  if (error && !users) {
    return (
      <p className="text-sm" style={{ color: 'rgb(var(--danger))' }}>
        {error}
      </p>
    );
  }
  if (!users) {
    return (
      <p className="text-sm" style={{ color: 'rgb(var(--text-muted))' }}>
        Loading…
      </p>
    );
  }

  const adminCount = users.filter((u) => u.role === 'admin' && !u.disabledAt).length;

  return (
    <div>
      {error && (
        <p className="mb-3 text-sm" style={{ color: 'rgb(var(--danger))' }}>
          {error}
        </p>
      )}

      <ul className="space-y-2">
        {users.map((u) => {
          const isSelf = u.id === me.id;
          const isLastAdmin = u.role === 'admin' && adminCount <= 1;
          const busy = busyId === u.id;

          return (
            <li
              key={u.id}
              className="rounded-lg border p-3"
              style={{
                borderColor: 'rgb(var(--border))',
                opacity: u.disabledAt ? 0.6 : 1,
              }}
            >
              <div className="flex items-start gap-3">
                <span
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold text-white"
                  style={{ backgroundColor: 'rgb(var(--accent))' }}
                >
                  {(u.displayName || u.email).slice(0, 1).toUpperCase()}
                </span>

                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">
                    {u.displayName || u.email}
                    {isSelf && (
                      <span className="ml-1.5 text-xs" style={{ color: 'rgb(var(--text-faint))' }}>
                        (you)
                      </span>
                    )}
                  </p>
                  <p className="truncate text-xs" style={{ color: 'rgb(var(--text-muted))' }}>
                    {u.email}
                  </p>
                  <p className="mt-1 text-[11px]" style={{ color: 'rgb(var(--text-faint))' }}>
                    {u.conversationCount} chat{u.conversationCount === 1 ? '' : 's'} ·{' '}
                    {u.activeSessions} active session{u.activeSessions === 1 ? '' : 's'} · last seen{' '}
                    {fmtDate(u.lastLoginAt)}
                    {u.googleLinked ? ' · Google' : ''}
                    {u.disabledAt ? ' · disabled' : ''}
                  </p>
                </div>

                <select
                  className="input w-auto py-1 text-xs"
                  value={u.role}
                  disabled={isSelf || busy}
                  title={isSelf ? 'You cannot change your own role.' : 'Change role'}
                  onChange={(e) =>
                    act(u.id, () => api.updateUser(u.id, { role: e.target.value as 'admin' | 'user' }))
                  }
                >
                  <option value="user">Member</option>
                  <option value="admin">Admin</option>
                </select>
              </div>

              <div className="mt-2 flex justify-end gap-2">
                <button
                  className="btn btn-ghost px-2 py-1 text-xs"
                  disabled={isSelf || busy || (isLastAdmin && !u.disabledAt)}
                  title={
                    isSelf
                      ? 'You cannot disable your own account.'
                      : isLastAdmin && !u.disabledAt
                        ? 'This is the only admin.'
                        : undefined
                  }
                  onClick={() => act(u.id, () => api.updateUser(u.id, { disabled: !u.disabledAt }))}
                >
                  {u.disabledAt ? 'Re-enable' : 'Disable'}
                </button>

                {confirmDelete === u.id ? (
                  <>
                    <button
                      className="btn px-2 py-1 text-xs text-white"
                      style={{ backgroundColor: 'rgb(var(--danger))' }}
                      disabled={busy}
                      onClick={() => act(u.id, () => api.deleteUser(u.id))}
                    >
                      Delete permanently
                    </button>
                    <button
                      className="btn btn-ghost px-2 py-1 text-xs"
                      onClick={() => setConfirmDelete(null)}
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <button
                    className="btn btn-ghost px-2 py-1 text-xs"
                    style={{ color: 'rgb(var(--danger))' }}
                    disabled={isSelf || busy || isLastAdmin}
                    title={isSelf ? 'You cannot delete your own account.' : undefined}
                    onClick={() => setConfirmDelete(u.id)}
                  >
                    Delete
                  </button>
                )}
              </div>

              {confirmDelete === u.id && (
                <p className="mt-2 text-[11px]" style={{ color: 'rgb(var(--danger))' }}>
                  This also deletes their {u.conversationCount} conversation
                  {u.conversationCount === 1 ? '' : 's'} and cannot be undone.
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
