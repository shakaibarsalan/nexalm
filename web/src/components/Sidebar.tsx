import { useState } from 'react';
import type { AuthUser, Conversation } from '../lib/types';
import { LogoWordmark } from './Logo';
import { ArchiveIcon, CloseIcon, PlusIcon, SettingsIcon, TrashIcon } from './Icons';

function groupByAge(items: Conversation[]): Array<[string, Conversation[]]> {
  const now = Date.now();
  const day = 86_400_000;
  const buckets: Record<string, Conversation[]> = {
    Today: [],
    Yesterday: [],
    'Previous 7 days': [],
    Older: [],
  };
  for (const c of items) {
    const age = now - new Date(c.updated_at).getTime();
    if (age < day) buckets['Today'].push(c);
    else if (age < 2 * day) buckets['Yesterday'].push(c);
    else if (age < 7 * day) buckets['Previous 7 days'].push(c);
    else buckets['Older'].push(c);
  }
  return Object.entries(buckets).filter(([, v]) => v.length > 0);
}

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onArchive: (id: string) => void;
  onOpenSettings: () => void;
  user: AuthUser | null;
  onSignOut: () => void;
}

export function Sidebar({
  conversations,
  activeId,
  open,
  onClose,
  onSelect,
  onNew,
  onDelete,
  onArchive,
  onOpenSettings,
  user,
  onSignOut,
}: Props) {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const groups = groupByAge(conversations);

  return (
    <>
      {/* Mobile scrim */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r transition-transform duration-200
                    md:static md:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}
        style={{ backgroundColor: 'rgb(var(--bg-elevated))', borderColor: 'rgb(var(--border))' }}
      >
        <div className="flex items-center justify-between px-3 pb-1 pt-3">
          <LogoWordmark size={24} />
        </div>

        <div className="flex items-center gap-2 p-3 pt-2">
          <button onClick={onNew} className="btn btn-primary flex-1">
            <PlusIcon />
            New chat
          </button>
          <button onClick={onClose} className="btn btn-ghost px-2 md:hidden" aria-label="Close sidebar">
            <CloseIcon />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2 pb-2">
          {conversations.length === 0 && (
            <p className="px-3 py-8 text-center text-sm" style={{ color: 'rgb(var(--text-faint))' }}>
              No conversations yet.
            </p>
          )}

          {groups.map(([label, items]) => (
            <div key={label} className="mb-3">
              <h2
                className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider"
                style={{ color: 'rgb(var(--text-faint))' }}
              >
                {label}
              </h2>
              <ul className="space-y-0.5">
                {items.map((c) => {
                  const active = c.id === activeId;
                  return (
                    <li key={c.id}>
                      <div
                        className="group flex items-center gap-1 rounded-lg pr-1 transition-colors"
                        style={active ? { backgroundColor: 'rgb(var(--bg-subtle))' } : undefined}
                      >
                        <button
                          onClick={() => onSelect(c.id)}
                          className="min-w-0 flex-1 truncate px-3 py-2 text-left text-sm"
                          style={{ color: active ? 'rgb(var(--text))' : 'rgb(var(--text-muted))' }}
                          title={c.title}
                        >
                          {c.title}
                        </button>

                        <div className="flex shrink-0 items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
                          <button
                            onClick={() => onArchive(c.id)}
                            className="btn btn-ghost p-1.5"
                            aria-label="Archive conversation"
                            title="Archive"
                          >
                            <ArchiveIcon className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={() => setConfirmId(c.id)}
                            className="btn btn-ghost p-1.5"
                            aria-label="Delete conversation"
                            title="Delete"
                          >
                            <TrashIcon className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>

                      {confirmId === c.id && (
                        <div
                          className="mx-2 mb-1 mt-1 rounded-lg border p-2 text-xs"
                          style={{ borderColor: 'rgb(var(--border))' }}
                        >
                          <p className="mb-2" style={{ color: 'rgb(var(--text-muted))' }}>
                            Delete this conversation permanently?
                          </p>
                          <div className="flex gap-2">
                            <button
                              className="btn px-2 py-1 text-xs text-white"
                              style={{ backgroundColor: 'rgb(var(--danger))' }}
                              onClick={() => {
                                onDelete(c.id);
                                setConfirmId(null);
                              }}
                            >
                              Delete
                            </button>
                            <button className="btn btn-ghost px-2 py-1 text-xs" onClick={() => setConfirmId(null)}>
                              Cancel
                            </button>
                          </div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>

        <div className="border-t p-2" style={{ borderColor: 'rgb(var(--border))' }}>
          <button onClick={onOpenSettings} className="btn btn-ghost w-full justify-start">
            <SettingsIcon />
            Settings
          </button>

          {user && (
            <div
              className="mt-1 flex items-center gap-2 rounded-lg px-2 py-2"
              style={{ backgroundColor: 'rgb(var(--bg-subtle))' }}
            >
              <span
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold text-white"
                style={{ backgroundColor: 'rgb(var(--accent))' }}
              >
                {(user.displayName || user.email).slice(0, 1).toUpperCase()}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-xs font-medium">
                  {user.displayName || user.email}
                </span>
                <span className="block text-[10px]" style={{ color: 'rgb(var(--text-faint))' }}>
                  {user.role === 'admin' ? 'Administrator' : 'Member'}
                </span>
              </span>
              <button
                onClick={onSignOut}
                className="btn btn-ghost px-2 py-1 text-[11px]"
                title="Sign out"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
