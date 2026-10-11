import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { config, isProd } from '../config.js';
import { query, withTransaction } from '../db/pool.js';

export const SESSION_COOKIE = 'llm_auth';
/** Pre-accounts cookie. Still read, never written: it identifies chats to adopt. */
export const LEGACY_SESSION_COOKIE = 'llm_session';
export const ACCESS_COOKIE = 'llm_access';

export type UserRole = 'admin' | 'user';

export interface AuthUser {
  id: string;
  email: string;
  displayName: string;
  avatarUrl: string | null;
  role: UserRole;
  createdAt: string;
  lastLoginAt: string | null;
}

interface UserRow {
  id: string;
  email: string;
  password_hash: string | null;
  display_name: string;
  avatar_url: string | null;
  role: UserRole;
  google_sub: string | null;
  created_at: Date;
  last_login_at: Date | null;
  disabled_at: Date | null;
}

export function toAuthUser(row: UserRow): AuthUser {
  return {
    id: row.id,
    email: row.email,
    displayName: row.display_name,
    avatarUrl: row.avatar_url,
    role: row.role,
    createdAt: row.created_at.toISOString(),
    lastLoginAt: row.last_login_at ? row.last_login_at.toISOString() : null,
  };
}

const USER_COLUMNS = `id, email, password_hash, display_name, avatar_url, role,
                      google_sub, created_at, last_login_at, disabled_at`;

/** Serialises the "am I the first user?" check in signup. */
const FIRST_USER_LOCK = 918_273_645;

export class EmailTakenError extends Error {
  constructor() {
    super('An account with that email already exists.');
    this.name = 'email_taken';
  }
}

// ── Users ─────────────────────────────────────────────────────

export async function findUserByEmail(email: string): Promise<UserRow | undefined> {
  const { rows } = await query<UserRow>(
    `SELECT ${USER_COLUMNS} FROM users WHERE lower(email) = lower($1)`,
    [email],
  );
  return rows[0];
}

export async function findUserByGoogleSub(sub: string): Promise<UserRow | undefined> {
  const { rows } = await query<UserRow>(`SELECT ${USER_COLUMNS} FROM users WHERE google_sub = $1`, [
    sub,
  ]);
  return rows[0];
}

export async function userCount(): Promise<number> {
  const { rows } = await query<{ count: number }>(`SELECT count(*)::int AS count FROM users`);
  return rows[0].count;
}

export interface NewUser {
  email: string;
  displayName: string;
  passwordHash?: string | null;
  googleSub?: string | null;
  avatarUrl?: string | null;
}

/**
 * Create an account, promoting the very first one to admin.
 *
 * The count and the insert share a transaction-scoped advisory lock because
 * two simultaneous first signups would otherwise both read zero and both come
 * out as admin.
 */
export async function createUser(input: NewUser): Promise<UserRow> {
  return withTransaction(async (client) => {
    await client.query(`SELECT pg_advisory_xact_lock($1::bigint)`, [FIRST_USER_LOCK]);
    const { rows: countRows } = await client.query<{ count: number }>(
      `SELECT count(*)::int AS count FROM users`,
    );
    const role: UserRole = countRows[0].count === 0 ? 'admin' : 'user';

    try {
      const { rows } = await client.query<UserRow>(
        `INSERT INTO users (email, password_hash, display_name, avatar_url, role, google_sub, last_login_at)
         VALUES ($1, $2, $3, $4, $5, $6, now())
         RETURNING ${USER_COLUMNS}`,
        [
          input.email,
          input.passwordHash ?? null,
          input.displayName,
          input.avatarUrl ?? null,
          role,
          input.googleSub ?? null,
        ],
      );
      return rows[0];
    } catch (err) {
      if ((err as { code?: string }).code === '23505') throw new EmailTakenError();
      throw err;
    }
  });
}

export async function markLogin(userId: string): Promise<void> {
  await query(`UPDATE users SET last_login_at = now() WHERE id = $1`, [userId]);
}

// ── Sessions ──────────────────────────────────────────────────

function hashToken(token: string): string {
  return createHash('sha256').update(token).digest('hex');
}

export interface ResolvedSession {
  user: AuthUser;
  sessionId: string;
}

/**
 * Mint a session. The raw token is returned once, for the cookie; only its
 * digest is persisted, so a leaked database dump cannot be replayed as a login.
 */
export async function createSession(userId: string, req: FastifyRequest): Promise<string> {
  const token = randomBytes(32).toString('base64url');
  await query(
    `INSERT INTO user_sessions (user_id, token_hash, expires_at, user_agent, ip)
     VALUES ($1, $2, now() + ($3 || ' days')::interval, $4, $5)`,
    [
      userId,
      hashToken(token),
      String(config.sessionTtlDays),
      (req.headers['user-agent'] ?? '').slice(0, 300),
      req.ip,
    ],
  );
  return token;
}

export async function resolveSession(token: string): Promise<ResolvedSession | undefined> {
  const { rows } = await query<UserRow & { session_id: string }>(
    `SELECT s.id AS session_id,
            u.id, u.email, u.password_hash, u.display_name, u.avatar_url, u.role,
            u.google_sub, u.created_at, u.last_login_at, u.disabled_at
       FROM user_sessions s
       JOIN users u ON u.id = s.user_id
      WHERE s.token_hash = $1
        AND s.expires_at > now()
        AND u.disabled_at IS NULL`,
    [hashToken(token)],
  );
  const row = rows[0];
  if (!row) return undefined;
  return { user: toAuthUser(row), sessionId: row.session_id };
}

/**
 * Best-effort activity stamp. Throttled in SQL so an active tab polling the
 * API does not turn every request into a write.
 */
export function touchSession(sessionId: string): void {
  void query(
    `UPDATE user_sessions SET last_seen_at = now()
      WHERE id = $1 AND last_seen_at < now() - interval '5 minutes'`,
    [sessionId],
  ).catch(() => {
    /* activity tracking is not worth failing a request over */
  });
}

export async function revokeSession(sessionId: string): Promise<void> {
  await query(`DELETE FROM user_sessions WHERE id = $1`, [sessionId]);
}

export async function revokeAllSessions(userId: string): Promise<number> {
  const { rowCount } = await query(`DELETE FROM user_sessions WHERE user_id = $1`, [userId]);
  return rowCount ?? 0;
}

/** Opportunistic cleanup, called from the login paths rather than on a timer. */
export function pruneExpiredSessions(): void {
  void query(`DELETE FROM user_sessions WHERE expires_at < now()`).catch(() => {
    /* the expiry predicate already keeps stale rows unusable */
  });
}

// ── Cookies ───────────────────────────────────────────────────

export function setSessionCookie(reply: FastifyReply, token: string): void {
  reply.setCookie(SESSION_COOKIE, token, {
    path: '/',
    httpOnly: true,
    secure: isProd,
    sameSite: 'lax',
    signed: true,
    maxAge: config.sessionTtlDays * 24 * 60 * 60,
  });
}

export function clearSessionCookie(reply: FastifyReply): void {
  reply.clearCookie(SESSION_COOKIE, { path: '/' });
}

/** Reads a signed cookie, returning undefined when absent or tampered with. */
export function readSignedCookie(req: FastifyRequest, name: string): string | undefined {
  const raw = req.cookies[name];
  if (!raw) return undefined;
  const unsigned = req.unsignCookie(raw);
  return unsigned.valid && unsigned.value ? unsigned.value : undefined;
}

export function constantTimeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

// ── Legacy data ───────────────────────────────────────────────

/**
 * Claim the chats a browser created before accounts existed.
 *
 * Guarded twice: only rows that nobody owns yet, and only when this account has
 * no chats of its own. Otherwise a second person signing up on a shared browser
 * would inherit the first person's history.
 */
export async function adoptLegacyConversations(
  userId: string,
  legacySessionId: string,
): Promise<number> {
  if (!legacySessionId) return 0;
  const { rowCount } = await query(
    `UPDATE conversations SET user_id = $1
      WHERE session_id = $2
        AND user_id IS NULL
        AND NOT EXISTS (SELECT 1 FROM conversations WHERE user_id = $1)`,
    [userId, legacySessionId],
  );
  return rowCount ?? 0;
}
