import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { config } from '../config.js';
import {
  ACCESS_COOKIE,
  LEGACY_SESSION_COOKIE,
  SESSION_COOKIE,
  readSignedCookie,
  resolveSession,
  touchSession,
} from '../services/authService.js';
import type { AuthUser } from '../services/authService.js';

/** Reachable without a signed-in user. Everything else under /api/ needs one. */
const PUBLIC_PATHS = new Set([
  '/api/health',
  '/api/auth/status',
  '/api/auth/signup',
  '/api/auth/login',
  '/api/auth/logout',
  '/api/auth/unlock',
  '/api/auth/verify-email',
  '/api/auth/resend-code',
  '/api/auth/lock',
  '/api/auth/google',
  '/api/auth/google/callback',
  // Worker bootstrap: authenticated by WORKER_TOKEN inside the route, since the
  // caller is a GPU host with no browser and no account.
  '/api/artifacts',
  // Authenticated by WORKER_TOKEN, not a session: the caller is a worker
  // that failed before it could connect and has no account.
  '/api/workers/report',
]);

declare module 'fastify' {
  interface FastifyRequest {
    /** The signed-in account, or null on public routes. */
    user: AuthUser | null;
    /** Row id in user_sessions, for logout. */
    authSessionId: string;
    /**
     * Value of the pre-accounts `llm_session` cookie, when the browser still
     * has one. Empty otherwise — and empty never matches a stored session_id.
     */
    legacySessionId: string;
  }
}

export class UnauthenticatedError extends Error {
  statusCode = 401;
  constructor() {
    super('Sign in required.');
    this.name = 'unauthenticated';
  }
}

/**
 * Narrow `req.user` for routes the gate has already protected. The throw is a
 * guard against a route being added to PUBLIC_PATHS by accident, not an
 * expected path.
 */
/**
 * Returns the caller when they are an admin, otherwise replies 403.
 *
 * Lives here rather than in routes/users.ts because it is not only the user
 * routes that need it: starting a GPU session spends a finite weekly quota,
 * and that was reachable by any signed-in account.
 */
export function requireAdmin(req: FastifyRequest, reply: FastifyReply): AuthUser | undefined {
  const user = requireUser(req);
  if (user.role !== 'admin') {
    reply.code(403).send({ error: 'forbidden', message: 'Administrator access required.' });
    return undefined;
  }
  return user;
}

export function requireUser(req: FastifyRequest): AuthUser {
  if (!req.user) throw new UnauthenticatedError();
  return req.user;
}

/**
 * SQL predicate for "this user owns the conversation".
 *
 * Chats created before accounts existed have a session_id and no owner. They
 * are adopted at first login, but a browser that never signs in as their
 * original owner would otherwise lose them mid-migration, so unclaimed rows
 * stay reachable through the old cookie.
 */
export function ownedByUser(alias: string, userParam: string, legacyParam: string): string {
  const p = alias ? `${alias}.` : '';
  return `(${p}user_id = ${userParam} OR (${p}user_id IS NULL AND ${p}session_id = ${legacyParam}))`;
}

/** Owner parameters, in the order `ownedByUser` expects them. */
export function ownerParams(req: FastifyRequest): [string, string] {
  return [requireUser(req).id, req.legacySessionId];
}

/**
 * Per-request identity.
 *
 * Replaces the single shared passphrase: a session cookie resolves to a real
 * account, and conversations scope to that account rather than to a browser.
 * APP_ACCESS_TOKEN survives only as an optional extra gate in front of the
 * login screen (REQUIRE_ACCESS_TOKEN=true).
 */
export function registerAuth(app: FastifyInstance): void {
  app.decorateRequest('user', null);
  app.decorateRequest('authSessionId', '');
  app.decorateRequest('legacySessionId', '');

  app.addHook('preHandler', async (req: FastifyRequest, reply: FastifyReply) => {
    if (!req.url.startsWith('/api/')) return;

    req.legacySessionId = readSignedCookie(req, LEGACY_SESSION_COOKIE) ?? '';

    const token = readSignedCookie(req, SESSION_COOKIE);
    if (token) {
      const resolved = await resolveSession(token);
      if (resolved) {
        req.user = resolved.user;
        req.authSessionId = resolved.sessionId;
        touchSession(resolved.sessionId);
      } else {
        // Expired, revoked, or the account was disabled — stop presenting it.
        reply.clearCookie(SESSION_COOKIE, { path: '/' });
      }
    }

    const path = req.url.split('?')[0];
    if (PUBLIC_PATHS.has(path) || path.startsWith('/api/artifacts/')) return;

    // The image upload is a worker, which has no account and authenticates
    // with WORKER_TOKEN inside the route. Deliberately method-scoped: GET
    // /api/images/:id must stay behind a session, because that is what proves
    // the caller owns the conversation the picture belongs to. Exempting the
    // whole path would make every user's images readable by anyone.
    if (req.method === 'POST' && path === '/api/images') return;
    if (req.method === 'POST' && path.startsWith('/api/artifacts/')) return;

    if (config.requireAccessToken) {
      const access = readSignedCookie(req, ACCESS_COOKIE);
      if (access !== 'granted') {
        return reply.code(401).send({ error: 'locked', message: 'Access token required.' });
      }
    }

    if (!req.user) {
      return reply.code(401).send({ error: 'unauthenticated', message: 'Sign in required.' });
    }
  });
}
