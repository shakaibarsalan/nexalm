import { createHash, randomInt, timingSafeEqual } from 'node:crypto';
import type { FastifyInstance } from 'fastify';
import { config, isProd } from '../config.js';
import { query } from '../db/pool.js';
import {
  CODE_TTL_MINUTES,
  checkBudget,
  isConfigured as mailIsConfigured,
  sendCode,
  sendWelcome,
} from '../services/mailer.js';
import { hashPassword, normalizeEmail, passwordProblem, verifyPassword } from '../lib/password.js';
import {
  ACCESS_COOKIE,
  EmailTakenError,
  adoptLegacyConversations,
  clearSessionCookie,
  constantTimeEqual,
  createSession,
  createUser,
  findUserByEmail,
  markLogin,
  pruneExpiredSessions,
  readSignedCookie,
  revokeAllSessions,
  revokeSession,
  setSessionCookie,
  toAuthUser,
  userCount,
} from '../services/authService.js';
import { requireUser } from '../middleware/auth.js';

/**
 * Login throttle.
 *
 * Per-process and in-memory, so it resets on deploy and would not hold across
 * replicas. That is acceptable while this runs as a single container; moving to
 * more than one API instance means moving this state into Postgres or Redis.
 */
interface Attempt {
  fails: number;
  lockedUntil: number;
  lastSeen: number;
}

const attempts = new Map<string, Attempt>();
const FREE_ATTEMPTS = 5;
const ATTEMPT_TTL_MS = 15 * 60_000;
const MAX_LOCK_MS = 15 * 60_000;

function throttleKey(ip: string, email: string): string {
  return `${ip}|${email}`;
}

/** Seconds the caller must wait, or 0 when they may try now. */
function throttleRetryAfter(key: string): number {
  const entry = attempts.get(key);
  if (!entry) return 0;
  const remaining = entry.lockedUntil - Date.now();
  return remaining > 0 ? Math.ceil(remaining / 1000) : 0;
}

function recordFailure(key: string): void {
  const now = Date.now();
  const entry = attempts.get(key) ?? { fails: 0, lockedUntil: 0, lastSeen: now };
  entry.fails += 1;
  entry.lastSeen = now;
  if (entry.fails > FREE_ATTEMPTS) {
    const backoff = Math.min(2 ** (entry.fails - FREE_ATTEMPTS) * 1000, MAX_LOCK_MS);
    entry.lockedUntil = now + backoff;
  }
  attempts.set(key, entry);

  // Bounded cleanup: a hostile client rotating emails must not grow this map
  // without limit.
  if (attempts.size > 1000) {
    for (const [k, v] of attempts) {
      if (now - v.lastSeen > ATTEMPT_TTL_MS && v.lockedUntil < now) attempts.delete(k);
    }
  }
}

function clearFailures(key: string): void {
  attempts.delete(key);
}

/**
 * Burned on unknown emails so that "no such account" and "wrong password" take
 * roughly the same time and cannot be told apart by a stopwatch.
 */
let decoyHash: Promise<string> | undefined;
function decoy(): Promise<string> {
  decoyHash ??= hashPassword('decoy-password-for-timing-equalisation');
  return decoyHash;
}

export function registerAuthRoutes(app: FastifyInstance): void {
/**
 * Whether this address has to prove itself before an account is made.
 *
 * Only when the server can actually send mail: demanding a code it cannot
 * deliver would lock everybody out. Reserved domains are exempt because they
 * cannot receive mail at all (see config.otpExemptDomains).
 */
function needsVerification(email: string): boolean {
  if (!mailIsConfigured()) return false;
  const domain = email.split('@')[1]?.toLowerCase() ?? '';
  return !config.otpExemptDomains.includes(domain);
}

/** Six digits, from a cryptographic source rather than Math.random. */
function newCode(): string {
  return String(randomInt(0, 1_000_000)).padStart(6, '0');
}

function hashCode(code: string): string {
  return createHash('sha256').update(code).digest('hex');
}

  app.post('/api/auth/signup', async (req, reply) => {
    if (!config.signupEnabled) {
      return reply
        .code(403)
        .send({ error: 'signup_disabled', message: 'Sign-ups are closed on this deployment.' });
    }

    const body = (req.body ?? {}) as { email?: unknown; password?: unknown; displayName?: unknown };

    const email = normalizeEmail(body.email);
    if (!email) {
      return reply.code(400).send({ error: 'invalid_email', message: 'Enter a valid email address.' });
    }
    const problem = passwordProblem(body.password);
    if (problem) return reply.code(400).send({ error: 'weak_password', message: problem });

    const displayName =
      typeof body.displayName === 'string' && body.displayName.trim()
        ? body.displayName.trim().slice(0, 80)
        : email.split('@')[0];

    const passwordHash = await hashPassword(body.password as string);

    if (needsVerification(email)) {
      // Refuse early on a taken address. Sending a code for an address that
      // already has an account would let someone use sign-up to spam a
      // stranger's inbox, and tells them nothing they can act on.
      if (await findUserByEmail(email)) {
        return reply.code(409).send({
          error: 'email_taken',
          message: 'An account already exists for that address. Sign in instead.',
        });
      }

      const budget = await checkBudget(email);
      if (!budget.allowed) {
        return reply.code(429).send({
          error: 'too_many_emails',
          message:
            `That address has already been sent ${budget.sentThisHour} messages in the ` +
            `last hour. Try again in about ${budget.retryAfterMinutes} minutes.`,
          retryAfterMinutes: budget.retryAfterMinutes,
        });
      }

      const code = newCode();
      // One pending sign-up per address: asking again replaces the previous
      // code rather than leaving several valid at once.
      await query(
        `INSERT INTO email_verifications (email, code_hash, password_hash, display_name, expires_at)
         VALUES ($1, $2, $3, $4, now() + ($5 || ' minutes')::interval)
         ON CONFLICT (lower(email)) DO UPDATE SET
           code_hash = EXCLUDED.code_hash, password_hash = EXCLUDED.password_hash,
           display_name = EXCLUDED.display_name, expires_at = EXCLUDED.expires_at,
           attempts = 0, created_at = now()`,
        [email, hashCode(code), passwordHash, displayName, String(CODE_TTL_MINUTES)],
      );

      try {
        await sendCode(email, code);
      } catch (err) {
        req.log.error({ err }, 'could not send a verification code');
        return reply.code(502).send({
          error: 'mail_failed',
          message: 'Could not send the verification email. Check the address and try again.',
        });
      }

      return reply.code(202).send({
        verificationRequired: true,
        email,
        expiresInMinutes: CODE_TTL_MINUTES,
        message: `We sent a 6-digit code to ${email}. It expires in ${CODE_TTL_MINUTES} minutes.`,
      });
    }

    try {
      const row = await createUser({ email, displayName, passwordHash });
      await query(`UPDATE users SET email_verified_at = now() WHERE id = $1`, [row.id]);

      const token = await createSession(row.id, req);
      setSessionCookie(reply, token);
      const adopted = await adoptLegacyConversations(row.id, req.legacySessionId);
      pruneExpiredSessions();

      return reply.code(201).send({ user: toAuthUser(row), adoptedConversations: adopted });
    } catch (err) {
      if (err instanceof EmailTakenError) {
        return reply.code(409).send({ error: 'email_taken', message: err.message });
      }
      throw err;
    }
  });

  /**
   * Exchange a code for the account it was issued for.
   *
   * The account is created here, not at sign-up, so an address that is never
   * confirmed never becomes a user row at all.
   */
  app.post('/api/auth/verify-email', async (req, reply) => {
    const body = (req.body ?? {}) as { email?: unknown; code?: unknown };
    const email = normalizeEmail(body.email);
    const code = typeof body.code === 'string' ? body.code.trim() : '';
    if (!email || !/^\d{6}$/.test(code)) {
      return reply
        .code(400)
        .send({ error: 'invalid_request', message: 'Enter the 6-digit code from the email.' });
    }

    const { rows } = await query<{
      id: string;
      code_hash: string;
      password_hash: string;
      display_name: string | null;
      attempts: number;
      expired: boolean;
    }>(
      `SELECT id, code_hash, password_hash, display_name, attempts,
              (expires_at < now()) AS expired
         FROM email_verifications WHERE lower(email) = lower($1)`,
      [email],
    );
    if (rows.length === 0) {
      return reply.code(404).send({
        error: 'no_pending_signup',
        message: 'No sign-up is waiting for that address. Start again.',
      });
    }
    const pending = rows[0];

    if (pending.expired) {
      await query(`DELETE FROM email_verifications WHERE id = $1`, [pending.id]);
      return reply
        .code(410)
        .send({ error: 'code_expired', message: 'That code has expired. Start again.' });
    }
    // A six-digit code is guessable given unlimited tries, so the tries are
    // limited: five wrong answers burn the code rather than the attacker's
    // patience.
    if (pending.attempts >= 5) {
      await query(`DELETE FROM email_verifications WHERE id = $1`, [pending.id]);
      return reply.code(429).send({
        error: 'too_many_attempts',
        message: 'Too many incorrect codes. Start the sign-up again.',
      });
    }

    const supplied = Buffer.from(hashCode(code));
    const expected = Buffer.from(pending.code_hash);
    const matches = supplied.length === expected.length && timingSafeEqual(supplied, expected);
    if (!matches) {
      await query(`UPDATE email_verifications SET attempts = attempts + 1 WHERE id = $1`, [
        pending.id,
      ]);
      return reply.code(401).send({
        error: 'invalid_code',
        message: `That code is not right. ${4 - pending.attempts} attempts left.`,
      });
    }

    try {
      const row = await createUser({
        email,
        displayName: pending.display_name ?? email.split('@')[0],
        passwordHash: pending.password_hash,
      });
      await query(`UPDATE users SET email_verified_at = now() WHERE id = $1`, [row.id]);
      await query(`DELETE FROM email_verifications WHERE id = $1`, [pending.id]);

      const token = await createSession(row.id, req);
      setSessionCookie(reply, token);
      const adopted = await adoptLegacyConversations(row.id, req.legacySessionId);
      pruneExpiredSessions();

      // Best effort: the account exists and the user is signed in, so a mail
      // failure here must not turn a successful sign-up into an error.
      void sendWelcome(email, row.display_name ?? email.split('@')[0]).catch((err) =>
        req.log.warn({ err }, 'welcome email failed'),
      );

      return reply.code(201).send({ user: toAuthUser(row), adoptedConversations: adopted });
    } catch (err) {
      if (err instanceof EmailTakenError) {
        return reply.code(409).send({ error: 'email_taken', message: err.message });
      }
      throw err;
    }
  });

  /** Send the code again, against the same hourly budget. */
  app.post('/api/auth/resend-code', async (req, reply) => {
    const email = normalizeEmail((req.body as { email?: unknown })?.email);
    if (!email) {
      return reply.code(400).send({ error: 'invalid_email', message: 'Enter a valid address.' });
    }
    const { rows } = await query<{ id: string }>(
      `SELECT id FROM email_verifications WHERE lower(email) = lower($1)`,
      [email],
    );
    if (rows.length === 0) {
      return reply.code(404).send({
        error: 'no_pending_signup',
        message: 'No sign-up is waiting for that address.',
      });
    }

    const budget = await checkBudget(email);
    if (!budget.allowed) {
      return reply.code(429).send({
        error: 'too_many_emails',
        message:
          `That address has had ${budget.sentThisHour} messages in the last hour, which is ` +
          `the limit. Try again in about ${budget.retryAfterMinutes} minutes.`,
        retryAfterMinutes: budget.retryAfterMinutes,
      });
    }

    const code = newCode();
    await query(
      `UPDATE email_verifications
          SET code_hash = $2, attempts = 0,
              expires_at = now() + ($3 || ' minutes')::interval
        WHERE id = $1`,
      [rows[0].id, hashCode(code), String(CODE_TTL_MINUTES)],
    );
    try {
      await sendCode(email, code);
    } catch (err) {
      req.log.error({ err }, 'could not resend a verification code');
      return reply
        .code(502)
        .send({ error: 'mail_failed', message: 'Could not send the email. Try again shortly.' });
    }
    return { ok: true, expiresInMinutes: CODE_TTL_MINUTES, sentThisHour: budget.sentThisHour + 1 };
  });

  app.post('/api/auth/login', async (req, reply) => {
    const body = (req.body ?? {}) as { email?: unknown; password?: unknown };
    const email = normalizeEmail(body.email);
    const password = typeof body.password === 'string' ? body.password : '';

    if (!email || !password) {
      return reply
        .code(400)
        .send({ error: 'invalid_request', message: 'Email and password are required.' });
    }

    const key = throttleKey(req.ip, email);
    const retryAfter = throttleRetryAfter(key);
    if (retryAfter > 0) {
      return reply
        .code(429)
        .header('retry-after', String(retryAfter))
        .send({
          error: 'too_many_attempts',
          message: `Too many failed attempts. Try again in ${retryAfter}s.`,
        });
    }

    const row = await findUserByEmail(email);
    const ok = row
      ? await verifyPassword(password, row.password_hash)
      : await verifyPassword(password, await decoy());

    if (!row || !ok) {
      recordFailure(key);
      return reply
        .code(401)
        .send({ error: 'invalid_credentials', message: 'Incorrect email or password.' });
    }
    if (row.disabled_at) {
      return reply
        .code(403)
        .send({ error: 'account_disabled', message: 'This account has been disabled.' });
    }

    clearFailures(key);
    const token = await createSession(row.id, req);
    setSessionCookie(reply, token);
    await markLogin(row.id);
    const adopted = await adoptLegacyConversations(row.id, req.legacySessionId);
    pruneExpiredSessions();

    return { user: toAuthUser(row), adoptedConversations: adopted };
  });

  /** Public and idempotent: a stale cookie must still be clearable. */
  app.post('/api/auth/logout', async (req, reply) => {
    if (req.authSessionId) await revokeSession(req.authSessionId);
    clearSessionCookie(reply);
    return { ok: true };
  });

  app.post('/api/auth/logout-all', async (req, reply) => {
    const user = requireUser(req);
    const revoked = await revokeAllSessions(user.id);
    clearSessionCookie(reply);
    return { ok: true, revoked };
  });

  app.get('/api/auth/me', async (req) => {
    return { user: requireUser(req) };
  });

  /** Public: the sign-in screen reads this before any credentials exist. */
  app.get('/api/auth/status', async (req) => {
    const unlocked = readSignedCookie(req, ACCESS_COOKIE) === 'granted';
    return {
      authenticated: Boolean(req.user),
      user: req.user,
      googleEnabled: config.googleEnabled,
      signupEnabled: config.signupEnabled,
      emailVerification: mailIsConfigured(),
      requiresAccessToken: config.requireAccessToken,
      unlocked,
      /** No accounts yet — whoever signs up first gets admin. */
      setupRequired: (await userCount()) === 0,
    };
  });

  // ── Optional shared-passphrase gate (REQUIRE_ACCESS_TOKEN=true) ──

  app.post('/api/auth/unlock', async (req, reply) => {
    const body = (req.body ?? {}) as { token?: string };
    if (
      !config.appAccessToken ||
      typeof body.token !== 'string' ||
      !constantTimeEqual(body.token, config.appAccessToken)
    ) {
      // Blunt throttle against guessing the shared passphrase.
      await new Promise((r) => setTimeout(r, 600));
      return reply.code(401).send({ error: 'invalid_token', message: 'Incorrect access token.' });
    }
    reply.setCookie(ACCESS_COOKIE, 'granted', {
      path: '/',
      httpOnly: true,
      secure: isProd,
      sameSite: 'lax',
      signed: true,
      maxAge: 60 * 60 * 24 * 30,
    });
    return { ok: true };
  });

  app.post('/api/auth/lock', async (_req, reply) => {
    reply.clearCookie(ACCESS_COOKIE, { path: '/' });
    return { ok: true };
  });
}
