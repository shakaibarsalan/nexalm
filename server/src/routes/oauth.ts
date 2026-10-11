import { randomBytes } from 'node:crypto';
import type { FastifyInstance, FastifyReply } from 'fastify';
import { config, isProd } from '../config.js';
import { query } from '../db/pool.js';
import { normalizeEmail } from '../lib/password.js';
import { sendWelcome } from '../services/mailer.js';
import {
  adoptLegacyConversations,
  createSession,
  createUser,
  findUserByEmail,
  findUserByGoogleSub,
  markLogin,
  readSignedCookie,
  setSessionCookie,
} from '../services/authService.js';

const STATE_COOKIE = 'llm_oauth_state';
const AUTH_ENDPOINT = 'https://accounts.google.com/o/oauth2/v2/auth';
const TOKEN_ENDPOINT = 'https://oauth2.googleapis.com/token';

interface IdTokenClaims {
  iss?: string;
  aud?: string;
  exp?: number;
  sub?: string;
  email?: string;
  email_verified?: boolean | string;
  name?: string;
  picture?: string;
}

/**
 * Read the id_token payload without verifying its signature.
 *
 * Safe only because this token came back on our own TLS connection to Google's
 * token endpoint, authenticated with the client secret — the case Google's own
 * documentation exempts from signature checks. Never do this with an id_token
 * that arrived from the browser.
 */
function decodeIdToken(idToken: string): IdTokenClaims | undefined {
  const parts = idToken.split('.');
  if (parts.length !== 3) return undefined;
  try {
    return JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf8')) as IdTokenClaims;
  } catch {
    return undefined;
  }
}

function claimsAreValid(claims: IdTokenClaims): boolean {
  const issuerOk =
    claims.iss === 'https://accounts.google.com' || claims.iss === 'accounts.google.com';
  const notExpired = typeof claims.exp === 'number' && claims.exp * 1000 > Date.now();
  const verified = claims.email_verified === true || claims.email_verified === 'true';
  return issuerOk && notExpired && verified && claims.aud === config.google.clientId && !!claims.sub;
}

function failRedirect(reply: FastifyReply, code: string): FastifyReply {
  return reply.redirect(`${config.publicOrigin}/?auth_error=${encodeURIComponent(code)}`, 302);
}

export function registerOAuthRoutes(app: FastifyInstance): void {
  /**
   * Registered even when unconfigured so the routes 404 with a JSON body
   * instead of the app failing to boot without Google credentials.
   */
  const disabled = (reply: FastifyReply): FastifyReply =>
    reply
      .code(404)
      .send({ error: 'google_disabled', message: 'Google sign-in is not configured.' });

  app.get('/api/auth/google', async (_req, reply) => {
    if (!config.googleEnabled) return disabled(reply);

    // The state lives in a signed, short-lived cookie so the callback can prove
    // the redirect it is handling was started by this browser.
    const state = randomBytes(32).toString('base64url');
    reply.setCookie(STATE_COOKIE, state, {
      path: '/api/auth/google',
      httpOnly: true,
      secure: isProd,
      sameSite: 'lax',
      signed: true,
      maxAge: 600,
    });

    const url = new URL(AUTH_ENDPOINT);
    url.searchParams.set('client_id', config.google.clientId);
    url.searchParams.set('redirect_uri', config.google.redirectUri);
    url.searchParams.set('response_type', 'code');
    url.searchParams.set('scope', 'openid email profile');
    url.searchParams.set('state', state);
    // Force the chooser: on a shared browser, silently reusing whichever Google
    // account is already signed in is how you end up in someone else's chats.
    url.searchParams.set('prompt', 'select_account');
    return reply.redirect(url.toString(), 302);
  });

  app.get('/api/auth/google/callback', async (req, reply) => {
    if (!config.googleEnabled) return disabled(reply);

    const params = req.query as { code?: string; state?: string; error?: string };
    if (params.error) return failRedirect(reply, params.error);

    const expected = readSignedCookie(req, STATE_COOKIE);
    reply.clearCookie(STATE_COOKIE, { path: '/api/auth/google' });
    if (!expected || !params.state || params.state !== expected) {
      return failRedirect(reply, 'state_mismatch');
    }
    if (!params.code) return failRedirect(reply, 'missing_code');

    let idToken: string;
    try {
      const res = await fetch(TOKEN_ENDPOINT, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          code: params.code,
          client_id: config.google.clientId,
          client_secret: config.google.clientSecret,
          redirect_uri: config.google.redirectUri,
          grant_type: 'authorization_code',
        }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!res.ok) {
        // The body can echo the client secret back in an error description.
        app.log.warn({ status: res.status }, 'google token exchange rejected');
        return failRedirect(reply, 'token_exchange_failed');
      }
      const payload = (await res.json()) as { id_token?: string };
      if (!payload.id_token) return failRedirect(reply, 'token_exchange_failed');
      idToken = payload.id_token;
    } catch (err) {
      app.log.warn({ err: (err as Error).message }, 'google token exchange failed');
      return failRedirect(reply, 'token_exchange_failed');
    }

    const claims = decodeIdToken(idToken);
    if (!claims || !claimsAreValid(claims)) return failRedirect(reply, 'invalid_id_token');

    const email = normalizeEmail(claims.email);
    if (!email) return failRedirect(reply, 'no_verified_email');

    const sub = claims.sub as string;
    const displayName = (claims.name ?? email.split('@')[0]).slice(0, 80);
    const picture = claims.picture ?? null;

    let user = await findUserByGoogleSub(sub);
    if (!user) {
      // Linking by email is only sound because Google asserted it is verified,
      // which claimsAreValid() has already required.
      const byEmail = await findUserByEmail(email);
      if (byEmail) {
        await query(
          `UPDATE users SET google_sub = $2, avatar_url = COALESCE(avatar_url, $3) WHERE id = $1`,
          [byEmail.id, sub, picture],
        );
        user = { ...byEmail, google_sub: sub };
      } else {
        // Google sign-in must not be a way around a closed registration.
        // Without this, publishing the OAuth consent screen would let any
        // Google account self-register on a private deployment, silently
        // defeating SIGNUP_ENABLED.
        if (!config.signupEnabled) return failRedirect(reply, 'signup_disabled');

        user = await createUser({
          email,
          displayName,
          googleSub: sub,
          avatarUrl: picture,
          passwordHash: null,
        });
        // Google has already proved the address, so no code is needed -- but a
        // welcome is still owed. Signing up this way silently skipped it, so
        // half the new accounts never heard from us at all.
        await query(`UPDATE users SET email_verified_at = now() WHERE id = $1`, [user.id]);
        void sendWelcome(email, user.display_name ?? displayName).catch((err: unknown) =>
          req.log.warn({ err }, 'welcome email failed for a Google sign-up'),
        );
      }
    }

    if (user.disabled_at) return failRedirect(reply, 'account_disabled');

    const token = await createSession(user.id, req);
    setSessionCookie(reply, token);
    await markLogin(user.id);
    await adoptLegacyConversations(user.id, req.legacySessionId);

    return reply.redirect(`${config.publicOrigin}/`, 302);
  });
}
