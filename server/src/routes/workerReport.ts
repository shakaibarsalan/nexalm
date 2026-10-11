import { timingSafeEqual } from 'node:crypto';
import type { FastifyInstance, FastifyRequest } from 'fastify';
import { config } from '../config.js';

/**
 * Lets a worker say why it never arrived.
 *
 * A GPU session that fails during provisioning is otherwise a black box: a
 * Kaggle session started through the API is interactive, so it produces no
 * saved kernel output, and `kernels output` returns the last *committed* run
 * instead. A session can therefore burn several minutes of a finite weekly
 * quota and leave nothing at all to diagnose -- which is exactly what happened
 * before this existed.
 *
 * Authenticated with WORKER_TOKEN rather than a user session, for the same
 * reason as the artifact endpoint: the caller is a bootstrapping worker with no
 * browser and no account. Reports are written to the server log rather than the
 * database; they are diagnostics for an operator reading `docker logs`, not
 * application state, and a failing worker retrying should not be able to grow a
 * table without bound.
 */

const MAX_DETAIL = 8000;

function tokenMatches(candidate: string): boolean {
  const a = Buffer.from(candidate);
  const b = Buffer.from(config.workerToken);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

function presentedToken(req: FastifyRequest): string | undefined {
  const auth = req.headers['authorization'];
  if (typeof auth === 'string' && auth.startsWith('Bearer ')) return auth.slice(7).trim();
  const q = (req.query as { token?: string }).token;
  return typeof q === 'string' ? q : undefined;
}

export function registerWorkerReportRoutes(app: FastifyInstance): void {
  app.post('/api/workers/report', async (req, reply) => {
    const token = presentedToken(req);
    if (!token || !tokenMatches(token)) {
      return reply.code(401).send({ error: 'unauthorized', message: 'Worker token required.' });
    }

    const body = (req.body ?? {}) as {
      platform?: unknown;
      stage?: unknown;
      message?: unknown;
      detail?: unknown;
    };

    const platform = typeof body.platform === 'string' ? body.platform.slice(0, 32) : 'unknown';
    const stage = typeof body.stage === 'string' ? body.stage.slice(0, 64) : 'unknown';
    const message = typeof body.message === 'string' ? body.message.slice(0, 500) : '';
    // Truncated rather than rejected: a traceback that is too long is still
    // the most useful thing in the request, and losing it to a 400 would
    // defeat the point of the endpoint.
    const detail = typeof body.detail === 'string' ? body.detail.slice(-MAX_DETAIL) : '';

    req.log.warn(
      { platform, stage, detail },
      `worker provisioning failed on ${platform} at ${stage}: ${message}`,
    );

    return { ok: true };
  });
}
