import type { FastifyInstance } from 'fastify';
import { healthCheck } from '../db/pool.js';
import { workerRegistry } from '../services/workerRegistry.js';

export function registerHealthRoutes(app: FastifyInstance): void {
  /** Unauthenticated: used by the container healthcheck. */
  app.get('/api/health', async (_req, reply) => {
    const db = await healthCheck();
    const workers = workerRegistry.list();
    const ready = workers.filter((w) => w.status === 'WORKER_READY' || w.status === 'WORKER_BUSY');

    const body = {
      status: db.ok ? 'ok' : 'degraded',
      uptimeSec: Math.round(process.uptime()),
      db,
      workers: {
        connected: workers.length,
        available: ready.length,
        statuses: workers.map((w) => ({ id: w.id, name: w.name, status: w.status })),
      },
    };
    return reply.code(db.ok ? 200 : 503).send(body);
  });
}
