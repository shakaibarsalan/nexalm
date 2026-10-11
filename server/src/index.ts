import Fastify from 'fastify';
import type { FastifyError } from 'fastify';
import cookie from '@fastify/cookie';
import websocket from '@fastify/websocket';
import { config, isProd } from './config.js';
import { pool } from './db/pool.js';
import { runMigrations } from './db/migrate.js';
import { registerAuth } from './middleware/auth.js';
import { registerHealthRoutes } from './routes/health.js';
import { registerAuthRoutes } from './routes/auth.js';
import { registerOAuthRoutes } from './routes/oauth.js';
import { registerUserRoutes } from './routes/users.js';
import { registerConversationRoutes } from './routes/conversations.js';
import { registerChatRoutes } from './routes/chat.js';
import { registerSettingsRoutes } from './routes/settings.js';
import { registerWorkerRoutes } from './routes/workers.js';
import { registerArtifactRoutes } from './routes/artifacts.js';
import { registerImageRoutes } from './routes/images.js';
import { registerWorkerReportRoutes } from './routes/workerReport.js';
import { registerWorkerHub } from './ws/workerHub.js';
import { workerRegistry } from './services/workerRegistry.js';
import { watchPool } from './services/pendingQueue.js';
import { isConfigured as mailConfigured, warm as warmMail } from './services/mailer.js';

async function main(): Promise<void> {
  const app = Fastify({
    logger: {
      level: config.logLevel,
      transport: isProd ? undefined : { target: 'pino-pretty' },
    },
    trustProxy: true,
    // Long generations stream for minutes; the default 0 disables the timeout.
    connectionTimeout: 0,
    keepAliveTimeout: 72_000,
  });

  await app.register(cookie, { secret: config.cookieSecret });
  await app.register(websocket, {
    options: { maxPayload: 8 * 1024 * 1024 },
  });

  registerAuth(app);
  registerHealthRoutes(app);
  registerAuthRoutes(app);
  registerOAuthRoutes(app);
  registerUserRoutes(app);
  registerConversationRoutes(app);
  registerChatRoutes(app);
  registerSettingsRoutes(app);
  registerWorkerRoutes(app);
  // Build tarballs arrive as raw bytes. Fastify's default body limit is 1 MB
  // and it has no parser for this type, so without both of these a worker
  // publishing a ~90 MB binary is rejected before the route is reached.
  app.addContentTypeParser(
    'application/octet-stream',
    { parseAs: 'buffer', bodyLimit: 256 * 1024 * 1024 },
    (_req, body, done) => done(null, body),
  );
  registerArtifactRoutes(app);
  registerImageRoutes(app);
  registerWorkerReportRoutes(app);
  registerWorkerHub(app);

  app.setErrorHandler((err: FastifyError, _req, reply) => {
    app.log.error({ err }, 'unhandled route error');
    const status = err.statusCode && err.statusCode >= 400 ? err.statusCode : 500;
    reply.code(status).send({
      error: status === 500 ? 'internal_error' : err.name,
      message: status === 500 && isProd ? 'Internal server error' : err.message,
    });
  });

  app.log.info('running database migrations…');
  await runMigrations();

  workerRegistry.startSweeper();
  // Answer anything that queued while no GPU was connected, including across
  // an API restart -- the queue is a query, not in-memory state.
  watchPool();
  // Opens the SMTP connection in the background: the first verification code
  // should not be the one that pays for the TLS handshake.
  if (mailConfigured()) void warmMail();

  await app.listen({ port: config.port, host: config.host });
  app.log.info(`API listening on ${config.host}:${config.port}`);

  const shutdown = async (signal: string): Promise<void> => {
    app.log.info(`${signal} received — shutting down`);
    workerRegistry.stopSweeper();
    try {
      await app.close();
      await pool.end();
    } catch (err) {
      app.log.error({ err }, 'error during shutdown');
    }
    process.exit(0);
  };

  process.on('SIGTERM', () => void shutdown('SIGTERM'));
  process.on('SIGINT', () => void shutdown('SIGINT'));
}

main().catch((err) => {
  console.error('fatal startup error:', err);
  process.exit(1);
});
