import { timingSafeEqual } from 'node:crypto';
import type { FastifyInstance } from 'fastify';
import type { WebSocket } from 'ws';
import { config } from '../config.js';
import { query } from '../db/pool.js';
import { jobManager } from '../services/jobManager.js';
import { workerRegistry } from '../services/workerRegistry.js';
import type { WorkerInbound } from '../lib/types.js';

/** Constant-time comparison so the worker token cannot be probed by timing. */
function tokenMatches(candidate: string): boolean {
  const a = Buffer.from(candidate);
  const b = Buffer.from(config.workerToken);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

function extractToken(headers: Record<string, unknown>, url: string): string | undefined {
  const auth = headers['authorization'];
  if (typeof auth === 'string' && auth.startsWith('Bearer ')) return auth.slice(7).trim();
  const proto = headers['sec-websocket-protocol'];
  if (typeof proto === 'string' && proto.startsWith('bearer.')) return proto.slice(7).trim();
  // Query-string fallback: some notebook WS clients cannot set headers.
  try {
    const parsed = new URL(url, 'http://localhost');
    return parsed.searchParams.get('token') ?? undefined;
  } catch {
    return undefined;
  }
}

/**
 * WebSocket endpoint that inference workers dial into.
 *
 * The worker always initiates the connection — Kaggle/Colab runtimes sit behind
 * NAT and can never be reached inbound, so this direction is not negotiable.
 */
export function registerWorkerHub(app: FastifyInstance): void {
  app.get('/worker/ws', { websocket: true }, (socket: WebSocket, req) => {
    const token = extractToken(req.headers as Record<string, unknown>, req.url ?? '');

    if (!token || !tokenMatches(token)) {
      app.log.warn({ ip: req.ip }, '[worker-ws] rejected: bad or missing token');
      socket.send(JSON.stringify({ type: 'error', payload: { message: 'unauthorized' } }));
      socket.close(4401, 'unauthorized');
      return;
    }

    let workerId: string | undefined;
    let registered = false;

    const heartbeatWatchdog = setInterval(() => {
      if (socket.readyState === 1) socket.ping();
    }, config.heartbeatIntervalMs);
    heartbeatWatchdog.unref?.();

    socket.on('message', (raw: Buffer) => {
      let frame: WorkerInbound;
      try {
        frame = JSON.parse(raw.toString());
      } catch {
        app.log.warn('[worker-ws] dropped malformed frame');
        return;
      }

      void (async () => {
        try {
          switch (frame.type) {
            case 'register': {
              const reg = frame.payload;
              if (!reg?.workerId || !reg?.name) {
                socket.send(
                  JSON.stringify({ type: 'error', payload: { message: 'register requires workerId and name' } }),
                );
                return;
              }
              workerId = reg.workerId;
              registered = true;
              const worker = await workerRegistry.register(socket, reg);
              app.log.info(
                { workerId: worker.id, gpus: worker.gpus.length, model: worker.model, engine: worker.engine },
                '[worker-ws] registered',
              );
              // The idle timeout is decided here rather than baked into the
              // worker. It protects a metered quota, so it has to be
              // adjustable without editing a notebook and re-running it --
              // which is what a worker that shut down mid-session cost today.
              const { rows: settingRows } = await query<{ value: Record<string, unknown> }>(
                `SELECT value FROM app_settings WHERE key = 'defaults'`,
              );
              const idleMinutes = Number(settingRows[0]?.value?.idleShutdownMinutes ?? 30);

              socket.send(
                JSON.stringify({
                  type: 'registered',
                  payload: {
                    ok: true,
                    serverTime: new Date().toISOString(),
                    heartbeatIntervalMs: config.heartbeatIntervalMs,
                    idleShutdownMinutes: Number.isFinite(idleMinutes) ? idleMinutes : 30,
                  },
                }),
              );
              break;
            }

            case 'heartbeat':
              if (workerId) await workerRegistry.heartbeat(workerId, frame.payload ?? {});
              break;

            case 'job.accepted':
              break;

            case 'model.state':
              // The worker is the only authority on what is actually resident;
              // the server records rather than assumes, so a load that failed
              // half way cannot leave the pool advertising a model it lacks.
              if (workerId) await workerRegistry.setModelState(workerId, frame.payload);
              break;

            case 'model.error':
              if (workerId) {
                app.log.warn(
                  { workerId, modelId: frame.payload.modelId },
                  `[worker-ws] model load failed: ${frame.payload.message}`,
                );
                // Clearing `loading` matters more than the log line: without it
                // the worker looks permanently busy and never accepts another load.
                await workerRegistry.setModelState(workerId, {
                  loaded: workerRegistry.get(workerId)?.loaded ?? [],
                });
              }
              break;

            case 'job.first_token':
              jobManager.onFirstToken(frame.payload.jobId, frame.payload.ttftMs);
              break;

            case 'job.delta':
              jobManager.onDelta(frame.payload.jobId, frame.payload.text);
              break;

            case 'job.reasoning':
              jobManager.onReasoning(frame.payload.jobId, frame.payload.text);
              break;

            case 'job.progress':
              jobManager.onProgress(frame.payload.jobId, frame.payload);
              break;

            case 'job.done':
              await jobManager.onDone(frame.payload.jobId, frame.payload);
              break;

            case 'job.error':
              await jobManager.fail(frame.payload.jobId, frame.payload.message);
              break;

            case 'deregister':
              if (workerId) {
                app.log.info({ workerId }, '[worker-ws] graceful deregister');
                await workerRegistry.deregister(workerId, frame.payload?.reason);
                registered = false;
              }
              break;

            case 'pong':
              break;

            default:
              app.log.warn({ frame }, '[worker-ws] unknown frame type');
          }
        } catch (err) {
          app.log.error({ err, type: frame.type }, '[worker-ws] frame handler failed');
        }
      })();
    });

    socket.on('close', (code: number, reason: Buffer) => {
      clearInterval(heartbeatWatchdog);
      if (workerId && registered) {
        app.log.warn({ workerId, code, reason: reason?.toString() }, '[worker-ws] disconnected');
        void workerRegistry.deregister(workerId, `socket closed (${code})`);
      }
    });

    socket.on('error', (err: Error) => {
      app.log.error({ err, workerId }, '[worker-ws] socket error');
    });
  });
}
