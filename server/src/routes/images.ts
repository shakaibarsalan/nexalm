import { createReadStream } from 'node:fs';
import { mkdir, stat, writeFile, unlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import type { FastifyInstance, FastifyRequest } from 'fastify';
import { config } from '../config.js';
import { query } from '../db/pool.js';
import { ownedByUser, ownerParams } from '../middleware/auth.js';
import { jobManager } from '../services/jobManager.js';

/**
 * Generated images: the worker uploads them here, and the browser reads them
 * back through the user's own session.
 *
 * Images do not travel over the job socket. A PNG is one to two megabytes, the
 * socket has an 8 MB frame ceiling at both ends and through nginx, and every
 * other frame on that link is small text -- putting a megabyte of base64
 * through it would be the wrong shape even where it fits. The worker already
 * speaks HTTP to this server for artifacts, so it uploads the same way and the
 * job frame carries only the id.
 *
 * The two directions authenticate differently, and deliberately:
 *
 *   * **Upload** is the worker, holding WORKER_TOKEN. It has no account.
 *   * **Download** is a browser, and is checked against the conversation's
 *     owner. Serving images on the worker token instead would make every
 *     user's pictures readable by anything holding a single shared secret.
 */

function tokenMatches(candidate: string): boolean {
  const a = Buffer.from(candidate);
  const b = Buffer.from(config.workerToken);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

function workerAuthorised(req: FastifyRequest): boolean {
  const auth = req.headers['authorization'];
  if (typeof auth !== 'string' || !auth.startsWith('Bearer ')) return false;
  return tokenMatches(auth.slice(7).trim());
}

/** Guards against a decoded payload that is not actually a PNG. */
function looksLikePng(buf: Buffer): boolean {
  return (
    buf.length > 8 &&
    buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47 &&
    buf[4] === 0x0d && buf[5] === 0x0a && buf[6] === 0x1a && buf[7] === 0x0a
  );
}

// A 1024px PNG is one to two megabytes; this leaves room for a large one
// without letting a misbehaving worker write unbounded data to the disk.
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

export function registerImageRoutes(app: FastifyInstance): void {
  /** Worker → server: store one finished image and return its id. */
  app.post(
    '/api/images',
    {
      // Fastify's default body limit is 1 MB and applies per route. A 768px
      // PNG is one to two megabytes, and base64 adds a third on top -- so a
      // picture that generated perfectly was rejected before this handler ran,
      // with a Fastify error that says nothing about images. The cap below is
      // MAX_IMAGE_BYTES with room for base64 expansion and the JSON around it.
      bodyLimit: Math.ceil(MAX_IMAGE_BYTES * 1.4) + 64 * 1024,
    },
    async (req, reply) => {
    if (!workerAuthorised(req)) {
      return reply.code(401).send({ error: 'unauthorized', message: 'Worker token required.' });
    }

    const body = (req.body ?? {}) as {
      jobId?: string;
      b64?: string;
      mime?: string;
      width?: number;
      height?: number;
      seed?: number;
      steps?: number;
      model?: string;
    };
    if (!body.jobId || !body.b64) {
      return reply.code(400).send({ error: 'bad_request', message: 'jobId and b64 are required.' });
    }

    // The worker knows the job, not the message. Resolving it here keeps the
    // worker ignorant of the server's schema, which is the same reason it
    // reports tokens rather than writing inference_runs itself.
    const job = jobManager.get(body.jobId);
    if (!job) {
      return reply.code(404).send({ error: 'unknown_job', message: 'No such job.' });
    }

    const raw = Buffer.from(body.b64, 'base64');
    if (raw.length === 0 || raw.length > MAX_IMAGE_BYTES) {
      return reply.code(413).send({
        error: 'too_large',
        message: `Image must be between 1 byte and ${MAX_IMAGE_BYTES} bytes.`,
      });
    }
    if (!looksLikePng(raw)) {
      return reply.code(400).send({ error: 'not_png', message: 'Payload is not a PNG.' });
    }

    const id = randomUUID();
    // Foldered by date so one directory does not grow to hundreds of thousands
    // of entries over the life of the deployment.
    const day = new Date().toISOString().slice(0, 10);
    const dir = join(config.imagesDir, day);
    const relative = join(day, `${id}.png`);
    const absolute = join(config.imagesDir, relative);

    await mkdir(dir, { recursive: true });
    await writeFile(absolute, raw);

    try {
      await query(
        `INSERT INTO generated_images
           (id, message_id, job_id, path, mime, width, height, seed, steps, model, bytes)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
        [
          id,
          job.messageId,
          body.jobId,
          relative,
          body.mime ?? 'image/png',
          body.width ?? null,
          body.height ?? null,
          body.seed ?? null,
          body.steps ?? null,
          body.model ?? job.model ?? null,
          raw.length,
        ],
      );
    } catch (err) {
      // Never leave a file the database does not know about: it would be
      // unreachable and would never be cleaned up.
      await unlink(absolute).catch(() => undefined);
      throw err;
    }

    app.log.info({ id, jobId: body.jobId, bytes: raw.length }, '[images] stored');
    return { ok: true, id, bytes: raw.length };
  },
  );

  /** Browser → server: the image itself, for whoever owns the conversation. */
  app.get('/api/images/:id', async (req, reply) => {
    const { id } = req.params as { id: string };
    if (!/^[0-9a-f-]{36}$/i.test(id)) {
      return reply.code(404).send({ error: 'not_found' });
    }

    const { rows } = await query<{ path: string; mime: string }>(
      `SELECT g.path, g.mime
         FROM generated_images g
         JOIN messages m      ON m.id = g.message_id
         JOIN conversations c ON c.id = m.conversation_id
        WHERE g.id = $1 AND ${ownedByUser('c', '$2', '$3')}
        LIMIT 1`,
      [id, ...ownerParams(req)],
    );
    // Deliberately 404 rather than 403 for someone else's image: a 403 confirms
    // the id exists, which is the same reason job lookups answer this way.
    if (rows.length === 0) return reply.code(404).send({ error: 'not_found' });

    const absolute = resolve(join(config.imagesDir, rows[0].path));
    if (!absolute.startsWith(resolve(config.imagesDir))) {
      return reply.code(400).send({ error: 'bad_path' });
    }

    try {
      const info = await stat(absolute);
      reply
        .header('Content-Type', rows[0].mime || 'image/png')
        .header('Content-Length', String(info.size))
        // Immutable: the id is unique per image and the bytes never change.
        .header('Cache-Control', 'private, max-age=31536000, immutable');
      return reply.send(createReadStream(absolute));
    } catch {
      return reply.code(404).send({ error: 'missing_file', message: 'The image file is gone.' });
    }
  });
}
