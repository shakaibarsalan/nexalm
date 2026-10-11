import { createReadStream } from 'node:fs';
import { rename, writeFile } from 'node:fs/promises';
import { stat } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { timingSafeEqual } from 'node:crypto';
import type { FastifyInstance, FastifyRequest } from 'fastify';
import { config } from '../config.js';

/**
 * Serves prebuilt worker artifacts to remote GPU hosts.
 *
 * Exists because compiling llama.cpp on the worker is pure waste: the output is
 * identical every time, and Colab's GPU runtime gives ~2 vCPUs, so the build
 * takes over an hour there — during which the GPU sits idle and Colab's
 * watchdog may disconnect the session for exactly that reason.
 *
 * sm_75 is Turing, which covers the T4 on both Kaggle and Colab, so one build
 * serves either platform.
 *
 * Authenticated with WORKER_TOKEN rather than a user session: the caller is a
 * bootstrapping worker that has no browser and no account.
 */

function tokenMatches(candidate: string): boolean {
  const a = Buffer.from(candidate);
  const b = Buffer.from(config.workerToken);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

function presentedToken(req: FastifyRequest): string | undefined {
  const auth = req.headers['authorization'];
  if (typeof auth === 'string' && auth.startsWith('Bearer ')) return auth.slice(7).trim();
  // Query fallback: notebook shells reach for curl/wget long before they reach
  // for a header, and this is the first call a worker makes.
  const q = (req.query as { token?: string }).token;
  return typeof q === 'string' ? q : undefined;
}

/** Only these names are servable; anything else is a path-traversal attempt. */
const ALLOWED = new Set([
  'llamacpp-cuda-sm75.tar.gz',
  // Image generation. Upstream publishes Linux binaries for CPU, ROCm and
  // Vulkan but not CUDA, so there is nothing to download for an NVIDIA card
  // and this is built once and served from here -- exactly as llama.cpp is.
  'sdcpp-cuda-sm75.tar.gz',
]);

export function registerArtifactRoutes(app: FastifyInstance): void {
  app.get('/api/artifacts/:name', async (req, reply) => {
    const token = presentedToken(req);
    if (!token || !tokenMatches(token)) {
      return reply.code(401).send({ error: 'unauthorized', message: 'Worker token required.' });
    }

    const name = basename((req.params as { name: string }).name);
    if (!ALLOWED.has(name)) {
      return reply.code(404).send({ error: 'not_found', message: 'No such artifact.' });
    }

    const path = resolve(join(config.artifactsDir, name));
    if (!path.startsWith(resolve(config.artifactsDir))) {
      return reply.code(400).send({ error: 'bad_path' });
    }

    try {
      const info = await stat(path);
      reply
        .header('Content-Type', 'application/gzip')
        .header('Content-Length', String(info.size))
        .header('Cache-Control', 'public, max-age=86400');
      return reply.send(createReadStream(path));
    } catch {
      return reply.code(404).send({
        error: 'not_built',
        message: `${name} is not present on the server.`,
      });
    }
  });

  /**
   * A worker publishes a build it just produced, so the next session does not
   * repeat it.
   *
   * This exists because the build host has to be a machine of the same
   * architecture as the machines that run it, and this project's control plane
   * is not one: it is an Ampere A1 (aarch64) while both GPU hosts are x86_64.
   * A build produced on the server ran nowhere. So the only place that can
   * produce these binaries is a worker -- and without somewhere to put the
   * result, every new session would pay the ~30 minute compile again.
   *
   * Written to a temporary name and renamed into place, so a worker that dies
   * mid-upload cannot leave a truncated binary that every later session would
   * download and fail to execute.
   */
  app.post(
    '/api/artifacts/:name',
    // Per route, for the same reason as the image upload: the content-type
    // parser's limit is not the route's, and the 1 MB default turned a 65 MB
    // build into a 413 -- so every session recompiled sd-server from source,
    // which is the ~26 minutes this endpoint exists to remove.
    { bodyLimit: 512 * 1024 * 1024 },
    async (req, reply) => {
    const token = presentedToken(req);
    if (!token || !tokenMatches(token)) {
      return reply.code(401).send({ error: 'unauthorized', message: 'Worker token required.' });
    }

    const name = basename((req.params as { name: string }).name);
    if (!ALLOWED.has(name)) {
      return reply.code(404).send({ error: 'not_found', message: 'No such artifact.' });
    }

    const body = req.body;
    if (!Buffer.isBuffer(body) || body.length === 0) {
      return reply.code(400).send({
        error: 'bad_request',
        message: 'Send the tarball as application/octet-stream.',
      });
    }
    // gzip magic. A tarball that is not one would be served to every future
    // worker, so it is worth the two bytes of checking.
    if (body[0] !== 0x1f || body[1] !== 0x8b) {
      return reply.code(400).send({ error: 'not_gzip', message: 'Expected a .tar.gz.' });
    }

    const finalPath = resolve(join(config.artifactsDir, name));
    if (!finalPath.startsWith(resolve(config.artifactsDir))) {
      return reply.code(400).send({ error: 'bad_path' });
    }

    const temp = `${finalPath}.incoming`;
    await writeFile(temp, body);
    await rename(temp, finalPath);
    req.log.info({ name, bytes: body.length }, '[artifacts] published by a worker');
    return { ok: true, name, bytes: body.length };
  },
  );

  /** Lets a worker decide whether to download or build, without fetching 49 MB. */
  app.get('/api/artifacts', async (req, reply) => {
    const token = presentedToken(req);
    if (!token || !tokenMatches(token)) {
      return reply.code(401).send({ error: 'unauthorized', message: 'Worker token required.' });
    }

    const available: Array<{ name: string; bytes: number }> = [];
    for (const name of ALLOWED) {
      try {
        const info = await stat(join(config.artifactsDir, name));
        available.push({ name, bytes: info.size });
      } catch {
        /* not built yet */
      }
    }
    return { artifacts: available };
  });
}
