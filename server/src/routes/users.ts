import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { query } from '../db/pool.js';
import { requireAdmin, requireUser } from '../middleware/auth.js';
import type { AuthUser, UserRole } from '../services/authService.js';

interface AdminUserRow {
  id: string;
  email: string;
  display_name: string;
  avatar_url: string | null;
  role: UserRole;
  google_sub: string | null;
  created_at: Date;
  last_login_at: Date | null;
  disabled_at: Date | null;
  conversation_count: number;
  session_count: number;
}

/**
 * True when removing or demoting this user would leave nobody able to
 * administer the deployment.
 */
async function isLastAdmin(userId: string): Promise<boolean> {
  const { rows } = await query<{ count: number }>(
    `SELECT count(*)::int AS count
       FROM users
      WHERE role = 'admin' AND disabled_at IS NULL AND id <> $1`,
    [userId],
  );
  return rows[0].count === 0;
}

export function registerUserRoutes(app: FastifyInstance): void {
  app.get('/api/users', async (req, reply) => {
    if (!requireAdmin(req, reply)) return reply;

    const { rows } = await query<AdminUserRow>(
      `SELECT u.id, u.email, u.display_name, u.avatar_url, u.role, u.google_sub,
              u.created_at, u.last_login_at, u.disabled_at,
              (SELECT count(*)::int FROM conversations c   WHERE c.user_id = u.id) AS conversation_count,
              (SELECT count(*)::int FROM user_sessions s
                WHERE s.user_id = u.id AND s.expires_at > now())                   AS session_count
         FROM users u
        ORDER BY u.created_at ASC`,
    );

    return {
      users: rows.map((r) => ({
        id: r.id,
        email: r.email,
        displayName: r.display_name,
        avatarUrl: r.avatar_url,
        role: r.role,
        /** Whether the account can sign in with Google, not the identity itself. */
        googleLinked: r.google_sub !== null,
        createdAt: r.created_at.toISOString(),
        lastLoginAt: r.last_login_at ? r.last_login_at.toISOString() : null,
        disabledAt: r.disabled_at ? r.disabled_at.toISOString() : null,
        conversationCount: r.conversation_count,
        activeSessions: r.session_count,
      })),
    };
  });

  app.patch('/api/users/:id', async (req, reply) => {
    const actor = requireAdmin(req, reply);
    if (!actor) return reply;

    const { id } = req.params as { id: string };
    const body = (req.body ?? {}) as { role?: unknown; disabled?: unknown };

    const wantsRole = body.role !== undefined;
    const wantsDisabled = body.disabled !== undefined;
    if (!wantsRole && !wantsDisabled) {
      return reply.code(400).send({ error: 'no_fields', message: 'Nothing to update.' });
    }
    if (wantsRole && body.role !== 'admin' && body.role !== 'user') {
      return reply.code(400).send({ error: 'invalid_role', message: 'Role must be admin or user.' });
    }
    if (wantsDisabled && typeof body.disabled !== 'boolean') {
      return reply
        .code(400)
        .send({ error: 'invalid_disabled', message: 'Disabled must be a boolean.' });
    }

    const { rows: targets } = await query<{ role: UserRole; disabled_at: Date | null }>(
      `SELECT role, disabled_at FROM users WHERE id = $1`,
      [id],
    );
    const target = targets[0];
    if (!target) return reply.code(404).send({ error: 'not_found', message: 'No such user.' });

    // Privilege changes never apply to yourself: it is how an admin would grant
    // themselves more than they have, and how the last admin locks everyone out.
    if (id === actor.id) {
      if (wantsRole && body.role !== target.role) {
        return reply
          .code(403)
          .send({ error: 'self_role_change', message: 'You cannot change your own role.' });
      }
      if (body.disabled === true) {
        return reply
          .code(403)
          .send({ error: 'self_disable', message: 'You cannot disable your own account.' });
      }
    }

    const losesAdmin =
      (wantsRole && target.role === 'admin' && body.role === 'user') ||
      (body.disabled === true && target.role === 'admin');
    if (losesAdmin && (await isLastAdmin(id))) {
      return reply
        .code(409)
        .send({ error: 'last_admin', message: 'The last administrator cannot be removed.' });
    }

    const sets: string[] = [];
    const params: unknown[] = [id];
    if (wantsRole) {
      params.push(body.role);
      sets.push(`role = $${params.length}`);
    }
    if (wantsDisabled) {
      sets.push(`disabled_at = ${body.disabled ? 'now()' : 'NULL'}`);
    }

    const { rows } = await query<{
      id: string;
      email: string;
      display_name: string;
      role: UserRole;
      disabled_at: Date | null;
    }>(
      `UPDATE users SET ${sets.join(', ')} WHERE id = $1
       RETURNING id, email, display_name, role, disabled_at`,
      params,
    );
    const updated = rows[0];

    // A disabled account must lose its live sessions, not just its next login.
    if (body.disabled === true) {
      await query(`DELETE FROM user_sessions WHERE user_id = $1`, [id]);
    }

    return {
      user: {
        id: updated.id,
        email: updated.email,
        displayName: updated.display_name,
        role: updated.role,
        disabledAt: updated.disabled_at ? updated.disabled_at.toISOString() : null,
      },
    };
  });

  /** Deletes the account and, by cascade, its conversations and sessions. */
  app.delete('/api/users/:id', async (req, reply) => {
    const actor = requireAdmin(req, reply);
    if (!actor) return reply;

    const { id } = req.params as { id: string };
    if (id === actor.id) {
      return reply
        .code(403)
        .send({ error: 'self_delete', message: 'You cannot delete your own account.' });
    }

    const { rows: targets } = await query<{ role: UserRole }>(
      `SELECT role FROM users WHERE id = $1`,
      [id],
    );
    if (!targets[0]) return reply.code(404).send({ error: 'not_found', message: 'No such user.' });

    if (targets[0].role === 'admin' && (await isLastAdmin(id))) {
      return reply
        .code(409)
        .send({ error: 'last_admin', message: 'The last administrator cannot be deleted.' });
    }

    await query(`DELETE FROM users WHERE id = $1`, [id]);
    return { ok: true };
  });
}
