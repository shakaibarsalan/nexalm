import { readdir, readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pool } from './pool.js';

const here = dirname(fileURLToPath(import.meta.url));

/**
 * Migrations live outside the compiled tree so the same .sql files are used in
 * dev and in the image. Resolve against both layouts.
 */
function migrationsDir(): string {
  return process.env.MIGRATIONS_DIR ?? join(here, '..', '..', '..', 'db', 'migrations');
}

export async function runMigrations(): Promise<void> {
  const dir = migrationsDir();

  await pool.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      filename    TEXT PRIMARY KEY,
      applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `);

  const files = (await readdir(dir)).filter((f) => f.endsWith('.sql')).sort();
  const { rows } = await pool.query<{ filename: string }>('SELECT filename FROM schema_migrations');
  const applied = new Set(rows.map((r) => r.filename));

  for (const file of files) {
    if (applied.has(file)) {
      console.log(`[migrate] skip    ${file}`);
      continue;
    }
    const sql = await readFile(join(dir, file), 'utf8');
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query(sql);
      await client.query('INSERT INTO schema_migrations (filename) VALUES ($1)', [file]);
      await client.query('COMMIT');
      console.log(`[migrate] applied ${file}`);
    } catch (err) {
      await client.query('ROLLBACK');
      console.error(`[migrate] FAILED  ${file}:`, (err as Error).message);
      throw err;
    } finally {
      client.release();
    }
  }
  console.log(`[migrate] up to date (${files.length} migration(s))`);
}

// Allow `node dist/db/migrate.js` as a standalone entrypoint.
if (process.argv[1] && process.argv[1].endsWith('migrate.js')) {
  runMigrations()
    .then(() => pool.end())
    .then(() => process.exit(0))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
