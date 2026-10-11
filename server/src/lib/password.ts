import { randomBytes, scrypt, timingSafeEqual } from 'node:crypto';

/**
 * Password hashing on `node:crypto` scrypt.
 *
 * bcrypt/argon2 would be the obvious picks, but both are native addons and this
 * API ships as node:22-alpine on an ARM host — every deploy would have to
 * compile them against musl. scrypt is memory-hard and built in, which is the
 * better trade here.
 */

interface ScryptParams {
  N: number;
  r: number;
  p: number;
}

/**
 * N=16384/r=8 costs ~16 MB and ~50-80 ms per hash on the deployment host. Going
 * to 2^15 doubles both, which starts to matter on a shared 11 GB box where a
 * burst of login attempts would each hold their own buffer.
 */
const DEFAULT_PARAMS: ScryptParams = { N: 16_384, r: 8, p: 1 };
const KEY_LEN = 64;
const SALT_LEN = 16;

/** scrypt refuses to allocate past `maxmem`; 128*N*r must fit with headroom. */
function maxmemFor(params: ScryptParams): number {
  return 256 * params.N * params.r;
}

function derive(password: string, salt: Buffer, params: ScryptParams): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    // NFKC so the same typed password matches across platforms that normalise
    // accented characters differently.
    scrypt(
      password.normalize('NFKC'),
      salt,
      KEY_LEN,
      { ...params, maxmem: maxmemFor(params) },
      (err, key) => (err ? reject(err) : resolve(key)),
    );
  });
}

/** Self-describing so the cost parameters can be raised without invalidating old hashes. */
export async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(SALT_LEN);
  const key = await derive(password, salt, DEFAULT_PARAMS);
  const { N, r, p } = DEFAULT_PARAMS;
  return `scrypt$${N}$${r}$${p}$${salt.toString('base64')}$${key.toString('base64')}`;
}

export async function verifyPassword(password: string, stored: string | null): Promise<boolean> {
  if (!stored) return false;

  const parts = stored.split('$');
  if (parts.length !== 6 || parts[0] !== 'scrypt') return false;

  const N = Number(parts[1]);
  const r = Number(parts[2]);
  const p = Number(parts[3]);
  if (!Number.isInteger(N) || !Number.isInteger(r) || !Number.isInteger(p)) return false;
  // A corrupt or hostile record must not talk us into a huge allocation.
  if (N < 1024 || N > 1 << 20 || r < 1 || r > 32 || p < 1 || p > 16) return false;

  const salt = Buffer.from(parts[4], 'base64');
  const expected = Buffer.from(parts[5], 'base64');
  if (salt.length === 0 || expected.length !== KEY_LEN) return false;

  const actual = await derive(password, salt, { N, r, p });
  return timingSafeEqual(actual, expected);
}

export const MIN_PASSWORD_LENGTH = 10;
/** scrypt reads the whole input, so an unbounded password is a cheap DoS. */
export const MAX_PASSWORD_LENGTH = 200;

/**
 * Every rule a password must satisfy, with the wording shown to the user.
 *
 * Exported so the browser can show the same list live while someone types
 * instead of rejecting the whole thing after the fact — a checklist that fills
 * in as you go is far less annoying than a form that says "too weak" and makes
 * you guess which part.
 */
export const PASSWORD_RULES: { id: string; label: string; test: (v: string) => boolean }[] = [
  { id: 'length', label: `At least ${MIN_PASSWORD_LENGTH} characters`, test: (v) => v.length >= MIN_PASSWORD_LENGTH },
  { id: 'lower', label: 'A lowercase letter', test: (v) => /[a-z]/.test(v) },
  { id: 'upper', label: 'An uppercase letter', test: (v) => /[A-Z]/.test(v) },
  { id: 'digit', label: 'A number', test: (v) => /\d/.test(v) },
  { id: 'symbol', label: 'A symbol', test: (v) => /[^A-Za-z0-9]/.test(v) },
];

export function passwordProblem(password: unknown): string | undefined {
  if (typeof password !== 'string') return 'Password is required.';
  if (password.length < MIN_PASSWORD_LENGTH) {
    return `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`;
  }
  if (password.length > MAX_PASSWORD_LENGTH) {
    return `Password must be at most ${MAX_PASSWORD_LENGTH} characters.`;
  }
  // Named individually rather than as "too weak": a rejection that does not say
  // which rule failed leaves the person guessing, and they usually guess wrong.
  const missing = PASSWORD_RULES.filter((r) => !r.test(password)).map((r) => r.label.toLowerCase());
  if (missing.length > 0) {
    return `Password still needs: ${missing.join(', ')}.`;
  }
  return undefined;
}

// Deliberately loose: the only reliable proof an address exists is delivering
// to it, so this rejects obvious typos and nothing else.
const EMAIL_RE = /^[^\s@]+@[^\s@.]+(\.[^\s@.]+)+$/;

/** Returns the normalised address, or undefined when it is not usable. */
export function normalizeEmail(email: unknown): string | undefined {
  if (typeof email !== 'string') return undefined;
  const trimmed = email.trim().toLowerCase();
  if (trimmed.length === 0 || trimmed.length > 254) return undefined;
  return EMAIL_RE.test(trimmed) ? trimmed : undefined;
}
