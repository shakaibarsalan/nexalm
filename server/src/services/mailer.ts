import { createTransport, type Transporter } from 'nodemailer';
import { query } from '../db/pool.js';
import { config } from '../config.js';

/**
 * Outbound mail: a verification code, and a welcome note once the account exists.
 *
 * Two deliberate differences between the two messages.
 *
 * The code email carries **no branding at all** — no logo, no colours, no
 * marketing. A one-time code is a security message, and the plainer it is the
 * harder it is to imitate; a recipient who has learned that codes arrive as
 * bare text is harder to fool with a decorated fake. It is sent as plain text
 * for the same reason, and because every client renders it identically.
 *
 * The welcome email is the opposite: it is the first thing a new account sees,
 * so it carries the identity — logo and name in the header, the message in the
 * body, company details in the footer.
 */

const CODE_TTL_MINUTES = 15;

/** Most messages one address may receive in a rolling hour. */
const MAX_PER_HOUR = 3;

let transport: Transporter | null = null;

export function isConfigured(): boolean {
  return Boolean(config.smtp.host && config.smtp.user && config.smtp.pass);
}

function mailer(): Transporter {
  if (!transport) {
    transport = createTransport({
      host: config.smtp.host,
      port: config.smtp.port,
      // 465 is implicit TLS; 587 upgrades with STARTTLS, which nodemailer does
      // automatically when secure is false.
      secure: config.smtp.port === 465,
      auth: { user: config.smtp.user, pass: config.smtp.pass },

      // Pooled, and kept warm. Most of the wait on a verification code is the
      // TLS handshake and SMTP auth against Gmail, not the message itself, and
      // paying that on every sign-up makes the code feel slow to arrive. A
      // pooled connection pays it once and reuses it.
      pool: true,
      maxConnections: 2,
      maxMessages: 50,
      connectionTimeout: 10_000,
      greetingTimeout: 8_000,
      socketTimeout: 20_000,
    });

    // Open the connection now rather than on the first sign-up, so the person
    // who happens to register first is not the one who pays for the handshake.
    transport.verify().catch(() => {
      /* a bad credential surfaces on the first real send, with a real message */
    });
  }
  return transport;
}

export interface SendBudget {
  allowed: boolean;
  sentThisHour: number;
  retryAfterMinutes: number;
}

/**
 * Whether another message may go to this address right now.
 *
 * Counted in the database rather than in memory: an in-memory counter resets on
 * every deploy, so anyone willing to wait for a restart could ignore the cap.
 */
export async function checkBudget(email: string): Promise<SendBudget> {
  const { rows } = await query<{ n: string; oldest: Date | null }>(
    `SELECT count(*) AS n, min(sent_at) AS oldest
       FROM email_sends
      WHERE lower(email) = lower($1) AND sent_at > now() - interval '1 hour'`,
    [email],
  );
  const sentThisHour = Number(rows[0]?.n ?? 0);
  const oldest = rows[0]?.oldest;
  const retryAfterMinutes = oldest
    ? Math.max(1, 60 - Math.floor((Date.now() - new Date(oldest).getTime()) / 60_000))
    : 0;
  return { allowed: sentThisHour < MAX_PER_HOUR, sentThisHour, retryAfterMinutes };
}

async function recordSend(email: string, kind: string): Promise<void> {
  await query(`INSERT INTO email_sends (email, kind) VALUES ($1, $2)`, [email, kind]);
  // Keep the table from growing without bound; only the last hour is ever read.
  if (Math.random() < 0.05) {
    await query(`DELETE FROM email_sends WHERE sent_at < now() - interval '2 days'`);
  }
}

/**
 * A verification code, deliberately unbranded.
 *
 * Plain text only. No logo, no colours, no links — nothing for a forgery to
 * copy, and nothing that asks the reader to click anything.
 */
export async function sendCode(email: string, code: string): Promise<void> {
  await mailer().sendMail({
    from: config.smtp.from,
    to: email,
    subject: `${code} is your verification code`,
    text: [
      `Your verification code is ${code}`,
      '',
      `It expires in ${CODE_TTL_MINUTES} minutes and can be used once.`,
      '',
      'If you did not request this code, no account has been created and you',
      'can ignore this message. Nobody will ever ask you to share this code.',
    ].join('\n'),
  });
  await recordSend(email, 'verification');
}

/** Header, body, footer — the structure asked for, kept simple enough to survive Gmail. */
function welcomeHtml(name: string, origin: string): string {
  return `<!doctype html>
<html lang="en">
  <body style="margin:0;padding:0;background:#f4f5f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#16181f;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f8;padding:32px 12px;">
      <tr><td align="center">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e3e5eb;">

          <!-- Header: logo and name, nothing else -->
          <tr><td style="padding:22px 28px;border-bottom:1px solid #eef0f4;">
            <table role="presentation" cellpadding="0" cellspacing="0"><tr>
              <td style="padding-right:10px;vertical-align:middle;">
                <img src="${origin}/logo-mail.png" width="30" height="30" alt="" style="display:block;border:0;border-radius:7px;">
              </td>
              <td style="vertical-align:middle;font-size:19px;font-weight:600;letter-spacing:-0.2px;">NexaLM</td>
            </tr></table>
          </td></tr>

          <!-- Body -->
          <tr><td style="padding:30px 28px;font-size:15px;line-height:1.62;">
            <p style="margin:0 0 16px;font-size:18px;font-weight:600;">Welcome, ${name}.</p>
            <p style="margin:0 0 16px;">
              Your account is verified and ready. NexaLM runs open-weight language models
              on GPU hardware you control, so your conversations stay on your own
              infrastructure rather than someone else's.
            </p>
            <p style="margin:0 0 22px;">
              Sign in and pick a model to get started. Larger models need a GPU with
              enough memory, and the app tells you which ones will run before you wait
              on a download.
            </p>
            <a href="${origin}" style="display:inline-block;background:#2563eb;color:#ffffff;text-decoration:none;font-weight:600;font-size:14px;padding:11px 22px;border-radius:8px;">Open NexaLM</a>
          </td></tr>

          <!-- Footer: company information -->
          <tr><td style="padding:20px 28px;background:#fafbfc;border-top:1px solid #eef0f4;font-size:12px;line-height:1.6;color:#666e7d;">
            <p style="margin:0 0 4px;font-weight:600;color:#4b5163;">NexaLM — Your AI. Your Control.</p>
            <p style="margin:0 0 4px;">Self-hosted language model platform.</p>
            <p style="margin:0 0 4px;"><a href="${origin}" style="color:#4b5163;">${origin.replace(/^https?:\/\//, '')}</a></p>
            <p style="margin:8px 0 0;color:#8b92a1;">
              You are receiving this because an account was created with this address.
            </p>
          </td></tr>

        </table>
      </td></tr>
    </table>
  </body>
</html>`;
}

export async function sendWelcome(email: string, name: string): Promise<void> {
  const origin = config.publicOrigin.replace(/\/$/, '');
  await mailer().sendMail({
    from: config.smtp.from,
    to: email,
    subject: 'Welcome to NexaLM',
    // A text part is not optional: some clients show it, and a mail with only
    // an HTML part scores worse with spam filters.
    text: [
      `Welcome, ${name}.`,
      '',
      'Your NexaLM account is verified and ready.',
      `Sign in at ${origin} and pick a model to get started.`,
      '',
      'NexaLM — Your AI. Your Control.',
    ].join('\n'),
    html: welcomeHtml(name, origin),
  });
  await recordSend(email, 'welcome');
}

/** Open the SMTP connection ahead of the first real send. */
export async function warm(): Promise<void> {
  try {
    await mailer().verify();
  } catch {
    /* reported properly when a real send is attempted */
  }
}

export { CODE_TTL_MINUTES, MAX_PER_HOUR };
