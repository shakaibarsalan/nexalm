import { lookup } from 'node:dns/promises';
import { isIP } from 'node:net';
import { config } from '../config.js';

/**
 * Looking things up on the web, for models that cannot.
 *
 * A local model's knowledge stops at its training cut-off and it has no way to
 * know that. Asking it about anything recent produces a confident answer rather
 * than an admission, which is worse than no answer at all. This fetches real
 * sources and puts them in front of the model with citations, so the reply can
 * be checked.
 *
 * **Why not MCP.** MCP is the obvious-sounding answer and was considered: it is
 * a protocol for connecting an AI application to tools. But every MCP search
 * server still needs the same upstream API key this does, so MCP would add a
 * protocol hop without removing the requirement, and the local models' support
 * for tool-calling is uneven enough that the model deciding *when* to search is
 * not yet reliable. The `SearchProvider` interface below is the seam an MCP
 * client would implement, so adding one later is additive rather than a rewrite.
 */

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

interface SearchProvider {
  name: string;
  configured(): boolean;
  search(query: string, limit: number): Promise<SearchResult[]>;
}

const UA = 'NexaLM/1.0 (self-hosted assistant)';

async function getJson(url: string, headers: Record<string, string> = {}, timeoutMs = 12_000) {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': UA, Accept: 'application/json', ...headers },
      signal: abort.signal,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return (await res.json()) as unknown;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Brave Search. Needs a key, and is the only one of these that is a real web
 * search: DuckDuckGo's keyless API answers only entity queries and returns
 * nothing for most questions, and public SearXNG instances disable JSON.
 */
const brave: SearchProvider = {
  name: 'brave',
  configured: () => Boolean(config.search.braveKey),
  async search(query, limit) {
    const url =
      'https://api.search.brave.com/res/v1/web/search?q=' +
      encodeURIComponent(query) +
      `&count=${Math.min(limit, 10)}`;
    const data = (await getJson(url, {
      'X-Subscription-Token': config.search.braveKey,
      Accept: 'application/json',
    })) as { web?: { results?: { title: string; url: string; description?: string }[] } };

    return (data.web?.results ?? []).slice(0, limit).map((r) => ({
      title: r.title,
      url: r.url,
      snippet: stripTags(r.description ?? ''),
      source: 'Brave',
    }));
  },
};

/**
 * Wikipedia, which needs no key and always works.
 *
 * Not a general web search, but it is genuinely good at the factual questions a
 * small model most often gets wrong, and it means the feature degrades to
 * something useful rather than to nothing when no key is set.
 */
const wikipedia: SearchProvider = {
  name: 'wikipedia',
  configured: () => true,
  async search(query, limit) {
    const searchUrl =
      'https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=' +
      Math.min(limit, 5) +
      '&srsearch=' +
      encodeURIComponent(query);
    const found = (await getJson(searchUrl)) as {
      query?: { search?: { title: string; snippet?: string }[] };
    };
    const titles = (found.query?.search ?? []).slice(0, limit).map((h) => h.title);
    if (titles.length === 0) return [];

    // One extra call gets real extracts rather than the search snippets, which
    // are truncated mid-sentence and full of markup.
    const extractUrl =
      'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts' +
      '&exintro=1&explaintext=1&redirects=1&titles=' +
      encodeURIComponent(titles.join('|'));
    const pages = (await getJson(extractUrl)) as {
      query?: { pages?: Record<string, { title: string; extract?: string }> };
    };

    return Object.values(pages.query?.pages ?? {})
      .filter((p) => p.extract)
      .slice(0, limit)
      .map((p) => ({
        title: p.title,
        url: `https://en.wikipedia.org/wiki/${encodeURIComponent(p.title.replace(/ /g, '_'))}`,
        snippet: (p.extract ?? '').slice(0, 700),
        source: 'Wikipedia',
      }));
  },
};

/**
 * Apify's RAG Web Browser: real Google results with the pages already fetched.
 *
 * The best quality of the three by a distance -- it returns the actual sources a
 * person would find, not just encyclopaedia articles -- at the cost of about
 * thirteen seconds per search, which is actor start-up rather than scraping
 * (measured: turning scraping off saved 1.5s and replaced every real URL with a
 * Google redirect, which is useless for a citation).
 *
 * Each call spends Apify credit, which is why results are cached below.
 */
const apify: SearchProvider = {
  name: 'apify',
  configured: () => Boolean(config.search.apifyToken),
  async search(query, limit) {
    const url =
      'https://api.apify.com/v2/acts/apify~rag-web-browser/run-sync-get-dataset-items?token=' +
      encodeURIComponent(config.search.apifyToken);

    const abort = new AbortController();
    // Steady state is 7-16s; a cold actor start has been seen at 38. Bounded so
    // an unlucky cold start falls through to Wikipedia in a second and a half
    // rather than making someone wait most of a minute for their reply.
    const timer = setTimeout(() => abort.abort(), 25_000);
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'User-Agent': UA },
        body: JSON.stringify({
          query,
          maxResults: Math.min(limit, 5),
          outputFormats: ['markdown'],
          scrapingTool: 'raw-http',
        }),
        signal: abort.signal,
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const items = (await res.json()) as {
        metadata?: { title?: string; url?: string; description?: string };
        markdown?: string;
        searchResult?: { title?: string; description?: string };
      }[];

      return items
        .map((it) => {
          // metadata.url is the page itself; searchResult.url is a Google
          // redirect and cannot be shown to anyone as a source.
          const link = it.metadata?.url ?? '';
          const body = (it.markdown ?? '').replace(/\s+/g, ' ').trim();
          return {
            title: it.metadata?.title || it.searchResult?.title || link,
            url: link,
            snippet: (body || it.searchResult?.description || '').slice(0, 900),
            source: 'Web',
          };
        })
        .filter((r) => r.url.startsWith('http') && r.snippet.length > 40)
        .slice(0, limit);
    } finally {
      clearTimeout(timer);
    }
  },
};

const PROVIDERS = [apify, brave, wikipedia];

/**
 * Recent searches, so the same question does not pay twice.
 *
 * A search costs about thirteen seconds and a unit of Apify credit, and people
 * retry and rephrase constantly. Small and short-lived on purpose: this is here
 * to absorb repeats within one conversation, not to serve stale news.
 */
const CACHE_TTL_MS = 5 * 60 * 1000;
const CACHE_MAX = 100;
const cache = new Map<string, { at: number; results: SearchResult[] }>();

function stripTags(s: string): string {
  return s.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
}

/**
 * The part of a message worth searching for.
 *
 * Sending the whole message searches the instructions too: "What is a
 * mixture-of-experts model? Cite your sources." returned articles about large
 * language models in general, where the question alone finds the right one.
 * The question is the query; anything after it is addressed to the model.
 */
export function searchQuery(message: string): string {
  const trimmed = message.trim();
  const q = trimmed.indexOf('?');
  const base = q > 8 ? trimmed.slice(0, q + 1) : trimmed;
  // Long enough for a real question, short enough that a pasted essay does not
  // become the query.
  return base.slice(0, 300);
}

export function searchConfigured(): boolean {
  return PROVIDERS.some((p) => p.configured());
}

/** Which provider will actually answer, for the UI to say so honestly. */
export function activeProvider(): string {
  return (PROVIDERS.find((p) => p.configured()) ?? wikipedia).name;
}

/**
 * Search, preferring the best provider that is configured.
 *
 * Falls through on failure rather than propagating: a search that fails should
 * degrade the answer, never lose the user's message.
 */
export async function search(query: string, limit = 5): Promise<SearchResult[]> {
  const key = `${limit}:${query.toLowerCase()}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.results;

  for (const provider of PROVIDERS) {
    if (!provider.configured()) continue;
    try {
      const results = await provider.search(query, limit);
      if (results.length > 0) {
        if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value as string);
        cache.set(key, { at: Date.now(), results });
        return results;
      }
    } catch {
      /* try the next provider */
    }
  }
  return [];
}

/**
 * Render results as a briefing block for the system prompt.
 *
 * Numbered, with the URL on each, because a model given sources without a way
 * to refer to them invents citations instead. The instruction to say when the
 * sources do not answer the question is the important line: without it a model
 * handed five irrelevant results will still confidently use them.
 */
export function asContextBlock(query: string, results: SearchResult[]): string {
  if (results.length === 0) return '';
  const lines = results.map(
    (r, i) => `[${i + 1}] ${r.title}\n    ${r.url}\n    ${r.snippet.slice(0, 600)}`,
  );
  return [
    `Web search results for "${query}", retrieved just now.`,
    'Use them when they are relevant and cite the number, like [1].',
    'If they do not answer the question, say so rather than guessing from them.',
    '',
    ...lines,
  ].join('\n');
}

/**
 * Whether a URL is safe for the server to fetch.
 *
 * The server sits inside a Docker network alongside another tenant's database,
 * so a fetch tool that follows any URL a model produces is an SSRF hole with a
 * language model attached to it. Only http(s), only public addresses, and the
 * hostname is resolved first so a public name pointing at 127.0.0.1 is caught.
 */
export async function isFetchable(raw: string): Promise<{ ok: boolean; reason?: string }> {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return { ok: false, reason: 'not a URL' };
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return { ok: false, reason: 'only http and https are allowed' };
  }

  const host = url.hostname;
  const addresses = isIP(host) ? [host] : (await lookup(host, { all: true })).map((a) => a.address);

  for (const addr of addresses) {
    if (isPrivate(addr)) return { ok: false, reason: 'resolves to a private address' };
  }
  return { ok: true };
}

function isPrivate(addr: string): boolean {
  if (addr === '::1' || addr.startsWith('fc') || addr.startsWith('fd') || addr.startsWith('fe80')) {
    return true;
  }
  const p = addr.split('.').map(Number);
  if (p.length !== 4 || p.some(Number.isNaN)) return false;
  const [a, b] = p;
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 100 && b >= 64 && b <= 127)
  );
}

/** Fetch a page and reduce it to readable text. */
export async function fetchPage(raw: string, maxChars = 6000): Promise<string> {
  const allowed = await isFetchable(raw);
  if (!allowed.ok) throw new Error(`Refusing to fetch that address: ${allowed.reason}.`);

  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 15_000);
  try {
    const res = await fetch(raw, {
      headers: { 'User-Agent': UA, Accept: 'text/html,text/plain' },
      signal: abort.signal,
      redirect: 'follow',
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const type = res.headers.get('content-type') ?? '';
    if (!type.includes('text/') && !type.includes('json')) {
      throw new Error(`that address returned ${type || 'an unknown type'}, not text`);
    }
    const html = (await res.text()).slice(0, 400_000);
    return stripTags(
      html
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<\/(p|div|li|h[1-6]|tr)>/gi, '\n'),
    ).slice(0, maxChars);
  } finally {
    clearTimeout(timer);
  }
}
