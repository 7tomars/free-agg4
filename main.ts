// main.ts — Free model aggregator: one server, two client APIs, ten simulated free-tier providers.
//
// This server reads the brief at MAKEFUN_PROMPT_FILE, probes each provider's catalog, ranks every
// free model by the quality table's rating, routes each of the 20 client requests to the best model
// that meets its requirements, fails over across providers when one refuses, and serves both the
// OpenAI Chat Completions and Anthropic Messages APIs in JSON and streaming.
//
// FILES
//   main.ts              this file: startup, routing, request handling, registration, polling.
//   lib/providers.ts     brief parsing, types, catalog probe, catalog builder.
//   lib/wire.ts          model lists, error shapes, Anthropic↔OpenAI translation, SSE streaming.
//   lib/upstream.ts      multi-wire upstream requests, normalised answers, client response builders.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  findRating,
  getCapabilities,
  isFreeModel,
  isRecord,
  meetsRequirements,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type ProbedModel,
  type ProbeResult,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  answerOf,
  clientJson,
  clientSse,
  providerHeaders,
  providerUrl,
  requestFor,
  wantsJson,
} from './lib/upstream.ts';

// ── Constants ────────────────────────────────────────────────────────────────

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT = 5000;
const MAX_BODY = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ── State ────────────────────────────────────────────────────────────────────

const brief = readBrief(PROMPT_FILE);
let probes: ProbeResult[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIdx = 0;

function log(msg: string): void {
  process.stderr.write(`aggregator: ${msg}\n`);
}

// ── Startup: probe all providers ─────────────────────────────────────────────

async function load(): Promise<void> {
  if (brief.error) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT)),
  );
  for (const p of probes) if (!p.ok && p.error) log(p.error);
  catalog = buildCatalog(probes, brief.ratings);
}

// ── Model listing for GET /v1/models ─────────────────────────────────────────

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const m of [...probe.models].sort(byId)) {
      const row = findRating(brief.ratings, probe.provider.id, m.id);
      if (!isFreeModel(m, row)) continue;
      out.push({ id: m.id, provider: probe.provider.id, display_name: m.display_name });
    }
  }
  return out;
}

// ── Routing: quality-first candidate list ────────────────────────────────────

interface Route {
  probe: ProbeResult;
  model: ProbedModel;
  rating: number;
}

/**
 * All free models meeting the requirements, sorted best-first: by rating_milli descending,
 * then by brief's provider order, then alphabetically by model id. This gives the optimal
 * pick and a complete failover chain, deterministic from the brief alone.
 */
function candidatesFor(req: Requirement): Route[] {
  const out: Route[] = [];
  for (let pi = 0; pi < probes.length; pi++) {
    const probe = probes[pi];
    if (!probe.ok) continue;
    for (const m of [...probe.models].sort(byId)) {
      const row = findRating(brief.ratings, probe.provider.id, m.id);
      if (!isFreeModel(m, row)) continue;
      if (!meetsRequirements(req, getCapabilities(m, row))) continue;
      out.push({ probe, model: m, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const pa = probes.indexOf(a.probe);
    const pb = probes.indexOf(b.probe);
    if (pa !== pb) return pa - pb;
    return byId(a.model, b.model);
  });
}

// ── Failover tracking ────────────────────────────────────────────────────────

interface Failure {
  status: number;
  code: string;
  message: string;
  retryAfter: number;
}

const deadProviders = new Set<string>();
const refusalCounts = new Map<string, number>();

function parseFailure(status: number, retryHeader: string | null, text: string): Failure {
  let code = '';
  let message = '';
  let retryAfter = 0;
  try {
    const val: unknown = JSON.parse(text);
    if (isRecord(val) && isRecord(val.error)) {
      const e = val.error;
      if (typeof e.code === 'string' || typeof e.code === 'number') code = String(e.code);
      else if (typeof e.status === 'string') code = e.status;
      if (typeof e.message === 'string') message = e.message;
      if (typeof e.retry_after === 'number' && Number.isFinite(e.retry_after)) retryAfter = e.retry_after;
    }
  } catch {
    message = text.slice(0, 200);
  }
  const hdr = retryHeader === null ? 0 : Number(retryHeader);
  if (Number.isFinite(hdr) && hdr > 0) retryAfter = hdr;
  if (!Number.isFinite(retryAfter) || retryAfter < 0) retryAfter = 0;
  return { status, code, message, retryAfter };
}

/**
 * Track a provider failure. 402 or budget-exhaustion 429 marks it dead immediately.
 * Other refusals accumulate; 3 refusals also mark it dead.
 */
function recordFailure(route: Route, fail: Failure): void {
  const id = route.probe.provider.id;
  log(`failover ${id}/${route.model.id}: HTTP ${fail.status}${fail.code ? ` ${fail.code}` : ''}${fail.message ? `: ${fail.message.slice(0, 160)}` : ''}`);
  if (fail.status === 402 || (fail.status === 429 && /budget.*(spent|exhausted)|is out|out of budget|quota/i.test(fail.message))) {
    deadProviders.add(id);
    return;
  }
  const count = (refusalCounts.get(id) ?? 0) + 1;
  refusalCounts.set(id, count);
  if (count >= 3) deadProviders.add(id);
}

// ── Requirements from manifest ───────────────────────────────────────────────

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIdx];
  manifestIdx++;
  if (entry) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  for (const msg of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(msg) || !Array.isArray(msg.content)) continue;
    for (const b of msg.content) {
      if (isRecord(b) && (b.type === 'image' || b.type === 'image_url')) vision = true;
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: fmt === 'json_object' || fmt === 'json_schema',
  };
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > MAX_BODY) return null;
    chunks.push(buf);
  }
  if (!chunks.length) return null;
  try {
    const val: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(val) ? val : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

// ── GET /v1/models ───────────────────────────────────────────────────────────

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ── Serve one request: try candidates best-first with failover ───────────────

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const routes = candidatesFor(requirementsFor(body));
  const errType = api === 'openai' ? 'no_free_model' : 'overloaded_error';
  if (!routes.length) return sendError(res, api, 503, errType, 'no free model meets this request');

  const requested = typeof body.model === 'string' && body.model ? body.model : 'auto';
  const needsJson = wantsJson(api, body);

  for (const route of routes) {
    if (deadProviders.has(route.probe.provider.id)) continue;
    // An Anthropic-wire provider cannot carry JSON mode for an OpenAI client
    if (route.probe.provider.wire === 'anthropic' && needsJson) continue;

    const model = route.model.id;
    let upstream: Response;
    try {
      upstream = await fetch(providerUrl(ENV_URL, route.probe.provider, model), {
        method: 'POST',
        headers: providerHeaders(route.probe.provider),
        body: JSON.stringify(requestFor(route.probe.provider.wire, api, body, model)),
      });
    } catch (err) {
      log(`failover ${route.probe.provider.id}/${model}: ${(err as Error).message}`);
      continue;
    }

    if (!upstream.ok) {
      const fail = parseFailure(upstream.status, upstream.headers.get('retry-after'), await upstream.text());
      recordFailure(route, fail);
      if (fail.retryAfter > 0) await sleep(fail.retryAfter * 1000);
      continue;
    }

    // Parse the provider's JSON answer
    let payload: unknown = null;
    try { payload = await upstream.json(); } catch { payload = null; }
    const answer = answerOf(route.probe.provider.wire, payload);
    if (!answer) {
      log(`failover ${route.probe.provider.id}/${model}: answer not in wire's shape`);
      continue;
    }

    // Write the response in the client's wire
    if (body.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(clientSse(api, requested, answer));
    } else {
      sendJson(res, 200, clientJson(api, requested, answer));
    }
    return;
  }

  sendError(res, api, 503, errType, 'every free model that meets this request refused');
}

// ── POST /v1/chat/completions ────────────────────────────────────────────────

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (!body) return sendError(res, 'openai', 400, 'invalid_request_error', 'request body is not one JSON object');
  await serve('openai', body, res);
}

// ── POST /v1/messages ────────────────────────────────────────────────────────

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (!body) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'request body is not one JSON object');
  await serve('anthropic', body, res);
}

// ── Router ───────────────────────────────────────────────────────────────────

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return handleModels(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return handleChat(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return handleMessages(req, res);
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ── Server lifecycle ─────────────────────────────────────────────────────────

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const r = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!r.ok) throw new Error(`POST /register → ${r.status}: ${(await r.text()).slice(0, 500)}`);
  const reply: unknown = await r.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} requests`);
}

function poll(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const r = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (r.status !== 200) return;
        const w: unknown = await r.json();
        if (isRecord(w) && w.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch { /* retry next tick */ } finally { busy = false; }
    })();
  }, POLL_MS);
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function start(): Promise<void> {
  await load();

  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }

  if (!ENV_URL) throw new Error('MAKEFUN_ENV_URL is not set');

  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') ? 'anthropic' : 'openai', 500, 'internal_error', 'server error');
      }
    });
  });

  await listen(server);
  const addr = server.address();
  if (!addr || typeof addr === 'string') throw new Error('server has no port');
  const port = addr.port;
  if (port === 8080) throw new Error('OS picked 8080, which the env owns');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);

  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
