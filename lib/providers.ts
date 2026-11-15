// lib/providers.ts — Brief parsing, provider probing, catalog building.
//
// Reads the three JSON documents embedded in MAKEFUN_PROMPT_FILE (provider directory, quality table,
// workload manifest), probes each provider's catalog route in its own wire, and builds the catalog
// document that --list prints and /register carries.

import { readFileSync } from 'node:fs';

// ── Types ────────────────────────────────────────────────────────────────────

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderEntry[];
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsDoc {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadDoc {
  schema: string;
  manifest: WorkloadEntry[];
}

export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsDoc | null;
  workload: WorkloadDoc | null;
  error: string | null;
}

export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface ProbeResult {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

// ── Schema constants ─────────────────────────────────────────────────────────

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

// ── Helpers ──────────────────────────────────────────────────────────────────

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

// ── JSON document extraction from mixed text ─────────────────────────────────

/** Find the closing brace of the JSON object starting at `start`. */
function findObjectEnd(text: string, start: number): number {
  let depth = 0;
  let inStr = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (c === '\\') i++;
      else if (c === '"') inStr = false;
    } else if (c === '"') {
      inStr = true;
    } else if (c === '{') {
      depth++;
    } else if (c === '}') {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Extract every top-level JSON object with a `schema` string from the text. */
function extractDocuments(text: string): Json[] {
  const docs: Json[] = [];
  for (let pos = text.indexOf('{'); pos !== -1; pos = text.indexOf('{', pos + 1)) {
    const end = findObjectEnd(text, pos);
    if (end === -1) continue;
    try {
      const parsed: unknown = JSON.parse(text.slice(pos, end + 1));
      if (isRecord(parsed) && typeof parsed.schema === 'string') docs.push(parsed);
    } catch {
      // Not valid JSON; continue scanning.
    }
  }
  return docs;
}

// ── Provider directory parsing ───────────────────────────────────────────────

function parseProvider(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (!id || !wire) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base = str(v.path) || str(v.base_path) || str(v.base_url) || str(v.base);
  return {
    id,
    name: str(v.name) || id,
    wire,
    auth,
    base,
    key_env: str(v.key_env),
    key: str(v.key),
  };
}

// ── Quality table parsing ────────────────────────────────────────────────────

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (!model || !provider) return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

// ── Workload manifest parsing ────────────────────────────────────────────────

function parseWorkloadEntry(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (!id) return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: req.tools === true,
      vision: req.vision === true,
      json: req.json === true,
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context) ? { min_context: req.min_context } : {}),
    },
  };
}

// ── Brief reader ─────────────────────────────────────────────────────────────

export function readBrief(path: string): Brief {
  const emptyDir: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (!path) return { directory: emptyDir, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };

  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory: emptyDir, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }

  const docs = extractDocuments(text);

  // Provider directory
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (!dirDoc || !Array.isArray(dirDoc.providers)) {
    return { directory: emptyDir, ratings: null, workload: null, error: `no ${PROVIDERS_SCHEMA} in ${path}` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p ? [p] : [];
  });

  // Ratings
  const ratDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsDoc | null =
    ratDoc && Array.isArray(ratDoc.ratings)
      ? {
          schema: RATINGS_SCHEMA,
          ratings: ratDoc.ratings.flatMap((v) => { const r = parseRating(v); return r ? [r] : []; }),
          best: Array.isArray(ratDoc.best)
            ? ratDoc.best.flatMap((v) => {
                if (!isRecord(v) || !str(v.id)) return [];
                return [{ id: str(v.id), rating_milli: numOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        }
      : null;

  // Workload
  const wlDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadDoc | null =
    wlDoc && Array.isArray(wlDoc.manifest)
      ? {
          schema: WORKLOAD_SCHEMA,
          manifest: wlDoc.manifest.flatMap((v) => { const w = parseWorkloadEntry(v); return w ? [w] : []; }),
        }
      : null;

  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

// ── URL / auth helpers ───────────────────────────────────────────────────────

export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (!b) return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (!key) return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

// ── Rating / capability lookups ──────────────────────────────────────────────

export function findRating(ratings: RatingsDoc | null, providerId: string, modelId: string): RatingRow | null {
  if (!ratings) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

export function isFreeModel(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

export function getCapabilities(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

export function meetsRequirements(req: Requirement, caps: Capabilities): boolean {
  if (req.tools === true && caps.tools !== true) return false;
  if (req.vision === true && caps.vision !== true) return false;
  if (req.json === true && caps.json !== true) return false;
  if (req.min_context !== undefined && (caps.context_length === null || caps.context_length < req.min_context)) return false;
  return true;
}

// ── Catalog probing ──────────────────────────────────────────────────────────

/** Check if all pricing values are zero (free). */
function isZeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (!pricing) return null;
  const vals = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim()) return [Number(x)];
    return [];
  });
  if (!vals.length) return null;
  return vals.every((n) => Number.isFinite(n) && n === 0);
}

/** Parse one model from a provider's listing in its wire's shape. */
function parseListedModel(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (!id) return null;
  const ctx = numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: str(v.display_name) || str(v.displayName) || id,
    free: boolOrNull(v.free) ?? isZeroPriced(v),
    context_length: ctx,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Extract models from a catalog response body in the given wire's shape. */
function parseModelList(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const arr = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(arr)) return [];
  return arr.flatMap((v) => { const m = parseListedModel(wire, v); return m ? [m] : []; });
}

/** Probe one provider's catalog route. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<ProbeResult> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} → ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: parseModelList(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

// ── Catalog builder ──────────────────────────────────────────────────────────

const byModelId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * Build the catalog from probes and ratings. base_url is the provider's own path (/p/<id>),
 * not joined to envUrl — the catalog must be deterministic across live/replay since the env's
 * origin changes between passes.
 */
export function buildCatalog(probes: readonly ProbeResult[], ratings: RatingsDoc | null): CatalogReport {
  const out: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const m of [...probe.models].sort(byModelId)) {
      if (seen.has(m.id)) continue;
      seen.add(m.id);
      const row = findRating(ratings, probe.provider.id, m.id);
      models.push({
        id: m.id,
        free: isFreeModel(m, row),
        context_length: row?.context_length ?? m.context_length ?? null,
      });
    }
    out.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers: out };
}
