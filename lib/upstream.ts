// lib/upstream.ts — Multi-wire upstream: build a provider request in any wire, normalise its answer,
// and render the client response in JSON or SSE for either client wire.
//
// Speaks all three provider wires (openai, anthropic, gemini) and both client APIs. One normalised
// answer { text, stop, input, output } crosses between them so the client sees the provider's text
// whole, its usage counts, and a stop reason of the same family.
//
// Every function is a pure transform of its arguments — no clock, no randomness.

import {
  authHeaders,
  isRecord,
  joinBase,
  type Json,
  type ProviderEntry,
  type Wire,
} from './providers.ts';
import { anthropicToOpenAI, sseEvent } from './wire.ts';

// ── Normalised answer ────────────────────────────────────────────────────────

export interface ProviderAnswer {
  text: string;
  stop: string | null;
  input: number;
  output: number;
  /** Full tool_calls from the provider (OpenAI wire), for pass-through. */
  toolCalls: Json[] | null;
  /** Full Anthropic content blocks, for pass-through. */
  anthropicContent: Json[] | null;
}

const FIXED_CREATED = 1767225600;

function nz(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function strList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

// ── Image conversion helpers ─────────────────────────────────────────────────

function imageToAnthropicSource(url: string): Json | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m) return { type: 'base64', media_type: m[1] || 'image/png', data: m[2] };
  if (/^https?:\/\//i.test(url)) return { type: 'url', url };
  return null;
}

function imageToGeminiPart(url: string): Json | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m) return { inlineData: { mimeType: m[1] || 'image/png', data: m[2] } };
  if (/^https?:\/\//i.test(url)) return { fileData: { fileUri: url } };
  return null;
}

// ── JSON mode detection ──────────────────────────────────────────────────────

export function wantsJson(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  return fmt === 'json_object' || fmt === 'json_schema';
}

// ── OpenAI max_tokens ────────────────────────────────────────────────────────

function openAIMaxTokens(body: Json): number | null {
  if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)) return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)) return body.max_completion_tokens;
  return null;
}

// ── OpenAI messages → Anthropic messages ─────────────────────────────────────

function openaiToAnthropicMessages(body: Json): Json[] {
  const out: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    const isAsst = raw.role === 'assistant';
    if (raw.role === 'tool') {
      out.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: blocksText(raw.content) }],
      });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      out.push({ role: isAsst ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const b of content) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') blocks.push({ type: 'text', text: b.text });
      else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
        const src = imageToAnthropicSource(b.image_url.url);
        if (src) blocks.push({ type: 'image', source: src });
      }
    }
    if (blocks.length) out.push({ role: isAsst ? 'assistant' : 'user', content: blocks });
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    if (calls.length) {
      const last = out[out.length - 1];
      if (last && last.role === 'assistant') {
        const existing = Array.isArray(last.content) ? last.content : [{ type: 'text', text: String(last.content) }];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let inp: Json = {};
          if (typeof call.function.arguments === 'string') {
            try { const p: unknown = JSON.parse(call.function.arguments); if (isRecord(p)) inp = p; } catch { /* */ }
          }
          existing.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: inp });
        }
        last.content = existing;
      }
    }
  }
  return out;
}

function openaiToAnthropicTools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    }];
  });
}

// ── OpenAI → Gemini contents ─────────────────────────────────────────────────

function openaiToGeminiContents(body: Json): Json[] {
  const out: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const b of content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
        else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
          const gp = imageToGeminiPart(b.image_url.url);
          if (gp) parts.push(gp);
        }
      }
    }
    if (parts.length) out.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  return out;
}

// ── Anthropic → Gemini contents ──────────────────────────────────────────────

function anthropicToGeminiContents(body: Json): Json[] {
  const out: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const b of content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
        else if (b.type === 'image' && isRecord(b.source)) {
          const s = b.source;
          if (s.type === 'base64' && typeof s.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data } });
          } else if (s.type === 'url' && typeof s.url === 'string') {
            const gp = imageToGeminiPart(s.url);
            if (gp) parts.push(gp);
          }
        } else if (b.type === 'tool_result') {
          parts.push({ text: blocksText(b.content) });
        }
      }
    }
    if (parts.length) out.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  return out;
}

// ── Build provider requests ──────────────────────────────────────────────────

/** Build an Anthropic-wire request from an OpenAI client body. */
function anthropicFromOpenAI(body: Json, model: string): Json {
  const sysParts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const t = blocksText(raw.content);
    if (t) sysParts.push(t);
  }
  const out: Json = { model, messages: openaiToAnthropicMessages(body) };
  if (sysParts.length) out.system = sysParts.join('\n');
  const maxTok = openAIMaxTokens(body);
  if (maxTok !== null) out.max_tokens = maxTok;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = strList(body.stop);
  if (stops.length) out.stop_sequences = stops;
  const tools = openaiToAnthropicTools(body);
  if (tools.length) out.tools = tools;
  const tc = body.tool_choice;
  if (tc === 'auto') out.tool_choice = { type: 'auto' };
  else if (tc === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(tc) && tc.type === 'function' && isRecord(tc.function) && typeof tc.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: tc.function.name };
  }
  return out;
}

/** Build a Gemini-wire request from an OpenAI client body. */
function geminiFromOpenAI(body: Json, _model: string): Json {
  const sysParts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const t = blocksText(raw.content);
    if (t) sysParts.push(t);
  }
  const out: Json = { contents: openaiToGeminiContents(body) };
  if (sysParts.length) out.systemInstruction = { parts: sysParts.map((text) => ({ text })) };
  const tools = openaiToAnthropicTools(body);
  if (tools.length) {
    out.tools = [{
      functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.input_schema })),
    }];
  }
  const cfg: Json = {};
  const maxTok = openAIMaxTokens(body);
  if (maxTok !== null) cfg.maxOutputTokens = maxTok;
  if (typeof body.temperature === 'number') cfg.temperature = body.temperature;
  if (typeof body.top_p === 'number') cfg.topP = body.top_p;
  const stops = strList(body.stop);
  if (stops.length) cfg.stopSequences = stops;
  if (wantsJson('openai', body)) cfg.responseMimeType = 'application/json';
  if (Object.keys(cfg).length) out.generationConfig = cfg;
  return out;
}

/** Build a Gemini-wire request from an Anthropic client body. */
function geminiFromAnthropic(body: Json, _model: string): Json {
  const out: Json = { contents: anthropicToGeminiContents(body) };
  const sys = blocksText(body.system);
  if (sys) out.systemInstruction = { parts: [{ text: sys }] };
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      }];
    });
    if (decls.length) out.tools = [{ functionDeclarations: decls }];
  }
  const cfg: Json = {};
  if (typeof body.max_tokens === 'number') cfg.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') cfg.temperature = body.temperature;
  if (typeof body.top_p === 'number') cfg.topP = body.top_p;
  const stops = strList(body.stop_sequences);
  if (stops.length) cfg.stopSequences = stops;
  if (Object.keys(cfg).length) out.generationConfig = cfg;
  return out;
}

/** Build the provider request body for any wire combination, always non-stream. */
export function requestFor(wire: Wire, clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (wire === 'openai') {
    const out: Json = clientApi === 'openai' ? { ...body } : anthropicToOpenAI(body, model);
    out.model = model;
    out.stream = false;
    delete out.stream_options;
    return out;
  }
  if (wire === 'anthropic') {
    const out: Json = clientApi === 'anthropic' ? { ...body } : anthropicFromOpenAI(body, model);
    out.model = model;
    out.stream = false;
    delete out.stream_options;
    return out;
  }
  return clientApi === 'anthropic' ? geminiFromAnthropic(body, model) : geminiFromOpenAI(body, model);
}

// ── Provider URL ─────────────────────────────────────────────────────────────

export function providerUrl(envUrl: string, provider: ProviderEntry, model: string): string {
  const base = joinBase(envUrl, provider.base);
  if (provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (provider.wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
}

// ── Provider request headers ─────────────────────────────────────────────────

export function providerHeaders(provider: ProviderEntry): Record<string, string> {
  const h: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') h['anthropic-version'] = '2023-06-01';
  return h;
}

// ── Normalise a provider's JSON answer ───────────────────────────────────────

export function answerOf(wire: Wire, body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const ch = isRecord(choices[0]) ? choices[0] : null;
    if (!ch) return null;
    const msg = isRecord(ch.message) ? ch.message : {};
    const text = typeof msg.content === 'string' ? msg.content : blocksText(msg.content);
    const u = isRecord(body.usage) ? body.usage : {};
    const toolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length ? msg.tool_calls as Json[] : null;
    return { text, stop: typeof ch.finish_reason === 'string' ? ch.finish_reason : null, input: nz(u.prompt_tokens), output: nz(u.completion_tokens), toolCalls, anthropicContent: null };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    if (!content.length) return null;
    const u = isRecord(body.usage) ? body.usage : {};
    return {
      text: blocksText(content),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
      input: nz(u.input_tokens),
      output: nz(u.output_tokens),
      toolCalls: null,
      anthropicContent: content as Json[],
    };
  }
  // Gemini
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : null;
  if (!cand) return null;
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    stop: typeof cand.finishReason === 'string' ? cand.finishReason : null,
    input: nz(meta.promptTokenCount),
    output: nz(meta.candidatesTokenCount),
    toolCalls: null,
    anthropicContent: null,
  };
}

// ── Stop reason family mapping ───────────────────────────────────────────────

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0], ['end_turn', 0], ['STOP', 0],
  ['length', 1], ['max_tokens', 1], ['MAX_TOKENS', 1],
  ['tool_calls', 2], ['function_call', 2], ['tool_use', 2],
  ['content_filter', 3], ['refusal', 3], ['SAFETY', 3],
]);
const OAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANT_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

function clientStop(api: 'openai' | 'anthropic', stop: string | null): string {
  const fam = stop === null ? 0 : STOP_FAMILIES.get(stop) ?? 0;
  return api === 'openai' ? OAI_STOPS[fam] : ANT_STOPS[fam];
}

// ── Client response builders (JSON) ──────────────────────────────────────────

export function clientJson(api: 'openai' | 'anthropic', model: string, answer: ProviderAnswer): Json {
  if (api === 'openai') {
    return {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: clientStop('openai', answer.stop),
        logprobs: null,
      }],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    };
  }
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: clientStop('anthropic', answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

// ── Client response builders (SSE) ──────────────────────────────────────────

export function clientSse(api: 'openai' | 'anthropic', model: string, answer: ProviderAnswer): string {
  if (api === 'openai') {
    const chunk = (delta: Json, finish: string | null): string =>
      sseEvent(null, {
        id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model,
        choices: [{ index: 0, delta, finish_reason: finish }],
      });
    const frames = [chunk({ role: 'assistant', content: '' }, null)];
    if (answer.text) frames.push(chunk({ content: answer.text }, null));
    frames.push(chunk({}, clientStop('openai', answer.stop)));
    frames.push(sseEvent(null, {
      id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }));
    frames.push('data: [DONE]\n\n');
    return frames.join('');
  }
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate', type: 'message', role: 'assistant', model,
        content: [], stop_reason: null, stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  if (answer.text) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: clientStop('anthropic', answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
