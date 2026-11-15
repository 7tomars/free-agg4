// lib/wire.ts — Client wire shapes (OpenAI / Anthropic model lists, errors, request/response
// translation) and SSE serialisation.
//
// Every function here is a pure transform of its arguments — no clock, no randomness — so a
// replay builds the same bytes.

import { isRecord, type Json } from './providers.ts';

// ── Served model type ────────────────────────────────────────────────────────

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

// ── Fixed timestamps (determinism) ───────────────────────────────────────────

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

// ── Model lists ──────────────────────────────────────────────────────────────

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

// ── Error shapes ─────────────────────────────────────────────────────────────

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ── Text helpers ─────────────────────────────────────────────────────────────

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const t = textOf(content);
    return t || JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function parseArgs(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim()) {
    try { const p: unknown = JSON.parse(value); if (isRecord(p)) return p; } catch { /* empty */ }
  }
  return {};
}

// ── Stop reason mapping ──────────────────────────────────────────────────────

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

// ── Image block helpers ──────────────────────────────────────────────────────

function partOfBlock(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const s = block.source;
    if (s.type === 'base64' && typeof s.data === 'string') {
      const mt = typeof s.media_type === 'string' ? s.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${mt};base64,${s.data}` } };
    }
    if (s.type === 'url' && typeof s.url === 'string') return { type: 'image_url', image_url: { url: s.url } };
  }
  return null;
}

// ── Anthropic → OpenAI request translation ───────────────────────────────────

export function anthropicToOpenAI(body: Json, model: string): Json {
  const msgs: Json[] = [];
  const sys = textOf(body.system);
  if (sys) msgs.push({ role: 'system', content: sys });

  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const isAsst = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      msgs.push({ role: isAsst ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (isAsst) {
      const calls: Json[] = [];
      for (const b of content) {
        if (!isRecord(b) || b.type !== 'tool_use') continue;
        calls.push({
          id: typeof b.id === 'string' ? b.id : '',
          type: 'function',
          function: { name: typeof b.name === 'string' ? b.name : '', arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) },
        });
      }
      const t = textOf(content);
      msgs.push({ role: 'assistant', content: t || null, ...(calls.length ? { tool_calls: calls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const b of content) {
      if (!isRecord(b)) continue;
      if (b.type === 'tool_result') {
        msgs.push({ role: 'tool', tool_call_id: typeof b.tool_use_id === 'string' ? b.tool_use_id : '', content: toolResultText(b.content) });
        continue;
      }
      const p = partOfBlock(b);
      if (p) parts.push(p);
    }
    if (parts.length) msgs.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages: msgs };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      }];
    });
    if (tools.length) out.tools = tools;
  }
  const tc = body.tool_choice;
  if (isRecord(tc)) {
    if (tc.type === 'auto') out.tool_choice = 'auto';
    else if (tc.type === 'any') out.tool_choice = 'required';
    else if (tc.type === 'tool' && typeof tc.name === 'string') out.tool_choice = { type: 'function', function: { name: tc.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function nz(v: unknown): number { return typeof v === 'number' && Number.isFinite(v) ? v : 0; }

// ── OpenAI → Anthropic response translation ─────────────────────────────────

export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const msg = isRecord(choice.message) ? choice.message : {};
  const blocks: Json[] = [];
  const t = typeof msg.content === 'string' ? msg.content : textOf(msg.content);
  if (t) blocks.push({ type: 'text', text: t });
  for (const call of Array.isArray(msg.tool_calls) ? msg.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: parseArgs(call.function.arguments),
    });
  }
  if (!blocks.length) blocks.push({ type: 'text', text: '' });
  const u = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: blocks,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: nz(u.prompt_tokens), output_tokens: nz(u.completion_tokens) },
  };
}

// ── SSE event writer ─────────────────────────────────────────────────────────

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ── Streaming: OpenAI SSE → Anthropic event sequence ─────────────────────────

export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIdx = -1;
  let nextIdx = 0;
  const toolIdxMap = new Map<number, number>();
  let inTok = 0;
  let outTok = 0;

  const emitStart = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream', type: 'message', role: 'assistant', model,
        content: [], stop_reason: null, stop_sequence: null,
        usage: { input_tokens: inTok, output_tokens: 0 },
      },
    })];
  };

  const emitClose = (): string[] => {
    const out: string[] = [];
    if (textIdx !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIdx }));
      textIdx = -1;
    }
    for (const idx of [...toolIdxMap.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: idx }));
    }
    toolIdxMap.clear();
    return out;
  };

  const emitFinish = (reason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...emitClose(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: reason, stop_sequence: null }, usage: { output_tokens: outTok } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const processFrame = (frame: string): string[] => {
    const events: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        events.push(...emitStart(), ...emitFinish(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try { chunk = JSON.parse(payload); } catch { continue; }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inTok = nz(chunk.usage.prompt_tokens);
        outTok = nz(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const ch = isRecord(choices[0]) ? choices[0] : null;
      if (!ch) continue;
      const delta = isRecord(ch.delta) ? ch.delta : {};
      events.push(...emitStart());
      if (typeof delta.content === 'string' && delta.content) {
        if (textIdx === -1) {
          textIdx = nextIdx++;
          events.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIdx, content_block: { type: 'text', text: '' } }));
        }
        events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIdx, delta: { type: 'text_delta', text: delta.content } }));
      }
      for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
        if (!isRecord(call)) continue;
        const upIdx = typeof call.index === 'number' ? call.index : toolIdxMap.size;
        let idx = toolIdxMap.get(upIdx);
        if (idx === undefined) {
          if (textIdx !== -1) {
            events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIdx }));
            textIdx = -1;
          }
          idx = nextIdx++;
          toolIdxMap.set(upIdx, idx);
          const fn = isRecord(call.function) ? call.function : {};
          events.push(sseEvent('content_block_start', {
            type: 'content_block_start', index: idx,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id ? call.id : `toolu_${idx}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }));
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments) {
          events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: idx, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (ch.finish_reason !== null && ch.finish_reason !== undefined) pendingStop = stopReasonOf(ch.finish_reason);
    }
    return events;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buf.indexOf('\n\n');
      while (at !== -1) {
        const frame = buf.slice(0, at);
        buf = buf.slice(at + 2);
        for (const ev of processFrame(frame)) yield ev;
        at = buf.indexOf('\n\n');
      }
    }
    buf += decoder.decode();
    if (buf.trim()) for (const ev of processFrame(buf)) yield ev;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const ev of emitStart()) yield ev;
    for (const ev of emitFinish(pendingStop ?? 'end_turn')) yield ev;
  }
}
