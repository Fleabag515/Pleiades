/**
 * lib/proxy-helpers.js — pure helpers extracted from proxy.js so they are
 * testable without booting the HTTP server or touching the DB.
 *
 * Nothing here imports config or any stateful module; everything takes its
 * dependencies as arguments.
 */

const crypto = require('crypto');
const os = require('os');

/**
 * Safely flatten OpenAI-style chat-completion `content` into a plain string.
 *
 * OpenAI-compatible clients (including OpenClaw) may send a message's
 * `content` as either a string OR an array of content parts:
 *   [{ type: 'text', text: '...' }, { type: 'tool_result', ... }, ...]
 *
 * SQLite (via better-sqlite3) only binds strings/buffers/numbers/null, so
 * we used to crash with a TypeError on every multipart request. Falls back
 * to JSON.stringify() for non-text parts so nothing is silently dropped.
 *
 * Imported in the proxy (for storing the user turn) and the selector (so
 * the query embedding sees the same text the user actually wrote).
 */
function extractContentText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = content
      .filter((p) => p?.type === 'text' || p?.text)
      .map((p) => p?.text ?? p?.content ?? '')
      .join('\n')
      .trim();
    return text || JSON.stringify(content);
  }
  if (content && typeof content === 'object') return JSON.stringify(content);
  return String(content ?? '');
}

/**
 * Find "the new user turn" in an incoming chat-completion request's
 * `messages` array, the same way the proxy's persistence/dedup logic does:
 * reverse-scan for the first role==='user' entry.
 *
 * Why reverse-scan instead of just the last array entry: an agentic
 * tool-calling client (Pleiades) resends the whole growing array once per
 * tool round-trip within one turn, so the literal last entry during those
 * rounds is an assistant tool_calls message or a role:"tool" result, not
 * the user's actual message. This deliberately extends to SYNTHETIC tool
 * rounds too — e.g. Pleiades' engine.py periodically fabricates an
 * assistant tool_calls message + a role:"tool" result to deliver an
 * internal notice (a self-reflection check, a background-task-completion
 * notice) mid-turn, specifically so it does NOT read as user speech (see
 * fix/anamnesis-reflection-injection on the Pleiades side). Filtering on
 * role==='user' here is what makes that safe: neither a real nor a
 * synthetic tool round is ever mistaken for "the new user turn", no matter
 * how recently it was appended.
 *
 * Returns the message object itself (not just its text — callers that need
 * the text still run it through extractContentText), or undefined if
 * `messages` has no role==='user' entry at all.
 */
function findNewUserMessage(messages) {
  return [...messages].reverse().find((m) => m.role === 'user');
}

/**
 * Recursively expand `~`, `${HOME}` and `$HOME` in string values inside a
 * (possibly nested) config object. Lets config.json ship a portable default
 * like `~/.anamnesis/history.db` instead of a machine-specific absolute path.
 */
function expandHome(obj, home = os.homedir()) {
  if (obj == null) return obj;
  if (typeof obj === 'string') {
    return obj
      .replace(/^~(?=\/|$)/, home)
      .replace(/\$\{HOME\}/g, home)
      .replace(/\$HOME(?=\/|$)/g, home);
  }
  if (Array.isArray(obj)) return obj.map((v) => expandHome(v, home));
  if (typeof obj === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(obj)) out[k] = expandHome(v, home);
    return out;
  }
  return obj;
}

/**
 * Derive a stable session key from request headers.
 *
 * Order of precedence:
 *   1. X-OpenClaw-Session / X-Session-Id explicit headers
 *   2. SHA-256 prefix of the bearer token (avoids leaking credential bytes)
 *   3. "default" for unauthenticated local clients
 */
/**
 * Which memory-quota category this request's turns should be tagged with
 * (see history.js turns.category / selector.js _fillBudget's category-
 * partitioned quotas). Explicit opt-in only: a caller must send
 * `X-Memory-Category`. Everything else defaults to 'fleagle' (owner-
 * authored), which is the safe no-op default — category quotas have zero
 * effect on anyone who never sends this header. Pleiades does not send it
 * yet as of this writing; wiring it up (e.g. tagging cron/autonomous-task
 * requests as 'background') is a caller-side follow-up, not part of this
 * change.
 */
function getMemoryCategory(headers, fallback = 'fleagle') {
  const raw = headers['x-memory-category'];
  if (typeof raw !== 'string') return fallback;
  const cat = raw.trim().toLowerCase();
  return cat || fallback;
}

/**
 * Render a gap in seconds as a short human duration ("14h 22m", "3d 2h",
 * "40m"). Used to build the downtime-awareness continuity note — see
 * selector.js _buildDowntimeNote(). Always rounds down to whole units;
 * a gap under a minute renders as "0m" rather than disappearing, so a
 * caller can still tell a note was suppressed by the minGapMinutes floor
 * versus genuinely being ~0.
 */
function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const days = Math.floor(s / 86400);
  const hours = Math.floor((s % 86400) / 3600);
  const mins = Math.floor((s % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

function getSessionKey(headers, upstreamApiKey = '') {
  const explicit = headers['x-openclaw-session'] ?? headers['x-session-id'];
  if (explicit) return `oc:${explicit}`;

  const auth = headers['authorization'] ?? '';
  const token = auth.replace(/^Bearer\s+/i, '').trim();
  if (token && token !== upstreamApiKey) {
    const hash = crypto.createHash('sha256').update(token).digest('hex').slice(0, 16);
    return `auth:${hash}`;
  }
  return 'default';
}

/**
 * Build outgoing headers for an upstream HTTP request.
 *
 *   - inject our configured upstream API key when set
 *   - otherwise pass the client's Authorization through untouched
 *   - strip hop-by-hop and proxy-internal headers
 *   - drop Content-Length so the caller can recompute against the (possibly
 *     rewritten) body
 */
function buildUpstreamHeaders(incomingHeaders, { upstreamApiKey } = {}) {
  const headers = { ...incomingHeaders };
  delete headers['x-openclaw-session'];
  delete headers['x-session-id'];
  delete headers['host'];
  for (const k of Object.keys(headers)) {
    const lk = k.toLowerCase();
    if (lk === 'content-length' || lk === 'connection' || lk === 'transfer-encoding') {
      delete headers[k];
    }
  }
  if (upstreamApiKey) {
    for (const k of Object.keys(headers)) {
      if (k.toLowerCase() === 'authorization') delete headers[k];
    }
    headers['Authorization'] = `Bearer ${upstreamApiKey}`;
  }
  return headers;
}

/**
 * Stateful accumulator for OpenAI-style SSE chat-completion streams.
 * Pass each chunk through `.feed(chunk)`; read the reconstructed assistant
 * content from `.content` after the stream ends.
 *
 * Handles:
 *   - frames split across chunk boundaries (buffered until next newline pair)
 *   - keepalive lines that aren't valid JSON
 *   - both delta.content (streaming) and message.content (final) shapes
 *   - `[DONE]` sentinel
 */
function makeSseAccumulator() {
  let buf = '';
  let content = '';
  return {
    feed(chunk) {
      buf += chunk.toString('utf8');
      const parts = buf.split(/\n\n/);
      buf = parts.pop(); // possibly incomplete tail
      for (const frame of parts) {
        for (const line of frame.split(/\n/)) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          try {
            const j = JSON.parse(payload);
            const delta =
              j?.choices?.[0]?.delta?.content ?? j?.choices?.[0]?.message?.content ?? '';
            if (delta) content += delta;
          } catch {
            /* ignore keepalives / partials */
          }
        }
      }
    },
    get content() {
      return content;
    },
  };
}


/**
 * stripThinkingTokens — remove internal-reasoning blocks from an assistant
 * response before it is stored in memory.
 *
 * Without this, thinking tokens (emitted inline in `content` by some models)
 * get persisted to the turn history and are re-injected on future turns,
 * which causes the model to keep generating more thinking tokens — a
 * self-reinforcing loop that fills every response with noise.
 *
 * Patterns handled:
 *   Gemma 4        <|channel>thought\n … <channel|>
 *   Qwen3 / QwQ /
 *   DeepSeek-R1    <think> … </think>
 *
 * The function is intentionally model-agnostic: it strips all known patterns
 * regardless of whether disableThinking is set, so a mis-configured upstream
 * or a model that ignores enable_thinking:false can never corrupt the memory
 * store.
 */
function stripThinkingTokens(text) {
  if (!text) return text;

  // Gemma 4 text form: <|channel>thought … <channel|>
  text = text.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '');
  text = text.replace(/<\|channel>thought[\s\S]*/g, '');

  // Gemma 4 PUA form — llama.cpp streaming decodes <|channel> as U+F06C (\uf06c).
  // The closing tag appears in multiple observed variants depending on context:
  //   !</thought   (no trailing >)
  //   !</thought>  (with trailing >)
  //   </thought>   (without leading !)
  // Handle all of them in one alternation, then strip any orphaned closers.
  text = text.replace(/\uf06cthought[\s\S]*?(?:!<\/thought>?|<\/thought>?)/g, '');
  text = text.replace(/\uf06cthought[\s\S]*/g, '');
  // Strip orphaned closers/openers — e.g. when opener was in a prior chunk or
  // a partial regex match left a tail behind.
  // Text form: <channel|> without a preceding <|channel>thought
  text = text.replace(/<channel\|>/g, '');
  // Text form opener without closer (shouldn't survive the open-ended regex above,
  // but belt-and-suspenders):
  text = text.replace(/<\|channel>/g, '');
  // PUA form closers
  text = text.replace(/!<\/thought>?/g, '');
  text = text.replace(/<\/thought>/g, '');
  // PUA opener residue \uf06c without "thought" following (chunk boundary split)
  text = text.replace(/\uf06c/g, '');

  // Qwen3 / DeepSeek-R1 / QwQ: <think> … </think>
  text = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  text = text.replace(/<think>[\s\S]*/g, '');

  // Gemma tool-call tokens that leak into content on backends without a
  // reasoning/tool extractor (e.g. Ollama). Strip the whole block or residue.
  // Full form: <tool_call>…</tool_call>
  text = text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
  // Partial closer residue (e.g. "}<tool_call|>" or "<tool_call|>")
  text = text.replace(/[}\s]*<tool_call\|>/g, '');
  text = text.replace(/<tool_call>/g, '');
  // Bare JSON artifact from tool call body: lone "[]", "{}", or "null" after stripping
  text = text.replace(/^[\s\[\]{}null]*$/, '');

  return text.trim();
}


/**
 * makeStreamingThinkingFilter — wraps a writable response so that SSE chunks
 * containing thinking tokens are silently dropped before reaching the client.
 *
 * Returns a replacement `write(chunk)` function. Call it in place of
 * `clientRes.write(chunk)` inside a streaming pipe.
 *
 * State machine:
 *   NORMAL     — forward chunks as-is
 *   IN_THINKING — suppress chunks; exit on closing tag
 *
 * Token pairs handled:
 *   Gemma 4:   <|channel>thought … <channel|>
 *   Qwen3:     <think> … </think>
 */
function makeStreamingThinkingFilter(clientRes) {
  let inThinking = false;
  let leftovers = ''; // partial text held back waiting to confirm it isn't an opener

  // Text forms (non-streaming / most backends) and PUA forms (llama.cpp streaming).
  // llama.cpp maps <|channel> → U+F06C and <channel|> → "!</thought" (no closing >)
  // when streaming Gemma 4 tokens as delta chunks.
  const OPENERS = ['<|channel>thought', 'thought', '<think>'];
  const CLOSERS = {
    '<|channel>thought': '<channel|>',
    'thought': '!</thought',
    '<think>': '</think>',
  };
  let activeCloser = null;

  return function filteredWrite(chunk) {
    let text = leftovers + chunk.toString('utf8');
    leftovers = '';
    let out = '';

    while (text.length > 0) {
      if (inThinking) {
        const ci = text.indexOf(activeCloser);
        if (ci === -1) {
          text = ''; // all thinking noise — discard
        } else {
          text = text.slice(ci + activeCloser.length); // skip closer, resume after
          inThinking = false;
          activeCloser = null;
        }
      } else {
        // Check if any opener starts within this text
        let earliest = -1;
        let matchedOpener = null;
        for (const op of OPENERS) {
          const idx = text.indexOf(op);
          if (idx !== -1 && (earliest === -1 || idx < earliest)) {
            earliest = idx;
            matchedOpener = op;
          }
        }
        if (earliest === -1) {
          // No opener. But check for a partial opener at the tail (hold back).
          let held = 0;
          for (const op of OPENERS) {
            for (let len = Math.min(op.length - 1, text.length); len > 0; len--) {
              if (text.endsWith(op.slice(0, len))) {
                held = Math.max(held, len);
                break;
              }
            }
          }
          out += text.slice(0, text.length - held);
          leftovers = text.slice(text.length - held);
          text = '';
        } else {
          out += text.slice(0, earliest); // content before opener is safe
          text = text.slice(earliest + matchedOpener.length); // skip opener
          inThinking = true;
          activeCloser = CLOSERS[matchedOpener];
        }
      }
    }

    if (out) clientRes.write(out);
  };
}

/**
 * Decide the "last activity" timestamp fed to selector.js's downtime/
 * continuity note for one request, given a per-session snapshot cache.
 *
 * Fix 2026-07-23: recomputing this fresh on every request (including every
 * tool-call round-trip resend within one agentic task) let it flip mid-task
 * — the assistant's own reply from round 1 gets persisted as "the last turn"
 * before round 2's request arrives, changing or erasing the continuity note
 * and shifting the system prompt enough to defeat llama-server's context-
 * checkpoint reuse for recurrent/hybrid models (measured live on Mark/
 * Ornith: round 1->2 common-prefix collapsed to <40% and forced a full ~20s
 * reprocess that a stable prefix would have avoided). Anchoring the
 * snapshot to the start of each genuinely NEW user turn keeps it — and the
 * prompt prefix it feeds — byte-stable for every round of one task.
 *
 * @param {boolean} isNewUserTurn - true only when this request's message is
 *   a genuinely new user turn (not a tool-call round-trip resend of one
 *   already stored — see findNewUserMessage's dedup usage in proxy.js).
 * @param {string} sessionKey
 * @param {Map<string, number|null>} snapshotMap - mutated in place; caller
 *   owns its lifetime (one per proxy process, keyed by session).
 * @param {() => (number|null)} readFreshTimestamp - lazily reads a fresh
 *   "last turn" timestamp (e.g. history.getLastTurnTimestamp(sessionKey)).
 *   Only invoked when a fresh read is actually needed.
 * @returns {number|null}
 */
function resolveTaskActivitySnapshot(isNewUserTurn, sessionKey, snapshotMap, readFreshTimestamp) {
  if (isNewUserTurn) {
    const fresh = readFreshTimestamp();
    snapshotMap.set(sessionKey, fresh);
    return fresh;
  }
  if (snapshotMap.has(sessionKey)) {
    return snapshotMap.get(sessionKey);
  }
  // First time we've seen this session in this proxy process (e.g. right
  // after a daemon restart mid-task) — fall back to a fresh read rather
  // than crashing; only degrades to the old per-request behavior for that
  // one edge case.
  return readFreshTimestamp();
}

module.exports = {
  makeStreamingThinkingFilter,
  expandHome,
  extractContentText,
  findNewUserMessage,
  getSessionKey,
  getMemoryCategory,
  formatDuration,
  buildUpstreamHeaders,
  makeSseAccumulator,
  stripThinkingTokens,
  resolveTaskActivitySnapshot,
};
