#!/usr/bin/env node
/**
 * proxy.js — Anamnesis main server.
 *
 * Works with any OpenAI-compatible backend (llama-server, Ollama /v1,
 * LM Studio, koboldcpp, OpenAI itself, …). Set upstream.baseUrl +
 * upstream.apiKey in config.json.
 *
 * Per-request pipeline (POST /…/chat/completions):
 *   1. Persist user turn synchronously  (survives any subsequent crash)
 *   2. Scene-guided context selection   (drops in <memory>/<foresight> blocks)
 *   3. Forward to upstream
 *      - streaming    : pipe each SSE chunk through to the client and tee
 *                       a copy into an accumulator that reconstructs the
 *                       final assistant content from delta frames.
 *      - non-streaming: buffer upstream, return, then parse content.
 *   4. Persist the assistant turn + kick off background extraction
 *      (engram + foresight, both non-blocking).
 *
 * Graceful shutdown (SIGTERM/SIGINT):
 *   - Stop the consolidator timer
 *   - Wait up to 15s for in-flight extraction
 *   - Close the SQLite handle
 */

const http = require('http');
const https = require('https');
const path = require('path');
const fs = require('fs');

const {
  expandHome,
  extractContentText,
  findNewUserMessage,
  getSessionKey,
  getMemoryCategory,
  buildUpstreamHeaders,
  makeSseAccumulator,
  stripThinkingTokens,
  makeStreamingThinkingFilter,
  resolveTaskActivitySnapshot,
} = require('./lib/proxy-helpers.js');
const log = require('./lib/logger.js').make('anamnesis');

const brain = require('./lib/brain.js');
const HistoryStore = require('./history.js');
const embedder = require('./embedder.js'); // singleton
const Selector = require('./selector.js');
const Extractor = require('./extractor.js');
const ForesightExtractor = require('./foresight.js');
const PersonaManager = require('./persona.js');
const Consolidator = require('./consolidator.js');

function loadConfig() {
  return expandHome(JSON.parse(fs.readFileSync(path.join(__dirname, '../config.json'), 'utf8')));
}

async function start(config = loadConfig()) {
  brain.init(config); // start model download/load in background; chat() calls queue until ready

  const history = new HistoryStore(config.history.dbPath);
  // Task-level snapshot of "time since last activity" (fix 2026-07-23): this used
  // to be recomputed fresh via history.getLastTurnTimestamp() on EVERY request,
  // including every tool-call round-trip resend within one agentic task. Since
  // recordAssistantTurn() persists each assistant reply asynchronously right after
  // it streams, by the very next tool round the "last turn" was the reply from a
  // few seconds ago, not genuine downtime -- so _buildDowntimeNote's <continuity>
  // block (selector.js) would appear on round 1 and vanish (or change) by round 2,
  // shifting the system prompt and defeating llama-server's context-checkpoint
  // reuse for recurrent/hybrid models exactly once per task (measured live on
  // Mark/Ornith: round 1->2 common-prefix collapsed to <40% and forced a full
  // ~20s reprocess; every round after that, once the prompt stabilized, restored
  // cleanly). Anchoring this snapshot to the start of each genuinely NEW user turn
  // (keyed by sessionKey, refreshed only when the incoming message is new -- same
  // dedup signal already used below to avoid double-storing the user turn) keeps
  // the downtime note's content -- and therefore the prompt prefix -- stable for
  // every round of one task, changing only when the user actually sends a new one.
  const _taskActivitySnapshot = new Map(); // sessionKey -> lastActivityAt (number|null)
  const persona = new PersonaManager(config, history);
  await persona.init();
  const selector = new Selector(config, history, embedder, persona);
  const extractor = new Extractor(config, history, embedder);
  const foresightExtractor = new ForesightExtractor(config, history, embedder);
  const consolidator = new Consolidator(config, history, embedder);

  const pruned = history.prune(config.history.maxAgeDays);
  if (pruned > 0) log.info(`pruned ${pruned} old turns`);

  // Pick up turns that crashed mid-extraction in a previous session.
  extractor.processBacklog().catch((e) => log.warn('backlog (extractor):', e.message));
  foresightExtractor.processBacklog().catch((e) => log.warn('backlog (foresight):', e.message));

  consolidator.start(config.memory.consolidationIntervalMs);

  // ─── Upstream context-window probe ────────────────────────────────────────
  // If the upstream serves /props (llama-server and Pleiades' elastic engine
  // both do), clamp the selector's token budget to the real window. Without
  // this, a 50k-token context sent at an 8k-window server gets front-
  // truncated upstream — the system prompt and memory block die first, which
  // presents as sudden amnesia + rampant hallucination mid-conversation.
  let _upWin = { value: null, at: 0 };
  async function upstreamWindow() {
    const now = Date.now();
    if (now - _upWin.at < 60000) return _upWin.value;
    _upWin.at = now;
    try {
      const base = config.upstream.baseUrl.replace(/\/v1\/?$/, '');
      const r = await fetch(base + '/props', { signal: AbortSignal.timeout(1500) });
      if (r.ok) {
        const j = await r.json();
        const n = j.n_ctx ?? j.default_generation_settings?.n_ctx ?? null;
        _upWin.value = typeof n === 'number' && n > 0 ? n : null;
      }
    } catch {
      _upWin.value = null; // upstream has no /props — no clamp
    }
    return _upWin.value;
  }

  // ─── Upstream wiring ──────────────────────────────────────────────────────

  function upstreamUrl(reqPath) {
    const upUrl = new URL(config.upstream.baseUrl);
    return {
      upUrl,
      lib: upUrl.protocol === 'https:' ? https : http,
      port: upUrl.port || (upUrl.protocol === 'https:' ? 443 : 80),
      path: (() => {
        const base = upUrl.pathname.replace(/\/$/, '');
        // If the baseUrl already ends in /v1, strip /v1 from reqPath to avoid doubling
        const req = base.endsWith('/v1') ? reqPath.replace(/^\/v1/, '') : reqPath;
        return base + req;
      })(),
    };
  }

  /**
   * Pipe upstream response straight to the client, AND tee each chunk into
   * `onChunk` so callers can rebuild assistant content for storage.
   */
  function streamThrough(reqPath, method, headers, body, clientRes, onChunk, watchRes) {
    return new Promise((resolve, reject) => {
      const { upUrl, lib, port, path: outPath } = upstreamUrl(reqPath);
      const upReq = lib.request(
        { hostname: upUrl.hostname, port, path: outPath, method, headers },
        (upRes) => {
          const outHeaders = { ...upRes.headers };
          // Don't repeat hop-by-hop headers to the client.
          delete outHeaders['transfer-encoding'];
          delete outHeaders['connection'];
          clientRes.writeHead(upRes.statusCode, outHeaders);

          upRes.on('data', (chunk) => {
            clientRes.write(chunk);
            try {
              onChunk(chunk);
            } catch (e) {
              log.warn('onChunk error:', e.message);
            }
          });
          upRes.on('end', () => {
            detachAbort();
            clientRes.end();
            resolve();
          });
          upRes.on('error', (err) => {
            detachAbort();
            clientRes.end();
            reject(err);
          });
        }
      );
      // Stop button / dropped client: the webui closes its connection to us
      // (engine.py calls stream.close() on the SDK stream), but without this,
      // that close was never observed here — the upstream llama-server request
      // kept decoding to completion regardless, wasting the model's single
      // (--parallel 1) concurrency slot and blocking every other request.
      // Destroying upReq on the real incoming request's 'close' propagates the
      // cancellation all the way to llama-server, which aborts the in-flight
      // generation when its socket drops.
      function onClientAbort() {
        log.warn('client disconnected mid-stream — aborting upstream generation');
        upReq.destroy(new Error('client disconnected'));
      }
      function detachAbort() {
        if (watchRes) watchRes.removeListener('close', onClientAbort);
      }
      // Must watch the RAW response object, not a thinking-filter wrapper —
      // Object.create(res) wrappers don't share res's EventEmitter state, so
      // 'close' (the documented signal for a connection torn down before
      // response.end()) would never reach a listener added to the wrapper.
      if (watchRes) watchRes.once('close', onClientAbort);
      upReq.on('error', (err) => {
        detachAbort();
        if (!clientRes.headersSent) {
          clientRes.writeHead(502, { 'Content-Type': 'application/json' });
          clientRes.end(JSON.stringify({ error: err.message }));
        } else {
          clientRes.end();
        }
        reject(err);
      });
      upReq.setTimeout(1800000, () => upReq.destroy(new Error('upstream timeout')));
      upReq.write(body);
      upReq.end();
    });
  }

  function bufferedForward(reqPath, method, headers, body) {
    return new Promise((resolve, reject) => {
      const { upUrl, lib, port, path: outPath } = upstreamUrl(reqPath);
      const upReq = lib.request(
        { hostname: upUrl.hostname, port, path: outPath, method, headers },
        (upRes) => {
          const chunks = [];
          upRes.on('data', (d) => chunks.push(d));
          upRes.on('end', () =>
            resolve({
              status: upRes.statusCode,
              headers: upRes.headers,
              body: Buffer.concat(chunks),
            })
          );
          upRes.on('error', reject);
        }
      );
      upReq.on('error', reject);
      upReq.setTimeout(1800000, () => upReq.destroy(new Error('upstream timeout')));
      upReq.write(body);
      upReq.end();
    });
  }


  function recordAssistantTurn(sessionKey, content, category = 'fleagle') {
    if (!content) return;
    // Defer to the next tick so the client connection is fully closed first;
    // embedding + extraction never block the response.
    setImmediate(async () => {
      try {
        const vec = await embedder.embed(content.slice(0, 2000)).catch(() => null);
        const est = Math.ceil(content.length / config.context.charsPerToken);
        // Tag the reply with the SAME category as the request that produced
        // it (e.g. a reply generated during a 'background' autonomous task
        // stays 'background' too) — see history.js turns.category.
        history.insertTurn(sessionKey, 'assistant', content, vec, est, embedder.model, category);
        extractor.processBatch().catch((e) => log.warn('extractor:', e.message));
        foresightExtractor.processBatch().catch((e) => log.warn('foresight:', e.message));
        persona.observeResponse(sessionKey, null, content);
      } catch (e) {
        log.warn('recordAssistantTurn:', e.message);
      }
    });
  }

  // ─── Server ───────────────────────────────────────────────────────────────

  const server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/anamnesis/status') {
      const sessions = history.listSessions();
      const busiest = sessions[0]?.session_key ?? 'default';
      const stats = history.stats(busiest);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(
        JSON.stringify({
          status: 'ok',
          session: busiest,
          session_count: sessions.length,
          ...stats,
          upstream: config.upstream.baseUrl,
          embedding_model: brain.embeddingModel(),
        })
      );
    }

    const reqChunks = [];
    req.on('data', (d) => reqChunks.push(d));
    req.on('end', async () => {
      const rawBody = Buffer.concat(reqChunks);
      try {
        await handleRequest(req, res, rawBody);
      } catch (err) {
        log.error('unhandled request error (prevented crash):', err.message, err.stack?.split('\n')[1] ?? '');
        if (!res.headersSent) {
          try { res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'proxy internal error' })); } catch {}
        }
      }
    });
  });

  async function handleRequest(req, res, rawBody) {

      if (req.method === 'POST' && req.url.endsWith('/chat/completions')) {
        let parsed;
        try {
          parsed = JSON.parse(rawBody.toString());
        } catch {
          return passthrough(req, res, rawBody);
        }
        if (!Array.isArray(parsed.messages)) return passthrough(req, res, rawBody);

        const sessionKey = getSessionKey(req.headers, config.upstream.apiKey);
        const category = getMemoryCategory(req.headers);
        const streaming = parsed.stream === true;

        // 1. Persist user turn synchronously (only when genuinely new — see below).
        // Content may be a string OR an array of OpenAI content-parts
        // (text + tool_result + image_url etc). Flatten before storage —
        // better-sqlite3 only binds primitives, and the selector needs a
        // string to embed.
        const userMsg = findNewUserMessage(parsed.messages);
        const userText = extractContentText(userMsg?.content);
        // Agentic clients re-send the same growing messages array once per
        // tool round-trip, so the same user turn used to be stored N+1 times
        // for a turn with N tool calls — duplicate embeddings, skewed
        // retrieval, wasted extraction. Store only when it's actually new.
        const isNewUserTurn = !!(userText && history.lastUserTurnContent(sessionKey) !== userText);

        // Snapshot the session's last-activity timestamp BEFORE inserting the
        // incoming user turn below, so it reflects genuine elapsed downtime
        // rather than the turn that just arrived (see selector.js
        // _buildDowntimeNote / history.js getLastTurnTimestamp). Only
        // recomputed on a genuinely NEW user turn — every tool-call
        // round-trip resend within the same task reuses the snapshot taken
        // when the task started, so the <continuity> block this feeds stays
        // byte-stable across a whole task instead of flipping mid-task (see
        // the _taskActivitySnapshot comment near HistoryStore's construction
        // for why that matters — measured live to break context-checkpoint
        // reuse on Mark/Ornith otherwise).
        const lastActivityAt = resolveTaskActivitySnapshot(
          isNewUserTurn,
          sessionKey,
          _taskActivitySnapshot,
          () => history.getLastTurnTimestamp(sessionKey)
        );

        if (isNewUserTurn) {
          const vec = await embedder.embed(userText).catch(() => null);
          const est = Math.ceil(userText.length / config.context.charsPerToken);
          history.insertTurn(sessionKey, 'user', userText, vec, est, embedder.model, category);
        }

        // 2. Scene-guided context selection.
        let selectedMessages = parsed.messages;
        try {
          const upCtx = await upstreamWindow();
          const budgetCeiling = upCtx
            ? Math.max(2048, Math.floor(upCtx * 0.85) - (parsed.max_tokens ?? 600))
            : undefined;
          selectedMessages = await selector.select(sessionKey, parsed.messages, {
            budgetCeiling,
            lastActivityAt,
          });
        } catch (err) {
          log.error('selector error, falling back to original messages:', err.message);
        }

        // 3. Rewrite + forward.
        const rewritten = { ...parsed, messages: selectedMessages };
        // Override model name if upstream.model is set in config (e.g. Ollama needs exact model name).
        if (config.upstream.model) rewritten.model = config.upstream.model;
        if (config.upstream.disableThinking) {
          // Thinking management strategy:
          // - Do NOT strip tools/tool_choice — Mark needs full tool access to act autonomously.
          // - Actually tell the chat template to skip thinking: `chat_template_kwargs.
          //   enable_thinking: false` is honored by llama-server's --jinja rendering on
          //   Qwen3-family templates (verified live: with it, the model answers directly,
          //   ~0.5s, finish_reason "stop"; without it, reasoning-budget defaults to
          //   unrestricted server-side and the model can burn the entire completion
          //   window on invisible `reasoning_content` with finish_reason "length" and
          //   EMPTY `content` — i.e. disableThinking previously didn't disable thinking
          //   at all, just hoped the model would stop on its own before a padded
          //   max_tokens ran out, which is false for at least one real model).
          // - Still pad max_tokens a bit as a fallback for templates that don't honor
          //   enable_thinking — better a little unused headroom than a truncated answer.
          rewritten.chat_template_kwargs = {
            ...(rewritten.chat_template_kwargs || {}),
            enable_thinking: false,
          };
          const THINKING_OVERHEAD = 500; // safety margin only; true suppression is above
          const clientBudget = rewritten.max_tokens ?? 600;
          rewritten.max_tokens = clientBudget + THINKING_OVERHEAD;
        }
        log.debug('MSGS OUT: ' + JSON.stringify(selectedMessages.map(m => m.role + ':' + (m.content ?? '').slice(0,80))));
        const rewrittenBody = Buffer.from(JSON.stringify(rewritten));
        const headers = buildUpstreamHeaders(req.headers, {
          upstreamApiKey: config.upstream.apiKey,
        });
        headers['Content-Length'] = Buffer.byteLength(rewrittenBody);

        if (streaming) {
          const sse = makeSseAccumulator();
          // When thinking suppression is active, filter chunks before reaching the client.
          const filteredRes = config.upstream.disableThinking
            ? Object.assign(Object.create(res), { write: makeStreamingThinkingFilter(res) })
            : res;
          try {
            await streamThrough(req.url, req.method, headers, rewrittenBody, filteredRes, (c) =>
              sse.feed(c)
            , res);
          } catch (err) {
            log.error('streaming upstream error:', err.message);
            return;
          }
          const strippedContent = stripThinkingTokens(sse.content);
          // Only store if there's real content — guards against storing null/'[]'/
          // tool-call residue or partial chunks from dropped connections.
          if (strippedContent && strippedContent.trim().length >= 4) {
            recordAssistantTurn(sessionKey, strippedContent, category);
          }
          return;
        }

        let upRes;
        try {
          upRes = await bufferedForward(req.url, req.method, headers, rewrittenBody);
        } catch (err) {
          log.error('upstream error:', err.message);
          res.writeHead(502, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify({ error: err.message }));
        }
        // Strip thinking tokens from the response before sending to client AND before storing.
        let clientBody = upRes.body;
        let storedContent = '';
        try {
          const upParsed = JSON.parse(upRes.body.toString());
          const rawContent = upParsed.choices?.[0]?.message?.content ?? '';
          storedContent = stripThinkingTokens(rawContent);
          if (rawContent !== storedContent) {
            // Rewrite body with stripped content so client never sees thinking tokens
            upParsed.choices[0].message.content = storedContent;
            clientBody = Buffer.from(JSON.stringify(upParsed));
          }
        } catch { /* non-JSON — pass through as-is */ }
        // Strip transfer-encoding when we're rewriting the body; set fresh Content-Length.
        const outHeaders = { ...upRes.headers };
        delete outHeaders['transfer-encoding'];
        delete outHeaders['content-encoding']; // body is already decoded by bufferedForward
        outHeaders['content-length'] = Buffer.byteLength(clientBody);
        res.writeHead(upRes.status, outHeaders);
        res.end(clientBody);

        if (storedContent) recordAssistantTurn(sessionKey, storedContent, category);
        return;
      }

      passthrough(req, res, rawBody);
  }  // end handleRequest

  async function passthrough(req, res, body) {
    const headers = buildUpstreamHeaders(req.headers, { upstreamApiKey: config.upstream.apiKey });
    headers['Content-Length'] = Buffer.byteLength(body);
    try {
      const upRes = await bufferedForward(req.url, req.method, headers, body);
      res.writeHead(upRes.status, upRes.headers);
      res.end(upRes.body);
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
  }

  await new Promise((resolve, reject) => {
    server.once('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        log.error(`port ${config.proxy.port} already in use — stop the other instance first`);
      }
      reject(err);
    });
    server.listen(config.proxy.port, config.proxy.host, () => {
      log.info(`listening on ${config.proxy.host}:${config.proxy.port}`);
      log.info(`upstream: ${config.upstream.baseUrl}`);
      log.info(
        `token budget: ${config.context.tokenBudget} | recency: ${config.context.recencyTurns} turns | slots: ${config.context.rotatingSlots}`
      );
      resolve();
    });
  });

  let shuttingDown = false;
  async function shutdown(signal) {
    if (shuttingDown) return;
    shuttingDown = true;
    log.info(`received ${signal}, shutting down gracefully...`);
    consolidator.stop();
    server.close();
    await Promise.all([extractor.flushInFlight(), foresightExtractor.flushInFlight()]);
    try {
      history.close();
    } catch {
      /* already closed */
    }
    log.info('shutdown complete');
    // Do NOT call process.exit() here. When running inside the daemon (in-process),
    // exiting would kill the entire daemon process. Callers that want process exit
    // (standalone mode) do it themselves via the signal handlers below.
  }

  if (require.main === module) {
    process.on('SIGTERM', () => shutdown('SIGTERM').then(() => process.exit(0)));
    process.on('SIGINT', () => shutdown('SIGINT').then(() => process.exit(0)));
  }

  return { server, history, shutdown };
}

if (require.main === module) start();

module.exports = { start, loadConfig };
