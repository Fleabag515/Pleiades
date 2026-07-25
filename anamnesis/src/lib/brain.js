'use strict';

/**
 * lib/brain.js — unified inference interface.
 *
 * Replaces ollama.js as the import target for all callers.
 * Routes chat/generate to inference-engine.js (node-llama-cpp).
 * Routes embed to local-embedder.js (@huggingface/transformers).
 *
 * chat() and generate() queue internally while the model downloads/loads.
 * embed() returns null when not ready (callers handle gracefully).
 */

const { LocalEmbedder } = require('./local-embedder.js');
const { InferenceEngine } = require('./inference-engine.js');
const { ProductionModelManager } = require('./model-manager.js');
const log = require('./logger.js').make('brain');

const QUEUE_MAX = 200; // hardcoded — independent of startupBacklogLimit

// ─── Singleton state ──────────────────────────────────────────────────────────
let _engine = null;
let _embedder = null;
let _ready = false;
let _initCalled = false;
const _queue = []; // { messages, opts, resolve, reject }[]

// ─── Test dependency injection ────────────────────────────────────────────────

function _inject({ engine, embedder }) {
  _engine = engine;
  _embedder = embedder;
}

function _onEngineReady() {
  _ready = true;
  _drainQueue();
}

function _onEngineError(err) {
  const pending = _queue.splice(0);
  for (const item of pending) item.reject(err);
}

function _drainQueue() {
  const pending = _queue.splice(0);
  for (const item of pending) {
    const { timeoutMs, ...engineOpts } = item.opts;
    let call = _engine.chat(item.messages, engineOpts);
    if (timeoutMs != null) {
      const timer = new Promise((_, rej) =>
        setTimeout(() => rej(new Error(`inference timed out after ${timeoutMs}ms`)), timeoutMs)
      );
      call = Promise.race([call, timer]);
    }
    call.then(item.resolve).catch(item.reject);
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Call once at daemon startup (or from importer CLI path).
 * Idempotent — safe to call multiple times.
 */
function init(config) {
  if (_initCalled) return;
  _initCalled = true;

  const gpuLayerBudgetMB = config?.inference?.gpuLayerBudgetMB ?? 512;
  const threads = config?.inference?.threads ?? 4;

  const engineWasInjected = !!_engine; // true in tests via _inject()
  if (!_engine) {
    _engine = new InferenceEngine({ gpuLayerBudgetMB, threads });
  }
  if (!_embedder) {
    _embedder = new LocalEmbedder();
  }

  // When engine was injected (test DI), skip the production download path.
  // Tests call _onEngineReady() manually to control when the queue drains.
  if (engineWasInjected) return;

  const manager = new ProductionModelManager();

  // Fire and forget — errors propagate to queue rejection
  (async () => {
    try {
      const modelPath = await manager.ensureModel();
      await _engine.load(modelPath);
      await _embedder.load();
      _ready = true;
      log.info('brain ready');
      _drainQueue();
    } catch (err) {
      log.warn(`brain init failed: ${err.message}`);
      _onEngineError(err);
    }
  })();
}

/** Returns the embedding model name (constant — available before init). */
function embeddingModel() {
  return LocalEmbedder.MODEL_NAME;
}

/**
 * Chat inference. Queues if engine not ready.
 * timeoutMs is stripped from queued calls and applied only after dequeue.
 *
 * @param {Array<{role:string,content:string}>} messages
 * @param {object} opts — { maxTokens, temperature, timeoutMs }
 * @returns {Promise<string>}
 */
function chat(messages, opts = {}) {
  if (_ready && _engine) {
    return _engine.chat(messages, opts);
  }

  // Queue — strip timeoutMs, re-attach after dequeue
  const { timeoutMs, ...restOpts } = opts;
  return new Promise((resolve, reject) => {
    if (_queue.length >= QUEUE_MAX) {
      return reject(new Error('brain: queue full — too many pending calls'));
    }
    _queue.push({
      messages,
      opts: timeoutMs != null ? { ...restOpts, timeoutMs } : restOpts,
      resolve,
      reject,
    });
  });
}

/**
 * Raw completion — wraps prompt as single user message.
 * consolidator.js uses this; the instruct template is applied by inference-engine.
 */
function generate(prompt, opts = {}) {
  return chat([{ role: 'user', content: prompt }], opts);
}

/**
 * Embed text. Returns Float32Array or null when not ready.
 */
async function embed(text) {
  if (!_embedder?._loaded) return null;
  return _embedder.embed(text);
}

// ─── Utilities (unchanged from ollama.js) ────────────────────────────────────

function tryParseJsonArray(text) {
  if (!text) return null;
  const match = text.match(/\[[\s\S]*?\]/);
  if (!match) return null;
  try {
    const arr = JSON.parse(match[0]);
    return Array.isArray(arr) ? arr : null;
  } catch {
    return null;
  }
}

function tryParseJsonObject(text) {
  if (!text) return null;
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    const obj = JSON.parse(match[0]);
    return obj && typeof obj === 'object' ? obj : null;
  } catch {
    return null;
  }
}

module.exports = {
  init,
  embeddingModel,
  chat,
  generate,
  embed,
  tryParseJsonArray,
  tryParseJsonObject,
  // Test DI — not for production use
  _inject,
  _onEngineReady,
  _onEngineError,
};
