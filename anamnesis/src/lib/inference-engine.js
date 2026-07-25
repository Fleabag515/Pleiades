'use strict';

/**
 * lib/inference-engine.js — LLM inference via node-llama-cpp.
 *
 * Wraps node-llama-cpp v3 (ESM — loaded via dynamic import).
 * Manages one model instance, one context, serialized calls.
 */

const { exec } = require('child_process');
const os = require('os');
const log = require('./logger.js').make('inference-engine');

const VRAM_PER_LAYER_MB = 55; // Qwen2.5-1.5B approximate VRAM per layer
const CONTEXT_SIZE = 2048;

class InferenceEngine {
  /**
   * @param {object} opts
   * @param {boolean} opts._skipLoad     — skip everything (test DI)
   * @param {Function} opts._llamaFactory — async () => llamaInstance (test DI)
   * @param {string} opts._platform      — override os.platform() (test DI)
   * @param {number} opts.gpuLayerBudgetMB
   * @param {string} opts.modelPath
   */
  constructor({
    _skipLoad = false,
    _llamaFactory = null,
    _platform = null,
    gpuLayerBudgetMB = 512,
    modelPath = null,
    threads = 4,
  } = {}) {
    // threads: keep the helper brain SMALL. It shares the machine with the
    // primary inference server, whose MoE decode is CPU-bandwidth-bound —
    // letting node-llama-cpp default to every core starves the main model
    // (observed: load avg 25/16, Ornith decode 15 → 1-2 tok/s).
    this.threads = Math.max(1, threads);
    this._skipLoad = _skipLoad;
    this._llamaFactory = _llamaFactory;
    this._platform = _platform;
    this.gpuLayerBudgetMB = gpuLayerBudgetMB;
    this._modelPath = modelPath;

    this._llama = null;
    this._model = null;
    this._ctx = null;
    this._loaded = false;
    this._loadPromise = null;

    // Serialization queue — one inference at a time
    this._queue = Promise.resolve();
  }

  isLoaded() {
    return this._loaded;
  }

  async load(modelPath) {
    if (this._loaded) return;
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = this._doLoad(modelPath ?? this._modelPath);
    return this._loadPromise;
  }

  async _doLoad(modelPath) {
    if (this._skipLoad) return;
    try {
      const gpuLayers = await this._resolveGpuLayers();

      let getLlama;
      if (this._llamaFactory) {
        // Test DI — mock llama object
        this._llama = await this._llamaFactory();
      } else {
        // Production — dynamic import because node-llama-cpp is ESM-only
        ({ getLlama } = await import('node-llama-cpp'));
        // Pass gpu:false when budget is 0 to prevent CUDA initialisation entirely.
        // With gpu:'auto' (the default), node-llama-cpp initialises CUDA even when
        // gpuLayers=0, which aborts with a PTX toolchain mismatch on some systems.
        this._llama = await getLlama(gpuLayers === 0 ? { gpu: false } : {});
      }

      // Normalize to forward slashes — node-llama-cpp's native code may reject
      // Windows-style backslash paths even when Node.js itself handles them fine.
      const normalizedPath = modelPath.replace(/\\/g, '/');
      this._model = await this._llama.loadModel({ modelPath: normalizedPath, gpuLayers });
      this._ctx = await this._model.createContext({
        contextSize: CONTEXT_SIZE,
        threads: this.threads,
      });
      this._loaded = true;
      log.info(`model loaded: ${modelPath} (gpuLayers=${gpuLayers})`);
    } catch (err) {
      log.warn(`model load failed: ${err.message}`);
      throw err;
    }
  }

  /**
   * Resolve GPU layers. Platform-aware, fails safe to 0 (CPU only).
   */
  async _resolveGpuLayers() {
    const platform = this._platform ?? os.platform();
    const budgetLayers = Math.floor(this.gpuLayerBudgetMB / VRAM_PER_LAYER_MB);

    if (platform === 'darwin') {
      // Apple Silicon: unified memory not queryable, use budget directly
      log.info(
        `[inference-engine] gpuLayers=${budgetLayers} (budget=${this.gpuLayerBudgetMB}MB, platform=metal)`
      );
      return budgetLayers;
    }

    try {
      const freeMB = await this._probeGpuFreeMB();
      const fromVram = Math.floor(freeMB / VRAM_PER_LAYER_MB);
      const layers = Math.min(fromVram, budgetLayers);
      log.info(
        `[inference-engine] gpuLayers=${layers} (budget=${this.gpuLayerBudgetMB}MB, freeVram=${freeMB}MB, platform=${platform})`
      );
      return layers;
    } catch (err) {
      log.info(`[inference-engine] GPU probe failed (${err.message}) — CPU only`);
      return 0;
    }
  }

  /** Returns free VRAM in MB. Throws on failure. Overridable for tests. */
  async _probeGpuFreeMB() {
    return new Promise((resolve, reject) => {
      exec(
        'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits',
        { timeout: 5000 },
        (err, stdout) => {
          if (err) return reject(err);
          const values = stdout
            .trim()
            .split('\n')
            .map((l) => parseInt(l.trim(), 10))
            .filter((n) => !isNaN(n));
          if (!values.length) return reject(new Error('no VRAM data from nvidia-smi'));
          resolve(Math.max(...values));
        }
      );
    });
  }

  /** Enqueue a function so only one inference runs at a time */
  _serialize(fn) {
    const next = this._queue.then(fn);
    this._queue = next.catch(() => {}); // don't poison the queue on errors
    return next;
  }

  /**
   * Chat inference. messages: [{role, content}]
   * opts: { maxTokens, temperature, timeoutMs }
   */
  async chat(messages, opts = {}) {
    return this._serialize(() => this._doChat(messages, opts));
  }

  /**
   * Raw completion — wraps prompt as user message so instruct template applies.
   */
  async generate(prompt, opts = {}) {
    return this.chat([{ role: 'user', content: prompt }], opts);
  }

  async _doChat(messages, opts = {}) {
    if (!this._loaded) throw new Error('inference-engine: model not loaded');
    const { maxTokens = 512, temperature = 0.1, timeoutMs } = opts;

    const seq = this._ctx.getSequence();
    try {
      const systemMsg = messages.find((m) => m.role === 'system')?.content;
      const userMsg = messages.filter((m) => m.role === 'user').pop()?.content ?? '';

      let resultPromise;
      if (this._llamaFactory) {
        // Test path — mock doesn't do real inference
        resultPromise = Promise.resolve(userMsg);
      } else {
        // Production path — use LlamaChatSession
        const { LlamaChatSession } = await import('node-llama-cpp');
        const session = new LlamaChatSession({
          contextSequence: seq,
          ...(systemMsg ? { systemPrompt: systemMsg } : {}),
        });
        resultPromise = session.prompt(userMsg, { maxTokens, temperature });
      }

      if (timeoutMs) {
        const timeout = new Promise((_, rej) =>
          setTimeout(() => rej(new Error(`inference timed out after ${timeoutMs}ms`)), timeoutMs)
        );
        const result = await Promise.race([resultPromise, timeout]);
        return (typeof result === 'string' ? result : String(result)).trim();
      }

      const result = await resultPromise;
      return (typeof result === 'string' ? result : String(result)).trim();
    } finally {
      await seq.dispose?.();
    }
  }
}

module.exports = { InferenceEngine };
