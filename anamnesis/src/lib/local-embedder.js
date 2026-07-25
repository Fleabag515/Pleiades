'use strict';

/**
 * lib/local-embedder.js — sentence embedding via @huggingface/transformers.
 *
 * Loads all-MiniLM-L6-v2 (ONNX/WASM, ~23MB, CPU-only) and returns
 * mean-pooled, L2-normalized Float32Array embeddings.
 */

const log = require('./logger.js').make('local-embedder');

class LocalEmbedder {
  /**
   * @param {object} opts
   * @param {boolean} opts._skipLoad — skip everything (test DI)
   * @param {Function} opts._pipeline — pre-built pipeline function (test DI)
   * @param {Function} opts._pipelineFactory — async (task, model) => pipeline (test DI)
   */
  constructor({ _skipLoad = false, _pipeline = null, _pipelineFactory = null } = {}) {
    this._loaded = false;
    this._pipeline = _pipeline;
    this._pipelineFactory = _pipelineFactory;
    this._skipLoad = _skipLoad;
    this._loadPromise = null;
  }

  async load() {
    if (this._loaded) return;
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = this._doLoad();
    return this._loadPromise;
  }

  async _doLoad() {
    if (this._skipLoad) return;
    if (this._pipeline) {
      // Already provided via test DI
      this._loaded = true;
      return;
    }
    try {
      const factory =
        this._pipelineFactory ??
        (async (task, model) => {
          const { pipeline } = require('@huggingface/transformers');
          return pipeline(task, model);
        });
      this._pipeline = await factory('feature-extraction', LocalEmbedder.MODEL_NAME);
      this._loaded = true;
      log.info(`embedding model loaded: ${LocalEmbedder.MODEL_NAME}`);
    } catch (err) {
      log.warn(`failed to load embedding model: ${err.message}`);
      throw err;
    }
  }

  /**
   * Embed text. Returns Float32Array (mean-pooled, L2-normalized) or null if not loaded.
   */
  async embed(text) {
    if (!this._loaded || !this._pipeline) return null;
    try {
      const output = await this._pipeline(text);
      // output.data: Float32Array of shape [batchSize * seqLen * hiddenDim]
      // output.dims: [batchSize, seqLen, hiddenDim]
      const [, seqLen, hiddenDim] = output.dims;
      const raw = output.data;

      // Mean pool across sequence dimension
      const pooled = new Float32Array(hiddenDim);
      for (let s = 0; s < seqLen; s++) {
        for (let h = 0; h < hiddenDim; h++) {
          pooled[h] += raw[s * hiddenDim + h];
        }
      }
      for (let h = 0; h < hiddenDim; h++) pooled[h] /= seqLen;

      // L2 normalize
      let norm = 0;
      for (let h = 0; h < hiddenDim; h++) norm += pooled[h] * pooled[h];
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let h = 0; h < hiddenDim; h++) pooled[h] /= norm;
      }

      return pooled;
    } catch (err) {
      log.warn(`embed() error: ${err.message}`);
      return null;
    }
  }
}

LocalEmbedder.MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';

module.exports = { LocalEmbedder };
