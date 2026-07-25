'use strict';

/**
 * embedder.js — embedding singleton backed by brain.js (local-embedder).
 *
 * Replaces the Ollama-based implementation. Delegates embed() to brain so
 * the actual model (all-MiniLM-L6-v2) is loaded once at daemon startup.
 *
 * Callers (selector.js, extractor.js, consolidator.js, proxy.js) receive
 * the exported singleton and call embedder.embed() / embedder.model —
 * the interface is unchanged.
 */

const brain = require('./lib/brain.js');

class Embedder {
  /** Model name — used to tag stored vectors for compatibility checks. */
  get model() {
    return brain.embeddingModel();
  }

  /**
   * Embed a single string. Returns Float32Array or null when model not ready.
   * Failures are logged inside brain and swallowed — callers handle null.
   */
  async embed(text) {
    if (!text) return null;
    return brain.embed(text);
  }

  /**
   * Cosine similarity between two Float32Arrays.
   * Returns 0 for missing vectors or mismatched lengths.
   */
  static cosine(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0,
      na = 0,
      nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }
}

module.exports = new Embedder();
