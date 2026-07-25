'use strict';

/**
 * extractor.js — Engram extraction with importance + category scoring.
 *
 * Each assistant turn is decomposed into atomic facts. Each fact gets:
 *   - importance: 0.0–1.0 (how durable/useful is this fact?)
 *   - category:   technical | decision | preference | personal | context | other
 *
 * Importance affects decay rate — high-importance facts resist pruning.
 * Category enables targeted retrieval and shields decisions/preferences
 * from decay-based pruning entirely.
 *
 * Resilience:
 *   - Turns are stored in SQLite BEFORE extraction — survives process death.
 *   - `extracted=0` turns are retried on next startup via processBacklog().
 *   - JSON parse failures retried up to maxRetries times.
 *   - shouldProcessTurn() filters trivial/noisy turns before hitting the LLM.
 */

const brain = require('./lib/brain.js');
const HistoryStore = require('./history.js');
const { ENGRAM_EXTRACTION, ENGRAM_EXTRACTION_USER } = require('./lib/prompts.js');
const { shouldProcessTurn } = require('./lib/heuristics.js');
const log = require('./lib/logger.js').make('extractor');

// Local cosine — deliberately not this.embedder.constructor.cosine, so the
// grounding check also works with duck-typed embedders (tests, future
// backends) that only implement embed()/model.
function cosine(a, b) {
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

const CATEGORIES = ['technical', 'decision', 'preference', 'personal', 'context', 'other'];

class Extractor {
  constructor(config, historyStore, embedder) {
    this.cfg = config.extraction;
    this.history = historyStore;
    this.embedder = embedder;
    this._running = false;
    this._inflight = null;
  }

  async processBacklog() {
    const pending = this.history.getUnextractedTurns(
      this.cfg.startupBacklogLimit,
      this.cfg.includeUserTurns ?? true
    );
    if (!pending.length) return;
    log.info(`processing ${pending.length} unextracted turn(s) from backlog`);
    for (const turn of pending) await this._extractTurn(turn);
    log.info('backlog cleared');
  }

  processBatch() {
    if (this._running) return this._inflight ?? Promise.resolve();
    this._running = true;
    this._inflight = this._runBatch().finally(() => {
      this._running = false;
      this._inflight = null;
    });
    return this._inflight;
  }

  async flushInFlight() {
    if (this._inflight) {
      log.info('flushing in-flight extraction before shutdown...');
      await Promise.race([this._inflight, new Promise((r) => setTimeout(r, 15000))]);
    }
  }

  async _runBatch() {
    const turns = this.history.getUnextractedTurns(5, this.cfg.includeUserTurns ?? true);
    for (const turn of turns) await this._extractTurn(turn);
  }

  async _extractTurn(turn) {
    if (!shouldProcessTurn(turn.content)) {
      this.history.markExtracted(turn.id);
      return;
    }

    let facts = null;
    for (let attempt = 0; attempt <= this.cfg.maxRetries; attempt++) {
      try {
        facts = await this._callLLM(turn.content, turn.role);
        if (facts?.length) break;
      } catch (err) {
        if (attempt === this.cfg.maxRetries)
          log.warn(`turn ${turn.id} failed after ${attempt + 1} attempts:`, err.message);
      }
    }

    if (!facts?.length) {
      this.history.markExtracted(turn.id);
      return;
    }

    // Grounding vector: what the source turn actually says. Facts whose
    // embedding is far from the source are the extractor model inventing —
    // small local models pad and embellish under pressure to produce output.
    let srcVec = null;
    if (turn.embedding && (!turn.embedding_model || turn.embedding_model === this.embedder.model)) {
      srcVec = HistoryStore.toFloat32(turn.embedding);
    }
    if (!srcVec) {
      srcVec = await this.embedder.embed(turn.content.slice(0, 2000)).catch(() => null);
    }
    const minGrounding = this.cfg.groundingMinSim ?? 0.42;

    let count = 0;
    let dropped = 0;
    for (const item of facts.slice(0, 6)) {
      const fact = typeof item === 'string' ? item : item?.fact;
      if (!fact || fact.trim().length < 10) continue;

      const importance =
        typeof item?.importance === 'number' ? Math.min(1, Math.max(0, item.importance)) : 0.5;
      const category = CATEGORIES.includes(item?.category) ? item.category : 'other';

      const embedding = await this.embedder.embed(fact.trim()).catch(() => null);
      if (srcVec && embedding) {
        const g = cosine(embedding, srcVec);
        if (g < minGrounding) {
          dropped++;
          continue; // fact does not resemble the turn it claims to come from
        }
      }
      this.history.insertMemcell(
        turn.session_key,
        turn.id,
        fact.trim(),
        embedding,
        importance,
        category,
        this.embedder.model
      );
      count++;
    }

    this.history.markExtracted(turn.id);
    if (count > 0 || dropped > 0)
      log.info(
        `turn ${turn.id} → ${count} engram(s)` +
          (dropped ? ` (+${dropped} dropped ungrounded)` : '') +
          ` (session=${turn.session_key.slice(0, 8)})`
      );
  }

  async _callLLM(content, role = 'assistant') {
    const prompt = role === 'user' ? ENGRAM_EXTRACTION_USER : ENGRAM_EXTRACTION;
    const truncated = content.length > 2500 ? content.slice(0, 2500) + '...' : content;
    const text = await brain.chat([{ role: 'user', content: prompt + truncated }], {
      maxTokens: 500,
      temperature: 0.1,
      timeoutMs: this.cfg.timeoutMs,
    });
    return brain.tryParseJsonArray(text);
  }
}

module.exports = Extractor;
