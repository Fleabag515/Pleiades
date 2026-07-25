'use strict';

/**
 * foresight.js — Future-intention extraction (EverMemOS-style Foresight).
 *
 * Runs in parallel with engram extraction. Each assistant turn is scanned
 * for statements about future plans, goals, or intended actions. These are
 * stored as foresights so the model always knows what it was about to do.
 *
 * Fields per foresight:
 *   - intention : what is planned (≤25 words)
 *   - target    : the object/subject (file, tool, project, system) or ""
 *   - timeframe : soon | days | weeks | months | ongoing
 *   - confidence: 0.0–1.0 — confidence ≥ 0.4 only, hedged language skipped
 *
 * IMPORTANT — own flag, not shared with extractor:
 *   This module tracks its own `foresight_scanned` column. Previously it
 *   piggybacked on `turns.extracted`, which caused silent starvation when
 *   whichever extractor finished a turn first set the flag.
 */

const brain = require('./lib/brain.js');
const HistoryStore = require('./history.js');
const { FORESIGHT } = require('./lib/prompts.js');
const { shouldProcessTurn } = require('./lib/heuristics.js');
const log = require('./lib/logger.js').make('foresight');

const VALID_TIMEFRAMES = ['soon', 'days', 'weeks', 'months', 'ongoing'];

class ForesightExtractor {
  constructor(config, historyStore, embedder = null) {
    this.cfg = config.foresight;
    this.history = historyStore;
    this.embedder = embedder; // optional: enables relevance vectors + auto-fulfill
    this._running = false;
    this._inflight = null;
  }

  async processBacklog() {
    const pending = this.history.getUnscannedAssistantTurns(this.cfg.startupBacklogLimit);
    if (!pending.length) return;
    log.info(`scanning ${pending.length} backlog turn(s) for intentions`);
    for (const turn of pending) await this._scanTurn(turn);
    log.info('backlog scan complete');
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
      log.info('flushing in-flight scan before shutdown...');
      await Promise.race([this._inflight, new Promise((r) => setTimeout(r, 15000))]);
    }
  }

  async _runBatch() {
    const turns = this.history.getUnscannedAssistantTurns(5);
    for (const turn of turns) await this._scanTurn(turn);
  }

  async _scanTurn(turn) {
    if (!shouldProcessTurn(turn.content)) {
      this.history.markForesightScanned(turn.id);
      return;
    }
    // Auto-fulfillment: if this turn is highly similar to an *older* active
    // foresight, the model has circled back to that plan — retire it so it
    // stops being re-injected forever. (Same-day repeats are just the plan
    // being restated, so require ≥1 day of age.) markForesightFulfilled did
    // not have a single caller before this: foresights were immortal.
    this._autoFulfill(turn).catch(() => {});

    let items = null;
    for (let attempt = 0; attempt <= this.cfg.maxRetries; attempt++) {
      try {
        items = await this._callLLM(turn.content);
        if (items !== null) break;
      } catch (err) {
        if (attempt === this.cfg.maxRetries)
          log.warn(`turn ${turn.id} failed after ${attempt + 1} attempts:`, err.message);
      }
    }

    if (!items?.length) {
      this.history.markForesightScanned(turn.id);
      return;
    }

    let count = 0;
    for (const item of items) {
      const intention = (item?.intention ?? '').trim();
      if (!intention || intention.length < 8) continue;

      const target = (item?.target ?? '').trim();
      const timeframe = VALID_TIMEFRAMES.includes(item?.timeframe) ? item.timeframe : 'soon';
      const confidence =
        typeof item?.confidence === 'number' ? Math.min(1, Math.max(0, item.confidence)) : 0.7;
      if (confidence < 0.4) continue;

      const vec = this.embedder
        ? await this.embedder.embed(intention).catch(() => null)
        : null;
      this.history.insertForesight(
        turn.session_key,
        turn.id,
        intention,
        target,
        timeframe,
        confidence,
        vec,
        vec ? this.embedder.model : null
      );
      count++;
    }

    this.history.markForesightScanned(turn.id);
    if (count > 0)
      log.info(`turn ${turn.id} → ${count} foresight(s) (session=${turn.session_key.slice(0, 8)})`);
  }

  async _autoFulfill(turn) {
    if (!this.embedder || !turn.embedding) return;
    if (turn.embedding_model && turn.embedding_model !== this.embedder.model) return;
    const tVec = HistoryStore.toFloat32(turn.embedding);
    if (!tVec) return;
    const cosine = this.embedder.constructor.cosine;
    const minAgeSec = this.cfg.fulfillMinAgeSec ?? 86400;
    const fulfillSim = this.cfg.fulfillSim ?? 0.72;
    const now = Math.floor(Date.now() / 1000);
    for (const f of this.history.getActiveForesights(turn.session_key, 50)) {
      if (!f.embedding) continue;
      if (f.embedding_model && f.embedding_model !== this.embedder.model) continue;
      if (now - f.created_at < minAgeSec) continue;
      const fVec = HistoryStore.toFloat32(f.embedding);
      if (fVec && cosine(tVec, fVec) >= fulfillSim) {
        this.history.markForesightFulfilled(f.id);
        log.info(`foresight ${f.id} auto-fulfilled ("${f.intention.slice(0, 50)}...")`);
      }
    }
  }

  async _callLLM(content) {
    const truncated = content.length > 2500 ? content.slice(0, 2500) + '...' : content;
    const text = await brain.chat([{ role: 'user', content: FORESIGHT + truncated }], {
      maxTokens: 400,
      temperature: 0.1,
      timeoutMs: this.cfg.timeoutMs,
    });
    return brain.tryParseJsonArray(text) ?? [];
  }
}

module.exports = ForesightExtractor;
