'use strict';

/**
 * consolidator.js — Episode building.
 *
 * Periodically clusters unclustered Engrams into thematic Episodes.
 * Uses cosine similarity to group related facts, then generates a title +
 * summary for each scene via an LLM call.
 *
 * Algorithm:
 *   1. Fetch all unclustered engrams for each active session.
 *   2. Build similarity graph between cells (cosine, single pass).
 *   3. Greedily cluster by sceneClusterThreshold.
 *   4. For each cluster ≥ minSceneSize: generate title+summary, upsert Episode.
 *   5. Update decay scores across all engrams, prune below threshold.
 *
 * Scheduling: self-rescheduling setTimeout chain with a `_running` guard so
 * a slow run can never overlap with the next tick.
 */

const brain = require('./lib/brain.js');
const { EPISODE_SCENE } = require('./lib/prompts.js');
const HistoryStore = require('./history.js');
const log = require('./lib/logger.js').make('consolidator');

// Static cosine helper (embedder instance method table lives on the class).
function cosineStatic(a, b) {
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

class Consolidator {
  constructor(config, historyStore, embedder) {
    this.cfg = config;
    this.history = historyStore;
    this.embedder = embedder;
    this._timer = null;
    this._running = false;
    this._stopped = false;
  }

  start(intervalMs) {
    this._stopped = false;
    this._intervalMs = intervalMs;
    log.info(`running every ${intervalMs / 1000}s`);
    this._scheduleNext(intervalMs);
  }

  stop() {
    this._stopped = true;
    if (this._timer) clearTimeout(this._timer);
    this._timer = null;
  }

  _scheduleNext(delayMs) {
    if (this._stopped) return;
    this._timer = setTimeout(async () => {
      if (this._running) {
        this._scheduleNext(this._intervalMs);
        return;
      }
      this._running = true;
      try {
        await this.run();
      } catch (e) {
        log.warn('run error:', e.message);
      } finally {
        this._running = false;
        this._scheduleNext(this._intervalMs);
      }
    }, delayMs);
  }

  async run() {
    // Retire stale foresights first — cheap, and keeps the selector's
    // active set meaningful (see HistoryStore.FORESIGHT_TTL_DAYS).
    const expired = this.history.expireForesights();
    if (expired > 0) log.info(`expired ${expired} stale foresight(s)`);

    const sessions = this.history.db
      .prepare('SELECT DISTINCT session_key FROM engrams WHERE scene_id IS NULL')
      .all()
      .map((r) => r.session_key);

    for (const sessionKey of sessions) {
      await this._consolidateSession(sessionKey);
      this.history.updateDecayScores(sessionKey);
      const pruned = this.history.pruneDecayedMemcells(
        sessionKey,
        this.cfg.memory.decayPruneThreshold
      );
      if (pruned > 0)
        log.info(`session=${sessionKey.slice(0, 8)} pruned ${pruned} decayed engram(s)`);
    }
  }

  async _consolidateSession(sessionKey) {
    const cells = this.history.getUnclusteredMemcells(
      sessionKey,
      this.cfg.memory.consolidationBatchSize,
      this.cfg.memory.minEngramAgeSec ?? 300
    );
    if (cells.length < this.cfg.memory.minSceneSize) return;

    const currentModel = this.embedder.model;
    const cosine = this.embedder.constructor.cosine;
    const decoded = cells
      .map((c) => ({ ...c, vec: HistoryStore.toFloat32(c.embedding) }))
      .filter((c) => c.vec && (!c.embedding_model || c.embedding_model === currentModel));

    const threshold = this.cfg.memory.sceneClusterThreshold;
    const clusters = [];
    const assigned = new Set();

    // Centroid clustering: a candidate joins only if it is similar to the
    // cluster's *running mean*, not merely to the seed. Star-to-seed
    // clustering chains topics (A~seed, B~seed, but A and B unrelated) and
    // produces mixed-topic scenes whose summaries match everything a
    // little and nothing well — a primary drift vector at retrieval time.
    for (let i = 0; i < decoded.length; i++) {
      if (assigned.has(i)) continue;
      const cluster = [decoded[i]];
      assigned.add(i);
      const dim = decoded[i].vec.length;
      const sum = Float32Array.from(decoded[i].vec);
      const centroid = () => {
        const c = new Float32Array(dim);
        let norm = 0;
        for (let k = 0; k < dim; k++) {
          c[k] = sum[k] / cluster.length;
          norm += c[k] * c[k];
        }
        norm = Math.sqrt(norm) || 1;
        for (let k = 0; k < dim; k++) c[k] /= norm;
        return c;
      };
      for (let j = i + 1; j < decoded.length; j++) {
        if (assigned.has(j)) continue;
        if (decoded[j].vec.length !== dim) continue;
        const sim = cosine(centroid(), decoded[j].vec);
        if (sim >= threshold) {
          cluster.push(decoded[j]);
          assigned.add(j);
          for (let k = 0; k < dim; k++) sum[k] += decoded[j].vec[k];
        }
      }
      clusters.push({ cluster, centroid: centroid() });
    }

    let sceneCount = 0;
    let mergeCount = 0;
    const mergeThreshold = this.cfg.memory.sceneMergeThreshold ?? 0.8;
    const existing = this.history
      .getScenes(sessionKey)
      .map((s) => ({
        ...s,
        vec:
          !s.embedding_model || s.embedding_model === currentModel
            ? HistoryStore.toFloat32(s.embedding)
            : null,
      }))
      .filter((s) => s.vec);

    for (const { cluster, centroid } of clusters) {
      if (cluster.length < this.cfg.memory.minSceneSize) continue;

      const facts = cluster.map((c) => c.content);
      const cellIds = cluster.map((c) => c.id);
      const avgImportance = cluster.reduce((s, c) => s + (c.importance ?? 0.5), 0) / cluster.length;

      // Merge path: if an existing scene already covers this topic, extend
      // it instead of minting scene #47 about the same thing. Fragmented
      // near-duplicate scenes crowd each other out of the injection block.
      let merged = false;
      for (const s of existing) {
        if (cosine(centroid, s.vec) < mergeThreshold) continue;
        let oldIds = [];
        try {
          oldIds = JSON.parse(s.engram_ids);
        } catch {
          oldIds = [];
        }
        const combinedFacts = [s.summary, ...facts].slice(0, 12);
        const sceneData =
          (await this._generateScene(combinedFacts)) ?? this._fallbackScene(cluster, s.title);
        const grounded = await this._groundedEmbed(sceneData.summary, centroid);
        this.history.updateScene(
          s.id,
          sceneData.title,
          sceneData.summary,
          grounded,
          [...new Set([...oldIds, ...cellIds])],
          (s.avg_importance + avgImportance) / 2,
          currentModel
        );
        for (const cell of cluster) this.history.assignMemcellToScene(cell.id, s.id);
        merged = true;
        mergeCount++;
        break;
      }
      if (merged) continue;

      const sceneData = (await this._generateScene(facts)) ?? this._fallbackScene(cluster, null);
      const grounded = await this._groundedEmbed(sceneData.summary, centroid);
      const sceneId = this.history.insertScene(
        sessionKey,
        sceneData.title,
        sceneData.summary,
        grounded,
        cellIds,
        avgImportance,
        currentModel
      );
      for (const cell of cluster) this.history.assignMemcellToScene(cell.id, sceneId);
      sceneCount++;
    }

    if (sceneCount > 0 || mergeCount > 0)
      log.info(
        `session=${sessionKey.slice(0, 8)} built ${sceneCount} scene(s), extended ${mergeCount} ` +
          `from ${cells.length} engram(s)`
      );
  }

  /**
   * Deterministic, grounded-by-construction scene when the LLM fails or
   * confabulates: title from the highest-importance fact, summary from the
   * top facts verbatim. Boring beats invented.
   */
  _fallbackScene(cluster, title = null) {
    const byImp = [...cluster].sort((a, b) => (b.importance ?? 0.5) - (a.importance ?? 0.5));
    const top = byImp[0]?.content ?? 'Untitled scene';
    return {
      title: title ?? top.split(/\s+/).slice(0, 6).join(' '),
      summary: byImp
        .slice(0, 3)
        .map((c) => c.content)
        .join(' — ')
        .slice(0, 300),
    };
  }

  /**
   * Embed a generated summary, then verify it actually resembles the
   * cluster it claims to summarize. A summary that drifted from its own
   * facts (small-model confabulation) would otherwise be *indexed under
   * the confabulation* — retrieval then surfaces it for the wrong queries
   * forever. On failure, index under the cluster centroid itself.
   */
  async _groundedEmbed(summary, centroid) {
    const minSim = this.cfg.memory.summaryGroundingMinSim ?? 0.45;
    const v = await this.embedder.embed(summary).catch(() => null);
    if (v && cosineStatic(v, centroid) >= minSim) return v;
    log.warn('scene summary failed grounding check — indexing under cluster centroid');
    return centroid;
  }

  async _generateScene(facts) {
    const factList = facts.map((f, i) => `${i + 1}. ${f}`).join('\n');
    try {
      const text = await brain.chat([{ role: 'user', content: EPISODE_SCENE + factList }], {
        maxTokens: 256,
        temperature: 0.2,
        timeoutMs: 90000,
      });
      const parsed = brain.tryParseJsonObject(text);
      if (!parsed?.title || !parsed?.summary) return null;
      return { title: String(parsed.title), summary: String(parsed.summary) };
    } catch {
      return null;
    }
  }
}

module.exports = Consolidator;
