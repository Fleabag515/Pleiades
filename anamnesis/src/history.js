/**
 * history.js — SQLite-backed memory store for Anamnesis.
 *
 * Schema overview:
 *   turns       — every prompt/response, raw text + embedding + token estimate.
 *                 `extracted` flags engram-extraction status; `foresight_scanned`
 *                 flags future-intention scan status. These are *independent*
 *                 because the two extractors run in parallel and shouldn't
 *                 starve each other.
 *   engrams    — atomic facts derived from assistant turns. `embedding_model`
 *                 records which model produced the vector so we never compare
 *                 vectors from different model families.
 *   episodes   — thematic clusters of engrams (turn → cell → scene).
 *   foresights  — extracted intentions / future plans.
 */

const Database = require('better-sqlite3');
const fs = require('fs');
const path = require('path');

const CATEGORIES = ['technical', 'decision', 'preference', 'personal', 'context', 'other'];
const TIMEFRAMES = ['soon', 'days', 'weeks', 'months', 'ongoing'];

class HistoryStore {
  // Days before an unfulfilled foresight is retired, per declared timeframe.
  static FORESIGHT_TTL_DAYS = { soon: 3, days: 7, weeks: 30, months: 90, ongoing: 45 };

  constructor(dbPath) {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this._init();
  }

  _init() {
    // 1. Base tables — no columns added here that aren't safe for old DBs;
    //    new columns are introduced via _migrate() so existing DBs upgrade.
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS turns (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        session_key       TEXT    NOT NULL,
        role              TEXT    NOT NULL,
        content           TEXT    NOT NULL,
        embedding         BLOB,
        token_est         INTEGER NOT NULL DEFAULT 0,
        recall_count      INTEGER NOT NULL DEFAULT 0,
        importance        REAL    NOT NULL DEFAULT 0.5,
        extracted         INTEGER NOT NULL DEFAULT 0,
        category          TEXT    NOT NULL DEFAULT 'fleagle',
        created_at        INTEGER NOT NULL DEFAULT (unixepoch())
      );

      CREATE TABLE IF NOT EXISTS engrams (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id       INTEGER REFERENCES turns(id) ON DELETE CASCADE,
        session_key   TEXT    NOT NULL,
        content       TEXT    NOT NULL,
        embedding     BLOB,
        recall_count  INTEGER NOT NULL DEFAULT 0,
        decay_score   REAL    NOT NULL DEFAULT 1.0,
        scene_id      INTEGER,
        created_at    INTEGER NOT NULL DEFAULT (unixepoch())
      );

      CREATE TABLE IF NOT EXISTS episodes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_key   TEXT    NOT NULL,
        title         TEXT    NOT NULL,
        summary       TEXT    NOT NULL,
        embedding     BLOB,
        engram_ids   TEXT    NOT NULL DEFAULT '[]',
        recall_count  INTEGER NOT NULL DEFAULT 0,
        created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
        updated_at    INTEGER NOT NULL DEFAULT (unixepoch())
      );

      CREATE TABLE IF NOT EXISTS foresights (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id       INTEGER REFERENCES turns(id) ON DELETE CASCADE,
        session_key   TEXT    NOT NULL,
        intention     TEXT    NOT NULL,
        target        TEXT    NOT NULL DEFAULT '',
        timeframe     TEXT    NOT NULL DEFAULT 'soon',
        confidence    REAL    NOT NULL DEFAULT 0.7,
                fulfilled     INTEGER NOT NULL DEFAULT 0,
        created_at    INTEGER NOT NULL DEFAULT (unixepoch())
      );

      CREATE TABLE IF NOT EXISTS character_profile (
        id                  INTEGER PRIMARY KEY,
        source_type         TEXT    NOT NULL,
        source_path         TEXT,
        source_mtime        INTEGER,
        raw_content         TEXT    NOT NULL DEFAULT '',
        parsed_summary      TEXT    NOT NULL DEFAULT '{}',
        evolution_notes     TEXT    NOT NULL DEFAULT '',
        drift_reminder      TEXT    NOT NULL DEFAULT '',
        drift_checked_at    INTEGER NOT NULL DEFAULT 0,
        updated_at          INTEGER NOT NULL DEFAULT (unixepoch())
      );

      CREATE TABLE IF NOT EXISTS character_observations (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_key   TEXT    NOT NULL,
        turn_id       INTEGER,
        observed_at   INTEGER NOT NULL DEFAULT (unixepoch()),
        obs_type      TEXT    NOT NULL,
        detail        TEXT    NOT NULL,
        consolidated  INTEGER NOT NULL DEFAULT 0
      );
    `);
    this._migrate();

    // Indices last, so all migrated columns are guaranteed to exist.
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_turns_session       ON turns(session_key, created_at);
      CREATE INDEX IF NOT EXISTS idx_turns_extracted     ON turns(extracted);
      CREATE INDEX IF NOT EXISTS idx_turns_foresight     ON turns(foresight_scanned);
      CREATE INDEX IF NOT EXISTS idx_engrams_session    ON engrams(session_key, created_at);
      CREATE INDEX IF NOT EXISTS idx_engrams_scene      ON engrams(scene_id);
      CREATE INDEX IF NOT EXISTS idx_engrams_cat        ON engrams(category);
      CREATE INDEX IF NOT EXISTS idx_engrams_turn       ON engrams(turn_id);
      CREATE INDEX IF NOT EXISTS idx_scenes_session      ON episodes(session_key, updated_at);
      CREATE INDEX IF NOT EXISTS idx_foresights_session  ON foresights(session_key, created_at);
      CREATE INDEX IF NOT EXISTS idx_foresights_active   ON foresights(session_key, fulfilled);
      CREATE INDEX IF NOT EXISTS idx_char_obs_session   ON character_observations(session_key, observed_at);
      CREATE INDEX IF NOT EXISTS idx_char_obs_pending   ON character_observations(consolidated, observed_at);
    `);
  }

  _migrate() {
    // v0.4 → v0.5: rename memcells/memscenes tables to engrams/episodes
    const tables = this.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table'")
      .all()
      .map((r) => r.name);
    if (tables.includes('memcells')) {
      if (!tables.includes('engrams')) {
        this.db.exec('ALTER TABLE memcells RENAME TO engrams');
      } else {
        // engrams table already exists (created by schema) — merge orphaned memcells in
        this.db.exec(`
          INSERT INTO engrams (turn_id, session_key, content, embedding, recall_count, decay_score, scene_id, created_at, importance, category, embedding_model)
          SELECT turn_id, session_key, content, embedding, recall_count, decay_score, scene_id, created_at, importance, category, embedding_model
          FROM memcells WHERE id NOT IN (SELECT id FROM engrams)
        `);
        this.db.exec('DROP TABLE memcells');
      }
    }
    if (tables.includes('memscenes') && !tables.includes('episodes'))
      this.db.exec('ALTER TABLE memscenes RENAME TO episodes');

    const has = (table, col) =>
      this.db
        .prepare(`PRAGMA table_info(${table})`)
        .all()
        .some((c) => c.name === col);

    // turns.foresight_scanned — replaces the previous abuse of turns.extracted
    // by the foresight extractor (which never set it, leading to a silent race).
    if (!has('turns', 'foresight_scanned')) {
      this.db.exec('ALTER TABLE turns ADD COLUMN foresight_scanned INTEGER NOT NULL DEFAULT 0');
      // For any existing turn that already has engrams, assume foresight has
      // also been processed — otherwise we'd re-scan the entire history.
      this.db.exec(`
        UPDATE turns SET foresight_scanned = 1
        WHERE extracted = 1
      `);
    }

    // engrams.importance / category — added in 0.2.0
    if (!has('engrams', 'importance'))
      this.db.exec('ALTER TABLE engrams ADD COLUMN importance REAL NOT NULL DEFAULT 0.5');
    if (!has('engrams', 'category'))
      this.db.exec("ALTER TABLE engrams ADD COLUMN category TEXT NOT NULL DEFAULT 'other'");

    // episodes.avg_importance — added in 0.2.0
    if (!has('episodes', 'avg_importance'))
      this.db.exec('ALTER TABLE episodes ADD COLUMN avg_importance REAL NOT NULL DEFAULT 0.5');

    // embedding_model — track which model produced each vector so we never
    // do cosine across incompatible vector spaces. Old NULL rows are treated
    // as legacy and skipped from similarity if cur. model differs.
    if (!has('turns', 'embedding_model'))
      this.db.exec('ALTER TABLE turns ADD COLUMN embedding_model TEXT');
    if (!has('engrams', 'embedding_model'))
      this.db.exec('ALTER TABLE engrams ADD COLUMN embedding_model TEXT');
    if (!has('episodes', 'embedding_model'))
      this.db.exec('ALTER TABLE episodes ADD COLUMN embedding_model TEXT');

    // foresights.embedding — added for relevance-gated foresight injection
    // and similarity-based auto-fulfillment (0.7.x). Legacy NULL rows are
    // injected by recency only, exactly like before.
    if (!has('foresights', 'embedding'))
      this.db.exec('ALTER TABLE foresights ADD COLUMN embedding BLOB');
    if (!has('foresights', 'embedding_model'))
      this.db.exec('ALTER TABLE foresights ADD COLUMN embedding_model TEXT');

    // turns.category — added for category-partitioned memory quotas (see
    // selector.js _fillBudget). Default 'fleagle' means every pre-existing
    // turn (and every turn from a caller that never sends the category
    // header) is treated as owner-authored content, which is the safe
    // default: nothing gets silently deprioritized just because this
    // shipped. Only a caller that explicitly tags requests (e.g. an
    // autonomous/background task) as a different category gets partitioned
    // away from the guaranteed floor.
    if (!has('turns', 'category'))
      this.db.exec("ALTER TABLE turns ADD COLUMN category TEXT NOT NULL DEFAULT 'fleagle'");
  }

  // ─── Turns ────────────────────────────────────────────────────────────────

  insertTurn(sessionKey, role, content, embedding, tokenEst, embeddingModel = null, category = 'fleagle') {
    const blob = embedding ? Buffer.from(embedding.buffer) : null;
    return this.db
      .prepare(
        `
      INSERT INTO turns (session_key, role, content, embedding, token_est, embedding_model, category)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `
      )
      .run(sessionKey, role, content, blob, tokenEst, embeddingModel, category || 'fleagle').lastInsertRowid;
  }

  getSessionTurns(sessionKey) {
    return this.db
      .prepare(
        `
      SELECT id, role, content, embedding, embedding_model, token_est, recall_count, importance, category, created_at
      FROM turns WHERE session_key=? ORDER BY created_at ASC, id ASC
    `
      )
      .all(sessionKey);
  }

  getUnextractedTurns(limit = 20, includeUser = false) {
    const roles = includeUser ? `('assistant','user')` : `('assistant')`;
    return this.db
      .prepare(
        `
      SELECT id, session_key, role, content, embedding, embedding_model FROM turns
      WHERE extracted=0 AND role IN ${roles}
      ORDER BY created_at ASC LIMIT ?
    `
      )
      .all(limit);
  }

  getUnextractedAssistantTurns(limit = 20) {
    return this.getUnextractedTurns(limit, false);
  }

  getUnscannedAssistantTurns(limit = 20) {
    return this.db
      .prepare(
        `
      SELECT id, session_key, role, content, embedding, embedding_model FROM turns
      WHERE foresight_scanned=0 AND role='assistant'
      ORDER BY created_at ASC LIMIT ?
    `
      )
      .all(limit);
  }

  markExtracted(id) {
    this.db.prepare('UPDATE turns SET extracted=1 WHERE id=?').run(id);
  }
  markForesightScanned(id) {
    this.db.prepare('UPDATE turns SET foresight_scanned=1 WHERE id=?').run(id);
  }
  bumpTurnRecall(id) {
    this.db.prepare('UPDATE turns SET recall_count=recall_count+1 WHERE id=?').run(id);
  }

  // ─── Engrams ─────────────────────────────────────────────────────────────

  insertMemcell(
    sessionKey,
    turnId,
    content,
    embedding,
    importance = 0.5,
    category = 'other',
    embeddingModel = null
  ) {
    const blob = embedding ? Buffer.from(embedding.buffer) : null;
    const cat = CATEGORIES.includes(category) ? category : 'other';
    return this.db
      .prepare(
        `
      INSERT INTO engrams (session_key, turn_id, content, embedding, importance, category, embedding_model)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `
      )
      .run(sessionKey, turnId, content, blob, importance, cat, embeddingModel).lastInsertRowid;
  }

  getUnclusteredMemcells(sessionKey, limit = 100, minAgeSec = 0) {
    // Session-scoped: engrams from one session cluster with their own kind.
    // (This used to be cross-session, which let scenes absorb engrams from
    // unrelated buckets and then claim them under whichever session
    // triggered consolidation — instant cross-contamination.) A minimum age
    // lets a topic finish unfolding before it is frozen into a scene.
    const cutoff = Math.floor(Date.now() / 1000) - Math.max(0, minAgeSec);
    return this.db
      .prepare(
        `
      SELECT id, content, embedding, embedding_model, importance FROM engrams
      WHERE scene_id IS NULL AND session_key=? AND created_at<=?
      ORDER BY created_at ASC LIMIT ?
    `
      )
      .all(sessionKey, cutoff, limit);
  }

  getAllMemcells(sessionKey) {
    return this.db
      .prepare(
        `
      SELECT id, content, embedding, embedding_model, importance, category, scene_id, decay_score, created_at
      FROM engrams WHERE session_key=? ORDER BY created_at ASC
    `
      )
      .all(sessionKey);
  }

  assignMemcellToScene(id, sceneId) {
    this.db.prepare('UPDATE engrams SET scene_id=? WHERE id=?').run(sceneId, id);
  }

  /**
   * Look up the turn IDs that produced a set of engrams. Used by the
   * selector to expand a relevant scene back to the underlying conversation.
   * Replaces an earlier `selector.history.db.prepare(...)` reach into internals.
   */
  getTurnIdsForMemcells(engramIds) {
    if (!engramIds.length) return [];
    const ph = engramIds.map(() => '?').join(',');
    return this.db
      .prepare(`SELECT DISTINCT turn_id FROM engrams WHERE id IN (${ph}) AND turn_id IS NOT NULL`)
      .all(...engramIds)
      .map((r) => r.turn_id);
  }

  updateDecayScores(sessionKey) {
    const now = Math.floor(Date.now() / 1000);
    const cells = this.db
      .prepare('SELECT id, created_at, recall_count, importance FROM engrams WHERE session_key=?')
      .all(sessionKey);
    const update = this.db.prepare('UPDATE engrams SET decay_score=? WHERE id=?');
    this.db.transaction(() => {
      for (const c of cells) {
        const ageDays = (now - c.created_at) / 86400;
        const halfLife = 30 + (c.importance ?? 0.5) * 60;
        const recency = Math.exp(-ageDays / halfLife);
        const recall = Math.log1p(c.recall_count) / 5;
        update.run(Math.min(1.0, recency + recall), c.id);
      }
    })();
  }

  pruneDecayedMemcells(sessionKey, threshold = 0.05) {
    return this.db
      .prepare(
        `
      DELETE FROM engrams
      WHERE session_key=? AND decay_score<? AND recall_count=0
        AND category NOT IN ('decision','preference')
    `
      )
      .run(sessionKey, threshold).changes;
  }

  // ─── Episodes ────────────────────────────────────────────────────────────

  insertScene(
    sessionKey,
    title,
    summary,
    embedding,
    engramIds,
    avgImportance = 0.5,
    embeddingModel = null
  ) {
    const blob = embedding ? Buffer.from(embedding.buffer) : null;
    return this.db
      .prepare(
        `
      INSERT INTO episodes (session_key, title, summary, embedding, engram_ids, avg_importance, embedding_model)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `
      )
      .run(
        sessionKey,
        title,
        summary,
        blob,
        JSON.stringify(engramIds),
        avgImportance,
        embeddingModel
      ).lastInsertRowid;
  }

  updateScene(sceneId, title, summary, embedding, engramIds, avgImportance, embeddingModel = null) {
    const blob = embedding ? Buffer.from(embedding.buffer) : null;
    this.db
      .prepare(
        `
      UPDATE episodes
      SET title=?, summary=?, embedding=?, engram_ids=?, avg_importance=?, embedding_model=?, updated_at=unixepoch()
      WHERE id=?
    `
      )
      .run(title, summary, blob, JSON.stringify(engramIds), avgImportance, embeddingModel, sceneId);
  }

  getScenes(_sessionKey) {
    return this.db
      .prepare(
        `
      SELECT id, title, summary, embedding, embedding_model, engram_ids, avg_importance, recall_count, updated_at
      FROM episodes ORDER BY updated_at DESC
    `
      )
      .all();
  }

  bumpSceneRecall(id) {
    this.db.prepare('UPDATE episodes SET recall_count=recall_count+1 WHERE id=?').run(id);
  }

  getTurnsByIds(ids) {
    if (!ids.length) return [];
    const ph = ids.map(() => '?').join(',');
    return this.db
      .prepare(`SELECT id, role, content, token_est, category FROM turns WHERE id IN (${ph})`)
      .all(...ids);
  }

  // ─── Foresights ───────────────────────────────────────────────────────────

  insertForesight(
    sessionKey,
    turnId,
    intention,
    target,
    timeframe,
    confidence,
    embedding = null,
    embeddingModel = null
  ) {
    const tf = TIMEFRAMES.includes(timeframe) ? timeframe : 'soon';
    const blob = embedding ? Buffer.from(embedding.buffer) : null;
    return this.db
      .prepare(
        `
      INSERT INTO foresights (session_key, turn_id, intention, target, timeframe, confidence, embedding, embedding_model)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `
      )
      .run(sessionKey, turnId, intention, target || '', tf, confidence, blob, embeddingModel)
      .lastInsertRowid;
  }

  getActiveForesights(sessionKey, limit = 10) {
    return this.db
      .prepare(
        `
      SELECT id, intention, target, timeframe, confidence, created_at, embedding, embedding_model
      FROM foresights
      WHERE session_key=? AND fulfilled=0
      ORDER BY created_at DESC LIMIT ?
    `
      )
      .all(sessionKey, limit);
  }

  markForesightFulfilled(id) {
    this.db.prepare('UPDATE foresights SET fulfilled=1 WHERE id=?').run(id);
  }

  // ─── Shared ───────────────────────────────────────────────────────────────

  /**
   * Decode a stored embedding BLOB into a Float32Array.
   *
   * Important: SQLite returns the BLOB as a Node Buffer, which is itself a
   * Uint8Array view into a (possibly shared) ArrayBuffer. Reading the raw
   * `.buffer` of that view exposes neighbouring bytes when the underlying
   * buffer is pooled — we copy via Float32Array.from(...) of a fresh view
   * sized exactly to the stored bytes so callers get an isolated, safely
   * owned vector.
   */
  /**
   * The last N turn embeddings for a session (newest first) — the
   * selector's short-term topic anchor. Rows whose vectors come from a
   * different embedding model are filtered out (cosine across vector
   * spaces is meaningless).
   */
  getRecentTurnVectors(sessionKey, limit = 6, embeddingModel = null) {
    const rows = this.db
      .prepare(
        `
      SELECT embedding, embedding_model FROM turns
      WHERE session_key=? AND embedding IS NOT NULL
      ORDER BY created_at DESC, id DESC LIMIT ?
    `
      )
      .all(sessionKey, limit);
    if (!embeddingModel) return rows;
    return rows.filter((r) => !r.embedding_model || r.embedding_model === embeddingModel);
  }

  /** Content of the most recent stored user turn (dedup guard in the proxy). */
  lastUserTurnContent(sessionKey) {
    const row = this.db
      .prepare(
        `SELECT content FROM turns WHERE session_key=? AND role='user' ORDER BY id DESC LIMIT 1`
      )
      .get(sessionKey);
    return row ? row.content : null;
  }

  /**
   * Unix-seconds timestamp of the most recent turn in this session (any
   * role), or null if the session has no history yet. Callers (proxy.js)
   * read this BEFORE inserting the new incoming user turn, so it reflects
   * genuine elapsed downtime rather than the turn that just arrived — the
   * selector uses the gap to decide whether to inject a downtime-awareness
   * note (see selector.js _buildDowntimeNote).
   */
  getLastTurnTimestamp(sessionKey) {
    const row = this.db
      .prepare(
        `SELECT created_at FROM turns WHERE session_key=? ORDER BY created_at DESC, id DESC LIMIT 1`
      )
      .get(sessionKey);
    return row ? row.created_at : null;
  }

  /** All session buckets in this store, busiest first. */
  listSessions() {
    return this.db
      .prepare(
        `
      SELECT session_key, COUNT(*) AS turns, MAX(created_at) AS last_at
      FROM turns GROUP BY session_key ORDER BY turns DESC
    `
      )
      .all();
  }

  /**
   * Merge session buckets into one canonical key across every
   * session-scoped table. One character = one continuous relationship;
   * fragmented keys (bearer-hash accidents, per-window ids) make rotating
   * retrieval + foresights blind to most of the character's own history.
   * fromKeys=null merges everything that isn't already intoKey.
   */
  mergeSessions(intoKey, fromKeys = null) {
    const tables = ['turns', 'engrams', 'episodes', 'foresights', 'character_observations'];
    let total = 0;
    this.db.transaction(() => {
      for (const t of tables) {
        if (fromKeys && fromKeys.length) {
          const ph = fromKeys.map(() => '?').join(',');
          total += this.db
            .prepare(`UPDATE ${t} SET session_key=? WHERE session_key IN (${ph})`)
            .run(intoKey, ...fromKeys).changes;
        } else {
          total += this.db
            .prepare(`UPDATE ${t} SET session_key=? WHERE session_key<>?`)
            .run(intoKey, intoKey).changes;
        }
      }
    })();
    return total;
  }

  /**
   * Retire unfulfilled foresights past their timeframe's TTL
   * (fulfilled=2 — distinct from done=1 so nothing is lost, just no longer
   * injected). Stale intentions are a drift vector: the model keeps being
   * told about plans from weeks ago as if they were current.
   */
  expireForesights(ttlDaysByTimeframe = HistoryStore.FORESIGHT_TTL_DAYS) {
    const now = Math.floor(Date.now() / 1000);
    let changed = 0;
    for (const [tf, days] of Object.entries(ttlDaysByTimeframe)) {
      changed += this.db
        .prepare(`UPDATE foresights SET fulfilled=2 WHERE fulfilled=0 AND timeframe=? AND created_at<?`)
        .run(tf, now - days * 86400).changes;
    }
    return changed;
  }

  static toFloat32(blob) {
    if (!blob) return null;
    if (blob.byteLength % 4 !== 0) return null;
    // Slice gives us an independent ArrayBuffer; safe to wrap as Float32.
    const ab = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
    return new Float32Array(ab);
  }

  prune(maxAgeDays) {
    const cutoff = Math.floor(Date.now() / 1000) - maxAgeDays * 86400;
    return this.db.prepare('DELETE FROM turns WHERE created_at<?').run(cutoff).changes;
  }

  stats(sessionKey) {
    const q = (k) =>
      this.db.prepare(`SELECT COUNT(*) as n FROM ${k} WHERE session_key=?`).get(sessionKey).n;
    return {
      turns: q('turns'),
      cells: q('engrams'),
      scenes: q('episodes'),
      foresights: this.db
        .prepare('SELECT COUNT(*) as n FROM foresights WHERE session_key=? AND fulfilled=0')
        .get(sessionKey).n,
    };
  }

  // ─── Character Profile ────────────────────────────────────────────────────

  getCharacterProfile() {
    return (
      this.db.prepare('SELECT * FROM character_profile ORDER BY id DESC LIMIT 1').get() || null
    );
  }

  upsertCharacterProfile({
    sourceType,
    sourcePath,
    sourceMtime,
    rawContent,
    parsedSummary,
    evolutionNotes,
    driftReminder,
    driftCheckedAt,
  }) {
    const existing = this.getCharacterProfile();
    const now = Math.floor(Date.now() / 1000);
    if (existing) {
      this.db
        .prepare(
          `
        UPDATE character_profile
        SET source_type=?, source_path=?, source_mtime=?, raw_content=?, parsed_summary=?,
            evolution_notes=?, drift_reminder=?, drift_checked_at=?, updated_at=?
        WHERE id=?
      `
        )
        .run(
          sourceType,
          sourcePath ?? null,
          sourceMtime ?? null,
          rawContent ?? existing.raw_content,
          parsedSummary ?? existing.parsed_summary,
          evolutionNotes ?? existing.evolution_notes,
          driftReminder ?? existing.drift_reminder,
          driftCheckedAt ?? existing.drift_checked_at,
          now,
          existing.id
        );
    } else {
      this.db
        .prepare(
          `
        INSERT INTO character_profile
          (source_type, source_path, source_mtime, raw_content, parsed_summary,
           evolution_notes, drift_reminder, drift_checked_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
      `
        )
        .run(
          sourceType,
          sourcePath ?? null,
          sourceMtime ?? null,
          rawContent ?? '',
          parsedSummary ?? '{}',
          evolutionNotes ?? '',
          driftReminder ?? '',
          driftCheckedAt ?? 0,
          now
        );
    }
  }

  insertCharacterObservation(sessionKey, turnId, obsType, detail) {
    return this.db
      .prepare(
        `
      INSERT INTO character_observations (session_key, turn_id, obs_type, detail)
      VALUES (?, ?, ?, ?)
    `
      )
      .run(sessionKey, turnId ?? null, obsType, detail).lastInsertRowid;
  }

  getPendingObservations(limit = 50) {
    return this.db
      .prepare(
        `
      SELECT * FROM character_observations
      WHERE consolidated=0 ORDER BY observed_at ASC LIMIT ?
    `
      )
      .all(limit);
  }

  countPendingObservations() {
    return this.db
      .prepare('SELECT COUNT(*) as n FROM character_observations WHERE consolidated=0')
      .get().n;
  }

  markObservationsConsolidated(ids) {
    if (!ids.length) return;
    const ph = ids.map(() => '?').join(',');
    this.db
      .prepare(`UPDATE character_observations SET consolidated=1 WHERE id IN (${ph})`)
      .run(...ids);
  }

  close() {
    this.db.close();
  }
}

module.exports = HistoryStore;
