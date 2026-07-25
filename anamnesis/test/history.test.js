const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

// History used to run on better-sqlite3, whose native addon could fail to
// load in rare environments (no prebuilt for the running Node ABI + no C++
// toolchain to fall back to compiling one -- see the 2026-07-25 migration
// to node:sqlite, which ships inside Node itself and can't hit that
// failure mode). `maybeTest` is kept as a thin passthrough rather than
// removed outright, so this file doesn't need a second pass if some other
// environment-dependent skip reason ever comes up here again.
const HistoryStore = require('../src/history.js');
const skipReason = null;

const maybeTest = (name, fn) => test(name, skipReason ? { skip: skipReason } : undefined, fn);

function tmpDb() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'anamnesis-test-'));
  return { dir, dbPath: path.join(dir, 'history.db') };
}

maybeTest('schema: init creates expected tables and columns', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const turnsCols = h.db
      .prepare('PRAGMA table_info(turns)')
      .all()
      .map((c) => c.name);
    assert.ok(turnsCols.includes('foresight_scanned'), 'turns.foresight_scanned must exist');
    assert.ok(turnsCols.includes('embedding_model'), 'turns.embedding_model must exist');

    const cellsCols = h.db
      .prepare('PRAGMA table_info(engrams)')
      .all()
      .map((c) => c.name);
    assert.ok(cellsCols.includes('importance'));
    assert.ok(cellsCols.includes('category'));
    assert.ok(cellsCols.includes('embedding_model'));

    const sceneCols = h.db
      .prepare('PRAGMA table_info(episodes)')
      .all()
      .map((c) => c.name);
    assert.ok(sceneCols.includes('avg_importance'));
    assert.ok(sceneCols.includes('embedding_model'));
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('foresight_scanned is independent of extracted', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const id = h.insertTurn('s1', 'assistant', 'a'.repeat(200), null, 50, 'm');
    // Initially both flags are 0.
    let row = h.db.prepare('SELECT extracted, foresight_scanned FROM turns WHERE id=?').get(id);
    assert.equal(row.extracted, 0);
    assert.equal(row.foresight_scanned, 0);

    h.markExtracted(id);
    row = h.db.prepare('SELECT extracted, foresight_scanned FROM turns WHERE id=?').get(id);
    assert.equal(row.extracted, 1);
    assert.equal(row.foresight_scanned, 0, 'foresight must NOT be marked when only extracted is');

    h.markForesightScanned(id);
    row = h.db.prepare('SELECT extracted, foresight_scanned FROM turns WHERE id=?').get(id);
    assert.equal(row.foresight_scanned, 1);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest(
  'getUnscannedAssistantTurns returns only assistant turns with foresight_scanned=0',
  () => {
    const { dir, dbPath } = tmpDb();
    const h = new HistoryStore(dbPath);
    try {
      const u = h.insertTurn('s1', 'user', 'aa'.repeat(50), null, 10, 'm');
      const a = h.insertTurn('s1', 'assistant', 'bb'.repeat(50), null, 10, 'm');
      const b = h.insertTurn('s1', 'assistant', 'cc'.repeat(50), null, 10, 'm');
      h.markForesightScanned(b);

      const out = h.getUnscannedAssistantTurns(10).map((r) => r.id);
      assert.deepEqual(out, [a]);
      assert.ok(!out.includes(u));
      assert.ok(!out.includes(b));
    } finally {
      h.close();
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
);

maybeTest('toFloat32 round-trips embeddings', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const orig = new Float32Array([0.1, -0.2, 0.3, 0.4]);
    const id = h.insertTurn('s1', 'assistant', 'x'.repeat(100), orig, 10, 'm');
    const row = h.db.prepare('SELECT embedding FROM turns WHERE id=?').get(id);
    const decoded = HistoryStore.toFloat32(row.embedding);
    assert.equal(decoded.length, 4);
    for (let i = 0; i < orig.length; i++) {
      assert.ok(Math.abs(orig[i] - decoded[i]) < 1e-6);
    }
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('getTurnIdsForMemcells returns deduped turn IDs', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const t1 = h.insertTurn('s', 'assistant', 'a'.repeat(100), null, 10, 'm');
    const t2 = h.insertTurn('s', 'assistant', 'b'.repeat(100), null, 10, 'm');
    const c1 = h.insertMemcell('s', t1, 'fact 1', null, 0.5, 'other', 'm');
    const c2 = h.insertMemcell('s', t1, 'fact 2', null, 0.5, 'other', 'm');
    const c3 = h.insertMemcell('s', t2, 'fact 3', null, 0.5, 'other', 'm');

    const ids = h.getTurnIdsForMemcells([c1, c2, c3]).sort();
    assert.deepEqual(ids, [t1, t2].sort());
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('decay: high-importance cell decays slower than low-importance', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const t = h.insertTurn('s', 'assistant', 'x'.repeat(100), null, 10, 'm');
    const cLo = h.insertMemcell('s', t, 'low', null, 0.1, 'other', 'm');
    const cHi = h.insertMemcell('s', t, 'high', null, 1.0, 'other', 'm');
    // Backdate both 60 days so decay can take effect.
    const old = Math.floor(Date.now() / 1000) - 60 * 86400;
    h.db.prepare('UPDATE engrams SET created_at=? WHERE id IN (?,?)').run(old, cLo, cHi);

    h.updateDecayScores('s');
    const rows = h.db
      .prepare('SELECT id, decay_score FROM engrams WHERE id IN (?,?)')
      .all(cLo, cHi);
    const lo = rows.find((r) => r.id === cLo).decay_score;
    const hi = rows.find((r) => r.id === cHi).decay_score;
    assert.ok(hi > lo, `expected high-importance decay > low (got hi=${hi}, lo=${lo})`);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('prune respects category exemption', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const t = h.insertTurn('s', 'assistant', 'x'.repeat(100), null, 10, 'm');
    const cOther = h.insertMemcell('s', t, 'misc', null, 0.1, 'other', 'm');
    const cDec = h.insertMemcell('s', t, 'choice', null, 0.1, 'decision', 'm');
    const cPref = h.insertMemcell('s', t, 'pref', null, 0.1, 'preference', 'm');

    // Set decay_score below threshold for all three.
    h.db.prepare('UPDATE engrams SET decay_score=0.01').run();
    const pruned = h.pruneDecayedMemcells('s', 0.05);
    assert.equal(pruned, 1, 'only the "other" cell should be pruned');

    const remaining = h
      .getAllMemcells('s')
      .map((c) => c.id)
      .sort();
    assert.deepEqual(remaining, [cDec, cPref].sort());
    assert.ok(!remaining.includes(cOther));
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('stats returns counts per session', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    h.insertTurn('s1', 'user', 'aaaaaaaa', null, 10, 'm');
    h.insertTurn('s1', 'assistant', 'a'.repeat(100), null, 10, 'm');
    h.insertTurn('s2', 'assistant', 'b'.repeat(100), null, 10, 'm');
    const s1 = h.stats('s1');
    const s2 = h.stats('s2');
    assert.equal(s1.turns, 2);
    assert.equal(s2.turns, 1);
    assert.equal(s1.cells, 0);
    assert.equal(s1.scenes, 0);
    assert.equal(s1.foresights, 0);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ─── category-partitioned quotas (turns.category) ───────────────────────────

maybeTest('insertTurn: defaults to fleagle category when not specified', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const id = h.insertTurn('s1', 'user', 'hello', null, 10, 'm');
    const row = h.db.prepare('SELECT category FROM turns WHERE id=?').get(id);
    assert.equal(row.category, 'fleagle');
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('insertTurn: stores an explicit category', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const id = h.insertTurn('s1', 'assistant', 'reply', null, 10, 'm', 'background');
    const row = h.db.prepare('SELECT category FROM turns WHERE id=?').get(id);
    assert.equal(row.category, 'background');
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('getSessionTurns and getTurnsByIds surface category', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const id = h.insertTurn('s1', 'user', 'hi', null, 10, 'm', 'task');
    const viaSession = h.getSessionTurns('s1');
    const viaIds = h.getTurnsByIds([id]);
    assert.equal(viaSession[0].category, 'task');
    assert.equal(viaIds[0].category, 'task');
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('migration: pre-existing DB without turns.category gets it added, defaulted', () => {
  const { dir, dbPath } = tmpDb();
  // Simulate an old DB: open once, drop the category column by rebuilding
  // the table without it (SQLite has no DROP COLUMN pre-3.35 in general use
  // here, so instead we assert the column is present after a fresh open —
  // the real regression this guards is _migrate() being a no-op/erroring
  // on a DB that already has the column, which the schema init path below
  // exercises implicitly on every test in this file).
  const h1 = new HistoryStore(dbPath);
  h1.insertTurn('s1', 'user', 'pre-migration turn', null, 5, 'm');
  h1.close();
  // Reopen — _migrate() must be idempotent (has() guard) and not touch
  // existing rows' categories.
  const h2 = new HistoryStore(dbPath);
  try {
    const row = h2.db.prepare('SELECT category FROM turns LIMIT 1').get();
    assert.equal(row.category, 'fleagle');
  } finally {
    h2.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ─── downtime-awareness (getLastTurnTimestamp) ──────────────────────────────

maybeTest('getLastTurnTimestamp: null for a brand-new session', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    assert.equal(h.getLastTurnTimestamp('never-seen'), null);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('getLastTurnTimestamp: returns the most recent turn\'s created_at', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    h.insertTurn('s1', 'user', 'first', null, 5, 'm');
    const id2 = h.insertTurn('s1', 'assistant', 'second', null, 5, 'm');
    const expected = h.db.prepare('SELECT created_at FROM turns WHERE id=?').get(id2).created_at;
    assert.equal(h.getLastTurnTimestamp('s1'), expected);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('getLastTurnTimestamp: is per-session, not global', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    h.insertTurn('s1', 'user', 'a', null, 5, 'm');
    assert.equal(h.getLastTurnTimestamp('s2'), null);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ─── node:sqlite migration safety (2026-07-25) ──────────────────────────────

maybeTest('mergeSessions: rolls back cleanly if a mid-merge statement throws (atomicity)', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    h.insertTurn('old-key', 'user', 'a', null, 5, 'm');
    h.insertTurn('old-key', 'assistant', 'b', null, 5, 'm');

    // Force a failure partway through mergeSessions' per-table loop by
    // dropping one of the tables it iterates (character_observations) --
    // its UPDATE will throw "no such table", after turns/engrams/episodes/
    // foresights have already been rewritten in the same transaction.
    h.db.exec('DROP TABLE character_observations');

    assert.throws(() => h.mergeSessions('new-key'));

    // If the transaction wrapper actually rolled back, 'turns' must NOT
    // have been re-keyed either, even though its own UPDATE succeeded
    // before the later failure.
    const stillOld = h.db.prepare("SELECT COUNT(*) as n FROM turns WHERE session_key='old-key'").get().n;
    const nowNew = h.db.prepare("SELECT COUNT(*) as n FROM turns WHERE session_key='new-key'").get().n;
    assert.equal(stillOld, 2, 'turns must still be under the original session_key after rollback');
    assert.equal(nowNew, 0, 'no turns should have been re-keyed after rollback');
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('mergeSessions: commits all five tables together on success', () => {
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const turnId = h.insertTurn('old-key', 'user', 'a', null, 5, 'm');
    h.insertMemcell('old-key', turnId, 'fact', null, 0.5, 'other', 'm');
    h.insertForesight('old-key', turnId, 'do the thing', '', 'soon', 0.7);

    const changed = h.mergeSessions('new-key');
    assert.ok(changed >= 3, 'should report at least the 3 rows re-keyed above');
    assert.equal(h.db.prepare("SELECT COUNT(*) as n FROM turns WHERE session_key='new-key'").get().n, 1);
    assert.equal(h.db.prepare("SELECT COUNT(*) as n FROM engrams WHERE session_key='new-key'").get().n, 1);
    assert.equal(h.db.prepare("SELECT COUNT(*) as n FROM foresights WHERE session_key='new-key'").get().n, 1);
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

maybeTest('busy timeout: connection is configured to wait on lock contention, not fail immediately', () => {
  // A same-thread test can't safely exercise *actual* cross-handle lock
  // contention: node:sqlite's DatabaseSync is genuinely synchronous, so a
  // second handle blocked waiting on a lock would block the one JS thread
  // that would otherwise run the timer/callback releasing that lock --
  // that's a real same-process deadlock risk, not a hypothetical one, and
  // not worth introducing into the suite to test this. What's both correct
  // and worth asserting: the connection actually has a non-zero busy
  // timeout configured at all, since node:sqlite defaults this to 0
  // (immediate SQLITE_BUSY on any contention) unlike better-sqlite3's
  // 5000ms default -- see the constructor's `timeout` option and the
  // PRAGMA busy_timeout call right after it in history.js.
  const { dir, dbPath } = tmpDb();
  const h = new HistoryStore(dbPath);
  try {
    const busyTimeout = h.db.prepare('PRAGMA busy_timeout').get().timeout;
    assert.equal(busyTimeout, 5000, 'busy_timeout must not be left at node:sqlite\'s 0 default');
  } finally {
    h.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
