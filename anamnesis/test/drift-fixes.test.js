'use strict';
/**
 * drift-fixes.test.js — regression tests for the topic-drift / hallucination
 * fix set: topic-anchored queries, relevance floors, foresight TTL +
 * relevance gating + auto-fulfillment, extraction grounding, centroid
 * clustering, scene merging, session merging, and user-turn dedup support.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const os = require('node:os');
const path = require('node:path');
const fs = require('node:fs');

const HistoryStore = require('../src/history.js');
const Selector = require('../src/selector.js');

const MODEL = 'test-model';

function buf(arr) {
  return Buffer.from(new Float32Array(arr).buffer);
}
function vec(arr) {
  return new Float32Array(arr);
}
function tmpDb(tag) {
  return path.join(os.tmpdir(), `anamnesis-drift-${tag}-${process.pid}-${Date.now()}.db`);
}
function cleanup(p) {
  for (const f of [p, p + '-wal', p + '-shm']) {
    try {
      fs.unlinkSync(f);
    } catch {
      /* fine */
    }
  }
}

function makeEmbedder(map = {}, fallback = [0.1, 0.2]) {
  return {
    model: MODEL,
    embed: async (text) => {
      for (const [prefix, v] of Object.entries(map)) {
        if (text.startsWith(prefix)) return vec(v);
      }
      return vec(fallback);
    },
  };
}

function makeConfig(ctxOverrides = {}) {
  return {
    context: {
      tokenBudget: 50000,
      systemReserveTokens: 500,
      recencyTurns: 8,
      rotatingSlots: 6,
      charsPerToken: 4,
      minChunkChars: 40,
      ...ctxOverrides,
    },
  };
}

function makeMockHistory(overrides = {}) {
  return {
    getScenes: () => [],
    getActiveForesights: () => [],
    getSessionTurns: () => [],
    getRecentTurnVectors: () => [],
    getTurnIdsForMemcells: () => [],
    getTurnsByIds: () => [],
    stats: () => ({ turns: 0, cells: 0, scenes: 0, foresights: 0 }),
    bumpTurnRecall: () => {},
    bumpSceneRecall: () => {},
    ...overrides,
  };
}

// ── HistoryStore: new primitives ────────────────────────────────────────────

test('mergeSessions re-keys every session-scoped table into one bucket', () => {
  const p = tmpDb('merge');
  const h = new HistoryStore(p);
  h.insertTurn('default', 'user', 'hello from default', null, 4);
  h.insertTurn('auth:abcd', 'assistant', 'hi from auth bucket', null, 4);
  const tid = h.insertTurn('oc:ses_x', 'assistant', 'hi from oc bucket', null, 4);
  h.insertMemcell('auth:abcd', tid, 'a fact', null, 0.5, 'other', MODEL);
  h.insertForesight('default', tid, 'do the thing', '', 'soon', 0.9);

  const changed = h.mergeSessions('oc:pleiades:Mark');
  assert.ok(changed >= 5, `expected >=5 re-keyed rows, got ${changed}`);
  const sessions = h.listSessions();
  assert.equal(sessions.length, 1);
  assert.equal(sessions[0].session_key, 'oc:pleiades:Mark');
  assert.equal(h.stats('oc:pleiades:Mark').turns, 3);
  h.close();
  cleanup(p);
});

test('expireForesights retires stale-by-timeframe, keeps fresh + fulfilled semantics distinct', () => {
  const p = tmpDb('expire');
  const h = new HistoryStore(p);
  const t = h.insertTurn('s', 'assistant', 'planning turn', null, 4);
  const oldId = h.insertForesight('s', t, 'stale soon plan', '', 'soon', 0.9);
  const freshId = h.insertForesight('s', t, 'fresh soon plan', '', 'soon', 0.9);
  const monthsId = h.insertForesight('s', t, 'long horizon plan', '', 'months', 0.9);
  // Age the first one 5 days (soon TTL = 3d) and the months one 10 days (TTL 90d).
  const now = Math.floor(Date.now() / 1000);
  h.db.prepare('UPDATE foresights SET created_at=? WHERE id=?').run(now - 5 * 86400, oldId);
  h.db.prepare('UPDATE foresights SET created_at=? WHERE id=?').run(now - 10 * 86400, monthsId);

  const changed = h.expireForesights();
  assert.equal(changed, 1);
  const active = h.getActiveForesights('s', 10).map((f) => f.id);
  assert.ok(!active.includes(oldId), 'stale soon foresight still active');
  assert.ok(active.includes(freshId), 'fresh foresight was wrongly expired');
  assert.ok(active.includes(monthsId), 'months foresight expired way before its TTL');
  h.close();
  cleanup(p);
});

test('getRecentTurnVectors filters other embedding models; lastUserTurnContent sees only user turns', () => {
  const p = tmpDb('recent');
  const h = new HistoryStore(p);
  h.insertTurn('s', 'user', 'first user msg', vec([1, 0]), 4, MODEL);
  h.insertTurn('s', 'assistant', 'assistant msg', vec([0, 1]), 4, MODEL);
  h.insertTurn('s', 'user', 'legacy model msg', vec([1, 1]), 4, 'other-model');
  const rows = h.getRecentTurnVectors('s', 10, MODEL);
  assert.equal(rows.length, 2);
  assert.equal(h.lastUserTurnContent('s'), 'legacy model msg');
  assert.equal(h.lastUserTurnContent('nope'), null);
  h.close();
  cleanup(p);
});

// ── Selector: topic anchor + floors + foresight gating ─────────────────────

test('short deictic queries lean on the topic anchor instead of their own noise vector', async () => {
  const embedder = makeEmbedder({ ok: [1, 0] });
  // Recent conversation is all about topic B ([0,1]).
  const history = makeMockHistory({
    getRecentTurnVectors: () => [{ embedding: buf([0, 1]), embedding_model: MODEL }],
  });
  const selector = new Selector(makeConfig(), history, embedder, null);

  const shortQ = await selector._buildQueryVec('s', 'ok', MODEL);
  const longQ = await selector._buildQueryVec(
    's',
    'ok'.padEnd(80, ' — a much longer, self-contained question about topic A'),
    MODEL
  );
  // Short query → anchor dominates (y component > x); long query → current
  // message dominates (x > y).
  assert.ok(shortQ[1] > shortQ[0], `short query should lean anchor, got [${shortQ}]`);
  assert.ok(longQ[0] > longQ[1], `long query should lean current text, got [${longQ}]`);
});

test('nothing is injected when no scene clears the relevance floor', async () => {
  const embedder = makeEmbedder({}, [1, 0]); // query embeds to [1,0]
  const now = Math.floor(Date.now() / 1000);
  const history = makeMockHistory({
    getScenes: () => [
      {
        id: 1,
        title: 'Unrelated scene',
        summary: 'totally different topic',
        embedding: buf([0, 1]), // orthogonal → sim 0
        embedding_model: MODEL,
        engram_ids: '[]',
        avg_importance: 0.9,
        recall_count: 0,
        updated_at: now,
      },
    ],
  });
  const selector = new Selector(makeConfig(), history, embedder, null);
  const out = await selector.select('s', [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'a completely new topic never discussed before, quite long indeed' },
  ]);
  const sys = out.filter((m) => m.role === 'system');
  assert.equal(sys.length, 1);
  assert.ok(!sys[0].content.includes('<memory>'), 'irrelevant scene was injected');
  // No rotating turns either: only system + the user message.
  assert.equal(out.length, 2);
});

test('relevant scenes DO inject, with date stamps and authoritative-conversation framing', async () => {
  const embedder = makeEmbedder({}, [1, 0]);
  const now = Math.floor(Date.now() / 1000);
  const history = makeMockHistory({
    getScenes: () => [
      {
        id: 1,
        title: 'Pleiades work',
        summary: 'built the autofit planner',
        embedding: buf([0.95, 0.31]),
        embedding_model: MODEL,
        engram_ids: '[]',
        avg_importance: 0.7,
        recall_count: 0,
        updated_at: now - 86400,
      },
    ],
  });
  const selector = new Selector(makeConfig(), history, embedder, null);
  const out = await selector.select('s', [
    { role: 'user', content: 'tell me about the pleiades autofit work we did, in detail please' },
  ]);
  const sys = out.find((m) => m.role === 'system');
  assert.ok(sys, 'no system message injected');
  assert.ok(sys.content.includes('<memory>'));
  assert.ok(sys.content.includes('[Pleiades work] built the autofit planner'));
  assert.match(sys.content, /• \[\d{4}-\d{2}-\d{2}\] \[Pleiades work\]/, 'memory line not dated');
  assert.ok(sys.content.includes('CURRENT conversation is always'), 'missing authority framing');
});

test('foresights are TTL-expired and relevance-gated at injection time', async () => {
  const embedder = makeEmbedder({}, [1, 0]);
  const now = Math.floor(Date.now() / 1000);
  const history = makeMockHistory({
    getActiveForesights: () => [
      // stale: 'soon' from 5 days ago
      {
        id: 1,
        intention: 'stale plan',
        target: '',
        timeframe: 'soon',
        confidence: 0.9,
        created_at: now - 5 * 86400,
        embedding: buf([1, 0]),
        embedding_model: MODEL,
      },
      // irrelevant: fresh but orthogonal to the query
      {
        id: 2,
        intention: 'irrelevant plan',
        target: '',
        timeframe: 'soon',
        confidence: 0.9,
        created_at: now - 3600,
        embedding: buf([0, 1]),
        embedding_model: MODEL,
      },
      // good: fresh and aligned
      {
        id: 3,
        intention: 'relevant fresh plan',
        target: 'pleiades',
        timeframe: 'soon',
        confidence: 0.9,
        created_at: now - 3600,
        embedding: buf([0.9, 0.44]),
        embedding_model: MODEL,
      },
    ],
  });
  const selector = new Selector(makeConfig(), history, embedder, null);
  const out = await selector.select('s', [
    { role: 'user', content: 'ready to continue the pleiades work from where we left off yesterday?' },
  ]);
  const sys = out.find((m) => m.role === 'system');
  assert.ok(sys, 'no system message');
  assert.ok(sys.content.includes('relevant fresh plan'));
  assert.ok(!sys.content.includes('stale plan'), 'TTL-expired foresight injected');
  assert.ok(!sys.content.includes('irrelevant plan'), 'irrelevant foresight injected');
  assert.ok(
    sys.content.includes('do not steer the conversation back'),
    'missing anti-drift foresight framing'
  );
});

// ── Extractor: grounding check ──────────────────────────────────────────────

async function withMockBrain(chatFn, fn) {
  const brainPath = require.resolve('../src/lib/brain.js');
  const extractorPath = require.resolve('../src/extractor.js');
  const savedBrain = require.cache[brainPath];
  const savedExtractor = require.cache[extractorPath];
  require.cache[brainPath] = {
    id: brainPath,
    filename: brainPath,
    loaded: true,
    exports: {
      chat: chatFn,
      tryParseJsonArray: (text) => {
        const m = text.match(/\[[\s\S]*?\]/);
        if (!m) return null;
        try {
          return JSON.parse(m[0]);
        } catch {
          return null;
        }
      },
      embeddingModel: () => MODEL,
      embed: async () => null,
    },
  };
  delete require.cache[extractorPath];
  try {
    return await fn(require('../src/extractor.js'));
  } finally {
    if (savedBrain) require.cache[brainPath] = savedBrain;
    else delete require.cache[brainPath];
    if (savedExtractor) require.cache[extractorPath] = savedExtractor;
    else delete require.cache[extractorPath];
  }
}

test('extractor drops facts that fail the grounding check against the source turn', async () => {
  const cells = [];
  const history = {
    markExtracted: () => {},
    insertMemcell: (...args) => cells.push(args),
  };
  const embedder = makeEmbedder(
    { Grounded: [0.95, 0.31], Hallucinated: [0, 1] },
    [0.1, 0.1]
  );
  await withMockBrain(
    async () =>
      JSON.stringify([
        { fact: 'Grounded: Redis was chosen for the caching layer', importance: 0.8, category: 'decision' },
        { fact: 'Hallucinated: the user plans to move to Mars', importance: 0.9, category: 'personal' },
      ]),
    async (Extractor) => {
      const e = new Extractor(
        { extraction: { maxRetries: 0, timeoutMs: 5000, startupBacklogLimit: 10 } },
        history,
        embedder
      );
      await e._extractTurn({
        id: 1,
        session_key: 's',
        role: 'assistant',
        content:
          'After comparing the options in depth we went with Redis for the caching layer.\n' +
          'The decision came down to latency: Redis kept p99 under two milliseconds in our tests.',
        embedding: buf([1, 0]),
        embedding_model: MODEL,
      });
    }
  );
  assert.equal(cells.length, 1, `expected 1 grounded cell, got ${cells.length}`);
  assert.ok(String(cells[0][2]).startsWith('Grounded:'));
});

// ── Consolidator: centroid clustering + merge ───────────────────────────────

async function withMockBrainConsolidator(chatFn, fn) {
  const brainPath = require.resolve('../src/lib/brain.js');
  const consPath = require.resolve('../src/consolidator.js');
  const savedBrain = require.cache[brainPath];
  const savedCons = require.cache[consPath];
  require.cache[brainPath] = {
    id: brainPath,
    filename: brainPath,
    loaded: true,
    exports: {
      chat: chatFn,
      generate: (prompt, opts) => chatFn([{ role: 'user', content: prompt }], opts),
      tryParseJsonObject: (text) => {
        const m = text.match(/\{[\s\S]*\}/);
        if (!m) return null;
        try {
          return JSON.parse(m[0]);
        } catch {
          return null;
        }
      },
      embeddingModel: () => MODEL,
      embed: async () => null,
    },
  };
  delete require.cache[consPath];
  try {
    return await fn(require('../src/consolidator.js'));
  } finally {
    if (savedBrain) require.cache[brainPath] = savedBrain;
    else delete require.cache[brainPath];
    if (savedCons) require.cache[consPath] = savedCons;
    else delete require.cache[consPath];
  }
}

function consolidatorConfig() {
  return {
    memory: {
      consolidationIntervalMs: 10000,
      consolidationBatchSize: 200,
      sceneClusterThreshold: 0.72,
      minSceneSize: 2,
      decayPruneThreshold: 0.05,
      minEngramAgeSec: 0, // no age gate in tests
    },
  };
}

test('centroid clustering rejects chained topics that star-to-seed would absorb', async () => {
  const p = tmpDb('cluster');
  const h = new HistoryStore(p);
  const t = h.insertTurn('s', 'assistant', 'some turn', null, 4);
  // A=[1,0]; B sim(A)=0.73 (joins, pulls centroid up); C sim(A)=0.73 but
  // sim(centroid(A,B)) ≈ 0.43 → must NOT join under centroid clustering.
  h.insertMemcell('s', t, 'fact A base topic vector', vec([1, 0]), 0.5, 'other', MODEL);
  h.insertMemcell('s', t, 'fact B near A pulls centroid', vec([0.73, 0.683]), 0.5, 'other', MODEL);
  h.insertMemcell('s', t, 'fact C chained off A other way', vec([0.73, -0.683]), 0.5, 'other', MODEL);

  const embedder = Object.assign(makeEmbedder({}, [0.9, 0.2]), {});
  // consolidator uses this.embedder.constructor.cosine — give the mock a
  // constructor with a static cosine like the real Embedder class.
  function FakeEmbedder() {}
  FakeEmbedder.cosine = (a, b) => {
    let dot = 0,
      na = 0,
      nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const d = Math.sqrt(na) * Math.sqrt(nb);
    return d === 0 ? 0 : dot / d;
  };
  Object.setPrototypeOf(embedder, FakeEmbedder.prototype);

  await withMockBrainConsolidator(
    async () => '{"title":"Scene title","summary":"fact A base topic vector and B"}',
    async (Consolidator) => {
      const c = new Consolidator(consolidatorConfig(), h, embedder);
      await c._consolidateSession('s');
    }
  );

  const scenes = h.getScenes('s');
  assert.equal(scenes.length, 1, `expected 1 scene (A+B), got ${scenes.length}`);
  const ids = JSON.parse(scenes[0].engram_ids);
  assert.equal(ids.length, 2, `chained engram C leaked into the scene (got ${ids.length} members)`);
  h.close();
  cleanup(p);
});

test('a new cluster on an existing topic extends the existing scene instead of fragmenting', async () => {
  const p = tmpDb('mergescene');
  const h = new HistoryStore(p);
  const t = h.insertTurn('s', 'assistant', 'turn', null, 4);
  // Existing scene about topic [1,0]
  const c1 = h.insertMemcell('s', t, 'old fact one', vec([1, 0]), 0.5, 'other', MODEL);
  const sceneId = h.insertScene('s', 'Old scene', 'old summary', vec([1, 0]), [c1], 0.5, MODEL);
  h.assignMemcellToScene(c1, sceneId);
  // Two new unclustered engrams on the same topic
  h.insertMemcell('s', t, 'new fact one same topic', vec([0.99, 0.14]), 0.5, 'other', MODEL);
  h.insertMemcell('s', t, 'new fact two same topic', vec([0.98, 0.19]), 0.5, 'other', MODEL);

  const embedder = makeEmbedder({}, [1, 0.05]);
  function FakeEmbedder() {}
  FakeEmbedder.cosine = (a, b) => {
    let dot = 0,
      na = 0,
      nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const d = Math.sqrt(na) * Math.sqrt(nb);
    return d === 0 ? 0 : dot / d;
  };
  Object.setPrototypeOf(embedder, FakeEmbedder.prototype);

  await withMockBrainConsolidator(
    async () => '{"title":"Same topic","summary":"new fact one same topic; new fact two"}',
    async (Consolidator) => {
      const c = new Consolidator(consolidatorConfig(), h, embedder);
      await c._consolidateSession('s');
    }
  );

  const scenes = h.getScenes('s');
  assert.equal(scenes.length, 1, `scene fragmented: ${scenes.length} scenes for one topic`);
  const ids = JSON.parse(scenes[0].engram_ids);
  assert.equal(ids.length, 3, `merged scene should own all 3 engrams, has ${ids.length}`);
  h.close();
  cleanup(p);
});
