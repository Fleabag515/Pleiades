'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');

const Selector = require('../src/selector.js');

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeMockHistory() {
  return {
    getScenes: () => [],
    getActiveForesights: () => [],
    getSessionTurns: () => [],
    stats: () => ({ turns: 0, cells: 0, scenes: 0, foresights: 0 }),
    bumpTurnRecall: () => {},
    bumpSceneRecall: () => {},
  };
}

function makeMockEmbedder() {
  return {
    model: 'test-model',
    embed: async () => new Float32Array([0.1, 0.2]),
  };
}

function makeConfig() {
  return {
    context: {
      tokenBudget: 50000,
      systemReserveTokens: 500,
      recencyTurns: 8, // recencyWindow = 16 messages
      rotatingSlots: 6,
      charsPerToken: 4,
      minChunkChars: 40,
    },
  };
}

// Simulate Pleiades's engine.py tool-calling loop for a SINGLE user turn:
// one user message, then many (assistant tool_call, tool result) round-trips,
// all within the one ever-growing `messages` array for that one turn (this is
// exactly what _loop()/stream_events() in pleiades/engine.py send on every
// round -- the full array, original user message included).
function buildLongToolLoopTurn(rounds) {
  const messages = [
    { role: 'system', content: 'You are Mark.' },
    { role: 'user', content: 'go log into instagram for me' },
  ];
  for (let i = 0; i < rounds; i++) {
    messages.push({ role: 'assistant', content: `calling tool round ${i}` });
    messages.push({ role: 'tool', content: `tool result ${i}` });
  }
  messages.push({ role: 'assistant', content: 'final reply' });
  return messages;
}

test('select() must not drop the turn-anchoring user message when the incoming tool-call loop exceeds the recency window', async () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);

  // recencyWindow = recencyTurns(8) * 2 = 16 non-system messages.
  // 9 tool rounds -> 1 user + 18 assistant/tool messages = 19 convo messages,
  // well past the 16-message window.
  const incoming = buildLongToolLoopTurn(9);
  const result = await selector.select('session-1', incoming);

  const hasUser = result.some((m) => m.role === 'user');
  assert.ok(
    hasUser,
    'selected context sent upstream has NO user-role message at all -- ' +
      'this is exactly what makes the Qwen3 chat template raise ' +
      `"No user query found in messages." Got roles: ${result.map((m) => m.role).join(',')}`
  );
});

test('select() still includes the user message for a normal short turn (no regression)', async () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);
  const incoming = buildLongToolLoopTurn(1); // well under the window
  const result = await selector.select('session-1', incoming);
  assert.ok(result.some((m) => m.role === 'user'));
});

test('select() anchors the retrieval query to the last REAL user message, not a tool result, across every round of one agentic turn', async () => {
  const embeddedTexts = [];
  const spyEmbedder = {
    model: 'test-model',
    embed: async (text) => {
      embeddedTexts.push(text);
      return new Float32Array([0.1, 0.2]);
    },
  };
  const selector = new Selector(makeConfig(), makeMockHistory(), spyEmbedder, null);

  // Simulate Pleiades' engine.py resending the growing messages array once
  // per tool round -- same request each time except more (assistant, tool)
  // pairs appended at the tail. The literal last message is a tool result
  // in every round after the first.
  for (let round = 0; round < 4; round++) {
    const incoming = buildLongToolLoopTurn(round);
    await selector.select('session-2', incoming);
  }

  // Every round's embed() call for the query vector must have used the
  // ORIGINAL user message text, never a "tool result N" string -- otherwise
  // the <memory>/rotating-slot selection recomputes differently each round
  // (defeating KV-cache reuse) and retrieval drifts onto irrelevant tool
  // output instead of the user's actual request.
  assert.ok(embeddedTexts.length > 0, 'expected at least one embed() call');
  for (const t of embeddedTexts) {
    assert.equal(
      t,
      'go log into instagram for me',
      `expected every round to embed the original user message, got: ${JSON.stringify(t)}`
    );
  }
});

// ─── downtime-awareness ──────────────────────────────────────────────────────

function makeConfigWithDowntime(overrides = {}) {
  const cfg = makeConfig();
  cfg.context.downtimeAwareness = { enabled: true, minGapMinutes: 30, ...overrides };
  return cfg;
}

test('_buildDowntimeNote: null when there is no prior activity (brand-new session)', () => {
  const selector = new Selector(makeConfigWithDowntime(), makeMockHistory(), makeMockEmbedder(), null);
  assert.equal(selector._buildDowntimeNote(null), null);
  assert.equal(selector._buildDowntimeNote(undefined), null);
});

test('_buildDowntimeNote: null when the gap is under minGapMinutes', () => {
  const selector = new Selector(makeConfigWithDowntime({ minGapMinutes: 30 }), makeMockHistory(), makeMockEmbedder(), null);
  const fiveMinAgo = Date.now() / 1000 - 5 * 60;
  assert.equal(selector._buildDowntimeNote(fiveMinAgo), null);
});

test('_buildDowntimeNote: returns a formatted note when the gap exceeds minGapMinutes', () => {
  const selector = new Selector(makeConfigWithDowntime({ minGapMinutes: 30 }), makeMockHistory(), makeMockEmbedder(), null);
  const fourteenHoursAgo = Date.now() / 1000 - (14 * 3600 + 22 * 60);
  const note = selector._buildDowntimeNote(fourteenHoursAgo);
  assert.ok(note, 'expected a continuity note for a 14h+ gap');
  assert.match(note, /14h 22m/);
  assert.match(note, /aware the gap happened/);
});

test('_buildDowntimeNote: disabled via config returns null regardless of gap', () => {
  const selector = new Selector(makeConfigWithDowntime({ enabled: false }), makeMockHistory(), makeMockEmbedder(), null);
  const daysAgo = Date.now() / 1000 - 5 * 86400;
  assert.equal(selector._buildDowntimeNote(daysAgo), null);
});

test('select(): injects a <continuity> system block when lastActivityAt shows a long gap', async () => {
  const selector = new Selector(makeConfigWithDowntime({ minGapMinutes: 30 }), makeMockHistory(), makeMockEmbedder(), null);
  const incoming = [
    { role: 'system', content: 'You are Mark.' },
    { role: 'user', content: 'hey, back' },
  ];
  const twoDaysAgo = Date.now() / 1000 - 2 * 86400;
  const result = await selector.select('session-1', incoming, { lastActivityAt: twoDaysAgo });
  const sys = result.find((m) => m.role === 'system');
  assert.ok(sys, 'expected a system message in the selected context');
  assert.match(sys.content, /<continuity>/);
  assert.match(sys.content, /2d 0h/);
});

test('select(): no <continuity> block for a normal back-and-forth (no lastActivityAt passed)', async () => {
  const selector = new Selector(makeConfigWithDowntime(), makeMockHistory(), makeMockEmbedder(), null);
  const incoming = [
    { role: 'system', content: 'You are Mark.' },
    { role: 'user', content: 'what about the other one' },
  ];
  const result = await selector.select('session-1', incoming);
  const sys = result.find((m) => m.role === 'system');
  // No memory/foresight/downtime and no persona -> system messages pass through untouched.
  assert.equal(sys.content, 'You are Mark.');
});

// ─── category-partitioned quotas (_fillBudget) ───────────────────────────────

function makeItem(id, category, content = 'x'.repeat(40)) {
  return { id, role: 'assistant', content, category, created_at: id };
}

test('_fillBudget: an explicit empty categoryQuotas ({}) opts out entirely -- plain score-ranked fill', () => {
  const cfg = makeConfig();
  cfg.context.categoryQuotas = {}; // explicit opt-out, distinct from "key never set"
  const bumped = [];
  const spyHistory = makeMockHistory();
  spyHistory.bumpTurnRecall = (id) => bumped.push(id);
  const selector = new Selector(cfg, spyHistory, makeMockEmbedder(), null);
  // 10 background items outrank (are listed before) 2 fleagle items; maxSlots=3.
  const ranked = [
    ...Array.from({ length: 10 }, (_, i) => makeItem(i, 'background')),
    makeItem(100, 'fleagle'),
    makeItem(101, 'fleagle'),
  ];
  const out = selector._fillBudget(ranked, 3, 10000, 4, 40);
  assert.equal(out.length, 3);
  // Pure score order with quotas opted out: the first 3 (all background) win.
  assert.deepEqual(bumped.sort((a, b) => a - b), [0, 1, 2]);
});

test('_fillBudget: categoryQuotas key never set at all -> module default (fleagle floor) applies automatically', () => {
  const bumped = [];
  const spyHistory = makeMockHistory();
  spyHistory.bumpTurnRecall = (id) => bumped.push(id);
  const selector2 = new Selector(makeConfig(), spyHistory, makeMockEmbedder(), null);
  const ranked = [
    ...Array.from({ length: 10 }, (_, i) => makeItem(i, 'background')),
    makeItem(100, 'fleagle'),
    makeItem(101, 'fleagle'),
  ];
  selector2._fillBudget(ranked, 3, 10000, 4, 40);
  assert.ok(
    bumped.includes(100) || bumped.includes(101),
    `expected the default fleagle floor to protect at least one fleagle item even with no ` +
      `explicit config, got ids: ${bumped.join(',')}`
  );
});

test('_fillBudget: categoryQuotas guarantees a fleagle floor even under a background flood', () => {
  const cfg = makeConfig();
  cfg.context.categoryQuotas = { fleagle: 0.5 };
  const selector = new Selector(cfg, makeMockHistory(), makeMockEmbedder(), null);
  // 20 background items all ranked ahead of 2 fleagle items -- under the old
  // unconditional fill, maxSlots=3 would be entirely background.
  const ranked = [
    ...Array.from({ length: 20 }, (_, i) => makeItem(i, 'background')),
    makeItem(200, 'fleagle'),
    makeItem(201, 'fleagle'),
  ];
  const out = selector._fillBudget(ranked, 3, 10000, 4, 40);
  assert.equal(out.length, 3);
  // We can't see category on the returned {role, content} shape, so assert
  // indirectly via content: fleagle items are id 200/201 -> content 'x'*40
  // for all items (same content), so instead check floor logic directly by
  // re-deriving which ids got selected through a spy-free approach: rerun
  // fillBudget's bookkeeping via the ids' recall bumps.
  const bumped = [];
  const spyHistory = makeMockHistory();
  spyHistory.bumpTurnRecall = (id) => bumped.push(id);
  const selector2 = new Selector(cfg, spyHistory, makeMockEmbedder(), null);
  selector2._fillBudget(ranked, 3, 10000, 4, 40);
  assert.ok(
    bumped.includes(200) || bumped.includes(201),
    `expected at least one fleagle-category item (200/201) to survive the floor guarantee, got ids: ${bumped.join(',')}`
  );
});

test('_fillBudget: unclaimed category floor budget falls back to the shared slack pool', () => {
  const cfg = makeConfig();
  cfg.context.categoryQuotas = { fleagle: 0.5 };
  const bumped = [];
  const spyHistory = makeMockHistory();
  spyHistory.bumpTurnRecall = (id) => bumped.push(id);
  const selector = new Selector(cfg, spyHistory, makeMockEmbedder(), null);
  // No fleagle candidates at all -- the floor share must not shrink the
  // total selected below maxSlots when enough background candidates exist.
  const ranked = Array.from({ length: 10 }, (_, i) => makeItem(i, 'background'));
  const out = selector._fillBudget(ranked, 3, 10000, 4, 40);
  assert.equal(out.length, 3, 'unclaimed floor budget should fall back to slack, not shrink the result');
});

test('_fillBudget: a category with no configured quota is treated as ordinary slack (backward compatible)', () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);
  const ranked = [makeItem(1, undefined), makeItem(2, undefined)];
  const out = selector._fillBudget(ranked, 5, 10000, 4, 40);
  assert.equal(out.length, 2);
});

// ─── vision content-parts arrays (Pleiades vision-routing, 2026-07-23) ──────
// engine.py sends the current turn's `content` as an OpenAI content-parts
// array (text + image_url) once a character's assigned model is vision-
// capable. Two things must hold: (1) the array survives select() completely
// unmangled so the image actually reaches the upstream model, and (2) the
// token-budget math must not silently under-count it (see _estOne's
// docstring in src/selector.js for the bug this used to be).

function makeImageMessage(text, dataUrl = 'data:image/png;base64,AAAAAAAAAAAAAAAAAAAA') {
  return {
    role: 'user',
    content: [
      { type: 'text', text },
      { type: 'image_url', image_url: { url: dataUrl } },
    ],
  };
}

test('select() passes an OpenAI content-parts (image_url) array through completely unmangled', async () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);
  const incoming = [
    { role: 'system', content: 'You are Mark.' },
    { role: 'user', content: 'earlier turn, plain text' },
    { role: 'assistant', content: 'ok' },
    makeImageMessage('what is in this image?', 'data:image/png;base64,SGVsbG8='),
  ];
  const result = await selector.select('session-vision-1', incoming);
  const last = result[result.length - 1];
  assert.ok(Array.isArray(last.content), 'expected the vision message content to still be an array');
  assert.equal(last.content.length, 2);
  assert.equal(last.content[0].type, 'text');
  assert.equal(last.content[0].text, 'what is in this image?');
  assert.equal(last.content[1].type, 'image_url');
  assert.equal(last.content[1].image_url.url, 'data:image/png;base64,SGVsbG8=');
});

test('select() does not crash and still anchors retrieval sensibly when the newest message is an image', async () => {
  const embeddedTexts = [];
  const spyEmbedder = {
    model: 'test-model',
    embed: async (text) => {
      embeddedTexts.push(text);
      return new Float32Array([0.1, 0.2]);
    },
  };
  const selector = new Selector(makeConfig(), makeMockHistory(), spyEmbedder, null);
  const incoming = [
    { role: 'system', content: 'You are Mark.' },
    makeImageMessage('describe this logo'),
  ];
  await selector.select('session-vision-2', incoming);
  // The query embedding must see the TEXT part only, never raw base64/JSON
  // noise from the image_url part (extractContentText already guarantees
  // this -- asserting it here pins the behavior at the selector-integration
  // level, not just proxy-helpers' own unit tests).
  assert.ok(embeddedTexts.length > 0);
  for (const t of embeddedTexts) {
    assert.equal(t, 'describe this logo');
  }
});

test('_estOne: a content-parts array with an image is costed far above the ~1-token undercount the old `.length` gave it', () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);
  const content = [
    { type: 'text', text: 'what is in this image?' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,' + 'A'.repeat(200000) } },
  ];
  const cpt = 4;
  const cost = selector._estOne(content, cpt);
  // Old behavior: Math.ceil(content.length / cpt) === Math.ceil(2 / 4) === 1.
  assert.ok(cost > 100, `expected a substantial per-image token estimate, got ${cost}`);
});

test('_estOne: plain string content is unaffected (byte-for-byte same as before)', () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);
  assert.equal(selector._estOne('hello world', 4), Math.ceil('hello world'.length / 4));
  assert.equal(selector._estOne('', 4), 0);
  assert.equal(selector._estOne(undefined, 4), 0);
});

test('_estOne: text-only content-parts array (no image) costs by real character length, not element count', () => {
  const selector = new Selector(makeConfig(), makeMockHistory(), makeMockEmbedder(), null);
  const content = [{ type: 'text', text: 'x'.repeat(400) }];
  const cost = selector._estOne(content, 4);
  assert.equal(cost, Math.ceil(400 / 4));
});
