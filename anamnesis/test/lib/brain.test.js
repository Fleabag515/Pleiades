'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');

// brain.js is a singleton — reset require cache between tests
function freshBrain() {
  delete require.cache[require.resolve('../../src/lib/brain.js')];
  return require('../../src/lib/brain.js');
}

// ─── tryParseJson tests (migrated from ollama.test.js) ───────────────────────

test('tryParseJsonArray: plain JSON array', () => {
  assert.deepEqual(freshBrain().tryParseJsonArray('[1,2,3]'), [1, 2, 3]);
});

test('tryParseJsonArray: array embedded in prose', () => {
  assert.deepEqual(freshBrain().tryParseJsonArray('Here:\n[{"a":1}]\nDone.'), [{ a: 1 }]);
});

test('tryParseJsonArray: strips markdown fences', () => {
  assert.deepEqual(freshBrain().tryParseJsonArray('```json\n[1, 2]\n```'), [1, 2]);
});

test('tryParseJsonArray: returns null on no array', () => {
  assert.equal(freshBrain().tryParseJsonArray('no brackets'), null);
});

test('tryParseJsonArray: returns null on malformed', () => {
  assert.equal(freshBrain().tryParseJsonArray('[1, 2,'), null);
});

test('tryParseJsonArray: handles null/empty input', () => {
  const b = freshBrain();
  assert.equal(b.tryParseJsonArray(''), null);
  assert.equal(b.tryParseJsonArray(null), null);
});

test('tryParseJsonObject: plain object', () => {
  assert.deepEqual(freshBrain().tryParseJsonObject('{"a":1}'), { a: 1 });
});

test('tryParseJsonObject: object in prose', () => {
  assert.deepEqual(freshBrain().tryParseJsonObject('Sure: {"title":"x"} done.'), { title: 'x' });
});

// ─── Embedding ────────────────────────────────────────────────────────────────

test('brain.embed() returns null before engine is ready', async () => {
  const brain = freshBrain();
  const result = await brain.embed('hello');
  assert.equal(result, null);
});

test('brain.embeddingModel() returns model name constant before init', () => {
  const brain = freshBrain();
  assert.equal(brain.embeddingModel(), 'Xenova/all-MiniLM-L6-v2');
});

// ─── Queue behavior ───────────────────────────────────────────────────────────

test('brain.chat() queues when not ready, resolves after ready signal', async () => {
  const brain = freshBrain();

  const mockEngine = {
    isLoaded: () => false,
    load: async () => {},
    chat: async (messages) => `echo:${messages[0].content}`,
    generate: async (prompt) => `echo:${prompt}`,
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});

  const pending = brain.chat([{ role: 'user', content: 'hello' }]);
  let resolved = false;
  pending.then(() => (resolved = true));

  await new Promise((r) => setImmediate(r));
  assert.equal(resolved, false, 'should still be queued');

  mockEngine.isLoaded = () => true;
  brain._onEngineReady();

  const result = await pending;
  assert.equal(result, 'echo:hello');
});

test('brain.chat() before init() queues same as loading state', async () => {
  const brain = freshBrain();
  const mockEngine = {
    isLoaded: () => false,
    load: async () => {},
    chat: async (messages) => `reply:${messages[0].content}`,
    generate: async () => '',
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  // Call chat BEFORE init
  const pending = brain.chat([{ role: 'user', content: 'pre-init' }]);

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});

  mockEngine.isLoaded = () => true;
  brain._onEngineReady();

  const result = await pending;
  assert.equal(result, 'reply:pre-init');
});

test('brain.chat() timeoutMs applies after dequeue, not while queued', async () => {
  const brain = freshBrain();
  const mockEngine = {
    isLoaded: () => false,
    load: async () => {},
    chat: async () => {
      await new Promise((r) => setTimeout(r, 80)); // slow response
      return 'slow reply';
    },
    generate: async () => '',
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});

  // 20ms timeout — starts counting from dequeue, not from queue entry
  const pending = brain.chat([{ role: 'user', content: 'hi' }], { timeoutMs: 20 });

  // Wait 40ms before signaling ready — if timeout counted from here, would have fired
  await new Promise((r) => setTimeout(r, 40));
  mockEngine.isLoaded = () => true;
  brain._onEngineReady();

  // Now engine takes 80ms which exceeds the 20ms timeout
  await assert.rejects(() => pending, /timed out/);
});

test('brain: queued calls are rejected when engine load permanently fails', async () => {
  const brain = freshBrain();
  const mockEngine = {
    isLoaded: () => false,
    load: async () => {},
    chat: async () => 'should not reach',
    generate: async () => '',
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});

  const pending = brain.chat([{ role: 'user', content: 'hi' }]);
  brain._onEngineError(new Error('download failed permanently'));

  await assert.rejects(() => pending, /download failed/);
});

test('brain.generate() routes through chat as user message', async () => {
  const brain = freshBrain();
  let capturedMessages = null;
  const mockEngine = {
    isLoaded: () => true,
    load: async () => {},
    chat: async (messages) => {
      capturedMessages = messages;
      return 'generated';
    },
    generate: async () => 'raw',
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});
  brain._onEngineReady();

  const result = await brain.generate('my prompt');
  assert.equal(result, 'generated');
  assert.ok(Array.isArray(capturedMessages));
  assert.equal(capturedMessages[0].content, 'my prompt');
  assert.equal(capturedMessages[0].role, 'user');
});
