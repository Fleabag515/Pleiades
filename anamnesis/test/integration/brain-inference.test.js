'use strict';

/**
 * test/integration/brain-inference.test.js
 *
 * Integration smoke-test for the brain.js → InferenceEngine → LocalEmbedder stack.
 * Uses test DI (mock engine + mock embedder) to verify the full queue→drain→chat
 * path without requiring a real model on disk.
 *
 * Run with: node --test test/integration/brain-inference.test.js
 */

const test = require('node:test');
const assert = require('node:assert/strict');

function freshBrain() {
  delete require.cache[require.resolve('../../src/lib/brain.js')];
  return require('../../src/lib/brain.js');
}

// ─── Full queue-drain integration ─────────────────────────────────────────────

test('brain: chat queues while loading, drains after init completes', async () => {
  const brain = freshBrain();

  const mockEngine = {
    isLoaded: () => false,
    load: async () => {},
    chat: async (messages) => `echo:${messages[0].content}`,
    generate: async () => '',
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});

  // Fire 3 concurrent chat calls before ready
  const calls = [
    brain.chat([{ role: 'user', content: 'a' }]),
    brain.chat([{ role: 'user', content: 'b' }]),
    brain.chat([{ role: 'user', content: 'c' }]),
  ];

  // None should be resolved yet
  let anyResolved = false;
  Promise.all(calls).then(() => (anyResolved = true));
  await new Promise((r) => setImmediate(r));
  assert.equal(anyResolved, false, 'calls should be queued before ready');

  // Signal ready
  brain._onEngineReady();

  const results = await Promise.all(calls);
  assert.deepEqual(results, ['echo:a', 'echo:b', 'echo:c']);
});

// ─── embed() returns null while not ready, non-null when loaded ───────────────

test('brain: embed() returns null before embedder loads', async () => {
  const brain = freshBrain();
  const result = await brain.embed('hello');
  assert.equal(result, null);
});

test('brain: embed() returns Float32Array when embedder is loaded', async () => {
  const brain = freshBrain();

  const vec = new Float32Array([0.1, 0.2, 0.3]);
  const mockEngine = {
    isLoaded: () => false,
    load: async () => {},
    chat: async () => '',
    generate: async () => '',
  };
  const mockEmbedder = {
    _loaded: true, // already loaded
    load: async () => {},
    embed: async () => vec,
  };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});

  const result = await brain.embed('hello');
  assert.ok(result instanceof Float32Array);
  assert.ok(Math.abs(result[0] - 0.1) < 1e-6);
});

// ─── error propagation ────────────────────────────────────────────────────────

test('brain: queued calls reject when engine fails', async () => {
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

  const pending = brain.chat([{ role: 'user', content: 'test' }]);
  brain._onEngineError(new Error('GPU out of memory'));

  await assert.rejects(() => pending, /GPU out of memory/);
});

// ─── generate() is sugar over chat() ─────────────────────────────────────────

test('brain.generate() wraps prompt as user message', async () => {
  const brain = freshBrain();
  let seen = null;

  const mockEngine = {
    isLoaded: () => true,
    load: async () => {},
    chat: async (msgs) => {
      seen = msgs;
      return 'ok';
    },
    generate: async () => '',
  };
  const mockEmbedder = { _loaded: false, load: async () => {}, embed: async () => null };

  brain._inject({ engine: mockEngine, embedder: mockEmbedder });
  brain.init({});
  brain._onEngineReady();

  await brain.generate('write a haiku');
  assert.equal(seen?.[0]?.role, 'user');
  assert.equal(seen?.[0]?.content, 'write a haiku');
});
