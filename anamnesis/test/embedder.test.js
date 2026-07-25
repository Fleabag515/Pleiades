'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');

// embedder is now a singleton — require it directly
const embedder = require('../src/embedder.js');
// cosine is still a static method on the constructor
const Embedder = embedder.constructor;

test('cosine: identical vectors => 1', () => {
  const v = new Float32Array([1, 2, 3, 4]);
  assert.equal(Embedder.cosine(v, v), 1);
});

test('cosine: orthogonal vectors => 0', () => {
  const a = new Float32Array([1, 0]);
  const b = new Float32Array([0, 1]);
  assert.equal(Embedder.cosine(a, b), 0);
});

test('cosine: opposite vectors => -1', () => {
  const a = new Float32Array([1, 2]);
  const b = new Float32Array([-1, -2]);
  assert.ok(Math.abs(Embedder.cosine(a, b) + 1) < 1e-6);
});

test('cosine: mismatched lengths => 0', () => {
  assert.equal(Embedder.cosine(new Float32Array([1, 2]), new Float32Array([1, 2, 3])), 0);
});

test('cosine: missing vectors => 0', () => {
  assert.equal(Embedder.cosine(null, new Float32Array([1])), 0);
  assert.equal(Embedder.cosine(new Float32Array([1]), null), 0);
  assert.equal(Embedder.cosine(null, null), 0);
});

test('cosine: zero vector vs any => 0 (no NaN)', () => {
  const z = new Float32Array([0, 0, 0]);
  const v = new Float32Array([1, 2, 3]);
  const s = Embedder.cosine(z, v);
  assert.equal(s, 0);
  assert.ok(!Number.isNaN(s));
});

test('cosine: monotonic w.r.t. similarity', () => {
  const q = new Float32Array([1, 0, 0]);
  const close = new Float32Array([0.9, 0.1, 0]);
  const far = new Float32Array([0, 1, 0]);
  assert.ok(Embedder.cosine(q, close) > Embedder.cosine(q, far));
});

// ─── Singleton behaviour ──────────────────────────────────────────────────────

test('embedder.model returns brain embedding model name', () => {
  assert.equal(embedder.model, 'Xenova/all-MiniLM-L6-v2');
});

test('embedder.embed() returns null before brain is ready', async () => {
  // brain not init'd in this test process — embed returns null gracefully
  const result = await embedder.embed('hello');
  assert.equal(result, null);
});

test('embedder.embed() returns null for empty string', async () => {
  assert.equal(await embedder.embed(''), null);
  assert.equal(await embedder.embed(null), null);
});
