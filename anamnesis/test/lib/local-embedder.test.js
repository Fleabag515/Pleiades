'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { LocalEmbedder } = require('../../src/lib/local-embedder.js');

test('LocalEmbedder: MODEL_NAME is the Xenova model string', () => {
  assert.equal(LocalEmbedder.MODEL_NAME, 'Xenova/all-MiniLM-L6-v2');
});

test('LocalEmbedder: embed() returns null when not loaded', async () => {
  const e = new LocalEmbedder({ _skipLoad: true });
  const result = await e.embed('hello world');
  assert.equal(result, null);
});

test('LocalEmbedder: embed() returns Float32Array after load (mock pipeline)', async () => {
  const fakeOutput = {
    data: new Float32Array(Array(384).fill(0.1)),
    dims: [1, 1, 384],
  };
  const e = new LocalEmbedder({ _pipeline: async () => fakeOutput });
  e._loaded = true;
  const result = await e.embed('hello');
  assert.ok(result instanceof Float32Array, 'should be Float32Array');
  assert.equal(result.length, 384);
});

test('LocalEmbedder: embed() output is L2-normalized (magnitude ≈ 1)', async () => {
  // Raw [3, 4, 0, 0] has magnitude 5 — should be normalized to 1
  const rawData = new Float32Array([3, 4, 0, 0]);
  const fakeOutput = { data: rawData, dims: [1, 1, 4] };
  const e = new LocalEmbedder({ _pipeline: async () => fakeOutput });
  e._loaded = true;
  const result = await e.embed('test');
  const mag = Math.sqrt(result.reduce((s, v) => s + v * v, 0));
  assert.ok(Math.abs(mag - 1) < 1e-5, `magnitude should be 1, got ${mag}`);
});

test('LocalEmbedder: load() sets _loaded = true', async () => {
  let callCount = 0;
  const mockFactory = async (_task, _model) => {
    callCount++;
    return async () => ({ data: new Float32Array(384), dims: [1, 1, 384] });
  };
  const e = new LocalEmbedder({ _pipelineFactory: mockFactory });
  assert.equal(e._loaded, false);
  await e.load();
  assert.equal(e._loaded, true);
  assert.equal(callCount, 1);
});

test('LocalEmbedder: calling load() twice only calls factory once', async () => {
  let callCount = 0;
  const mockFactory = async () => {
    callCount++;
    return async () => ({ data: new Float32Array(384), dims: [1, 1, 384] });
  };
  const e = new LocalEmbedder({ _pipelineFactory: mockFactory });
  await Promise.all([e.load(), e.load()]);
  assert.equal(callCount, 1);
});

test('LocalEmbedder: mean-pool across multiple sequence tokens', async () => {
  // 2 tokens, 4 dims: [[1,0,0,0],[0,0,0,1]] → mean = [0.5,0,0,0.5]
  const rawData = new Float32Array([1, 0, 0, 0, 0, 0, 0, 1]);
  const fakeOutput = { data: rawData, dims: [1, 2, 4] };
  const e = new LocalEmbedder({ _pipeline: async () => fakeOutput });
  e._loaded = true;
  const result = await e.embed('test');
  // After mean-pool: [0.5, 0, 0, 0.5], magnitude = sqrt(0.5) ≈ 0.707
  // After L2 norm: [0.707, 0, 0, 0.707]
  const mag = Math.sqrt(result.reduce((s, v) => s + v * v, 0));
  assert.ok(Math.abs(mag - 1) < 1e-5, `magnitude should be 1, got ${mag}`);
  assert.ok(Math.abs(result[0] - result[3]) < 1e-5, 'first and last dims should be equal');
});
