'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { InferenceEngine } = require('../../src/lib/inference-engine.js');

// All tests mock node-llama-cpp — no model needed in CI.

function makeMockLlama({ chatResponse = 'response text' } = {}) {
  const mockSession = {
    prompt: async (msg) => chatResponse + ':' + msg,
    dispose: async () => {},
  };
  const mockCtx = {
    getSequence: () => ({
      chatSession: mockSession,
      dispose: async () => {},
    }),
    dispose: async () => {},
  };
  const mockModel = {
    createContext: async () => mockCtx,
    dispose: async () => {},
  };
  return {
    loadModel: async () => mockModel,
    // Expose for test assertions
    _mockSession: mockSession,
    _mockCtx: mockCtx,
  };
}

test('InferenceEngine: isLoaded() returns false before load()', () => {
  const eng = new InferenceEngine({ _skipLoad: true });
  assert.equal(eng.isLoaded(), false);
});

test('InferenceEngine: isLoaded() returns true after load()', async () => {
  const mock = makeMockLlama();
  const eng = new InferenceEngine({
    _llamaFactory: async () => mock,
    modelPath: '/fake/path.gguf',
    gpuLayerBudgetMB: 0,
  });
  await eng.load('/fake/path.gguf');
  assert.equal(eng.isLoaded(), true);
});

test('InferenceEngine: GPU probe failure falls back to gpuLayers=0', async () => {
  const eng = new InferenceEngine({ _skipLoad: true, gpuLayerBudgetMB: 512 });
  eng._probeGpuFreeMB = async () => {
    throw new Error('nvidia-smi not found');
  };
  const layers = await eng._resolveGpuLayers();
  assert.equal(layers, 0);
});

test('InferenceEngine: GPU probe success caps at budget', async () => {
  const eng = new InferenceEngine({ _skipLoad: true, gpuLayerBudgetMB: 512 });
  // Probe returns 2000MB free — more than budget
  eng._probeGpuFreeMB = async () => 2000;
  const layers = await eng._resolveGpuLayers();
  // floor(512/55) = 9
  assert.equal(layers, 9);
});

test('InferenceEngine: GPU probe caps at free VRAM when less than budget', async () => {
  const eng = new InferenceEngine({ _skipLoad: true, gpuLayerBudgetMB: 512 });
  // Only 220MB free → floor(220/55) = 4, which is less than budget (9)
  eng._probeGpuFreeMB = async () => 220;
  const layers = await eng._resolveGpuLayers();
  assert.equal(layers, 4);
});

test('InferenceEngine: macOS Metal path uses budget directly', async () => {
  const eng = new InferenceEngine({ _skipLoad: true, gpuLayerBudgetMB: 512, _platform: 'darwin' });
  const layers = await eng._resolveGpuLayers();
  assert.equal(layers, Math.floor(512 / 55));
});

test('InferenceEngine: concurrent chat() calls are serialized', async () => {
  const callOrder = [];

  // Build a mock that records start/end order
  const mockCtx = {
    getSequence: () => ({
      dispose: async () => {},
    }),
    dispose: async () => {},
  };
  const mockModel = {
    createContext: async () => mockCtx,
    dispose: async () => {},
  };
  const mockLlama = { loadModel: async () => mockModel };

  const eng = new InferenceEngine({
    _llamaFactory: async () => mockLlama,
    modelPath: '/fake/path.gguf',
    gpuLayerBudgetMB: 0,
  });
  await eng.load('/fake/path.gguf');

  // Override _doChat to record order
  eng._doChat = async (messages) => {
    const id = messages[0].content;
    callOrder.push(`start:${id}`);
    await new Promise((r) => setTimeout(r, 10));
    callOrder.push(`end:${id}`);
    return `reply:${id}`;
  };

  await Promise.all([
    eng.chat([{ role: 'user', content: 'A' }]),
    eng.chat([{ role: 'user', content: 'B' }]),
  ]);

  // A must fully finish before B starts
  assert.deepEqual(callOrder, ['start:A', 'end:A', 'start:B', 'end:B']);
});
