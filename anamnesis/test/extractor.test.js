'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');

// ── Helpers ────────────────────────────────────────────────────────────────────

function makeMockHistory() {
  const cells = [];
  return {
    getUnextractedAssistantTurns: (limit) => [],
    markExtracted: () => {},
    insertMemcell: (...args) => cells.push(args),
    _cells: cells,
  };
}

function makeMockEmbedder() {
  return {
    model: 'Xenova/all-MiniLM-L6-v2',
    embed: async () => new Float32Array([0.1, 0.2]),
  };
}

function makeConfig(overrides = {}) {
  return {
    extraction: { maxRetries: 1, timeoutMs: 5000, startupBacklogLimit: 10 },
    ...overrides,
  };
}

// Patch brain module before requiring Extractor — async so finally runs after awaited work
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
      embeddingModel: () => 'Xenova/all-MiniLM-L6-v2',
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

// ── Tests ──────────────────────────────────────────────────────────────────────

test('Extractor._callLLM calls brain.chat with correct shape', async () => {
  let captured = null;
  await withMockBrain(
    async (messages, opts) => {
      captured = { messages, opts };
      return '[{"fact":"Redis was chosen","importance":0.8,"category":"decision"}]';
    },
    (Extractor) => {
      const e = new Extractor(makeConfig(), makeMockHistory(), makeMockEmbedder());
      return e._callLLM('some assistant response');
    }
  ).then((facts) => {
    assert.ok(Array.isArray(facts));
    assert.equal(facts[0].fact, 'Redis was chosen');
    assert.ok(captured.messages[0].role === 'user');
    assert.ok(captured.opts.maxTokens > 0);
    assert.equal(captured.opts.temperature, 0.1);
  });
});

test('Extractor._extractTurn inserts memcells with embedder.model', async () => {
  const history = makeMockHistory();
  const embedder = makeMockEmbedder();

  history.markExtracted = () => {};
  history.insertMemcell = (...args) => history._cells.push(args);

  await withMockBrain(
    async () => '[{"fact":"User prefers dark mode","importance":0.9,"category":"preference"}]',
    (Extractor) => {
      const e = new Extractor(makeConfig(), history, embedder);
      return e._extractTurn({
        id: 1,
        session_key: 'abc123',
        content:
          'The user mentioned they prefer dark mode interfaces for all apps.\n' +
          'They also noted that light mode causes eye strain after long sessions.',
      });
    }
  );

  assert.equal(history._cells.length, 1);
  const [sessionKey, turnId, fact, , , category, model] = history._cells[0];
  assert.equal(sessionKey, 'abc123');
  assert.equal(turnId, 1);
  assert.ok(fact.includes('dark mode'));
  assert.equal(category, 'preference');
  assert.equal(model, 'Xenova/all-MiniLM-L6-v2');
});

test('Extractor._extractTurn skips facts shorter than 10 chars', async () => {
  const history = makeMockHistory();
  history.insertMemcell = (...args) => history._cells.push(args);

  await withMockBrain(
    async () => '[{"fact":"ok","importance":0.5,"category":"other"}]',
    (Extractor) => {
      const e = new Extractor(makeConfig(), history, makeMockEmbedder());
      return e._extractTurn({
        id: 2,
        session_key: 'sess',
        content: 'A meaningful enough assistant response that passes shouldProcessTurn.',
      });
    }
  );

  assert.equal(history._cells.length, 0, 'short fact should be skipped');
});

test('Extractor._extractTurn handles LLM failure gracefully', async () => {
  const history = makeMockHistory();
  let marked = false;
  history.markExtracted = () => {
    marked = true;
  };
  history.insertMemcell = () => {
    throw new Error('should not be called');
  };

  await withMockBrain(
    async () => {
      throw new Error('inference failed');
    },
    (Extractor) => {
      const e = new Extractor(makeConfig(), history, makeMockEmbedder());
      return e._extractTurn({
        id: 3,
        session_key: 'sess',
        content:
          'Some content that would normally be processed by the extractor.\n' +
          'It spans multiple lines so that shouldProcessTurn returns true.',
      });
    }
  );

  assert.ok(marked, 'turn should be marked extracted even on LLM failure');
});
