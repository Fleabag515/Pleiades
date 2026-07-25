'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const Registry = require('../src/lib/registry.js');
const CharacterManager = require('../src/character-manager.js');

function tmpSetup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'anamnesis-mgr-'));
  const reg = new Registry(path.join(dir, 'registry.json'));
  const mgr = new CharacterManager(reg, dir);
  return { dir, reg, mgr };
}

function minimalCharConfig(port = 19900, baseDir = os.tmpdir()) {
  return {
    proxy: { port, host: '127.0.0.1' },
    upstream: { baseUrl: 'http://127.0.0.1:9999/v1', apiKey: 'test', disableThinking: false },
    embedding: { ollamaUrl: 'http://127.0.0.1:11434', model: 'nomic-embed-cpu:latest' },
    extraction: { model: 'qwen3:0.6b', maxRetries: 1, timeoutMs: 5000, startupBacklogLimit: 0 },
    context: {
      tokenBudget: 4096,
      systemReserveTokens: 512,
      recencyTurns: 4,
      rotatingSlots: 2,
      charsPerToken: 3.5,
      minChunkChars: 50,
    },
    memory: {
      consolidationIntervalMs: 999999,
      consolidationBatchSize: 10,
      sceneClusterThreshold: 0.72,
      minSceneSize: 2,
      decayPruneThreshold: 0.05,
    },
    history: {
      dbPath: path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'anam-hist-')), 'history.db'),
      maxAgeDays: 30,
    },
    foresight: { model: 'qwen3:0.6b', maxRetries: 1, timeoutMs: 5000, startupBacklogLimit: 0 },
    persona: { enabled: false },
  };
}

test('createCharacter adds entry to registry', () => {
  const { dir, reg, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19900, dir));
  assert.ok(reg.get('mark'));
  assert.equal(reg.get('mark').name, 'mark');
});

test('createCharacter rejects duplicate names', () => {
  const { dir, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19901, dir));
  assert.throws(() => mgr.createCharacter('mark', minimalCharConfig(19902, dir)), /already exists/);
});

test('createCharacter rejects invalid names', () => {
  const { dir, mgr } = tmpSetup();
  assert.throws(
    () => mgr.createCharacter('my character', minimalCharConfig(19903, dir)),
    /invalid name/i
  );
  assert.throws(
    () => mgr.createCharacter('../evil', minimalCharConfig(19904, dir)),
    /invalid name/i
  );
});

test('deleteCharacter removes from registry', () => {
  const { dir, reg, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19905, dir));
  mgr.deleteCharacter('mark');
  assert.equal(reg.get('mark'), undefined);
});

test('deleteCharacter throws when character is active', () => {
  const { dir, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19906, dir));
  mgr._running.set('mark', { shutdown: async () => {} });
  assert.throws(() => mgr.deleteCharacter('mark'), /stop .* before delet/i);
  mgr._running.delete('mark');
});

test('listCharacters returns all entries with running flag', () => {
  const { dir, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19907, dir));
  mgr.createCharacter('aria', minimalCharConfig(19908, dir));
  const list = mgr.listCharacters();
  assert.equal(list.length, 2);
  assert.equal(list[0].running, false);
});

test('isActive returns false for stopped character', () => {
  const { dir, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19909, dir));
  assert.equal(mgr.isActive('mark'), false);
});

test('stopCharacter sets active=false in registry', async () => {
  const { dir, reg, mgr } = tmpSetup();
  mgr.createCharacter('mark', minimalCharConfig(19910, dir));
  mgr._running.set('mark', { shutdown: async () => {} });
  reg.setActive('mark', true);
  await mgr.stopCharacter('mark');
  assert.equal(mgr.isActive('mark'), false);
  assert.equal(reg.get('mark').active, false);
});

test('updateConfig deep-merges a partial patch and preserves siblings', () => {
  const { dir, reg, mgr } = tmpSetup();
  mgr.createCharacter('alice', minimalCharConfig(19950));
  const merged = mgr.updateConfig('alice', { upstream: { baseUrl: 'http://x/v1' } });
  assert.equal(merged.upstream.baseUrl, 'http://x/v1');
  assert.equal(merged.upstream.apiKey, 'test'); // sibling preserved
  assert.equal(merged.proxy.port, 19950); // untouched section preserved
  const onDisk = JSON.parse(
    fs.readFileSync(path.join(dir, 'characters', 'alice', 'config.json'), 'utf8'),
  );
  assert.equal(onDisk.upstream.baseUrl, 'http://x/v1');
  assert.equal(reg.get('alice') !== null, true);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('updateConfig syncs the registry port when proxy.port changes', () => {
  const { dir, reg, mgr } = tmpSetup();
  mgr.createCharacter('bob', minimalCharConfig(19960));
  mgr.updateConfig('bob', { proxy: { port: 19961 } });
  assert.equal(reg.get('bob').port, 19961);
  fs.rmSync(dir, { recursive: true, force: true });
});
