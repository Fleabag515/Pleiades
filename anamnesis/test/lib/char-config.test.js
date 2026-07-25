'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { buildConfig } = require('../../src/lib/char-config.js');

test('buildConfig produces valid config shape', () => {
  const cfg = buildConfig({
    name: 'mark',
    port: 8084,
    upstreamUrl: 'http://127.0.0.1:8083/v1',
    apiKey: 'localqwen',
  });
  assert.equal(cfg.proxy.port, 8084);
  assert.equal(cfg.upstream.baseUrl, 'http://127.0.0.1:8083/v1');
  assert.equal(cfg.upstream.apiKey, 'localqwen');
  assert.ok(cfg.history.dbPath.includes('mark'));
  assert.ok(cfg.context.tokenBudget > 0);
  assert.equal(cfg.inference.gpuLayerBudgetMB, 512);
  assert.equal(cfg.embedding, undefined, 'embedding section should be removed');
  assert.equal(cfg.extraction.model, undefined, 'extraction.model should be removed');
  assert.equal(cfg.foresight.model, undefined, 'foresight.model should be removed');
});

test('buildConfig sets persona.source.inline when description provided', () => {
  const cfg = buildConfig({
    name: 'aria',
    port: 8085,
    upstreamUrl: 'http://127.0.0.1:8083/v1',
    apiKey: 'key',
    characterDescription: 'sarcastic hacker',
  });
  assert.equal(cfg.persona.source.type, 'inline');
  assert.equal(cfg.persona.source.inline.content, 'sarcastic hacker');
});

test('buildConfig disables persona when blank', () => {
  const cfg = buildConfig({
    name: 'blank',
    port: 8086,
    upstreamUrl: 'http://localhost:8083/v1',
    apiKey: 'k',
    blank: true,
  });
  assert.equal(cfg.persona.enabled, false);
});
