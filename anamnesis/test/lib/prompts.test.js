'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const prompts = require('../../src/lib/prompts.js');

const REQUIRED_EXPORTS = [
  'ENGRAM_EXTRACTION',
  'FORESIGHT',
  'EPISODE_SCENE',
  'PERSONA_EXTRACT',
  'PERSONA_DRIFT',
  'PERSONA_EVOLUTION',
  'IMPORT_EXTRACTION',
];

test('prompts: all required exports exist', () => {
  for (const key of REQUIRED_EXPORTS) {
    assert.ok(key in prompts, `missing export: ${key}`);
  }
});

test('prompts: all exports are non-empty strings', () => {
  for (const key of REQUIRED_EXPORTS) {
    const val = prompts[key];
    assert.equal(typeof val, 'string', `${key} is not a string`);
    assert.ok(val.length > 20, `${key} is suspiciously short`);
  }
});
