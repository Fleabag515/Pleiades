const test = require('node:test');
const assert = require('node:assert/strict');

const { shouldProcessTurn } = require('../src/lib/heuristics.js');

test('rejects empty / null', () => {
  assert.equal(shouldProcessTurn(''), false);
  assert.equal(shouldProcessTurn(null), false);
  assert.equal(shouldProcessTurn(undefined), false);
});

test('rejects too-short content (<80 chars)', () => {
  assert.equal(shouldProcessTurn('hello world'), false);
});

test('rejects XML/HTML-looking payloads', () => {
  const xml = '<response><status>ok</status><data>'.padEnd(200, 'x') + '</data></response>';
  assert.equal(shouldProcessTurn(xml), false);
});

test('rejects one-liner under 200 chars', () => {
  const line = 'a'.repeat(150);
  assert.equal(shouldProcessTurn(line), false);
});

test('accepts multi-paragraph content', () => {
  const text = 'first line here that is reasonably long\nsecond line here that is also long enough';
  assert.equal(shouldProcessTurn(text), true);
});

test('accepts long one-liner', () => {
  const text = 'a'.repeat(250);
  assert.equal(shouldProcessTurn(text), true);
});
