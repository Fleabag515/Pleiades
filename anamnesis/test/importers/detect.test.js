// test/importers/detect.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { detectFormat } = require('../../src/importers/detect.js');

test('detects anamnesis export by key', () => {
  const buf = Buffer.from(JSON.stringify({ anamnesis_export: true, version: 1 }));
  assert.equal(detectFormat('export.json', buf), 'anamnesis-export');
});

test('detects odysseus by memories array', () => {
  const buf = Buffer.from(JSON.stringify({ memories: [], version: 1 }));
  assert.equal(detectFormat('dump.json', buf), 'odysseus');
});

test('detects characterai by participants key', () => {
  const buf = Buffer.from(JSON.stringify({ participants: [], histories: [] }));
  assert.equal(detectFormat('export.json', buf), 'characterai');
});

test('detects markdown as text', () => {
  const buf = Buffer.from('# Character\nShe is...');
  assert.equal(detectFormat('SOUL.md', buf), 'text');
});

test('detects .txt as text', () => {
  assert.equal(detectFormat('notes.txt', Buffer.from('hello')), 'text');
});

test('detects .png as sillytavern', () => {
  const buf = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.equal(detectFormat('char.png', buf), 'sillytavern');
});

test('falls back to text for unknown json', () => {
  const buf = Buffer.from(JSON.stringify({ something: 'else' }));
  assert.equal(detectFormat('file.json', buf), 'text');
});
