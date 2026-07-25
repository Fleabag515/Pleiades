// test/importers/odysseus.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { extract } = require('../../src/importers/odysseus.js');

test('extracts memories as direct engrams', () => {
  const data = {
    memories: [
      { id: 'a1', text: 'user likes coffee', timestamp: 1000, session_id: 's1' },
      { id: 'a2', text: 'user works at night', timestamp: 2000, session_id: 's1' },
    ],
  };
  const result = extract(Buffer.from(JSON.stringify(data)));
  assert.equal(result.direct, true);
  assert.equal(result.data.engrams.length, 2);
  assert.equal(result.data.engrams[0].content, 'user likes coffee');
});

test('returns summary text for preview', () => {
  const data = { memories: [{ id: 'x', text: 'hello', timestamp: 1000, session_id: 's' }] };
  const result = extract(Buffer.from(JSON.stringify(data)));
  assert.ok(result.summary.includes('1 memor'));
});
