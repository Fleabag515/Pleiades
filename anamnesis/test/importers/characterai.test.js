// test/importers/characterai.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { extract } = require('../../src/importers/characterai.js');

test('flattens participants and histories to text', () => {
  const data = {
    participants: [{ name: 'Aria' }],
    histories: {
      histories: [
        {
          msgs: [
            { src: { is_human: false, name: 'Aria' }, text: 'Hello!' },
            { src: { is_human: true, name: 'User' }, text: 'Hi!' },
          ],
        },
      ],
    },
  };
  const result = extract(Buffer.from(JSON.stringify(data)));
  assert.equal(result.direct, false);
  assert.ok(result.text.includes('Aria'));
  assert.ok(result.text.includes('Hello!'));
});
