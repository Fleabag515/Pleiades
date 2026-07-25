// test/importers/sillytavern.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { extract, extractCharaFromPng } = require('../../src/importers/sillytavern.js');

function makeFakePng(charData) {
  const PNG_HEADER = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const b64 = Buffer.from(JSON.stringify(charData)).toString('base64');
  const keyword = 'chara';
  const payload = Buffer.concat([Buffer.from(keyword + '\0'), Buffer.from(b64)]);
  const type = Buffer.from('tEXt');
  const len = Buffer.alloc(4);
  len.writeUInt32BE(payload.length);
  const crc = Buffer.alloc(4);
  const chunk = Buffer.concat([len, type, payload, crc]);
  return Buffer.concat([PNG_HEADER, chunk]);
}

test('extractCharaFromPng returns parsed character JSON', () => {
  const charData = { name: 'Aria', description: 'A cool character', personality: 'witty' };
  const png = makeFakePng(charData);
  const result = extractCharaFromPng(png);
  assert.deepEqual(result, charData);
});

test('extract returns text representation of character', () => {
  const charData = {
    name: 'Aria',
    description: 'cool',
    personality: 'witty',
    scenario: 'sci-fi',
    mes_example: '',
  };
  const png = makeFakePng(charData);
  const result = extract(png);
  assert.equal(result.direct, false);
  assert.ok(result.text.includes('Aria'));
  assert.ok(result.text.includes('cool'));
});

test('extract throws on PNG without chara chunk', () => {
  const PNG_HEADER = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.throws(() => extract(PNG_HEADER), /no chara/i);
});
