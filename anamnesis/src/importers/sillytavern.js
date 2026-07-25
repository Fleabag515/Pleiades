// src/importers/sillytavern.js
'use strict';
const PNG_HEADER = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function extractCharaFromPng(buf) {
  if (!buf.slice(0, 8).equals(PNG_HEADER)) throw new Error('not a PNG file');
  let offset = 8;
  while (offset < buf.length - 12) {
    const len = buf.readUInt32BE(offset);
    const type = buf.slice(offset + 4, offset + 8).toString('ascii');
    const data = buf.slice(offset + 8, offset + 8 + len);
    offset += 12 + len;
    if (type === 'tEXt') {
      const nullIdx = data.indexOf(0);
      if (nullIdx === -1) continue;
      const keyword = data.slice(0, nullIdx).toString('ascii');
      if (keyword !== 'chara') continue;
      const b64 = data.slice(nullIdx + 1).toString('ascii');
      return JSON.parse(Buffer.from(b64, 'base64').toString('utf8'));
    }
  }
  throw new Error('no chara tEXt chunk found in PNG — not a SillyTavern character card');
}

function extract(buf) {
  const char = extractCharaFromPng(buf);
  const lines = [];
  if (char.name) lines.push(`Name: ${char.name}`);
  if (char.description) lines.push(`Description: ${char.description}`);
  if (char.personality) lines.push(`Personality: ${char.personality}`);
  if (char.scenario) lines.push(`Scenario: ${char.scenario}`);
  if (char.mes_example) lines.push(`Example messages:\n${char.mes_example}`);
  if (char.system_prompt) lines.push(`System: ${char.system_prompt}`);
  for (const [k, v] of Object.entries(char)) {
    if (
      ![
        'name',
        'description',
        'personality',
        'scenario',
        'mes_example',
        'system_prompt',
        'spec',
        'spec_version',
      ].includes(k)
    ) {
      if (typeof v === 'string' && v.trim()) lines.push(`${k}: ${v}`);
    }
  }
  return { text: lines.join('\n\n'), direct: false };
}

module.exports = { extract, extractCharaFromPng };
