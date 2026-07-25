// src/importers/characterai.js
'use strict';
function extract(buf) {
  const data = JSON.parse(buf.toString('utf8'));
  const lines = [];
  if (data.participants)
    lines.push('Characters: ' + data.participants.map((p) => p.name).join(', '));
  const histories = data.histories?.histories || [];
  for (const h of histories) {
    for (const msg of h.msgs || []) {
      const speaker = msg.src?.name || (msg.src?.is_human ? 'User' : 'Character');
      lines.push(`${speaker}: ${msg.text}`);
    }
  }
  return { text: lines.join('\n'), direct: false };
}
module.exports = { extract };
