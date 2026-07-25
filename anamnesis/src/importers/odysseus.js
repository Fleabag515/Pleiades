// src/importers/odysseus.js
'use strict';
function extract(buf) {
  const data = JSON.parse(buf.toString('utf8'));
  const memories = data.memories || [];
  const engrams = memories.map((m) => ({
    content: m.text,
    category: 'other',
    decay_score: 1.0,
    created_at: m.timestamp || Math.floor(Date.now() / 1000),
    session_key: m.session_id || 'imported',
  }));
  const summary = `${engrams.length} memor${engrams.length === 1 ? 'y' : 'ies'} from Odysseus export`;
  return { text: null, direct: true, summary, data: { engrams } };
}
module.exports = { extract };
