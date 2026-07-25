// src/importers/detect.js
'use strict';
const path = require('path');
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function detectFormat(filename, buf) {
  if (buf.slice(0, 8).equals(PNG_MAGIC)) return 'sillytavern';

  const ext = path.extname(filename).toLowerCase();
  if (ext === '.json' || buf[0] === 0x7b) {
    try {
      const obj = JSON.parse(buf.toString('utf8').slice(0, 2000));
      if (obj.anamnesis_export) return 'anamnesis-export';
      if (Array.isArray(obj.memories)) return 'odysseus';
      if (obj.participants && obj.histories) return 'characterai';
    } catch {
      /* not valid JSON */
    }
  }

  return 'text';
}

module.exports = { detectFormat };
