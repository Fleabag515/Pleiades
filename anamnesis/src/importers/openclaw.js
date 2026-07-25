// src/importers/openclaw.js
'use strict';
function extract(buf) {
  return { text: buf.toString('utf8'), direct: false };
}
module.exports = { extract };
