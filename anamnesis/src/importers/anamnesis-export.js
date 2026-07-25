'use strict';
function extract(buf) {
  const data = JSON.parse(buf.toString('utf8'));
  if (!data.anamnesis_export) throw new Error('not an anamnesis export file');
  const supported = [1];
  if (!supported.includes(data.version))
    throw new Error(`unsupported export version ${data.version}`);
  const summary = `${(data.engrams || []).length} engrams, ${(data.scenes || []).length} scenes, ${(data.foresights || []).length} foresights`;
  return { text: null, direct: true, summary, data };
}
module.exports = { extract };
