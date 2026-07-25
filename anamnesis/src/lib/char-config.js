'use strict';

const path = require('path');
const os = require('os');

const DEFAULTS = {
  context: {
    tokenBudget: 50000,
    systemReserveTokens: 4096,
    recencyTurns: 8,
    rotatingSlots: 6,
    charsPerToken: 3.5,
    minChunkChars: 50,
    // Category-partitioned quotas: guaranteed floor SHARE (0..1) of the
    // rotating-slot budget reserved for each category, filled from that
    // category's own candidates before anything competes in the shared
    // slack pool. Only categories a caller actually opts into (via the
    // X-Memory-Category header) participate — see history.js/selector.js.
    categoryQuotas: { fleagle: 0.3 },
    // Downtime-awareness: inject an explicit elapsed-time note when the gap
    // since the session's last turn exceeds minGapMinutes, instead of
    // silently continuing as if no time passed.
    downtimeAwareness: { enabled: true, minGapMinutes: 30 },
  },
  memory: {
    consolidationIntervalMs: 120000,
    consolidationBatchSize: 50,
    sceneClusterThreshold: 0.72,
    minSceneSize: 2,
    decayPruneThreshold: 0.05,
  },
  extraction: { maxRetries: 2, timeoutMs: 45000, startupBacklogLimit: 200 },
  foresight: { maxRetries: 2, timeoutMs: 45000, startupBacklogLimit: 200 },
  inference: { gpuLayerBudgetMB: 512 },
  history: { maxAgeDays: 90 },
};

const PERSONA_SHARED = {
  timeoutMs: 45000,
  drift: { enabled: true, checkEveryNTurns: 4, driftThreshold: 0.55 },
  evolution: { enabled: true, consolidateAfterNObservations: 8, maxEvolutionChars: 600 },
  injection: { maxProfileChars: 700 },
};

function buildConfig({
  name,
  port,
  upstreamUrl,
  apiKey,
  characterDescription,
  blank,
  overrides = {},
}) {
  const dbPath = path.join(os.homedir(), '.anamnesis', 'characters', name, 'history.db');

  let persona;
  if (blank) {
    persona = { enabled: false };
  } else if (characterDescription) {
    persona = {
      enabled: true,
      source: { type: 'inline', inline: { content: characterDescription }, openclaw: {}, file: {} },
      ...PERSONA_SHARED,
    };
  } else {
    persona = {
      enabled: true,
      source: {
        type: 'auto',
        openclaw: { soulPath: '~/.openclaw/SOUL.md' },
        file: { path: '~/.anamnesis/character.md' },
        inline: { content: '' },
      },
      ...PERSONA_SHARED,
    };
  }

  return {
    proxy: { port, host: '127.0.0.1' },
    upstream: { baseUrl: upstreamUrl, apiKey, disableThinking: true },
    inference: DEFAULTS.inference,
    extraction: DEFAULTS.extraction,
    context: DEFAULTS.context,
    memory: DEFAULTS.memory,
    history: { dbPath, maxAgeDays: DEFAULTS.history.maxAgeDays },
    foresight: DEFAULTS.foresight,
    persona,
    ...overrides,
  };
}

module.exports = { buildConfig, DEFAULTS };
