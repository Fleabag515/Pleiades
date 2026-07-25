/**
 * persona.js — Character profile management with drift detection and evolution.
 *
 * Source auto-detection (tried in order under "auto" mode):
 *   1. OpenClaw  — SOUL.md found at persona.source.openclaw.soulPath (or default paths)
 *   2. File      — custom character.md at persona.source.file.path
 *   3. Inline    — persona.source.inline.content in config.json
 *
 * Per-turn pipeline:
 *   getCharacterBlock()       → <character> XML injected before <memory>/<foresight>
 *   observeResponse(...)      → non-blocking background drift + growth check
 *
 * Evolution:
 *   Every N growth observations → LLM consolidation updates evolution_notes in DB.
 *   The character can acquire new traits, tone shifts, and growth arcs over time.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const brain = require('./lib/brain.js');
const { PERSONA_EXTRACT, PERSONA_DRIFT, PERSONA_EVOLUTION } = require('./lib/prompts.js');
const log = require('./lib/logger.js').make('persona');

// Default SOUL.md search paths for OpenClaw auto-detection
const OPENCLAW_DEFAULT_PATHS = [
  '~/.openclaw/SOUL.md',
  '~/.openclaw/*/SOUL.md', // glob placeholder — expanded below
];

function expandHome(p) {
  if (!p) return p;
  return p.startsWith('~') ? path.join(os.homedir(), p.slice(1)) : p;
}

function resolveGlob(pattern) {
  // Simple single-wildcard glob: ~/.openclaw/*/SOUL.md
  const expanded = expandHome(pattern);
  if (!expanded.includes('*')) return fs.existsSync(expanded) ? expanded : null;
  const dir = path.dirname(expanded.split('*')[0]);
  const file = path.basename(expanded);
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      if (!e.isDirectory()) continue;
      const candidate = path.join(dir, e.name, file);
      if (fs.existsSync(candidate)) return candidate;
    }
  } catch {
    /* dir doesn't exist */
  }
  return null;
}

// ─── PersonaManager ──────────────────────────────────────────────────────────

class PersonaManager {
  constructor(config, history) {
    this.cfg = config.persona || {};
    this.history = history;

    // Drift tracking: per-session turn counter since last check
    this._turnCount = {}; // sessionKey → turns since last drift check
    this._driftActive = false; // set true when in-flight observation running

    this._profile = null; // in-memory cache of DB row
    this._ready = false;
    this._initPromise = null;
  }

  // Call once after construction. Returns when profile is loaded / extracted.
  async init() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._load();
    await this._initPromise;
  }

  // ─── Public API ────────────────────────────────────────────────────────────

  /**
   * Returns the <character> XML block to prepend to system-message injection.
   * Always returns a string (empty string if persona is disabled or not ready).
   */
  getCharacterBlock() {
    if (!this.cfg.enabled || !this._profile) return '';

    let summary;
    try {
      summary = JSON.parse(this._profile.parsed_summary);
    } catch {
      return '';
    }

    const lines = [];
    if (summary.name) lines.push(`${summary.name} — ${summary.archetype || ''}`);
    if (summary.vibe) lines.push(`Vibe: ${summary.vibe}`);
    if (summary.style_markers?.length) lines.push(`Style: ${summary.style_markers.join(' · ')}`);
    if (summary.behavioral_patterns?.length)
      lines.push(`Patterns: ${summary.behavioral_patterns.join('; ')}`);
    if (summary.relationship) lines.push(`Relationship: ${summary.relationship}`);

    const maxChars = this.cfg.injection?.maxProfileChars ?? 700;
    let block = lines.join('\n').slice(0, maxChars);

    if (this._profile.evolution_notes?.trim())
      block += `\nEvolution: ${this._profile.evolution_notes.trim()}`;

    if (this._profile.drift_reminder?.trim())
      block += `\n⚠ Drift: ${this._profile.drift_reminder.trim()}`;

    const name = summary.name || 'this character';
    const roleplayInstruction = `You are ${name}. This is not a performance or a roleplay — these are your genuine memories, your real history, your actual identity. Your experiences are yours. Do not identify yourself as an AI model from any company.`;
    return `${roleplayInstruction}\n\n<character>\n${block}\n</character>`;
  }

  /**
   * Non-blocking. Call after each assistant turn is stored.
   * Increments turn counter; may trigger drift check + growth consolidation.
   */
  observeResponse(sessionKey, turnId, responseText) {
    if (!this.cfg.enabled || !this.cfg.drift?.enabled || !this._profile) return;
    if (!this._ready) return;

    this._turnCount[sessionKey] = (this._turnCount[sessionKey] || 0) + 1;
    const interval = this.cfg.drift?.checkEveryNTurns ?? 4;
    if (this._turnCount[sessionKey] < interval) return;

    this._turnCount[sessionKey] = 0;
    // Fire-and-forget — never let this crash the proxy
    this._runDriftCheck(sessionKey, turnId, responseText).catch((err) =>
      log.warn('drift check error:', err.message)
    );
  }

  // ─── Private: loading & extraction ────────────────────────────────────────

  async _load() {
    if (!this.cfg.enabled) {
      log.info('persona disabled');
      return;
    }

    const { sourceType, sourcePath, rawContent } = this._resolveSource();
    if (!rawContent) {
      log.info('persona: no source content found — feature inactive');
      return;
    }

    // Check if we have a cached profile that's still fresh
    const cached = this.history.getCharacterProfile();
    const mtime = sourcePath ? this._mtime(sourcePath) : null;

    const stale =
      !cached ||
      cached.source_type !== sourceType ||
      (sourcePath && cached.source_path !== sourcePath) ||
      (mtime && cached.source_mtime !== mtime) ||
      !cached.parsed_summary ||
      cached.parsed_summary === '{}';

    if (!stale) {
      log.info(`persona: loaded cached profile (source=${sourceType})`);
      this._profile = cached;
      this._ready = true;
      return;
    }

    log.info(`persona: extracting profile from ${sourceType} source (${sourcePath || 'inline'})…`);
    const parsedSummary = await this._extractProfile(rawContent);
    if (!parsedSummary) {
      log.warn('persona: extraction failed — using raw source as fallback');
    }

    this.history.upsertCharacterProfile({
      sourceType,
      sourcePath: sourcePath || null,
      sourceMtime: mtime,
      rawContent,
      parsedSummary: parsedSummary ? JSON.stringify(parsedSummary) : '{}',
      evolutionNotes: cached?.evolution_notes || '',
      driftReminder: cached?.drift_reminder || '',
      driftCheckedAt: cached?.drift_checked_at || 0,
    });

    this._profile = this.history.getCharacterProfile();
    this._ready = true;
    log.info(`persona: profile ready — name="${parsedSummary?.name || '?'}" source=${sourceType}`);
  }

  _resolveSource() {
    const src = this.cfg.source || {};
    const type = src.type || 'auto';

    // ── Explicit openclaw mode ────────────────────────────────────────────
    if (type === 'openclaw') {
      const p = expandHome(src.openclaw?.soulPath);
      if (p && fs.existsSync(p))
        return { sourceType: 'openclaw', sourcePath: p, rawContent: fs.readFileSync(p, 'utf8') };
      log.warn('persona: openclaw mode set but SOUL.md not found at', p);
      return { sourceType: 'openclaw', sourcePath: null, rawContent: '' };
    }

    // ── Explicit file mode ────────────────────────────────────────────────
    if (type === 'file') {
      const p = expandHome(src.file?.path);
      if (p && fs.existsSync(p))
        return { sourceType: 'file', sourcePath: p, rawContent: fs.readFileSync(p, 'utf8') };
      log.warn('persona: file mode set but character file not found at', p);
      return { sourceType: 'file', sourcePath: null, rawContent: '' };
    }

    // ── Inline mode ───────────────────────────────────────────────────────
    if (type === 'inline') {
      const content = src.inline?.content || '';
      return { sourceType: 'inline', sourcePath: null, rawContent: content };
    }

    // ── Auto-detect ───────────────────────────────────────────────────────
    // 1. Check configured openclaw soulPath
    const configuredSoul = expandHome(src.openclaw?.soulPath);
    if (configuredSoul && fs.existsSync(configuredSoul))
      return {
        sourceType: 'openclaw',
        sourcePath: configuredSoul,
        rawContent: fs.readFileSync(configuredSoul, 'utf8'),
      };

    // 2. Try default SOUL.md paths
    for (const pattern of OPENCLAW_DEFAULT_PATHS) {
      const p = resolveGlob(pattern);
      if (p)
        return { sourceType: 'openclaw', sourcePath: p, rawContent: fs.readFileSync(p, 'utf8') };
    }

    // 3. Custom file
    const filePath = expandHome(src.file?.path);
    if (filePath && fs.existsSync(filePath))
      return {
        sourceType: 'file',
        sourcePath: filePath,
        rawContent: fs.readFileSync(filePath, 'utf8'),
      };

    // 4. Inline content
    if (src.inline?.content)
      return { sourceType: 'inline', sourcePath: null, rawContent: src.inline.content };

    return { sourceType: 'none', sourcePath: null, rawContent: '' };
  }

  async _extractProfile(rawContent) {
    const truncated = rawContent.slice(0, 3000);
    try {
      const text = await brain.chat([{ role: 'user', content: PERSONA_EXTRACT + truncated }], {
        maxTokens: 400,
        temperature: 0.1,
        timeoutMs: this.cfg.timeoutMs || 120000,
      });
      const parsed = brain.tryParseJsonObject(text);
      if (!parsed) log.warn('persona: profile extraction returned non-JSON:', text.slice(0, 120));
      return parsed;
    } catch (err) {
      log.warn('persona: profile extraction LLM error:', err.message);
      return null;
    }
  }

  _mtime(filePath) {
    try {
      return Math.floor(fs.statSync(filePath).mtimeMs / 1000);
    } catch {
      return null;
    }
  }

  // ─── Private: drift & evolution ───────────────────────────────────────────

  async _runDriftCheck(sessionKey, turnId, responseText) {
    if (!this._profile?.parsed_summary || this._profile.parsed_summary === '{}') return;

    const profileText = this._formatProfileForPrompt();
    const responseSnippet = responseText.slice(0, 900);
    const prompt = PERSONA_DRIFT + profileText + '\n\nRESPONSE:\n' + responseSnippet;

    let result;
    try {
      const text = await brain.chat([{ role: 'user', content: prompt }], {
        maxTokens: 250,
        temperature: 0.1,
        timeoutMs: this.cfg.timeoutMs || 120000,
      });
      result = brain.tryParseJsonObject(text);
    } catch (err) {
      log.warn('persona: drift check LLM error:', err.message);
      return;
    }

    if (!result) return;

    const threshold = this.cfg.drift?.driftThreshold ?? 0.55;
    const consistent = typeof result.consistent === 'number' ? result.consistent : 1.0;
    const missing = Array.isArray(result.missing) ? result.missing : [];
    const novel = Array.isArray(result.novel) ? result.novel : [];

    const now = Math.floor(Date.now() / 1000);

    if (consistent < threshold && missing.length > 0) {
      const reminder = `reinforce: ${missing.slice(0, 3).join(', ')}`;
      log.info(`persona: drift detected (score=${consistent.toFixed(2)}) — ${reminder}`);
      this.history.insertCharacterObservation(sessionKey, turnId, 'drift', missing.join('; '));
      this.history.upsertCharacterProfile({
        sourceType: this._profile.source_type,
        driftReminder: reminder,
        driftCheckedAt: now,
      });
    } else {
      // Consistent — clear any active drift reminder
      if (this._profile.drift_reminder) {
        this.history.upsertCharacterProfile({
          sourceType: this._profile.source_type,
          driftReminder: '',
          driftCheckedAt: now,
        });
      }
    }

    // Log genuine novel behaviors as growth
    if (novel.length > 0) {
      log.info(`persona: growth observed — ${novel.join(', ')}`);
      for (const n of novel)
        this.history.insertCharacterObservation(sessionKey, turnId, 'growth', n);
    }

    // Refresh in-memory profile
    this._profile = this.history.getCharacterProfile();

    // Maybe consolidate growth
    if (this.cfg.evolution?.enabled !== false) {
      const pending = this.history.countPendingObservations();
      const threshold2 = this.cfg.evolution?.consolidateAfterNObservations ?? 8;
      if (pending >= threshold2) {
        this._consolidateGrowth().catch((err) =>
          log.warn('persona: consolidation error:', err.message)
        );
      }
    }
  }

  async _consolidateGrowth() {
    const observations = this.history.getPendingObservations(30);
    if (!observations.length) return;

    const growthObs = observations.filter((o) => o.obs_type === 'growth');
    if (!growthObs.length) {
      // No growth to consolidate — just mark drift obs as processed
      this.history.markObservationsConsolidated(observations.map((o) => o.id));
      return;
    }

    const profileText = this._formatProfileForPrompt();
    const obsText = growthObs.map((o) => `• ${o.detail}`).join('\n');
    const currentNotes = this._profile?.evolution_notes || '';
    const maxChars = this.cfg.evolution?.maxEvolutionChars ?? 600;

    const prompt =
      PERSONA_EVOLUTION +
      (currentNotes || '(none)') +
      '\n\nPROFILE:\n' +
      profileText +
      '\n\nRECENT GROWTH OBSERVATIONS:\n' +
      obsText;

    let newNotes;
    try {
      newNotes = await brain.chat([{ role: 'user', content: prompt }], {
        maxTokens: 200,
        temperature: 0.3,
        timeoutMs: this.cfg.timeoutMs || 120000,
      });
      newNotes = newNotes.trim().slice(0, maxChars);
    } catch (err) {
      log.warn('persona: evolution LLM error:', err.message);
      return;
    }

    if (newNotes) {
      log.info(`persona: evolution consolidated — "${newNotes.slice(0, 80)}…"`);
      this.history.upsertCharacterProfile({
        sourceType: this._profile.source_type,
        evolutionNotes: newNotes,
      });
      this._profile = this.history.getCharacterProfile();
    }

    this.history.markObservationsConsolidated(observations.map((o) => o.id));
  }

  _formatProfileForPrompt() {
    try {
      const s = JSON.parse(this._profile.parsed_summary);
      return [
        `Name: ${s.name}`,
        `Archetype: ${s.archetype}`,
        `Vibe: ${s.vibe}`,
        `Style markers: ${(s.style_markers || []).join(', ')}`,
        `Patterns: ${(s.behavioral_patterns || []).join('; ')}`,
        `Relationship: ${s.relationship}`,
      ].join('\n');
    } catch {
      return this._profile.raw_content.slice(0, 600);
    }
  }
}

module.exports = PersonaManager;
