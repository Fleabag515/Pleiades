/**
 * selector.js — Scene-guided context selection with memory injection.
 *
 * Two-stage pipeline inspired by EverMemOS and claude-mem's before_prompt_build:
 *
 * Stage 1 — System-message injection:
 *   Find top relevant Episodes, build a compact <memory> block, append it
 *   to the last system message. The model is explicitly *told* what it
 *   already knows. Only scenes above injectionMinSim are included, which
 *   prevents the "skiptracer" problem (irrelevant past context leaking in).
 *
 * Stage 2 — Rotating turn slots:
 *   Fill remaining token budget with turns from relevant scenes. Falls back
 *   to raw turn similarity if no scenes exist yet.
 *
 * Topic anchoring (drift fix):
 *   The retrieval query is NOT just the latest user message. Short or
 *   deictic messages ("yes", "do that", "what about the other one") embed
 *   to near-noise, and ranking every memory against noise is exactly how
 *   off-topic memories get injected and the conversation drifts. The query
 *   vector is a blend of the current message and a *topic anchor* — the
 *   normalized mean of the last few stored turn embeddings for this
 *   session. The shorter the current message, the harder the query leans
 *   on the anchor. See _buildQueryVec().
 *
 * Relevance floors (hallucination fix):
 *   Every retrieval path has a minimum-similarity floor. When nothing
 *   clears the floor, NOTHING is injected — an empty memory block is
 *   strictly better than a wrong one, because the model treats whatever we
 *   inject as ground truth.
 *
 * Final context shape:
 *   [system + <memory> block] + [rotating relevant turns] + [last N turns verbatim]
 */

const HistoryStore = require('./history.js');
const Embedder = require('./embedder.js');
const { extractContentText, findNewUserMessage, stripThinkingTokens, formatDuration } = require('./lib/proxy-helpers.js');
const log = require('./lib/logger.js').make('selector');

// Defaults — every one of these is overridable via config.context.*
const INJECTION_SCENES = 3; // scenes summarised in the injection block
const INJECTION_MIN_SIM = 0.45; // min scene similarity to inject
const SLOT_MIN_SIM = 0.32; // min raw similarity for rotating-slot candidates
const ANCHOR_TURNS = 6; // recent turns blended into the topic anchor
const ANCHOR_WEIGHT = 0.35; // anchor share of the query vector (long queries)
const ANCHOR_WEIGHT_SHORT = 0.65; // anchor share for short/deictic queries
const SHORT_QUERY_CHARS = 40; // "short query" threshold
const FORESIGHT_MIN_SIM = 0.22; // min relevance for a foresight to inject
const FORESIGHT_MAX = 3; // max foresights injected
const DOWNTIME_MIN_GAP_MINUTES = 30; // below this gap, no continuity note (normal back-and-forth)
const CATEGORY_QUOTAS_DEFAULT = { fleagle: 0.3 }; // guaranteed floor share of rotating budget
const IMAGE_TOKEN_ESTIMATE = 600; // flat per-image budget cost -- see _estOne()
// Days before an unfulfilled foresight stops being injected, per timeframe.
// Single source of truth lives on HistoryStore (the consolidator's expiry
// pass uses the same map).
const FORESIGHT_TTL_DAYS = HistoryStore.FORESIGHT_TTL_DAYS ?? {
  soon: 3,
  days: 7,
  weeks: 30,
  months: 90,
  ongoing: 45,
};

// ─── small vector helpers ────────────────────────────────────────────────────
function meanVec(vecs) {
  const usable = vecs.filter((v) => v && v.length);
  if (!usable.length) return null;
  const dim = usable[0].length;
  const out = new Float32Array(dim);
  let n = 0;
  for (const v of usable) {
    if (v.length !== dim) continue;
    for (let i = 0; i < dim; i++) out[i] += v[i];
    n++;
  }
  if (!n) return null;
  for (let i = 0; i < dim; i++) out[i] /= n;
  return normalize(out);
}

function normalize(v) {
  if (!v) return null;
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

function blend(a, b, wb) {
  // (1-wb)*a + wb*b, normalized. a and b must be same-dim.
  if (!a) return b ? normalize(b) : null;
  if (!b || b.length !== a.length) return normalize(a);
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (1 - wb) * a[i] + wb * b[i];
  return normalize(out);
}

function fmtDate(unixSec) {
  if (!unixSec) return '';
  try {
    return new Date(unixSec * 1000).toISOString().slice(0, 10);
  } catch {
    return '';
  }
}

function ageDays(unixSec) {
  return (Date.now() / 1000 - (unixSec ?? 0)) / 86400;
}

class Selector {
  constructor(config, historyStore, embedder, persona = null) {
    this.cfg = config.context;
    this.history = historyStore;
    this.embedder = embedder;
    this.persona = persona; // PersonaManager instance or null
  }

  // Tunable with config.context override, falling back to module default.
  _t(key, fallback) {
    const v = this.cfg?.[key];
    return typeof v === 'number' ? v : fallback;
  }

  /**
   * @param {string} sessionKey
   * @param {Array} incoming — the request's messages array
   * @param {object} [opts]
   * @param {number} [opts.budgetCeiling] — effective token ceiling discovered
   *   from the upstream server's real context window (see proxy.js). The
   *   configured tokenBudget is clamped to this so we never build a context
   *   the upstream will silently truncate — front-truncation eats the system
   *   prompt + memory block first, which reads as amnesia + hallucination.
   */
  async select(sessionKey, incoming, opts = {}) {
    const { systemReserveTokens, recencyTurns, rotatingSlots, charsPerToken, minChunkChars } =
      this.cfg;
    let tokenBudget = this.cfg.tokenBudget;
    if (typeof opts.budgetCeiling === 'number' && opts.budgetCeiling > 0) {
      tokenBudget = Math.min(tokenBudget, opts.budgetCeiling);
    }

    const systemMsgs = incoming.filter((m) => m.role === 'system');
    const convoMsgs = incoming.filter((m) => m.role !== 'system');
    // Anchor retrieval to the last REAL user message, not just the literal
    // last array entry. For an agentic tool-calling client (Pleiades), one
    // user turn resends the whole growing array once per tool round-trip,
    // and the literal last message during those rounds is a tool result or
    // an assistant tool_call, not the user's actual question. Embedding that
    // as the query (a) drifts retrieval away from what the user is actually
    // asking about (a shell command's stdout, an email body, etc. is not a
    // meaningful memory query) and (b) recomputes a DIFFERENT <memory>/
    // rotating-slot selection on every single round, which defeats
    // llama-server's longest-common-prefix KV-cache reuse for the rest of
    // that turn -- measured as real 20-30s+ stalls between tool rounds on a
    // 35B model (see Pleiades' native-inference-engine design doc, Phase 6).
    // Falls back to the literal last message only if no user-role message
    // exists at all (shouldn't happen for Pleiades' calling convention, but
    // keeps this from ever throwing on an unusual caller). Shares its
    // role==='user' reverse-scan with the proxy's own persistence dedup
    // (see findNewUserMessage's docstring) so a synthetic tool round (e.g.
    // Pleiades' self-reflection check) is uniformly invisible to both --
    // it never becomes the retrieval anchor here any more than it becomes
    // "the new user turn" there.
    const lastUserMsg = findNewUserMessage(convoMsgs);
    const currentMsg = lastUserMsg ?? convoMsgs[convoMsgs.length - 1];
    // Normalise possibly-array content (OpenAI multipart) into plain text
    // so the embedding sees the same string that gets stored as the turn.
    const queryText = extractContentText(currentMsg?.content);
    const currentModel = this.embedder.model;
    const rawVec = queryText ? await this.embedder.embed(queryText) : null;
    const queryVec = await this._buildQueryVec(sessionKey, queryText, currentModel, rawVec);

    // Recency buffer — always included verbatim.
    //
    // convoMsgs is the *incoming request's* non-system messages, not a log of
    // distinct conversational turns -- for an agentic tool-calling client
    // (e.g. Pleiades) a single user turn can itself expand into many
    // (assistant tool_call, tool result) round-trips, all resent in the one
    // growing array for that turn. A blind count-based slice can therefore
    // cut deeper than the turn's own anchoring user message and drop it
    // entirely, leaving an upstream request with zero user-role messages.
    // Some chat templates assume at least one is always present (e.g. Qwen3's
    // multi_step_tool branch raises "No user query found in messages."
    // otherwise), so never slice past the most recent user message, even if
    // that means the window holds more than recencyTurns*2 messages for one
    // oversized turn.
    const recencyWindow = recencyTurns * 2;
    let recencyCut = Math.max(0, convoMsgs.length - recencyWindow);
    const lastUserIdx = convoMsgs.reduce((acc, m, i) => (m.role === 'user' ? i : acc), -1);
    if (lastUserIdx !== -1 && lastUserIdx < recencyCut) recencyCut = lastUserIdx;
    const recencyMsgs = convoMsgs.slice(recencyCut);

    const scenes = this.history.getScenes(sessionKey);

    // ─── Stage 1: build memory + foresight + continuity injection block ─────
    const foresights = this._relevantForesights(sessionKey, queryVec, currentModel, rawVec);
    const downtimeNote = this._buildDowntimeNote(opts.lastActivityAt);
    const enrichedSystem = this._buildSystemWithMemory(
      systemMsgs,
      scenes,
      queryVec,
      currentModel,
      foresights,
      rawVec,
      downtimeNote
    );

    // ─── Budget accounting ─────────────────────────────────────────────────
    const systemTokens = this._est(enrichedSystem, charsPerToken);
    const recencyTokens = this._est(recencyMsgs, charsPerToken);
    const budget = tokenBudget - systemReserveTokens - systemTokens - recencyTokens;

    // ─── Stage 2: rotating turn slots ──────────────────────────────────────
    let rotatingMsgs = [];
    if (budget > 0 && queryVec) {
      if (scenes.length > 0) {
        rotatingMsgs = this._sceneGuidedRetrieval(
          sessionKey,
          queryVec,
          currentModel,
          scenes,
          rotatingSlots,
          budget,
          charsPerToken,
          minChunkChars,
          rawVec
        );
      } else {
        rotatingMsgs = this._rawTurnRetrieval(
          sessionKey,
          queryVec,
          currentModel,
          recencyMsgs.length,
          rotatingSlots,
          budget,
          charsPerToken,
          minChunkChars
        );
      }
      // NOTE: category quotas are applied inside _fillBudget (both retrieval
      // paths funnel through it), reading this.cfg.categoryQuotas directly —
      // no extra params needed on the two call sites above.
    }

    const final = [...enrichedSystem, ...rotatingMsgs, ...recencyMsgs];

    // Debug-level: noisy when many requests/sec. Bump ANAMNESIS_LOG=debug.
    const stats = this.history.stats(sessionKey);
    log.debug(
      `session=${sessionKey.slice(0, 8)} ` +
        `turns=${stats.turns} cells=${stats.cells} scenes=${stats.scenes} foresights=${stats.foresights} ` +
        `injected=${enrichedSystem.length > systemMsgs.length ? 'yes' : 'no'} ` +
        `rotating=${rotatingMsgs.length} recency=${recencyMsgs.length} budget=${tokenBudget}`
    );

    return final;
  }

  /**
   * Build the retrieval query vector: current message blended with the
   * session's short-term topic anchor (mean of recent turn embeddings).
   *
   * Returns null only when we have neither a usable current message nor any
   * recent turn vectors — in which case no similarity-based injection
   * happens at all (recency still flows through untouched).
   */
  async _buildQueryVec(sessionKey, queryText, currentModel, precomputedCur = null) {
    const cur = precomputedCur ?? (queryText ? await this.embedder.embed(queryText) : null);

    let anchor = null;
    try {
      const anchorTurns = this._t('anchorTurns', ANCHOR_TURNS);
      const recent = this.history.getRecentTurnVectors(sessionKey, anchorTurns, currentModel);
      anchor = meanVec(recent.map((r) => HistoryStore.toFloat32(r.embedding)));
    } catch (e) {
      log.debug('anchor build failed:', e.message);
    }

    if (!cur && !anchor) return null;
    if (!cur) return anchor;
    if (!anchor) return normalize(cur);

    const short = (queryText?.length ?? 0) < this._t('shortQueryChars', SHORT_QUERY_CHARS);
    const w = short
      ? this._t('anchorWeightShort', ANCHOR_WEIGHT_SHORT)
      : this._t('anchorWeight', ANCHOR_WEIGHT);
    return blend(cur, anchor, w);
  }

  /**
   * Foresights worth injecting: unfulfilled, not expired for their declared
   * timeframe, and (when both vectors exist) at least loosely relevant to
   * the current query. Stale "I was about to X" lines from weeks ago are a
   * classic drift vector — the model keeps steering back to dead plans.
   */
  _relevantForesights(sessionKey, queryVec, currentModel, rawVec = null) {
    let rows = [];
    try {
      rows = this.history.getActiveForesights(sessionKey, 25);
    } catch {
      return [];
    }
    const minSim = this._t('foresightMinSim', FORESIGHT_MIN_SIM);
    const out = [];
    for (const f of rows) {
      const ttl = FORESIGHT_TTL_DAYS[f.timeframe] ?? 7;
      if (ageDays(f.created_at) > ttl) continue; // expired — consolidator will retire it
      if (queryVec && f.embedding && (!f.embedding_model || f.embedding_model === currentModel)) {
        const fVec = HistoryStore.toFloat32(f.embedding);
        let sim = null;
        if (fVec) {
          const b = Embedder.constructor.cosine(queryVec, fVec);
          const r = rawVec ? Embedder.constructor.cosine(rawVec, fVec) : 0;
          sim = Math.max(b, r);
        }
        if (sim !== null && sim < minSim) continue;
        out.push({ ...f, sim: sim ?? 0 });
      } else if (ageDays(f.created_at) <= 7) {
        // Legacy row without a comparable vector: relevance can't be
        // checked, so only surface it while fresh. `anamnesis reembed`
        // backfills vectors and retires this branch.
        out.push({ ...f, sim: 0 });
      }
    }
    out.sort((a, b) => b.sim - a.sim || b.created_at - a.created_at);
    return out.slice(0, this._t('foresightMax', FORESIGHT_MAX));
  }

  /**
   * Score a scene against the query. Returns { sim, score }:
   *   sim   — raw cosine, used against floors;
   *   score — sim weighted by importance and a mild recency factor, used
   *           for ranking. Recency matters: two scenes at equal similarity
   *           should not tie when one is from yesterday and one from three
   *           months ago.
   */
  _sceneScore(scene, queryVec, currentModel, rawVec = null) {
    if (scene.embedding_model && scene.embedding_model !== currentModel) {
      return { sim: null, score: 0.2 * (0.7 + (scene.avg_importance ?? 0.5) * 0.3) };
    }
    const sVec = HistoryStore.toFloat32(scene.embedding);
    // Eligibility takes the best of the blended query and the raw current
    // message: blending rescues short deictic turns ("yeah do that"), but
    // dilutes long self-contained questions whenever the recent-turn anchor
    // is about something else. max() keeps both cases honest — noise vectors
    // very rarely clear the floors on their own.
    let sim = 0;
    if (sVec) {
      const b = queryVec ? Embedder.constructor.cosine(queryVec, sVec) : 0;
      const r = rawVec ? Embedder.constructor.cosine(rawVec, sVec) : 0;
      sim = Math.max(b, r);
    }
    const recency = 0.85 + 0.15 * Math.exp(-ageDays(scene.updated_at) / 45);
    return { sim, score: sim * (0.7 + (scene.avg_importance ?? 0.5) * 0.3) * recency };
  }

  /**
   * Build the downtime-awareness continuity note, or null if none is
   * warranted. `lastActivityAt` is the unix-seconds timestamp of the
   * session's most recent turn BEFORE the incoming one (captured by
   * proxy.js prior to inserting the new user turn) — null/undefined for a
   * brand-new session, in which case there's nothing to be aware of.
   *
   * Deliberately factual, not emotive: this states the elapsed time and
   * lets the character react to it in its own voice, rather than
   * pre-scripting a reaction (that's persona's job, not the memory layer's).
   */
  _buildDowntimeNote(lastActivityAt) {
    const cfg = this.cfg?.downtimeAwareness;
    if (cfg?.enabled === false) return null;
    if (!lastActivityAt) return null;
    const gapSeconds = Date.now() / 1000 - lastActivityAt;
    const minGapMinutes = typeof cfg?.minGapMinutes === 'number' ? cfg.minGapMinutes : DOWNTIME_MIN_GAP_MINUTES;
    if (gapSeconds < minGapMinutes * 60) return null;
    const last = new Date(lastActivityAt * 1000).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
    return (
      `${formatDuration(gapSeconds)} have passed since the last exchange (last active ${last}). ` +
      `This is real elapsed time, not a new topic — you are aware the gap happened. React to it ` +
      `naturally in character if it comes up; do not pretend no time passed, and do not treat this ` +
      `note itself as something to repeat verbatim.`
    );
  }

  /**
   * Append <continuity>/<memory>/<foresight> blocks to the last system message.
   */
  _buildSystemWithMemory(
    systemMsgs,
    scenes,
    queryVec,
    currentModel,
    foresights = [],
    rawVec = null,
    downtimeNote = null
  ) {
    const hasMemory = scenes.length > 0 && queryVec;
    const hasForesight = foresights.length > 0;
    const hasDowntime = !!downtimeNote;
    const _charBlock = this.persona ? this.persona.getCharacterBlock() : '';
    if (!hasMemory && !hasForesight && !hasDowntime) {
      if (!_charBlock) return systemMsgs;
      // Inject character block even when no memories or foresights are present
      if (systemMsgs.length === 0) return [{ role: 'system', content: _charBlock.trim() }];
      return [
        {
          role: 'system',
          content: _charBlock.trim() + '\n\n' + systemMsgs.map((m) => m.content).join('\n\n'),
        },
      ];
    }

    // Character block is always prepended (persona handles its own enabled check)
    const characterBlock = this.persona ? this.persona.getCharacterBlock() : '';
    let injection = characterBlock;

    if (hasDowntime) {
      injection += `\n\n<continuity>\n${downtimeNote}\n</continuity>`;
    }

    if (hasMemory) {
      const injMin = this._t('injectionMinSim', INJECTION_MIN_SIM);
      const relevant = scenes
        .map((s) => ({ ...s, ...this._sceneScore(s, queryVec, currentModel, rawVec) }))
        // sim === null → vector from another embedding model: unverifiable
        // relevance. Never inject those as "memories" — they go through the
        // rotating-slot path at reduced priority instead.
        .filter((s) => s.sim !== null && s.sim >= injMin)
        .sort((a, b) => b.score - a.score)
        .slice(0, this._t('injectionScenes', INJECTION_SCENES));

      if (relevant.length) {
        const memLines = relevant
          .map((s) => {
            const d = fmtDate(s.updated_at);
            return `• ${d ? `[${d}] ` : ''}[${s.title}] ${s.summary}`;
          })
          .join('\n');
        injection +=
          `\n\n<memory>\nExcerpts from your long-term memory of earlier sessions. ` +
          `They may be incomplete or summarized — the CURRENT conversation is always ` +
          `authoritative. If a memory conflicts with what is being said now, trust the ` +
          `conversation. Never invent details beyond what is written here:\n${memLines}\n</memory>`;
      }
    }

    if (hasForesight) {
      const fLines = foresights
        .map((f) => {
          const tag = f.target ? ` (${f.target})` : '';
          const age = Math.floor(ageDays(f.created_at));
          const ageTag = age >= 1 ? ` — noted ${age}d ago` : '';
          return `• [${f.timeframe}]${tag} ${f.intention}${ageTag}`;
        })
        .join('\n');
      injection +=
        `\n\n<foresight>\nIntentions you noted in earlier sessions. Only act on one if it is ` +
        `relevant to what is being discussed NOW — do not steer the conversation back to ` +
        `these on your own:\n${fLines}\n</foresight>`;
    }

    if (!injection) return systemMsgs;

    if (systemMsgs.length === 0) {
      return [{ role: 'system', content: injection.trim() }];
    }
    // Prepend so character persona takes priority over any client system prompt
    const enriched = [
      {
        role: 'system',
        content: injection.trim() + '\n\n' + systemMsgs.map((m) => m.content).join('\n\n'),
      },
    ];
    return enriched;
  }

  _sceneGuidedRetrieval(
    sessionKey,
    queryVec,
    currentModel,
    scenes,
    maxSlots,
    budget,
    cpt,
    minChars,
    rawVec = null
  ) {
    const slotMin = this._t('slotMinSim', SLOT_MIN_SIM);
    const scoredAll = scenes.map((s) => ({
      ...s,
      ...this._sceneScore(s, queryVec, currentModel, rawVec),
    }));

    // Same-model scenes must clear the relevance floor. Unverifiable
    // (model-mismatch) scenes are only a COLD-START fallback: right after an
    // embedding-model swap nearly nothing is comparable, and unverifiable
    // beats total amnesia. But when the store has plenty of comparable
    // scenes and none are relevant, inject nothing — that's the floor doing
    // its job (`anamnesis reembed` is the real fix for a mixed store).
    let qualified = scoredAll.filter((s) => s.sim !== null && s.sim >= slotMin);
    const comparable = scoredAll.filter((s) => s.sim !== null).length;
    if (qualified.length < 2 && comparable < 3) {
      qualified = qualified.concat(scoredAll.filter((s) => s.sim === null));
    }
    const scored = qualified.sort((a, b) => b.score - a.score).slice(0, maxSlots * 2);

    const turnIdSet = new Set();
    for (const scene of scored) {
      this.history.bumpSceneRecall(scene.id);
      let ids;
      try {
        ids = JSON.parse(scene.engram_ids);
      } catch {
        continue;
      }
      if (!ids.length) continue;

      for (const id of this.history.getTurnIdsForMemcells(ids)) turnIdSet.add(id);
      if (turnIdSet.size >= maxSlots * 4) break;
    }

    const candidateTurns = this.history.getTurnsByIds([...turnIdSet]);
    const allTurns = this.history.getSessionTurns(sessionKey);
    const turnMap = new Map(allTurns.map((t) => [t.id, t]));

    const ranked = candidateTurns
      .map((t) => {
        const full = turnMap.get(t.id);
        if (full?.embedding_model && full.embedding_model !== currentModel) {
          return { ...t, sim: null, score: (full?.importance ?? 0.5) * 0.5 };
        }
        const tVec = full?.embedding ? HistoryStore.toFloat32(full.embedding) : null;
        const sim = tVec ? Embedder.constructor.cosine(queryVec, tVec) : null;
        const imp = full?.importance ?? 0.5;
        return { ...t, sim, score: (sim ?? 0.3) * 0.7 + imp * 0.3 };
      })
      // Floor: a turn with a comparable vector must be at least loosely
      // on-topic. Turns without vectors (sim === null) are allowed through
      // on importance alone — they were selected via a qualified scene.
      .filter((t) => t.sim === null || t.sim >= slotMin)
      .sort((a, b) => b.score - a.score);

    return this._fillBudget(ranked, maxSlots, budget, cpt, minChars);
  }

  _rawTurnRetrieval(
    sessionKey,
    queryVec,
    currentModel,
    recencyCount,
    maxSlots,
    budget,
    cpt,
    minChars
  ) {
    const slotMin = this._t('slotMinSim', SLOT_MIN_SIM);
    const allTurns = this.history.getSessionTurns(sessionKey);
    const candidates = allTurns.slice(0, Math.max(0, allTurns.length - recencyCount));

    const scored = candidates
      .filter((t) => t.role === 'assistant' && t.content.length >= minChars)
      .map((t) => {
        if (t.embedding_model && t.embedding_model !== currentModel) {
          return { ...t, score: 0 };
        }
        const tVec = HistoryStore.toFloat32(t.embedding);
        const sim = tVec && queryVec ? Embedder.constructor.cosine(queryVec, tVec) : 0;
        return { ...t, score: sim };
      })
      .filter((t) => t.score >= slotMin)
      .sort((a, b) => b.score - a.score);

    return this._fillBudget(scored, maxSlots, budget, cpt, minChars);
  }

  /**
   * Fill up to `maxSlots` from `ranked` (descending score) without exceeding
   * the token budget. When an item doesn't fit we `continue` rather than
   * `break` so a small high-value item can still slot in after a too-big
   * one is rejected — at the cost of slight non-monotonicity in selection
   * order. We re-sort the selected items by created_at so the conversation
   * reads chronologically when re-injected.
   *
   * Category-partitioned quotas (config.context.categoryQuotas, e.g.
   * `{ fleagle: 0.3 }`): before the plain score-ranked fill above ran
   * unconditionally, so a long stretch of unrelated/background-tagged
   * content (see history.js turns.category / proxy-helpers.js
   * getMemoryCategory) could out-compete older owner-relevant turns for
   * every slot in this budget purely by volume. Each configured category
   * now gets first crack at a reserved floor SHARE of the token budget and
   * (at least one, proportionally more) slot, filled from that category's
   * own best-ranked candidates only. Whatever floor budget/slots a category
   * doesn't use (too few qualifying candidates) falls back into the shared
   * slack pool below, exactly like unclaimed capacity in a resource quota.
   * The remaining budget+slots after all floors are honored are filled
   * from every not-yet-selected candidate by pure score, same as before —
   * so a caller that never sets categoryQuotas (or never tags a category)
   * sees byte-for-byte the old behavior.
   */
  _fillBudget(ranked, maxSlots, budget, cpt, _minChars) {
    const selected = [];
    const seenIds = new Set();
    let remaining = budget;
    let remainingSlots = maxSlots;

    const take = (items, budgetCap, slotCap) => {
      let spentTokens = 0;
      let spentSlots = 0;
      for (const t of items) {
        if (spentSlots >= slotCap) break;
        if (seenIds.has(t.id)) continue;
        const cost = Math.ceil((t.content?.length ?? 0) / cpt);
        if (cost > budgetCap - spentTokens) continue; // skip oversized, try a smaller one
        selected.push(t);
        seenIds.add(t.id);
        spentTokens += cost;
        spentSlots += 1;
        this.history.bumpTurnRecall(t.id);
      }
      return { spentTokens, spentSlots };
    };

    // Undefined (key never set) -> module default applies automatically, so
    // existing characters get the floor without a config migration. An
    // EXPLICIT {} opts a character out entirely (nullish-coalescing only
    // falls back on null/undefined, not on an empty object).
    const quotas = this.cfg?.categoryQuotas ?? CATEGORY_QUOTAS_DEFAULT;
    if (quotas && typeof quotas === 'object') {
      // Largest floor first: with small slot counts, rounding a floor up to
      // "at least 1 slot" for every category could otherwise overcommit
      // maxSlots — processing biggest-guarantee-first means any such
      // overcommit eats into the smaller, later floors rather than the
      // largest one.
      const cats = Object.entries(quotas)
        .filter(([, share]) => typeof share === 'number' && share > 0)
        .sort((a, b) => b[1] - a[1]);
      for (const [cat, share] of cats) {
        if (remaining <= 0 || remainingSlots <= 0) break;
        const floorTokens = Math.min(Math.floor(budget * share), remaining);
        const floorSlots = Math.min(Math.max(1, Math.round(maxSlots * share)), remainingSlots);
        const bucket = ranked.filter((t) => (t.category || 'fleagle') === cat && !seenIds.has(t.id));
        if (!bucket.length) continue;
        const { spentTokens, spentSlots } = take(bucket, floorTokens, floorSlots);
        remaining -= spentTokens;
        remainingSlots -= spentSlots;
      }
    }

    // Shared slack pool: every not-yet-selected candidate, pure score order.
    take(ranked, remaining, remainingSlots);

    selected.sort((a, b) => (a.created_at ?? 0) - (b.created_at ?? 0));
    return selected.map((t) => ({ role: t.role, content: stripThinkingTokens(t.content) }));
  }

  _est(msgs, cpt) {
    return msgs.reduce((s, m) => s + this._estOne(m.content, cpt), 0);
  }

  /**
   * Token-cost estimate for ONE message's content, used by the budget math
   * in select() (systemTokens/recencyTokens -> how much room is left for
   * rotating memory slots).
   *
   * Bug this fixes (found 2026-07-23 auditing vision-routing support, see
   * Pleiades' docs/specs/2026-07-23-vision-routing-design.md step 4a): a
   * vision-capable client (Pleiades' engine.py, once a character's assigned
   * model can take image input) sends the current turn's `content` as an
   * OpenAI content-parts ARRAY, e.g.
   *   [{type:'text', text:'...'}, {type:'image_url', image_url:{url:'data:
   *   image/png;base64,<huge>'}}]
   * The old `m.content?.length` blindly used JS's `.length` regardless of
   * type. For a string that's the char count (correct); for an ARRAY,
   * `.length` is the number of array ELEMENTS (typically 1-3) -- a message
   * carrying a full image was silently costed at ~1 estimated token instead
   * of the real, substantial cost an image actually represents. That let
   * select() believe far more budget was free than truly was, over-filling
   * rotating-memory slots and risking the assembled request overflowing the
   * upstream server's real n_ctx once the image's actual token cost lands
   * server-side -- exactly the kind of silent breakage this project's
   * vision-routing work was told to explicitly verify for, not assume away.
   *
   * Does NOT crash either way (JS never throws on `.length` for an array),
   * which is also worth recording: the failure mode here was silent
   * misestimation, not an exception.
   */
  _estOne(content, cpt) {
    if (typeof content === 'string') return Math.ceil(content.length / cpt);
    if (Array.isArray(content)) {
      let chars = 0;
      let imageParts = 0;
      for (const part of content) {
        if (part && typeof part.text === 'string') chars += part.text.length;
        else if (part?.type === 'image_url' || part?.image_url) imageParts += 1;
      }
      // No exact way to know an image's real post-encoder token cost from
      // here (varies by tiling/resolution/model) -- IMAGE_TOKEN_ESTIMATE is
      // a conservative flat per-image guess (roughly a single-tile budget
      // for common small vision encoders). Costing a little rotating-memory
      // headroom on every vision turn is a much safer failure mode than the
      // previous ~1-token estimate, which cost none at all.
      return Math.ceil(chars / cpt) + imageParts * IMAGE_TOKEN_ESTIMATE;
    }
    return 0;
  }
}

module.exports = Selector;
module.exports._internals = { meanVec, normalize, blend, FORESIGHT_TTL_DAYS };
