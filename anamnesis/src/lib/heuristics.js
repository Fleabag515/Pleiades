/**
 * lib/heuristics.js — pre-LLM filters for which turns are worth processing.
 *
 * Used by both the engram extractor and the foresight scanner so they share
 * one definition of "worth an Ollama call".
 */

/**
 * Decide whether a turn's text is worth sending to an LLM for extraction.
 * Cheap-and-strict: we'd rather miss a few facts than waste a 45s timeout
 * window on a "thanks!" or an XML payload.
 */
function shouldProcessTurn(content) {
  if (!content || content.length < 80) return false;
  // Looks like a raw XML/HTML payload (e.g. tool result echoed back) — skip.
  if (content.startsWith('<') && content.includes('</')) return false;
  // One-liner under 200 chars — probably an acknowledgement.
  if (content.split('\n').length < 2 && content.length < 200) return false;
  return true;
}

module.exports = { shouldProcessTurn };
