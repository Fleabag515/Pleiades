'use strict';

/**
 * node:sqlite's DatabaseSync has no `.transaction()` sugar the way
 * better-sqlite3 did -- this is the manual equivalent: BEGIN, run fn(),
 * COMMIT; roll back and rethrow on any failure so a partial multi-statement
 * write never lands half-applied. Used anywhere multiple writes must
 * succeed or fail together -- see history.js's mergeSessions() (repoints
 * session_key across five tables; a partial failure there would split a
 * character's memory across two keys) and updateDecayScores(), and
 * importers/index.js's bulk import.
 */
function runInTransaction(db, fn) {
  db.exec('BEGIN');
  try {
    const result = fn();
    db.exec('COMMIT');
    return result;
  } catch (err) {
    try {
      db.exec('ROLLBACK');
    } catch {
      // The connection itself is in a bad state if ROLLBACK fails too --
      // surface the original error, not this secondary one.
    }
    throw err;
  }
}

module.exports = { runInTransaction };
