'use strict';

/**
 * lib/logger.js — small, dependency-free logger.
 *
 * Always appends to ~/.anamnesis/daemon.log so logs survive even when the
 * daemon is started with stdio:ignore (detached on Windows via Task Scheduler,
 * or as a systemd service). Also writes to stdout when it is writable.
 *
 * Levels: error | warn | info | debug
 * Set ANAMNESIS_LOG=debug for verbose output, ANAMNESIS_LOG=warn to quiet down.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 };
const ACTIVE = LEVELS[(process.env.ANAMNESIS_LOG ?? 'info').toLowerCase()] ?? LEVELS.info;

const LOG_PATH = path.join(os.homedir(), '.anamnesis', 'daemon.log');
try { fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true }); } catch {}

// Rotate log if >5 MB so it doesn't grow forever
try {
  if (fs.existsSync(LOG_PATH) && fs.statSync(LOG_PATH).size > 5 * 1024 * 1024) {
    fs.renameSync(LOG_PATH, LOG_PATH + '.old');
  }
} catch {}

function writeLine(line) {
  try { fs.appendFileSync(LOG_PATH, line + '\n'); } catch {}
  try { if (process.stdout.writable) process.stdout.write(line + '\n'); } catch {}
}

function make(tag) {
  const prefix = `[${tag}]`;
  return {
    error: (...args) => { if (ACTIVE >= LEVELS.error) writeLine(prefix + ' ' + args.join(' ')); },
    warn:  (...args) => { if (ACTIVE >= LEVELS.warn)  writeLine(prefix + ' ' + args.join(' ')); },
    info:  (...args) => { if (ACTIVE >= LEVELS.info)  writeLine(prefix + ' ' + args.join(' ')); },
    debug: (...args) => { if (ACTIVE >= LEVELS.debug) writeLine(prefix + ' ' + args.join(' ')); },
  };
}

module.exports = { make, LEVELS, ACTIVE };
