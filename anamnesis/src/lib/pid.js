// src/lib/pid.js
'use strict';

const fs = require('fs');
const path = require('path');

const DEFAULT_PATH = path.join(
  process.env.HOME || require('os').homedir(),
  '.anamnesis',
  'daemon.pid'
);

function write(pidPath = DEFAULT_PATH) {
  fs.mkdirSync(path.dirname(pidPath), { recursive: true });
  fs.writeFileSync(pidPath, String(process.pid), 'utf8');
}

function read(pidPath = DEFAULT_PATH) {
  try {
    const pid = parseInt(fs.readFileSync(pidPath, 'utf8'), 10);
    return isNaN(pid) ? null : pid;
  } catch {
    return null;
  }
}

function remove(pidPath = DEFAULT_PATH) {
  try {
    fs.unlinkSync(pidPath);
  } catch {
    /* already gone */
  }
}

function isRunning(pidPath = DEFAULT_PATH) {
  const pid = read(pidPath);
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === 'EPERM';
  }
}

module.exports = { write, read, remove, isRunning, DEFAULT_PATH };
