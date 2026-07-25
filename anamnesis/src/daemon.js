#!/usr/bin/env node
'use strict';

const path = require('path');
const os = require('os');
const fs = require('fs');

const Registry = require('./lib/registry.js');
const pid = require('./lib/pid.js');
const CharacterManager = require('./character-manager.js');
const { createControlServer } = require('./control-server.js');
const log = require('./lib/logger.js').make('daemon');

const ANAMNESIS_DIR = path.join(os.homedir(), '.anamnesis');
const DAEMON_CONFIG = path.join(ANAMNESIS_DIR, 'daemon.json');
const REGISTRY_PATH = path.join(ANAMNESIS_DIR, 'registry.json');

function loadDaemonConfig() {
  try {
    return JSON.parse(fs.readFileSync(DAEMON_CONFIG, 'utf8'));
  } catch {
    return {};
  }
}

async function main() {
  fs.mkdirSync(ANAMNESIS_DIR, { recursive: true });

  // v0.4 → v0.5 migration (if migrator exists and migration is needed)
  try {
    const { needsMigration, migrate } = require('./migrator.js');
    if (needsMigration()) migrate(log);
  } catch {
    /* migrator not yet implemented — safe to skip */
  }

  const daemonCfg = loadDaemonConfig();
  const controlPort = daemonCfg.controlPort || 9000;
  const controlHost = '127.0.0.1';
  const startedAt = Date.now();

  const registry = new Registry(REGISTRY_PATH);
  const manager = new CharacterManager(registry, ANAMNESIS_DIR, controlPort);
  const controlServer = createControlServer(manager, startedAt);

  await new Promise((resolve, reject) => {
    controlServer.listen(controlPort, controlHost, resolve);
    controlServer.once('error', reject);
  });

  pid.write();
  log.info(`daemon listening on ${controlHost}:${controlPort}`);

  await manager.reactivateAll();

  async function shutdown(signal) {
    log.info(`received ${signal}, shutting down...`);
    await manager.stopAll();
    await new Promise((resolve) => controlServer.close(resolve));
    pid.remove();
    log.info('shutdown complete');
    process.exit(0);
  }

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

main().catch((e) => {
  console.error('[daemon] fatal:', e.message);
  process.exit(1);
});
