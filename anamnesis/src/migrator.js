'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');

const ANAMNESIS_DIR = path.join(os.homedir(), '.anamnesis');
const REGISTRY_PATH = path.join(ANAMNESIS_DIR, 'registry.json');
const OLD_DB_PATH = path.join(ANAMNESIS_DIR, 'history.db');

function findOldConfig() {
  const candidates = [
    path.join(os.homedir(), 'anamnesis', 'config.json'),
    path.join('/opt', 'anamnesis', 'config.json'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return { path: c, config: JSON.parse(fs.readFileSync(c, 'utf8')) };
  }
  return null;
}

function needsMigration() {
  if (fs.existsSync(REGISTRY_PATH)) return false;
  if (fs.existsSync(OLD_DB_PATH)) return true;
  const old = findOldConfig();
  return !!old;
}

function migrate(log) {
  const old = findOldConfig();
  const characterName = 'default';
  const charDir = path.join(ANAMNESIS_DIR, 'characters', characterName);

  fs.mkdirSync(charDir, { recursive: true });
  log.info('migrating v0.4 layout to v0.5...');

  if (fs.existsSync(OLD_DB_PATH)) {
    const newDbPath = path.join(charDir, 'history.db');
    fs.renameSync(OLD_DB_PATH, newDbPath);
    log.info(`moved history.db → characters/${characterName}/history.db`);
  }

  let charConfig;
  if (old) {
    charConfig = { ...old.config };
    charConfig.history = { ...charConfig.history, dbPath: path.join(charDir, 'history.db') };
  } else {
    charConfig = {
      proxy: { port: 8084, host: '127.0.0.1' },
      upstream: { baseUrl: 'http://127.0.0.1:8083/v1', apiKey: '', disableThinking: true },
      history: { dbPath: path.join(charDir, 'history.db'), maxAgeDays: 90 },
    };
  }

  fs.writeFileSync(path.join(charDir, 'config.json'), JSON.stringify(charConfig, null, 2));

  const registry = {
    characters: [{ name: characterName, port: charConfig.proxy?.port || 8084, active: false }],
  };
  fs.writeFileSync(REGISTRY_PATH, JSON.stringify(registry, null, 2));

  // Disable legacy systemd unit ONLY if it points to the old proxy.js (not the new daemon.js)
  try {
    const { execSync } = require('child_process');
    const unitOut = execSync('systemctl cat anamnesis 2>/dev/null || true', {
      shell: true,
      encoding: 'utf8',
    });
    const isLegacy = unitOut.includes('proxy.js') && !unitOut.includes('daemon.js');
    if (isLegacy) {
      execSync(
        'systemctl stop anamnesis 2>/dev/null; systemctl disable anamnesis 2>/dev/null; true',
        { shell: true, stdio: 'pipe' }
      );
      log.info('disabled legacy anamnesis.service (was pointing to proxy.js)');
    }
  } catch {
    /* not on systemd or already removed */
  }

  log.info(`migration complete — character '${characterName}' created from v0.4 data`);
  log.info(`start it with: anamnesis start ${characterName}`);
}

module.exports = { needsMigration, migrate };
