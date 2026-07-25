'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');

const { start: startProxy } = require('./proxy.js');
const { findFreePort } = require('./lib/ports.js');
const log = require('./lib/logger.js').make('manager');

const NAME_RE = /^[a-z0-9_-]+$/i;
const DEFAULT_BASE = path.join(os.homedir(), '.anamnesis');

function deepMerge(target, src) {
  for (const k of Object.keys(src || {})) {
    const v = src[k];
    if (
      v && typeof v === 'object' && !Array.isArray(v) &&
      target[k] && typeof target[k] === 'object' && !Array.isArray(target[k])
    ) {
      deepMerge(target[k], v);
    } else {
      target[k] = v;
    }
  }
  return target;
}

class CharacterManager {
  constructor(registry, baseDir = DEFAULT_BASE, controlPort = 9000) {
    this._registry = registry;
    this._baseDir = baseDir;
    this._controlPort = controlPort;
    this._running = new Map();
  }

  _characterDir(name) {
    return path.join(this._baseDir, 'characters', name);
  }

  _configPath(name) {
    return path.join(this._characterDir(name), 'config.json');
  }

  _loadConfig(name) {
    return JSON.parse(fs.readFileSync(this._configPath(name), 'utf8'));
  }

  _saveConfig(name, config) {
    const dir = this._characterDir(name);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this._configPath(name), JSON.stringify(config, null, 2), 'utf8');
  }

  createCharacter(name, config) {
    if (!NAME_RE.test(name))
      throw new Error(`invalid name '${name}' — use letters, digits, hyphens, underscores only`);
    if (this._registry.get(name)) throw new Error(`character '${name}' already exists`);
    this._saveConfig(name, config);
    this._registry.add({ name, port: config.proxy.port, active: false });
    log.info(`created character '${name}' on port ${config.proxy.port}`);
  }

  async startCharacter(name) {
    if (this._running.has(name)) return this._registry.get(name).port;
    const entry = this._registry.get(name);
    if (!entry) throw new Error(`character '${name}' not found`);

    const config = this._loadConfig(name);
    const reserved = this._registry.usedPorts();
    reserved.delete(entry.port);
    reserved.add(this._controlPort);

    const freePort = await findFreePort(entry.port, reserved);
    if (freePort !== entry.port) {
      log.info(`port ${entry.port} in use — starting '${name}' on ${freePort} instead`);
      config.proxy.port = freePort;
      this._registry.updatePort(name, freePort);
      this._saveConfig(name, config);
    }

    const instance = await startProxy(config);
    this._running.set(name, instance);
    this._registry.setActive(name, true);
    log.info(`started '${name}' on port ${freePort}`);
    return freePort;
  }

  async stopCharacter(name) {
    const instance = this._running.get(name);
    if (!instance) return;
    await instance.shutdown('stop');
    this._running.delete(name);
    this._registry.setActive(name, false);
    log.info(`stopped '${name}'`);
  }

  deleteCharacter(name) {
    if (this._running.has(name)) throw new Error(`stop '${name}' before deleting`);
    const dir = this._characterDir(name);
    fs.rmSync(dir, { recursive: true, force: true });
    this._registry.remove(name);
    log.info(`deleted character '${name}'`);
  }

  listCharacters() {
    return this._registry.list().map((entry) => ({
      ...entry,
      running: this._running.has(entry.name),
    }));
  }

  getCharacter(name) {
    const entry = this._registry.get(name);
    if (!entry) return null;
    return { ...entry, running: this._running.has(name) };
  }

  // Deep-merge a partial config patch and persist it. Does NOT restart a
  // running proxy — the caller restarts to apply (matches CLI edit + the
  // Pleiades client). Returns the merged config.
  updateConfig(name, patch) {
    if (!this._registry.get(name)) throw new Error(`character '${name}' not found`);
    const config = this._loadConfig(name);
    deepMerge(config, patch || {});
    this._saveConfig(name, config);
    if (config.proxy && typeof config.proxy.port === 'number') {
      this._registry.updatePort(name, config.proxy.port);
    }
    log.info(`updated config for '${name}'`);
    return config;
  }

  isActive(name) {
    return this._running.has(name);
  }

  async reactivateAll() {
    for (const entry of this._registry.list()) {
      if (entry.active) {
        try {
          await this.startCharacter(entry.name);
        } catch (e) {
          log.warn(`failed to reactivate '${entry.name}':`, e.message);
        }
      }
    }
  }

  async stopAll() {
    for (const name of [...this._running.keys()]) {
      await this.stopCharacter(name);
    }
  }
}

module.exports = CharacterManager;
