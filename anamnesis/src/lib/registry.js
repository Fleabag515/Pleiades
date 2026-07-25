// src/lib/registry.js
'use strict';

const fs = require('fs');
const path = require('path');

// Always use os.homedir() — process.env.HOME is unreliable on Windows
// (Git Bash sets HOME to a Unix-style path that differs from the actual homedir).
const DEFAULT_PATH = path.join(require('os').homedir(), '.anamnesis', 'registry.json');

class Registry {
  constructor(filePath = DEFAULT_PATH) {
    this._path = filePath;
    this._data = this._load();
  }

  _load() {
    try {
      return JSON.parse(fs.readFileSync(this._path, 'utf8'));
    } catch {
      return { characters: [] };
    }
  }

  _save() {
    fs.mkdirSync(path.dirname(this._path), { recursive: true });
    fs.writeFileSync(this._path, JSON.stringify(this._data, null, 2), 'utf8');
  }

  list() {
    return this._data.characters;
  }
  get(name) {
    return this._data.characters.find((c) => c.name === name);
  }

  add(entry) {
    if (this.get(entry.name)) throw new Error(`character '${entry.name}' already exists`);
    this._data.characters.push(entry);
    this._save();
  }

  setActive(name, active) {
    const c = this.get(name);
    if (!c) throw new Error(`character '${name}' not found`);
    c.active = active;
    this._save();
  }

  updatePort(name, port) {
    const c = this.get(name);
    if (!c) throw new Error(`character '${name}' not found`);
    c.port = port;
    this._save();
  }

  update(name, fields) {
    const c = this.get(name);
    if (!c) throw new Error(`character '${name}' not found`);
    Object.assign(c, fields);
    this._save();
  }

  remove(name) {
    this._data.characters = this._data.characters.filter((c) => c.name !== name);
    this._save();
  }

  usedPorts() {
    return new Set(this._data.characters.map((c) => c.port));
  }
}

module.exports = Registry;
