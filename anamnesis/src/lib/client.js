'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const DAEMON_CONFIG = path.join(os.homedir(), '.anamnesis', 'daemon.json');

function getControlPort() {
  try {
    return JSON.parse(fs.readFileSync(DAEMON_CONFIG, 'utf8')).controlPort || 9000;
  } catch {
    return 9000;
  }
}

function request(method, urlPath, body) {
  return new Promise((resolve, reject) => {
    const port = getControlPort();
    const payload = body ? JSON.stringify(body) : null;
    const options = {
      hostname: '127.0.0.1',
      port,
      path: urlPath,
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(payload ? { 'Content-Length': Buffer.byteLength(payload) } : {}),
      },
    };
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (d) => {
        data += d;
      });
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, body: data });
        }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

const client = {
  get: (p) => request('GET', p),
  post: (p, body) => request('POST', p, body),
  delete: (p) => request('DELETE', p),

  status: () => client.get('/status'),
  listCharacters: () => client.get('/characters'),
  getCharacter: (name) => client.get(`/characters/${name}`),
  startCharacter: (name) => client.post(`/characters/${name}/start`),
  stopCharacter: (name) => client.post(`/characters/${name}/stop`),
  createCharacter: (name, config) => client.post('/characters', { name, config }),
  deleteCharacter: (name) => client.delete(`/characters/${name}`),
};

module.exports = client;
