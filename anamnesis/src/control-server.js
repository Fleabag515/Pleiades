// src/control-server.js
'use strict';

const http = require('http');
const { Router } = require('./lib/router.js');
const log = require('./lib/logger.js').make('control');

function json(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

async function readBody(req) {
  return new Promise((resolve) => {
    let buf = '';
    req.on('data', (d) => {
      buf += d;
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(buf));
      } catch {
        resolve({});
      }
    });
  });
}

function createControlServer(manager, daemonStartedAt) {
  const router = new Router();

  // GET /status
  router.get('/status', (req, res) => {
    json(res, 200, {
      status: 'ok',
      uptime: Math.floor((Date.now() - daemonStartedAt) / 1000),
      active: manager.listCharacters().filter((c) => c.running).length,
    });
  });

  // GET /characters
  router.get('/characters', (req, res) => {
    json(res, 200, { characters: manager.listCharacters() });
  });

  // GET /characters/:name
  router.get('/characters/:name', (req, res) => {
    const c = manager.getCharacter(req.params.name);
    if (!c) return json(res, 404, { error: `character '${req.params.name}' not found` });
    json(res, 200, c);
  });

  // POST /characters — create
  router.post('/characters', async (req, res) => {
    const body = await readBody(req);
    try {
      manager.createCharacter(body.name, body.config);
      json(res, 201, { name: body.name });
    } catch (e) {
      json(res, 400, { error: e.message });
    }
  });

  // PATCH /characters/:name — update config (persist; caller restarts to apply)
  router.patch('/characters/:name', async (req, res) => {
    const { name } = req.params;
    if (!manager.getCharacter(name))
      return json(res, 404, { error: `character '${name}' not found` });
    const body = await readBody(req);
    try {
      const config = manager.updateConfig(name, body.config || body);
      json(res, 200, { name, config });
    } catch (e) {
      json(res, 400, { error: e.message });
    }
  });

  // POST /characters/:name/start
  router.post('/characters/:name/start', async (req, res) => {
    const { name } = req.params;
    if (!manager.getCharacter(name))
      return json(res, 404, { error: `character '${name}' not found` });
    try {
      const port = await manager.startCharacter(name);
      json(res, 200, { name, port });
    } catch (e) {
      json(res, 500, { error: e.message });
    }
  });

  // POST /characters/:name/stop
  router.post('/characters/:name/stop', async (req, res) => {
    const { name } = req.params;
    if (!manager.getCharacter(name))
      return json(res, 404, { error: `character '${name}' not found` });
    try {
      await manager.stopCharacter(name);
      json(res, 200, { name, stopped: true });
    } catch (e) {
      json(res, 500, { error: e.message });
    }
  });

  // DELETE /characters/:name — auto-stops if active, then deletes
  router.delete('/characters/:name', async (req, res) => {
    const { name } = req.params;
    if (!manager.getCharacter(name))
      return json(res, 404, { error: `character '${name}' not found` });
    try {
      if (manager.isActive(name)) await manager.stopCharacter(name);
      manager.deleteCharacter(name);
      json(res, 200, { name, deleted: true });
    } catch (e) {
      json(res, 500, { error: e.message });
    }
  });

  const server = http.createServer(async (req, res) => {
    try {
      await router.handle(req, res);
    } catch (e) {
      log.error('control server error:', e.message);
      json(res, 500, { error: 'internal error' });
    }
  });

  return server;
}

module.exports = { createControlServer };
