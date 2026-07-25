// src/lib/ports.js
'use strict';

const net = require('net');
const BASE_PORT = 8084;
const MAX_SCAN = 200;

function isPortFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.listen(port, '127.0.0.1', () => server.close(() => resolve(true)));
  });
}

async function findFreePort(preferred = BASE_PORT, reserved = new Set()) {
  for (let i = 0; i < MAX_SCAN; i++) {
    const candidate = preferred + i;
    if (reserved.has(candidate)) continue;
    if (await isPortFree(candidate)) return candidate;
  }
  throw new Error(
    `could not find a free port starting from ${preferred} after ${MAX_SCAN} attempts`
  );
}

async function suggestPort(registry) {
  const used = registry.usedPorts();
  return findFreePort(BASE_PORT, used);
}

module.exports = { isPortFree, findFreePort, suggestPort };
