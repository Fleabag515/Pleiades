// test/lib/ports.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const net = require('net');
const { isPortFree, findFreePort } = require('../../src/lib/ports.js');

function occupy(port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(port, '127.0.0.1', () => resolve(server));
    server.on('error', reject);
  });
}

test('isPortFree returns true for an unbound port', async () => {
  const free = await isPortFree(19999);
  assert.equal(free, true);
});

test('isPortFree returns false for an occupied port', async () => {
  const server = await occupy(19998);
  try {
    const free = await isPortFree(19998);
    assert.equal(free, false);
  } finally {
    await new Promise((r) => server.close(r));
  }
});

test('findFreePort returns requested port when free', async () => {
  const port = await findFreePort(19997, new Set());
  assert.equal(port, 19997);
});

test('findFreePort skips occupied ports', async () => {
  const server = await occupy(19996);
  try {
    const port = await findFreePort(19996, new Set());
    assert.ok(port > 19996);
  } finally {
    await new Promise((r) => server.close(r));
  }
});

test('findFreePort skips ports in reserved set', async () => {
  const port = await findFreePort(19995, new Set([19995, 19994]));
  assert.ok(port !== 19995 && port !== 19994);
});

test('findFreePort skips multiple consecutive reserved ports', async () => {
  const port = await findFreePort(19990, new Set([19990, 19991, 19992]));
  assert.ok(port >= 19993);
});

const { suggestPort } = require('../../src/lib/ports.js');
const Registry = require('../../src/lib/registry.js');
const fs = require('fs');
const path = require('path');
const os = require('os');

test('suggestPort returns port not in registry', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'anam-ports-'));
  const reg = new Registry(path.join(dir, 'registry.json'));
  reg.add({ name: 'mark', port: 19985, active: false });
  const port = await suggestPort(reg);
  assert.notEqual(port, 19985);
});
