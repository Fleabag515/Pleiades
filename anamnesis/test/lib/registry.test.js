// test/lib/registry.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Registry = require('../../src/lib/registry.js');

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'anamnesis-test-'));
}

test('load returns empty list when file absent', () => {
  const dir = tmpDir();
  const reg = new Registry(path.join(dir, 'registry.json'));
  assert.deepEqual(reg.list(), []);
});

test('add and persist a character', () => {
  const dir = tmpDir();
  const reg = new Registry(path.join(dir, 'registry.json'));
  reg.add({ name: 'mark', port: 8084, active: false });
  const reg2 = new Registry(path.join(dir, 'registry.json'));
  assert.equal(reg2.list().length, 1);
  assert.equal(reg2.list()[0].name, 'mark');
});

test('setActive updates active flag and persists', () => {
  const dir = tmpDir();
  const reg = new Registry(path.join(dir, 'registry.json'));
  reg.add({ name: 'mark', port: 8084, active: false });
  reg.setActive('mark', true);
  const reg2 = new Registry(path.join(dir, 'registry.json'));
  assert.equal(reg2.get('mark').active, true);
});

test('updatePort updates port and persists to disk', () => {
  const dir = tmpDir();
  const reg = new Registry(path.join(dir, 'registry.json'));
  reg.add({ name: 'mark', port: 8084, active: false });
  reg.updatePort('mark', 8085);
  assert.equal(reg.get('mark').port, 8085);
  const reg2 = new Registry(path.join(dir, 'registry.json'));
  assert.equal(reg2.get('mark').port, 8085);
});

test('remove deletes character and persists to disk', () => {
  const dir = tmpDir();
  const reg = new Registry(path.join(dir, 'registry.json'));
  reg.add({ name: 'mark', port: 8084, active: false });
  reg.remove('mark');
  assert.equal(reg.list().length, 0);
  const reg2 = new Registry(path.join(dir, 'registry.json'));
  assert.equal(reg2.list().length, 0);
});

test('get returns undefined for unknown name', () => {
  const dir = tmpDir();
  const reg = new Registry(path.join(dir, 'registry.json'));
  assert.equal(reg.get('nobody'), undefined);
});
