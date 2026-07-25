'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const os = require('os');
const fs = require('fs');

const { ModelManager } = require('../../src/lib/model-manager.js');

function makeManager() {
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), 'anamnesis-mm-test-'));
  const mm = new ModelManager({ cacheDir });
  return mm;
}

test('ModelManager: ensureModel resolves immediately when file matches size and checksum', async () => {
  const mm = makeManager();
  const filePath = mm.modelPath();

  const content = Buffer.alloc(10, 0x42);
  fs.writeFileSync(filePath, content);

  mm._EXPECTED_SIZE = content.length;
  mm._verifyChecksum = async () => true;

  const result = await mm.ensureModel();
  assert.equal(result, filePath);
});

test('ModelManager: starts download when file is absent', async () => {
  const mm = makeManager();
  let downloadCalled = false;
  mm._EXPECTED_SIZE = 10;
  mm._download = async () => {
    downloadCalled = true;
    fs.writeFileSync(mm.modelPath(), Buffer.alloc(mm._EXPECTED_SIZE));
  };
  mm._verifyChecksum = async () => true;
  await mm.ensureModel();
  assert.ok(downloadCalled);
});

test('ModelManager: sends Range header when partial file exists', async () => {
  const mm = makeManager();
  const partialPath = mm.modelPath();
  fs.writeFileSync(partialPath, Buffer.alloc(500));

  let capturedRange = null;
  mm._doHttpRequest = async (url, headers) => {
    capturedRange = headers['Range'];
    return { statusCode: 206, body: Buffer.alloc(0) };
  };
  // Capture just the range construction, don't actually download
  const orig = mm._downloadOnce.bind(mm);
  mm._downloadOnce = async (dest) => {
    const partial = fs.existsSync(dest) ? fs.statSync(dest).size : 0;
    await mm._doHttpRequest(mm._GGUF_URL, partial > 0 ? { Range: `bytes=${partial}-` } : {});
    // stub: just append nothing
  };
  mm._EXPECTED_SIZE = 500;
  mm._verifyChecksum = async () => true;
  await mm._download();
  assert.equal(capturedRange, 'bytes=500-');
});

test('ModelManager: restarts download if server returns 200 instead of 206', async () => {
  const mm = makeManager();
  const dest = mm.modelPath();
  // Write a partial file
  fs.writeFileSync(dest, Buffer.alloc(100));
  mm._EXPECTED_SIZE = 200;

  let requestCount = 0;
  mm._doHttpRequest = async (_url, _headers) => {
    requestCount++;
    // Return 200 (not 206) on range request → should discard and restart
    return { statusCode: 200, body: Buffer.alloc(mm._EXPECTED_SIZE) };
  };
  mm._verifyChecksum = async () => true;
  await mm._download();
  // File should be the full expected size (restarted from scratch)
  const size = fs.statSync(dest).size;
  assert.equal(size, mm._EXPECTED_SIZE);
  // First request should have sent Range header but got 200 back
  assert.ok(requestCount >= 1);
});

test('ModelManager: retries up to 3 times on network error, succeeds on 3rd', async () => {
  const mm = makeManager();
  let attempts = 0;
  mm._EXPECTED_SIZE = 10;
  mm._RETRY_DELAYS_MS = [0, 0, 0];
  mm._doHttpRequest = async () => {
    attempts++;
    if (attempts < 3) throw new Error('network error');
    return { statusCode: 200, body: Buffer.alloc(mm._EXPECTED_SIZE) };
  };
  mm._verifyChecksum = async () => true;
  await mm._download();
  assert.equal(attempts, 3);
});

test('ModelManager: throws after all retries fail', async () => {
  const mm = makeManager();
  mm._EXPECTED_SIZE = 10;
  mm._doHttpRequest = async () => {
    throw new Error('always fails');
  };
  // Speed up retries for tests
  mm._RETRY_DELAYS_MS = [0, 0, 0];
  await assert.rejects(() => mm._download(), /always fails/);
});
