// test/lib/router.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const { Router } = require('../../src/lib/router.js');

function fakeReq(method, url) {
  return { method, url };
}

function fakeRes() {
  const res = { headers: {}, body: null, status: null };
  res.writeHead = (s, h) => {
    res.status = s;
    Object.assign(res.headers, h || {});
  };
  res.end = (b) => {
    res.body = b;
  };
  return res;
}

test('routes GET /characters to handler', async () => {
  const router = new Router();
  router.get('/characters', (req, res) => res.end('ok'));
  const res = fakeRes();
  await router.handle(fakeReq('GET', '/characters'), res);
  assert.equal(res.body, 'ok');
});

test('extracts :name param', async () => {
  const router = new Router();
  router.get('/characters/:name', (req, res) => res.end(req.params.name));
  const res = fakeRes();
  await router.handle(fakeReq('GET', '/characters/mark'), res);
  assert.equal(res.body, 'mark');
});

test('404 on unmatched route', async () => {
  const router = new Router();
  const res = fakeRes();
  await router.handle(fakeReq('GET', '/nope'), res);
  assert.equal(res.status, 404);
});

test('routes POST separately from GET', async () => {
  const router = new Router();
  router.get('/x', (req, res) => res.end('get'));
  router.post('/x', (req, res) => res.end('post'));
  const res = fakeRes();
  await router.handle(fakeReq('POST', '/x'), res);
  assert.equal(res.body, 'post');
});

test('routes DELETE', async () => {
  const router = new Router();
  router.delete('/characters/:name', (req, res) => res.end(req.params.name));
  const res = fakeRes();
  await router.handle(fakeReq('DELETE', '/characters/mark'), res);
  assert.equal(res.body, 'mark');
});

test('populates req.query from URL query string', async () => {
  const router = new Router();
  router.get('/status', (req, res) => res.end(req.query.active));
  const res = fakeRes();
  await router.handle(fakeReq('GET', '/status?active=true'), res);
  assert.equal(res.body, 'true');
});
