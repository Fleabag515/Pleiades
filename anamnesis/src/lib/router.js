// src/lib/router.js
'use strict';

class Router {
  constructor() {
    this._routes = [];
  }

  _add(method, path, handler) {
    const keys = [];
    // Escape static segments per-slash-separated-part, substitute :param
    const src = path
      .split('/')
      .map((seg) => {
        if (seg.startsWith(':')) {
          keys.push(seg.slice(1));
          return '([^/]+)';
        }
        return seg.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      })
      .join('/');
    this._routes.push({ method, re: new RegExp(`^${src}$`), keys, handler });
  }

  get(path, handler) {
    this._add('GET', path, handler);
  }
  post(path, handler) {
    this._add('POST', path, handler);
  }
  delete(path, handler) {
    this._add('DELETE', path, handler);
  }
  patch(path, handler) {
    this._add('PATCH', path, handler);
  }

  async handle(req, res) {
    const url = new URL(req.url, 'http://localhost');
    for (const route of this._routes) {
      if (route.method !== req.method) continue;
      const m = url.pathname.match(route.re);
      if (!m) continue;
      req.params = Object.fromEntries(route.keys.map((k, i) => [k, m[i + 1]]));
      req.query = Object.fromEntries(url.searchParams);
      try {
        await route.handler(req, res);
      } catch {
        if (!res.headersSent) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'internal server error' }));
        }
      }
      return;
    }
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'not found' }));
  }
}

module.exports = { Router };
