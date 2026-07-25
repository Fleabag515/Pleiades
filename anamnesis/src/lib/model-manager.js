'use strict';

/**
 * lib/model-manager.js — GGUF model download, cache, and integrity verification.
 *
 * Downloads Qwen2.5-1.5B-Instruct-Q4_K_M.gguf to ~/.anamnesis/models/ on first run.
 * Supports HTTP Range resume for interrupted downloads. Verifies SHA256 on completion.
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const log = require('./logger.js').make('model-manager');

const GGUF_URL =
  'https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf';
const GGUF_FILENAME = 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf';
// Verified against HuggingFace CDN x-linked-size and x-linked-etag headers.
// Update both constants if the upstream file is replaced.
const GGUF_EXPECTED_SIZE = 986048768;
const GGUF_SHA256 = '1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370';

const DEFAULT_CACHE_DIR = path.join(os.homedir(), '.anamnesis', 'models');
const DEFAULT_RETRY_DELAYS_MS = [1000, 4000, 16000];

class ModelManager {
  /**
   * @param {object} opts
   * @param {string} opts.cacheDir — override default cache location (test DI)
   */
  constructor({ cacheDir = DEFAULT_CACHE_DIR } = {}) {
    this.cacheDir = cacheDir;
    this.filename = GGUF_FILENAME;
    this._GGUF_URL = GGUF_URL;
    this._EXPECTED_SIZE = GGUF_EXPECTED_SIZE;
    this._SHA256 = GGUF_SHA256;
    this._RETRY_DELAYS_MS = DEFAULT_RETRY_DELAYS_MS;
  }

  modelPath() {
    return path.join(this.cacheDir, this.filename);
  }

  async ensureModel() {
    fs.mkdirSync(this.cacheDir, { recursive: true });
    const dest = this.modelPath();

    if (fs.existsSync(dest)) {
      const stat = fs.statSync(dest);
      if (stat.size === this._EXPECTED_SIZE) {
        const ok = await this._verifyChecksum(dest);
        if (ok) {
          log.info(`model ready: ${dest}`);
          return dest;
        }
        log.warn('checksum mismatch on cached file — re-downloading');
        fs.unlinkSync(dest);
      }
    }

    await this._download();
    return dest;
  }

  /** Overridable for tests */
  async _verifyChecksum(filePath) {
    return new Promise((resolve, reject) => {
      const hash = crypto.createHash('sha256');
      const stream = fs.createReadStream(filePath);
      stream.on('data', (d) => hash.update(d));
      stream.on('end', () => resolve(hash.digest('hex') === this._SHA256));
      stream.on('error', reject);
    });
  }

  /**
   * Overridable for tests. In production, subclass overrides this to stream to disk.
   * Returns { statusCode, body: Buffer }.
   */
  async _doHttpRequest(url, headers = {}) {
    return new Promise((resolve, reject) => {
      const u = new URL(url);
      const lib = u.protocol === 'https:' ? https : http;
      const req = lib.get({ hostname: u.hostname, path: u.pathname + u.search, headers }, (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ statusCode: res.statusCode, body: Buffer.concat(chunks) }));
      });
      req.on('error', reject);
    });
  }

  async _download() {
    const dest = this.modelPath();
    const delays = this._RETRY_DELAYS_MS;
    let lastErr;

    for (let attempt = 0; attempt <= delays.length; attempt++) {
      try {
        await this._downloadOnce(dest);
        const ok = await this._verifyChecksum(dest);
        if (!ok) {
          log.warn('SHA256 mismatch after download — retrying from scratch');
          if (fs.existsSync(dest)) fs.unlinkSync(dest);
          lastErr = new Error('SHA256 verification failed');
          if (attempt < delays.length) {
            await new Promise((r) => setTimeout(r, delays[attempt]));
          }
          continue;
        }
        log.info(`model downloaded and verified: ${dest}`);
        return;
      } catch (err) {
        lastErr = err;
        if (attempt < delays.length) {
          const delay = delays[attempt];
          log.warn(
            `download attempt ${attempt + 1} failed: ${err.message} — retrying in ${delay}ms`
          );
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    }
    throw lastErr;
  }

  async _downloadOnce(dest) {
    fs.mkdirSync(path.dirname(dest), { recursive: true });

    let existingBytes = 0;
    if (fs.existsSync(dest)) {
      existingBytes = fs.statSync(dest).size;
    }

    const headers = existingBytes > 0 ? { Range: `bytes=${existingBytes}-` } : {};
    const { statusCode, body } = await this._doHttpRequest(this._GGUF_URL, headers);

    // Server returned 200 on range request → doesn't support resume; start over
    if (existingBytes > 0 && statusCode === 200) {
      log.info('server returned 200 for range request — restarting from byte 0');
      existingBytes = 0;
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
    }

    if (statusCode !== 200 && statusCode !== 206) {
      throw new Error(`unexpected HTTP ${statusCode} from HuggingFace`);
    }

    const flag = existingBytes > 0 ? 'a' : 'w';
    fs.writeFileSync(dest, body, { flag });

    const finalSize = fs.existsSync(dest) ? fs.statSync(dest).size : 0;
    const pct = Math.round((finalSize / this._EXPECTED_SIZE) * 100);
    log.info(`download progress: ${pct}% (${finalSize}/${this._EXPECTED_SIZE} bytes)`);
  }
}

/**
 * Production variant — streams download to disk instead of buffering.
 * Follows HTTP redirects (HuggingFace CDN redirects to storage).
 */
class ProductionModelManager extends ModelManager {
  async _doHttpRequest(url, headers = {}) {
    return new Promise((resolve, reject) => {
      const u = new URL(url);
      const req = https.get(
        { hostname: u.hostname, path: u.pathname + u.search, headers },
        (res) => {
          // Follow redirects
          if ([301, 302, 307, 308].includes(res.statusCode) && res.headers.location) {
            resolve(this._doHttpRequest(res.headers.location, headers));
            return;
          }
          resolve({ statusCode: res.statusCode, _stream: res });
        }
      );
      req.on('error', reject);
    });
  }

  async _downloadOnce(dest) {
    fs.mkdirSync(path.dirname(dest), { recursive: true });

    let existingBytes = 0;
    if (fs.existsSync(dest)) {
      existingBytes = fs.statSync(dest).size;
    }

    const headers = existingBytes > 0 ? { Range: `bytes=${existingBytes}-` } : {};
    const { statusCode, _stream } = await this._doHttpRequest(this._GGUF_URL, headers);

    if (existingBytes > 0 && statusCode === 200) {
      log.info('server returned 200 for range request — restarting from byte 0');
      existingBytes = 0;
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
    }

    if (statusCode !== 200 && statusCode !== 206) {
      _stream.resume(); // drain
      throw new Error(`unexpected HTTP ${statusCode}`);
    }

    const flag = existingBytes > 0 ? 'a' : 'w';
    const fileStream = fs.createWriteStream(dest, { flags: flag });
    let received = existingBytes;
    let lastLoggedPct = Math.floor((existingBytes / this._EXPECTED_SIZE) * 10) * 10;

    await new Promise((resolve, reject) => {
      _stream.on('data', (chunk) => {
        received += chunk.length;
        const pct = Math.floor((received / this._EXPECTED_SIZE) * 100);
        const bucket = Math.floor(pct / 10) * 10;
        if (bucket > lastLoggedPct) {
          lastLoggedPct = bucket;
          log.info(`download: ${pct}% (${received}/${this._EXPECTED_SIZE} bytes)`);
        }
      });
      _stream.pipe(fileStream);
      fileStream.on('finish', resolve);
      fileStream.on('error', reject);
      _stream.on('error', reject);
    });
  }
}

module.exports = { ModelManager, ProductionModelManager, GGUF_FILENAME, DEFAULT_CACHE_DIR };
