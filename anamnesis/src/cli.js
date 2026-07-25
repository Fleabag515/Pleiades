#!/usr/bin/env node
'use strict';

const { execFileSync, spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');
const pid = require('./lib/pid.js');
const client = require('./lib/client.js');

const DAEMON_JS = path.join(__dirname, 'daemon.js');

// ─── Daemon auto-start ────────────────────────────────────────────────────────

async function ensureDaemon() {
  if (pid.isRunning()) return;

  const managed = isManagedService();
  if (managed) {
    console.error(
      'daemon not running — start it with: anamnesis\n  or check: systemctl status anamnesis'
    );
    process.exit(1);
  }

  const logPath = path.join(os.homedir(), '.anamnesis', 'daemon.log');
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  const logFd = fs.openSync(logPath, 'a');
  const child = spawn(process.execPath, [DAEMON_JS], {
    detached: true,
    stdio: ['ignore', logFd, logFd],
    env: { ...process.env, ANAMNESIS_LOG: process.env.ANAMNESIS_LOG || 'info' },
  });
  child.unref();
  fs.closeSync(logFd);

  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 200));
    try {
      const res = await client.status();
      if (res.status === 200) return;
    } catch {
      /* not ready yet */
    }
  }
  console.error('daemon failed to start — check ~/.anamnesis/daemon.log');
  process.exit(1);
}

function isManagedService() {
  try {
    execFileSync('systemctl', ['is-enabled', 'anamnesis'], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

// ─── Output helpers ───────────────────────────────────────────────────────────

function printCharacters(characters) {
  if (!characters.length) {
    console.log('No characters. Run: anamnesis new');
    return;
  }
  const W = { name: 4, port: 4 };
  for (const c of characters) W.name = Math.max(W.name, c.name.length);
  console.log(`${'NAME'.padEnd(W.name)}  ${'PORT'.padEnd(W.port)}  STATUS`);
  for (const c of characters) {
    console.log(
      `${c.name.padEnd(W.name)}  \x1b[1m${String(c.port).padEnd(W.port)}\x1b[0m  ${c.running ? '\x1b[32mactive\x1b[0m' : 'inactive'}`
    );
  }
}

function die(msg) {
  console.error(`error: ${msg}`);
  process.exit(1);
}

// ─── Commands ─────────────────────────────────────────────────────────────────

const commands = {
  async status() {
    await ensureDaemon();
    const r = await client.status();
    if (r.status !== 200) die('daemon not responding');
    const chars = (await client.listCharacters()).body.characters || [];
    console.log(`anamnesis daemon — uptime ${r.body.uptime}s, ${r.body.active} active`);
    printCharacters(chars);
  },

  async list() {
    await ensureDaemon();
    const r = await client.listCharacters();
    printCharacters(r.body.characters || []);
  },

  async start([name]) {
    if (!name) die('usage: anamnesis start <name>');
    await ensureDaemon();
    const r = await client.startCharacter(name);
    if (r.status !== 200) die(r.body.error || 'failed to start');
    console.log(`✓ ${name} running on http://127.0.0.1:${r.body.port}/v1`);
  },

  async stop([name]) {
    if (!name) die('usage: anamnesis stop <name>');
    await ensureDaemon();
    const r = await client.stopCharacter(name);
    if (r.status !== 200) die(r.body.error || 'failed to stop');
    console.log(`✓ ${name} stopped`);
  },

  async restart([name]) {
    if (name) {
      await commands.stop([name]);
      await commands.start([name]);
    } else {
      // No name — restart all currently active characters
      await ensureDaemon();
      const res = await client.listCharacters();
      const active = (res.body.characters || []).filter((c) => c.running);
      if (!active.length) {
        console.log('no active characters to restart');
        return;
      }
      for (const c of active) {
        await commands.stop([c.name]);
        await commands.start([c.name]);
      }
    }
  },

  async show([name]) {
    if (!name) die('usage: anamnesis show <name>');
    await ensureDaemon();
    const r = await client.getCharacter(name);
    if (r.status === 404) die(`character '${name}' not found`);
    const c = r.body;
    console.log(`name:   ${c.name}`);
    console.log(`port:   ${c.port}`);
    console.log(`status: ${c.running ? 'active' : 'inactive'}`);
  },

  async remove([name, ...rest]) {
    if (!name) die('usage: anamnesis remove <name>');
    const yes = rest.includes('--yes');
    await ensureDaemon();
    if (!yes) {
      const prompts = require('prompts');
      const { ok } = await prompts({
        type: 'confirm',
        name: 'ok',
        message: `Delete '${name}' and all its memories?`,
        initial: false,
      });
      if (!ok) {
        console.log('aborted');
        return;
      }
    }
    const r = await client.deleteCharacter(name);
    if (r.status !== 200) die(r.body.error || 'failed to delete');
    console.log(`✓ ${name} deleted`);
  },

  async logs([name]) {
    if (!name) die('usage: anamnesis logs <name>');
    console.log(`(showing daemon logs — filter for '${name}' manually)\n`);

    const logPath = path.join(os.homedir(), '.anamnesis', 'daemon.log');
    const isWindows = process.platform === 'win32';

    if (!isWindows) {
      // Linux/macOS: try journalctl first, fall back to log file
      try {
        const child = spawn('journalctl', ['-u', 'anamnesis', '-f', '--no-pager', '-n', '50'], {
          stdio: 'inherit',
        });
        child.on('error', () => tailFile(logPath));
        return;
      } catch {
        // fall through
      }
    }

    tailFile(logPath);

    function tailFile(p) {
      if (!fs.existsSync(p)) {
        console.log(`no log file found at ${p}`);
        console.log('start a character first: anamnesis start <name>');
        return;
      }
      // Print last 50 lines then watch for new content
      const stat = fs.statSync(p);
      let pos = Math.max(0, stat.size - 8000);
      const printFrom = (offset) => {
        const stream = fs.createReadStream(p, { start: offset });
        stream.pipe(process.stdout);
        stream.on('end', () => { pos = stat.size; });
      };
      printFrom(pos);
      fs.watchFile(p, { interval: 500 }, (curr) => {
        if (curr.size > pos) {
          fs.createReadStream(p, { start: pos }).pipe(process.stdout);
          pos = curr.size;
        }
      });
    }
  },

  async new(args) {
    await ensureDaemon();
    const wizard = require('./wizard.js');
    await wizard.run(args);
  },

  async edit([name, ...rest]) {
    if (!name) die('usage: anamnesis edit <name>');
    await ensureDaemon();
    const wizard = require('./wizard.js');
    await wizard.edit(name);
  },

  async import(args) {
    const importer = require('./importers/index.js');
    await importer.runCli(args);
  },

  async export([name]) {
    if (!name) die('usage: anamnesis export <name>');
    console.log('export not yet implemented');
  },

  async install() {
    const svc = require('./service.js');
    await svc.install();
  },

  async uninstall() {
    const svc = require('./service.js');
    await svc.uninstall();
  },

  async update() {
    const { execSync } = require('child_process');
    const isWindows = process.platform === 'win32';

    const INSTALL_DIR = isWindows
      ? path.join(
          process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local'),
          'anamnesis'
        )
      : path.join(os.homedir(), '.local', 'share', 'anamnesis');

    const isGitRepo = fs.existsSync(path.join(INSTALL_DIR, '.git'));

    if (isGitRepo) {
      // ── Git install (Linux / macOS / developer setup) ──────────────────────
      let current, remote;
      try {
        current = execSync('git rev-parse HEAD', { cwd: INSTALL_DIR, encoding: 'utf8' }).trim();
        remote = execSync('git ls-remote origin HEAD', { cwd: INSTALL_DIR, encoding: 'utf8' })
          .trim()
          .split('\t')[0];
      } catch {
        console.error('update failed: git error or no network access');
        process.exit(1);
      }

      if (current === remote) {
        console.log('anamnesis is up to date (' + current.slice(0, 7) + ')');
        return;
      }

      console.log('update available: ' + current.slice(0, 7) + ' → ' + remote.slice(0, 7));
      console.log('updating...');

      try {
        execSync('git pull --ff-only', { cwd: INSTALL_DIR, stdio: 'inherit' });
        execSync('npm install --omit=dev', { cwd: INSTALL_DIR, stdio: 'inherit' });
      } catch {
        console.error(
          'update failed — try manually: cd ~/.local/share/anamnesis && git pull && npm install --omit=dev'
        );
        process.exit(1);
      }
    } else if (isWindows) {
      // ── ZIP install (Windows installer) ────────────────────────────────────
      const REPO_URL = 'https://github.com/Fleabag515/anamnesis/archive/refs/heads/main.zip';
      const TmpZip = path.join(os.tmpdir(), 'anamnesis-update.zip');
      const TmpDir = path.join(os.tmpdir(), 'anamnesis-update');

      console.log('downloading update...');
      try {
        execSync(
          `powershell -Command "Invoke-WebRequest -Uri '${REPO_URL}' -OutFile '${TmpZip}' -UseBasicParsing"`,
          { stdio: 'inherit' }
        );
        execSync(
          `powershell -Command "if (Test-Path '${TmpDir}') { Remove-Item '${TmpDir}' -Recurse -Force }; Expand-Archive -Path '${TmpZip}' -DestinationPath '${TmpDir}' -Force"`,
          { stdio: 'inherit' }
        );
        execSync(
          `powershell -Command "Copy-Item '${TmpDir}\\anamnesis-main\\*' '${INSTALL_DIR}' -Recurse -Force"`,
          { stdio: 'inherit' }
        );
        execSync('npm install --omit=dev', { cwd: INSTALL_DIR, stdio: 'inherit' });
        execSync('npm link --silent', { cwd: INSTALL_DIR, stdio: 'pipe' });
        execSync(
          `powershell -Command "Remove-Item '${TmpZip}','${TmpDir}' -Recurse -Force -ErrorAction SilentlyContinue"`,
          { stdio: 'pipe' }
        );
      } catch {
        console.error(
          'update failed — re-run the installer:\n  irm https://raw.githubusercontent.com/Fleabag515/anamnesis/main/install.ps1 | iex'
        );
        process.exit(1);
      }
    } else {
      console.error('update failed: not a git repo. Re-run the installer from the README.');
      process.exit(1);
    }

    console.log('updated — restart active characters: anamnesis restart');
  },
  async doctor() {
    const http = require('http');
    const ok   = (s) => console.log(`  ✓ ${s}`);
    const warn = (s) => console.log(`  ⚠ ${s}`);
    const fail = (s) => console.log(`  ✗ ${s}`);
    console.log('\nanamnesis doctor\n');

    // 1. Daemon
    let daemonOk = false;
    try {
      const res = await client.status();
      if (res.status === 200) { ok(`daemon running (uptime ${res.body.uptime}s)`); daemonOk = true; }
      else fail('daemon not responding');
    } catch { fail('daemon not running — start with: anamnesis start <name>'); }

    // 2. Characters
    if (daemonOk) {
      const chars = (await client.listCharacters()).body.characters || [];
      if (!chars.length) warn('no characters — create one with: anamnesis new');
      for (const c of chars) {
        const label = c.running ? '✓ active  ' : '⚠ inactive';
        const cfgPath = path.join(os.homedir(), '.anamnesis', 'characters', c.name, 'config.json');
        const hasConfig = fs.existsSync(cfgPath);
        console.log(`  ${label} ${c.name} (port ${c.port})${hasConfig ? '' : '  — ✗ MISSING config.json'}`);
        if (!hasConfig) { fail(`    re-create with: anamnesis new`); continue; }

        try {
          const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
          // Upstream reachability
          const upUrl = new URL(cfg.upstream?.baseUrl || '');
          await new Promise((resolve, reject) => {
            const req = http.get({ hostname: upUrl.hostname, port: upUrl.port || 80,
              path: '/health', timeout: 3000 }, resolve);
            req.on('error', reject);
            req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
          }).then(() => ok(`    upstream reachable: ${cfg.upstream.baseUrl}`))
            .catch(() => warn(`    upstream unreachable: ${cfg.upstream.baseUrl}`));
          // Embedding model mismatch
          const dbPath = cfg.history?.dbPath?.replace(/^~/, os.homedir());
          if (dbPath && fs.existsSync(dbPath)) {
            const Database = require('better-sqlite3');
            const db = new Database(dbPath, { readonly: true });
            const rows = db.prepare(
              'SELECT DISTINCT embedding_model FROM engrams WHERE embedding_model IS NOT NULL LIMIT 5'
            ).all();
            db.close();
            const brain = require('./lib/brain.js');
            const cur = brain.embeddingModel();
            const bad = rows.filter(r => r.embedding_model && r.embedding_model !== cur);
            if (bad.length)
              warn(`    embedding mismatch — stored: ${[...new Set(bad.map(r=>r.embedding_model))].join(', ')} | current: ${cur} — run: anamnesis reembed ${c.name}`);
            else
              ok(`    embeddings consistent (${cur})`);
          }
        } catch {}
      }
    }

    // 3. Log file
    const logPath = path.join(os.homedir(), '.anamnesis', 'daemon.log');
    fs.existsSync(logPath) ? ok(`log: ${logPath}`) : warn(`no log file yet (created on first start)`);

    // 4. Startup registration
    if (process.platform === 'win32') {
      try { require('child_process').execSync('schtasks /Query /TN "Anamnesis Daemon"', { stdio: 'pipe' });
        ok('registered as Task Scheduler logon task'); }
      catch { warn('not a startup task — run: anamnesis install'); }
    } else if (process.platform === 'darwin') {
      try { require('child_process').execSync('launchctl print system/com.anamnesis.daemon', { stdio: 'pipe' });
        ok('registered as launchd service'); }
      catch { warn('not a launchd service — run: sudo anamnesis install'); }
    } else {
      try { require('child_process').execSync('systemctl is-enabled anamnesis', { stdio: 'pipe' });
        ok('systemd service enabled'); }
      catch { warn('not a systemd service — run: sudo anamnesis install'); }
    }
    console.log('');
  },

  async reembed([name]) {
    if (!name) die('usage: anamnesis reembed <name>');
    const cfgPath = path.join(os.homedir(), '.anamnesis', 'characters', name, 'config.json');
    if (!fs.existsSync(cfgPath)) die(`character '${name}' not found`);
    const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    const dbPath = (cfg.history?.dbPath || '').replace(/^~/, os.homedir());
    if (!fs.existsSync(dbPath)) die(`no database found at ${dbPath}`);

    const Database = require('better-sqlite3');
    const brain = require('./lib/brain.js');
    brain.init(cfg);
    // Wait for embedder to load
    // brain.embed returns NULL (does not throw) until the embedder has
    // loaded — the old try/catch broke out on the very first null and the
    // whole re-embed then no-opped row by row. Wait for a real vector.
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
      const probe = await brain.embed('test').catch(() => null);
      if (probe) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const currentModel = brain.embeddingModel();
    console.log(`re-embedding with model: ${currentModel}`);

    const db = new Database(dbPath);
    db.pragma('journal_mode = WAL');
    const rows = db.prepare(
      'SELECT id, content FROM engrams WHERE embedding_model != ? OR embedding_model IS NULL'
    ).all(currentModel);
    console.log(`${rows.length} engram(s) need re-embedding...`);

    const upd = db.prepare('UPDATE engrams SET embedding=?, embedding_model=? WHERE id=?');
    let done = 0;
    for (const row of rows) {
      try {
        const vec = await brain.embed(row.content);
        if (vec) {
          upd.run(Buffer.from(new Float32Array(vec).buffer), currentModel, row.id);
          done++;
          if (done % 50 === 0) process.stdout.write(`  ${done}/${rows.length}\r`);
        }
      } catch { /* skip */ }
    }

    const epRows = db.prepare(
      'SELECT id, title, summary FROM episodes WHERE embedding_model != ? OR embedding_model IS NULL'
    ).all(currentModel);
    console.log(`\n${epRows.length} episode(s) need re-embedding...`);
    const updEp = db.prepare('UPDATE episodes SET embedding=?, embedding_model=? WHERE id=?');
    for (const row of epRows) {
      try {
        const vec = await brain.embed(`${row.title}: ${row.summary}`);
        if (vec) updEp.run(Buffer.from(new Float32Array(vec).buffer), currentModel, row.id);
      } catch { /* skip */ }
    }

    const turnRows = db.prepare(
      'SELECT id, content FROM turns WHERE embedding_model != ? OR embedding_model IS NULL'
    ).all(currentModel);
    console.log(`\n${turnRows.length} turn(s) need re-embedding...`);
    const updTurn = db.prepare('UPDATE turns SET embedding=?, embedding_model=? WHERE id=?');
    let doneTurns = 0;
    for (const row of turnRows) {
      try {
        const vec = await brain.embed(String(row.content).slice(0, 2000));
        if (vec) {
          updTurn.run(Buffer.from(new Float32Array(vec).buffer), currentModel, row.id);
          doneTurns++;
          if (doneTurns % 100 === 0) process.stdout.write(`  ${doneTurns}/${turnRows.length}\r`);
        }
      } catch { /* skip */ }
    }

    const fsRows = db.prepare(
      'SELECT id, intention FROM foresights WHERE fulfilled=0 AND (embedding IS NULL OR embedding_model != ?)'
    ).all(currentModel);
    console.log(`\n${fsRows.length} active foresight(s) need embedding...`);
    const updFs = db.prepare('UPDATE foresights SET embedding=?, embedding_model=? WHERE id=?');
    for (const row of fsRows) {
      try {
        const vec = await brain.embed(row.intention);
        if (vec) updFs.run(Buffer.from(new Float32Array(vec).buffer), currentModel, row.id);
      } catch { /* skip */ }
    }

    db.close();
    console.log(`\n✓ re-embedded ${done}/${rows.length} engrams, ${epRows.length} episodes, ` +
                `${doneTurns}/${turnRows.length} turns, ${fsRows.length} foresights`);
    console.log('  restart the character to apply: anamnesis restart ' + name);
  },


  /**
   * Regenerate every episode's title+summary from its own engrams using the
   * current grounded prompts, then merge near-duplicate episodes. One-shot
   * repair for stores whose scenes were built by the old pipeline (star-to-
   * seed clustering + "extract 3-6 facts" prompts): those summaries are
   * vague-to-invented, which is why retrieval either misses them (floors)
   * or drifts (no floors). Usage: anamnesis reconsolidate <character>
   */
  async reconsolidate([name]) {
    if (!name) die('usage: anamnesis reconsolidate <name>');
    const cfgPath = path.join(os.homedir(), '.anamnesis', 'characters', name, 'config.json');
    if (!fs.existsSync(cfgPath)) die(`character '${name}' not found`);
    const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    const dbPath = (cfg.history?.dbPath || '').replace(/^~/, os.homedir());
    if (!fs.existsSync(dbPath)) die(`no database found at ${dbPath}`);

    const brain = require('./lib/brain.js');
    const { EPISODE_SCENE } = require('./lib/prompts.js');
    brain.init(cfg);
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
      const probe = await brain.embed('test').catch(() => null);
      if (probe) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const currentModel = brain.embeddingModel();

    const HistoryStore = require('./history.js');
    const store = new HistoryStore(dbPath);
    const cosine = (a, b) => {
      if (!a || !b || a.length !== b.length) return 0;
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
      const d = Math.sqrt(na) * Math.sqrt(nb);
      return d === 0 ? 0 : dot / d;
    };
    const norm = (v) => {
      let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1;
      return Float32Array.from(v, (x) => x / n);
    };

    const scenes = store.getScenes('unused');
    console.log(`${scenes.length} episode(s) to reconsolidate...`);
    let redone = 0, kept = 0, dropped = 0;
    for (const s of scenes) {
      let ids = [];
      try { ids = JSON.parse(s.engram_ids); } catch { ids = []; }
      const cells = ids.length
        ? store.db.prepare(`SELECT content, embedding, importance FROM engrams WHERE id IN (${ids.map(() => '?').join(',')})`).all(...ids)
        : [];
      if (!cells.length) {
        // Orphaned scene: its engrams were pruned — the summary is backed by
        // nothing. Remove rather than keep an unfalsifiable memory.
        store.db.prepare('DELETE FROM episodes WHERE id=?').run(s.id);
        dropped++;
        continue;
      }
      const vecs = cells.map((c) => HistoryStore.toFloat32(c.embedding)).filter(Boolean);
      let centroid = null;
      if (vecs.length) {
        const dim = vecs[0].length;
        const sum = new Float32Array(dim);
        for (const v of vecs) { if (v.length === dim) for (let k = 0; k < dim; k++) sum[k] += v[k]; }
        centroid = norm(sum);
      }
      const facts = cells.map((c) => c.content).slice(0, 12);
      const factList = facts.map((f, i) => `${i + 1}. ${f}`).join('\n');
      let data = null;
      try {
        const text = await brain.chat([{ role: 'user', content: EPISODE_SCENE + factList }],
          { maxTokens: 256, temperature: 0.2, timeoutMs: 90000 });
        data = brain.tryParseJsonObject(text);
      } catch { data = null; }
      if (!data?.title || !data?.summary) {
        const byImp = [...cells].sort((a, b) => (b.importance ?? 0.5) - (a.importance ?? 0.5));
        data = {
          title: byImp[0].content.split(/\s+/).slice(0, 6).join(' '),
          summary: byImp.slice(0, 3).map((c) => c.content).join(' — ').slice(0, 300),
        };
      }
      let emb = await brain.embed(String(data.summary)).catch(() => null);
      if (centroid && (!emb || cosine(emb, centroid) < 0.45)) emb = centroid;
      store.updateScene(s.id, String(data.title), String(data.summary),
        emb ? new Float32Array(emb) : null, ids, s.avg_importance ?? 0.5, currentModel);
      redone++;
      if (redone % 10 === 0) console.log(`  ${redone}/${scenes.length}...`);
    }

    // Merge pass: absorb near-duplicate episodes into their older sibling.
    const fresh = store.getScenes('unused')
      .map((x) => ({ ...x, vec: HistoryStore.toFloat32(x.embedding) }))
      .filter((x) => x.vec);
    const gone = new Set();
    let merged = 0;
    for (let i = 0; i < fresh.length; i++) {
      if (gone.has(fresh[i].id)) continue;
      for (let j = i + 1; j < fresh.length; j++) {
        if (gone.has(fresh[j].id)) continue;
        if (cosine(fresh[i].vec, fresh[j].vec) < 0.85) continue;
        let ia = [], ja = [];
        try { ia = JSON.parse(fresh[i].engram_ids); } catch {}
        try { ja = JSON.parse(fresh[j].engram_ids); } catch {}
        const ids = [...new Set([...ia, ...ja])];
        store.updateScene(fresh[i].id, fresh[i].title, fresh[i].summary,
          fresh[i].vec, ids, ((fresh[i].avg_importance ?? 0.5) + (fresh[j].avg_importance ?? 0.5)) / 2,
          currentModel);
        for (const cid of ja) store.assignMemcellToScene(cid, fresh[i].id);
        store.db.prepare('DELETE FROM episodes WHERE id=?').run(fresh[j].id);
        gone.add(fresh[j].id);
        merged++;
      }
    }
    store.close();
    console.log(`✓ reconsolidated ${redone} episode(s), dropped ${dropped} orphaned, merged ${merged} near-duplicate(s) (${kept} untouched)`);
    console.log('  restart the character to apply: anamnesis restart ' + name);
  },

  /**
   * Merge fragmented session buckets into one canonical key.
   * Usage: anamnesis migrate-sessions <character> [--into <sessionKey>]
   * Default target: oc:pleiades:<character> (what Pleiades' Engine sends
   * via X-Session-Id). Requires the character to be stopped.
   */
  async 'migrate-sessions'(args) {
    const name = args[0];
    if (!name) die('usage: anamnesis migrate-sessions <character> [--into <sessionKey>]');
    const intoIdx = args.indexOf('--into');
    const into = intoIdx !== -1 ? args[intoIdx + 1] : `oc:pleiades:${name}`;
    if (!into) die('--into requires a value');

    await ensureDaemon();
    const r = await client.getCharacter(name);
    if (r.status !== 200) die(`character '${name}' not found`);
    const running = r.body.running ?? r.body.active;
    if (running) die(`'${name}' is running — stop it first: anamnesis stop ${name}`);

    const HistoryStore = require('./history.js');
    const path = require('path');
    const os = require('os');
    const dbPath = path.join(os.homedir(), '.anamnesis', 'characters', name, 'history.db');
    const store = new HistoryStore(dbPath);
    const before = store.listSessions();
    console.log('session buckets before:');
    for (const s of before) console.log(`  ${s.session_key}  (${s.turns} turns)`);
    const changed = store.mergeSessions(into);
    store.close();
    console.log(`✓ merged ${before.length} bucket(s) → '${into}' (${changed} rows re-keyed)`);
    console.log(`  start it again with: anamnesis start ${name}`);
  },
};

// ─── Aliases ──────────────────────────────────────────────────────────────────
commands.ls = commands.list;
commands.ps = commands.status;
commands.run = commands.start;
commands.kill = commands.stop;
commands.rm = commands.remove;

// ─── Entry point ─────────────────────────────────────────────────────────────

const [, , cmd, ...args] = process.argv;

if (!cmd) {
  commands.status().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
} else if (commands[cmd]) {
  commands[cmd](args).catch((e) => {
    console.error('error:', e.message);
    process.exit(1);
  });
} else if (cmd === '--help' || cmd === '-h' || cmd === 'help') {
  console.log(`anamnesis — multi-character memory proxy

Usage: anamnesis <command> [options]

Commands:
  new                   Create a new character (interactive wizard)
  list, ls              List all characters and their status
  start <name>          Start a character's proxy
  stop <name>           Stop a character's proxy
  restart <name>        Restart a character's proxy
  status, ps            Show daemon status and active characters
  show <name>           Show character config
  edit <name>           Edit character settings (interactive)
  remove, rm <name>     Delete a character and its data
  import <name> <file>  Import memories from a file
  export <name>         Export character memories
  logs [name]           Tail logs
  install               Register anamnesis as a system service
  uninstall             Remove the system service
  update                Check for updates and install if available

Options:
  -h, --help            Show this help message`);
} else {
  console.error(`unknown command: ${cmd}`);
  console.error('run: anamnesis --help   for a list of commands');
  process.exit(1);
}
