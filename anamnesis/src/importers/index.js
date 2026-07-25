// src/importers/index.js
'use strict';

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');
const os = require('os');

const { detectFormat } = require('./detect.js');
const brain = require('../lib/brain.js');
const { IMPORT_EXTRACTION } = require('../lib/prompts.js');
const log = require('../lib/logger.js').make('importer');

const ADAPTERS = {
  text: require('./text.js'),
  openclaw: require('./openclaw.js'),
  'anamnesis-export': require('./anamnesis-export.js'),
  odysseus: require('./odysseus.js'),
  characterai: require('./characterai.js'),
  sillytavern: require('./sillytavern.js'),
};

async function fetchUrl(url) {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith('https') ? https : http;
    lib
      .get(url, (res) => {
        if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode} from ${url}`));
        const chunks = [];
        res.on('data', (d) => chunks.push(d));
        res.on('end', () => resolve(Buffer.concat(chunks)));
      })
      .on('error', reject);
  });
}

async function loadSource(source) {
  if (source.startsWith('http://') || source.startsWith('https://')) {
    log.info(`fetching ${source}...`);
    return {
      filename: path.basename(new URL(source).pathname) || 'download',
      buf: await fetchUrl(source),
    };
  }
  if (!fs.existsSync(source)) throw new Error(`file not found: ${source}`);
  return { filename: path.basename(source), buf: fs.readFileSync(source) };
}

async function llmExtract(text) {
  const result = await brain.chat(
    [
      { role: 'system', content: IMPORT_EXTRACTION },
      { role: 'user', content: text.slice(0, 12000) },
    ],
    { maxTokens: 500, temperature: 0.1, timeoutMs: 30000 }
  );
  try {
    return JSON.parse(result);
  } catch {
    return { other: result };
  }
}

async function importSources(sources, extraDescription = '') {
  const textParts = [];
  const directData = [];
  const summaries = [];

  // Validate all sources before any work
  for (const src of sources) {
    if (!src.startsWith('http://') && !src.startsWith('https://') && !fs.existsSync(src)) {
      throw new Error(`file not found: ${src}`);
    }
  }

  for (const src of sources) {
    const { filename, buf } = await loadSource(src);
    const format = detectFormat(filename, buf);
    const adapter = ADAPTERS[format] || ADAPTERS['text'];
    log.info(`${filename} → ${format}`);
    const result = adapter.extract(buf);
    if (result.direct) {
      directData.push(result.data);
      if (result.summary) summaries.push(result.summary);
    } else {
      textParts.push(result.text);
    }
  }

  if (extraDescription) textParts.push(extraDescription);

  let profile = null;
  if (textParts.length > 0) {
    const combined = textParts.join('\n\n---\n\n');
    log.info('running LLM extraction...');
    profile = await llmExtract(combined);
  }

  return { profile, directData, summaries, hasText: textParts.length > 0 };
}

function writeToDb(characterName, { profile, directData }) {
  const dbPath = path.join(os.homedir(), '.anamnesis', 'characters', characterName, 'history.db');
  if (!fs.existsSync(dbPath)) {
    console.warn(
      `warning: '${characterName}' has no history DB yet — start it first with: anamnesis start ${characterName}`
    );
    console.warn('import data was not written; re-run after starting the character.');
    return;
  }
  const Database = require('better-sqlite3');
  const db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  const insert = db.prepare(`
    INSERT OR IGNORE INTO engrams (session_key, content, category, decay_score, created_at)
    VALUES (?, ?, ?, ?, ?)
  `);
  const importMany = db.transaction((cells) => {
    for (const cell of cells) {
      insert.run(
        cell.session_key || 'imported',
        cell.content,
        cell.category || 'other',
        cell.decay_score || 1.0,
        cell.created_at || Math.floor(Date.now() / 1000)
      );
    }
  });
  for (const d of directData) {
    if (d.engrams) importMany(d.engrams);
  }
  db.close();
}

/**
 * Returns true if the arg looks like a file path or URL rather than a character name.
 * Used to distinguish `anamnesis import NyzKh file.md` from `anamnesis import file.md`.
 */
function _looksLikeSource(arg) {
  return (
    arg.startsWith('http://') ||
    arg.startsWith('https://') ||
    arg.startsWith('./') ||
    arg.startsWith('../') ||
    arg.startsWith('/') ||
    arg.startsWith('~') ||
    /^[A-Za-z]:[\\\/]/.test(arg) || // Windows drive letter e.g. C:\ or C:/
    arg.includes('/') ||
    arg.includes('\\') ||
    arg.includes('.')
  );
}

async function runCli(args) {
  const sources = [];
  let intoName = null;
  let description = '';
  let yes = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--into') intoName = args[++i];
    else if (args[i] === '--description') description = args[++i];
    else if (args[i] === '--yes') yes = true;
    else sources.push(args[i]);
  }

  // Support positional syntax: anamnesis import <name> <file> [...]
  // If the first positional arg doesn't look like a file or URL, treat it as
  // the character name (equivalent to --into <name>).
  if (!intoName && sources.length >= 1 && !_looksLikeSource(sources[0])) {
    intoName = sources.shift();
  }

  if (!sources.length) {
    console.error('usage: anamnesis import <name> <file|url> [...]');
    process.exit(1);
  }

  console.log(`\nimporting ${sources.length} source(s)...\n`);
  const result = await importSources(sources, description);

  if (result.hasText && result.profile) {
    console.log('\nExtracted profile preview:');
    console.log(JSON.stringify(result.profile, null, 2));
  }
  if (result.summaries.length) {
    for (const s of result.summaries) console.log(`  ${s}`);
  }

  if (!yes) {
    const prompts = require('prompts');
    const { ok } = await prompts({
      type: 'confirm',
      name: 'ok',
      message: 'Import this?',
      initial: true,
    });
    if (!ok) {
      console.log('aborted');
      return;
    }
  }

  if (intoName) {
    writeToDb(intoName, result);
    console.log(`\n✓ imported into '${intoName}'`);
  } else {
    console.log(
      '\nNo --into specified. Use --into <name> to merge into an existing character, or run anamnesis new to create one.'
    );
  }
}

async function runWizard({ name }) {
  const prompts = require('prompts');
  const { sourceLine } = await prompts({
    type: 'text',
    name: 'sourceLine',
    message: 'Files or URLs (space-separated):',
  });
  const sources = (sourceLine || '').split(/\s+/).filter(Boolean);
  const { desc } = await prompts({
    type: 'text',
    name: 'desc',
    message: 'Add a written description? (optional):',
  });
  return { sources, description: desc || '' };
}

async function importInto(name, sources, description) {
  const result = await importSources(sources, description);
  writeToDb(name, result);
}

module.exports = { importSources, writeToDb, runCli, runWizard, importInto };
