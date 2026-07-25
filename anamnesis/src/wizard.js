'use strict';

const prompts = require('prompts');
const path = require('path');
const os = require('os');
const fs = require('fs');

const client = require('./lib/client.js');
const { buildConfig } = require('./lib/char-config.js');
const { suggestPort } = require('./lib/ports.js');
const Registry = require('./lib/registry.js');

const REGISTRY_PATH = path.join(os.homedir(), '.anamnesis', 'registry.json');
const NAME_RE = /^[a-z0-9_-]+$/i;

// ─── Auto-detect running LLM endpoint ────────────────────────────────────────

async function detectEndpoint() {
  const candidates = [
    { url: 'http://127.0.0.1:8083/v1', label: 'llama-server' },
    { url: 'http://127.0.0.1:11434/v1', label: 'ollama' },
    { url: 'http://127.0.0.1:1234/v1', label: 'lm-studio' },
    { url: 'http://127.0.0.1:5001/v1', label: 'koboldcpp' },
    { url: 'http://127.0.0.1:8080/v1', label: 'other' },
  ];
  for (const c of candidates) {
    try {
      const { port, hostname } = new URL(c.url);
      await new Promise((resolve, reject) => {
        const net = require('net');
        const sock = net.connect(Number(port) || 80, hostname, () => {
          sock.destroy();
          resolve();
        });
        sock.setTimeout(300);
        sock.on('timeout', () => {
          sock.destroy();
          reject(new Error('timeout'));
        });
        sock.on('error', reject);
      });
      return c.url; // first one that accepts a connection wins
    } catch {
      /* not running */
    }
  }
  return candidates[0].url; // fallback to llama-server
}

async function run(args) {
  const flags = parseFlags(args);
  const registry = new Registry(REGISTRY_PATH);

  console.log('\n✨ anamnesis — new character\n');

  let name = flags.name;
  if (!name) {
    ({ name } = await prompts({
      type: 'text',
      name: 'name',
      message: 'Character name:',
      validate: (v) => (NAME_RE.test(v) ? true : 'letters, digits, hyphens, underscores only'),
    }));
  }
  if (!name) process.exit(0);
  if (registry.get(name)) {
    console.error(`\n  a character named '${name}' already exists — use: anamnesis edit ${name}`);
    process.exit(1);
  }

  const suggested = flags.port || (await suggestPort(registry));
  let port = suggested;
  if (!flags.port) {
    ({ port } = await prompts({
      type: 'number',
      name: 'port',
      message: 'Port:',
      initial: suggested,
    }));
  }

  let upstreamUrl = flags.upstream || '';
  if (!upstreamUrl) {
    const detectedUrl = await detectEndpoint();
    ({ upstreamUrl } = await prompts({
      type: 'text',
      name: 'upstreamUrl',
      message: 'Model endpoint URL:',
      initial: detectedUrl,
    }));
  }

  let apiKey = flags.key || '';
  if (!apiKey) {
    ({ apiKey } = await prompts({
      type: 'text',
      name: 'apiKey',
      message: 'API key:',
      initial: '',
    }));
  }

  let mode = flags.blank ? 'blank' : flags.description ? 'describe' : null;
  if (!mode) {
    ({ mode } = await prompts({
      type: 'select',
      name: 'mode',
      message: 'Character setup:',
      choices: [
        { title: 'Import from files or URLs', value: 'import' },
        { title: 'Describe the character concept', value: 'describe' },
        { title: 'Start blank — let it develop on its own', value: 'blank' },
      ],
    }));
  }

  let characterDescription = '';
  let importSources = [];

  if (mode === 'import') {
    const importer = require('./importers/index.js');
    const result = await importer.runWizard({ name });
    characterDescription = result.description;
    importSources = result.sources;
  } else if (mode === 'describe') {
    ({ characterDescription } = await prompts({
      type: 'text',
      name: 'characterDescription',
      message: 'Describe the character:',
    }));
  }

  const config = buildConfig({
    name,
    port,
    upstreamUrl,
    apiKey,
    characterDescription: characterDescription || undefined,
    blank: mode === 'blank',
  });

  const createRes = await client.createCharacter(name, config);
  if (createRes.status !== 201) {
    console.error('failed to create character:', createRes.body.error);
    process.exit(1);
  }

  // Start first so DB exists before import
  let startNow = flags.yes;
  if (startNow === undefined) {
    ({ startNow } = await prompts({
      type: 'confirm',
      name: 'startNow',
      message: `Start ${name} now?`,
      initial: true,
    }));
  }
  if (startNow) {
    const startRes = await client.startCharacter(name);
    if (startRes.status === 200) {
      console.log(`\n✓ ${name} is running on http://127.0.0.1:${startRes.body.port}/v1`);
    } else {
      console.error('failed to start:', startRes.body.error);
    }
  } else {
    console.log(`\n✓ ${name} created. Start with: anamnesis start ${name}`);
  }

  // Import after start (DB now exists)
  if (importSources.length > 0) {
    if (!startNow) {
      console.log(
        `\nnote: import data will be written once you start '${name}' (run: anamnesis start ${name})`
      );
    } else {
      const importer = require('./importers/index.js');
      await importer.importInto(name, importSources, characterDescription);
    }
  }
  console.log('');
}

async function edit(name) {
  const registry = new Registry(REGISTRY_PATH);
  const entry = registry.get(name);
  const configPath = path.join(os.homedir(), '.anamnesis', 'characters', name, 'config.json');
  if (!fs.existsSync(configPath)) {
    console.error(`character '${name}' not found`);
    process.exit(1);
  }
  const current = JSON.parse(fs.readFileSync(configPath, 'utf8'));

  console.log(`\n✏️  editing ${name}\n`);

  const { upstreamUrl } = await prompts({
    type: 'text',
    name: 'upstreamUrl',
    message: 'Model endpoint URL:',
    initial: current.upstream.baseUrl,
  });
  const { apiKey } = await prompts({
    type: 'text',
    name: 'apiKey',
    message: 'API key:',
    initial: current.upstream.apiKey,
  });
  const { port } = await prompts({
    type: 'number',
    name: 'port',
    message: 'Port:',
    initial: current.proxy.port,
  });
  const { tokenBudget } = await prompts({
    type: 'number',
    name: 'tokenBudget',
    message: 'Token budget:',
    initial: current.context.tokenBudget,
  });

  current.upstream.baseUrl = upstreamUrl;
  current.upstream.apiKey = apiKey;
  current.proxy.port = port;
  current.context.tokenBudget = tokenBudget;

  fs.writeFileSync(configPath, JSON.stringify(current, null, 2), 'utf8');
  registry.updatePort(name, port);
  console.log(`\n✓ ${name} updated`);

  // Always restart if active (daemon holds old config in memory)
  if (entry.active) {
    console.log('restarting to apply changes...');
    await client.stopCharacter(name);
    await client.startCharacter(name);
    console.log(`✓ ${name} restarted`);
  }
}

function parseFlags(args) {
  const flags = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--name') flags.name = args[++i];
    else if (args[i] === '--port') flags.port = parseInt(args[++i]);
    else if (args[i] === '--upstream') flags.upstream = args[++i];
    else if (args[i] === '--key') flags.key = args[++i];
    else if (args[i] === '--blank') flags.blank = true;
    else if (args[i] === '--yes') flags.yes = true;
    else if (args[i] === '--description') flags.description = args[++i];
  }
  return flags;
}

module.exports = { run, edit };
