const test = require('node:test');
const assert = require('node:assert/strict');

const {
  expandHome,
  extractContentText,
  findNewUserMessage,
  getSessionKey,
  getMemoryCategory,
  formatDuration,
  buildUpstreamHeaders,
  makeSseAccumulator,
  resolveTaskActivitySnapshot,
} = require('../src/lib/proxy-helpers.js');

// ─── extractContentText ─────────────────────────────────────────────────────

test('extractContentText: passes strings through', () => {
  assert.equal(extractContentText('hello'), 'hello');
});

test('extractContentText: joins OpenAI text parts', () => {
  const parts = [
    { type: 'text', text: 'first' },
    { type: 'text', text: 'second' },
  ];
  assert.equal(extractContentText(parts), 'first\nsecond');
});

test('extractContentText: mixed parts keep only text', () => {
  const parts = [
    { type: 'text', text: 'hello' },
    { type: 'tool_result', tool_use_id: 'x', content: 'ignored' },
    { type: 'text', text: 'world' },
  ];
  assert.equal(extractContentText(parts), 'hello\nworld');
});

test('extractContentText: array with no text parts falls back to JSON', () => {
  const parts = [{ type: 'image_url', image_url: { url: 'data:foo' } }];
  const out = extractContentText(parts);
  // Either JSON of the parts or empty — must be a string, never the array.
  assert.equal(typeof out, 'string');
  assert.ok(out.includes('image_url'));
});

test('extractContentText: object → JSON', () => {
  assert.equal(extractContentText({ a: 1 }), '{"a":1}');
});

test('extractContentText: null/undefined → empty string', () => {
  assert.equal(extractContentText(null), '');
  assert.equal(extractContentText(undefined), '');
});

// ─── findNewUserMessage ─────────────────────────────────────────────────────
// This is the core claim fix/anamnesis-reflection-injection rests on: the
// proxy persists "the new user turn" by reverse-scanning `messages` for the
// first role==='user' entry (see findNewUserMessage's docstring and its one
// call site in proxy.js). A machine-generated notice delivered as a
// SYNTHETIC TOOL ROUND — a fabricated assistant tool_calls message the
// model never actually emitted, immediately followed by a role:"tool"
// result carrying the real notice text (Pleiades engine.py's periodic
// self-check, or any future "background task finished" notice) — must
// never be mistaken for that new user turn, no matter where in the array
// it sits. These tests build exactly that shape by hand (synthetic data
// only, no real character content) and assert on it directly, rather than
// just reasoning about the reverse-scan.

test('findNewUserMessage: finds the only user message', () => {
  const messages = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'hello' },
  ];
  assert.equal(findNewUserMessage(messages).content, 'hello');
});

test('findNewUserMessage: an agentic resend with real tool_calls/tool pairs still finds the real user message', () => {
  // Shape of a normal (non-synthetic) tool round-trip: assistant calls a
  // REAL tool, gets a REAL result. The literal last message is role:"tool",
  // not role:"user" -- must not be picked up as the new user turn.
  const messages = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'what files are in this repo?' },
    {
      role: 'assistant', content: '',
      tool_calls: [{ id: 'call_1', type: 'function',
                    function: { name: 'list_files', arguments: '{}' } }],
    },
    { role: 'tool', tool_call_id: 'call_1', content: 'a.txt\nb.txt' },
  ];
  const found = findNewUserMessage(messages);
  assert.equal(found.content, 'what files are in this repo?');
});

test('findNewUserMessage: a SYNTHETIC tool round (fabricated assistant tool_calls + its role:"tool" result) is invisible to the scan even as the last entries in the array', () => {
  // Exactly the shape fix/anamnesis-reflection-injection produces: the
  // model never emitted this tool_calls entry -- it's fabricated by the
  // caller (Pleiades' engine.py / harness/agent.py) to deliver an internal
  // notice through a channel that is provably NOT "the new user turn".
  const messages = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'help me refactor this function' },
    {
      role: 'assistant', content: '',
      tool_calls: [{ id: 'call_1', type: 'function',
                    function: { name: 'read_file', arguments: '{"path":"a.py"}' } }],
    },
    { role: 'tool', tool_call_id: 'call_1', content: 'def f(): ...' },
    // Synthetic self-reflection round -- the model never called this tool.
    {
      role: 'assistant', content: '',
      tool_calls: [{ id: 'reflect_40', type: 'function',
                    function: { name: 'system_reflection', arguments: '{}' } }],
    },
    {
      role: 'tool', tool_call_id: 'reflect_40',
      content: '[Self-check after 40 steps] Before continuing, genuinely evaluate...',
    },
  ];

  const found = findNewUserMessage(messages);
  assert.ok(found, 'expected a role==user message to be found');
  assert.equal(found.role, 'user');
  assert.equal(found.content, 'help me refactor this function');
  // Directly disprove the bug: the found message must never be (or resolve
  // to, via the exact same extractContentText step proxy.js runs next) the
  // fabricated notice text.
  assert.notEqual(extractContentText(found.content), '[Self-check after 40 steps] Before continuing, genuinely evaluate...');
  assert.equal(extractContentText(found.content), 'help me refactor this function');
});

test('findNewUserMessage: a synthetic tool round with NO real user message anywhere finds nothing (never falls back to tool content)', () => {
  // Degenerate case, for robustness: if a request somehow carried only a
  // synthetic notice and no real user turn, the scan must return undefined
  // rather than ever treating tool content as user speech.
  const messages = [
    { role: 'system', content: 'sys' },
    {
      role: 'assistant', content: '',
      tool_calls: [{ id: 'reflect_1', type: 'function',
                    function: { name: 'system_reflection', arguments: '{}' } }],
    },
    { role: 'tool', tool_call_id: 'reflect_1', content: '[Self-check after 40 steps] ...' },
  ];
  const found = findNewUserMessage(messages);
  assert.equal(found, undefined);
  // proxy.js's actual guard (`if (userText && ...)`) would then correctly
  // skip persistence entirely -- confirm the text side is also empty, not
  // some stringified fallback that would sneak the tool content in anyway.
  assert.equal(extractContentText(found?.content), '');
});

test('findNewUserMessage: multiple real user turns in one resent array (agentic re-send) resolves to the LAST one, not the first', () => {
  const messages = [
    { role: 'user', content: 'first question' },
    { role: 'assistant', content: 'reply' },
    { role: 'user', content: 'second question, same array' },
  ];
  assert.equal(findNewUserMessage(messages).content, 'second question, same array');
});

// ─── expandHome ─────────────────────────────────────────────────────────────

test('expandHome: rewrites ~ to HOME', () => {
  const out = expandHome({ p: '~/foo' }, '/home/fake');
  assert.equal(out.p, '/home/fake/foo');
});

test('expandHome: rewrites ${HOME} and $HOME', () => {
  assert.equal(expandHome('${HOME}/x', '/h'), '/h/x');
  assert.equal(expandHome('$HOME/y', '/h'), '/h/y');
});

test('expandHome: leaves non-leading ~ alone', () => {
  assert.equal(expandHome('a/~/b', '/h'), 'a/~/b');
});

test('expandHome: deeply nested', () => {
  const cfg = { a: { b: ['~/x', { c: '~/y' }] } };
  const out = expandHome(cfg, '/H');
  assert.deepEqual(out, { a: { b: ['/H/x', { c: '/H/y' }] } });
});

test('expandHome: non-string scalars untouched', () => {
  assert.equal(expandHome(42, '/h'), 42);
  assert.equal(expandHome(null, '/h'), null);
  assert.equal(expandHome(true, '/h'), true);
});

// ─── getSessionKey ──────────────────────────────────────────────────────────

test('getSessionKey: explicit X-OpenClaw-Session wins', () => {
  const k = getSessionKey({ 'x-openclaw-session': 'abc123' }, 'apikey');
  assert.equal(k, 'oc:abc123');
});

test('getSessionKey: X-Session-Id fallback', () => {
  const k = getSessionKey({ 'x-session-id': 'sess-xyz' }, 'apikey');
  assert.equal(k, 'oc:sess-xyz');
});

test('getSessionKey: hashes bearer token, prefix only', () => {
  const k = getSessionKey({ authorization: 'Bearer super-secret-12345' }, 'somethingelse');
  assert.match(k, /^auth:[a-f0-9]{16}$/);
  // The raw token must not appear in the derived key.
  assert.ok(!k.includes('super-secret'));
});

test('getSessionKey: ignores upstream apiKey echo', () => {
  // If the client (or a misconfigured tool) sent our own configured upstream
  // apiKey, we shouldn't fingerprint that as a "real" user session.
  const k = getSessionKey({ authorization: 'Bearer localqwen' }, 'localqwen');
  assert.equal(k, 'default');
});

test('getSessionKey: returns default when no auth', () => {
  assert.equal(getSessionKey({}, 'anything'), 'default');
});

test('getSessionKey: same token => same key (deterministic)', () => {
  const a = getSessionKey({ authorization: 'Bearer T' }, '');
  const b = getSessionKey({ authorization: 'Bearer T' }, '');
  assert.equal(a, b);
});

// ─── buildUpstreamHeaders ───────────────────────────────────────────────────

test('buildUpstreamHeaders: strips proxy-internal + hop-by-hop headers', () => {
  const out = buildUpstreamHeaders(
    {
      host: 'localhost:8084',
      'x-openclaw-session': 's',
      'x-session-id': 's2',
      'content-length': '999',
      connection: 'keep-alive',
      'transfer-encoding': 'chunked',
      'x-other': 'keep',
    },
    { upstreamApiKey: 'K' }
  );
  assert.equal(out['host'], undefined);
  assert.equal(out['x-openclaw-session'], undefined);
  assert.equal(out['x-session-id'], undefined);
  assert.equal(out['content-length'], undefined);
  assert.equal(out['connection'], undefined);
  assert.equal(out['transfer-encoding'], undefined);
  assert.equal(out['x-other'], 'keep');
});

test('buildUpstreamHeaders: injects Authorization when apiKey is set', () => {
  const out = buildUpstreamHeaders(
    { authorization: 'Bearer client-token' },
    { upstreamApiKey: 'SERVER-KEY' }
  );
  // Should be overwritten, not duplicated.
  const authKeys = Object.keys(out).filter((k) => k.toLowerCase() === 'authorization');
  assert.equal(authKeys.length, 1);
  assert.equal(out[authKeys[0]], 'Bearer SERVER-KEY');
});

test('buildUpstreamHeaders: passes Authorization through when apiKey empty', () => {
  const out = buildUpstreamHeaders(
    { authorization: 'Bearer client-token' },
    { upstreamApiKey: '' }
  );
  // The client-supplied header must survive untouched.
  const authKey = Object.keys(out).find((k) => k.toLowerCase() === 'authorization');
  assert.equal(out[authKey], 'Bearer client-token');
});

// ─── makeSseAccumulator ─────────────────────────────────────────────────────

function sseFrame(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

test('SSE: reconstructs content from delta frames', () => {
  const acc = makeSseAccumulator();
  acc.feed(Buffer.from(sseFrame({ choices: [{ delta: { content: 'Hello' } }] })));
  acc.feed(Buffer.from(sseFrame({ choices: [{ delta: { content: ', world' } }] })));
  acc.feed(Buffer.from(sseFrame({ choices: [{ delta: { content: '!' } }] })));
  acc.feed(Buffer.from('data: [DONE]\n\n'));
  assert.equal(acc.content, 'Hello, world!');
});

test('SSE: handles frame split across chunk boundaries', () => {
  const acc = makeSseAccumulator();
  const frame = sseFrame({ choices: [{ delta: { content: 'split-frame' } }] });
  const mid = Math.floor(frame.length / 2);
  acc.feed(Buffer.from(frame.slice(0, mid)));
  acc.feed(Buffer.from(frame.slice(mid)));
  assert.equal(acc.content, 'split-frame');
});

test('SSE: ignores keepalives and non-JSON lines', () => {
  const acc = makeSseAccumulator();
  acc.feed(Buffer.from(': ping\n\n'));
  acc.feed(Buffer.from('data: not-json\n\n'));
  acc.feed(Buffer.from(sseFrame({ choices: [{ delta: { content: 'OK' } }] })));
  assert.equal(acc.content, 'OK');
});

test('SSE: also picks up message.content shape', () => {
  const acc = makeSseAccumulator();
  acc.feed(Buffer.from(sseFrame({ choices: [{ message: { content: 'final' } }] })));
  assert.equal(acc.content, 'final');
});

test('SSE: empty stream => empty content', () => {
  const acc = makeSseAccumulator();
  acc.feed(Buffer.from('data: [DONE]\n\n'));
  assert.equal(acc.content, '');
});

// ─── stripThinkingTokens ─────────────────────────────────────────────────────
const { stripThinkingTokens } = require('../src/lib/proxy-helpers.js');

test('stripThinkingTokens: passthrough when no thinking tokens', () => {
  assert.equal(stripThinkingTokens('Hello, how can I help?'), 'Hello, how can I help?');
});

test('stripThinkingTokens: returns falsy input unchanged', () => {
  assert.equal(stripThinkingTokens(''), '');
  assert.equal(stripThinkingTokens(null), null);
  assert.equal(stripThinkingTokens(undefined), undefined);
});

test('stripThinkingTokens: strips Qwen3 <think> block', () => {
  const raw = '<think>\nLet me reason through this carefully.\n</think>\nThe answer is 42.';
  assert.equal(stripThinkingTokens(raw), 'The answer is 42.');
});

test('stripThinkingTokens: strips Gemma 4 <|channel>thought block (empty)', () => {
  const raw = '<|channel>thought\n<channel|>online';
  assert.equal(stripThinkingTokens(raw), 'online');
});

test('stripThinkingTokens: strips Gemma 4 <|channel>thought block (with content)', () => {
  const raw = '<|channel>thought\nI should greet the user warmly.\n<channel|>Hello there!';
  assert.equal(stripThinkingTokens(raw), 'Hello there!');
});

test('stripThinkingTokens: strips multiple Gemma 4 blocks', () => {
  const raw = '<|channel>thought\n<channel|><|channel>thought\n<channel|>Ready.';
  assert.equal(stripThinkingTokens(raw), 'Ready.');
});

test('stripThinkingTokens: strips multiple Qwen3 blocks', () => {
  const raw = '<think>step 1</think>\nIntermediate.\n<think>step 2</think>\nFinal answer.';
  assert.equal(stripThinkingTokens(raw), 'Intermediate.\n\nFinal answer.');
});

test('stripThinkingTokens: mixed Gemma 4 and Qwen3 blocks', () => {
  const raw = '<|channel>thought\ngemma thinking\n<channel|><think>qwen thinking</think>actual reply';
  assert.equal(stripThinkingTokens(raw), 'actual reply');
});

test('stripThinkingTokens: trims surrounding whitespace after stripping', () => {
  const raw = '   <think>noise</think>   real content   ';
  assert.equal(stripThinkingTokens(raw), 'real content');
});

test('stripThinkingTokens: leaves normal angle-bracket content alone', () => {
  const html = '<b>bold</b> and <em>italic</em>';
  assert.equal(stripThinkingTokens(html), html);
});

test('stripThinkingTokens: thinking-only content collapses to empty string', () => {
  const raw = '<|channel>thought\nsome internal reasoning\n<channel|>';
  assert.equal(stripThinkingTokens(raw), '');
});

test('stripThinkingTokens: Qwen3 thinking-only collapses to empty string', () => {
  assert.equal(stripThinkingTokens('<think>chain of thought</think>'), '');
});

test('stripThinkingTokens: strips truncated Gemma 4 block (no closing tag)', () => {
  // Model hit max_tokens mid-thought — opener present, <channel|> never emitted
  const raw = 'some prefix<|channel>thought\npartial reasoning that was cut off';
  assert.equal(stripThinkingTokens(raw), 'some prefix');
});

test('stripThinkingTokens: strips truncated Qwen3 block (no closing tag)', () => {
  const raw = 'prefix text\n<think>partial chain of thought never closed';
  assert.equal(stripThinkingTokens(raw), 'prefix text');
});

test('stripThinkingTokens: complete block before truncated block both stripped', () => {
  const raw = '<|channel>thought\nfirst thought\n<channel|>real reply<|channel>thought\ntruncated';
  assert.equal(stripThinkingTokens(raw), 'real reply');
});

// PUA-encoded Gemma 4 tokens (llama.cpp streaming decodes <|channel> as U+F06C)
test('stripThinkingTokens: Gemma 4 PUA complete block stripped', () => {
  const input = '\uf06cthought\nI am thinking\n!</thought\nActual response';
  assert.strictEqual(stripThinkingTokens(input), 'Actual response');
});

test('stripThinkingTokens: Gemma 4 PUA orphaned opener stripped', () => {
  const input = '\uf06cthought\ntruncated by max_tokens';
  assert.strictEqual(stripThinkingTokens(input), '');
});

test('stripThinkingTokens: Gemma 4 PUA block with closing > variant', () => {
  const input = '\uf06cthought\nthinking...\n!</thought>\nReply here';
  assert.strictEqual(stripThinkingTokens(input), 'Reply here');
});

test('stripThinkingTokens: Gemma 4 PUA only in content leaves empty', () => {
  const input = '\uf06cthought\n!</thought';
  assert.strictEqual(stripThinkingTokens(input), '');
});

// Text-form orphaned closer/opener (the forms stored when stripping failed)
test('stripThinkingTokens: orphaned text-form closer <channel|> stripped', () => {
  // e.g. model emitted only the closer — the opener was in a previous chunk
  assert.strictEqual(stripThinkingTokens('<channel|>'), '');
});

test('stripThinkingTokens: orphaned text-form opener <|channel> stripped', () => {
  assert.strictEqual(stripThinkingTokens('<|channel>'), '');
});

test('stripThinkingTokens: PUA opener residue \\uf06c stripped', () => {
  // opener was split across chunks; only the PUA char survived
  assert.strictEqual(stripThinkingTokens(''), '');
});

test('stripThinkingTokens: real content with <channel|> orphan preserved', () => {
  // ensure real text before the orphaned closer is kept
  assert.strictEqual(stripThinkingTokens('Hello!<channel|>'), 'Hello!');
});

// ─── getMemoryCategory ───────────────────────────────────────────────────────

test('getMemoryCategory: defaults to fleagle when header absent', () => {
  assert.equal(getMemoryCategory({}), 'fleagle');
});

test('getMemoryCategory: reads X-Memory-Category header, lowercased/trimmed', () => {
  assert.equal(getMemoryCategory({ 'x-memory-category': ' Background ' }), 'background');
});

test('getMemoryCategory: blank header falls back to default', () => {
  assert.equal(getMemoryCategory({ 'x-memory-category': '   ' }), 'fleagle');
});

test('getMemoryCategory: custom fallback is honored', () => {
  assert.equal(getMemoryCategory({}, 'task'), 'task');
});

// ─── formatDuration ──────────────────────────────────────────────────────────

test('formatDuration: minutes only', () => {
  assert.equal(formatDuration(40 * 60), '40m');
});

test('formatDuration: hours and minutes', () => {
  assert.equal(formatDuration(14 * 3600 + 22 * 60), '14h 22m');
});

test('formatDuration: days and hours', () => {
  assert.equal(formatDuration(3 * 86400 + 2 * 3600), '3d 2h');
});

test('formatDuration: negative/zero clamps to 0m', () => {
  assert.equal(formatDuration(-5), '0m');
  assert.equal(formatDuration(0), '0m');
});


// ─── resolveTaskActivitySnapshot ─────────────────────────────────────────────
// Fix 2026-07-23: keeps the downtime/continuity note's basis timestamp stable
// across every tool-call round-trip resend within one agentic task, instead of
// flipping when the assistant's own reply from round 1 gets recorded as "the
// last turn" before round 2 arrives (which broke llama-server's context-
// checkpoint reuse for recurrent/hybrid models — see the function's own
// docstring / proxy.js for the full story).

test('resolveTaskActivitySnapshot: a new user turn reads fresh and caches it', () => {
  const snapshotMap = new Map();
  let reads = 0;
  const readFresh = () => { reads += 1; return 12345; };

  const result = resolveTaskActivitySnapshot(true, 'sess-a', snapshotMap, readFresh);

  assert.equal(result, 12345);
  assert.equal(reads, 1);
  assert.equal(snapshotMap.get('sess-a'), 12345);
});

test('resolveTaskActivitySnapshot: a tool-call round-trip resend reuses the cached snapshot without a fresh read', () => {
  const snapshotMap = new Map([['sess-a', 12345]]);
  let reads = 0;
  const readFresh = () => { reads += 1; return 99999; }; // would be wrong if used

  const result = resolveTaskActivitySnapshot(false, 'sess-a', snapshotMap, readFresh);

  assert.equal(result, 12345);
  assert.equal(reads, 0, 'must not call readFreshTimestamp on a cached hit');
});

test('resolveTaskActivitySnapshot: a resend for a session never seen this process falls back to a fresh read', () => {
  const snapshotMap = new Map();
  let reads = 0;
  const readFresh = () => { reads += 1; return 42; };

  const result = resolveTaskActivitySnapshot(false, 'sess-new', snapshotMap, readFresh);

  assert.equal(result, 42);
  assert.equal(reads, 1);
  // Does not backfill the map -- only a genuinely new user turn should own
  // that write, so a second resend for the same never-seen session falls
  // back to a fresh read again rather than silently caching a guess.
  assert.equal(snapshotMap.has('sess-new'), false);
});

test('resolveTaskActivitySnapshot: null (brand-new session, no prior turns) is cached and reused just like a real timestamp', () => {
  const snapshotMap = new Map();
  const readFresh = () => null;

  const first = resolveTaskActivitySnapshot(true, 'sess-b', snapshotMap, readFresh);
  const second = resolveTaskActivitySnapshot(false, 'sess-b', snapshotMap, readFresh);

  assert.equal(first, null);
  assert.equal(second, null);
  assert.equal(snapshotMap.has('sess-b'), true, 'null must still be cached, not treated as "no entry"');
});

test('resolveTaskActivitySnapshot: a later new user turn refreshes the snapshot (does not stick forever)', () => {
  const snapshotMap = new Map();
  const timestamps = [100, 200];
  const readFresh = () => timestamps.shift();

  const turn1 = resolveTaskActivitySnapshot(true, 'sess-c', snapshotMap, readFresh);
  const midTaskResend = resolveTaskActivitySnapshot(false, 'sess-c', snapshotMap, readFresh);
  const turn2 = resolveTaskActivitySnapshot(true, 'sess-c', snapshotMap, readFresh);

  assert.equal(turn1, 100);
  assert.equal(midTaskResend, 100, 'stays pinned to turn 1s snapshot for the whole task');
  assert.equal(turn2, 200, 'a genuinely new user turn is free to move the snapshot forward');
});

test('resolveTaskActivitySnapshot: independent sessions do not clobber each other', () => {
  const snapshotMap = new Map();
  resolveTaskActivitySnapshot(true, 'sess-x', snapshotMap, () => 111);
  resolveTaskActivitySnapshot(true, 'sess-y', snapshotMap, () => 222);

  assert.equal(resolveTaskActivitySnapshot(false, 'sess-x', snapshotMap, () => { throw new Error('must not be called'); }), 111);
  assert.equal(resolveTaskActivitySnapshot(false, 'sess-y', snapshotMap, () => { throw new Error('must not be called'); }), 222);
});
