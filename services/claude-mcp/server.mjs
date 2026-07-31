import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { spawn } from "child_process";
import { createServer } from "http";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { createHash, randomBytes, timingSafeEqual } from "crypto";
import { z } from "zod";

const PORT = process.env.PORT || 3456;
const VERSION = "0.3.0";

// ── CLAUDE.md → default system prompt ───────────────────────────────────────

// fileURLToPath, not URL.pathname.slice(1): pathname yields "/C:/…" on Windows
// (where slice(1) was a fix) but "/home/…" on POSIX, where slicing the leading
// "/" turned it into a CWD-relative path — so on Linux/macOS the prompt never
// loaded and config.json was (re)created at a garbage path unless CWD was "/".
function here(rel) {
  return fileURLToPath(new URL(rel, import.meta.url));
}

function loadSystemPrompt() {
  const claudeMd = here("./CLAUDE.md");
  if (existsSync(claudeMd)) {
    console.log("  Loaded CLAUDE.md as system prompt.");
    return readFileSync(claudeMd, "utf8");
  }
  return process.env.SYSTEM_PROMPT || "";
}

const SYSTEM_PROMPT = loadSystemPrompt();

// ── Config (token + rate limit) ──────────────────────────────────────────────

const CONFIG_FILE = here("./config.json");

function loadConfig() {
  if (!existsSync(CONFIG_FILE)) {
    const token = randomBytes(24).toString("hex");
    const cfg = { token, requestsPerHour: 20 };
    writeFileSync(CONFIG_FILE, JSON.stringify(cfg, null, 2));
    console.log(`\n🔑  Generated bearer token: ${token}`);
    console.log(`    Share this with the group — keep it secret!\n`);
    return cfg;
  }
  let parsed;
  try {
    parsed = JSON.parse(readFileSync(CONFIG_FILE, "utf8"));
  } catch (e) {
    // Don't regenerate on a corrupt file: that would silently rotate the token
    // and lock out every configured client. Fail loudly instead.
    console.error(`Corrupt ${CONFIG_FILE}: ${e.message}`);
    console.error("Fix it by hand, or delete it to generate a NEW token (clients must update).");
    process.exit(1);
  }
  if (!parsed.token) {
    console.error(`${CONFIG_FILE} has no "token" — delete the file to generate one.`);
    process.exit(1);
  }
  // Older config files predate requestsPerHour; without this default the
  // limiter compared against undefined and never fired.
  return { requestsPerHour: 20, ...parsed };
}

const config = loadConfig();

// ── Rate limiter (rolling window) ────────────────────────────────────────────
// Guards *Claude invocations* (the expensive, subscription-metered thing), not
// protocol plumbing: the MCP handshake + tools/list used to count against the
// hourly budget, so agents that mount at startup could exhaust it without ever
// asking Claude anything.

const requestLog = [];

function isRateLimited() {
  const now = Date.now();
  const windowMs = 60 * 60 * 1000;
  while (requestLog.length && requestLog[0] < now - windowMs) requestLog.shift();
  if (requestLog.length >= config.requestsPerHour) return true;
  requestLog.push(now);
  return false;
}

const RATE_LIMIT_MSG = () => `Rate limit: ${config.requestsPerHour} Claude calls/hour — try again later.`;

// ── Auth ─────────────────────────────────────────────────────────────────────

function safeEqual(a, b) {
  // Hash both sides so timingSafeEqual gets equal-length buffers regardless
  // of attacker-controlled input length.
  const ha = createHash("sha256").update(String(a)).digest();
  const hb = createHash("sha256").update(String(b)).digest();
  return timingSafeEqual(ha, hb);
}

// Two ways to present the token:
//   1. Authorization: Bearer <token>            (MCP clients, Anamnesis upstream.apiKey)
//   2. a /t/<token>/ URL prefix on any route    (clients that can only carry a
//      base_url, e.g. Pleiades' "openai" tier: base_url http://host:3456/t/<token>/v1)
function isAuthorized(req, pathToken = "") {
  if (pathToken && safeEqual(pathToken, config.token)) return true;
  const auth = req.headers["authorization"] || "";
  return safeEqual(auth, `Bearer ${config.token}`);
}

// ── Claude CLI ───────────────────────────────────────────────────────────

// IMPORTANT (Windows): point at the real claude.exe, NOT claude.cmd.
// The .cmd wrapper npm installs requires shell:true to spawn at all, and
// shell:true makes Node naively string-join argv into one command line
// for cmd.exe to re-parse — which silently truncates/mangles any
// multi-line argument (like a whole CLAUDE.md system prompt) at the
// first newline. Calling the real .exe directly needs no shell, so argv
// (including embedded newlines) passes through the Windows process API
// intact. Confirmed via a marker-word test: a multi-line system prompt
// with a planted secret word round-tripped correctly only after this fix.
// Find it via the claude.cmd wrapper's own %dp0%\node_modules\...\bin\claude.exe reference.
// On POSIX the npm global install is a plain symlink to an executable JS file,
// so a bare "claude" resolved via PATH spawns fine without a shell.
const CLAUDE_BIN = process.env.CLAUDE_BIN ||
  (process.platform === "win32"
    ? `${process.env.APPDATA}\\npm\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe`
    : "claude");

const CLAUDE_TIMEOUT_MS = 120_000;

console.log(`  Claude binary: ${CLAUDE_BIN}`);

// Async (non-blocking) — spawnSync would freeze the whole event loop for
// every other caller until claude.exe returns, which is unacceptable once
// more than one person (Nyzkh, Fleagle, etc.) can hit this server at once.
function askClaude(prompt, systemPrompt = SYSTEM_PROMPT) {
  return new Promise((resolve) => {
    const args = ["-p", prompt];
    if (systemPrompt) args.push("--system-prompt", systemPrompt);

    const child = spawn(CLAUDE_BIN, args, { timeout: CLAUDE_TIMEOUT_MS });
    let stdout = "", stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", (e) => {
      // Most common cause: CLI not installed / wrong CLAUDE_BIN. Log it so the
      // operator sees the real problem, not just callers getting error text.
      console.error(`[askClaude] spawn failed: ${e.message}`);
      resolve(`Error: could not run the Claude CLI (${e.message}). ` +
              `Check it is installed and CLAUDE_BIN / PATH is correct.`);
    });
    child.on("close", (code, signal) => {
      if (signal) {
        // spawn's timeout option kills the child with a signal.
        resolve(`Error: Claude CLI timed out after ${CLAUDE_TIMEOUT_MS / 1000}s.`);
        return;
      }
      resolve(stdout.trim() || stderr.trim() || "No response");
    });
  });
}

// ── MCP server ──────────────────────────────────────────────────────────────

// One McpServer PER SSE CONNECTION. The SDK's Protocol.connect() binds a
// single transport; sharing one instance across sessions made a second
// client's connect steal the first client's reply channel, so concurrent
// users received each other's responses.
function buildMcp() {
  const mcp = new McpServer({ name: "vern-claude", version: VERSION });
  mcp.tool(
    "ask_claude",
    "Ask Vern's Claude anything",
    { prompt: z.string().describe("Your question or task") },
    async ({ prompt }) => {
      if (isRateLimited()) {
        return { content: [{ type: "text", text: RATE_LIMIT_MSG() }], isError: true };
      }
      return { content: [{ type: "text", text: await askClaude(prompt) }] };
    }
  );
  return mcp;
}

// ── OpenAI-compatible chat completions (for Anamnesis / Pleiades) ────────────

function flattenContent(content) {
  return Array.isArray(content)
    ? content.filter(c => c.type === "text").map(c => c.text).join("\n")
    : (content ?? "");
}

function formatMessagesForClaude(messages) {
  const systemMsg = messages.find(m => m.role === "system");
  const chat = messages.filter(m => m.role !== "system");
  const conversation = chat.map(m => {
    const role = m.role === "user" ? "Human" : "Assistant";
    return `${role}: ${flattenContent(m.content)}`;
  }).join("\n\n");
  return { prompt: conversation, system: flattenContent(systemMsg?.content) || SYSTEM_PROMPT };
}

function handleChatCompletions(req, res, pathToken) {
  if (!isAuthorized(req, pathToken)) { res.writeHead(401); res.end("Unauthorized"); return; }
  if (isRateLimited()) {
    res.writeHead(429); res.end(RATE_LIMIT_MSG()); return;
  }
  let body = "";
  req.on("data", c => body += c);
  req.on("end", async () => {
    try {
      const { messages = [], model, stream } = JSON.parse(body);
      const { prompt, system } = formatMessagesForClaude(messages);
      const text = await askClaude(prompt, system);
      const id = `chatcmpl-${Date.now()}`;
      const created = Math.floor(Date.now() / 1000);
      const mdl = model || "claude";
      if (stream) {
        // The CLI hands us the full text at once, so "streaming" is one content
        // chunk + [DONE] — it exists so stream:true clients get a valid SSE body
        // instead of choking on unexpected JSON.
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        });
        const chunk = (delta, finish = null) => res.write("data: " + JSON.stringify({
          id, object: "chat.completion.chunk", created, model: mdl,
          choices: [{ index: 0, delta, finish_reason: finish }],
        }) + "\n\n");
        chunk({ role: "assistant", content: text });
        chunk({}, "stop");
        res.write("data: [DONE]\n\n");
        res.end();
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        id,
        object: "chat.completion",
        created,
        model: mdl,
        choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      }));
    } catch (e) { res.writeHead(500); res.end(e.message); }
  });
}

// ── HTTP server ───────────────────────────────────────────────────────────────

const sessions = {};

const http = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  // Strip an optional /t/<token>/ auth prefix before routing (see isAuthorized).
  let pathname = url.pathname;
  let pathToken = "";
  if (pathname.startsWith("/t/")) {
    const rest = pathname.slice(3);
    const slash = rest.indexOf("/");
    pathToken = slash === -1 ? rest : rest.slice(0, slash);
    pathname = slash === -1 ? "/" : rest.slice(slash);
  }

  if (req.method === "GET" && pathname === "/sse") {
    if (!isAuthorized(req, pathToken)) { res.writeHead(401); res.end("Unauthorized"); return; }
    // Advertise the endpoint WITH the token prefix when the client connected
    // through one, so URL-token clients can POST back without headers.
    const endpoint = pathToken ? `/t/${pathToken}/messages` : "/messages";
    const transport = new SSEServerTransport(endpoint, res);
    sessions[transport.sessionId] = transport;
    transport.onclose = () => delete sessions[transport.sessionId];
    try {
      await buildMcp().connect(transport);
    } catch (e) {
      delete sessions[transport.sessionId];
      console.error(`[sse] connect failed: ${e.message}`);
      if (!res.headersSent) { res.writeHead(500); res.end(); }
    }

  } else if (req.method === "POST" && pathname === "/messages") {
    if (!isAuthorized(req, pathToken)) { res.writeHead(401); res.end("Unauthorized"); return; }
    // No rate limit here: these are JSON-RPC frames (handshake, tools/list, …).
    // Claude usage is limited inside the ask_claude handler instead.
    const sessionId = url.searchParams.get("sessionId");
    const transport = sessions[sessionId];
    if (!transport) { res.writeHead(404); res.end("Session not found"); return; }
    let body = "";
    req.on("data", c => body += c);
    req.on("end", async () => {
      // Guarded: an unparseable body / transport error used to throw in this
      // callback and take down the whole process.
      try {
        await transport.handlePostMessage(req, res, JSON.parse(body));
      } catch (e) {
        console.error(`[messages] ${e.message}`);
        if (!res.headersSent) { res.writeHead(400); res.end("Bad request"); }
      }
    });

  } else if (req.method === "POST" && pathname === "/v1/chat/completions") {
    handleChatCompletions(req, res, pathToken);

  } else if (req.method === "GET" && pathname === "/v1/models") {
    // Intentionally auth'd but NOT rate-limited: it's a cheap static list that
    // OpenAI clients poll for discovery; only Claude invocations are metered.
    if (!isAuthorized(req, pathToken)) { res.writeHead(401); res.end("Unauthorized"); return; }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ object: "list", data: [{ id: "claude", object: "model", created: 0, owned_by: "vern" }] }));

  } else {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("Vern's Claude MCP Server is running :3");
  }
});

http.listen(PORT, "0.0.0.0", () => {
  console.log(`\n🤖  Vern's Claude — v${VERSION}`);
  console.log(`    MCP  → http://0.0.0.0:${PORT}/sse`);
  console.log(`    OAI  → http://0.0.0.0:${PORT}/v1/chat/completions`);
  console.log(`    ping → http://0.0.0.0:${PORT}/\n`);
});
