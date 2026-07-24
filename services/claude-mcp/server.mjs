import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { spawnSync } from "child_process";
import { createServer } from "http";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { randomBytes } from "crypto";
import { z } from "zod";

const PORT = process.env.PORT || 3456;

// ── CLAUDE.md → default system prompt ───────────────────────────────────────

function loadSystemPrompt() {
  const claudeMd = new URL("./CLAUDE.md", import.meta.url).pathname.slice(1);
  if (existsSync(claudeMd)) {
    console.log("  Loaded CLAUDE.md as system prompt.");
    return readFileSync(claudeMd, "utf8");
  }
  return process.env.SYSTEM_PROMPT || "";
}

const SYSTEM_PROMPT = loadSystemPrompt();

// ── Config (token + rate limit) ──────────────────────────────────────────────

const CONFIG_FILE = new URL("./config.json", import.meta.url).pathname.slice(1);

function loadConfig() {
  if (!existsSync(CONFIG_FILE)) {
    const token = randomBytes(24).toString("hex");
    const cfg = { token, requestsPerHour: 20 };
    writeFileSync(CONFIG_FILE, JSON.stringify(cfg, null, 2));
    console.log(`\n🔑  Generated bearer token: ${token}`);
    console.log(`    Share this with the group — keep it secret!\n`);
    return cfg;
  }
  return JSON.parse(readFileSync(CONFIG_FILE, "utf8"));
}

const config = loadConfig();

// ── Rate limiter (rolling window) ────────────────────────────────────────────

const requestLog = [];

function isRateLimited() {
  const now = Date.now();
  const windowMs = 60 * 60 * 1000;
  while (requestLog.length && requestLog[0] < now - windowMs) requestLog.shift();
  if (requestLog.length >= config.requestsPerHour) return true;
  requestLog.push(now);
  return false;
}

// ── Auth ─────────────────────────────────────────────────────────────────────

function isAuthorized(req) {
  const auth = req.headers["authorization"] || "";
  return auth === `Bearer ${config.token}`;
}

// ── Claude CLI call ───────────────────────────────────────────────────────────

// Resolve claude binary explicitly so it works when node is spawned without
// the user's full PATH (e.g. Task Scheduler, background Start-Process)
const CLAUDE_BIN = process.env.CLAUDE_BIN ||
  (process.platform === "win32"
    ? `${process.env.APPDATA}\\npm\\claude.cmd`
    : "claude");

function askClaude(prompt, systemPrompt = SYSTEM_PROMPT) {
  const args = ["-p", prompt];
  if (systemPrompt) args.push("--system-prompt", systemPrompt);
  const result = spawnSync(CLAUDE_BIN, args, {
    encoding: "utf8",
    shell: true,
    env: { ...process.env, COMSPEC: "C:\\Windows\\System32\\cmd.exe" },
    timeout: 120_000,
  });
  return result.stdout?.trim() || result.stderr?.trim() || "No response";
}

// ── MCP server ──────────────────────────────────────────────────────────────

const mcp = new McpServer({ name: "vern-claude", version: "0.2.0" });

mcp.tool(
  "ask_claude",
  "Ask Vern's Claude anything",
  { prompt: z.string().describe("Your question or task") },
  async ({ prompt }) => ({
    content: [{ type: "text", text: askClaude(prompt) }],
  })
);

// ── OpenAI-compatible chat completions (for Anamnesis / Pleiades) ────────────

function formatMessagesForClaude(messages) {
  const systemMsg = messages.find(m => m.role === "system");
  const chat = messages.filter(m => m.role !== "system");
  const conversation = chat.map(m => {
    const role = m.role === "user" ? "Human" : "Assistant";
    const content = Array.isArray(m.content)
      ? m.content.filter(c => c.type === "text").map(c => c.text).join("\n")
      : (m.content ?? "");
    return `${role}: ${content}`;
  }).join("\n\n");
  return { prompt: conversation, system: systemMsg?.content || SYSTEM_PROMPT };
}

function handleChatCompletions(req, res) {
  if (!isAuthorized(req)) { res.writeHead(401); res.end("Unauthorized"); return; }
  if (isRateLimited()) {
    res.writeHead(429); res.end(`Rate limit: ${config.requestsPerHour} req/hour`); return;
  }
  let body = "";
  req.on("data", c => body += c);
  req.on("end", () => {
    try {
      const { messages = [], model } = JSON.parse(body);
      const { prompt, system } = formatMessagesForClaude(messages);
      const text = askClaude(prompt, system);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model || "claude",
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

  if (req.method === "GET" && url.pathname === "/sse") {
    if (!isAuthorized(req)) { res.writeHead(401); res.end("Unauthorized"); return; }
    const transport = new SSEServerTransport("/messages", res);
    sessions[transport.sessionId] = transport;
    transport.onclose = () => delete sessions[transport.sessionId];
    await mcp.connect(transport);

  } else if (req.method === "POST" && url.pathname === "/messages") {
    if (!isAuthorized(req)) { res.writeHead(401); res.end("Unauthorized"); return; }
    if (isRateLimited()) { res.writeHead(429); res.end(`Rate limit: ${config.requestsPerHour} req/hour`); return; }
    const sessionId = url.searchParams.get("sessionId");
    const transport = sessions[sessionId];
    if (!transport) { res.writeHead(404); res.end("Session not found"); return; }
    let body = "";
    req.on("data", c => body += c);
    req.on("end", () => transport.handlePostMessage(req, res, JSON.parse(body)));

  } else if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
    handleChatCompletions(req, res);

  } else if (req.method === "GET" && url.pathname === "/v1/models") {
    if (!isAuthorized(req)) { res.writeHead(401); res.end("Unauthorized"); return; }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ object: "list", data: [{ id: "claude", object: "model", created: 0, owned_by: "vern" }] }));

  } else {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("Vern's Claude MCP Server is running :3");
  }
});

http.listen(PORT, "0.0.0.0", () => {
  console.log(`\n🤖  Vern's Claude — v0.2.0`);
  console.log(`    MCP  → http://0.0.0.0:${PORT}/sse`);
  console.log(`    OAI  → http://0.0.0.0:${PORT}/v1/chat/completions`);
  console.log(`    ping → http://0.0.0.0:${PORT}/\n`);
});
