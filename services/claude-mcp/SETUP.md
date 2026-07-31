# Claude MCP Service — Setup

Turns a Claude Code subscription into a shared MCP tool server and
OpenAI-compatible endpoint, reachable over Tailscale. No API key needed.

## Prerequisites

- Node.js ≥ 18
- Claude subscription (Max, Pro, or Team)
- Tailscale

## Install

```powershell
# Windows
$env:COMSPEC = "C:\Windows\System32\cmd.exe"
npm install -g @anthropic-ai/claude-code
claude auth   # opens browser — log in with Anthropic account

cd services/claude-mcp
npm install
node server.mjs
```

```bash
# macOS / Linux
npm install -g @anthropic-ai/claude-code && claude auth
cd services/claude-mcp && npm install && node server.mjs
```

First run generates `config.json` with a random bearer token. Share it with
anyone who should have access. **Never commit config.json.**

## Firewall + auto-start (Windows, run as admin once)

```powershell
# Allow Tailscale peers to reach port 3456
netsh advfirewall firewall add rule name="Claude MCP (Tailscale)" dir=in action=allow protocol=TCP localport=3456

# Start on login
schtasks /create /tn "ClaudeMCP-OnLogin" /tr "pwsh -NonInteractive -WindowStyle Hidden -File '%~dp0watchdog.ps1'" /sc ONLOGON /f

# Watchdog every 3 hours
schtasks /create /tn "ClaudeMCP-Watchdog" /tr "pwsh -NonInteractive -File '%~dp0watchdog.ps1'" /sc HOURLY /mo 3 /f
```

## Connect as MCP tool

```json
{
  "mcpServers": {
    "vern-claude": {
      "url": "http://<tailscale-ip>:3456/sse",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

## Connect as Anamnesis / Pleiades backend

```bash
anamnesis new vern-claude
# edit ~/.anamnesis/characters/vern-claude/config.json:
#   upstream.baseUrl = "http://<tailscale-ip>:3456/v1"
#   upstream.apiKey  = "<token>"
anamnesis start vern-claude
pleiades chat --as vern-claude
```

Anamnesis handles persistent memory automatically.

## Connect natively (no Anamnesis)

Point Pleiades itself at a running claude-mcp server -- either as a backend
tier, or as an MCP tool (`ask_claude`) any agent can call, or both:

```bash
export PLEIADES_CLAUDE_MCP_URL="http://<tailscale-ip>:3456"
export PLEIADES_CLAUDE_MCP_TOKEN="<token from config.json>"
export PLEIADES_CLAUDE_MCP_TOOLS=1   # optional: also mount ask_claude as a tool

pleiades chat --tier claude-mcp      # backend tier
pleiades work "..." --tier claude-mcp   # agent harness, ask_claude tool available if TOOLS=1
```

The same three keys (`claude_mcp_url`, `claude_mcp_token`, `claude_mcp_tools`)
work in `config.json` instead of env vars. No `ANTHROPIC_API_KEY` needed --
the bearer token rides in the URL for the backend tier (`/t/<token>/v1`,
since the `openai` backend sends no `Authorization` header) and in headers
for the MCP connection. See `pleiades/config.py`'s `claude_mcp_*` fields.

## Customise identity

Edit `CLAUDE.md` — loaded on each server start as the default system prompt.
This is the character's identity layer. Commit it and grow it over time.
