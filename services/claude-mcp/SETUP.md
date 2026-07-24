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

## Customise identity

Edit `CLAUDE.md` — loaded on each server start as the default system prompt.
This is the character's identity layer. Commit it and grow it over time.
