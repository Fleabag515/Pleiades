# Pleiades UI — workstation prototype

`workstation-prototype.html` is the approved redesign direction (open it in a browser).
It's a self-contained, **mock-data** prototype — not yet wired to the backend.

## What it establishes
- **Workstation-first**, not chat-first. Hero surface is the **model foundry**: Hugging Face
  GGUF search → hardware-aware autofit (optimal quant for the exact GPU/VRAM, predicted tps,
  GPU/CPU layer placement, MoE/RAM-spill, speed↔quality dial) → progressive download → library.
- **Command deck** (hardware/services/models/characters monitoring; Monitor folded in here + Talk rail).
- **Characters**: model + email (IMAP/SMTP presets) + Discord + Anamnesis memory + avatar/persona,
  with vault secrets in their own modal.
- **Tools & connectors**: 75 tools in collapsible category grids with hover "what's this" tips;
  Ask↔Act exec-policy toggle; connectors (email/Discord/Camoufox/SearXNG/Anamnesis) with
  start/stop/fix/debug/settings/edit actions.
- **Claude lab**: subscription account connect + model dropdown (incl. claude-fable-5), a
  build/debug side-chat, tool-audit, and an optional **debug-only** brain. Local models remain
  the only default chat/agent brains.
- **Talk**: one unified surface (chat == agent, all tools always on) with model info + recent
  events in-room. Three views of the model working — Stream, **Pipeline** (default; animated,
  wraps vertically, click any node for args/output/timing), and Mind (synchronized
  thought/tool/output timeline).

## Next step
Implement against the live FastAPI endpoints (`/api/models/hf-search`, `/api/models/fetch`,
`/api/hardware` + autofit, `/api/profiles/*`, `/api/chats/*`) and replace `pleiades/webui` once verified.
