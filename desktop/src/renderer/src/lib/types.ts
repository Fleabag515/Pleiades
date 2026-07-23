// Shared frontend types mirroring pleiades/webui/server.py response shapes.
// Field names verified directly against server.py, pleiades/chats.py,
// pleiades/profiles.py and pleiades/models.py (2026-07-21).

export interface Profile {
  name: string
  email_address: string
  imap_host: string
  imap_port: number
  smtp_host: string
  smtp_port: number
  discord_enabled: boolean
  discord_require_mention: boolean
  discord_respond_to_bots: boolean
  discord_allowed_channels: string
  persona_source: string
  model: string
  has_email: boolean
  // Per-character tool-call approval override (composer's Approve/Ask
  // dropdown; see pleiades/profiles.py Profile.exec_policy). null means
  // "not configured" -- ProfileManager migrates any profile.json missing
  // this key to "allow" on first read, so in practice this is always a
  // real string for any profile loaded through the API, but the type stays
  // nullable to match what the backend can technically return.
  exec_policy: 'allow' | 'ask' | 'deny' | null
  // Agents panel toggle + model pick (see pleiades/profiles.py
  // Profile.agents_enabled/agents_model). Default false/"" for every
  // character unless explicitly turned on.
  agents_enabled: boolean
  agents_model: string
}

export interface ChatSummary {
  id: string
  character: string
  title: string
  updated: number
  messages: number
}

export interface ToolItem {
  t: 'tool'
  name: string
  args: string
  output: string | null
  ok: boolean | null
}

export interface TextItem {
  t: 'text'
  text: string
}

export interface ReasoningItem {
  t: 'reasoning'
  text: string
}

export interface UserInjectedItem {
  t: 'user_injected'
  text: string
}

export type AssistantItem = TextItem | ToolItem | ReasoningItem | UserInjectedItem

export interface TurnMeta {
  tokens?: number
  seconds?: number
  tps?: number
  stopped?: boolean
}

// A real file the user attached to a message (see pleiades/attachments.py +
// POST /api/chats/{id}/attachments) -- `path` is an absolute path on THIS
// machine, injected into the turn's context so the character's file/shell
// tools can act on it directly (see engine.py's `_attachments_note`).
// MessageBubble.tsx renders a chip for each of these on the user message
// they're attached to -- `name` doubles as the id GET
// /api/chats/{id}/attachments/{name} (lib/api.ts's `attachmentUrl`) needs to
// stream the real bytes back for an image thumbnail/`<audio>` player.
export interface MessageAttachment {
  name: string
  path: string
  mime: string
}

export interface UserMessage {
  role: 'user'
  content: string
  attachments?: MessageAttachment[]
}

export interface AssistantMessage {
  role: 'assistant'
  items: AssistantItem[]
  meta: TurnMeta
}

export type ChatMessageEntry = UserMessage | AssistantMessage

export interface ChatDetail {
  id: string
  character: string
  title: string
  created: number
  updated: number
  messages: ChatMessageEntry[]
}

export interface ModelEntry {
  name: string
  path: string
  n_ctx: number | 'auto'
  n_gpu_layers: number | 'auto'
  chat_format: string
  // UI-only rename override (pleiades/models.py Model.display_name, added
  // alongside this feature) — blank/absent means "show `name`". Never used
  // as a registry key; start/stop/delete/logs all still address by `name`.
  display_name?: string
  port: number
  state: 'running' | 'loading' | 'crashed' | 'stopped'
  running: boolean
}

export interface ApiStatus {
  services?: {
    anamnesis?: { up: boolean; url?: string; characters?: number }
    inference?: { up: boolean; url?: string; model_path?: string; state?: string }
    searxng?: { up: boolean; url?: string }
  }
  counts?: {
    profiles?: number
    models?: number
    models_running?: number
    orphans?: number
  }
  running_models?: { name: string; port: number; n_gpu_layers: number | string }[]
}

// NDJSON stream event union from POST /api/chats/{id}/message. Verified
// against the live engine, not just the server.py docstring: thinking
// models also emit `reasoning` events (engine.py yields these from
// `delta.reasoning_content`/`reasoning`) that the docstring omits.
// `chats_message()`'s persistence loop now folds `reasoning` bursts into the
// saved turn's `items` as their own `ReasoningItem`s (same flush-on-transition
// pattern as `text`/`tool`), so a reasoning block survives reload and each
// distinct burst within a turn becomes its own sequential item — it used to
// be forwarded to the client live and then silently dropped.
export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'tool_call'; name: string; args: string }
  | { type: 'tool_result'; name: string; output: string; ok: boolean }
  // A message sent via POST .../interject while this turn was in flight,
  // woven in by Engine.stream_events' poll_injections at a safe round/tool
  // boundary (see engine.py) -- rendered inline, in order, like the rest of
  // the turn's events.
  | { type: 'user_injected'; text: string }
  // An interjection that arrived too late to be woven into the turn (it was
  // still sitting in the chat's inbox when the turn ended) -- never silently
  // dropped; the frontend should offer to resend it as a fresh message.
  | { type: 'undelivered'; text: string }
  | { type: 'done'; tokens: number; seconds: number; tps: number }
  | { type: 'stopped' }
  | { type: 'error'; error: string }
  | { type: 'ping' }

export interface PendingApproval {
  tool: string
  args: string
}

export interface VaultEntryMeta {
  key: string
  meta: { note?: string; reserved?: boolean } | null
  reserved: boolean
  site: string | null
  created_at: number
  updated_at: number
}

export interface ModelInfo {
  registered: boolean
  running: boolean
}

export interface ProfileDetail extends Profile {
  browser_dir: string
  vault: VaultEntryMeta[]
  model_info?: ModelInfo
}

export interface EmailPreset {
  imap_host: string
  imap_port: number
  smtp_host: string
  smtp_port: number
  note?: string
}

export type EmailPresets = Record<string, EmailPreset>

export interface DiscordInfo {
  configured: boolean
  valid?: boolean
  error?: string
  bot?: { username: string; id: string }
  guilds?: { id: string; name: string }[]
}

export interface GpuInfo {
  vendor: string
  name: string
  vram_total: number
  vram_free: number
}

export interface ModelPlan {
  model: string
  n_gpu_layers: number | string
  n_layers: number
  strategy: string
  est_tps: number
  moe: boolean
  n_cpu_moe: number
  reason: string
  fits_fully: boolean
}

export interface RuntimeStatus {
  native: string | null
  moe_offload: boolean
  version?: string
}

export interface HardwareInfo {
  gpus: GpuInfo[]
  ram_total: number
  ram_available: number
  cpu_threads: number
  unified_memory: boolean
  summary: string
  plans: ModelPlan[]
  runtime: RuntimeStatus
}

// ---- Model Foundry (Phase G): HF search, quant planning, downloads, cloud ----

export interface HfSearchResult {
  id: string
  downloads: number
  likes: number
}

export interface QuantOption {
  quant: string
  file: string
  size: number
  quality: number
  strategy: string
  n_gpu_layers: number | string
  est_tps: number
  vram_used: number
  fits_fully: boolean
  feasible: boolean
  reason: string
}

export interface QuantOptionsResponse {
  repo: string
  preference: string
  n_layers: number
  is_moe: boolean
  hardware: {
    summary: string
    gpu: string
    vendor: string
    vram_total: number
    vram_free: number
    ram_available: number
  }
  recommended: string | null
  options: QuantOption[]
}

export interface FetchStatus {
  status: 'idle' | 'downloading' | 'done' | 'error'
  repo?: string
  error?: string
  file?: string
  done?: number
  total?: number
  result?: ModelEntry
}

export interface CloudSearchResult {
  id: string
  name: string
  context?: number | null
  prompt_price?: string | number | null
  completion_price?: string | number | null
  is_free?: boolean
}

// ---- Right panel (Phase 14): work/progress jobs, browser-use embed -------

export interface WorkJobSummary {
  id: string
  task: string
  character: string
  status: 'running' | 'done' | 'error' | 'cancelled'
  started: number
  pending_approval: { tool: string; args: string } | null
}

export interface WorkEvent {
  kind: 'tool_call' | 'tool_result' | 'text'
  ts: number
  name?: string
  ok?: boolean
  output?: string
  text?: string
}

export interface WorkJobDetail extends WorkJobSummary {
  finished: number | null
  result: string
  error: string
  steps: number
  events: WorkEvent[]
  event_count: number
}

export interface WorkTask {
  id: string
  subject: string
  description: string
  activeForm: string
  status: 'pending' | 'in_progress' | 'completed'
  created: number
  updated: number
}

export interface ScheduledTask {
  id: string
  task: string
  character: string
  cron: string
  fire_at: number
  tier: string
  policy: string
  enabled: boolean
  created: number
  last_run: number
  next_run: number
  last_job_id: string
}

// ---- Right panel: character tool belt -----------------------------------

export interface ToolInfo {
  name: string
  description: string
  safe: boolean
  status: 'available' | 'needs_approval' | 'blocked'
  use_count: number
  last_used: number | null
}

export interface ProfileTools {
  character: string
  exec_policy: string
  tools: ToolInfo[]
  bridge_count: number
}

// ---- Right panel: Agents (ephemeral background sub-agents, pleiades/agents.py) --

export interface AgentSummary {
  id: string
  task: string
  character: string
  model: string
  status: 'queued' | 'running' | 'done' | 'error' | 'stopped'
  result: string
  error: string
  started: number
  finished: number | null
  steps: number
}

// Same event shape as WorkEvent (kind/ts/name/ok/output/text) -- the backend
// deliberately mirrors work_jobs' event schema so eventLine()-style rendering
// works unchanged for both.
export type AgentEvent = WorkEvent

export interface AgentDetail extends AgentSummary {
  events: AgentEvent[]
  event_count: number
}

export interface BrowserViewStatus {
  character: string
  status: 'stopped' | 'starting' | 'running' | 'error'
  error: string | null
  url: string | null
  viewport: { width: number; height: number }
  interactive: boolean
  backend: string
  headed: boolean
}

export interface CloudSearchResponse {
  source: string
  results: CloudSearchResult[]
  error?: string
  detail?: string
}
