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

export type AssistantItem = TextItem | ToolItem

export interface TurnMeta {
  tokens?: number
  seconds?: number
  tps?: number
  stopped?: boolean
}

export interface UserMessage {
  role: 'user'
  content: string
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
// `delta.reasoning_content`/`reasoning`) that the docstring omits. Unlike
// `token`, the server-side persistence loop in chats_message() never folds
// `reasoning` into the saved turn's `items` — it's forwarded to the client
// live and then dropped, so it only matters for in-flight rendering.
export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'tool_call'; name: string; args: string }
  | { type: 'tool_result'; name: string; output: string; ok: boolean }
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

export interface BrowserViewStatus {
  character: string
  status: 'stopped' | 'starting' | 'running' | 'error'
  error: string | null
  url: string | null
  viewport: { width: number; height: number }
  interactive: boolean
  backend: string
}

export interface CloudSearchResponse {
  source: string
  results: CloudSearchResult[]
  error?: string
  detail?: string
}
