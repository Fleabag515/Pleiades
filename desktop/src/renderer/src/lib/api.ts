import type {
  ApiStatus,
  BrowserViewStatus,
  ChatDetail,
  ChatSummary,
  CloudSearchResponse,
  DiscordInfo,
  EmailPresets,
  FetchStatus,
  HardwareInfo,
  HfSearchResult,
  ModelEntry,
  PendingApproval,
  Profile,
  ProfileDetail,
  QuantOptionsResponse,
  StreamEvent,
  VaultEntryMeta,
  WorkJobDetail,
  WorkJobSummary
} from './types'

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = body?.detail ?? JSON.stringify(body)
    } catch {
      detail = await res.text().catch(() => '')
    }
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<T>
}

export function avatarUrl(base: string, character: string): string {
  return `${base}/api/profiles/${encodeURIComponent(character)}/avatar`
}

export async function getStatus(base: string): Promise<ApiStatus> {
  return asJson(await fetch(`${base}/api/status`))
}

export async function listProfiles(
  base: string
): Promise<{ profiles: Profile[]; orphans: string[] }> {
  return asJson(await fetch(`${base}/api/profiles`))
}

export async function createProfile(
  base: string,
  body: {
    name: string
    email_address?: string
    imap_host?: string
    imap_port?: number
    smtp_host?: string
    smtp_port?: number
    email_password?: string | null
    discord_token?: string | null
    persona_source?: string
  }
): Promise<Profile> {
  return asJson(
    await fetch(`${base}/api/profiles`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  )
}

export async function listChats(base: string): Promise<ChatSummary[]> {
  const data = await asJson<{ chats: ChatSummary[] }>(await fetch(`${base}/api/chats`))
  return data.chats
}

export async function createChat(base: string, character: string): Promise<ChatDetail> {
  return asJson(
    await fetch(`${base}/api/chats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character })
    })
  )
}

export async function getChat(base: string, chatId: string): Promise<ChatDetail> {
  return asJson(await fetch(`${base}/api/chats/${chatId}`))
}

export async function deleteChat(base: string, chatId: string): Promise<void> {
  await fetch(`${base}/api/chats/${chatId}`, { method: 'DELETE' })
}

export async function stopChat(base: string, chatId: string): Promise<void> {
  await fetch(`${base}/api/chats/${chatId}/stop`, { method: 'POST' })
}

export async function getApproval(base: string, chatId: string): Promise<PendingApproval | null> {
  const data = await asJson<{ pending: PendingApproval | null }>(
    await fetch(`${base}/api/chats/${chatId}/approval`)
  )
  return data.pending
}

export async function postApproval(base: string, chatId: string, approve: boolean): Promise<void> {
  await fetch(`${base}/api/chats/${chatId}/approval`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approve })
  })
}

export async function listModels(base: string): Promise<ModelEntry[]> {
  const data = await asJson<{ models: ModelEntry[] }>(await fetch(`${base}/api/models`))
  return data.models
}

export async function startModel(base: string, name: string): Promise<void> {
  await fetch(`${base}/api/models/${encodeURIComponent(name)}/start`, { method: 'POST' })
}

export async function stopModel(base: string, name: string): Promise<void> {
  await fetch(`${base}/api/models/${encodeURIComponent(name)}/stop`, { method: 'POST' })
}

/** PUT /api/models/{name} — currently only used from the UI to set/clear
 * `display_name` (the rename feature), but mirrors the full backend
 * ModelUpdate shape so it isn't a dead end if more fields get editable later. */
export async function updateModel(
  base: string,
  name: string,
  body: { display_name?: string; path?: string; n_ctx?: number | 'auto'; n_gpu_layers?: number | 'auto'; chat_format?: string }
): Promise<ModelEntry> {
  return asJson(
    await fetch(`${base}/api/models/${encodeURIComponent(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  )
}

/** DELETE /api/models/{name} — the backend also deletes the underlying GGUF
 * file from disk (ModelManager.remove(name) defaults delete_file=True), so
 * callers must confirm with the user before calling this; it's not
 * reversible. */
export async function deleteModel(base: string, name: string): Promise<{ ok: boolean }> {
  return asJson(await fetch(`${base}/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' }))
}

export async function modelLogs(base: string, name: string, lines = 250): Promise<{ name: string; state: string; log: string }> {
  return asJson(await fetch(`${base}/api/models/${encodeURIComponent(name)}/logs?lines=${lines}`))
}

/**
 * Streams a chat turn, invoking `onEvent` for each NDJSON line as it arrives.
 * Resolves when the stream ends (naturally or via `signal` abort).
 */
export async function streamMessage(
  base: string,
  chatId: string,
  message: string,
  onEvent: (evt: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${base}/api/chats/${chatId}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal
  })
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let newlineIdx: number
    while ((newlineIdx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, newlineIdx).trim()
      buffer = buffer.slice(newlineIdx + 1)
      if (!line) continue
      try {
        onEvent(JSON.parse(line) as StreamEvent)
      } catch {
        // Ignore malformed lines rather than killing the whole stream.
      }
    }
  }
  const tail = buffer.trim()
  if (tail) {
    try {
      onEvent(JSON.parse(tail) as StreamEvent)
    } catch {
      // ignore
    }
  }
}

export async function getProfile(base: string, name: string): Promise<ProfileDetail> {
  return asJson(await fetch(`${base}/api/profiles/${encodeURIComponent(name)}`))
}

export async function getEmailPresets(base: string): Promise<EmailPresets> {
  return asJson(await fetch(`${base}/api/email/presets`))
}

export async function setEmailConfig(
  base: string,
  name: string,
  body: {
    email_address: string
    imap_host: string
    imap_port: number
    smtp_host: string
    smtp_port: number
    password?: string | null
  }
): Promise<ProfileDetail> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  )
}

export async function setDiscordConfig(
  base: string,
  name: string,
  body: {
    token?: string | null
    enabled: boolean
    require_mention?: boolean
    respond_to_bots?: boolean
    allowed_channels?: string
  }
): Promise<ProfileDetail> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/discord`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  )
}

export async function getDiscordInfo(base: string, name: string): Promise<DiscordInfo> {
  return asJson(await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/discord/info`))
}

export async function assignModel(
  base: string,
  name: string,
  model: string
): Promise<ProfileDetail> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model })
    })
  )
}

export async function listVault(base: string, name: string): Promise<VaultEntryMeta[]> {
  const data = await asJson<{ entries: VaultEntryMeta[] }>(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/vault`)
  )
  return data.entries
}

export async function revealVaultEntry(base: string, name: string, key: string): Promise<string> {
  const data = await asJson<{ key: string; value: string }>(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/vault/${encodeURIComponent(key)}`)
  )
  return data.value
}

export async function setVaultEntry(
  base: string,
  name: string,
  body: { key: string; value: string; note?: string }
): Promise<VaultEntryMeta[]> {
  const data = await asJson<{ entries: VaultEntryMeta[] }>(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/vault`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  )
  return data.entries
}

export async function deleteVaultEntry(base: string, name: string, key: string): Promise<void> {
  await asJson(
    await fetch(
      `${base}/api/profiles/${encodeURIComponent(name)}/vault/${encodeURIComponent(key)}`,
      {
        method: 'DELETE'
      }
    )
  )
}

export async function getHardware(base: string): Promise<HardwareInfo> {
  return asJson(await fetch(`${base}/api/hardware`))
}

export async function hfSearch(base: string, q: string, limit = 8): Promise<HfSearchResult[]> {
  const data = await asJson<{ results: HfSearchResult[] }>(
    await fetch(`${base}/api/models/hf-search?q=${encodeURIComponent(q)}&limit=${limit}`)
  )
  return data.results
}

export async function quantOptions(
  base: string,
  repo: string,
  n_ctx = 8192,
  preference = 'balanced'
): Promise<QuantOptionsResponse> {
  return asJson(
    await fetch(
      `${base}/api/models/quant-options?repo=${encodeURIComponent(repo)}&n_ctx=${n_ctx}&preference=${encodeURIComponent(preference)}`
    )
  )
}

export async function fetchModel(
  base: string,
  body: { repo: string; name?: string; quant?: string }
): Promise<{ ok: boolean }> {
  return asJson(
    await fetch(`${base}/api/models/fetch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: body.repo, name: body.name ?? '', quant: body.quant ?? '' })
    })
  )
}

export async function fetchStatus(base: string): Promise<FetchStatus> {
  return asJson(await fetch(`${base}/api/models/fetch/status`))
}

export async function cloudSearch(
  base: string,
  source: 'openrouter' | 'ollama',
  q = '',
  limit = 25
): Promise<CloudSearchResponse> {
  return asJson(
    await fetch(
      `${base}/api/models/cloud-search?source=${source}&q=${encodeURIComponent(q)}&limit=${limit}`
    )
  )
}

export async function assignCloudModel(
  base: string,
  name: string,
  source: 'openrouter' | 'ollama',
  model: string
): Promise<ProfileDetail> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(name)}/cloud-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, model })
    })
  )
}

// ---- Right panel: work/progress jobs -------------------------------------

export async function listWorkJobs(base: string): Promise<WorkJobSummary[]> {
  const data = await asJson<{ jobs: WorkJobSummary[] }>(await fetch(`${base}/api/work`))
  return data.jobs
}

export async function getWorkJob(base: string, id: string, since = 0): Promise<WorkJobDetail> {
  return asJson(await fetch(`${base}/api/work/${id}?since=${since}`))
}

// ---- Right panel: browser-use embed (Playwright/Chromium, see browser_view.py) ----

export async function browserViewStatus(
  base: string,
  character: string
): Promise<BrowserViewStatus> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(character)}/browser-view/status`)
  )
}

export async function browserViewStart(
  base: string,
  character: string,
  url = ''
): Promise<BrowserViewStatus> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(character)}/browser-view/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    })
  )
}

export async function browserViewStop(base: string, character: string): Promise<BrowserViewStatus> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(character)}/browser-view/stop`, {
      method: 'POST'
    })
  )
}

export async function browserViewNavigate(
  base: string,
  character: string,
  url: string
): Promise<{ ok: boolean; url: string }> {
  return asJson(
    await fetch(`${base}/api/profiles/${encodeURIComponent(character)}/browser-view/navigate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    })
  )
}

/** http(s):// base -> ws(s):// URL for the browser-view WebSocket. */
export function browserViewWsUrl(base: string, character: string): string {
  const wsBase = base.replace(/^http/, 'ws')
  return `${wsBase}/api/profiles/${encodeURIComponent(character)}/browser-view/ws`
}
