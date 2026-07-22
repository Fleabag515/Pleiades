import type { ApiStatus, ChatDetail, ChatSummary, ModelEntry, PendingApproval, Profile, StreamEvent } from './types'

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

export async function listProfiles(base: string): Promise<{ profiles: Profile[]; orphans: string[] }> {
  return asJson(await fetch(`${base}/api/profiles`))
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
