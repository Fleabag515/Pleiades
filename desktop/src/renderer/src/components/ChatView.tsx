import { useEffect, useRef, useState } from 'react'
import { getApproval, getChat, postApproval, stopChat, streamMessage } from '../lib/api'
import type { AssistantItem, ChatDetail, ChatMessageEntry, PendingApproval } from '../lib/types'
import Avatar from './Avatar'
import ApprovalCard from './ApprovalCard'
import ModelBadge from './ModelBadge'
import Composer from './Composer'
import EmptyState from './EmptyState'
import MessageBubble from './MessageBubble'

interface ChatViewProps {
  base: string
  chat: ChatDetail | null
  activeCharacter: string
  onNewChat: () => void
  onChatChanged: () => void
}

/** Main pane: top bar, streaming message list, composer, inline tool
 * approval gate. Owns the fetch+ReadableStream NDJSON parsing loop against
 * POST /api/chats/{id}/message. */
function ChatView({ base, chat, activeCharacter, onNewChat, onChatChanged }: ChatViewProps): React.JSX.Element {
  const [history, setHistory] = useState<ChatMessageEntry[]>(chat?.messages ?? [])
  const [draft, setDraft] = useState<AssistantMessageDraft | null>(null)
  // `reasoning` (thinking-model output) is never persisted server-side (see
  // the StreamEvent comment in lib/types.ts), so it lives only here and is
  // dropped once we reconcile with the canonical transcript after the turn.
  const [reasoning, setReasoning] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [approval, setApproval] = useState<PendingApproval | null>(null)
  const [approvalBusy, setApprovalBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    setHistory(chat?.messages ?? [])
    setDraft(null)
    setReasoning('')
    setError(null)
    setApproval(null)
  }, [chat?.id])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [history, draft, approval])

  // Poll the approval gate only while a turn is in flight.
  useEffect(() => {
    if (!streaming || !chat) return
    let cancelled = false
    const tick = async (): Promise<void> => {
      try {
        const pending = await getApproval(base, chat.id)
        if (!cancelled) setApproval(pending)
      } catch {
        // ignore transient poll failures
      }
    }
    const id = setInterval(tick, 900)
    tick()
    return () => clearInterval(id)
  }, [streaming, chat, base])

  if (!chat) {
    return (
      <div className="flex h-full flex-1 flex-col bg-bg-app">
        <EmptyState characterName={activeCharacter} onNewChat={onNewChat} />
      </div>
    )
  }

  const send = async (text: string): Promise<void> => {
    setError(null)
    setReasoning('')
    const userMsg: ChatMessageEntry = { role: 'user', content: text }
    setHistory((h) => [...h, userMsg])
    setDraft({ items: [], meta: {} })
    setStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    let items: AssistantItem[] = []
    let textAcc = ''

    const flushText = (): void => {
      if (textAcc) {
        items = [...items, { t: 'text', text: textAcc }]
        textAcc = ''
      }
    }

    try {
      await streamMessage(
        base,
        chat.id,
        text,
        (evt) => {
          if (evt.type === 'reasoning') {
            setReasoning((r) => r + evt.text)
          } else if (evt.type === 'token') {
            textAcc += evt.text
            setDraft({ items: [...items, ...(textAcc ? [{ t: 'text', text: textAcc } as AssistantItem] : [])], meta: {} })
          } else if (evt.type === 'tool_call') {
            flushText()
            items = [...items, { t: 'tool', name: evt.name, args: evt.args, output: null, ok: null }]
            setDraft({ items, meta: {} })
          } else if (evt.type === 'tool_result') {
            for (let i = items.length - 1; i >= 0; i--) {
              const it = items[i]
              if (it.t === 'tool' && it.output === null) {
                items = items.map((x, idx) => (idx === i ? { ...it, output: evt.output.slice(0, 4000), ok: evt.ok } : x))
                break
              }
            }
            setDraft({ items, meta: {} })
          } else if (evt.type === 'done') {
            flushText()
            setDraft({ items, meta: { tokens: evt.tokens, seconds: evt.seconds, tps: evt.tps } })
          } else if (evt.type === 'stopped') {
            flushText()
            setDraft({ items, meta: { stopped: true } })
          } else if (evt.type === 'error') {
            flushText()
            setError(evt.error)
          }
        },
        controller.signal
      )
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      setStreaming(false)
      setApproval(null)
      abortRef.current = null
      // Reconcile with the server's persisted transcript (it may have set
      // the chat title from this first message, etc.) rather than trusting
      // our locally-reconstructed draft.
      try {
        const fresh = await getChat(base, chat.id)
        setHistory(fresh.messages)
        setDraft(null)
        setReasoning('')
      } catch {
        // keep the local optimistic render if the refetch fails
      }
      onChatChanged()
    }
  }

  const stop = async (): Promise<void> => {
    try {
      await stopChat(base, chat.id)
    } catch {
      // best-effort
    }
  }

  const approve = async (ok: boolean): Promise<void> => {
    setApprovalBusy(true)
    try {
      await postApproval(base, chat.id, ok)
      setApproval(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setApprovalBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-1 flex-col bg-bg-app">
      <div className="flex h-14 flex-none items-center gap-2.5 border-b border-border px-5">
        <Avatar base={base} character={chat.character} size={24} />
        <span className="font-medium text-ink-bright">{chat.character}</span>
        <ModelBadge base={base} character={chat.character} />
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto py-3">
        {history.map((m, i) => (
          <MessageBubble key={i} message={m} base={base} character={chat.character} />
        ))}
        {reasoning && streaming && (
          <div className="mx-5 my-1 max-h-32 overflow-y-auto rounded-xl bg-bg-100/70 px-3 py-2 text-xs italic text-ink-dim">
            {reasoning}
          </div>
        )}
        {draft && (
          <MessageBubble
            message={{ role: 'assistant', items: draft.items, meta: draft.meta }}
            base={base}
            character={chat.character}
            streaming
          />
        )}
        {approval && (
          <div className="px-5 py-2">
            <ApprovalCard
              approval={approval}
              busy={approvalBusy}
              onApprove={() => approve(true)}
              onDeny={() => approve(false)}
            />
          </div>
        )}
        {error && (
          <div className="mx-5 my-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
            {error}
          </div>
        )}
      </div>

      <Composer disabled={!chat} streaming={streaming} onSend={send} onStop={stop} />
    </div>
  )
}

interface AssistantMessageDraft {
  items: AssistantItem[]
  meta: { tokens?: number; seconds?: number; tps?: number; stopped?: boolean }
}

export default ChatView
