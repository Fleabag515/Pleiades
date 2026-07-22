import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { createChat, getApproval, getChat, listChats, postApproval, stopChat, streamMessage } from './api'
import type { AssistantItem, ChatMessageEntry, PendingApproval, TurnMeta } from './types'

export interface AssistantDraft {
  items: AssistantItem[]
  meta: TurnMeta
}

export interface ChatSessionState {
  chatId: string | null
  loading: boolean
  history: ChatMessageEntry[]
  draft: AssistantDraft | null
  reasoning: string
  streaming: boolean
  approval: PendingApproval | null
  approvalBusy: boolean
  error: string | null
}

const EMPTY_SESSION: ChatSessionState = {
  chatId: null,
  loading: false,
  history: [],
  draft: null,
  reasoning: '',
  streaming: false,
  approval: null,
  approvalBusy: false,
  error: null
}

interface ChatStoreContextValue {
  sessions: Record<string, ChatSessionState>
  ensureLiveChat: (base: string, character: string) => Promise<void>
  send: (base: string, character: string, text: string) => Promise<void>
  stop: (base: string, character: string) => Promise<void>
  respondApproval: (base: string, character: string, ok: boolean) => Promise<void>
  startNewChat: (base: string, character: string) => Promise<void>
}

const ChatStoreContext = createContext<ChatStoreContextValue | null>(null)

/**
 * Persistent-chat-state fix (owner brief item 3).
 *
 * Root cause of the old "chat disappears when you navigate away and back"
 * bug: `ChatView` owned all streaming/history state in local `useState`, so
 * unmounting it (switching sections, or in the old sidebar model just
 * losing the `chat` prop) tore the state down, and the in-flight
 * fetch+ReadableStream reading loop in `streamMessage` had nothing left to
 * write its results into once the component doing the writing was gone.
 *
 * Fix: this Provider is mounted exactly once, at the app root (see
 * App.tsx), above the navigation layer (character switch / Settings
 * overlay / anything else that mounts and unmounts views). It owns one
 * `ChatSessionState` per character in React state that lives as long as the
 * Provider does. `send()` below is a stable callback defined *inside this
 * component* — when a caller (ChatView) invokes it, the resulting promise
 * chain (the NDJSON reader loop, the approval-poll interval, every
 * `setSessions` call) is not tied to the caller's lifecycle at all; it's a
 * plain JS async function closing over `setSessions`/refs that belong to
 * this always-mounted Provider. So a character's stream keeps running and
 * accumulating into `sessions[character]` even while no `ChatView` for that
 * character is currently rendered (e.g. the user switched to another
 * character or opened Settings). When the user comes back, `useChatSession`
 * just reads the current accumulated state out of context — nothing was
 * ever paused, cancelled, or reset.
 */
export function ChatSessionsProvider({ children }: { children: React.ReactNode }): React.JSX.Element {
  const [sessions, setSessions] = useState<Record<string, ChatSessionState>>({})
  const sessionsRef = useRef(sessions)
  useEffect(() => {
    sessionsRef.current = sessions
  }, [sessions])

  const approvalTimers = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())
  const ensureInFlight = useRef<Map<string, Promise<void>>>(new Map())

  const patchSession = useCallback(
    (character: string, patch: Partial<ChatSessionState> | ((s: ChatSessionState) => Partial<ChatSessionState>)) => {
      setSessions((prev) => {
        const cur = prev[character] ?? EMPTY_SESSION
        const p = typeof patch === 'function' ? patch(cur) : patch
        return { ...prev, [character]: { ...cur, ...p } }
      })
    },
    []
  )

  const clearApprovalTimer = useCallback((character: string) => {
    const t = approvalTimers.current.get(character)
    if (t) {
      clearInterval(t)
      approvalTimers.current.delete(character)
    }
  }, [])

  // Resolves "the live chat" for a character: its most-recently-updated
  // chat (list_chats() on the backend already sorts desc by `updated`, so
  // no backend change is needed), or a fresh one if it has none yet. Once
  // resolved for a character, this is a no-op — every send() after that
  // appends to the same chat, matching the "one ongoing chat per character"
  // model. Older chats are left untouched on disk as reference (see
  // HistoryOverlay); they're just never picked as the live chat again.
  const ensureLiveChat = useCallback(
    (base: string, character: string): Promise<void> => {
      if (!character) return Promise.resolve()
      if (sessionsRef.current[character]?.chatId) return Promise.resolve()
      const inFlight = ensureInFlight.current.get(character)
      if (inFlight) return inFlight

      const run = (async (): Promise<void> => {
        patchSession(character, { loading: true, error: null })
        try {
          const all = await listChats(base)
          const mine = all.filter((c) => c.character === character)
          const detail = mine.length > 0 ? await getChat(base, mine[0].id) : await createChat(base, character)
          patchSession(character, { chatId: detail.id, history: detail.messages, loading: false })
        } catch (e) {
          patchSession(character, { loading: false, error: (e as Error).message })
        } finally {
          ensureInFlight.current.delete(character)
        }
      })()
      ensureInFlight.current.set(character, run)
      return run
    },
    [patchSession]
  )

  const send = useCallback(
    async (base: string, character: string, text: string): Promise<void> => {
      await ensureLiveChat(base, character)
      const chatId = sessionsRef.current[character]?.chatId
      if (!chatId) return

      const userMsg: ChatMessageEntry = { role: 'user', content: text }
      patchSession(character, (s) => ({
        history: [...s.history, userMsg],
        draft: { items: [], meta: {} },
        reasoning: '',
        streaming: true,
        error: null
      }))

      clearApprovalTimer(character)
      const timer = setInterval(async () => {
        try {
          const pending = await getApproval(base, chatId)
          patchSession(character, { approval: pending })
        } catch {
          // transient poll failures are ignored
        }
      }, 900)
      approvalTimers.current.set(character, timer)

      let items: AssistantItem[] = []
      let textAcc = ''
      const flushText = (): void => {
        if (textAcc) {
          items = [...items, { t: 'text', text: textAcc }]
          textAcc = ''
        }
      }

      try {
        await streamMessage(base, chatId, text, (evt) => {
          if (evt.type === 'reasoning') {
            patchSession(character, (s) => ({ reasoning: s.reasoning + evt.text }))
          } else if (evt.type === 'token') {
            textAcc += evt.text
            patchSession(character, {
              draft: { items: [...items, ...(textAcc ? [{ t: 'text', text: textAcc } as AssistantItem] : [])], meta: {} }
            })
          } else if (evt.type === 'tool_call') {
            flushText()
            items = [...items, { t: 'tool', name: evt.name, args: evt.args, output: null, ok: null }]
            patchSession(character, { draft: { items, meta: {} } })
          } else if (evt.type === 'tool_result') {
            for (let i = items.length - 1; i >= 0; i--) {
              const it = items[i]
              if (it.t === 'tool' && it.output === null) {
                items = items.map((x, idx) => (idx === i ? { ...it, output: evt.output.slice(0, 4000), ok: evt.ok } : x))
                break
              }
            }
            patchSession(character, { draft: { items, meta: {} } })
          } else if (evt.type === 'done') {
            flushText()
            patchSession(character, { draft: { items, meta: { tokens: evt.tokens, seconds: evt.seconds, tps: evt.tps } } })
          } else if (evt.type === 'stopped') {
            flushText()
            patchSession(character, { draft: { items, meta: { stopped: true } } })
          } else if (evt.type === 'error') {
            flushText()
            patchSession(character, { error: evt.error })
          }
        })
      } catch (e) {
        if ((e as Error).name !== 'AbortError') patchSession(character, { error: (e as Error).message })
      } finally {
        clearApprovalTimer(character)
        patchSession(character, { streaming: false, approval: null })
        try {
          const fresh = await getChat(base, chatId)
          patchSession(character, { history: fresh.messages, draft: null, reasoning: '' })
        } catch {
          // keep the local optimistic render if the refetch fails
        }
      }
    },
    [ensureLiveChat, patchSession, clearApprovalTimer]
  )

  const stop = useCallback(async (base: string, character: string): Promise<void> => {
    const chatId = sessionsRef.current[character]?.chatId
    if (!chatId) return
    try {
      await stopChat(base, chatId)
    } catch {
      // best-effort
    }
  }, [])

  const respondApproval = useCallback(
    async (base: string, character: string, ok: boolean): Promise<void> => {
      const chatId = sessionsRef.current[character]?.chatId
      if (!chatId) return
      patchSession(character, { approvalBusy: true })
      try {
        await postApproval(base, chatId, ok)
        patchSession(character, { approval: null })
      } catch (e) {
        patchSession(character, { error: (e as Error).message })
      } finally {
        patchSession(character, { approvalBusy: false })
      }
    },
    [patchSession]
  )

  // "Clear chat" (owner brief item 6): the owner has no way today to
  // intentionally end the current live conversation and start fresh -- every
  // character has exactly one ongoing "live chat" (ensureLiveChat above
  // always resolves to whichever chat has the most recent `updated`
  // timestamp), and older chats just become read-only History
  // (RightPanel's HistorySection), never deleted. This does the same thing
  // ensureLiveChat would naturally converge on if you just started
  // chatting again: create() a brand new chat (chats.py stamps it with
  // `updated = now`, so it's automatically what ensureLiveChat picks next
  // time), then swap the whole in-memory session over to it. The old chat
  // is untouched on disk -- "archiving" it is just "no longer being
  // pointed at as the live one", exactly like the read-only-reference
  // design HistorySection already documents.
  const startNewChat = useCallback(
    async (base: string, character: string): Promise<void> => {
      if (!character) return
      clearApprovalTimer(character)
      patchSession(character, { loading: true, error: null })
      try {
        const fresh = await createChat(base, character)
        patchSession(character, {
          chatId: fresh.id,
          history: [],
          draft: null,
          reasoning: '',
          streaming: false,
          approval: null,
          approvalBusy: false,
          error: null,
          loading: false
        })
      } catch (e) {
        patchSession(character, { loading: false, error: (e as Error).message })
      }
    },
    [patchSession, clearApprovalTimer]
  )

  const value = useMemo(
    () => ({ sessions, ensureLiveChat, send, stop, respondApproval, startNewChat }),
    [sessions, ensureLiveChat, send, stop, respondApproval, startNewChat]
  )

  return <ChatStoreContext.Provider value={value}>{children}</ChatStoreContext.Provider>
}

export function useChatSession(character: string): ChatSessionState {
  const ctx = useContext(ChatStoreContext)
  if (!ctx) throw new Error('useChatSession must be used within ChatSessionsProvider')
  return ctx.sessions[character] ?? EMPTY_SESSION
}

export function useChatStoreActions(): Omit<ChatStoreContextValue, 'sessions'> {
  const ctx = useContext(ChatStoreContext)
  if (!ctx) throw new Error('useChatStoreActions must be used within ChatSessionsProvider')
  const { ensureLiveChat, send, stop, respondApproval, startNewChat } = ctx
  return { ensureLiveChat, send, stop, respondApproval, startNewChat }
}
