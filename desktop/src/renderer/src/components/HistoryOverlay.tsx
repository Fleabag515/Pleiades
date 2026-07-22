import { useEffect, useState } from 'react'
import { getChat, listChats } from '../lib/api'
import type { ChatDetail, ChatSummary } from '../lib/types'
import { relativeTime } from '../lib/format'
import MessageBubble from './MessageBubble'

interface HistoryOverlayProps {
  base: string
  character: string
  onClose: () => void
}

/**
 * Stub/placeholder for chat history browsing (owner brief item 4). Full
 * history browsing UI is Phase 14's right-panel job — this is deliberately
 * minimal: a read-only list of a character's past chats plus a read-only
 * transcript viewer. Selecting one never makes it "the live chat" again —
 * under the new one-live-chat-per-character model, old chats are reference
 * only, never resumed. Nothing here deletes or mutates old chats.
 */
function HistoryOverlay({ base, character, onClose }: HistoryOverlayProps): React.JSX.Element {
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [viewing, setViewing] = useState<ChatDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listChats(base)
      .then((all) => setChats(all.filter((c) => c.character === character)))
      .catch((e) => setError((e as Error).message))
  }, [base, character])

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/50 p-6" onClick={onClose}>
      <div
        className="flex h-[70vh] w-full max-w-lg flex-col overflow-hidden rounded-3xl border border-border bg-bg-100 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <h2 className="truncate text-base font-semibold text-ink-bright">
            {viewing ? viewing.title || '(new chat)' : `${character}'s history`}
          </h2>
          <div className="flex flex-none items-center gap-2">
            {viewing && (
              <button
                onClick={() => setViewing(null)}
                className="rounded-lg px-2 py-1 text-xs text-ink-dim transition hover:text-ink"
              >
                Back
              </button>
            )}
            <button
              onClick={onClose}
              className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {error && (
            <div className="m-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
              {error}
            </div>
          )}

          {!viewing ? (
            <div className="p-3">
              <p className="mb-2 px-1 text-xs text-ink-faint">
                Read-only reference — {character}&rsquo;s active conversation is always the live chat behind this
                overlay. Opening an old one here doesn&rsquo;t resume it.
              </p>
              {chats.length === 0 && <div className="px-2 py-3 text-sm text-ink-faint">No past chats yet.</div>}
              {chats.map((c) => (
                <button
                  key={c.id}
                  onClick={() =>
                    getChat(base, c.id)
                      .then(setViewing)
                      .catch((e) => setError((e as Error).message))
                  }
                  className="flex w-full flex-col items-start rounded-lg px-3 py-2 text-left transition hover:bg-bg-surface-hover"
                >
                  <span className="truncate text-sm text-ink">{c.title || '(new chat)'}</span>
                  <span className="text-xs text-ink-faint">
                    {relativeTime(c.updated)} &middot; {c.messages} messages
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="py-2">
              {viewing.messages.length === 0 && <div className="p-4 text-sm text-ink-faint">Empty chat.</div>}
              {viewing.messages.map((m, i) => (
                <MessageBubble key={i} message={m} base={base} character={character} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default HistoryOverlay
