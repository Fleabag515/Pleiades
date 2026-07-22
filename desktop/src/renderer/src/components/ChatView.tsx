import { useEffect, useRef, useState } from 'react'
import { useChatSession, useChatStoreActions } from '../lib/chatStore'
import Avatar from './Avatar'
import ApprovalCard from './ApprovalCard'
import Composer from './Composer'
import HistoryOverlay from './HistoryOverlay'
import MessageBubble from './MessageBubble'
import ReasoningBlock from './ReasoningBlock'
import RightPanelStub from './RightPanelStub'

interface ChatViewProps {
  base: string
  character: string
}

/**
 * Main pane: top bar, streaming message list, composer. Owner brief item 4:
 * there's no "pick a chat" list anymore — this always shows the one live
 * chat for `character` (resolved/created by `ensureLiveChat` in
 * lib/chatStore.tsx). Owner brief item 3: unlike the old version, none of
 * the message history/streaming/approval state lives here anymore — it's
 * all read from `useChatSession`, which is backed by a Provider mounted
 * once at the app root, so switching characters or opening Settings and
 * coming back never resets or loses anything that streamed in the
 * meantime.
 */
function ChatView({ base, character }: ChatViewProps): React.JSX.Element {
  const session = useChatSession(character)
  const { ensureLiveChat, send, stop, respondApproval } = useChatStoreActions()
  const scrollRef = useRef<HTMLDivElement>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [rightPanelOpen, setRightPanelOpen] = useState(false)

  useEffect(() => {
    if (character) ensureLiveChat(base, character)
  }, [base, character, ensureLiveChat])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [session.history, session.draft, session.reasoning, session.approval])

  if (!character) {
    return (
      <div className="flex h-full flex-1 flex-col items-center justify-center bg-bg-app text-center text-sm text-ink-dim">
        Pick a character above, or create one with the + bubble.
      </div>
    )
  }

  const showEmpty = !session.loading && session.history.length === 0 && !session.draft

  return (
    <div className="relative flex h-full flex-1 flex-col overflow-hidden bg-bg-app">
      <div className="flex h-14 flex-none items-center gap-2.5 border-b border-border px-5">
        <Avatar base={base} character={character} size={24} />
        <span className="font-medium text-ink-bright">{character}</span>
        <button
          onClick={() => setHistoryOpen(true)}
          title="Past chats (reference only)"
          className="ml-auto rounded-full p-1.5 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
        >
          <span aria-hidden>&#8986;</span>
        </button>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto py-3">
        {session.loading && session.history.length === 0 && (
          <div className="px-5 py-4 text-sm text-ink-dim">Loading…</div>
        )}

        {showEmpty && (
          <div className="flex h-full w-full flex-col items-center justify-center gap-3 px-6 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/15 text-2xl text-accent">
              ✳
            </div>
            <h1 className="text-xl font-semibold text-ink-bright">Say hello to {character}</h1>
            <p className="max-w-sm text-sm text-ink-dim">
              Persistent memory via Anamnesis. Tools stay on standby until the task calls for them.
            </p>
          </div>
        )}

        {session.history.map((m, i) => (
          <MessageBubble key={i} message={m} base={base} character={character} />
        ))}

        {session.reasoning && session.streaming && <ReasoningBlock text={session.reasoning} streaming={session.streaming} />}

        {session.draft && (
          <MessageBubble
            message={{ role: 'assistant', items: session.draft.items, meta: session.draft.meta }}
            base={base}
            character={character}
            streaming
          />
        )}

        {session.approval && (
          <div className="px-5 py-2">
            <ApprovalCard
              approval={session.approval}
              busy={session.approvalBusy}
              onApprove={() => respondApproval(base, character, true)}
              onDeny={() => respondApproval(base, character, false)}
            />
          </div>
        )}

        {session.error && (
          <div className="mx-5 my-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
            {session.error}
          </div>
        )}
      </div>

      <Composer
        base={base}
        character={character}
        disabled={!character}
        streaming={session.streaming}
        onSend={(text) => send(base, character, text)}
        onStop={() => stop(base, character)}
      />

      {/* Reserved toggle for the future right-side panel (history/progress/
          browser/scheduled-tasks) — see RightPanelStub. */}
      <button
        onClick={() => setRightPanelOpen((v) => !v)}
        title="History, tasks &amp; browser panel (coming soon)"
        className="absolute bottom-24 right-4 z-10 flex h-9 w-9 items-center justify-center rounded-full border border-border bg-bg-100/90 text-ink-dim shadow-lg backdrop-blur transition hover:bg-bg-surface-hover hover:text-ink"
      >
        <span aria-hidden>&#9776;</span>
      </button>
      {rightPanelOpen && <RightPanelStub onClose={() => setRightPanelOpen(false)} />}

      {historyOpen && <HistoryOverlay base={base} character={character} onClose={() => setHistoryOpen(false)} />}
    </div>
  )
}

export default ChatView
