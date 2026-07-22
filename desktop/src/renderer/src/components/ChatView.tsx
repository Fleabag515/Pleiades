import { useEffect, useRef } from 'react'
import { useChatSession, useChatStoreActions } from '../lib/chatStore'
import ApprovalCard from './ApprovalCard'
import Composer from './Composer'
import MessageBubble from './MessageBubble'
import ReasoningBlock from './ReasoningBlock'
import RightPanel from './RightPanel'

interface ChatViewProps {
  base: string
  character: string
  rightPanelOpen: boolean
}

/**
 * Main pane: streaming message list + composer, no separate header bar.
 * Owner brief item 4: there's no "pick a chat" list anymore — this always
 * shows the one live chat for `character` (resolved/created by
 * `ensureLiveChat` in lib/chatStore.tsx). Owner brief item 3: unlike the
 * old version, none of the message history/streaming/approval state lives
 * here anymore — it's all read from `useChatSession`, which is backed by a
 * Provider mounted once at the app root, so switching characters or
 * opening Settings and coming back never resets or loses anything that
 * streamed in the meantime.
 *
 * Owner feedback round 2:
 *   - The right-panel open/closed toggle used to float here, oddly placed
 *     relative to everything else. It's lifted up to Shell.tsx now, so it
 *     can sit in the same row and at the same size as the Settings
 *     cogwheel bubble -- this component just receives `rightPanelOpen` as a
 *     prop and renders RightPanel accordingly.
 *   - The History trigger (previously its own icon under Composer) is
 *     retired; History now lives inside RightPanel as a collapsible block.
 *   - The scrollable message list now carries `.scroll-fade-top` (see
 *     main.css) so content fades to transparent as it scrolls up underneath
 *     TopCharacterBar, instead of being hard-clipped at a bordered edge --
 *     TopCharacterBar's own bottom border was also removed, so there's
 *     nothing but that soft mask marking the boundary.
 *
 * Phase 14: the outer return is a real flex ROW — `[chat column]
 * [RightPanel]` — instead of a single column with an absolutely-positioned
 * overlay. RightPanel is always mounted (so its own width transition can
 * animate open/close); only its wrapper's width changes, which is why
 * opening it visibly narrows this chat column instead of floating on top
 * of it.
 */
function ChatView({ base, character, rightPanelOpen }: ChatViewProps): React.JSX.Element {
  const session = useChatSession(character)
  const { ensureLiveChat, send, stop, respondApproval, startNewChat } = useChatStoreActions()
  const scrollRef = useRef<HTMLDivElement>(null)

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
    <div className="flex h-full flex-1 overflow-hidden">
      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-bg-app">
        <div
          ref={scrollRef}
          className="scroll-fade-top min-h-0 flex-1 overflow-y-auto py-3"
        >
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
                Persistent memory via Anamnesis. Tools stay on standby until the task calls for
                them.
              </p>
            </div>
          )}

          {session.history.map((m, i) => (
            <MessageBubble key={i} message={m} base={base} character={character} />
          ))}

          {session.reasoning && session.streaming && (
            <ReasoningBlock text={session.reasoning} streaming={session.streaming} />
          )}

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
          onNewChat={() => {
            if (
              window.confirm(
                `Start a new chat with ${character}? This conversation moves to History — nothing is deleted.`
              )
            ) {
              startNewChat(base, character)
            }
          }}
        />
      </div>

      <RightPanel base={base} character={character} open={rightPanelOpen} />
    </div>
  )
}

export default ChatView
