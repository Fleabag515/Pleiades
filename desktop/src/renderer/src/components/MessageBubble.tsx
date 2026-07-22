import { memo } from 'react'
import type { ChatMessageEntry } from '../lib/types'
import Avatar from './Avatar'

interface MessageBubbleProps {
  message: ChatMessageEntry
  base: string
  character: string
  streaming?: boolean
}

function ToolBlock({
  name,
  args,
  output,
  ok
}: {
  name: string
  args: string
  output: string | null
  ok: boolean | null
}): React.JSX.Element {
  return (
    <div className="my-1.5 rounded-lg bg-bg-100 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 font-mono text-ink-dim">
        <span className={ok === false ? 'text-rose-400' : ok === true ? 'text-emerald-400' : 'text-amber-400'}>
          {ok === null ? '…' : ok ? '✓' : '✕'}
        </span>
        <span className="text-ink">{name}</span>
        <span className="truncate opacity-70">{args}</span>
      </div>
      {output != null && (
        <pre className="mt-1.5 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-ink-dim">
          {output}
        </pre>
      )}
    </div>
  )
}

/** Renders one persisted (or in-progress) chat turn.
 *
 * Structure matches what live accessibility-tree inspection of the real
 * Claude Desktop app showed (see Phase F notes): the user's turn is a
 * right-aligned, content-hugging bubble bounded well short of the full
 * column width, while the assistant's turn is *not* boxed at all -- it
 * spans almost the full column as plain text with no card/border,
 * which is the detail that made the previous version (bubbles on both
 * sides) read as generic/ChatGPT-like rather than Claude-like. */
function MessageBubble({ message, base, character, streaming }: MessageBubbleProps): React.JSX.Element {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end px-5 py-2">
        <div className="max-w-[85%] rounded-2xl bg-bubble-user px-4 py-2.5 text-[15px] leading-relaxed text-ink-bright">
          <div className="whitespace-pre-wrap break-words">{message.content}</div>
        </div>
      </div>
    )
  }

  const hasContent = message.items.length > 0
  return (
    <div className="flex gap-3 px-5 py-2.5">
      <Avatar base={base} character={character} size={24} />
      <div className="min-w-0 max-w-3xl flex-1 pt-0.5 text-[15px] leading-relaxed text-ink">
        {!hasContent && streaming && (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:0ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:300ms]" />
          </span>
        )}
        {message.items.map((item, i) =>
          item.t === 'text' ? (
            <div key={i} className="whitespace-pre-wrap break-words">
              {item.text}
            </div>
          ) : (
            <ToolBlock key={i} name={item.name} args={item.args} output={item.output} ok={item.ok} />
          )
        )}
        {!streaming && (message.meta?.tps || message.meta?.stopped) && (
          <div className="mt-1.5 text-xs text-ink-faint">
            {message.meta.stopped ? 'Stopped · ' : ''}
            {message.meta.tps ? `${message.meta.tps.toFixed(1)} tok/s` : ''}
          </div>
        )}
      </div>
    </div>
  )
}

// Memoized: ChatView re-renders on every streamed token (draft/reasoning
// state changes), which would otherwise re-run every historical
// MessageBubble's render function on every token for long conversations.
// Historical `message`/`base`/`character` props are referentially stable
// between those re-renders, so memo turns that into a no-op for anything
// but the live streaming bubble.
export default memo(MessageBubble)
