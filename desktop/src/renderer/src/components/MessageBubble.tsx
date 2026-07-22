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
    <div className="my-1.5 rounded-lg border border-border bg-bg-surface/70 px-3 py-2 text-xs">
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

/** Renders one persisted (or in-progress) chat turn. User turns are plain
 * `content` strings; assistant turns are an ordered `items` list mixing text
 * chunks and tool calls, matching pleiades/chats.py's on-disk shape exactly
 * so the same renderer works for history and live streaming. */
function MessageBubble({ message, base, character, streaming }: MessageBubbleProps): React.JSX.Element {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end px-4 py-2">
        <div className="max-w-2xl rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[15px] leading-relaxed text-white">
          <div className="whitespace-pre-wrap break-words">{message.content}</div>
        </div>
      </div>
    )
  }

  const hasContent = message.items.length > 0
  return (
    <div className="flex gap-3 px-4 py-2">
      <Avatar base={base} character={character} size={28} />
      <div className="min-w-0 max-w-2xl flex-1">
        <div className="rounded-2xl rounded-tl-md border border-border bg-bg-surface px-4 py-2.5 text-[15px] leading-relaxed text-ink">
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
        </div>
        {!streaming && (message.meta?.tps || message.meta?.stopped) && (
          <div className="mt-1 px-1 text-xs text-ink-dim">
            {message.meta.stopped ? 'Stopped · ' : ''}
            {message.meta.tps ? `${message.meta.tps.toFixed(1)} tok/s` : ''}
          </div>
        )}
      </div>
    </div>
  )
}

export default MessageBubble
