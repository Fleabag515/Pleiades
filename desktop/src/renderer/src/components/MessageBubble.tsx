import { memo } from 'react'
import type { ChatMessageEntry } from '../lib/types'
import Avatar from './Avatar'

interface MessageBubbleProps {
  message: ChatMessageEntry
  base: string
  character: string
  streaming?: boolean
}

/**
 * Renders one completed (or in-flight) tool call inline in the transcript.
 *
 * Structure/clarity deliberately matches the legacy webui's `toolBlock()`
 * (pleiades/webui/static/app.js): a collapsible `<details>` with a summary
 * line (status icon + tool name + a pending/done/error tag) and a body
 * showing the raw args followed by a `<pre>` of the output — read-only
 * reference styling, not the approval-gating `ApprovalCard` (that's a
 * different concern: a pending permission gate on a tool call that hasn't
 * run yet, vs. this, a completed call's name/args/result on display).
 */
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
  const pending = output === null
  const tagClass = pending ? 'text-amber-400' : ok === false ? 'text-rose-400' : 'text-emerald-400'
  const tagText = pending ? 'running…' : ok === false ? 'error' : 'done'
  return (
    <details className="my-1.5 rounded-lg border border-border bg-bg-100 text-xs">
      <summary className="flex cursor-pointer select-none items-center gap-1.5 px-3 py-2 text-ink-dim [&::-webkit-details-marker]:hidden">
        <span className={tagClass}>{pending ? '◌' : ok === false ? '✕' : '✓'}</span>
        <span className="text-ink">{name}</span>
        <span className={`ml-auto text-[10.5px] uppercase tracking-wide ${tagClass}`}>{tagText}</span>
      </summary>
      <div className="border-t border-border px-3 py-2">
        {args && <div className="mb-1.5 break-all font-mono text-[11px] text-ink-faint">{args}</div>}
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-ink-dim">
          {pending ? '…' : output || '(no output)'}
        </pre>
      </div>
    </details>
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
