import { memo, useState } from 'react'
import type { AssistantItem, ChatMessageEntry, ToolItem } from '../lib/types'

interface MessageBubbleProps {
  message: ChatMessageEntry
  base: string
  character: string
  streaming?: boolean
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s
}

/**
 * One row in a `ToolChain`'s shared detail card (owner brief item 3): a
 * label/value pair, monospace value, matching the legacy webui's `.kv`
 * rows in `showNodeDetail()` (pleiades/webui/static/next.html) rather than
 * a JSON blob.
 */
function DetailRow({ label, value, valueClassName }: { label: string; value: string; valueClassName?: string }): React.JSX.Element {
  return (
    <div className="flex items-start gap-2 py-0.5">
      <span className="w-14 flex-none text-[11px] text-ink-dim">{label}</span>
      <span className={`min-w-0 flex-1 break-all font-mono text-[11px] ${valueClassName ?? 'text-ink'}`}>{value}</span>
    </div>
  )
}

/**
 * Renders one turn's tool call(s) the way the real legacy webui does it —
 * this replaced a from-scratch guess (a per-call `<details>` accordion
 * showing icon+name+tag then full args/output) after actually launching
 * `pleiades ui`, opening a chat with Claude, and watching a live tool call
 * render: pleiades/webui/static/next.html's "pipeline" pattern
 * (`buildToolNodes`/`.pipe-node`/`.pipe-detail`), NOT the older
 * `pleiades/webui/static/app.js` `toolBlock()` the previous attempt read
 * (that file is a stale fallback the server only serves when next.html is
 * missing — see server.py's `@app.get("/")`).
 *
 * What that looks like: a horizontal row of small pill-shaped chips (one
 * per tool call, monospace, just the tool name — no icon, no "done"/
 * "running" tag text), chained with a plain "→" when several calls happen
 * back to back in the same turn. State is conveyed purely by the chip's
 * border color: amber + a slow blink while running, green once it
 * succeeds, red on error; hovering (or the currently selected chip) gets
 * a cyan border. There is no per-chip expansion — clicking a chip opens a
 * *single* shared detail card below the whole row (cyan-bordered) showing
 * tool/args/output/status as short key-value rows, each value truncated
 * to 400 chars, not a scrolling `<pre>` dump.
 */
function ToolChain({ tools }: { tools: ToolItem[] }): React.JSX.Element {
  const [selected, setSelected] = useState<number | null>(null)
  const active = selected !== null ? tools[selected] : null

  return (
    <>
      {/* `inline-flex` (not `div`/block) so a tool call sitting between two
       * chunks of assistant text stays on the same line as the text around
       * it whenever there's room, and wraps together with it as one
       * paragraph flow instead of always forcing its own line -- that
       * "doesn't flow together nicely" was literally block-level layout
       * fighting the surrounding `whitespace-pre-wrap` text. `align-middle`
       * plus a small negative-y nudge keeps the chip's baseline visually
       * centered against the text line instead of sitting low/high. */}
      <span className="my-0.5 inline-flex flex-wrap items-center gap-1.5 align-middle">
        {tools.map((t, i) => {
          const pending = t.output === null
          const err = t.ok === false
          const isSel = selected === i
          const stateClass = pending
            ? 'border-amber-400 animate-tool-blink'
            : err
              ? 'border-rose-400'
              : 'border-emerald-400'
          return (
            <span key={i} className="inline-flex items-center gap-1.5">
              {i > 0 && <span className="select-none text-[13px] text-ink-faint">&rarr;</span>}
              <button
                onClick={() => setSelected(isSel ? null : i)}
                className={`whitespace-nowrap rounded-lg border bg-bg-100 px-3 py-1 font-mono text-[11.5px] text-ink transition-colors hover:border-cyan-400 hover:bg-bg-200 ${
                  isSel ? 'border-cyan-400 bg-bg-200' : stateClass
                }`}
              >
                {t.name}
              </button>
            </span>
          )
        })}
      </span>

      {active && (
        <div className="my-1 rounded-lg border border-cyan-400/70 bg-bg-surface px-3 py-2.5 text-xs">
          <DetailRow label="tool" value={active.name} valueClassName="text-cyan-300" />
          {active.args && <DetailRow label="args" value={truncate(active.args, 400)} />}
          {active.output != null && <DetailRow label="output" value={truncate(active.output, 400)} />}
          <DetailRow
            label="status"
            value={active.output === null ? '…' : active.ok === false ? 'error' : 'ok'}
            valueClassName={active.output === null ? 'text-amber-400' : active.ok === false ? 'text-rose-400' : 'text-emerald-400'}
          />
        </div>
      )}
    </>
  )
}

/** Groups an assistant turn's items so consecutive tool calls render as one
 * `ToolChain` (matching the legacy webui, where every tool call from a turn
 * lands in a single `.pipe-nodes` row) instead of one box per call. Text
 * segments in between still break the chain, same as the source data. */
function groupItems(
  items: AssistantItem[]
): Array<{ kind: 'text'; text: string } | { kind: 'tools'; tools: ToolItem[] }> {
  const groups: Array<{ kind: 'text'; text: string } | { kind: 'tools'; tools: ToolItem[] }> = []
  for (const item of items) {
    if (item.t === 'tool') {
      const last = groups[groups.length - 1]
      if (last && last.kind === 'tools') last.tools.push(item)
      else groups.push({ kind: 'tools', tools: [item] })
    } else {
      groups.push({ kind: 'text', text: item.text })
    }
  }
  return groups
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
function MessageBubble({ message, streaming }: MessageBubbleProps): React.JSX.Element {
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
  const groups = groupItems(message.items)
  return (
    <div className="px-5 py-2.5">
      <div className="max-w-3xl whitespace-pre-wrap break-words text-[15px] leading-relaxed text-ink">
        {!hasContent && streaming && (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:0ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:300ms]" />
          </span>
        )}
        {groups.map((g, i) =>
          g.kind === 'text' ? <span key={i}>{g.text}</span> : <ToolChain key={i} tools={g.tools} />
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
