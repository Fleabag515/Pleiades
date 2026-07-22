import { memo, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import type { AssistantItem, ChatMessageEntry, ToolItem } from '../lib/types'
import ReasoningBlock from './ReasoningBlock'

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
    <div className="my-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
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
            <div key={i} className="flex items-center gap-1.5">
              {i > 0 && <span className="select-none text-[13px] text-ink-faint">&rarr;</span>}
              <button
                onClick={() => setSelected(isSel ? null : i)}
                className={`whitespace-nowrap rounded-lg border bg-bg-100 px-3 py-1.5 font-mono text-[11.5px] text-ink transition-colors hover:border-cyan-400 hover:bg-bg-200 ${
                  isSel ? 'border-cyan-400 bg-bg-200' : stateClass
                }`}
              >
                {t.name}
              </button>
            </div>
          )
        })}
      </div>

      {active && (
        <div className="mt-1.5 rounded-lg border border-cyan-400/70 bg-bg-surface px-3 py-2.5 text-xs">
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
    </div>
  )
}

/** Groups an assistant turn's items so consecutive tool calls render as one
 * `ToolChain` (matching the legacy webui, where every tool call from a turn
 * lands in a single `.pipe-nodes` row) instead of one box per call. Text
 * segments in between still break the chain, same as the source data.
 * Reasoning items are never merged into a text or tool group -- each
 * `{t:'reasoning'}` item is already one distinct burst (chatStore.tsx and
 * server.py's chats_message() both only ever flush a new one once
 * something else has happened in between), so it always gets its own
 * `ReasoningBlock`, in sequence with everything else in the turn. */
function groupItems(
  items: AssistantItem[]
): Array<
  { kind: 'text'; text: string } | { kind: 'tools'; tools: ToolItem[] } | { kind: 'reasoning'; text: string }
> {
  const groups: Array<
    { kind: 'text'; text: string } | { kind: 'tools'; tools: ToolItem[] } | { kind: 'reasoning'; text: string }
  > = []
  for (const item of items) {
    if (item.t === 'tool') {
      const last = groups[groups.length - 1]
      if (last && last.kind === 'tools') last.tools.push(item)
      else groups.push({ kind: 'tools', tools: [item] })
    } else if (item.t === 'reasoning') {
      groups.push({ kind: 'reasoning', text: item.text })
    } else {
      groups.push({ kind: 'text', text: item.text })
    }
  }
  return groups
}

/**
 * Style overrides so `react-markdown`'s output fits the existing dark
 * theme/tokens (main.css's --color-ink, --color-bg, --color-accent ramps)
 * instead of default browser styling -- every tag react-markdown can emit for basic
 * CommonMark (the owner asked for bold/italic/code/lists/headings, not a
 * full GFM+extensions stack, so no tables/strikethrough/task-list overrides
 * here) gets a small, deliberately understated rule matching the rest of
 * the chat UI's visual weight (ToolChain/ReasoningBlock use the same
 * ink-dim/ink-faint/border tokens).
 *
 * `code`/`pre`: react-markdown v9+ dropped the old `inline` prop (it was
 * never reliable), so the standard replacement is checking the `code`
 * element's className for a `language-xxx` token, which remark-rehype only
 * ever adds to FENCED code blocks -- inline `` `code` `` never gets one.
 * `pre` is overridden once for the block-code wrapper (a bordered card like
 * ToolChain's detail card); the inline variant gets its own small pill
 * instead so a stray backtick never causes a nested/double `<pre>`.
 */
const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2 whitespace-pre-wrap break-words last:mb-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-ink-bright">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-accent underline underline-offset-2 hover:text-accent-hover"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => <ul className="my-1.5 ml-5 list-disc space-y-0.5 marker:text-ink-faint">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 ml-5 list-decimal space-y-0.5 marker:text-ink-faint">{children}</ol>,
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  h1: ({ children }) => <h1 className="mb-1.5 mt-3 text-lg font-semibold text-ink-bright first:mt-0">{children}</h1>,
  h2: ({ children }) => (
    <h2 className="mb-1.5 mt-3 text-base font-semibold text-ink-bright first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-1 mt-2.5 text-[15px] font-semibold text-ink-bright first:mt-0">{children}</h3>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-1.5 border-l-2 border-border pl-3 text-ink-dim">{children}</blockquote>
  ),
  hr: () => <hr className="my-3 border-border" />,
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg border border-border bg-bg-100 p-3 text-[13px]">{children}</pre>
  ),
  code: ({ className, children, ...rest }) => {
    const isBlock = /(?:^|\s)language-/.test(className ?? '')
    if (isBlock) {
      return (
        <code className={`font-mono text-ink ${className ?? ''}`} {...rest}>
          {children}
        </code>
      )
    }
    return (
      <code className="rounded bg-bg-300 px-1 py-0.5 font-mono text-[13px] text-ink-bright" {...rest}>
        {children}
      </code>
    )
  }
}

/** Renders assistant message text as markdown (bold/italic/inline code/
 * lists/headings) instead of raw literal syntax -- the owner's ask was
 * `**Progress:**` showing up literally with asterisks instead of bold.
 * `remark-breaks` turns a single newline into a real line break (CommonMark
 * alone only breaks paragraphs on a BLANK line, which would otherwise
 * silently swallow single `\n`s that used to render via the old
 * `whitespace-pre-wrap` plain-text path -- a real regression for models
 * that format with single newlines between short lines/list items).
 *
 * Verified live against a partial/streaming buffer (e.g. an unterminated
 * "**bold" while tokens are still arriving): react-markdown's underlying
 * parser (micromark, via remark) treats unmatched `**`/`*`/`` ` ``
 * delimiters as literal text rather than erroring, so a mid-stream chunk
 * just renders as plain text until its closing delimiter arrives, then
 * re-renders as the formatted element on the very next chunk -- no crash,
 * no leftover raw asterisks once the turn finishes.
 */
function Markdown({ text }: { text: string }): React.JSX.Element {
  return (
    <ReactMarkdown remarkPlugins={[remarkBreaks]} components={markdownComponents}>
      {text}
    </ReactMarkdown>
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
      <div className="max-w-3xl text-[15px] leading-relaxed text-ink">
        {!hasContent && streaming && (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:0ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-dim [animation-delay:300ms]" />
          </span>
        )}
        {groups.map((g, i) => {
          if (g.kind === 'tools') return <ToolChain key={i} tools={g.tools} />
          if (g.kind === 'reasoning') {
            const isLive = Boolean(streaming) && i === groups.length - 1
            return <ReasoningBlock key={i} text={g.text} streaming={isLive} />
          }
          return (
            <div key={i} className="break-words">
              <Markdown text={g.text} />
            </div>
          )
        })}
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

// Memoized: ChatView re-renders on every streamed token/reasoning-chunk
// (draft state changes), which would otherwise re-run every historical
// MessageBubble's render function on every token for long conversations.
// Historical `message`/`base`/`character` props are referentially stable
// between those re-renders, so memo turns that into a no-op for anything
// but the live streaming bubble.
export default memo(MessageBubble)
