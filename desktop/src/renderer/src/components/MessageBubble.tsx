import { memo } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import { attachmentUrl } from '../lib/api'
import type { AssistantItem, ChatMessageEntry, MessageAttachment, ToolItem } from '../lib/types'
import ActivityBlock, { type ActivityPart } from './ActivityBlock'
import ToolChain from './ToolChain'

interface MessageBubbleProps {
  message: ChatMessageEntry
  base: string
  character: string
  // This message's chat id -- needed to build the GET .../attachments/{name}
  // URL (see lib/api.ts's `attachmentUrl`) each persisted attachment chip
  // below points its thumbnail/`<audio>` src at. Optional/nullable because
  // a couple of call sites (the live streaming draft bubble, which is
  // always an assistant turn and never carries attachments) don't have one
  // handy -- attachments only ever land on `role: 'user'` messages, so it's
  // simply not read on any path where it'd be missing.
  chatId?: string | null
  streaming?: boolean
}

/** Groups an assistant turn's items for rendering.
 *
 * Owner brief (thinking-redesign): a reasoning burst and every tool-call
 * run that follows it -- up to the next real text/final-answer segment or
 * a `user_injected` interruption -- now collapse into ONE `'activity'`
 * group instead of a `ReasoningBlock` and one-or-more separate `ToolChain`
 * groups sitting next to each other as siblings. `parts` preserves the
 * original in-order sequence of reasoning bursts and tool runs inside that
 * span (a multi-hop turn can have several of each before the model's next
 * real text), so nothing about *when* things happened is lost -- only the
 * outer chrome around them changes (see ActivityBlock.tsx).
 *
 * `tools` items that appear with nothing preceding them in the current run
 * (no open 'activity' group to fold into -- the model called a tool with
 * no `<think>` burst at all) stay as a bare `'tools'` group, unwrapped,
 * exactly like before: there's no thinking to anchor a collapsible summary
 * to, so there's nothing to collapse. In practice this is rare for models
 * that reason before every tool call, but it's a real, currently-untested
 * path worth flagging if tool-only bursts turn out to be common in
 * practice -- see report.
 *
 * `text`/`user_injected` items always close out any in-progress 'activity'
 * (or bare 'tools') run and start a fresh group -- a plain-text segment or
 * a live user interjection is never folded into the dim collapsed region.
 */
type ItemGroup =
  | { kind: 'text'; text: string }
  | { kind: 'tools'; tools: ToolItem[] }
  | { kind: 'activity'; parts: ActivityPart[] }
  // A message the user sent mid-turn (see chatStore.tsx's `interject` /
  // engine.py's poll_injections) -- rendered as its own distinct group, in
  // place, so it's clear this was a live interruption and not part of the
  // assistant's own text.
  | { kind: 'user_injected'; text: string }

function groupItems(items: AssistantItem[]): ItemGroup[] {
  const groups: ItemGroup[] = []
  for (const item of items) {
    const last = groups[groups.length - 1]
    if (item.t === 'reasoning') {
      if (last && last.kind === 'activity') {
        last.parts.push({ kind: 'reasoning', text: item.text })
      } else {
        groups.push({ kind: 'activity', parts: [{ kind: 'reasoning', text: item.text }] })
      }
    } else if (item.t === 'tool') {
      if (last && last.kind === 'activity') {
        const lastPart = last.parts[last.parts.length - 1]
        if (lastPart && lastPart.kind === 'tools') lastPart.tools.push(item)
        else last.parts.push({ kind: 'tools', tools: [item] })
      } else if (last && last.kind === 'tools') {
        last.tools.push(item)
      } else {
        groups.push({ kind: 'tools', tools: [item] })
      }
    } else if (item.t === 'user_injected') {
      groups.push({ kind: 'user_injected', text: item.text })
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
 * the chat UI's visual weight (ToolChain/ActivityBlock use the same
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
  ul: ({ children }) => (
    <ul className="my-1.5 ml-5 list-disc space-y-0.5 marker:text-ink-faint">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-1.5 ml-5 list-decimal space-y-0.5 marker:text-ink-faint">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  h1: ({ children }) => (
    <h1 className="mb-1.5 mt-3 text-lg font-semibold text-ink-bright first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-1.5 mt-3 text-base font-semibold text-ink-bright first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-1 mt-2.5 text-[15px] font-semibold text-ink-bright first:mt-0">{children}</h3>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-1.5 border-l-2 border-border pl-3 text-ink-dim">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-border" />,
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg border border-border bg-bg-100 p-3 text-[13px]">
      {children}
    </pre>
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
      <code
        className="rounded bg-bg-300 px-1 py-0.5 font-mono text-[13px] text-ink-bright"
        {...rest}
      >
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

/** Small glyph differentiating an attachment chip by MIME prefix -- same
 * unicode-glyph-as-icon convention Composer.tsx already uses for its own
 * attach button (&#128206;) and per-chip error state (&#9888;), so this
 * isn't introducing a new visual idiom, just extending the existing one. */
function attachmentIcon(mime: string): string {
  if (mime.startsWith('image/')) return '\u{1F5BC}'
  if (mime.startsWith('audio/')) return '\u{1F3B5}'
  return '\u{1F4C4}'
}

/** One persisted attachment on a historical (already-sent) user message.
 *
 * Deliberately mirrors Composer.tsx's own pre-send chip (`attachedFiles`
 * rendering: `rounded-lg border border-border px-2 py-1 text-xs text-ink-dim`)
 * so the same visual pattern reads as "this is an attachment" whether it's
 * still uploading in the composer or long since sent and sitting in
 * history -- that consistency was the whole point of this pass (see owner
 * feedback: "no indication a file was uploaded... some kind of file
 * indicator would be nice").
 *
 * Images get a real thumbnail -- fetched from GET .../attachments/{name}
 * (lib/api.ts's `attachmentUrl`, added alongside this component; see
 * pleiades/webui/server.py's `chats_attachment_get`) -- since the upload
 * response's `path` is a path on THIS machine the renderer process can't
 * load directly. Audio gets a native `<audio controls>` against that same
 * URL (cheap, and a real player is more useful than a mute chip for
 * something the user very likely wants to play back). Everything else
 * falls back to the plain icon+filename chip.
 */
function AttachmentChip({
  attachment,
  base,
  chatId
}: {
  attachment: MessageAttachment
  base: string
  chatId: string
}): React.JSX.Element {
  const url = attachmentUrl(base, chatId, attachment.name)

  if (attachment.mime.startsWith('image/')) {
    return (
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        title={attachment.name}
        className="block h-16 w-16 overflow-hidden rounded-lg border border-border bg-bg-300/40"
      >
        <img src={url} alt={attachment.name} className="h-full w-full object-cover" />
      </a>
    )
  }

  if (attachment.mime.startsWith('audio/')) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-border px-2 py-1 text-xs text-ink-dim">
        <span aria-hidden>{attachmentIcon(attachment.mime)}</span>
        <span className="max-w-[140px] truncate">{attachment.name}</span>
        <audio controls src={url} className="h-7 max-w-[200px]" />
      </div>
    )
  }

  return (
    <div
      title={attachment.name}
      className="flex items-center gap-1.5 rounded-lg border border-border px-2 py-1 text-xs text-ink-dim"
    >
      <span aria-hidden>{attachmentIcon(attachment.mime)}</span>
      <span className="max-w-[160px] truncate">{attachment.name}</span>
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
function MessageBubble({ message, base, chatId, streaming }: MessageBubbleProps): React.JSX.Element {
  if (message.role === 'user') {
    const attachments = message.attachments
    return (
      <div className="flex flex-col items-end gap-1.5 px-5 py-2">
        {attachments && attachments.length > 0 && chatId && (
          <div className="flex max-w-[85%] flex-wrap justify-end gap-1.5">
            {attachments.map((a, i) => (
              <AttachmentChip key={`${a.name}-${i}`} attachment={a} base={base} chatId={chatId} />
            ))}
          </div>
        )}
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
          if (g.kind === 'activity') {
            const isLive = Boolean(streaming) && i === groups.length - 1
            return <ActivityBlock key={i} parts={g.parts} live={isLive} />
          }
          if (g.kind === 'user_injected') {
            return (
              <div key={i} className="my-1.5 flex justify-end">
                <div className="max-w-[85%] rounded-2xl border border-accent/30 bg-bubble-user/70 px-4 py-2 text-[14px] leading-relaxed text-ink-bright">
                  <div className="mb-0.5 text-[11px] font-medium uppercase tracking-wide text-accent">
                    sent mid-task
                  </div>
                  <div className="whitespace-pre-wrap break-words">{g.text}</div>
                </div>
              </div>
            )
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
