import { useEffect, useRef, useState } from 'react'
import type { ToolItem } from '../lib/types'
import ToolChain from './ToolChain'

/** One piece of an `'activity'` `ItemGroup` (see MessageBubble's
 * `groupItems`) -- either a distinct reasoning burst or a run of tool
 * calls, in the order they actually happened. Kept as a discriminated
 * union (mirroring `ItemGroup` itself) rather than flattening reasoning
 * text and tool calls into one blob, so `ToolChain` still only ever sees
 * a plain `ToolItem[]` -- its own horizontal-chip rendering/chaining logic
 * is completely untouched by this restructuring. */
export type ActivityPart =
  { kind: 'reasoning'; text: string } | { kind: 'tools'; tools: ToolItem[] }

interface ActivityBlockProps {
  parts: ActivityPart[]
  /** True only for the single activity group that's still the live tail of
   * an in-progress stream (see MessageBubble's `isLive`, same formula the
   * old `ReasoningBlock` used: `streaming && i === groups.length - 1`). */
  live: boolean
}

/** Claude-Desktop-style collapsed "activity" strip: replaces the old
 * bordered-box `ReasoningBlock` (rounded-xl border + bg card) with
 * borderless, dim inline text that brightens on hover and carries a
 * rotating chevron for collapsed/expanded state -- owner's ask verbatim:
 * "a darker slightly smaller text, that when hovered over lightens up and
 * adds an arrow to indicate its collapsed... when you click the text, it
 * expands to show the thinking and tool calls that happened between each
 * message the model sent". Still a native `<details>/<summary>` under the
 * hood (free keyboard/a11y + click-anywhere-on-summary toggle), just
 * restyled -- no reason to rebuild the disclosure primitive from scratch.
 *
 * Structural change from the old two-sibling-blocks layout (council
 * review, both Fable and Opus independently flagged the same thing): a
 * reasoning burst and every tool-call run that follows it, up to the next
 * real text/final-answer segment, is now ONE group (`groupItems` builds
 * this as `parts: ActivityPart[]`, preserving the original sequence --
 * multiple reasoning bursts and tool runs can interleave inside a single
 * multi-hop tool-use turn), rendered through exactly one collapsible
 * wrapper instead of a `ReasoningBlock` sitting next to a separate
 * `ToolChain`. `ToolChain` itself renders unchanged inside that wrapper --
 * only the outer chrome changed.
 *
 * Truncation (owner brief item 3, "the thinking should also be collapsed
 * in that if its too big"): the `max-h-64 overflow-y-auto` scroll region
 * from the old `ReasoningBlock` is kept, but now applied per reasoning
 * *part* rather than to the whole expanded region -- so a very long
 * thinking burst scrolls within its own box instead of pushing later tool
 * chips (or a second, later reasoning burst) far down the page. Tool runs
 * are never height-clamped.
 *
 * Live-tool-activity gotcha (owner brief item 5, flagged specifically by
 * Opus): a collapsed dim strip must never hide a tool call that's still
 * running or waiting on `exec_policy=ask` approval. `hasPendingTool` below
 * mirrors `ToolChain`'s own `pending = t.output === null` check across
 * every tool part in this group, and gets OR'd into the same
 * streaming-driven auto-expand/auto-collapse effect `ReasoningBlock` used
 * (`effectiveLive = live || hasPendingTool`) -- so the region forces itself
 * open for as long as anything inside it is actually in flight, live turn
 * or not, and only auto-collapses once every tool call in it has resolved.
 * The summary label also gets a short live hint appended whenever that's
 * the reason it's open, so it's never just a silent dim strip while
 * something is actually running underneath it.
 */
function ActivityBlock({ parts, live }: ActivityBlockProps): React.JSX.Element {
  const hasPendingTool = parts.some(
    (p) => p.kind === 'tools' && p.tools.some((t) => t.output === null)
  )
  const effectiveLive = live || hasPendingTool

  const [open, setOpen] = useState(effectiveLive)
  const wasLive = useRef(effectiveLive)

  useEffect(() => {
    if (wasLive.current && !effectiveLive) setOpen(false)
    wasLive.current = effectiveLive
  }, [effectiveLive])

  const hasReasoning = parts.some((p) => p.kind === 'reasoning')
  const label = hasReasoning ? 'Thinking' : 'Tool use'
  const hint = effectiveLive ? (hasPendingTool ? ' · running' : '…') : ''

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="my-1"
    >
      <summary className="flex cursor-pointer select-none items-center gap-1 text-xs text-ink-faint transition-colors hover:text-ink-dim [&::-webkit-details-marker]:hidden">
        <span
          aria-hidden
          className={`inline-block text-[9px] transition-transform duration-200 ${open ? '' : '-rotate-90'}`}
        >
          &#9662;
        </span>
        <span>
          {label}
          {hint}
        </span>
      </summary>
      <div className="mt-1 flex flex-col gap-1.5 border-l border-border/60 pl-3">
        {parts.map((part, i) =>
          part.kind === 'tools' ? (
            <ToolChain key={i} tools={part.tools} />
          ) : (
            <div
              key={i}
              className="max-h-64 overflow-y-auto whitespace-pre-wrap text-xs italic text-ink-faint"
            >
              {part.text}
            </div>
          )
        )}
      </div>
    </details>
  )
}

export default ActivityBlock
