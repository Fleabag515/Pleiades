import { useEffect, useRef, useState } from 'react'

interface ReasoningBlockProps {
  text: string
  streaming: boolean
}

/** Collapsible `<think>` block, matching the legacy webui's `.think-block`
 * (see pleiades/webui/static/app.js's `renderSegments`/`toolBlock`, which
 * uses native `<details>` for both thinking and tool blocks). Owner brief
 * item 6: togglable both during and after streaming, not just always-on
 * dimmed text.
 *
 * Default: expanded while actively streaming (so the user can watch it
 * think, same as the legacy `d.open = seg.live && streaming`), then
 * auto-collapses once streaming ends for this turn. A manual toggle at any
 * point (native `onToggle`) overrides that default going forward. */
function ReasoningBlock({ text, streaming }: ReasoningBlockProps): React.JSX.Element {
  const [open, setOpen] = useState(streaming)
  const wasStreaming = useRef(streaming)

  useEffect(() => {
    if (wasStreaming.current && !streaming) setOpen(false)
    wasStreaming.current = streaming
  }, [streaming])

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="mx-5 my-1.5 rounded-xl border border-border bg-white/[0.02] text-xs"
    >
      <summary className="flex cursor-pointer select-none items-center gap-1.5 px-3 py-1.5 text-ink-faint [&::-webkit-details-marker]:hidden">
        <span aria-hidden>✦</span>
        <span>thinking{streaming ? '…' : ''}</span>
      </summary>
      <div className="max-h-64 overflow-y-auto whitespace-pre-wrap border-t border-border px-3 py-2 italic text-ink-faint">
        {text}
      </div>
    </details>
  )
}

export default ReasoningBlock
