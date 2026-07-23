import { useState } from 'react'
import type { ToolItem } from '../lib/types'

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s
}

/**
 * One row in a `ToolChain`'s shared detail card (owner brief item 3): a
 * label/value pair, monospace value, matching the legacy webui's `.kv`
 * rows in `showNodeDetail()` (pleiades/webui/static/next.html) rather than
 * a JSON blob.
 */
function DetailRow({
  label,
  value,
  valueClassName
}: {
  label: string
  value: string
  valueClassName?: string
}): React.JSX.Element {
  return (
    <div className="flex items-start gap-2 py-0.5">
      <span className="w-14 flex-none text-[11px] text-ink-dim">{label}</span>
      <span
        className={`min-w-0 flex-1 break-all font-mono text-[11px] ${valueClassName ?? 'text-ink'}`}
      >
        {value}
      </span>
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
 *
 * Extracted verbatim out of MessageBubble.tsx into its own module so both
 * MessageBubble (standalone tool runs with no preceding reasoning) and
 * ActivityBlock (tool runs bundled inside a collapsed thinking region) can
 * import the exact same component -- this thinking-block redesign only
 * ever touches the *wrapper* around tool calls, never this file's own
 * rendering/chaining/detail-card logic.
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
          {active.output != null && (
            <DetailRow label="output" value={truncate(active.output, 400)} />
          )}
          <DetailRow
            label="status"
            value={active.output === null ? '…' : active.ok === false ? 'error' : 'ok'}
            valueClassName={
              active.output === null
                ? 'text-amber-400'
                : active.ok === false
                  ? 'text-rose-400'
                  : 'text-emerald-400'
            }
          />
        </div>
      )}
    </div>
  )
}

export default ToolChain
