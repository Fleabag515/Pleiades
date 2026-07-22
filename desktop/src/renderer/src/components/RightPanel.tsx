import { useCallback, useEffect, useRef, useState } from 'react'
import {
  browserViewNavigate,
  browserViewOpenSeparate,
  browserViewResize,
  browserViewStart,
  browserViewStatus,
  browserViewStop,
  browserViewWsUrl,
  getChat,
  getWorkJob,
  listChats,
  listWorkJobs
} from '../lib/api'
import type {
  BrowserViewStatus,
  ChatDetail,
  ChatSummary,
  WorkEvent,
  WorkJobDetail,
  WorkJobSummary
} from '../lib/types'
import { relativeTime } from '../lib/format'
import MessageBubble from './MessageBubble'

interface RightPanelProps {
  base: string
  character: string
  open: boolean
}

/**
 * The right-side contextual panel, rebuilt (owner feedback round 2) to match
 * the reference screenshots precisely: NOT a bordered sub-panel with its own
 * header bar (that's what produced the redundant "{character}" label the
 * owner kept seeing even after a prior phase claimed to remove name bars
 * elsewhere -- it had quietly come back here). Now there is no panel header
 * at all: just stacked, individually-rounded "block" sections (Progress,
 * History, Browser, Scheduled tasks) sitting directly on the app's normal
 * bg-app background, each its own bg-200 surface with an icon/label/chevron
 * header and a gap between blocks -- no wrapping frame or border around the
 * whole panel, exactly per the reference. History used to be a separate
 * modal triggered from an icon under Composer (see HistoryOverlay's prior
 * incarnation); that trigger is retired and its content lives here as one
 * more collapsible block instead, per owner brief item 2.
 *
 * ChatView still lays out `[chat column][this panel]` as one flex row, and
 * only this wrapper's width animates open/closed -- see ChatView.tsx.
 */
function RightPanel({ base, character, open }: RightPanelProps): React.JSX.Element {
  const [progressOpen, setProgressOpen] = useState(true)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [browserOpen, setBrowserOpen] = useState(false)
  const [scheduledOpen, setScheduledOpen] = useState(false)

  return (
    <div
      className={`h-full flex-none overflow-hidden bg-bg-app transition-[width] duration-300 ease-in-out ${
        open ? 'w-96' : 'w-0'
      }`}
    >
      <div className="flex h-full w-96 flex-col overflow-y-auto px-3 py-3">
        <div className="flex flex-col gap-2.5">
          <CollapsibleBlock
            icon={<span aria-hidden>&#9642;</span>}
            label="Progress"
            open={progressOpen}
            onToggle={() => setProgressOpen((v) => !v)}
          >
            <ProgressSection base={base} character={character} active={open && progressOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#8635;</span>}
            label="History"
            open={historyOpen}
            onToggle={() => setHistoryOpen((v) => !v)}
          >
            <HistorySection base={base} character={character} active={open && historyOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#8859;</span>}
            label="Browser"
            open={browserOpen}
            onToggle={() => setBrowserOpen((v) => !v)}
          >
            <BrowserViewSection base={base} character={character} active={open && browserOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#9711;</span>}
            label="Scheduled tasks"
            open={scheduledOpen}
            onToggle={() => setScheduledOpen((v) => !v)}
          >
            <ScheduledTasksSection />
          </CollapsibleBlock>
        </div>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Shared collapsible "block" shell -- the one visual unit the whole panel
// is built from. bg-200 on top of the panel's bg-app background is the
// "barely distinguished, not a bright card" surface pairing from the
// reference; radius-2xl (1rem/16px) matches its rounded-corner blocks;
// gap between blocks comes from the parent's flex gap-2.5, not anything
// here. No border/outline anywhere on this shell -- the surface-color step
// is the only thing that reads as "a block".
// --------------------------------------------------------------------------

function CollapsibleBlock({
  icon,
  label,
  open,
  onToggle,
  children
}: {
  icon: React.ReactNode
  label: string
  open: boolean
  onToggle: () => void
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="overflow-hidden rounded-2xl bg-bg-200">
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2.5 px-3.5 py-3 text-left transition hover:bg-bg-300/40"
      >
        <span className="flex h-5 w-5 flex-none items-center justify-center text-sm text-ink-dim">
          {icon}
        </span>
        <span className="flex-1 text-sm font-medium text-ink">{label}</span>
        <span
          aria-hidden
          className={`flex-none text-[10px] text-ink-faint transition-transform duration-200 ${
            open ? '' : '-rotate-90'
          }`}
        >
          &#9662;
        </span>
      </button>
      {open && <div className="px-3.5 pb-3.5">{children}</div>}
    </div>
  )
}

// --------------------------------------------------------------------------
// Progress list — real /api/work job data, no fake data.
// --------------------------------------------------------------------------

function statusColor(status: string): string {
  if (status === 'running') return 'text-accent'
  if (status === 'error') return 'text-rose-300'
  if (status === 'cancelled') return 'text-ink-faint'
  return 'text-emerald-300'
}

function eventLine(e: WorkEvent): string {
  if (e.kind === 'tool_call') return `→ ${e.name}`
  if (e.kind === 'tool_result') return `${e.ok === false ? '✗' : '✓'} ${e.name}`
  return (e.text || '').slice(0, 80)
}

/**
 * First-pass stub, honestly labelled as such: shows whatever real job/event
 * data /api/work already has for this character. The owner is planning a
 * dedicated harness "task list" tool later that will make this richer
 * (multi-step plans, retries, etc.) — this just surfaces what exists today.
 */
function ProgressSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [jobs, setJobs] = useState<WorkJobSummary[]>([])
  const [details, setDetails] = useState<Record<string, WorkJobDetail>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!active) return
    let cancelled = false

    const refresh = async (): Promise<void> => {
      try {
        const all = await listWorkJobs(base)
        if (cancelled) return
        const mine = all.filter((j) => j.character === character)
        setJobs(mine)
        setError(null)
        const running = mine.filter((j) => j.status === 'running').slice(0, 5)
        const fetched = await Promise.all(
          running.map((j) => getWorkJob(base, j.id).catch(() => null))
        )
        if (cancelled) return
        setDetails((prev) => {
          const next = { ...prev }
          fetched.forEach((d, i) => {
            if (d) next[running[i].id] = d
          })
          return next
        })
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }

    refresh()
    const id = setInterval(refresh, 2500)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [base, character, active])

  return (
    <div>
      {error && <div className="mb-2 text-xs text-rose-300">{error}</div>}
      {jobs.length === 0 ? (
        <p className="text-xs text-ink-faint">
          Nothing running for {character || 'this character'} right now.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {jobs.slice(0, 8).map((j) => {
            const detail = details[j.id]
            return (
              <li key={j.id} className="rounded-lg bg-bg-300/50 px-2.5 py-2">
                <div className="flex items-start justify-between gap-2">
                  <span className="truncate text-xs text-ink">{j.task}</span>
                  <span
                    className={`flex-none text-[10px] font-medium uppercase ${statusColor(j.status)}`}
                  >
                    {j.status}
                  </span>
                </div>
                <div className="mt-0.5 text-[10px] text-ink-faint">{relativeTime(j.started)}</div>
                {j.pending_approval && (
                  <div className="mt-1 rounded bg-amber-500/10 px-1.5 py-1 text-[10px] text-amber-300">
                    Waiting on approval: {j.pending_approval.tool}
                  </div>
                )}
                {detail && detail.events.length > 0 && (
                  <ul className="mt-1.5 flex flex-col gap-0.5 border-t border-border pt-1.5">
                    {detail.events.slice(-3).map((e, i) => (
                      <li key={i} className="truncate text-[10px] text-ink-faint">
                        {eventLine(e)}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// History — folded into the panel per owner brief item 2 (previously a
// separate modal, HistoryOverlay, triggered by its own icon under Composer).
// Same data/behavior as before: a read-only list of a character's past
// chats plus a read-only transcript viewer. Selecting one never makes it
// "the live chat" again -- old chats are reference only, never resumed.
// --------------------------------------------------------------------------

function HistorySection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [viewing, setViewing] = useState<ChatDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!active) return
    setViewing(null)
    listChats(base)
      .then((all) => setChats(all.filter((c) => c.character === character)))
      .catch((e) => setError((e as Error).message))
  }, [base, character, active])

  if (viewing) {
    return (
      <div>
        <button
          onClick={() => setViewing(null)}
          className="mb-2 text-xs text-ink-dim transition hover:text-ink"
        >
          &larr; Back to {character}&rsquo;s history
        </button>
        <div className="max-h-80 overflow-y-auto rounded-lg bg-bg-300/40">
          {viewing.messages.length === 0 && (
            <div className="p-3 text-xs text-ink-faint">Empty chat.</div>
          )}
          {viewing.messages.map((m, i) => (
            <MessageBubble key={i} message={m} base={base} character={character} />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div>
      <p className="mb-2 text-[11px] text-ink-faint">
        Read-only reference — {character || 'this character'}&rsquo;s active conversation is always
        the live chat above. Opening an old one here doesn&rsquo;t resume it.
      </p>
      {error && <div className="mb-2 text-xs text-rose-300">{error}</div>}
      {chats.length === 0 ? (
        <p className="text-xs text-ink-faint">No past chats yet.</p>
      ) : (
        <ul className="flex flex-col gap-1">
          {chats.map((c) => (
            <li key={c.id}>
              <button
                onClick={() =>
                  getChat(base, c.id)
                    .then(setViewing)
                    .catch((e) => setError((e as Error).message))
                }
                className="flex w-full flex-col items-start rounded-lg px-2.5 py-2 text-left transition hover:bg-bg-300/50"
              >
                <span className="truncate text-xs text-ink">{c.title || '(new chat)'}</span>
                <span className="text-[10px] text-ink-faint">
                  {relativeTime(c.updated)} &middot; {c.messages} messages
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Scheduled tasks — honest placeholder. No fake data: the backend has no
// real scheduler yet (that's a separate future phase).
// --------------------------------------------------------------------------

function ScheduledTasksSection(): React.JSX.Element {
  return (
    <div className="rounded-lg border border-dashed border-border px-2.5 py-3 text-center">
      <p className="text-xs text-ink-faint">Coming soon.</p>
      <p className="mt-1 text-[10px] text-ink-faint">
        Pleiades doesn&rsquo;t have a real task scheduler yet — this section is reserved for that
        future phase.
      </p>
    </div>
  )
}

// --------------------------------------------------------------------------
// Browser-use embedded view — live Playwright/Chromium CDP screencast.
//
// Owner brief items 5-6: this session is now headless by default (no OS
// window pops up), it's the SAME session the model's own `browser` tool
// drives in chat (see pleiades/tools/browser.py + pleiades/webui/
// browser_view.py), and it offers two explicit, human-triggered escape
// hatches this component wires up: "Open separately" (switches the running
// session to a real headed OS window in place) and "Expand" (a larger
// in-app view, requesting a bigger server-side viewport via resize() rather
// than just upscaling a blurry small image).
// --------------------------------------------------------------------------

/** Map a pointer/keyboard-capable client rect to the backend's fixed
 * viewport pixel space (frames are always rendered at that resolution
 * regardless of how large the <img> is drawn on screen). */
function toViewportCoords(
  e: { clientX: number; clientY: number },
  el: HTMLElement,
  viewport: { width: number; height: number }
): { x: number; y: number } {
  const rect = el.getBoundingClientRect()
  const scaleX = viewport.width / rect.width
  const scaleY = viewport.height / rect.height
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
}

const EXPANDED_VIEWPORT = { width: 1600, height: 1000 }
const DEFAULT_VIEWPORT = { width: 1280, height: 800 }

function BrowserViewSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [status, setStatus] = useState<BrowserViewStatus | null>(null)
  const [urlInput, setUrlInput] = useState('')
  const [interactive, setInteractive] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  const expandedImgRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const frameUrlRef = useRef<string | null>(null)
  const [frameSrc, setFrameSrc] = useState<string | null>(null)

  const closeSocket = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  // Poll status even when the socket isn't open, so a session started/
  // stopped from elsewhere (the model's own browser tool, or a backend
  // restart) is reflected here live.
  useEffect(() => {
    if (!active || !character) return
    let cancelled = false
    const poll = async (): Promise<void> => {
      try {
        const s = await browserViewStatus(base, character)
        if (!cancelled) setStatus(s)
      } catch {
        // backend not reachable / no session yet — leave last-known status
      }
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [base, character, active])

  // Own the WebSocket lifecycle: connect while a session is running/
  // starting, tear down on stop/unmount/character switch.
  useEffect(() => {
    if (!active || !character) return
    if (!status || status.status === 'stopped') return

    const ws = new WebSocket(browserViewWsUrl(base, character))
    ws.binaryType = 'blob'
    wsRef.current = ws
    ws.onmessage = (evt) => {
      if (typeof evt.data === 'string') {
        try {
          const msg = JSON.parse(evt.data)
          if (msg.type === 'status') setStatus(msg)
        } catch {
          // ignore
        }
        return
      }
      const blob = evt.data as Blob
      const url = URL.createObjectURL(blob)
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
      frameUrlRef.current = url
      setFrameSrc(url)
    }
    ws.onerror = () => setError('Live view connection error.')
    return () => {
      ws.close()
      if (wsRef.current === ws) wsRef.current = null
    }
    // Reconnect whenever the session's status transitions (e.g. stopped -> starting).
  }, [base, character, active, status?.status])

  useEffect(
    () => () => {
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
    },
    []
  )

  // Model calls the chat browser tool -> goto starts the session headless
  // -> this component's status poll picks it up within ~3s and the WS
  // effect above connects automatically, exactly like a human-triggered
  // start. Nothing special is needed here for that path to "just show up".

  const start = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      const s = await browserViewStart(base, character, urlInput || 'https://example.com')
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const stop = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      closeSocket()
      setFrameSrc(null)
      setInteractive(false)
      setExpanded(false)
      const s = await browserViewStop(base, character)
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const navigate = async (): Promise<void> => {
    if (!urlInput.trim()) return
    setBusy(true)
    setError(null)
    try {
      await browserViewNavigate(base, character, urlInput.trim())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const openSeparate = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      const s = await browserViewOpenSeparate(base, character)
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const toggleExpand = async (): Promise<void> => {
    const next = !expanded
    setExpanded(next)
    if (status?.status !== 'running') return
    try {
      const target = next ? EXPANDED_VIEWPORT : DEFAULT_VIEWPORT
      const s = await browserViewResize(base, character, target.width, target.height)
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const sendInput = (evt: Record<string, unknown>): void => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(evt))
    }
  }

  const viewport = status?.viewport || DEFAULT_VIEWPORT

  const onMouseEvent = (
    ref: React.RefObject<HTMLImageElement | null>,
    type: 'mousePressed' | 'mouseMoved' | 'mouseReleased',
    e: React.MouseEvent<HTMLImageElement>
  ): void => {
    if (!interactive || !ref.current) return
    const { x, y } = toViewportCoords(e, ref.current, viewport)
    sendInput({ kind: 'mouse', type, x, y, button: 'left', clickCount: 1 })
  }

  const onWheel = (
    ref: React.RefObject<HTMLImageElement | null>,
    e: React.WheelEvent<HTMLImageElement>
  ): void => {
    if (!interactive || !ref.current) return
    const { x, y } = toViewportCoords(e, ref.current, viewport)
    sendInput({ kind: 'mouse', type: 'mouseWheel', x, y, deltaX: e.deltaX, deltaY: e.deltaY })
  }

  const onKey = (type: 'keyDown' | 'keyUp', e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (!interactive) return
    e.preventDefault()
    const printable = e.key.length === 1 ? e.key : undefined
    sendInput({
      kind: 'key',
      type,
      key: e.key,
      code: e.code,
      keyCode: e.keyCode,
      text: type === 'keyDown' ? printable : undefined
    })
  }

  const running = status?.status === 'running'
  const starting = status?.status === 'starting'

  const frameView = (imgRefToUse: React.RefObject<HTMLImageElement | null>, big: boolean): React.JSX.Element => (
    <div
      tabIndex={interactive ? 0 : -1}
      onKeyDown={(e) => onKey('keyDown', e)}
      onKeyUp={(e) => onKey('keyUp', e)}
      className={`relative overflow-hidden rounded-lg bg-black/40 outline-none ${big ? 'h-full w-full' : ''}`}
    >
      {frameSrc ? (
        <img
          ref={imgRefToUse}
          src={frameSrc}
          alt="Live browser view"
          draggable={false}
          onMouseDown={(e) => onMouseEvent(imgRefToUse, 'mousePressed', e)}
          onMouseUp={(e) => onMouseEvent(imgRefToUse, 'mouseReleased', e)}
          onMouseMove={(e) => onMouseEvent(imgRefToUse, 'mouseMoved', e)}
          onWheel={(e) => onWheel(imgRefToUse, e)}
          className={`w-full select-none ${big ? 'h-full object-contain' : ''} ${
            interactive ? 'cursor-crosshair' : 'cursor-default'
          }`}
        />
      ) : (
        <div className="flex h-40 items-center justify-center text-xs text-ink-faint">
          {starting ? 'Starting browser…' : 'Waiting for the first frame…'}
        </div>
      )}
    </div>
  )

  return (
    <div>
      {!running && !starting && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-ink-faint">
            Live, embeddable browser session for {character || 'this character'} — the same session
            the model&rsquo;s own browser tool drives in chat. Hidden by default (headless); use
            &ldquo;Open separately&rdquo; below to pop a real window once it&rsquo;s running.
          </p>
          <button
            onClick={start}
            disabled={busy || !character}
            className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? 'Starting…' : 'Start browser session'}
          </button>
        </div>
      )}

      {(running || starting) && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1.5">
            <input
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && navigate()}
              placeholder={status?.url || 'https://…'}
              className="min-w-0 flex-1 rounded-lg bg-bg-surface px-2 py-1 text-xs text-ink placeholder:text-ink-faint focus:outline-none"
            />
            <button
              onClick={navigate}
              disabled={busy}
              className="flex-none rounded-lg bg-bg-300 px-2 py-1 text-xs text-ink transition hover:bg-bg-400 disabled:opacity-40"
            >
              Go
            </button>
          </div>

          {frameView(imgRef, false)}

          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => setInteractive((v) => !v)}
              className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition ${
                interactive ? 'bg-accent text-white' : 'bg-bg-300 text-ink-dim hover:text-ink'
              }`}
              title="Forward clicks/keys from this view into the real page"
            >
              {interactive ? 'Interaction: ON' : 'Enter interaction mode'}
            </button>
            <button
              onClick={toggleExpand}
              disabled={!running}
              className="rounded-lg bg-bg-300 px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-bg-400 hover:text-ink disabled:opacity-40"
              title="Open a larger view within the app"
            >
              Expand
            </button>
            <button
              onClick={openSeparate}
              disabled={busy || !running || !!status?.headed}
              className="rounded-lg bg-bg-300 px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-bg-400 hover:text-ink disabled:opacity-40"
              title="Switch this session to a real OS window (same session, same login state)"
            >
              {status?.headed ? 'Open window: on-screen' : 'Open separately'}
            </button>
            <button
              onClick={stop}
              disabled={busy}
              className="ml-auto rounded-lg px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-rose-500/20 hover:text-rose-300"
            >
              Stop session
            </button>
          </div>
          {interactive && (
            <p className="text-[10px] text-ink-faint">
              Click the view to focus it, then your clicks/scroll/keys go to the real page.
            </p>
          )}
        </div>
      )}

      {status?.status === 'error' && (
        <div className="mt-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-2.5 py-2 text-xs text-rose-300">
          {status.error}
        </div>
      )}
      {error && <div className="mt-2 text-xs text-rose-300">{error}</div>}

      {expanded && (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-8"
          onClick={() => toggleExpand()}
        >
          <div
            className="flex h-full max-h-[90vh] w-full max-w-5xl flex-col gap-2 rounded-2xl bg-bg-100 p-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-none items-center justify-between">
              <span className="truncate text-sm text-ink-dim">{status?.url || character}</span>
              <button
                onClick={() => toggleExpand()}
                className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
              >
                &#10005;
              </button>
            </div>
            <div className="min-h-0 flex-1">{frameView(expandedImgRef, true)}</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default RightPanel
