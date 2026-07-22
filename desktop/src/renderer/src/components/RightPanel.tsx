import { useCallback, useEffect, useRef, useState } from 'react'
import {
  browserViewNavigate,
  browserViewStart,
  browserViewStatus,
  browserViewStop,
  browserViewWsUrl,
  getWorkJob,
  listWorkJobs
} from '../lib/api'
import type { BrowserViewStatus, WorkEvent, WorkJobDetail, WorkJobSummary } from '../lib/types'
import { relativeTime } from '../lib/format'

interface RightPanelProps {
  base: string
  character: string
  open: boolean
  onClose: () => void
}

/**
 * The right-side contextual panel (Phase 14). Unlike RightPanelStub (the
 * placeholder this replaces), this is a real flex sibling of the chat
 * column, not an overlay: `ChatView` lays out `[chat column][this panel]`
 * in one flex row, and only this component's own width animates between
 * 0 and its open width — the chat column's `flex-1 min-w-0` does the rest,
 * so opening the panel visibly squishes the chat column narrower exactly
 * like Claude Desktop's own side panel, instead of floating on top of it.
 *
 * Deliberately does NOT include a fourth "History" section. HistoryOverlay
 * already exists as its own complete, separately-triggered modal (icon
 * docked under Composer) from a prior phase. Folding it into this panel
 * too would mean two History entry points for the same read-only data —
 * the owner's brief explicitly says: when in doubt, leave that alone. This
 * panel covers the three pieces that didn't already have a home: live work
 * ("Progress"), the (not-yet-real) task scheduler placeholder, and the new
 * browser-use embed.
 */
function RightPanel({ base, character, open }: RightPanelProps): React.JSX.Element {
  return (
    <div
      className={`h-full flex-none overflow-hidden border-l border-border bg-bg-100 transition-[width] duration-300 ease-in-out ${
        open ? 'w-96' : 'w-0'
      }`}
    >
      <div className="flex h-full w-96 flex-col">
        <div className="flex-none border-b border-border px-4 py-3">
          <span className="text-sm font-medium text-ink-bright">{character || 'Panel'}</span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <ProgressSection base={base} character={character} active={open} />
          <ScheduledTasksSection />
          <BrowserViewSection base={base} character={character} active={open} />
        </div>
      </div>
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
    <div className="border-b border-border px-4 py-3">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-faint">
        Progress
      </h3>
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
              <li key={j.id} className="rounded-lg bg-bg-200/60 px-2.5 py-2">
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
// Scheduled tasks — honest placeholder. No fake data: the backend has no
// real scheduler yet (that's a separate future phase).
// --------------------------------------------------------------------------

function ScheduledTasksSection(): React.JSX.Element {
  return (
    <div className="border-b border-border px-4 py-3">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-faint">
        Scheduled tasks
      </h3>
      <div className="rounded-lg border border-dashed border-border px-2.5 py-3 text-center">
        <p className="text-xs text-ink-faint">Coming soon.</p>
        <p className="mt-1 text-[10px] text-ink-faint">
          Pleiades doesn&rsquo;t have a real task scheduler yet — this section is reserved for that
          future phase.
        </p>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Browser-use embedded view — live Playwright/Chromium CDP screencast.
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
  const imgRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const frameUrlRef = useRef<string | null>(null)
  const [frameSrc, setFrameSrc] = useState<string | null>(null)

  const closeSocket = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  // Poll status even when the socket isn't open, so a session started/
  // stopped from elsewhere (or lost on a backend restart) is reflected.
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
    const id = setInterval(poll, 4000)
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

  const sendInput = (evt: Record<string, unknown>): void => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(evt))
    }
  }

  const viewport = status?.viewport || { width: 1280, height: 800 }

  const onMouseEvent = (
    type: 'mousePressed' | 'mouseMoved' | 'mouseReleased',
    e: React.MouseEvent<HTMLImageElement>
  ): void => {
    if (!interactive || !imgRef.current) return
    const { x, y } = toViewportCoords(e, imgRef.current, viewport)
    sendInput({ kind: 'mouse', type, x, y, button: 'left', clickCount: 1 })
  }

  const onWheel = (e: React.WheelEvent<HTMLImageElement>): void => {
    if (!interactive || !imgRef.current) return
    const { x, y } = toViewportCoords(e, imgRef.current, viewport)
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

  return (
    <div className="px-4 py-3">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-faint">Browser</h3>

      {!running && !starting && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-ink-faint">
            Live, embeddable browser session for {character || 'this character'} (Playwright +
            Chromium — separate from the agent&rsquo;s own Camoufox browsing tool).
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

          <div
            tabIndex={interactive ? 0 : -1}
            onKeyDown={(e) => onKey('keyDown', e)}
            onKeyUp={(e) => onKey('keyUp', e)}
            className="relative overflow-hidden rounded-lg border border-border bg-black/40 outline-none"
          >
            {frameSrc ? (
              <img
                ref={imgRef}
                src={frameSrc}
                alt="Live browser view"
                draggable={false}
                onMouseDown={(e) => onMouseEvent('mousePressed', e)}
                onMouseUp={(e) => onMouseEvent('mouseReleased', e)}
                onMouseMove={(e) => onMouseEvent('mouseMoved', e)}
                onWheel={onWheel}
                className={`w-full select-none ${interactive ? 'cursor-crosshair' : 'cursor-default'}`}
              />
            ) : (
              <div className="flex h-40 items-center justify-center text-xs text-ink-faint">
                {starting ? 'Starting browser…' : 'Waiting for the first frame…'}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between gap-2">
            <button
              onClick={() => setInteractive((v) => !v)}
              className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition ${
                interactive ? 'bg-accent text-white' : 'bg-bg-300 text-ink-dim hover:text-ink'
              }`}
              title="Forward clicks/keys from this view into the real page"
            >
              {interactive ? 'Interaction mode: ON' : 'Enter interaction mode'}
            </button>
            <button
              onClick={stop}
              disabled={busy}
              className="rounded-lg px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-rose-500/20 hover:text-rose-300"
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
    </div>
  )
}

export default RightPanel
