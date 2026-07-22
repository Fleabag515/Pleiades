import { useCallback, useEffect, useRef, useState } from 'react'
import Shell from './components/Shell'
import TitleBar from './components/TitleBar'
import { ChatSessionsProvider } from './lib/chatStore'

interface BackendStatus {
  phase: 'starting' | 'ready' | 'error'
  port: number | null
  url: string | null
  error: string | null
}

type ViewState =
  | { kind: 'connecting'; detail: string }
  | { kind: 'connected'; url: string }
  | { kind: 'error'; message: string }

function App(): React.JSX.Element {
  const [view, setView] = useState<ViewState>({ kind: 'connecting', detail: 'Starting backend…' })
  const cancelled = useRef(false)

  const connect = useCallback(async () => {
    cancelled.current = false
    setView({ kind: 'connecting', detail: 'Starting backend…' })

    if (!window.pleiades) {
      setView({ kind: 'error', message: 'Preload bridge unavailable (window.pleiades missing).' })
      return
    }

    const backendUrl = await window.pleiades.getBackendUrl()

    // Poll main's view of backend readiness until it flips to ready/error.
    const deadline = Date.now() + 25_000
    let status: BackendStatus = await window.pleiades.getBackendStatus()
    while (!cancelled.current && status.phase === 'starting' && Date.now() < deadline) {
      setView({ kind: 'connecting', detail: 'Waiting for Pleiades backend to come online…' })
      await new Promise((r) => setTimeout(r, 500))
      status = await window.pleiades.getBackendStatus()
    }

    if (cancelled.current) return

    if (status.phase === 'error') {
      setView({ kind: 'error', message: status.error ?? 'Backend failed to start.' })
      return
    }

    if (status.phase !== 'ready' || !backendUrl) {
      setView({ kind: 'error', message: 'Timed out waiting for the backend to become ready.' })
      return
    }

    try {
      const res = await fetch(`${backendUrl}/api/status`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      if (!cancelled.current) setView({ kind: 'connected', url: backendUrl })
    } catch (err) {
      if (!cancelled.current) {
        setView({
          kind: 'error',
          message: `Backend reachable but /api/status failed: ${(err as Error).message}`
        })
      }
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      cancelled.current = true
    }
  }, [connect])

  if (view.kind === 'connected') {
    return (
      <div className="flex h-screen w-full flex-col overflow-hidden">
        <TitleBar />
        <div className="min-h-0 flex-1">
          <ChatSessionsProvider>
            <Shell base={view.url} />
          </ChatSessionsProvider>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden">
      <TitleBar />
      <div className="flex min-h-0 flex-1 items-center justify-center bg-bg-app p-8 text-ink">
        <div className="w-full max-w-lg rounded-2xl border border-border bg-bg-surface p-8 shadow-xl">
          <h1 className="mb-1 text-xl font-semibold tracking-tight">Pleiades</h1>
          <p className="mb-6 text-sm text-ink-dim">Connecting to the local backend…</p>

          {view.kind === 'connecting' && (
            <div className="flex items-center gap-3 text-ink">
              <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-accent" />
              <span>{view.detail}</span>
            </div>
          )}

          {view.kind === 'error' && (
            <div className="space-y-4">
              <div className="flex items-start gap-3 text-rose-300">
                <span className="mt-1 h-2.5 w-2.5 flex-none rounded-full bg-rose-400" />
                <span className="text-sm">{view.message}</span>
              </div>
              <button
                onClick={connect}
                className="rounded-lg bg-bg-surface-hover px-4 py-2 text-sm font-medium text-ink transition hover:bg-accent hover:text-white"
              >
                Retry
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
