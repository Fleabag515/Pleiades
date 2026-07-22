import { useCallback, useEffect, useRef, useState } from 'react'

interface BackendStatus {
  phase: 'starting' | 'ready' | 'error'
  port: number | null
  url: string | null
  error: string | null
}

interface ApiStatus {
  services?: {
    anamnesis?: { up: boolean; characters?: number }
    inference?: { up: boolean; state?: string }
    searxng?: { up: boolean }
  }
  counts?: { profiles?: number; models?: number; models_running?: number }
}

type ViewState =
  | { kind: 'connecting'; detail: string }
  | { kind: 'connected'; api: ApiStatus }
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
      const api = (await res.json()) as ApiStatus
      if (!cancelled.current) setView({ kind: 'connected', api })
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

  return (
    <div className="flex min-h-screen w-full items-center justify-center p-8 text-slate-100">
      <div className="w-full max-w-lg rounded-2xl border border-slate-700/60 bg-slate-900/60 p-8 shadow-xl">
        <h1 className="mb-1 text-xl font-semibold tracking-tight">Pleiades</h1>
        <p className="mb-6 text-sm text-slate-400">Desktop shell — Phase A wiring check</p>

        {view.kind === 'connecting' && (
          <div className="flex items-center gap-3 text-slate-300">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-amber-400" />
            <span>{view.detail}</span>
          </div>
        )}

        {view.kind === 'connected' && (
          <div className="space-y-2 text-slate-200">
            <div className="flex items-center gap-3">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
              <span className="font-medium">Connected</span>
            </div>
            <p className="text-sm text-slate-300">
              {view.api.counts?.profiles ?? 0} characters, engine:{' '}
              {view.api.services?.inference?.state ??
                (view.api.services?.inference?.up ? 'up' : 'down')}
            </p>
            <p className="text-xs text-slate-500">
              Anamnesis {view.api.services?.anamnesis?.up ? 'up' : 'down'} · SearXNG{' '}
              {view.api.services?.searxng?.up ? 'up' : 'down'} · Models{' '}
              {view.api.counts?.models ?? 0} ({view.api.counts?.models_running ?? 0} running)
            </p>
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
              className="rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-slate-600"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
