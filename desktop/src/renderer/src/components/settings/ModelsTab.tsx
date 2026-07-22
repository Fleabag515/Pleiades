import { useEffect, useRef, useState } from 'react'
import { deleteModel, listModels, modelLogs, startModel, stopModel, updateModel } from '../../lib/api'
import type { ModelEntry } from '../../lib/types'
import { modelDisplayName } from '../../lib/format'

interface ModelsTabProps {
  base: string
}

function StatePill({ state }: { state: ModelEntry['state'] }): React.JSX.Element {
  const styles: Record<ModelEntry['state'], string> = {
    running: 'bg-emerald-500/15 text-emerald-300',
    loading: 'bg-amber-500/15 text-amber-300',
    crashed: 'bg-rose-500/15 text-rose-300',
    stopped: 'bg-bg-surface-hover text-ink-dim'
  }
  return <span className={`rounded-full px-2 py-0.5 text-xs ${styles[state]}`}>{state}</span>
}

/** Registered GGUFs: start/stop, live log tail, rename (display-name
 * override, separate from the registry key `name`), and delete. Delete
 * removes the underlying GGUF file too (backend default), so it's gated
 * behind a native confirm() that says so explicitly. */
function ModelsTab({ base }: ModelsTabProps): React.JSX.Element {
  const [models, setModels] = useState<ModelEntry[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [renameError, setRenameError] = useState<string | null>(null)

  const [logsFor, setLogsFor] = useState<string | null>(null)
  const [logText, setLogText] = useState('')
  const [logError, setLogError] = useState<string | null>(null)
  const logPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const refresh = async (): Promise<void> => {
    try {
      setModels(await listModels(base))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base])

  // Stop polling logs when the panel closes or the tab unmounts.
  useEffect(() => {
    return () => {
      if (logPollRef.current) clearInterval(logPollRef.current)
    }
  }, [])

  const toggle = async (m: ModelEntry): Promise<void> => {
    setBusy(m.name)
    setError(null)
    try {
      if (m.running) await stopModel(base, m.name)
      else await startModel(base, m.name)
      // Model start is async on the backend; give it a beat then refresh.
      await new Promise((r) => setTimeout(r, 400))
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const startRename = (m: ModelEntry): void => {
    setRenaming(m.name)
    setRenameValue(m.display_name ?? '')
    setRenameError(null)
  }

  const cancelRename = (): void => {
    setRenaming(null)
    setRenameError(null)
  }

  const saveRename = async (name: string): Promise<void> => {
    setRenameError(null)
    try {
      await updateModel(base, name, { display_name: renameValue.trim() })
      setRenaming(null)
      await refresh()
    } catch (e) {
      setRenameError((e as Error).message)
    }
  }

  const openLogs = async (name: string): Promise<void> => {
    if (logsFor === name) {
      // toggle closed
      setLogsFor(null)
      if (logPollRef.current) clearInterval(logPollRef.current)
      return
    }
    setLogsFor(name)
    setLogText('')
    setLogError(null)
    const tick = async (): Promise<void> => {
      try {
        const r = await modelLogs(base, name, 300)
        setLogText(r.log)
      } catch (e) {
        setLogError((e as Error).message)
      }
    }
    await tick()
    if (logPollRef.current) clearInterval(logPollRef.current)
    logPollRef.current = setInterval(tick, 2000)
  }

  const remove = async (m: ModelEntry): Promise<void> => {
    const label = modelDisplayName(m)
    const ok = window.confirm(
      `Delete "${label}"? This stops it if running and permanently deletes the GGUF file(s) from disk. This cannot be undone.`
    )
    if (!ok) return
    setBusy(m.name)
    setError(null)
    try {
      await deleteModel(base, m.name)
      if (logsFor === m.name) setLogsFor(null)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {error && (
        <div className="mb-1 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}
      {models.length === 0 && <p className="text-sm text-ink-dim">No models registered yet.</p>}
      {models.map((m) => (
        <div key={m.name} className="rounded-xl border border-border bg-bg-app px-4 py-3">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              {renaming === m.name ? (
                <div className="flex items-center gap-1.5">
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveRename(m.name)
                      if (e.key === 'Escape') cancelRename()
                    }}
                    placeholder={m.name}
                    className="min-w-0 flex-1 rounded-lg border border-border bg-bg-surface px-2 py-1 text-sm text-ink focus:outline-none"
                  />
                  <button
                    onClick={() => saveRename(m.name)}
                    className="flex-none rounded-lg border border-border px-2 py-1 text-xs text-ink transition hover:bg-bg-surface-hover"
                  >
                    Save
                  </button>
                  <button
                    onClick={cancelRename}
                    className="flex-none rounded-lg px-2 py-1 text-xs text-ink-dim transition hover:text-ink"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium text-ink">{modelDisplayName(m)}</span>
                  {m.display_name && (
                    <span className="truncate text-xs text-ink-faint" title="Underlying registered name">
                      ({m.name})
                    </span>
                  )}
                  <StatePill state={m.state} />
                  <button
                    onClick={() => startRename(m)}
                    title="Rename display name"
                    aria-label="Rename"
                    className="flex-none rounded px-1 text-xs text-ink-faint transition hover:text-ink"
                  >
                    ✎
                  </button>
                </div>
              )}
              {renameError && <div className="mt-1 text-xs text-rose-300">{renameError}</div>}
              <div className="truncate text-xs text-ink-dim">
                ctx {m.n_ctx} · gpu_layers {m.n_gpu_layers} {m.chat_format ? `· ${m.chat_format}` : ''}
              </div>
            </div>
            <div className="flex flex-none items-center gap-1.5">
              <button
                onClick={() => openLogs(m.name)}
                className={`rounded-lg border px-3 py-1.5 text-sm transition hover:bg-bg-surface-hover ${
                  logsFor === m.name ? 'border-accent text-ink-bright' : 'border-border text-ink'
                }`}
              >
                Logs
              </button>
              <button
                disabled={busy === m.name}
                onClick={() => toggle(m)}
                className="rounded-lg border border-border px-3 py-1.5 text-sm text-ink transition hover:bg-bg-surface-hover disabled:opacity-50"
              >
                {busy === m.name ? '…' : m.running ? 'Stop' : 'Start'}
              </button>
              <button
                disabled={busy === m.name}
                onClick={() => remove(m)}
                title="Delete this model (also deletes the GGUF file from disk)"
                className="rounded-lg border border-rose-500/30 px-3 py-1.5 text-sm text-rose-300 transition hover:bg-rose-500/10 disabled:opacity-50"
              >
                Delete
              </button>
            </div>
          </div>

          {logsFor === m.name && (
            <div className="mt-2 rounded-lg border border-border bg-bg-000/60 p-2">
              {logError && <div className="mb-1 text-xs text-rose-300">{logError}</div>}
              <pre className="max-h-56 overflow-y-auto whitespace-pre-wrap break-all text-[11px] leading-snug text-ink-dim">
                {logText || 'No log output yet.'}
              </pre>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export default ModelsTab
