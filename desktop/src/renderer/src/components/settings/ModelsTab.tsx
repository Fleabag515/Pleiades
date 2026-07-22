import { useEffect, useState } from 'react'
import { listModels, startModel, stopModel } from '../../lib/api'
import type { ModelEntry } from '../../lib/types'

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

/** Registered GGUFs: start/stop lifecycle. Unchanged behavior from the
 * read-only-Characters phase — this tab was already fully functional. */
function ModelsTab({ base }: ModelsTabProps): React.JSX.Element {
  const [models, setModels] = useState<ModelEntry[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

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

  return (
    <div className="flex flex-col gap-2">
      {error && (
        <div className="mb-1 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}
      {models.length === 0 && <p className="text-sm text-ink-dim">No models registered yet.</p>}
      {models.map((m) => (
        <div
          key={m.name}
          className="flex items-center justify-between rounded-xl border border-border bg-bg-app px-4 py-3"
        >
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-ink">{m.name}</span>
              <StatePill state={m.state} />
            </div>
            <div className="truncate text-xs text-ink-dim">
              ctx {m.n_ctx} · gpu_layers {m.n_gpu_layers} {m.chat_format ? `· ${m.chat_format}` : ''}
            </div>
          </div>
          <button
            disabled={busy === m.name}
            onClick={() => toggle(m)}
            className="flex-none rounded-lg border border-border px-3 py-1.5 text-sm text-ink transition hover:bg-bg-surface-hover disabled:opacity-50"
          >
            {busy === m.name ? '…' : m.running ? 'Stop' : 'Start'}
          </button>
        </div>
      ))}
    </div>
  )
}

export default ModelsTab
