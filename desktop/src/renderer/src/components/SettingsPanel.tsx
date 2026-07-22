import { useEffect, useState } from 'react'
import { listModels, listProfiles, startModel, stopModel } from '../lib/api'
import type { ModelEntry, Profile } from '../lib/types'
import Avatar from './Avatar'

interface SettingsPanelProps {
  base: string
  onClose: () => void
}

type Tab = 'models' | 'characters'

function StatePill({ state }: { state: ModelEntry['state'] }): React.JSX.Element {
  const styles: Record<ModelEntry['state'], string> = {
    running: 'bg-emerald-500/15 text-emerald-300',
    loading: 'bg-amber-500/15 text-amber-300',
    crashed: 'bg-rose-500/15 text-rose-300',
    stopped: 'bg-bg-surface-hover text-ink-dim'
  }
  return <span className={`rounded-full px-2 py-0.5 text-xs ${styles[state]}`}>{state}</span>
}

/** Gear-icon modal: Models (start/stop registered GGUFs) and Characters
 * (read-only roster). Full profile editing is explicitly out of scope for
 * this phase per the brief. */
function SettingsPanel({ base, onClose }: SettingsPanelProps): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('models')
  const [models, setModels] = useState<ModelEntry[]>([])
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refreshModels = async (): Promise<void> => {
    try {
      setModels(await listModels(base))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    refreshModels()
    listProfiles(base)
      .then((d) => setProfiles(d.profiles))
      .catch((e) => setError((e as Error).message))
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
      await refreshModels()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/50 p-6" onClick={onClose}>
      <div
        className="flex h-[560px] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-border bg-bg-surface shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <h2 className="text-base font-semibold text-ink">Settings</h2>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
          >
            ✕
          </button>
        </div>

        <div className="flex gap-1 border-b border-border px-5 pt-3">
          {(['models', 'characters'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-t-lg px-3 py-2 text-sm capitalize transition ${
                tab === t ? 'border-b-2 border-accent text-ink' : 'text-ink-dim hover:text-ink'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <div className="mb-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
              {error}
            </div>
          )}

          {tab === 'models' && (
            <div className="flex flex-col gap-2">
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
          )}

          {tab === 'characters' && (
            <div className="flex flex-col gap-2">
              {profiles.length === 0 && <p className="text-sm text-ink-dim">No characters yet.</p>}
              {profiles.map((p) => (
                <div
                  key={p.name}
                  className="flex items-center gap-3 rounded-xl border border-border bg-bg-app px-4 py-3"
                >
                  <Avatar base={base} character={p.name} size={32} />
                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-ink">{p.name}</div>
                    <div className="truncate text-xs text-ink-dim">
                      {p.model ? `model: ${p.model}` : 'default engine'}
                      {p.has_email ? ' · email configured' : ''}
                      {p.discord_enabled ? ' · discord enabled' : ''}
                    </div>
                  </div>
                </div>
              ))}
              <p className="mt-2 text-xs text-ink-dim">
                Read-only for now — full character editing (email, vault, Discord) is a later phase.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel
