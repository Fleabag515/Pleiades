import { useEffect, useState } from 'react'
import { listProfiles } from '../lib/api'
import type { Profile } from '../lib/types'
import ModelsTab from './settings/ModelsTab'
import CharactersTab from './settings/CharactersTab'
import HardwareTab from './settings/HardwareTab'

interface SettingsPanelProps {
  base: string
  onClose: () => void
}

type Tab = 'models' | 'characters' | 'hardware'

const TABS: Tab[] = ['models', 'characters', 'hardware']

/** Gear-icon modal: Models (start/stop registered GGUFs), Characters (full
 * per-character email/Discord/model/vault editing), and Hardware (this
 * machine's detected GPU/RAM/CPU + per-model offload plans). */
function SettingsPanel({ base, onClose }: SettingsPanelProps): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('models')
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listProfiles(base)
      .then((d) => setProfiles(d.profiles))
      .catch((e) => setError((e as Error).message))
  }, [base])

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/50 p-6" onClick={onClose}>
      <div
        className="flex h-[680px] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-border bg-bg-surface shadow-2xl"
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
          {TABS.map((t) => (
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

          {tab === 'models' && <ModelsTab base={base} />}
          {tab === 'characters' && <CharactersTab base={base} profiles={profiles} />}
          {tab === 'hardware' && <HardwareTab base={base} />}
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel
