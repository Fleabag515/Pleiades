import { useEffect, useState } from 'react'
import { listProfiles } from '../lib/api'
import type { Profile } from '../lib/types'
import ModelsSection from './settings/ModelsSection'
import CharactersTab from './settings/CharactersTab'
import HardwareTab from './settings/HardwareTab'
import UpdatesSection from './settings/UpdatesSection'

interface SettingsPanelProps {
  base: string
  onClose: () => void
}

type Section = 'characters' | 'hardware' | 'models' | 'updates'

const SECTIONS: { key: Section; label: string }[] = [
  { key: 'characters', label: 'Characters' },
  { key: 'hardware', label: 'Hardware' },
  { key: 'models', label: 'Models' },
  { key: 'updates', label: 'Updates' }
]

/** Gear-icon modal, restructured onto a left vertical nav (replacing the old
 * top tab bar): Characters (full per-character email/Discord/model/vault
 * editing), Hardware (this machine's detected GPU/RAM/CPU + per-model
 * offload plans), and Models — which now owns its own internal top tab bar
 * (Models / Download / Ollama Cloud / OpenRouter, see ModelsSection) rather
 * than being a single flat list. The standalone "Foundry" top-level tab from
 * the previous phase is retired: its HF-search/download/cloud-browsing
 * content fully relocated into Models' Download/Ollama Cloud/OpenRouter
 * tabs, so there is no longer a separate Foundry entry point here. */
function SettingsPanel({ base, onClose }: SettingsPanelProps): React.JSX.Element {
  const [section, setSection] = useState<Section>('characters')
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [error, setError] = useState<string | null>(null)
  const [updateReady, setUpdateReady] = useState(false)

  useEffect(() => {
    listProfiles(base)
      .then((d) => setProfiles(d.profiles))
      .catch((e) => setError((e as Error).message))
  }, [base])

  useEffect(() => {
    // Small nav-item dot only -- UpdatesSection itself owns the full status
    // detail/actions. 'available'/'downloading' means autoUpdater found one
    // and is still fetching it; 'ready' means it's fully downloaded and
    // installUpdateAndRestart() will work right now.
    window.pleiades
      .getUpdateStatus()
      .then((s) =>
        setUpdateReady(s.phase === 'available' || s.phase === 'downloading' || s.phase === 'ready')
      )
    return window.pleiades.onUpdateStatusChanged((s) =>
      setUpdateReady(s.phase === 'available' || s.phase === 'downloading' || s.phase === 'ready')
    )
  }, [])

  return (
    <div
      className="fixed inset-0 z-20 flex items-center justify-center bg-black/50 p-6"
      onClick={onClose}
    >
      <div
        className="flex h-[680px] max-h-[calc(100vh-3rem)] w-full max-w-4xl flex-col overflow-hidden rounded-3xl border border-border bg-bg-100 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <h2 className="text-base font-semibold text-ink-bright">Settings</h2>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
          >
            ✕
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
          <nav className="flex w-44 flex-none flex-col gap-0.5 border-r border-border p-3">
            {SECTIONS.map((s) => (
              <button
                key={s.key}
                onClick={() => setSection(s.key)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-left text-sm font-medium transition ${
                  section === s.key
                    ? 'bg-bg-300 text-ink-bright'
                    : 'text-ink-dim hover:bg-bg-surface-hover hover:text-ink'
                }`}
              >
                {s.label}
                {s.key === 'updates' && updateReady && (
                  <span className="h-1.5 w-1.5 flex-none rounded-full bg-accent" aria-hidden />
                )}
              </button>
            ))}
          </nav>

          <div className="flex-1 overflow-y-auto p-5">
            {error && (
              <div className="mb-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
                {error}
              </div>
            )}

            {section === 'characters' && <CharactersTab base={base} profiles={profiles} />}
            {section === 'hardware' && <HardwareTab base={base} />}
            {section === 'models' && <ModelsSection base={base} />}
            {section === 'updates' && <UpdatesSection />}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel
