import { useCallback, useEffect, useState } from 'react'
import { listProfiles } from '../lib/api'
import type { Profile } from '../lib/types'
import ChatView from './ChatView'
import CreateCharacterOverlay from './CreateCharacterOverlay'
import SettingsPanel from './SettingsPanel'
import TopCharacterBar from './TopCharacterBar'

interface ShellProps {
  base: string
}

/**
 * Top-level app shell after the sidebar-removal rebuild (owner brief items
 * 1-2): a top bar of character bubbles (+ the "+" create bubble) replaces
 * the entire old left sidebar, and a floating circular cogwheel bubble in
 * the bottom-left corner opens Settings (unchanged content, still a modal
 * overlay). The main pane below the character bar is always that
 * character's one live `ChatView` — no separate "pick a chat" list, no
 * mode-switch for Local Models/Foundry (folded into Settings, see
 * SettingsPanel's new Foundry tab).
 */
function Shell({ base }: ShellProps): React.JSX.Element {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [activeCharacter, setActiveCharacter] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshProfiles = useCallback(async () => {
    try {
      const d = await listProfiles(base)
      setProfiles(d.profiles)
      setActiveCharacter((c) => c || d.profiles[0]?.name || '')
    } catch (e) {
      setError((e as Error).message)
    }
  }, [base])

  useEffect(() => {
    refreshProfiles()
  }, [refreshProfiles])

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-bg-app text-ink">
      <TopCharacterBar
        base={base}
        profiles={profiles}
        activeCharacter={activeCharacter}
        onSelectCharacter={setActiveCharacter}
        onOpenCreate={() => setCreateOpen(true)}
      />

      {error && (
        <div className="flex-none border-b border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="min-h-0 flex-1">
        <ChatView base={base} character={activeCharacter} />
      </div>

      <button
        onClick={() => setSettingsOpen(true)}
        title="Settings"
        aria-label="Settings"
        className="absolute bottom-4 left-4 z-10 flex h-11 w-11 items-center justify-center rounded-full border border-border bg-bg-200 text-lg text-ink-dim shadow-lg transition hover:bg-bg-300 hover:text-ink"
      >
        <span aria-hidden>&#9881;</span>
      </button>

      {settingsOpen && <SettingsPanel base={base} onClose={() => setSettingsOpen(false)} />}
      {createOpen && (
        <CreateCharacterOverlay
          base={base}
          onClose={() => setCreateOpen(false)}
          onCreated={refreshProfiles}
        />
      )}
    </div>
  )
}

export default Shell
