import { useCallback, useEffect, useState } from 'react'
import { listProfiles } from '../lib/api'
import { useChatSession, useChatStoreActions } from '../lib/chatStore'
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
 * mode-switch for Local Models/Foundry (folded into Settings; Settings
 * itself now has a left-nav Characters/Hardware/Models layout, with Models
 * owning its own Models/Download/Ollama Cloud/OpenRouter internal tabs —
 * see SettingsPanel and ModelsSection).
 *
 * Owner feedback round 4: the right-panel toggle that used to float at
 * bottom-4 right-4 has moved up into TopCharacterBar (top right of the chat
 * header). Its old bottom-right spot is now a "New chat" bubble instead —
 * same size/position/styling as the Settings cogwheel it mirrors, so the
 * bottom row still reads as a matched pair. That bubble sits *outside* the
 * `[chat column][RightPanel]` flex row (it's a sibling, absolutely
 * positioned over the whole Shell), so it doesn't get reflowed for free the
 * way ChatView's chat column does when RightPanel's width animates — it's
 * pushed left "by hand" with its own `transition-transform duration-300
 * ease-in-out` + `-translate-x-96` keyed off the same `rightPanelOpen`
 * boolean, matching RightPanel's own `transition-[width] duration-300
 * ease-in-out` timing/easing exactly (see RightPanel.tsx) so the two motions
 * read as one push rather than two independently-timed animations.
 */
function Shell({ base }: ShellProps): React.JSX.Element {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [activeCharacter, setActiveCharacter] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [rightPanelOpen, setRightPanelOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [updateReady, setUpdateReady] = useState(false)
  const { startNewChat } = useChatStoreActions()
  const activeSession = useChatSession(activeCharacter)

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

  useEffect(() => {
    // Gear-button badge: same three "there's something to look at in
    // Settings > Updates" phases SettingsPanel's own nav-dot uses.
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
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-bg-app text-ink">
      <TopCharacterBar
        base={base}
        profiles={profiles}
        activeCharacter={activeCharacter}
        onSelectCharacter={setActiveCharacter}
        onOpenCreate={() => setCreateOpen(true)}
        rightPanelOpen={rightPanelOpen}
        onToggleRightPanel={() => setRightPanelOpen((v) => !v)}
      />

      {error && (
        <div className="flex-none border-b border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="min-h-0 flex-1">
        <ChatView base={base} character={activeCharacter} rightPanelOpen={rightPanelOpen} />
      </div>

      <button
        onClick={() => setSettingsOpen(true)}
        title={updateReady ? 'Settings — update ready' : 'Settings'}
        aria-label="Settings"
        className="absolute bottom-4 left-4 z-10 flex h-11 w-11 items-center justify-center rounded-full border border-border bg-bg-200 text-lg text-ink-dim shadow-lg transition hover:bg-bg-300 hover:text-ink"
      >
        <span aria-hidden>&#9881;</span>
        {updateReady && (
          <span
            className="absolute right-0.5 top-0.5 h-2.5 w-2.5 rounded-full border-2 border-bg-200 bg-accent"
            aria-hidden
          />
        )}
      </button>

      {/* Owner feedback round 4: the right-panel toggle that used to live
        here moved up into TopCharacterBar's top-right corner. This bubble
        takes over its old bottom-right spot instead: a "New chat" shortcut,
        same size/shape as the Settings cogwheel it mirrors. It's pushed left
        by exactly RightPanel's open width (`w-96` / 24rem, so `-translate-x-96`)
        whenever the panel is open, animating with the identical
        duration-300/ease-in-out timing RightPanel uses for its own
        `transition-[width]` (see RightPanel.tsx) -- as if this bubble were
        one more element of the chat pane getting shoved over by the panel
        sliding in, and it slides back on close the same way. Disabled while
        the active character has no name yet or a turn is mid-stream, same
        guard Composer's inline "New chat" control uses. */}
      <button
        onClick={() => {
          if (!activeCharacter || activeSession.streaming) return
          if (
            window.confirm(
              `Start a new chat with ${activeCharacter}? This conversation moves to History — nothing is deleted.`
            )
          ) {
            startNewChat(base, activeCharacter)
          }
        }}
        disabled={!activeCharacter || activeSession.streaming}
        title="Start a new chat — this conversation moves to History"
        aria-label="Start a new chat"
        className={`absolute bottom-4 right-4 z-10 flex h-11 w-11 items-center justify-center rounded-full border border-border bg-bg-200 text-lg text-ink-dim shadow-lg transition-transform duration-300 ease-in-out hover:bg-bg-300 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40 ${
          rightPanelOpen ? '-translate-x-96' : 'translate-x-0'
        }`}
      >
        <span aria-hidden>&#10133;</span>
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
