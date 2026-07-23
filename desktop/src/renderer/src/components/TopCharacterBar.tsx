import type { Profile } from '../lib/types'
import Avatar from './Avatar'

interface TopCharacterBarProps {
  base: string
  profiles: Profile[]
  activeCharacter: string
  onSelectCharacter: (name: string) => void
  onOpenCreate: () => void
  rightPanelOpen: boolean
  onToggleRightPanel: () => void
}

/**
 * Replaces the old left sidebar entirely (owner brief item 1): a horizontal
 * row of oval character bubbles, each auto-sized to its name (plain
 * `inline`/`flex` sizing, no fixed width class), plus a circular "+" bubble
 * at the end that opens character creation. Clicking a bubble switches the
 * active character, which (via ChatSessionsProvider) always shows that
 * character's one persistent live chat.
 *
 * Owner feedback round 2: no border/divider under this bar anymore -- the
 * chat content below (see ChatView's `.scroll-fade-top`) now fades out as it
 * scrolls underneath instead, so the boundary reads as a soft mask, not a
 * hard line.
 *
 * Owner feedback round 4: the right-panel open/close toggle moves up here
 * (top right of the chat header) instead of floating at the bottom-right
 * corner -- see Shell.tsx, which now puts a "New chat" bubble in that old
 * bottom-right spot instead. The bubble row keeps its own `overflow-x-auto`
 * scroller (wrapped in a `flex-1 min-w-0` div) so a long character list
 * still scrolls independently, while this toggle sits outside that scroller
 * as a `flex-none` sibling and stays pinned flush right instead of
 * scrolling away with the bubbles.
 */
function TopCharacterBar({
  base,
  profiles,
  activeCharacter,
  onSelectCharacter,
  onOpenCreate,
  rightPanelOpen,
  onToggleRightPanel
}: TopCharacterBarProps): React.JSX.Element {
  return (
    <div className="flex flex-none items-center gap-2 bg-bg-app px-4 py-3">
      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto">
        {profiles.map((p) => (
          <button
            key={p.name}
            onClick={() => onSelectCharacter(p.name)}
            className={`flex flex-none items-center gap-2 whitespace-nowrap rounded-full px-3.5 py-1.5 text-sm font-medium transition ${
              activeCharacter === p.name
                ? 'bg-bg-400 text-ink-bright'
                : 'bg-bg-200 text-ink-dim hover:bg-bg-300 hover:text-ink'
            }`}
          >
            <Avatar base={base} character={p.name} size={20} />
            {p.name}
          </button>
        ))}
        <button
          onClick={onOpenCreate}
          title="New character"
          aria-label="New character"
          className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-bg-200 text-lg leading-none text-ink-dim transition hover:bg-accent hover:text-white"
        >
          +
        </button>
        {profiles.length === 0 && (
          <span className="px-1 text-xs text-ink-faint">
            No characters yet — create one to get started.
          </span>
        )}
      </div>

      <button
        onClick={onToggleRightPanel}
        title={rightPanelOpen ? 'Close panel' : 'Progress, history, browser & scheduled tasks'}
        aria-label="Toggle right panel"
        className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-bg-200 text-lg text-ink-dim transition hover:bg-bg-300 hover:text-ink"
      >
        <span aria-hidden>&#9776;</span>
      </button>
    </div>
  )
}

export default TopCharacterBar
