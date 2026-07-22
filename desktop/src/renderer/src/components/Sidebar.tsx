import type { ChatSummary, Profile } from '../lib/types'
import { relativeTime } from '../lib/format'
import Avatar from './Avatar'

export type Section = 'chats' | 'foundry'

interface SidebarProps {
  base: string
  section: Section
  onSectionChange: (section: Section) => void
  profiles: Profile[]
  activeCharacter: string
  onSelectCharacter: (name: string) => void
  chats: ChatSummary[]
  activeChatId: string | null
  onSelectChat: (id: string) => void
  onNewChat: () => void
  onDeleteChat: (id: string) => void
  onOpenSettings: () => void
}

/** Real Claude Desktop's sidebar (measured live via its accessibility
 * tree): ~300px wide, a two-way segmented mode switch at the very top
 * (their "Home"/"Code" -> our "Chats"/"Model Foundry"), tight ~28-32px
 * list-item rows, and a dim uppercase section header above the chat
 * list. This mirrors that density instead of the previous ad-hoc
 * spacing. */
function Sidebar({
  base,
  section,
  onSectionChange,
  profiles,
  activeCharacter,
  onSelectCharacter,
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onOpenSettings
}: SidebarProps): React.JSX.Element {
  return (
    <aside className="flex h-full w-[300px] flex-none flex-col bg-bg-sidebar">
      <div className="p-2.5 pb-2">
        <div className="flex gap-0.5 rounded-lg bg-bg-000/40 p-0.5">
          <button
            onClick={() => onSectionChange('chats')}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition ${
              section === 'chats' ? 'bg-bg-400 text-ink-bright' : 'text-ink-dim hover:text-ink'
            }`}
          >
            Chats
          </button>
          <button
            onClick={() => onSectionChange('foundry')}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition ${
              section === 'foundry' ? 'bg-bg-400 text-ink-bright' : 'text-ink-dim hover:text-ink'
            }`}
          >
            Model Foundry
          </button>
        </div>
      </div>

      {section === 'chats' ? (
        <>
          <div className="px-2.5 pb-2">
            <button
              onClick={onNewChat}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-accent-hover"
            >
              <span className="text-base leading-none">+</span> New chat
            </button>
          </div>

          {profiles.length > 0 && (
            <div className="px-2.5 pb-2">
              <div className="mb-1 px-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
                Character
              </div>
              <div className="flex flex-col gap-0.5">
                {profiles.map((p) => (
                  <button
                    key={p.name}
                    onClick={() => onSelectCharacter(p.name)}
                    className={`flex items-center gap-2 rounded-md px-2 py-1 text-left text-sm transition ${
                      activeCharacter === p.name
                        ? 'bg-bg-300 text-ink-bright'
                        : 'text-ink-dim hover:bg-bg-300/60 hover:text-ink'
                    }`}
                  >
                    <Avatar base={base} character={p.name} size={20} />
                    <span className="truncate">{p.name}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto px-2.5 pb-2">
            <div className="mb-1 px-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
              Chats
            </div>
            <div className="flex flex-col gap-0.5">
              {chats.map((c) => (
                <div
                  key={c.id}
                  className={`group flex items-center gap-1 rounded-md pr-1 transition ${
                    activeChatId === c.id ? 'bg-bg-300' : 'hover:bg-bg-300/60'
                  }`}
                >
                  <button onClick={() => onSelectChat(c.id)} className="min-w-0 flex-1 px-2 py-1.5 text-left">
                    <div className="truncate text-sm text-ink">{c.title || 'New chat'}</div>
                    <div className="truncate text-xs text-ink-faint">
                      {c.character} &middot; {relativeTime(c.updated)}
                    </div>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteChat(c.id)
                    }}
                    title="Delete chat"
                    className="flex-none rounded px-1.5 py-1 text-xs text-ink-faint opacity-0 transition hover:text-rose-300 group-hover:opacity-100"
                  >
                    &#10005;
                  </button>
                </div>
              ))}
              {chats.length === 0 && <div className="px-2 py-2 text-xs text-ink-faint">No chats yet.</div>}
            </div>
          </div>
        </>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto px-2.5 pb-2 pt-1">
          <div className="mb-1 px-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
            Model Foundry
          </div>
          <div className="px-2 py-2 text-xs text-ink-faint">
            Search Hugging Face, plan quantization, download, and manage models in the main panel.
          </div>
        </div>
      )}

      <div className="border-t border-border p-2">
        <button
          onClick={onOpenSettings}
          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-ink-dim transition hover:bg-bg-300 hover:text-ink"
        >
          <span aria-hidden>&#9881;</span> Settings
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
