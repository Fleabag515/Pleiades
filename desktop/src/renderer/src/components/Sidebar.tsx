import type { ChatSummary, Profile } from '../lib/types'
import { relativeTime } from '../lib/format'
import Avatar from './Avatar'

interface SidebarProps {
  base: string
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

function Sidebar({
  base,
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
    <aside className="flex h-full w-[270px] flex-none flex-col bg-bg-sidebar">
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-accent px-3 py-2.5 text-sm font-medium text-white transition hover:bg-accent-hover"
        >
          <span className="text-base leading-none">+</span> New chat
        </button>
      </div>

      {profiles.length > 0 && (
        <div className="px-3 pb-2">
          <div className="mb-1.5 px-1 text-xs font-medium uppercase tracking-wide text-ink-dim">
            Character
          </div>
          <div className="flex flex-col gap-0.5">
            {profiles.map((p) => (
              <button
                key={p.name}
                onClick={() => onSelectCharacter(p.name)}
                className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm transition ${
                  activeCharacter === p.name
                    ? 'bg-bg-surface-hover text-ink'
                    : 'text-ink-dim hover:bg-bg-surface-hover/60 hover:text-ink'
                }`}
              >
                <Avatar base={base} character={p.name} size={22} />
                <span className="truncate">{p.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-2">
        <div className="mb-1.5 px-1 text-xs font-medium uppercase tracking-wide text-ink-dim">Chats</div>
        <div className="flex flex-col gap-0.5">
          {chats.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center gap-1 rounded-lg pr-1 transition ${
                activeChatId === c.id ? 'bg-bg-surface-hover' : 'hover:bg-bg-surface-hover/60'
              }`}
            >
              <button onClick={() => onSelectChat(c.id)} className="min-w-0 flex-1 px-2 py-2 text-left">
                <div className="truncate text-sm text-ink">{c.title || 'New chat'}</div>
                <div className="truncate text-xs text-ink-dim">
                  {c.character} · {relativeTime(c.updated)}
                </div>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onDeleteChat(c.id)
                }}
                title="Delete chat"
                className="flex-none rounded px-1.5 py-1 text-xs text-ink-dim opacity-0 transition hover:text-rose-300 group-hover:opacity-100"
              >
                ✕
              </button>
            </div>
          ))}
          {chats.length === 0 && <div className="px-2 py-2 text-xs text-ink-dim">No chats yet.</div>}
        </div>
      </div>

      <div className="border-t border-border p-2">
        <button
          onClick={onOpenSettings}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
        >
          <span aria-hidden>⚙</span> Settings
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
