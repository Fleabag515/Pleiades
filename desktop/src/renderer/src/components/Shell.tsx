import { useCallback, useEffect, useState } from 'react'
import { createChat, deleteChat, getChat, listChats, listProfiles } from '../lib/api'
import type { ChatDetail, ChatSummary, Profile } from '../lib/types'
import ChatView from './ChatView'
import ModelFoundryView from './ModelFoundryView'
import Sidebar, { type Section } from './Sidebar'
import SettingsPanel from './SettingsPanel'

interface ShellProps {
  base: string
}

/** Top-level authenticated app shell: sidebar + main pane + settings
 * modal, all backed by the live pleiades.webui REST API at `base`.
 * Mirrors Claude Desktop's shape: a top-level section switch in the
 * sidebar (their Home/Code -> our Chats/Model Foundry) alongside the
 * chat UI, rather than the previous single chat-only view. */
function Shell({ base }: ShellProps): React.JSX.Element {
  const [section, setSection] = useState<Section>('chats')
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [activeCharacter, setActiveCharacter] = useState('')
  const [activeChat, setActiveChat] = useState<ChatDetail | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshChats = useCallback(async () => {
    try {
      setChats(await listChats(base))
    } catch (e) {
      setError((e as Error).message)
    }
  }, [base])

  useEffect(() => {
    listProfiles(base)
      .then((d) => {
        setProfiles(d.profiles)
        if (d.profiles.length > 0) setActiveCharacter((c) => c || d.profiles[0].name)
      })
      .catch((e) => setError((e as Error).message))
    refreshChats()
  }, [base, refreshChats])

  const openChat = async (id: string): Promise<void> => {
    try {
      const detail = await getChat(base, id)
      setActiveChat(detail)
      setActiveCharacter(detail.character)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const newChat = async (): Promise<void> => {
    if (!activeCharacter) {
      setError('Create a character first (no profiles found).')
      return
    }
    try {
      const chat = await createChat(base, activeCharacter)
      setActiveChat(chat)
      await refreshChats()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const removeChat = async (id: string): Promise<void> => {
    try {
      await deleteChat(base, id)
      if (activeChat?.id === id) setActiveChat(null)
      await refreshChats()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-app text-ink">
      <Sidebar
        base={base}
        section={section}
        onSectionChange={setSection}
        profiles={profiles}
        activeCharacter={activeCharacter}
        onSelectCharacter={setActiveCharacter}
        chats={chats}
        activeChatId={activeChat?.id ?? null}
        onSelectChat={openChat}
        onNewChat={newChat}
        onDeleteChat={removeChat}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        {error && (
          <div className="flex-none border-b border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-300">
            {error}
          </div>
        )}
        {section === 'chats' ? (
          <ChatView
            base={base}
            chat={activeChat}
            activeCharacter={activeCharacter}
            onNewChat={newChat}
            onChatChanged={refreshChats}
          />
        ) : (
          <ModelFoundryView base={base} />
        )}
      </div>
      {settingsOpen && <SettingsPanel base={base} onClose={() => setSettingsOpen(false)} />}
    </div>
  )
}

export default Shell
