interface EmptyStateProps {
  characterName: string
  onNewChat: () => void
}

/** Claude-Desktop-style centered prompt shown when no chat is open. */
function EmptyState({ characterName, onNewChat }: EmptyStateProps): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/15 text-2xl text-accent">
        ✳
      </div>
      <h1 className="text-xl font-semibold text-ink-bright">Start a conversation</h1>
      <p className="max-w-sm text-sm text-ink-dim">
        {characterName
          ? `Say hello to ${characterName}, or pick a different character in the sidebar first.`
          : 'Pick a character in the sidebar, then start a new chat.'}
      </p>
      <button
        onClick={onNewChat}
        className="mt-2 rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover"
      >
        New chat
      </button>
    </div>
  )
}

export default EmptyState
