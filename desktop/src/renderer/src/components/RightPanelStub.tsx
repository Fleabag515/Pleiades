interface RightPanelStubProps {
  onClose: () => void
}

/**
 * Reserved slot for the future right-side panel (history/progress-list/
 * browser/scheduled-tasks) — building it out is explicitly a separate
 * phase's job. This phase only needs an obvious toggle + a docked shell so
 * the affordance exists (see the bottom-right icon button in ChatView).
 */
function RightPanelStub({ onClose }: RightPanelStubProps): React.JSX.Element {
  return (
    <div className="absolute inset-y-0 right-0 z-20 flex w-80 flex-none flex-col border-l border-border bg-bg-100 shadow-2xl">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <span className="text-sm font-medium text-ink-bright">Panel</span>
        <button
          onClick={onClose}
          className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
        >
          ✕
        </button>
      </div>
      <div className="flex flex-1 items-center justify-center p-6 text-center text-sm text-ink-faint">
        History, progress list, browser, and scheduled tasks land here in a future phase.
      </div>
    </div>
  )
}

export default RightPanelStub
