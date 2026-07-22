import { useEffect, useState } from 'react'

/**
 * Minimal custom title bar for the frameless main window (see main/index.ts's
 * `frame: false`). Kept deliberately small: a draggable strip with the app
 * name plus three native-feeling window controls wired through the preload
 * bridge to the real BrowserWindow (minimize / maximize-toggle / close).
 * The close button goes through the exact same IPC-triggered
 * `mainWindow.close()` as a native close button would, so the existing
 * minimize-to-tray 'close' handler in main/index.ts applies unchanged.
 *
 * Height (36px/h-9) and button width (32px/w-8) match the real Claude
 * Desktop title bar's measured proportions (its own custom window
 * controls come out to ~32x34px, tightly packed with no gaps).
 */
function TitleBar(): React.JSX.Element {
  const [maximized, setMaximized] = useState(false)

  useEffect(() => {
    let cancelled = false
    window.pleiades
      .windowIsMaximized()
      .then((v) => {
        if (!cancelled) setMaximized(v)
      })
      .catch(() => {})
    const unsubscribe = window.pleiades.onWindowMaximizedChanged((v) => setMaximized(v))
    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  return (
    <div
      className="flex h-9 flex-none select-none items-center justify-between bg-bg-000 text-ink-dim"
      style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
    >
      <div className="flex items-center gap-2 pl-3 text-xs font-medium tracking-wide">
        <span>Pleiades</span>
      </div>
      <div className="flex h-full" style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
        <button
          aria-label="Minimize"
          onClick={() => window.pleiades.windowMinimize()}
          className="flex h-full w-8 items-center justify-center transition hover:bg-bg-surface-hover"
        >
          <svg width="10" height="10" viewBox="0 0 10 10">
            <rect x="0" y="4.5" width="10" height="1" fill="currentColor" />
          </svg>
        </button>
        <button
          aria-label={maximized ? 'Restore' : 'Maximize'}
          onClick={() => window.pleiades.windowMaximizeToggle()}
          className="flex h-full w-8 items-center justify-center transition hover:bg-bg-surface-hover"
        >
          {maximized ? (
            <svg width="10" height="10" viewBox="0 0 10 10">
              <rect
                x="1.5"
                y="0.5"
                width="7"
                height="7"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
              <rect x="0.5" y="2.5" width="7" height="7" fill="var(--color-bg-000)" />
              <rect
                x="0.5"
                y="2.5"
                width="7"
                height="7"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 10 10">
              <rect
                x="0.5"
                y="0.5"
                width="9"
                height="9"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
            </svg>
          )}
        </button>
        <button
          aria-label="Close"
          onClick={() => window.pleiades.windowClose()}
          className="flex h-full w-8 items-center justify-center transition hover:bg-rose-600 hover:text-white"
        >
          <svg width="10" height="10" viewBox="0 0 10 10">
            <line x1="0.5" y1="0.5" x2="9.5" y2="9.5" stroke="currentColor" strokeWidth="1" />
            <line x1="9.5" y1="0.5" x2="0.5" y2="9.5" stroke="currentColor" strokeWidth="1" />
          </svg>
        </button>
      </div>
    </div>
  )
}

export default TitleBar
