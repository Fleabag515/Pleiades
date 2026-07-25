import { useEffect, useState } from 'react'

// Mirrors main/index.ts's UpdateStatus / preload's PleiadesUpdateStatus.
// Redeclared locally rather than imported, matching how App.tsx's
// BackendStatus mirrors PleiadesBackendStatus -- this codebase's existing
// convention for renderer-side copies of preload-exposed shapes.
interface UpdateStatus {
  phase: 'idle' | 'checking' | 'available' | 'downloading' | 'ready' | 'not-available' | 'error'
  version: string | null
  progressPercent: number | null
  error: string | null
}

const PHASE_LABEL: Record<UpdateStatus['phase'], string> = {
  idle: 'No check run yet this session.',
  checking: 'Checking for updates…',
  available: 'Update found — downloading…',
  downloading: 'Downloading update…',
  ready: 'Update downloaded — ready to install.',
  'not-available': "You're on the latest version.",
  error: 'Update check failed.'
}

const PHASE_PILL: Record<UpdateStatus['phase'], string> = {
  idle: 'bg-bg-surface-hover text-ink-dim',
  checking: 'bg-amber-500/15 text-amber-300',
  available: 'bg-amber-500/15 text-amber-300',
  downloading: 'bg-amber-500/15 text-amber-300',
  ready: 'bg-emerald-500/15 text-emerald-300',
  'not-available': 'bg-emerald-500/15 text-emerald-300',
  error: 'bg-rose-500/15 text-rose-300'
}

/** Updates tab: current app version, live auto-update status (pushed from
 * main/index.ts's setupAutoUpdater via 'pleiades:update-status-changed'),
 * a manual re-check button, and — only once a download has actually
 * finished ('ready') — the one-click install+restart button. Packaged
 * builds already auto-check ~10s after launch and auto-download on found;
 * this tab exists for visibility and manual control, not to drive the
 * whole flow by hand. In dev (unpackaged) builds electron-updater has
 * nothing to check against, so checkForUpdates() will typically resolve to
 * an 'error' or 'not-available' phase -- expected, not a bug. */
function UpdatesSection(): React.JSX.Element {
  const [status, setStatus] = useState<UpdateStatus>({
    phase: 'idle',
    version: null,
    progressPercent: null,
    error: null
  })
  const [appVersion, setAppVersion] = useState<string | null>(null)

  useEffect(() => {
    window.pleiades.getUpdateStatus().then(setStatus)
    return window.pleiades.onUpdateStatusChanged(setStatus)
  }, [])

  useEffect(() => {
    window.pleiades.getAppVersion().then(setAppVersion)
  }, [])

  const busy = status.phase === 'checking' || status.phase === 'downloading'

  return (
    <div className="flex flex-col gap-4">
      <section className="rounded-xl border border-border bg-bg-app p-4">
        <h3 className="mb-3 text-sm font-semibold text-ink">Pleiades desktop</h3>
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-ink-dim">
              {appVersion ? `Version ${appVersion}` : 'Loading version…'}
            </span>
            <div className="flex items-center gap-2">
              <span className={`rounded-full px-2 py-0.5 text-[11px] ${PHASE_PILL[status.phase]}`}>
                {PHASE_LABEL[status.phase]}
                {status.phase === 'downloading' && status.progressPercent != null
                  ? ` ${status.progressPercent}%`
                  : ''}
              </span>
              {(status.phase === 'available' || status.phase === 'ready') && status.version && (
                <span className="text-[11px] text-ink-dim">v{status.version}</span>
              )}
            </div>
            {status.phase === 'error' && status.error && (
              <span className="text-[11px] text-rose-300/80">{status.error}</span>
            )}
          </div>

          <div className="flex flex-none items-center gap-2">
            {status.phase === 'ready' ? (
              <button
                onClick={() => window.pleiades.installUpdateAndRestart()}
                className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition hover:opacity-90"
              >
                Install &amp; restart
              </button>
            ) : (
              <button
                onClick={() => {
                  setStatus((s) => ({ ...s, phase: 'checking', error: null }))
                  window.pleiades.checkForUpdates()
                }}
                disabled={busy}
                className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
              >
                Check for updates
              </button>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}

export default UpdatesSection
