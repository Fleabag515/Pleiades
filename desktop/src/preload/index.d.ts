export interface PleiadesBackendStatus {
  phase: 'starting' | 'ready' | 'error'
  port: number | null
  url: string | null
  error: string | null
}

/** Mirrors main/index.ts's UpdateStatus. 'available'/'downloading' only occur
 * when autoDownload has kicked in; with autoDownload:true (current setting)
 * the flow goes checking -> available -> downloading -> ready, and 'ready'
 * is the only phase where installUpdateAndRestart() should be offered. */
export interface PleiadesUpdateStatus {
  phase: 'idle' | 'checking' | 'available' | 'downloading' | 'ready' | 'not-available' | 'error'
  version: string | null
  progressPercent: number | null
  error: string | null
}

export interface PleiadesApi {
  getBackendUrl: () => Promise<string | null>
  getBackendStatus: () => Promise<PleiadesBackendStatus>
  windowMinimize: () => Promise<void>
  windowMaximizeToggle: () => Promise<void>
  windowClose: () => Promise<void>
  windowIsMaximized: () => Promise<boolean>
  onWindowMaximizedChanged: (cb: (maximized: boolean) => void) => () => void
  getAppVersion: () => Promise<string>
  isSelfUpdatingInstall: () => Promise<boolean>
  checkForUpdates: () => Promise<void>
  getUpdateStatus: () => Promise<PleiadesUpdateStatus>
  installUpdateAndRestart: () => Promise<void>
  onUpdateStatusChanged: (cb: (status: PleiadesUpdateStatus) => void) => () => void
}

declare global {
  interface Window {
    pleiades: PleiadesApi
  }
}
