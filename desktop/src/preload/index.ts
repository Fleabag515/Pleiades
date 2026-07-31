import { contextBridge, ipcRenderer } from 'electron'

interface UpdateStatus {
  phase: 'idle' | 'checking' | 'available' | 'downloading' | 'ready' | 'not-available' | 'error'
  version: string | null
  progressPercent: number | null
  error: string | null
}

/**
 * Minimal, explicit bridge for the renderer. Intentionally does NOT expose
 * ipcRenderer, require, process, or any other Node/Electron primitive —
 * only the specific calls the renderer needs.
 */
const pleiadesApi = {
  /** Base URL of the local Pleiades backend, e.g. http://127.0.0.1:8750 */
  getBackendUrl: (): Promise<string | null> => ipcRenderer.invoke('pleiades:get-backend-url'),

  /** Current lifecycle state of the backend child process. */
  getBackendStatus: (): Promise<{
    phase: 'starting' | 'ready' | 'error'
    port: number | null
    url: string | null
    error: string | null
  }> => ipcRenderer.invoke('pleiades:get-backend-status'),

  // Custom title bar window controls (frame: false main window).
  windowMinimize: (): Promise<void> => ipcRenderer.invoke('pleiades:window-minimize'),
  windowMaximizeToggle: (): Promise<void> => ipcRenderer.invoke('pleiades:window-maximize-toggle'),
  windowClose: (): Promise<void> => ipcRenderer.invoke('pleiades:window-close'),
  windowIsMaximized: (): Promise<boolean> => ipcRenderer.invoke('pleiades:window-is-maximized'),
  onWindowMaximizedChanged: (cb: (maximized: boolean) => void): (() => void) => {
    const listener = (_event: unknown, maximized: boolean): void => cb(maximized)
    ipcRenderer.on('pleiades:window-maximized-changed', listener)
    return () => ipcRenderer.removeListener('pleiades:window-maximized-changed', listener)
  },

  getAppVersion: (): Promise<string> => ipcRenderer.invoke('pleiades:get-app-version'),
  /** True only for a Linux AppImage install -- the one case where
   * "Install & restart" replaces the running app in place and relaunches
   * automatically, with no separate installer step. See main/index.ts's
   * install-update-and-restart handler. */
  isSelfUpdatingInstall: (): Promise<boolean> =>
    ipcRenderer.invoke('pleiades:is-self-updating-install'),

  // Auto-update (see main/index.ts's "Auto-update" section for the full flow).
  checkForUpdates: (): Promise<void> => ipcRenderer.invoke('pleiades:check-for-updates'),
  getUpdateStatus: (): Promise<UpdateStatus> => ipcRenderer.invoke('pleiades:get-update-status'),
  installUpdateAndRestart: (): Promise<void> =>
    ipcRenderer.invoke('pleiades:install-update-and-restart'),
  onUpdateStatusChanged: (cb: (status: UpdateStatus) => void): (() => void) => {
    const listener = (_event: unknown, status: UpdateStatus): void => cb(status)
    ipcRenderer.on('pleiades:update-status-changed', listener)
    return () => ipcRenderer.removeListener('pleiades:update-status-changed', listener)
  }
}

contextBridge.exposeInMainWorld('pleiades', pleiadesApi)

export type PleiadesApi = typeof pleiadesApi
