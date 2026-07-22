import { contextBridge, ipcRenderer } from 'electron'

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
  }
}

contextBridge.exposeInMainWorld('pleiades', pleiadesApi)

export type PleiadesApi = typeof pleiadesApi
