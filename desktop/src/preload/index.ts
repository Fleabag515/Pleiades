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
  }> => ipcRenderer.invoke('pleiades:get-backend-status')
}

contextBridge.exposeInMainWorld('pleiades', pleiadesApi)

export type PleiadesApi = typeof pleiadesApi
