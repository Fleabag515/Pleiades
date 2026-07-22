export interface PleiadesBackendStatus {
  phase: 'starting' | 'ready' | 'error'
  port: number | null
  url: string | null
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
}

declare global {
  interface Window {
    pleiades: PleiadesApi
  }
}
