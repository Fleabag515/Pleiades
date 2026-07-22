export interface PleiadesBackendStatus {
  phase: 'starting' | 'ready' | 'error'
  port: number | null
  url: string | null
  error: string | null
}

export interface PleiadesApi {
  getBackendUrl: () => Promise<string | null>
  getBackendStatus: () => Promise<PleiadesBackendStatus>
}

declare global {
  interface Window {
    pleiades: PleiadesApi
  }
}
