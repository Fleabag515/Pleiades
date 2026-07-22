import { useEffect, useRef, useState } from 'react'
import { fetchModel, fetchStatus, hfSearch, quantOptions } from '../../lib/api'
import type { FetchStatus, HfSearchResult, QuantOption, QuantOptionsResponse } from '../../lib/types'
import { formatGiB } from '../../lib/format'

interface DownloadTabProps {
  base: string
}

const STRATEGY_LABEL: Record<string, string> = {
  full_gpu: 'fits fully on GPU',
  moe_cpu: 'experts → RAM, rest → GPU',
  moe_partial: 'MoE hybrid split',
  layers: 'partial offload',
  cpu: 'CPU only'
}

function useDebounced(value: string, delayMs: number): string {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])
  return debounced
}

/** HF search -> quant options -> download-with-progress. Relocated from the
 * old standalone "Foundry" top-level Settings tab into Models' internal tab
 * bar (this is now the "Download" tab) — logic is unchanged from
 * ModelFoundryView's HfSearchSection, only the location moved. The backend
 * only tracks one in-flight fetch at a time (models_fetch 409s otherwise),
 * so download progress is a single global strip fed by polling
 * GET /api/models/fetch/status rather than per-card state. */
function DownloadTab({ base }: DownloadTabProps): React.JSX.Element {
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounced(query, 400)
  const [results, setResults] = useState<HfSearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  const [selectedRepo, setSelectedRepo] = useState<string | null>(null)
  const [quantData, setQuantData] = useState<QuantOptionsResponse | null>(null)
  const [quantLoading, setQuantLoading] = useState(false)
  const [quantError, setQuantError] = useState<string | null>(null)

  const [status, setStatus] = useState<FetchStatus>({ status: 'idle' })
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([])
      return
    }
    let cancelled = false
    setSearching(true)
    setSearchError(null)
    hfSearch(base, debouncedQuery.trim(), 12)
      .then((r) => {
        if (!cancelled) setResults(r)
      })
      .catch((e) => {
        if (!cancelled) setSearchError((e as Error).message)
      })
      .finally(() => {
        if (!cancelled) setSearching(false)
      })
    return () => {
      cancelled = true
    }
  }, [base, debouncedQuery])

  // Poll fetch status continuously while this tab is mounted so a download
  // kicked off earlier still shows live progress. See ModelFoundryView's
  // retired history for why this poll must never trigger a remount of the
  // separate local-models list (it doesn't — this tab owns no such list).
  useEffect(() => {
    const tick = async (): Promise<void> => {
      try {
        const s = await fetchStatus(base)
        setStatus(s)
      } catch {
        // ignore transient poll failures
      }
    }
    tick()
    pollRef.current = setInterval(tick, 1000)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base])

  const selectRepo = async (repo: string): Promise<void> => {
    setSelectedRepo(repo)
    setQuantData(null)
    setQuantError(null)
    setQuantLoading(true)
    try {
      setQuantData(await quantOptions(base, repo))
    } catch (e) {
      setQuantError((e as Error).message)
    } finally {
      setQuantLoading(false)
    }
  }

  const download = async (opt: QuantOption): Promise<void> => {
    if (!selectedRepo) return
    try {
      await fetchModel(base, { repo: selectedRepo, quant: opt.quant })
      setStatus({ status: 'downloading', repo: selectedRepo, file: opt.file, done: 0, total: opt.size })
    } catch (e) {
      setSearchError((e as Error).message)
    }
  }

  const busy = status.status === 'downloading'
  const pct = busy && status.total ? Math.min(100, Math.round((100 * (status.done ?? 0)) / status.total)) : 0

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search Hugging Face for GGUF models (e.g. “qwen 7b”, “llama 3”)…"
          className="w-full rounded-lg border border-border bg-bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:outline-none"
        />
        {searching && <span className="flex-none text-xs text-ink-faint">searching…</span>}
      </div>

      {searchError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {searchError}
        </div>
      )}

      {busy && (
        <div className="rounded-xl border border-border bg-bg-app p-3">
          <div className="mb-1.5 flex items-center justify-between text-xs">
            <span className="truncate text-ink">
              Downloading {status.file || status.repo} {status.repo ? `(${status.repo})` : ''}
            </span>
            <span className="text-ink-dim">
              {status.total ? `${formatGiB(status.done ?? 0)} / ${formatGiB(status.total)} GiB · ${pct}%` : '…'}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-surface-hover">
            <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
      {status.status === 'error' && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          Download failed: {status.error}
        </div>
      )}
      {status.status === 'done' && status.result && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300">
          Downloaded and registered “{status.result.name}”. Find it in the Models tab.
        </div>
      )}

      {results.length > 0 && (
        <div className="flex flex-col gap-1 rounded-xl border border-border bg-bg-app p-2">
          {results.map((r) => (
            <button
              key={r.id}
              onClick={() => selectRepo(r.id)}
              className={`flex items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition ${
                selectedRepo === r.id ? 'bg-bg-300 text-ink-bright' : 'text-ink hover:bg-bg-surface-hover'
              }`}
            >
              <span className="truncate">{r.id}</span>
              <span className="flex-none text-xs text-ink-faint">
                {r.downloads.toLocaleString()} downloads · {r.likes.toLocaleString()} likes
              </span>
            </button>
          ))}
        </div>
      )}

      {selectedRepo && (
        <div className="rounded-xl border border-border bg-bg-app p-4">
          <h3 className="mb-2 text-sm font-semibold text-ink">{selectedRepo}</h3>
          {quantLoading && <p className="text-xs text-ink-dim">Planning quant placement for this machine…</p>}
          {quantError && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
              {quantError}
            </div>
          )}
          {quantData && (
            <div className="flex flex-col gap-2">
              <div className="text-xs text-ink-dim">
                {quantData.hardware.summary}
                {quantData.is_moe ? ' · MoE architecture' : ''}
                {quantData.n_layers ? ` · ${quantData.n_layers} layers` : ''}
              </div>
              {quantData.options.map((opt) => (
                <div
                  key={opt.quant}
                  className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg-surface px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-ink">{opt.quant}</span>
                      {quantData.recommended === opt.quant && (
                        <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
                          recommended
                        </span>
                      )}
                      {!opt.feasible && (
                        <span className="rounded-full bg-rose-500/15 px-1.5 py-0.5 text-[10px] text-rose-300">
                          may not fit
                        </span>
                      )}
                    </div>
                    <div className="truncate text-xs text-ink-dim">
                      {formatGiB(opt.size)} GiB · {STRATEGY_LABEL[opt.strategy] ?? opt.strategy}
                      {opt.est_tps ? ` · ~${opt.est_tps.toFixed(1)} tok/s est.` : ''}
                    </div>
                    {opt.reason && <div className="truncate text-[11px] text-ink-faint">{opt.reason}</div>}
                  </div>
                  <button
                    disabled={busy}
                    onClick={() => download(opt)}
                    className="flex-none rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
                  >
                    Download
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default DownloadTab
