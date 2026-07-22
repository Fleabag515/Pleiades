import { useEffect, useRef, useState } from 'react'
import {
  assignCloudModel,
  cloudSearch,
  fetchModel,
  fetchStatus,
  hfSearch,
  listProfiles,
  quantOptions
} from '../lib/api'
import type {
  CloudSearchResult,
  FetchStatus,
  HfSearchResult,
  Profile,
  QuantOption,
  QuantOptionsResponse
} from '../lib/types'
import { formatGiB } from '../lib/format'
import ModelsTab from './settings/ModelsTab'

interface ModelFoundryViewProps {
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

/** HF search -> quant options -> download-with-progress. The backend only
 * tracks one in-flight fetch at a time (models_fetch 409s otherwise), so
 * download progress is a single global strip fed by polling
 * GET /api/models/fetch/status rather than per-card state. */
function HfSearchSection({ base, onDownloaded }: { base: string; onDownloaded: () => void }): React.JSX.Element {
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

  // Poll fetch status continuously while this view is mounted so a download
  // kicked off earlier (or from a previous mount) still shows live progress,
  // and stop the interval once it settles into done/error/idle.
  useEffect(() => {
    const tick = async (): Promise<void> => {
      try {
        const s = await fetchStatus(base)
        setStatus(s)
        if (s.status === 'done') onDownloaded()
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
          Downloaded and registered “{status.result.name}”. See Local models below.
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

function CloudModelsSection({ base }: { base: string }): React.JSX.Element {
  const [source, setSource] = useState<'openrouter' | 'ollama'>('openrouter')
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounced(query, 400)
  const [results, setResults] = useState<CloudSearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [assignTarget, setAssignTarget] = useState<Record<string, string>>({})
  const [assignStatus, setAssignStatus] = useState<Record<string, string>>({})

  useEffect(() => {
    listProfiles(base)
      .then((d) => setProfiles(d.profiles))
      .catch(() => {
        // leave assign controls disabled if profiles can't load
      })
  }, [base])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    cloudSearch(base, source, debouncedQuery.trim(), 25)
      .then((r) => {
        if (cancelled) return
        setResults(r.results)
        if (r.error) setError(r.error)
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [base, source, debouncedQuery])

  const assign = async (modelId: string): Promise<void> => {
    const character = assignTarget[modelId]
    if (!character) return
    setAssignStatus((s) => ({ ...s, [modelId]: 'saving' }))
    try {
      await assignCloudModel(base, character, source, modelId)
      setAssignStatus((s) => ({ ...s, [modelId]: `assigned to ${character}` }))
    } catch (e) {
      setAssignStatus((s) => ({ ...s, [modelId]: `error: ${(e as Error).message}` }))
    }
  }

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="flex gap-0.5 rounded-lg bg-bg-000/40 p-0.5">
          {(['openrouter', 'ollama'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                source === s ? 'bg-bg-400 text-ink-bright' : 'text-ink-dim hover:text-ink'
              }`}
            >
              {s === 'openrouter' ? 'OpenRouter' : 'Ollama Cloud'}
            </button>
          ))}
        </div>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter cloud models…"
          className="flex-1 rounded-lg border border-border bg-bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:outline-none"
        />
        {loading && <span className="flex-none text-xs text-ink-faint">loading…</span>}
      </div>

      {error && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          {error}
        </div>
      )}

      <div className="flex flex-col gap-1.5">
        {results.map((m) => (
          <div
            key={m.id}
            className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg-app px-3 py-2"
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-ink">{m.name}</span>
                {m.is_free && (
                  <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
                    free
                  </span>
                )}
              </div>
              <div className="truncate text-xs text-ink-dim">
                {m.id}
                {m.context ? ` · ${m.context.toLocaleString()} ctx` : ''}
              </div>
            </div>
            <div className="flex flex-none items-center gap-1.5">
              <select
                value={assignTarget[m.id] ?? ''}
                onChange={(e) => setAssignTarget((t) => ({ ...t, [m.id]: e.target.value }))}
                className="rounded-lg border border-border bg-bg-surface px-2 py-1.5 text-xs text-ink focus:outline-none"
              >
                <option value="">Assign to…</option>
                {profiles.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button
                disabled={!assignTarget[m.id] || assignStatus[m.id] === 'saving'}
                onClick={() => assign(m.id)}
                className="rounded-lg border border-border px-2.5 py-1.5 text-xs text-ink transition hover:bg-bg-surface-hover disabled:opacity-50"
              >
                Assign
              </button>
            </div>
            {assignStatus[m.id] && (
              <span className="flex-none text-[11px] text-ink-faint">{assignStatus[m.id]}</span>
            )}
          </div>
        ))}
        {!loading && results.length === 0 && !error && (
          <p className="px-1 py-2 text-xs text-ink-dim">No results.</p>
        )}
      </div>
    </section>
  )
}

/** Model Foundry: search Hugging Face for GGUF repos, plan quant placement
 * for this machine, download with live progress, manage already-registered
 * local models (reuses the Settings "Models" tab lifecycle logic rather
 * than duplicating it), and browse/assign cloud models (OpenRouter /
 * Ollama Cloud) to a character. */
function ModelFoundryView({ base }: ModelFoundryViewProps): React.JSX.Element {
  const [refreshKey, setRefreshKey] = useState(0)

  return (
    <div className="h-full w-full overflow-y-auto bg-bg-app px-6 py-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-8">
        <div>
          <h1 className="mb-1 text-lg font-semibold text-ink-bright">Model Foundry</h1>
          <p className="text-sm text-ink-dim">
            Search Hugging Face, plan quantization for this machine, and download GGUF models.
          </p>
          <div className="mt-4">
            <HfSearchSection base={base} onDownloaded={() => setRefreshKey((k) => k + 1)} />
          </div>
        </div>

        <div>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-ink-faint">Local models</h2>
          <ModelsTab key={refreshKey} base={base} />
        </div>

        <div>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-ink-faint">Cloud models</h2>
          <CloudModelsSection base={base} />
        </div>
      </div>
    </div>
  )
}

export default ModelFoundryView
