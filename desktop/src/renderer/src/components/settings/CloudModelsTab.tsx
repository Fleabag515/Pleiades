import { useEffect, useState } from 'react'
import { assignCloudModel, cloudSearch, listProfiles } from '../../lib/api'
import type { CloudSearchResult, Profile } from '../../lib/types'

interface CloudModelsTabProps {
  base: string
  // Backend's GET /api/models/cloud-search?source=... only accepts these two
  // literal values (verified against pleiades/webui/server.py's cloud_search
  // handler) — "ollama" means Ollama Cloud, not local Ollama.
  source: 'openrouter' | 'ollama'
}

function useDebounced(value: string, delayMs: number): string {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])
  return debounced
}

/** Browse hosted models from ONE cloud provider and assign one to a
 * character. Split out of the old combined CloudModelsSection (which had
 * both providers behind an internal toggle) into two distinct top tabs —
 * Ollama Cloud and OpenRouter — per the Models section restructure; `source`
 * is now a fixed prop instead of local state, so each tab is a separate
 * mount with its own query/results, not a shared toggle. */
function CloudModelsTab({ base, source }: CloudModelsTabProps): React.JSX.Element {
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
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Filter ${source === 'openrouter' ? 'OpenRouter' : 'Ollama Cloud'} models…`}
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

export default CloudModelsTab
