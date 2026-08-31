import { useEffect, useState } from 'react'
import {
  assignCloudModel,
  cloudFeatured,
  cloudSearch,
  getSettings,
  listProfiles
} from '../../lib/api'
import type { CloudSearchResult, FeaturedCloudModel, Profile } from '../../lib/types'

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

interface AssignableRowProps {
  id: string
  name: string
  subtitle?: string
  badge?: string
  profiles: Profile[]
  assignTarget: Record<string, string>
  assignStatus: Record<string, string>
  onPick: (id: string, character: string) => void
  onAssign: (id: string) => void
}

/** One assignable model row — shared by the curated "free picks" pins and
 * the live search results so both lists behave identically. */
function AssignableRow({
  id,
  name,
  subtitle,
  badge,
  profiles,
  assignTarget,
  assignStatus,
  onPick,
  onAssign
}: AssignableRowProps): React.JSX.Element {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg-app px-3 py-2">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-ink">{name}</span>
          {badge && (
            <span className="flex-none rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
              {badge}
            </span>
          )}
        </div>
        <div className="truncate text-xs text-ink-dim">{subtitle ?? id}</div>
      </div>
      <div className="flex flex-none items-center gap-1.5">
        <select
          value={assignTarget[id] ?? ''}
          onChange={(e) => onPick(id, e.target.value)}
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
          disabled={!assignTarget[id] || assignStatus[id] === 'saving'}
          onClick={() => onAssign(id)}
          className="rounded-lg border border-border px-2.5 py-1.5 text-xs text-ink transition hover:bg-bg-surface-hover disabled:opacity-50"
        >
          Assign
        </button>
      </div>
      {assignStatus[id] && (
        <span className="flex-none text-[11px] text-ink-faint">{assignStatus[id]}</span>
      )}
    </div>
  )
}

/** Browse hosted models from ONE cloud provider and assign one to a
 * character. Split out of the old combined CloudModelsSection (which had
 * both providers behind an internal toggle) into two distinct top tabs —
 * Ollama Cloud and OpenRouter — per the Models section restructure; `source`
 * is now a fixed prop instead of local state, so each tab is a separate
 * mount with its own query/results, not a shared toggle.
 *
 * Curated FREE picks from GET /api/models/cloud-featured are pinned above
 * the search results while the query box is empty, so the fastest path to a
 * working cloud brain ("I just want Ox Alpha on this character") needs zero
 * searching and zero typing. */
function CloudModelsTab({ base, source }: CloudModelsTabProps): React.JSX.Element {
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounced(query, 400)
  const [results, setResults] = useState<CloudSearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [featured, setFeatured] = useState<FeaturedCloudModel[]>([])
  const [hasKey, setHasKey] = useState<boolean | null>(null)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [assignTarget, setAssignTarget] = useState<Record<string, string>>({})
  const [assignStatus, setAssignStatus] = useState<Record<string, string>>({})

  useEffect(() => {
    listProfiles(base)
      .then((d) => setProfiles(d.profiles))
      .catch(() => {
        // leave assign controls disabled if profiles can't load
      })
    cloudFeatured(base)
      .then((d) => setFeatured(d.featured.filter((f) => f.provider === source)))
      .catch(() => {
        // pins are a convenience; search still works without them
      })
    getSettings(base)
      .then((d) =>
        setHasKey(
          source === 'openrouter'
            ? (d.settings.openrouter_api_keys?.length ?? 0) > 0
            : (d.settings.ollama_cloud_api_keys?.length ?? 0) > 0
        )
      )
      .catch(() => setHasKey(null))
  }, [base, source])

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

  const pick = (modelId: string, character: string): void =>
    setAssignTarget((t) => ({ ...t, [modelId]: character }))

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

  const showPins = debouncedQuery.trim() === '' && featured.length > 0

  return (
    <section className="flex flex-col gap-3">
      {!showPins ? null : hasKey === false ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          A platform API key is needed before these will run — add one under Settings → Cloud APIs.
        </div>
      ) : null}

      {showPins && (
        <div className="flex flex-col gap-1.5">
          <h4 className="px-1 text-xs font-semibold uppercase tracking-wide text-ink-dim">
            Free picks · one click to assign
          </h4>
          {featured.map((f) => (
            <AssignableRow
              key={f.id}
              id={f.id}
              name={f.name}
              subtitle={f.note}
              badge="free"
              profiles={profiles}
              assignTarget={assignTarget}
              assignStatus={assignStatus}
              onPick={pick}
              onAssign={(id) => assign(id)}
            />
          ))}
        </div>
      )}

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
          <AssignableRow
            key={m.id}
            id={m.id}
            name={m.name}
            subtitle={`${m.id}${m.context ? ` · ${m.context.toLocaleString()} ctx` : ''}${
              m.is_free ? ' · $0' : ''
            }`}
            badge={m.is_free ? 'free' : undefined}
            profiles={profiles}
            assignTarget={assignTarget}
            assignStatus={assignStatus}
            onPick={pick}
            onAssign={(id) => assign(id)}
          />
        ))}
        {!loading && results.length === 0 && !error && (
          <p className="px-1 py-2 text-xs text-ink-dim">No results.</p>
        )}
      </div>
    </section>
  )
}

export default CloudModelsTab
