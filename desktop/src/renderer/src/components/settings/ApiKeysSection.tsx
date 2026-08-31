import { useEffect, useState } from 'react'
import { addApiKey, getSettings, removeApiKey, validateApiKey } from '../../lib/api'
import type { SettingsView } from '../../lib/types'

interface ApiKeysSectionProps {
  base: string
}

type ProviderId = 'openrouter' | 'ollama-cloud'

interface ProviderSpec {
  id: ProviderId
  label: string
  blurb: string
  keysUrl: string
  keyHint: string
  field: 'openrouter_api_keys' | 'ollama_cloud_api_keys'
}

const PROVIDERS: ProviderSpec[] = [
  {
    id: 'openrouter',
    label: 'OpenRouter',
    blurb: 'Hosted models from many labs behind one key — including the free Ox Alpha preview.',
    keysUrl: 'https://openrouter.ai/settings/keys',
    keyHint: 'sk-or-…',
    field: 'openrouter_api_keys'
  },
  {
    id: 'ollama-cloud',
    label: 'Ollama Cloud',
    blurb: 'Ollama’s hosted models (ollama.com) — same models, no local hardware needed.',
    keysUrl: 'https://ollama.com/settings/keys',
    keyHint: 'API key from ollama.com',
    field: 'ollama_cloud_api_keys'
  }
]

interface RowState {
  busy: boolean
}

/** Cloud platform API-key management INSIDE the desktop app (no web UI
 * needed). Keys are stored in the project .env by the backend (multiple per
 * provider; engine.py rotates around rate limits automatically), always
 * displayed masked, and can be live-checked against the provider before
 * saving. Nothing here ever sends a key anywhere except its own provider. */
function ApiKeysSection({ base }: ApiKeysSectionProps): React.JSX.Element {
  const [settings, setSettings] = useState<SettingsView | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [addingFor, setAddingFor] = useState<ProviderId | null>(null)
  const [draft, setDraft] = useState('')
  const [saveBusy, setSaveBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [rows, setRows] = useState<Record<string, RowState>>({})
  // Validation state keyed by provider — one check at a time is enough here.
  const [validating, setValidating] = useState<ProviderId | null>(null)
  const [validation, setValidation] = useState<Record<ProviderId, string>>({
    openrouter: '',
    'ollama-cloud': ''
  })

  const refresh = async (): Promise<void> => {
    try {
      const d = await getSettings(base)
      setSettings(d.settings)
      setLoadError(null)
    } catch (e) {
      setLoadError((e as Error).message)
    }
  }

  useEffect(() => {
    refresh()
    setAddingFor(null)
    setDraft('')
    setStatus(null)
    setError(null)
    setValidation({ openrouter: '', 'ollama-cloud': '' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base])

  const rowBusy = (provider: ProviderId, index: number): boolean =>
    rows[`${provider}:${index}`]?.busy ?? false

  const setRowBusy = (provider: ProviderId, index: number, busy: boolean): void =>
    setRows((r) => ({ ...r, [`${provider}:${index}`]: { busy } }))

  const saveDraft = async (spec: ProviderSpec): Promise<void> => {
    const key = draft.trim()
    if (!key) return
    setSaveBusy(true)
    setError(null)
    setStatus(null)
    try {
      const d = await addApiKey(base, spec.id, key)
      setSettings(d.settings)
      setDraft('')
      setAddingFor(null)
      setStatus(`${spec.label} key saved`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaveBusy(false)
    }
  }

  const checkDraft = async (spec: ProviderSpec): Promise<void> => {
    const key = draft.trim()
    if (!key) return
    setValidating(spec.id)
    setValidation((v) => ({ ...v, [spec.id]: '' }))
    try {
      const r = await validateApiKey(base, spec.id, key)
      setValidation((v) => ({
        ...v,
        [spec.id]: r.valid ? `✓ ${r.detail}` : `✕ ${r.detail}`
      }))
    } catch (e) {
      setValidation((v) => ({ ...v, [spec.id]: `✕ ${(e as Error).message}` }))
    } finally {
      setValidating(null)
    }
  }

  const remove = async (spec: ProviderSpec, index: number): Promise<void> => {
    if (!window.confirm(`Remove this ${spec.label} key?`)) return
    setRowBusy(spec.id, index, true)
    try {
      const d = await removeApiKey(base, spec.id, index)
      setSettings(d.settings)
      setStatus(`${spec.label} key removed`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRowBusy(spec.id, index, false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-ink-dim">
        Keys for cloud model platforms. Stored in this machine&rsquo;s <code>.env</code>, shown
        masked, and used automatically whenever a character runs a cloud brain. Multiple keys per
        platform are allowed — Pleiades rotates them around rate limits.
      </p>

      {loadError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {loadError}
        </div>
      )}
      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}
      {status && !error && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300">
          {status}
        </div>
      )}

      {PROVIDERS.map((spec) => {
        const keys = settings?.[spec.field] ?? []
        return (
          <section key={spec.id} className="rounded-xl border border-border bg-bg-app p-4">
            <div className="mb-1 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-ink">{spec.label}</h3>
              <span
                className={`rounded-full px-2 py-0.5 text-xs ${
                  keys.length > 0
                    ? 'bg-emerald-500/15 text-emerald-300'
                    : 'bg-amber-500/15 text-amber-300'
                }`}
              >
                {keys.length > 0 ? `${keys.length} key${keys.length > 1 ? 's' : ''}` : 'no key'}
              </span>
            </div>
            <p className="text-xs text-ink-dim">{spec.blurb}</p>

            {keys.length > 0 && (
              <div className="mt-3 flex flex-col gap-1.5">
                {keys.map((k, i) => (
                  <div
                    key={`${i}-${k}`}
                    className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg-surface px-3 py-2"
                  >
                    <code className="truncate font-mono text-xs text-ink">{k}</code>
                    <button
                      onClick={() => remove(spec, i)}
                      disabled={rowBusy(spec.id, i)}
                      className="flex-none rounded-lg border border-border px-2.5 py-1 text-xs text-rose-300 transition hover:bg-rose-500/10 disabled:opacity-50"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}

            {addingFor === spec.id ? (
              <div className="mt-3 flex flex-col gap-2 rounded-lg border border-border bg-bg-surface p-3">
                <input
                  autoFocus
                  type="password"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder={spec.keyHint}
                  className="rounded-lg border border-border bg-bg-app px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
                />
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-xs ${validation[spec.id].startsWith('✓') ? 'text-emerald-300' : 'text-rose-300'}`}
                  >
                    {validation[spec.id]}
                  </span>
                  <div className="flex flex-none items-center gap-2">
                    <button
                      onClick={() => checkDraft(spec)}
                      disabled={!draft.trim() || validating === spec.id}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs text-ink transition hover:bg-bg-surface-hover disabled:opacity-50"
                    >
                      {validating === spec.id ? 'Checking…' : 'Test'}
                    </button>
                    <button
                      onClick={() => saveDraft(spec)}
                      disabled={!draft.trim() || saveBusy}
                      className="rounded-lg bg-accent px-3.5 py-1.5 text-xs font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
                    >
                      Save key
                    </button>
                    <button
                      onClick={() => {
                        setAddingFor(null)
                        setDraft('')
                        setValidation((v) => ({ ...v, [spec.id]: '' }))
                      }}
                      className="rounded-lg px-2 py-1.5 text-xs text-ink-dim transition hover:text-ink"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="mt-3 flex items-center gap-3">
                <button
                  onClick={() => {
                    setAddingFor(spec.id)
                    setDraft('')
                  }}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm text-ink transition hover:bg-bg-surface-hover"
                >
                  + Add key
                </button>
                <a
                  href={spec.keysUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-accent underline-offset-2 hover:underline"
                >
                  Get a key ↗
                </a>
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}

export default ApiKeysSection
