import { useEffect, useState } from 'react'
import {
  assignCloudModel,
  assignModel,
  cloudFeatured,
  getSettings,
  listModels
} from '../../lib/api'
import type { FeaturedCloudModel, ModelEntry, ProfileDetail } from '../../lib/types'
import { modelDisplayName } from '../../lib/format'
import SaveStatus, { type SaveStatusValue } from './SaveStatus'

interface ModelAssignSectionProps {
  base: string
  profile: ProfileDetail
  onUpdated: (p: ProfileDetail) => void
}

/** Prefix used by the curated free cloud brain (Ox Alpha via OpenRouter).
 * Mirrors engine.py's `_provider_for` convention: "<prefix>:<model id>" is a
 * cloud brain, anything else is a local GGUF registry name. */
const OX_ALPHA = 'openrouter:stealth/ox-alpha'

function cloudProviderOf(model: string): 'openrouter' | 'ollama' | null {
  if (model.startsWith('openrouter:')) return 'openrouter'
  if (model.startsWith('ollama-cloud:')) return 'ollama'
  return null
}

/** Assign this character's "brain": either a LOCAL model served by the
 * Pleiades engine (default / registered GGUFs) or a CLOUD model routed
 * through OpenRouter / Ollama Cloud. The dropdown groups both paths so the
 * choice is explicit and one click; curated free cloud brains (Ox Alpha
 * first) come from GET /api/models/cloud-featured, full catalogs live in
 * Settings → Models → OpenRouter / Ollama Cloud. Cloud assignment goes
 * through POST /api/profiles/{name}/cloud-model (the plain-model endpoint
 * rejects unregistered names); local assignment keeps POST .../model. */
function ModelAssignSection({
  base,
  profile,
  onUpdated
}: ModelAssignSectionProps): React.JSX.Element {
  const [models, setModels] = useState<ModelEntry[]>([])
  const [featured, setFeatured] = useState<FeaturedCloudModel[]>([])
  const [selected, setSelected] = useState(profile.model)
  const [status, setStatus] = useState<SaveStatusValue>({ kind: 'idle' })
  const [keys, setKeys] = useState<{ openrouter: boolean; ollama: boolean }>({
    openrouter: true,
    ollama: true
  })

  useEffect(() => {
    setSelected(profile.model)
    setStatus({ kind: 'idle' })
  }, [profile.name, profile.model])

  useEffect(() => {
    listModels(base)
      .then(setModels)
      .catch(() => {
        // Leave the dropdown as "default engine only" on failure.
      })
    cloudFeatured(base)
      .then((d) => setFeatured(d.featured))
      .catch(() => {
        // pins are optional sugar; local assignment still works without them
      })
    getSettings(base)
      .then((d) =>
        setKeys({
          openrouter: (d.settings.openrouter_api_keys?.length ?? 0) > 0,
          ollama: (d.settings.ollama_cloud_api_keys?.length ?? 0) > 0
        })
      )
      .catch(() => setKeys({ openrouter: true, ollama: true }))
  }, [base])

  const currentProvider = cloudProviderOf(selected)
  const keyMissing =
    currentProvider === 'openrouter'
      ? !keys.openrouter
      : currentProvider === 'ollama'
        ? !keys.ollama
        : false

  const save = async (): Promise<void> => {
    setStatus({ kind: 'saving' })
    try {
      const provider = cloudProviderOf(selected)
      const updated =
        provider != null
          ? await assignCloudModel(base, profile.name, provider, selected.split(':', 2)[1])
          : await assignModel(base, profile.name, selected)
      onUpdated(updated)
      setStatus({ kind: 'success', message: 'Brain assignment saved' })
    } catch (e) {
      setStatus({ kind: 'error', message: (e as Error).message })
    }
  }

  const statusChip = (() => {
    if (currentProvider != null) {
      return (
        <span className="rounded-full bg-violet-500/15 px-2 py-0.5 text-xs text-violet-300">
          cloud · {currentProvider === 'openrouter' ? 'OpenRouter' : 'Ollama Cloud'}
        </span>
      )
    }
    if (profile.model_info) {
      return (
        <span
          className={`rounded-full px-2 py-0.5 text-xs ${
            profile.model_info.registered
              ? profile.model_info.running
                ? 'bg-emerald-500/15 text-emerald-300'
                : 'bg-amber-500/15 text-amber-300'
              : 'bg-rose-500/15 text-rose-300'
          }`}
        >
          {profile.model_info.registered
            ? profile.model_info.running
              ? 'running'
              : 'registered, stopped'
            : 'model missing'}
        </span>
      )
    }
    return null
  })()

  return (
    <section className="rounded-xl border border-border bg-bg-app p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Brain</h3>
        {statusChip}
      </div>

      <div className="flex items-center gap-3">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="flex-1 rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink focus:outline-none"
        >
          <optgroup label="Local · Pleiades engine">
            <option value="">Default engine (PLEIADES_MODEL_PATH)</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {modelDisplayName(m)}
              </option>
            ))}
          </optgroup>
          {featured.length > 0 && (
            <optgroup label="Cloud · OpenRouter / Ollama Cloud">
              {featured.map((f) => (
                <option
                  key={f.id}
                  value={`${f.provider === 'ollama' ? 'ollama-cloud' : f.provider}:${f.id}`}
                >
                  {f.name} — {f.note}
                </option>
              ))}
              {currentProvider != null &&
                !featured.some(
                  (f) =>
                    `${f.provider === 'ollama' ? 'ollama-cloud' : f.provider}:${f.id}` === selected
                ) && <option value={selected}>{selected} (assigned)</option>}
            </optgroup>
          )}
        </select>
        <button
          onClick={save}
          disabled={status.kind === 'saving'}
          className="flex-none rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
        >
          Assign
        </button>
      </div>

      {(keyMissing || selected === OX_ALPHA) && (
        <p className="mt-2 text-xs text-amber-300">
          {keyMissing
            ? `No ${currentProvider === 'ollama' ? 'Ollama Cloud' : 'OpenRouter'} key yet — add one under Settings → Cloud APIs.`
            : 'Free preview listing: it may end or start billing at any time (see the note in Models → OpenRouter).'}
        </p>
      )}
      <p className="mt-1 text-xs text-ink-dim">
        Full cloud catalogs: Models → OpenRouter / Ollama Cloud.
      </p>

      <div className="mt-2">
        <SaveStatus status={status} />
      </div>
    </section>
  )
}

export default ModelAssignSection
