import { useEffect, useState } from 'react'
import { assignModel, listModels } from '../../lib/api'
import type { ModelEntry, ProfileDetail } from '../../lib/types'
import SaveStatus, { type SaveStatusValue } from './SaveStatus'

interface ModelAssignSectionProps {
  base: string
  profile: ProfileDetail
  onUpdated: (p: ProfileDetail) => void
}

/** Assign a registered local model to this character (blank = default
 * engine / PLEIADES_MODEL_PATH). Cloud model assignment (openrouter/ollama
 * cloud) is out of scope here — this covers the local-GGUF registry case
 * the brief calls out explicitly. */
function ModelAssignSection({ base, profile, onUpdated }: ModelAssignSectionProps): React.JSX.Element {
  const [models, setModels] = useState<ModelEntry[]>([])
  const [selected, setSelected] = useState(profile.model)
  const [status, setStatus] = useState<SaveStatusValue>({ kind: 'idle' })

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
  }, [base])

  const save = async (): Promise<void> => {
    setStatus({ kind: 'saving' })
    try {
      const updated = await assignModel(base, profile.name, selected)
      onUpdated(updated)
      setStatus({ kind: 'success', message: 'Model assignment saved' })
    } catch (e) {
      setStatus({ kind: 'error', message: (e as Error).message })
    }
  }

  return (
    <section className="rounded-xl border border-border bg-bg-app p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Model</h3>
        {profile.model_info && (
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
        )}
      </div>

      <div className="flex items-center gap-3">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="flex-1 rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink focus:outline-none"
        >
          <option value="">Default engine (PLEIADES_MODEL_PATH)</option>
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
        <button
          onClick={save}
          disabled={status.kind === 'saving'}
          className="flex-none rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
        >
          Assign
        </button>
      </div>
      <div className="mt-2">
        <SaveStatus status={status} />
      </div>
    </section>
  )
}

export default ModelAssignSection
