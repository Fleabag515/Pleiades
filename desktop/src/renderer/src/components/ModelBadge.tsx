import { useEffect, useRef, useState } from 'react'
import { assignModel, getProfile, listModels } from '../lib/api'
import type { ModelEntry, ProfileDetail } from '../lib/types'

interface ModelBadgeProps {
  base: string
  character: string
}

/** Inline model indicator/switcher for the chat top bar (Phase G, item 3).
 *
 * Backend reality (read directly from pleiades/chats.py + server.py):
 * a chat is bound to exactly one character forever (no PUT to change a
 * chat's character or model after creation), and the model is entirely a
 * per-character setting via POST /api/profiles/{name}/model — there is no
 * per-chat model field at all. So "choose the model in this chat" is
 * implemented here as "reassign this character's model", which takes
 * effect for every chat with this character, not just the one currently
 * open. That's flagged explicitly in the dropdown copy rather than
 * implying a scope it doesn't have. */
function ModelBadge({ base, character }: ModelBadgeProps): React.JSX.Element | null {
  const [profile, setProfile] = useState<ProfileDetail | null>(null)
  const [models, setModels] = useState<ModelEntry[]>([])
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    getProfile(base, character)
      .then((p) => {
        if (!cancelled) {
          setProfile(p)
          setSelected(p.model)
        }
      })
      .catch(() => {
        if (!cancelled) setProfile(null)
      })
    listModels(base)
      .then((m) => {
        if (!cancelled) setModels(m)
      })
      .catch(() => {
        // leave the dropdown local-only if the registry can't load
      })
    return () => {
      cancelled = true
    }
  }, [base, character])

  useEffect(() => {
    if (!open) return
    const onDocClick = (e: MouseEvent): void => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [open])

  if (!profile) return null

  const label = profile.model
    ? profile.model.startsWith('openrouter:')
      ? profile.model.slice('openrouter:'.length)
      : profile.model.startsWith('ollama-cloud:')
        ? profile.model.slice('ollama-cloud:'.length)
        : profile.model
    : 'Default engine'

  const save = async (): Promise<void> => {
    setSaving(true)
    setError(null)
    try {
      const updated = await assignModel(base, character, selected)
      setProfile(updated)
      setOpen(false)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div ref={boxRef} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        title="Change the model assigned to this character"
        className="flex items-center gap-1 rounded-full border border-border bg-bg-surface px-2.5 py-1 text-xs text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
      >
        <span aria-hidden>⬡</span>
        <span className="max-w-[160px] truncate">{label}</span>
        <span aria-hidden className="text-[10px]">▾</span>
      </button>

      {open && (
        <div className="absolute left-0 top-full z-30 mt-1.5 w-72 rounded-xl border border-border bg-bg-100 p-3 shadow-2xl">
          <p className="mb-2 text-[11px] text-ink-faint">
            Sets the model for <span className="text-ink-dim">{character}</span> as a whole — it applies to
            every chat with this character, not just this one (Pleiades has no per-chat model override).
          </p>
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="mb-2 w-full rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink focus:outline-none"
          >
            <option value="">Default engine (PLEIADES_MODEL_PATH)</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          {error && <div className="mb-2 text-[11px] text-rose-300">{error}</div>}
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setOpen(false)}
              className="rounded-lg px-2.5 py-1 text-xs text-ink-dim transition hover:text-ink"
            >
              Cancel
            </button>
            <button
              disabled={saving}
              onClick={save}
              className="rounded-lg bg-accent px-3 py-1 text-xs font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Assign'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelBadge
