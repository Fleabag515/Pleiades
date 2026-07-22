import { useEffect, useState } from 'react'
import { deleteVaultEntry, listVault, revealVaultEntry, setVaultEntry } from '../../lib/api'
import type { VaultEntryMeta } from '../../lib/types'
import { relativeTime } from '../../lib/format'
import SaveStatus, { type SaveStatusValue } from './SaveStatus'

interface VaultSectionProps {
  base: string
  character: string
}

interface RowState {
  revealed: boolean
  value: string | null
  busy: boolean
  error: string | null
}

/** Per-character credential vault: metadata-only list (key/note/timestamps —
 * never values), an explicit per-row "Reveal" button that decrypts on click
 * (mirrors the webui's security model: nothing sensitive is ever fetched
 * automatically), and add/delete for custom `site:<domain>` entries. */
function VaultSection({ base, character }: VaultSectionProps): React.JSX.Element {
  const [entries, setEntries] = useState<VaultEntryMeta[]>([])
  const [rows, setRows] = useState<Record<string, RowState>>({})
  const [loadError, setLoadError] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [newKey, setNewKey] = useState('')
  const [newValue, setNewValue] = useState('')
  const [newNote, setNewNote] = useState('')
  const [addStatus, setAddStatus] = useState<SaveStatusValue>({ kind: 'idle' })

  const refresh = async (): Promise<void> => {
    try {
      const list = await listVault(base, character)
      setEntries(list)
      setRows({})
      setLoadError(null)
    } catch (e) {
      setLoadError((e as Error).message)
    }
  }

  useEffect(() => {
    refresh()
    setAdding(false)
    setNewKey('')
    setNewValue('')
    setNewNote('')
    setAddStatus({ kind: 'idle' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, character])

  const rowFor = (key: string): RowState => rows[key] ?? { revealed: false, value: null, busy: false, error: null }

  const setRow = (key: string, patch: Partial<RowState>): void => {
    setRows((r) => ({ ...r, [key]: { ...rowFor(key), ...patch } }))
  }

  const toggleReveal = async (key: string): Promise<void> => {
    const row = rowFor(key)
    if (row.revealed) {
      setRow(key, { revealed: false })
      return
    }
    setRow(key, { busy: true, error: null })
    try {
      const value = await revealVaultEntry(base, character, key)
      setRow(key, { revealed: true, value, busy: false })
    } catch (e) {
      setRow(key, { busy: false, error: (e as Error).message })
    }
  }

  const remove = async (key: string): Promise<void> => {
    if (!window.confirm(`Delete vault entry "${key}"?`)) return
    setRow(key, { busy: true, error: null })
    try {
      await deleteVaultEntry(base, character, key)
      await refresh()
    } catch (e) {
      setRow(key, { busy: false, error: (e as Error).message })
    }
  }

  const addEntry = async (): Promise<void> => {
    if (!newKey.trim() || !newValue) {
      setAddStatus({ kind: 'error', message: 'Key and value are required' })
      return
    }
    setAddStatus({ kind: 'saving' })
    try {
      const list = await setVaultEntry(base, character, {
        key: newKey.trim(),
        value: newValue,
        note: newNote.trim()
      })
      setEntries(list)
      setRows({})
      setAdding(false)
      setNewKey('')
      setNewValue('')
      setNewNote('')
      setAddStatus({ kind: 'idle' })
    } catch (e) {
      setAddStatus({ kind: 'error', message: (e as Error).message })
    }
  }

  return (
    <section className="rounded-xl border border-border bg-bg-app p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-ink">Credentials / vault</h3>
          <p className="text-xs text-ink-dim">
            Encrypted secrets for this character. Values are revealed only on explicit request.
          </p>
        </div>
        <button
          onClick={() => setAdding((v) => !v)}
          className="flex-none rounded-lg border border-border px-3 py-1.5 text-sm text-ink transition hover:bg-bg-surface-hover"
        >
          {adding ? 'Cancel' : '+ Add entry'}
        </button>
      </div>

      {loadError && (
        <div className="mb-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {loadError}
        </div>
      )}

      {adding && (
        <div className="mb-3 flex flex-col gap-2 rounded-lg border border-border bg-bg-surface p-3">
          <input
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            placeholder="key (e.g. site:example.com)"
            className="rounded-lg border border-border bg-bg-app px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
          />
          <input
            type="password"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            placeholder="secret value"
            className="rounded-lg border border-border bg-bg-app px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
          />
          <input
            value={newNote}
            onChange={(e) => setNewNote(e.target.value)}
            placeholder="optional label"
            className="rounded-lg border border-border bg-bg-app px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
          />
          <div className="flex items-center justify-end gap-3">
            <SaveStatus status={addStatus} />
            <button
              onClick={addEntry}
              disabled={addStatus.kind === 'saving'}
              className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </div>
      )}

      {entries.length === 0 ? (
        <p className="text-sm text-ink-dim">No secrets stored.</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.map((en) => {
            const row = rowFor(en.key)
            const badge = en.reserved ? 'reserved' : en.site ? 'site' : 'custom'
            const badgeStyle = en.reserved
              ? 'bg-amber-500/15 text-amber-300'
              : en.site
                ? 'bg-violet-500/15 text-violet-300'
                : 'bg-emerald-500/15 text-emerald-300'
            return (
              <div
                key={en.key}
                className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-bg-surface px-3 py-2 text-sm"
              >
                <span className={`rounded-full px-2 py-0.5 text-xs ${badgeStyle}`}>{badge}</span>
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-xs text-ink">{en.key}</div>
                  <div className="truncate text-xs text-ink-dim">
                    {en.meta?.note ? `${en.meta.note} · ` : ''}
                    updated {relativeTime(en.updated_at)}
                  </div>
                  {row.error && <div className="text-xs text-rose-400">{row.error}</div>}
                </div>
                {row.revealed && row.value != null ? (
                  <code className="max-w-[220px] truncate rounded bg-bg-app px-2 py-1 text-xs text-ink">
                    {row.value}
                  </code>
                ) : (
                  <span className="tracking-widest text-ink-dim">••••••••</span>
                )}
                <button
                  onClick={() => toggleReveal(en.key)}
                  disabled={row.busy}
                  className="flex-none rounded-lg border border-border px-2.5 py-1 text-xs text-ink transition hover:bg-bg-surface-hover disabled:opacity-50"
                >
                  {row.busy ? '…' : row.revealed ? 'Hide' : 'Reveal'}
                </button>
                <button
                  onClick={() => remove(en.key)}
                  disabled={row.busy}
                  className="flex-none rounded-lg border border-border px-2.5 py-1 text-xs text-rose-300 transition hover:bg-rose-500/10 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

export default VaultSection
