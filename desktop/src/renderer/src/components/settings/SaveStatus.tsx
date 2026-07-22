type Status = { kind: 'idle' } | { kind: 'saving' } | { kind: 'success'; message?: string } | { kind: 'error'; message: string }

interface SaveStatusProps {
  status: Status
}

/** Small inline feedback strip for a save/write action: saving spinner text,
 * a success flash, or an error message. Used across the Characters tab so
 * every write gives real feedback instead of silently no-op'ing. */
function SaveStatus({ status }: SaveStatusProps): React.JSX.Element | null {
  if (status.kind === 'idle') return null
  if (status.kind === 'saving') {
    return <span className="text-xs text-ink-dim">Saving…</span>
  }
  if (status.kind === 'success') {
    return <span className="text-xs text-emerald-400">{status.message ?? 'Saved'}</span>
  }
  return <span className="text-xs text-rose-400">{status.message}</span>
}

export type { Status as SaveStatusValue }
export default SaveStatus
