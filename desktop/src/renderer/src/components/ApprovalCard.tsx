import type { PendingApproval } from '../lib/types'

interface ApprovalCardProps {
  approval: PendingApproval
  onApprove: () => void
  onDeny: () => void
  busy: boolean
}

/** Inline tool-call approval gate, rendered in the message flow while the
 * engine is blocked waiting on GET/POST /api/chats/{id}/approval. */
function ApprovalCard({ approval, onApprove, onDeny, busy }: ApprovalCardProps): React.JSX.Element {
  return (
    <div className="mx-auto w-full max-w-2xl rounded-2xl border border-accent/40 bg-accent/10 p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-ink">
        <span className="text-accent">⚠</span>
        Wants to run a tool
      </div>
      <div className="mt-2 rounded-lg border border-border bg-bg-surface/60 p-3 font-mono text-xs text-ink-dim">
        <div className="text-ink">{approval.tool}</div>
        <div className="mt-1 break-all">{approval.args}</div>
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <button
          onClick={onDeny}
          disabled={busy}
          className="rounded-lg border border-border px-3 py-1.5 text-sm text-ink transition hover:bg-bg-surface disabled:opacity-50"
        >
          Deny
        </button>
        <button
          onClick={onApprove}
          disabled={busy}
          className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
        >
          Approve
        </button>
      </div>
    </div>
  )
}

export default ApprovalCard
