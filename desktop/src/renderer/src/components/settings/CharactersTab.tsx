import { useEffect, useState } from 'react'
import { getProfile } from '../../lib/api'
import type { Profile, ProfileDetail } from '../../lib/types'
import Avatar from '../Avatar'
import EmailSection from './EmailSection'
import DiscordSection from './DiscordSection'
import ModelAssignSection from './ModelAssignSection'
import VaultSection from './VaultSection'

interface CharactersTabProps {
  base: string
  profiles: Profile[]
}

/** Full per-character settings: email, Discord, model assignment, and the
 * credentials vault. Replaces the earlier read-only roster — every section
 * here writes through to the real backend and reports success/failure. */
function CharactersTab({ base, profiles }: CharactersTabProps): React.JSX.Element {
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState<ProfileDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!selected && profiles.length > 0) setSelected(profiles[0].name)
  }, [profiles, selected])

  useEffect(() => {
    if (!selected) return
    // Guard against out-of-order resolution: switching characters quickly
    // can let an earlier fetch's response land after a later one's, which
    // would otherwise overwrite the correct detail (and let child sections
    // save edits under the wrong profile name). Same pattern as
    // DownloadTab/CloudModelsTab/Composer/RightPanel.
    let cancelled = false
    setLoading(true)
    setError(null)
    getProfile(base, selected)
      .then((d) => {
        if (!cancelled) setDetail(d)
      })
      .catch((e) => {
        if (!cancelled) {
          setError((e as Error).message)
          setDetail(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [base, selected])

  if (profiles.length === 0) {
    return <p className="text-sm text-ink-dim">No characters yet.</p>
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-1.5">
        {profiles.map((p) => (
          <button
            key={p.name}
            onClick={() => setSelected(p.name)}
            className={`flex items-center gap-2 rounded-full border px-2.5 py-1 text-sm transition ${
              selected === p.name
                ? 'border-accent bg-accent/15 text-ink'
                : 'border-border text-ink-dim hover:bg-bg-surface-hover hover:text-ink'
            }`}
          >
            <Avatar base={base} character={p.name} size={18} />
            {p.name}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}

      {loading && !detail && <p className="text-sm text-ink-dim">Loading…</p>}

      {detail && (
        <div className="flex flex-col gap-4">
          <EmailSection base={base} profile={detail} onUpdated={setDetail} />
          <DiscordSection base={base} profile={detail} onUpdated={setDetail} />
          <ModelAssignSection base={base} profile={detail} onUpdated={setDetail} />
          <VaultSection base={base} character={detail.name} />
        </div>
      )}
    </div>
  )
}

export default CharactersTab
