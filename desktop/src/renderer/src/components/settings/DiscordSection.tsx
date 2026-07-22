import { useEffect, useState } from 'react'
import { setDiscordConfig } from '../../lib/api'
import type { ProfileDetail } from '../../lib/types'
import SaveStatus, { type SaveStatusValue } from './SaveStatus'

interface DiscordSectionProps {
  base: string
  profile: ProfileDetail
  onUpdated: (p: ProfileDetail) => void
}

/** Editable Discord bot config. The token field is write-only, same rule as
 * the email password: never fetched/displayed, blank on save = keep current. */
function DiscordSection({ base, profile, onUpdated }: DiscordSectionProps): React.JSX.Element {
  const [enabled, setEnabled] = useState(profile.discord_enabled)
  const [requireMention, setRequireMention] = useState(profile.discord_require_mention)
  const [respondToBots, setRespondToBots] = useState(profile.discord_respond_to_bots)
  const [allowedChannels, setAllowedChannels] = useState(profile.discord_allowed_channels)
  const [token, setToken] = useState('')
  const [status, setStatus] = useState<SaveStatusValue>({ kind: 'idle' })

  useEffect(() => {
    setEnabled(profile.discord_enabled)
    setRequireMention(profile.discord_require_mention)
    setRespondToBots(profile.discord_respond_to_bots)
    setAllowedChannels(profile.discord_allowed_channels)
    setToken('')
    setStatus({ kind: 'idle' })
  }, [profile.name])

  const save = async (): Promise<void> => {
    setStatus({ kind: 'saving' })
    try {
      const updated = await setDiscordConfig(base, profile.name, {
        token: token || null,
        enabled,
        require_mention: requireMention,
        respond_to_bots: respondToBots,
        allowed_channels: allowedChannels.trim()
      })
      onUpdated(updated)
      setToken('')
      setStatus({ kind: 'success', message: 'Discord settings saved' })
    } catch (e) {
      setStatus({ kind: 'error', message: (e as Error).message })
    }
  }

  return (
    <section className="rounded-xl border border-border bg-bg-app p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Discord bot</h3>
        <span
          className={`rounded-full px-2 py-0.5 text-xs ${
            profile.discord_enabled ? 'bg-violet-500/15 text-violet-300' : 'bg-bg-surface-hover text-ink-dim'
          }`}
        >
          {profile.discord_enabled ? 'enabled' : 'disabled'}
        </span>
      </div>

      <div className="flex flex-col gap-3">
        <label className="flex items-center gap-2 text-sm text-ink">
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
          Host this character as a Discord bot
        </label>
        <label className="flex items-center gap-2 text-sm text-ink">
          <input
            type="checkbox"
            checked={requireMention}
            onChange={(e) => setRequireMention(e.target.checked)}
          />
          In servers, only respond when pinged (DMs always work)
        </label>
        <label className="flex items-center gap-2 text-sm text-ink">
          <input
            type="checkbox"
            checked={respondToBots}
            onChange={(e) => setRespondToBots(e.target.checked)}
          />
          Respond to other bots&apos; messages
        </label>

        <label className="flex flex-col gap-1 text-xs text-ink-dim">
          Allowed channels
          <input
            value={allowedChannels}
            onChange={(e) => setAllowedChannels(e.target.value)}
            placeholder="general, bot-chat, 123456789 (blank = all channels)"
            className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-ink-dim">
          Bot token
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="leave blank to keep current"
            className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
          />
        </label>

        <div className="flex items-center justify-end gap-3">
          <SaveStatus status={status} />
          <button
            onClick={save}
            disabled={status.kind === 'saving'}
            className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
          >
            Save Discord settings
          </button>
        </div>
      </div>
    </section>
  )
}

export default DiscordSection
