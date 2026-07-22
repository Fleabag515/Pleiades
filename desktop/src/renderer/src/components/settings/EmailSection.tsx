import { useEffect, useState } from 'react'
import { getEmailPresets, setEmailConfig } from '../../lib/api'
import type { EmailPresets, ProfileDetail } from '../../lib/types'
import SaveStatus, { type SaveStatusValue } from './SaveStatus'

interface EmailSectionProps {
  base: string
  profile: ProfileDetail
  onUpdated: (p: ProfileDetail) => void
}

/** Editable email account config. The password field is write-only: it is
 * never fetched or pre-filled from the server (matches the vault's "never
 * return secret values in list responses" rule) — leaving it blank on save
 * keeps whatever password is already stored. */
function EmailSection({ base, profile, onUpdated }: EmailSectionProps): React.JSX.Element {
  const [emailAddress, setEmailAddress] = useState(profile.email_address)
  const [imapHost, setImapHost] = useState(profile.imap_host)
  const [imapPort, setImapPort] = useState(profile.imap_port)
  const [smtpHost, setSmtpHost] = useState(profile.smtp_host)
  const [smtpPort, setSmtpPort] = useState(profile.smtp_port)
  const [password, setPassword] = useState('')
  const [preset, setPreset] = useState('')
  const [presetNote, setPresetNote] = useState<string | null>(null)
  const [presets, setPresets] = useState<EmailPresets>({})
  const [status, setStatus] = useState<SaveStatusValue>({ kind: 'idle' })

  useEffect(() => {
    setEmailAddress(profile.email_address)
    setImapHost(profile.imap_host)
    setImapPort(profile.imap_port)
    setSmtpHost(profile.smtp_host)
    setSmtpPort(profile.smtp_port)
    setPassword('')
    setPreset('')
    setPresetNote(null)
    setStatus({ kind: 'idle' })
  }, [profile.name])

  useEffect(() => {
    getEmailPresets(base)
      .then(setPresets)
      .catch(() => {
        // Presets are a convenience only; leave the dropdown empty on failure.
      })
  }, [base])

  const applyPreset = (key: string): void => {
    setPreset(key)
    const p = presets[key]
    if (!p) {
      setPresetNote(null)
      return
    }
    setImapHost(p.imap_host)
    setImapPort(p.imap_port)
    setSmtpHost(p.smtp_host)
    setSmtpPort(p.smtp_port)
    setPresetNote(p.note ?? null)
  }

  const save = async (): Promise<void> => {
    setStatus({ kind: 'saving' })
    try {
      const updated = await setEmailConfig(base, profile.name, {
        email_address: emailAddress.trim(),
        imap_host: imapHost.trim(),
        imap_port: imapPort || 993,
        smtp_host: smtpHost.trim(),
        smtp_port: smtpPort || 587,
        password: password || null
      })
      onUpdated(updated)
      setPassword('')
      setStatus({ kind: 'success', message: 'Email settings saved' })
    } catch (e) {
      setStatus({ kind: 'error', message: (e as Error).message })
    }
  }

  return (
    <section className="rounded-xl border border-border bg-bg-app p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Email account</h3>
        <span
          className={`rounded-full px-2 py-0.5 text-xs ${
            profile.has_email ? 'bg-emerald-500/15 text-emerald-300' : 'bg-bg-surface-hover text-ink-dim'
          }`}
        >
          {profile.has_email ? 'configured' : 'not set'}
        </span>
      </div>

      <div className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-ink-dim">
          Provider preset
          <select
            value={preset}
            onChange={(e) => applyPreset(e.target.value)}
            className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink focus:outline-none"
          >
            <option value="">Choose a provider preset…</option>
            {Object.keys(presets).map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        {presetNote && <p className="-mt-1 text-xs text-ink-dim">{presetNote}</p>}

        <label className="flex flex-col gap-1 text-xs text-ink-dim">
          Email address
          <input
            value={emailAddress}
            onChange={(e) => setEmailAddress(e.target.value)}
            placeholder="alice@example.com"
            className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
          />
        </label>

        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1 text-xs text-ink-dim">
            IMAP host
            <input
              value={imapHost}
              onChange={(e) => setImapHost(e.target.value)}
              placeholder="imap.example.com"
              className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-ink-dim">
            IMAP port
            <input
              type="number"
              value={imapPort}
              onChange={(e) => setImapPort(Number(e.target.value) || 993)}
              className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink focus:outline-none"
            />
          </label>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1 text-xs text-ink-dim">
            SMTP host
            <input
              value={smtpHost}
              onChange={(e) => setSmtpHost(e.target.value)}
              placeholder="smtp.example.com"
              className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-dim/60 focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-ink-dim">
            SMTP port
            <input
              type="number"
              value={smtpPort}
              onChange={(e) => setSmtpPort(Number(e.target.value) || 587)}
              className="rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink focus:outline-none"
            />
          </label>
        </div>

        <label className="flex flex-col gap-1 text-xs text-ink-dim">
          App password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
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
            Save email settings
          </button>
        </div>
      </div>
    </section>
  )
}

export default EmailSection
