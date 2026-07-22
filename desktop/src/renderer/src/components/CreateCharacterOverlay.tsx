import { useState } from 'react'
import { createProfile } from '../lib/api'

interface CreateCharacterOverlayProps {
  base: string
  onClose: () => void
  onCreated: () => void
}

const inputCls =
  'w-full rounded-lg border border-border bg-bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-faint focus:outline-none'

function Field({ label, children }: { label: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <label className="block">
      <div className="mb-1 text-xs text-ink-dim">{label}</div>
      {children}
    </label>
  )
}

/**
 * Character-creation overlay opened from the "+" bubble at the end of the
 * top character bar. Wired to the real `POST /api/profiles` endpoint —
 * fields mirror `ProfileCreate` in pleiades/webui/server.py exactly (name
 * required, everything else optional/defaulted server-side).
 */
function CreateCharacterOverlay({ base, onClose, onCreated }: CreateCharacterOverlayProps): React.JSX.Element {
  const [name, setName] = useState('')
  const [personaSource, setPersonaSource] = useState('auto')
  const [emailAddress, setEmailAddress] = useState('')
  const [imapHost, setImapHost] = useState('')
  const [imapPort, setImapPort] = useState(993)
  const [smtpHost, setSmtpHost] = useState('')
  const [smtpPort, setSmtpPort] = useState(587)
  const [emailPassword, setEmailPassword] = useState('')
  const [discordToken, setDiscordToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (): Promise<void> => {
    const trimmed = name.trim()
    if (!trimmed) {
      setError('Name is required.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      await createProfile(base, {
        name: trimmed,
        email_address: emailAddress,
        imap_host: imapHost,
        imap_port: imapPort,
        smtp_host: smtpHost,
        smtp_port: smtpPort,
        email_password: emailPassword || null,
        discord_token: discordToken || null,
        persona_source: personaSource
      })
      onCreated()
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/50 p-6" onClick={onClose}>
      <div
        className="flex max-h-[calc(100vh-3rem)] w-full max-w-md flex-col overflow-hidden rounded-3xl border border-border bg-bg-100 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <h2 className="text-base font-semibold text-ink-bright">New character</h2>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
          >
            ✕
          </button>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto p-5">
          {error && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
              {error}
            </div>
          )}

          <Field label="Name *">
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={inputCls}
              placeholder="e.g. Aria"
            />
          </Field>

          <Field label="Persona source">
            <input
              value={personaSource}
              onChange={(e) => setPersonaSource(e.target.value)}
              className={inputCls}
              placeholder="auto"
            />
          </Field>

          <details className="rounded-lg border border-border">
            <summary className="cursor-pointer px-3 py-2 text-xs text-ink-dim [&::-webkit-details-marker]:hidden">
              Email (optional)
            </summary>
            <div className="space-y-2 p-3 pt-0">
              <Field label="Address">
                <input value={emailAddress} onChange={(e) => setEmailAddress(e.target.value)} className={inputCls} />
              </Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="IMAP host">
                  <input value={imapHost} onChange={(e) => setImapHost(e.target.value)} className={inputCls} />
                </Field>
                <Field label="IMAP port">
                  <input
                    type="number"
                    value={imapPort}
                    onChange={(e) => setImapPort(Number(e.target.value))}
                    className={inputCls}
                  />
                </Field>
                <Field label="SMTP host">
                  <input value={smtpHost} onChange={(e) => setSmtpHost(e.target.value)} className={inputCls} />
                </Field>
                <Field label="SMTP port">
                  <input
                    type="number"
                    value={smtpPort}
                    onChange={(e) => setSmtpPort(Number(e.target.value))}
                    className={inputCls}
                  />
                </Field>
              </div>
              <Field label="Password">
                <input
                  type="password"
                  value={emailPassword}
                  onChange={(e) => setEmailPassword(e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>
          </details>

          <details className="rounded-lg border border-border">
            <summary className="cursor-pointer px-3 py-2 text-xs text-ink-dim [&::-webkit-details-marker]:hidden">
              Discord (optional)
            </summary>
            <div className="p-3 pt-0">
              <Field label="Bot token">
                <input
                  type="password"
                  value={discordToken}
                  onChange={(e) => setDiscordToken(e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>
          </details>
        </div>

        <div className="flex justify-end gap-2 border-t border-border px-5 py-4">
          <button onClick={onClose} className="rounded-lg px-3 py-1.5 text-sm text-ink-dim transition hover:text-ink">
            Cancel
          </button>
          <button
            disabled={busy}
            onClick={submit}
            className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-50"
          >
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default CreateCharacterOverlay
