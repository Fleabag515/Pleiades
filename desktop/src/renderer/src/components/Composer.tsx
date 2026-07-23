import { useEffect, useRef, useState } from 'react'
import { assignModel, getProfile, listModels, setExecPolicy, uploadAttachment } from '../lib/api'
import type { ModelEntry, ProfileDetail } from '../lib/types'
import { modelDisplayName } from '../lib/format'
import { subscribeModelsChanged } from '../lib/modelEvents'

interface ComposerProps {
  base: string
  character: string
  // The active live chat's id (see chatStore.tsx's `ensureLiveChat`) --
  // needed up front, before Send, because attachments upload to THIS chat's
  // own cache dir (see lib/api.ts's `uploadAttachment`) the moment they're
  // attached, not at send time. Attach controls stay disabled until this is
  // non-null (there's a brief window on mount/character-switch where the
  // live chat hasn't resolved yet).
  chatId: string | null
  disabled: boolean
  streaming: boolean
  // `attachments` are ids already uploaded via uploadAttachment() -- see
  // AttachedFile below. Undefined/omitted when nothing is attached.
  onSend: (text: string, attachments?: string[]) => void
  // Weave a message into the turn that's already streaming, instead of
  // starting a second one (see chatStore.tsx's `interject`). Called instead
  // of onSend whenever `streaming` is true. Attachments aren't supported on
  // this path (the attach control is disabled while streaming) -- see the
  // attach button's `disabled` below.
  onInterject: (text: string) => void
  onStop: () => void
}

/** One file the user has attached to the message that hasn't been sent yet.
 * Uploads start the moment a file is picked/dropped/pasted (not deferred to
 * Send) so the chip can show live progress/errors per file. `localId` is a
 * purely client-side key (never sent to the backend); `serverId` is the
 * `id` uploadAttachment() returned, which is what actually gets sent with
 * the message. */
interface AttachedFile {
  localId: string
  file: File
  status: 'uploading' | 'done' | 'error'
  serverId?: string
  filename?: string
  error?: string
}

function makeLocalId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function modelLabel(model: string, models: ModelEntry[]): string {
  if (!model) return 'Default engine'
  if (model.startsWith('openrouter:')) return model.slice('openrouter:'.length)
  if (model.startsWith('ollama-cloud:')) return model.slice('ollama-cloud:'.length)
  // Local registry model: show its custom display name if one is set.
  const entry = models.find((m) => m.name === model)
  return entry ? modelDisplayName(entry) : model
}

/** Green while the model process is up, red once it's crashed, otherwise a
 * neutral gray for "registered but not running" -- pulled straight from
 * `GET /api/models`'s `running`/`state` fields, no separate polling. */
function statusDotClass(m: ModelEntry): string {
  if (m.running) return 'bg-emerald-400'
  if (m.state === 'crashed') return 'bg-rose-400'
  return 'bg-ink-faint'
}

/** Composer's Approve/Ask control label. `null`/`"allow"` both read as
 * "Approve" -- new and migrated characters default to "allow" (see
 * pleiades/profiles.py), so a bare `null` should look identical to an
 * explicit "allow" here, never fall back to some third label. An explicit
 * "deny" (only reachable by hand-editing profile.json / the API, not from
 * this dropdown, which only ever writes allow/ask) still shows plainly
 * rather than silently rendering as one of the two offered choices. */
function policyLabel(execPolicy: 'allow' | 'ask' | 'deny' | null): string {
  if (execPolicy === 'ask') return 'Ask'
  if (execPolicy === 'deny') return 'Deny'
  return 'Approve'
}

/**
 * Bottom message composer: auto-growing textarea + send/stop button, plus
 * an inline model picker docked just under the input, Claude-Desktop style
 * — replacing the old top-bar `ModelBadge`, which is now retired since this
 * fully takes over its job. Backend reality is unchanged from ModelBadge's
 * docstring: a model is a per-character setting
 * (`POST /api/profiles/{name}/model`), not per-chat, so picking one here
 * reassigns the whole character.
 *
 * Owner feedback round 2: the History trigger that used to sit here as its
 * own icon is retired -- History is now one of the collapsible blocks in
 * the right panel instead (see RightPanel.tsx's HistorySection), so this
 * row is just the model picker again. Its styling was also toned down to
 * match the reference screenshots' understated composer controls: small
 * muted text with just a caret, no chip/badge border or background.
 *
 * Owner feedback round 3 (polish pass):
 *   - The picker + a new "New chat" control now sit in the SAME row as the
 *     textarea/send button, right before Send -- matching the reference
 *     screenshot's small, low-visual-weight text controls sitting inline
 *     with the other icon controls (mic, send) at the very bottom of the
 *     composer, instead of on their own row underneath.
 *   - Each model in the dropdown gets a small colored status dot (green =
 *     running, red = crashed, gray = stopped) so you can tell what's live
 *     without opening Settings. "Default engine" is no longer a selectable
 *     row at all -- it cluttered the list for something nobody actually
 *     picks day-to-day; the composer's own label still falls back to
 *     showing "Default engine" as *display* text if a character somehow
 *     has no model assigned, it's just not offered as a choice here anymore.
 *   - Rename-doesn't-refresh-live fix: this component used to fetch
 *     `listModels()` exactly once per (base, character) and never again, so
 *     a rename done in Settings' ModelsTab (a separate, simultaneously
 *     mounted tree) never reached it until something remounted this effect.
 *     It now also subscribes to `modelEvents` (see lib/modelEvents.ts) and
 *     refetches immediately whenever ANY view calls updateModel/deleteModel/
 *     startModel/stopModel, anywhere in the app.
 *   - "New chat" archived the current live chat into History and started a
 *     fresh one for this character (see chatStore.tsx's `startNewChat`) --
 *     it lived here, right next to the model picker. Removed in a later
 *     pass (see below) since it's now redundant with the header's own
 *     control.
 *
 * Attach-cache phase: the "New chat" control above is gone from this row --
 * an earlier session's header-rearrange work already put an equivalent
 * "New chat" bubble in Shell.tsx (bottom-right floating button, confirmed
 * live there, NOT in TopCharacterBar.tsx despite what an older comment on
 * this file implied), so keeping a second copy here would just be a
 * redundant control, not a needed one. Its old icon slot is repurposed into
 * an attach action instead, supporting all three ways a Claude-Desktop-like
 * composer needs to accept a file: (a) click -> native file picker, (b)
 * drag-and-drop onto this card, (c) paste onto the textarea (pasted
 * screenshots/clipboard images -- the single most-used path in practice).
 * Every attached file uploads immediately via uploadAttachment() into this
 * chat's own server-side cache dir (see pleiades/attachments.py) and shows
 * as a chip above this control row; only successfully-uploaded ids are
 * threaded through onSend so the model can be told their real paths (see
 * engine.py's `_attachments_note`). This endpoint/cache accepts images,
 * audio, and arbitrary other file types alike -- no vision/captioning/audio
 * work happens in this component at all, on purpose; that's separate,
 * parallel work building on top of this same cache.
 */
function Composer({
  base,
  character,
  chatId,
  disabled,
  streaming,
  onSend,
  onInterject,
  onStop
}: ComposerProps): React.JSX.Element {
  const [text, setText] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [profile, setProfile] = useState<ProfileDetail | null>(null)
  const [models, setModels] = useState<ModelEntry[]>([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const [assigning, setAssigning] = useState(false)
  const [pickerError, setPickerError] = useState<string | null>(null)
  const pickerRef = useRef<HTMLDivElement>(null)

  const [policyOpen, setPolicyOpen] = useState(false)
  const [policySaving, setPolicySaving] = useState(false)
  const [policyError, setPolicyError] = useState<string | null>(null)
  const policyRef = useRef<HTMLDivElement>(null)

  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const attachDisabled = disabled || streaming || !chatId

  useEffect(() => {
    let cancelled = false
    setProfile(null)
    getProfile(base, character)
      .then((p) => {
        if (!cancelled) setProfile(p)
      })
      .catch(() => {
        if (!cancelled) setProfile(null)
      })
    listModels(base)
      .then((m) => {
        if (!cancelled) setModels(m)
      })
      .catch(() => {
        // leave the picker local-only if the registry can't load
      })
    return () => {
      cancelled = true
    }
  }, [base, character])

  // Rename-doesn't-refresh-live fix: refetch the registry the instant any
  // view mutates it (rename, delete, start, stop), independent of whether
  // this component happens to remount around the same time.
  useEffect(() => {
    return subscribeModelsChanged(() => {
      listModels(base)
        .then(setModels)
        .catch(() => {
          // keep showing the last-known list if a refetch fails
        })
    })
  }, [base])

  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent): void => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) setPickerOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [pickerOpen])

  useEffect(() => {
    if (!policyOpen) return
    const onDocClick = (e: MouseEvent): void => {
      if (policyRef.current && !policyRef.current.contains(e.target as Node)) setPolicyOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [policyOpen])

  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`
  }, [text])

  // Chat switched (or the live chat id changed out from under us) -- any
  // attached-but-unsent chips belonged to the PREVIOUS chat's cache dir and
  // would upload into the wrong place / reference an id that no longer
  // makes sense for the newly active chat. Drop them rather than silently
  // sending stale ids.
  useEffect(() => {
    setAttachedFiles([])
  }, [chatId])

  /** Kicks off an upload for each newly attached file immediately (not
   * deferred to Send) so the chip can show per-file progress/errors right
   * away -- matches how Claude Desktop and friends behave, and means a
   * slow/failed upload is visible well before the user tries to hit Send. */
  const addFiles = (fileList: FileList | File[]): void => {
    if (attachDisabled || !chatId) return
    const files = Array.from(fileList)
    if (files.length === 0) return
    const chips: AttachedFile[] = files.map((file) => ({
      localId: makeLocalId(),
      file,
      status: 'uploading'
    }))
    setAttachedFiles((prev) => [...prev, ...chips])
    chips.forEach((chip) => {
      uploadAttachment(base, chatId, chip.file)
        .then((info) => {
          setAttachedFiles((prev) =>
            prev.map((f) =>
              f.localId === chip.localId
                ? { ...f, status: 'done', serverId: info.id, filename: info.filename }
                : f
            )
          )
        })
        .catch((e) => {
          setAttachedFiles((prev) =>
            prev.map((f) =>
              f.localId === chip.localId ? { ...f, status: 'error', error: (e as Error).message } : f
            )
          )
        })
    })
  }

  const removeAttachment = (localId: string): void => {
    setAttachedFiles((prev) => prev.filter((f) => f.localId !== localId))
  }

  const readyAttachmentIds = attachedFiles
    .filter((f) => f.status === 'done' && f.serverId)
    .map((f) => f.serverId as string)
  const hasUploadingAttachment = attachedFiles.some((f) => f.status === 'uploading')

  const submit = (): void => {
    const trimmed = text.trim()
    if (disabled || hasUploadingAttachment) return
    // Attachment-only sends are allowed (e.g. dropping a logo with no
    // caption, asking about it in a follow-up message) -- only block when
    // there's neither text nor a successfully-uploaded attachment.
    if (!trimmed && readyAttachmentIds.length === 0) return
    if (streaming) {
      // Mid-turn interjections don't carry attachments (the attach control
      // is disabled while streaming, so attachedFiles is always empty here
      // in practice) -- see ComposerProps.onInterject's docstring.
      onInterject(trimmed)
    } else {
      onSend(trimmed, readyAttachmentIds.length > 0 ? readyAttachmentIds : undefined)
    }
    setText('')
    setAttachedFiles([])
  }

  const chooseModel = async (name: string): Promise<void> => {
    setAssigning(true)
    setPickerError(null)
    try {
      const updated = await assignModel(base, character, name)
      setProfile(updated)
      setPickerOpen(false)
    } catch (e) {
      setPickerError((e as Error).message)
    } finally {
      setAssigning(false)
    }
  }

  const choosePolicy = async (value: 'allow' | 'ask'): Promise<void> => {
    setPolicySaving(true)
    setPolicyError(null)
    try {
      const updated = await setExecPolicy(base, character, value)
      setProfile(updated)
      setPolicyOpen(false)
    } catch (e) {
      setPolicyError((e as Error).message)
    } finally {
      setPolicySaving(false)
    }
  }

  const sendDisabled =
    disabled || hasUploadingAttachment || (!text.trim() && readyAttachmentIds.length === 0)

  return (
    <div className="flex-none bg-bg-app px-5 py-4">
      <div className="mx-auto max-w-3xl">
        {/* Stacked composer, Claude-Desktop style: the text area gets its own
         * full-width row on top, and a small low-visual-weight control row
         * (attach, model picker, send/stop) sits underneath it -- rather
         * than beside it competing for width. See owner feedback: the prior
         * single-row layout pushed the typing area narrower every time a
         * control was added next to it. */}
        <div
          className={`flex flex-col gap-2 rounded-3xl bg-bg-100 px-4 py-3 shadow-sm transition ${
            isDragging ? 'ring-2 ring-accent/60' : ''
          }`}
          onDragOver={(e) => {
            if (attachDisabled) return
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDragging(false)
            if (attachDisabled) return
            if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files)
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files)
              e.target.value = ''
            }}
          />

          {attachedFiles.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {attachedFiles.map((f) => (
                <div
                  key={f.localId}
                  title={f.error ?? f.filename ?? f.file.name}
                  className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs ${
                    f.status === 'error' ? 'border-rose-400/50 text-rose-300' : 'border-border text-ink-dim'
                  }`}
                >
                  <span className="max-w-[160px] truncate">{f.filename ?? f.file.name}</span>
                  {f.status === 'uploading' && <span className="text-ink-faint">Uploading…</span>}
                  {f.status === 'error' && <span aria-hidden>&#9888;</span>}
                  <button
                    onClick={() => removeAttachment(f.localId)}
                    aria-label={`Remove ${f.filename ?? f.file.name}`}
                    className="text-ink-faint transition hover:text-ink"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}

          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            onPaste={(e) => {
              if (attachDisabled) return
              const items = e.clipboardData?.items
              if (!items) return
              const files: File[] = []
              for (const item of Array.from(items)) {
                if (item.kind === 'file') {
                  const f = item.getAsFile()
                  if (f) files.push(f)
                }
              }
              // Only intercept the paste when it actually contains a file
              // (e.g. a pasted clipboard screenshot) -- ordinary text paste
              // must keep working exactly as before.
              if (files.length > 0) {
                e.preventDefault()
                addFiles(files)
              }
            }}
            disabled={disabled}
            rows={1}
            placeholder={
              disabled
                ? 'Select a character first…'
                : streaming
                  ? `Send ${character} a message mid-task…`
                  : `Message ${character}…`
            }
            className="max-h-60 min-h-[24px] w-full resize-none bg-transparent px-1 py-1 text-[15px] text-ink placeholder:text-ink-faint focus:outline-none"
          />

          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={attachDisabled}
                title={
                  streaming
                    ? "Can't attach files while a turn is mid-task"
                    : 'Attach a file, image, or audio clip'
                }
                aria-label="Attach a file"
                className="flex-none rounded-md px-1.5 py-1.5 text-xs text-ink-faint transition hover:text-ink-dim disabled:cursor-not-allowed disabled:opacity-40"
              >
                <span aria-hidden>&#128206;</span>
              </button>

              {!disabled && profile && (
                <div ref={pickerRef} className="relative flex-none">
                  <button
                    onClick={() => setPickerOpen((v) => !v)}
                    title="Change the model assigned to this character"
                    className="flex items-center gap-1 rounded-md px-1.5 py-1.5 text-xs text-ink-faint transition hover:text-ink-dim"
                  >
                    <span className="max-w-[140px] truncate">{modelLabel(profile.model, models)}</span>
                    <span aria-hidden className="text-[10px]">
                      ▾
                    </span>
                  </button>

                  {pickerOpen && (
                    <div className="absolute bottom-full left-0 z-30 mb-1.5 w-72 rounded-xl border border-border bg-bg-100 p-3 shadow-2xl">
                      <p className="mb-2 text-[11px] text-ink-faint">
                        Sets the model for <span className="text-ink-dim">{character}</span> as a whole — it applies
                        to every chat with this character (Pleiades has no per-chat model override).
                      </p>
                      <div className="flex max-h-52 flex-col gap-0.5 overflow-y-auto">
                        {models.map((m) => (
                          <button
                            key={m.name}
                            onClick={() => chooseModel(m.name)}
                            disabled={assigning}
                            className={`flex items-center gap-2 truncate rounded-lg px-2.5 py-1.5 text-left text-sm transition ${
                              profile.model === m.name
                                ? 'bg-bg-300 text-ink-bright'
                                : 'text-ink-dim hover:bg-bg-300/60 hover:text-ink'
                            }`}
                          >
                            <span
                              aria-hidden
                              className={`h-1.5 w-1.5 flex-none rounded-full ${statusDotClass(m)}`}
                            />
                            <span className="truncate">{modelDisplayName(m)}</span>
                          </button>
                        ))}
                        {models.length === 0 && (
                          <p className="px-2.5 py-1.5 text-xs text-ink-faint">No models registered yet.</p>
                        )}
                      </div>
                      {pickerError && <div className="mt-2 text-[11px] text-rose-300">{pickerError}</div>}
                      {assigning && <div className="mt-2 text-[11px] text-ink-faint">Saving…</div>}
                    </div>
                  )}
                </div>
              )}

              {!disabled && profile && (
                <div ref={policyRef} className="relative flex-none">
                  <button
                    onClick={() => setPolicyOpen((v) => !v)}
                    title="Tool-call approval for this character: Approve runs tools immediately, Ask prompts every time"
                    className="flex items-center gap-1 rounded-md px-1.5 py-1.5 text-xs text-ink-faint transition hover:text-ink-dim"
                  >
                    <span className="max-w-[140px] truncate">{policyLabel(profile.exec_policy)}</span>
                    <span aria-hidden className="text-[10px]">
                      ▾
                    </span>
                  </button>

                  {policyOpen && (
                    <div className="absolute bottom-full left-0 z-30 mb-1.5 w-64 rounded-xl border border-border bg-bg-100 p-3 shadow-2xl">
                      <p className="mb-2 text-[11px] text-ink-faint">
                        Tool-call approval for <span className="text-ink-dim">{character}</span> — Approve runs
                        every tool call immediately; Ask prompts you first, every time.
                      </p>
                      <div className="flex flex-col gap-0.5">
                        {(['allow', 'ask'] as const).map((value) => (
                          <button
                            key={value}
                            onClick={() => choosePolicy(value)}
                            disabled={policySaving}
                            className={`truncate rounded-lg px-2.5 py-1.5 text-left text-sm transition ${
                              (profile.exec_policy ?? 'allow') === value
                                ? 'bg-bg-300 text-ink-bright'
                                : 'text-ink-dim hover:bg-bg-300/60 hover:text-ink'
                            }`}
                          >
                            {value === 'allow' ? 'Approve' : 'Ask'}
                          </button>
                        ))}
                      </div>
                      {policyError && <div className="mt-2 text-[11px] text-rose-300">{policyError}</div>}
                      {policySaving && <div className="mt-2 text-[11px] text-ink-faint">Saving…</div>}
                    </div>
                  )}
                </div>
              )}
            </div>

            {streaming ? (
              <div className="flex flex-none items-center gap-1.5">
                {text.trim() && (
                  <button
                    onClick={submit}
                    title="Send this into the running turn, without interrupting it"
                    className="flex-none rounded-2xl bg-bg-300 px-3.5 py-2 text-sm font-medium text-ink-dim transition hover:bg-accent/20 hover:text-accent"
                  >
                    ↩ Send
                  </button>
                )}
                <button
                  onClick={onStop}
                  className="flex-none rounded-2xl bg-bg-300 px-3.5 py-2 text-sm font-medium text-ink transition hover:bg-rose-500/20 hover:text-rose-300"
                  title="Stop generating"
                >
                  ◼ Stop
                </button>
              </div>
            ) : (
              <button
                onClick={submit}
                disabled={sendDisabled}
                className="flex-none rounded-2xl bg-accent px-3.5 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
                title="Send"
              >
                Send
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Composer
