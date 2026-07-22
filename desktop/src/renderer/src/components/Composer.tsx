import { useEffect, useRef, useState } from 'react'
import { assignModel, getProfile, listModels } from '../lib/api'
import type { ModelEntry, ProfileDetail } from '../lib/types'

interface ComposerProps {
  base: string
  character: string
  disabled: boolean
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  onOpenHistory: () => void
}

function modelLabel(model: string): string {
  if (!model) return 'Default engine'
  if (model.startsWith('openrouter:')) return model.slice('openrouter:'.length)
  if (model.startsWith('ollama-cloud:')) return model.slice('ollama-cloud:'.length)
  return model
}

/**
 * Bottom message composer: auto-growing textarea + send/stop button, plus
 * (owner brief item 5) an inline model picker docked just under the input,
 * Claude-Desktop style — replacing the old top-bar `ModelBadge`, which is
 * now retired since this fully takes over its job. Backend reality is
 * unchanged from ModelBadge's docstring: a model is a per-character setting
 * (`POST /api/profiles/{name}/model`), not per-chat, so picking one here
 * reassigns the whole character.
 *
 * Post-launch owner feedback round: the History trigger used to live in
 * ChatView's now-removed header bar. The owner asked for it to get "its
 * own small dedicated icon under the chat box", separate from the
 * right-panel toggle — it now sits in this same under-input row, to the
 * right of the model picker, and just calls `onOpenHistory` (ChatView
 * still owns the `historyOpen` state and renders `HistoryOverlay` itself;
 * only the trigger's location changed).
 */
function Composer({ base, character, disabled, streaming, onSend, onStop, onOpenHistory }: ComposerProps): React.JSX.Element {
  const [text, setText] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)

  const [profile, setProfile] = useState<ProfileDetail | null>(null)
  const [models, setModels] = useState<ModelEntry[]>([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const [assigning, setAssigning] = useState(false)
  const [pickerError, setPickerError] = useState<string | null>(null)
  const pickerRef = useRef<HTMLDivElement>(null)

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

  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent): void => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) setPickerOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [pickerOpen])

  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`
  }, [text])

  const submit = (): void => {
    const trimmed = text.trim()
    if (!trimmed || disabled || streaming) return
    onSend(trimmed)
    setText('')
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

  return (
    <div className="flex-none bg-bg-app px-5 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-3xl bg-bg-100 px-4 py-3 shadow-sm">
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
            disabled={disabled}
            rows={1}
            placeholder={disabled ? 'Select a character first…' : `Message ${character}…`}
            className="max-h-60 min-h-[24px] flex-1 resize-none bg-transparent px-1 py-1 text-[15px] text-ink placeholder:text-ink-faint focus:outline-none"
          />
          {streaming ? (
            <button
              onClick={onStop}
              className="flex-none rounded-2xl bg-bg-300 px-3.5 py-2 text-sm font-medium text-ink transition hover:bg-rose-500/20 hover:text-rose-300"
              title="Stop generating"
            >
              ◼ Stop
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={disabled || !text.trim()}
              className="flex-none rounded-2xl bg-accent px-3.5 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
              title="Send"
            >
              Send
            </button>
          )}
        </div>

        <div className="mt-1.5 flex items-center justify-between px-2">
          <div className="min-w-0">
            {!disabled && profile && (
              <div ref={pickerRef} className="relative">
                <button
                  onClick={() => setPickerOpen((v) => !v)}
                  title="Change the model assigned to this character"
                  className="flex items-center gap-1 rounded-full border border-border bg-bg-surface px-2.5 py-1 text-xs text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
                >
                  <span aria-hidden>⬡</span>
                  <span className="max-w-[200px] truncate">{modelLabel(profile.model)}</span>
                  <span aria-hidden className="text-[10px]">
                    ▾
                  </span>
                </button>

                {pickerOpen && (
                  <div className="absolute bottom-full left-0 z-30 mb-1.5 w-72 rounded-xl border border-border bg-bg-100 p-3 shadow-2xl">
                    <p className="mb-2 text-[11px] text-ink-faint">
                      Sets the model for <span className="text-ink-dim">{character}</span> as a whole — it applies to
                      every chat with this character (Pleiades has no per-chat model override).
                    </p>
                    <div className="flex max-h-52 flex-col gap-0.5 overflow-y-auto">
                      <button
                        onClick={() => chooseModel('')}
                        disabled={assigning}
                        className={`rounded-lg px-2.5 py-1.5 text-left text-sm transition ${
                          !profile.model ? 'bg-bg-300 text-ink-bright' : 'text-ink-dim hover:bg-bg-300/60 hover:text-ink'
                        }`}
                      >
                        Default engine (PLEIADES_MODEL_PATH)
                      </button>
                      {models.map((m) => (
                        <button
                          key={m.name}
                          onClick={() => chooseModel(m.name)}
                          disabled={assigning}
                          className={`truncate rounded-lg px-2.5 py-1.5 text-left text-sm transition ${
                            profile.model === m.name
                              ? 'bg-bg-300 text-ink-bright'
                              : 'text-ink-dim hover:bg-bg-300/60 hover:text-ink'
                          }`}
                        >
                          {m.name}
                        </button>
                      ))}
                    </div>
                    {pickerError && <div className="mt-2 text-[11px] text-rose-300">{pickerError}</div>}
                    {assigning && <div className="mt-2 text-[11px] text-ink-faint">Saving…</div>}
                  </div>
                )}
              </div>
            )}
          </div>

          <button
            onClick={onOpenHistory}
            title="Past chats (reference only)"
            aria-label="Chat history"
            className="flex h-7 w-7 flex-none items-center justify-center rounded-full text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
          >
            <span aria-hidden>&#8986;</span>
          </button>
        </div>
      </div>
    </div>
  )
}

export default Composer
