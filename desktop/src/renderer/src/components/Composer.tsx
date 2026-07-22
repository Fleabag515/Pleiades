import { useEffect, useRef, useState } from 'react'

interface ComposerProps {
  disabled: boolean
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
}

/** Bottom message composer: auto-growing textarea + send/stop button. */
function Composer({ disabled, streaming, onSend, onStop }: ComposerProps): React.JSX.Element {
  const [text, setText] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)

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

  return (
    <div className="border-t border-border bg-bg-app px-4 py-3">
      <div className="mx-auto flex max-w-2xl items-end gap-2 rounded-2xl border border-border bg-bg-surface px-3 py-2 shadow-sm">
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
          placeholder={disabled ? 'Select or start a chat first…' : 'Message…'}
          className="max-h-60 min-h-[24px] flex-1 resize-none bg-transparent px-1 py-1 text-[15px] text-ink placeholder:text-ink-dim focus:outline-none"
        />
        {streaming ? (
          <button
            onClick={onStop}
            className="flex-none rounded-xl bg-bg-surface-hover px-3 py-2 text-sm font-medium text-ink transition hover:bg-rose-500/20 hover:text-rose-300"
            title="Stop generating"
          >
            ◼ Stop
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={disabled || !text.trim()}
            className="flex-none rounded-xl bg-accent px-3.5 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
            title="Send"
          >
            Send
          </button>
        )}
      </div>
    </div>
  )
}

export default Composer
