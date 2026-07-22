/** Compact "time ago" formatting for sidebar chat timestamps. Server sends
 * `updated`/`created` as Unix seconds (time.time()), not milliseconds. */
export function relativeTime(unixSeconds: number): string {
  if (!unixSeconds) return ''
  const deltaMs = Date.now() - unixSeconds * 1000
  const sec = Math.max(0, Math.round(deltaMs / 1000))
  if (sec < 45) return 'just now'
  const min = Math.round(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.round(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.round(hr / 24)
  if (day < 7) return `${day}d ago`
  const wk = Math.round(day / 7)
  if (wk < 5) return `${wk}w ago`
  const date = new Date(unixSeconds * 1000)
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function initial(name: string): string {
  const trimmed = name.trim()
  return trimmed ? trimmed[0].toUpperCase() : '?'
}

/** Bytes -> GiB, one decimal place — matches the legacy webui's `GiB()`. */
export function formatGiB(bytes: number): string {
  return (bytes / 1073741824).toFixed(1)
}

/** Custom display-name override (falls back to the real registered name).
 * Single source of truth for rendering a model's name anywhere in the UI —
 * Models tab, composer picker, character model-assign dropdown — so a
 * rename is consistent everywhere without hunting down each call site. */
export function modelDisplayName(m: { name: string; display_name?: string }): string {
  return m.display_name?.trim() || m.name
}
