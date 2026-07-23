// Lightweight client-side mirror of pleiades/scheduler.py's 5-field cron
// parser -- just enough to validate a cron string and render a
// human-readable preview before it round-trips to the backend, which stays
// the source of truth (parse_cron() there re-validates on every
// create/update regardless of what this says).

const FIELD_RANGES: Record<string, [number, number]> = {
  minute: [0, 59],
  hour: [0, 23],
  day: [1, 31],
  month: [1, 12],
  weekday: [0, 7]
}

const FIELD_NAMES = ['minute', 'hour', 'day', 'month', 'weekday'] as const

const WEEKDAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

function parseField(spec: string, lo: number, hi: number): number[] | null {
  const out = new Set<number>()
  for (const rawPart of spec.split(',')) {
    const part = rawPart.trim()
    if (!part) return null
    let step = 1
    let body = part
    if (part.includes('/')) {
      const pieces = part.split('/')
      if (pieces.length !== 2) return null
      body = pieces[0]
      step = Number(pieces[1])
      if (!Number.isInteger(step) || step <= 0) return null
    }
    let baseLo: number
    let baseHi: number
    if (body === '*') {
      baseLo = lo
      baseHi = hi
    } else if (body.includes('-')) {
      const pieces = body.split('-')
      if (pieces.length !== 2) return null
      baseLo = Number(pieces[0])
      baseHi = Number(pieces[1])
    } else {
      baseLo = baseHi = Number(body)
    }
    if (!Number.isInteger(baseLo) || !Number.isInteger(baseHi)) return null
    if (baseLo < lo || baseHi > hi || baseLo > baseHi) return null
    for (let v = baseLo; v <= baseHi; v += 1) {
      if ((v - baseLo) % step === 0) out.add(v)
    }
  }
  if (out.size === 0) return null
  return [...out].sort((a, b) => a - b)
}

export interface CronCheck {
  valid: boolean
  error?: string
  describe?: string
}

/** Validate a 5-field cron string and, if valid, produce a short
 * human-readable preview ("Runs every day at 07:00"). */
export function checkCron(expr: string): CronCheck {
  const fields = expr.trim().split(/\s+/)
  if (fields.length !== 5 || fields.some((f) => !f)) {
    return { valid: false, error: 'cron needs 5 fields: minute hour day month weekday' }
  }
  const parsed: number[][] = []
  for (let i = 0; i < 5; i += 1) {
    const name = FIELD_NAMES[i]
    const [lo, hi] = FIELD_RANGES[name]
    const values = parseField(fields[i], lo, hi)
    if (!values) {
      return { valid: false, error: `bad ${name} field: "${fields[i]}"` }
    }
    parsed.push(values)
  }
  return { valid: true, describe: describeCron(fields, parsed) }
}

function describeCron(fields: string[], parsed: number[][]): string {
  const [minutes, hours, days, weekdays] = [parsed[0], parsed[1], parsed[2], parsed[4]]
  const [minuteF, hourF, dayF, monthF, weekdayF] = fields
  const pad = (n: number): string => String(n).padStart(2, '0')

  const everyDay = dayF === '*' && monthF === '*' && weekdayF === '*'
  const singleTime = minutes.length === 1 && hours.length === 1

  if (minuteF === '*' && hourF === '*' && everyDay) {
    return 'Runs every minute'
  }
  if (singleTime && everyDay) {
    return `Runs every day at ${pad(hours[0])}:${pad(minutes[0])}`
  }
  if (singleTime && dayF === '*' && monthF === '*' && weekdays.length === 1) {
    return `Runs every ${WEEKDAY_NAMES[weekdays[0] % 7]} at ${pad(hours[0])}:${pad(minutes[0])}`
  }
  if (singleTime && dayF !== '*' && monthF === '*' && weekdayF === '*' && days.length === 1) {
    return `Runs on day ${days[0]} of every month at ${pad(hours[0])}:${pad(minutes[0])}`
  }
  return `Runs on a custom schedule (${fields.join(' ')})`
}
