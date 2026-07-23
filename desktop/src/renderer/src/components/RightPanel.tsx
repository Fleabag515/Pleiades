import { useCallback, useEffect, useRef, useState } from 'react'
import {
  browserViewNavigate,
  browserViewOpenSeparate,
  browserViewResize,
  browserViewStart,
  browserViewStatus,
  browserViewStop,
  browserViewWsUrl,
  createScheduledTask,
  deleteScheduledTask,
  getAgent,
  getChat,
  getProfile,
  getProfileTools,
  getWorkJob,
  getWorkTasks,
  listAgents,
  listChats,
  listModels,
  listScheduledTasks,
  listWorkJobs,
  runScheduledTaskNow,
  setAgentsSettings,
  stopAgent,
  updateScheduledTask
} from '../lib/api'
import type {
  AgentEvent,
  AgentSummary,
  BrowserViewStatus,
  ChatDetail,
  ChatSummary,
  ModelEntry,
  ProfileDetail,
  ProfileTools,
  ScheduledTask,
  ToolInfo,
  WorkEvent,
  WorkJobDetail,
  WorkJobSummary,
  WorkTask
} from '../lib/types'
import { checkCron } from '../lib/cron'
import { modelDisplayName, relativeTime } from '../lib/format'
import MessageBubble from './MessageBubble'

interface RightPanelProps {
  base: string
  character: string
  open: boolean
}

/**
 * The right-side contextual panel, rebuilt (owner feedback round 2) to match
 * the reference screenshots precisely: NOT a bordered sub-panel with its own
 * header bar (that's what produced the redundant "{character}" label the
 * owner kept seeing even after a prior phase claimed to remove name bars
 * elsewhere -- it had quietly come back here). Now there is no panel header
 * at all: just stacked, individually-rounded "block" sections (Progress,
 * History, Browser, Scheduled tasks) sitting directly on the app's normal
 * bg-app background, each its own bg-200 surface with an icon/label/chevron
 * header and a gap between blocks -- no wrapping frame or border around the
 * whole panel, exactly per the reference. History used to be a separate
 * modal triggered from an icon under Composer (see HistoryOverlay's prior
 * incarnation); that trigger is retired and its content lives here as one
 * more collapsible block instead, per owner brief item 2.
 *
 * ChatView still lays out `[chat column][this panel]` as one flex row, and
 * only this wrapper's width animates open/closed -- see ChatView.tsx.
 */
function RightPanel({ base, character, open }: RightPanelProps): React.JSX.Element {
  const [progressOpen, setProgressOpen] = useState(true)
  const [toolsOpen, setToolsOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [browserOpen, setBrowserOpen] = useState(false)
  const [scheduledOpen, setScheduledOpen] = useState(false)
  const [agentsOpen, setAgentsOpen] = useState(false)

  return (
    <div
      className={`h-full flex-none overflow-hidden bg-bg-app transition-[width] duration-300 ease-in-out ${
        open ? 'w-96' : 'w-0'
      }`}
    >
      <div className="flex h-full w-96 flex-col overflow-y-auto px-3 py-3">
        <div className="flex flex-col gap-2.5">
          <CollapsibleBlock
            icon={<span aria-hidden>&#9642;</span>}
            label="Progress"
            open={progressOpen}
            onToggle={() => setProgressOpen((v) => !v)}
          >
            <ProgressSection base={base} character={character} active={open && progressOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#9881;</span>}
            label="Tools"
            open={toolsOpen}
            onToggle={() => setToolsOpen((v) => !v)}
          >
            <ToolsSection base={base} character={character} active={open && toolsOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#8635;</span>}
            label="History"
            open={historyOpen}
            onToggle={() => setHistoryOpen((v) => !v)}
          >
            <HistorySection base={base} character={character} active={open && historyOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#8859;</span>}
            label="Browser"
            open={browserOpen}
            onToggle={() => setBrowserOpen((v) => !v)}
          >
            <BrowserViewSection base={base} character={character} active={open && browserOpen} />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#9711;</span>}
            label="Scheduled tasks"
            open={scheduledOpen}
            onToggle={() => setScheduledOpen((v) => !v)}
          >
            <ScheduledTasksSection
              base={base}
              character={character}
              active={open && scheduledOpen}
            />
          </CollapsibleBlock>

          <CollapsibleBlock
            icon={<span aria-hidden>&#9737;</span>}
            label="Agents"
            open={agentsOpen}
            onToggle={() => setAgentsOpen((v) => !v)}
          >
            <AgentsSection base={base} character={character} active={open && agentsOpen} />
          </CollapsibleBlock>
        </div>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Shared collapsible "block" shell -- the one visual unit the whole panel
// is built from. bg-200 on top of the panel's bg-app background is the
// "barely distinguished, not a bright card" surface pairing from the
// reference; radius-2xl (1rem/16px) matches its rounded-corner blocks;
// gap between blocks comes from the parent's flex gap-2.5, not anything
// here. No border/outline anywhere on this shell -- the surface-color step
// is the only thing that reads as "a block".
// --------------------------------------------------------------------------

function CollapsibleBlock({
  icon,
  label,
  open,
  onToggle,
  children
}: {
  icon: React.ReactNode
  label: string
  open: boolean
  onToggle: () => void
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="overflow-hidden rounded-2xl bg-bg-200">
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2.5 px-3.5 py-3 text-left transition hover:bg-bg-300/40"
      >
        <span className="flex h-5 w-5 flex-none items-center justify-center text-sm text-ink-dim">
          {icon}
        </span>
        <span className="flex-1 text-sm font-medium text-ink">{label}</span>
        <span
          aria-hidden
          className={`flex-none text-[10px] text-ink-faint transition-transform duration-200 ${
            open ? '' : '-rotate-90'
          }`}
        >
          &#9662;
        </span>
      </button>
      {open && <div className="px-3.5 pb-3.5">{children}</div>}
    </div>
  )
}

// --------------------------------------------------------------------------
// Progress list — real /api/work job data, no fake data.
// --------------------------------------------------------------------------

function statusColor(status: string): string {
  if (status === 'running') return 'text-accent'
  if (status === 'error') return 'text-rose-300'
  if (status === 'cancelled') return 'text-ink-faint'
  return 'text-emerald-300'
}

function eventLine(e: WorkEvent): string {
  if (e.kind === 'tool_call') return `→ ${e.name}`
  if (e.kind === 'tool_result') return `${e.ok === false ? '✗' : '✓'} ${e.name}`
  return (e.text || '').slice(0, 80)
}

function taskStatusColor(status: WorkTask['status']): string {
  if (status === 'completed') return 'text-emerald-300'
  if (status === 'in_progress') return 'text-accent'
  return 'text-ink-faint'
}

function taskMarker(status: WorkTask['status']): string {
  if (status === 'completed') return '\u2713'
  if (status === 'in_progress') return '\u25cf'
  return '\u25cb'
}

/**
 * The live task list a character keeps via create_task/update_task/list_tasks
 * (pleiades/harness/builtins/tasks.py) for this job -- the real plan-tracking
 * tool the earlier stub comment on this component used to say was still to
 * come. Renders subject + a small status indicator per task, and the
 * activeForm text while a task is in_progress so the human sees what's
 * actually happening right now, not just a static label.
 */
function TaskList({ tasks }: { tasks: WorkTask[] }): React.JSX.Element | null {
  if (tasks.length === 0) return null
  return (
    <ul className="mt-1.5 flex flex-col gap-1 border-t border-border pt-1.5">
      {tasks.map((t) => (
        <li key={t.id} className="flex items-start gap-1.5">
          <span className={`mt-0.5 flex-none text-[10px] ${taskStatusColor(t.status)}`}>
            {taskMarker(t.status)}
          </span>
          <div className="min-w-0 flex-1">
            <span
              className={`block truncate text-[11px] ${
                t.status === 'completed' ? 'text-ink-faint line-through' : 'text-ink'
              }`}
            >
              {t.subject}
            </span>
            {t.status === 'in_progress' && t.activeForm && (
              <span className="block truncate text-[10px] text-accent">{t.activeForm}</span>
            )}
          </div>
        </li>
      ))}
    </ul>
  )
}

/**
 * Real progress/task-list panel: /api/work job data (status, approvals,
 * recent tool events) plus each job's live task list from the harness's own
 * task-list tool (create_task/update_task/list_tasks -- the same
 * subject/description/status/activeForm shape Cowork's own TaskCreate/
 * TaskUpdate tools use). This replaced an earlier stub that only showed raw
 * job/event data with no plan-tracking of its own.
 */
function ProgressSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [jobs, setJobs] = useState<WorkJobSummary[]>([])
  const [details, setDetails] = useState<Record<string, WorkJobDetail>>({})
  const [taskLists, setTaskLists] = useState<Record<string, WorkTask[]>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!active) return
    let cancelled = false

    const refresh = async (): Promise<void> => {
      try {
        const all = await listWorkJobs(base)
        if (cancelled) return
        const mine = all.filter((j) => j.character === character)
        setJobs(mine)
        setError(null)
        const running = mine.filter((j) => j.status === 'running').slice(0, 5)
        const [fetchedDetails, fetchedTasks] = await Promise.all([
          Promise.all(running.map((j) => getWorkJob(base, j.id).catch(() => null))),
          Promise.all(running.map((j) => getWorkTasks(base, j.id).catch(() => null)))
        ])
        if (cancelled) return
        setDetails((prev) => {
          const next = { ...prev }
          fetchedDetails.forEach((d, i) => {
            if (d) next[running[i].id] = d
          })
          return next
        })
        setTaskLists((prev) => {
          const next = { ...prev }
          fetchedTasks.forEach((t, i) => {
            if (t) next[running[i].id] = t
          })
          return next
        })
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }

    refresh()
    const id = setInterval(refresh, 2500)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [base, character, active])

  return (
    <div>
      {error && <div className="mb-2 text-xs text-rose-300">{error}</div>}
      {jobs.length === 0 ? (
        <p className="text-xs text-ink-faint">
          Nothing running for {character || 'this character'} right now.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {jobs.slice(0, 8).map((j) => {
            const detail = details[j.id]
            const tasks = taskLists[j.id] ?? []
            return (
              <li key={j.id} className="rounded-lg bg-bg-300/50 px-2.5 py-2">
                <div className="flex items-start justify-between gap-2">
                  <span className="truncate text-xs text-ink">{j.task}</span>
                  <span
                    className={`flex-none text-[10px] font-medium uppercase ${statusColor(j.status)}`}
                  >
                    {j.status}
                  </span>
                </div>
                <div className="mt-0.5 text-[10px] text-ink-faint">{relativeTime(j.started)}</div>
                {j.pending_approval && (
                  <div className="mt-1 rounded bg-amber-500/10 px-1.5 py-1 text-[10px] text-amber-300">
                    Waiting on approval: {j.pending_approval.tool}
                  </div>
                )}
                {tasks.length > 0 ? (
                  <TaskList tasks={tasks} />
                ) : (
                  detail &&
                  detail.events.length > 0 && (
                    <ul className="mt-1.5 flex flex-col gap-0.5 border-t border-border pt-1.5">
                      {detail.events.slice(-3).map((e, i) => (
                        <li key={i} className="truncate text-[10px] text-ink-faint">
                          {eventLine(e)}
                        </li>
                      ))}
                    </ul>
                  )
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Tools — the character's actual tool belt (GET /api/profiles/{name}/tools).
//
// Everything here is real, live data off the same code paths the engine
// itself uses to decide what a character can do:
//   - the tool list + one-line descriptions come straight from each Tool's
//     own .description (the exact text sent to the model in its function-
//     calling schema) via build_default_belt() -- a tool this character
//     hasn't configured (no email set up) or that isn't installed (no
//     camoufox/[browser] extra) is simply absent, not shown "disabled".
//   - "status" is the real exec_policy gate: safe/read-only tools always
//     show available; everything else reflects this character's actual
//     allow/ask/deny setting (see profiles.Profile.exec_policy, set from
//     the composer's Approve/Ask dropdown or Settings).
//   - "used Nx / last used ..." is mined from this character's own
//     persisted chat transcripts (chats.tool_usage()) — real completed
//     tool calls, real timestamps. Calls made before per-call timestamps
//     existed still count toward N but are labelled "before tracking
//     started" instead of showing a guessed time.
//   - the "+N more via tool-search" footnote is a real count of the
//     harness's on-demand tool registry (files/shell/git/web/memory/
//     subagents/MCP...) that chat can reach through find_tools/call_tool
//     without those schemas being resident every turn -- see engine.py's
//     _bridge_harness_tools. Not broken out per-tool here on purpose: it's
//     a much larger, more volatile registry than the belt, and the whole
//     point of tool-search is lazy loading rather than a static list.
// --------------------------------------------------------------------------

function toolStatusMeta(status: ToolInfo['status']): { label: string; className: string } {
  if (status === 'blocked') return { label: 'blocked', className: 'text-rose-300' }
  if (status === 'needs_approval') return { label: 'needs approval', className: 'text-amber-300' }
  return { label: 'available', className: 'text-emerald-300' }
}

function execPolicyBlurb(policy: string): string {
  if (policy === 'deny') return 'Non-read-only tools are blocked (exec policy: deny).'
  if (policy === 'ask') return 'Non-read-only tools ask for approval before running (exec policy: ask).'
  return 'Tools run automatically without asking (exec policy: allow).'
}

function toolUsageLine(t: ToolInfo): string {
  if (t.use_count === 0) return 'Not used yet'
  if (t.last_used) return `Used ${t.use_count}× · last ${relativeTime(t.last_used)}`
  return `Used ${t.use_count}× (before tracking started)`
}

function ToolsSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [data, setData] = useState<ProfileTools | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async (): Promise<void> => {
    if (!character) return
    try {
      setData(await getProfileTools(base, character))
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [base, character])

  useEffect(() => {
    if (!active) return
    refresh()
    const id = setInterval(refresh, 8000)
    return () => clearInterval(id)
  }, [active, refresh])

  if (error) {
    return <div className="text-xs text-rose-300">{error}</div>
  }
  if (!data) {
    return (
      <p className="text-xs text-ink-faint">
        Loading {character || 'this character'}&rsquo;s tool belt&hellip;
      </p>
    )
  }

  // Most-recently-used first; never-used tools sort to the bottom together.
  const sorted = [...data.tools].sort((a, b) => (b.last_used ?? 0) - (a.last_used ?? 0))

  return (
    <div>
      <p className="mb-2 text-[11px] text-ink-faint">{execPolicyBlurb(data.exec_policy)}</p>
      {sorted.length === 0 ? (
        <p className="text-xs text-ink-faint">
          No tools configured for {character || 'this character'} yet.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {sorted.map((t) => {
            const meta = toolStatusMeta(t.status)
            return (
              <li key={t.name} className="rounded-lg bg-bg-300/50 px-2.5 py-2">
                <div className="flex items-start justify-between gap-2">
                  <span className="truncate text-xs font-medium text-ink">{t.name}</span>
                  <span
                    className={`flex-none text-[10px] font-medium uppercase ${meta.className}`}
                  >
                    {meta.label}
                  </span>
                </div>
                <p className="mt-0.5 line-clamp-2 text-[11px] text-ink-faint">{t.description}</p>
                <div className="mt-1 flex items-center gap-1.5 text-[10px] text-ink-faint">
                  {t.safe && (
                    <span className="flex-none rounded bg-bg-200 px-1 py-0.5">read-only</span>
                  )}
                  <span className="truncate">{toolUsageLine(t)}</span>
                </div>
              </li>
            )
          })}
        </ul>
      )}
      {data.bridge_count > 0 && (
        <p className="mt-2 text-[10px] text-ink-faint">
          +{data.bridge_count} more reachable on demand via tool-search (files, shell, git, web,
          memory, subagents, &hellip;) — loaded only when a turn actually needs one.
        </p>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// History — folded into the panel per owner brief item 2 (previously a

// separate modal, HistoryOverlay, triggered by its own icon under Composer).
// Same data/behavior as before: a read-only list of a character's past
// chats plus a read-only transcript viewer. Selecting one never makes it
// "the live chat" again -- old chats are reference only, never resumed.
// --------------------------------------------------------------------------

function HistorySection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [viewing, setViewing] = useState<ChatDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!active) return
    setViewing(null)
    listChats(base)
      .then((all) => setChats(all.filter((c) => c.character === character)))
      .catch((e) => setError((e as Error).message))
  }, [base, character, active])

  if (viewing) {
    return (
      <div>
        <button
          onClick={() => setViewing(null)}
          className="mb-2 text-xs text-ink-dim transition hover:text-ink"
        >
          &larr; Back to {character}&rsquo;s history
        </button>
        <div className="max-h-80 overflow-y-auto rounded-lg bg-bg-300/40">
          {viewing.messages.length === 0 && (
            <div className="p-3 text-xs text-ink-faint">Empty chat.</div>
          )}
          {viewing.messages.map((m, i) => (
            <MessageBubble key={i} message={m} base={base} character={character} chatId={viewing.id} />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div>
      <p className="mb-2 text-[11px] text-ink-faint">
        Read-only reference — {character || 'this character'}&rsquo;s active conversation is always
        the live chat above. Opening an old one here doesn&rsquo;t resume it.
      </p>
      {error && <div className="mb-2 text-xs text-rose-300">{error}</div>}
      {chats.length === 0 ? (
        <p className="text-xs text-ink-faint">No past chats yet.</p>
      ) : (
        <ul className="flex flex-col gap-1">
          {chats.map((c) => (
            <li key={c.id}>
              <button
                onClick={() =>
                  getChat(base, c.id)
                    .then(setViewing)
                    .catch((e) => setError((e as Error).message))
                }
                className="flex w-full flex-col items-start rounded-lg px-2.5 py-2 text-left transition hover:bg-bg-300/50"
              >
                <span className="truncate text-xs text-ink">{c.title || '(new chat)'}</span>
                <span className="text-[10px] text-ink-faint">
                  {relativeTime(c.updated)} &middot; {c.messages} messages
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Scheduled tasks — real data via /api/scheduled-tasks. A character can
// create these itself (create_scheduled_task/etc. harness tools, see
// pleiades/harness/builtins/schedule.py) or you can create/edit/delete them
// right here — both go through the same store
// (~/.pleiades/scheduled_tasks.json), never two sources of truth.
// --------------------------------------------------------------------------

function formatSchedule(t: ScheduledTask): string {
  if (t.cron) {
    const check = checkCron(t.cron)
    return check.valid && check.describe ? check.describe : `cron ${t.cron}`
  }
  if (t.next_run) return new Date(t.next_run * 1000).toLocaleString()
  return 'fired (one-time)'
}

function ScheduledTasksSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [tasks, setTasks] = useState<ScheduledTask[]>([])
  const [error, setError] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [taskText, setTaskText] = useState('')
  const [mode, setMode] = useState<'cron' | 'once'>('once')
  const [cronInput, setCronInput] = useState('0 7 * * *')
  const [whenInput, setWhenInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [runningId, setRunningId] = useState<string | null>(null)

  const cronCheck = mode === 'cron' ? checkCron(cronInput) : null

  const refresh = useCallback(async (): Promise<void> => {
    try {
      setTasks(await listScheduledTasks(base))
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [base])

  useEffect(() => {
    if (!active) return
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [active, refresh])

  const submit = async (): Promise<void> => {
    if (!taskText.trim()) return
    if (mode === 'cron' && !checkCron(cronInput).valid) return
    setBusy(true)
    try {
      const input =
        mode === 'cron'
          ? { task: taskText.trim(), character, cron: cronInput.trim() }
          : {
              task: taskText.trim(),
              character,
              fire_at: whenInput ? new Date(whenInput).getTime() / 1000 : 0
            }
      await createScheduledTask(base, input)
      setTaskText('')
      setWhenInput('')
      setAdding(false)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const toggle = async (t: ScheduledTask): Promise<void> => {
    try {
      await updateScheduledTask(base, t.id, { enabled: !t.enabled })
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const remove = async (t: ScheduledTask): Promise<void> => {
    try {
      await deleteScheduledTask(base, t.id)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const runNow = async (t: ScheduledTask): Promise<void> => {
    setRunningId(t.id)
    try {
      await runScheduledTaskNow(base, t.id)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunningId(null)
    }
  }

  return (
    <div>
      {error && <div className="mb-2 text-xs text-rose-300">{error}</div>}
      {tasks.length === 0 && !adding && (
        <p className="mb-2 text-xs text-ink-faint">No scheduled tasks yet.</p>
      )}
      {tasks.length > 0 && (
        <ul className="mb-2 flex flex-col gap-2">
          {tasks.map((t) => (
            <li key={t.id} className="rounded-lg bg-bg-300/50 px-2.5 py-2">
              <div className="flex items-start justify-between gap-2">
                <span className="truncate text-xs text-ink">{t.task}</span>
                <div className="flex flex-none items-center gap-1.5">
                  <button
                    onClick={() => runNow(t)}
                    disabled={runningId === t.id}
                    className="text-[10px] font-medium uppercase text-ink-faint transition hover:text-accent disabled:opacity-40"
                    aria-label="Run scheduled task now"
                    title="Run now"
                  >
                    {runningId === t.id ? '…' : '▶'}
                  </button>
                  <button
                    onClick={() => toggle(t)}
                    className={`text-[10px] font-medium uppercase transition ${
                      t.enabled ? 'text-accent' : 'text-ink-faint'
                    }`}
                  >
                    {t.enabled ? 'on' : 'off'}
                  </button>
                  <button
                    onClick={() => remove(t)}
                    className="text-ink-faint transition hover:text-rose-300"
                    aria-label="Delete scheduled task"
                  >
                    &#10005;
                  </button>
                </div>
              </div>
              <div className="mt-0.5 text-[10px] text-ink-faint">
                {formatSchedule(t)}
                {t.character ? ` · ${t.character}` : ''}
                {t.last_run ? ` · last ran ${relativeTime(t.last_run)}` : ''}
              </div>
            </li>
          ))}
        </ul>
      )}

      {adding ? (
        <div className="flex flex-col gap-1.5 rounded-lg bg-bg-300/40 p-2">
          <input
            value={taskText}
            onChange={(e) => setTaskText(e.target.value)}
            placeholder="What should run?"
            className="rounded bg-bg-200 px-2 py-1 text-xs text-ink outline-none"
          />
          <div className="flex gap-1.5 text-[10px]">
            <button
              onClick={() => setMode('once')}
              className={`rounded px-1.5 py-0.5 ${mode === 'once' ? 'bg-accent/20 text-accent' : 'text-ink-faint'}`}
            >
              One-time
            </button>
            <button
              onClick={() => setMode('cron')}
              className={`rounded px-1.5 py-0.5 ${mode === 'cron' ? 'bg-accent/20 text-accent' : 'text-ink-faint'}`}
            >
              Recurring
            </button>
          </div>
          {mode === 'once' ? (
            <input
              type="datetime-local"
              value={whenInput}
              onChange={(e) => setWhenInput(e.target.value)}
              className="rounded bg-bg-200 px-2 py-1 text-xs text-ink outline-none"
            />
          ) : (
            <input
              value={cronInput}
              onChange={(e) => setCronInput(e.target.value)}
              placeholder="minute hour day month weekday"
              className="rounded bg-bg-200 px-2 py-1 text-xs text-ink outline-none"
            />
          )}
          {mode === 'cron' && cronInput.trim() && (
            <p className={`text-[10px] ${cronCheck?.valid ? 'text-ink-faint' : 'text-rose-300'}`}>
              {cronCheck?.valid ? cronCheck.describe : cronCheck?.error}
            </p>
          )}
          <div className="flex justify-end gap-2 pt-0.5">
            <button
              onClick={() => setAdding(false)}
              className="text-[10px] text-ink-faint hover:text-ink"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={
                busy ||
                !taskText.trim() ||
                (mode === 'once' && !whenInput) ||
                (mode === 'cron' && !cronCheck?.valid)
              }
              className="rounded bg-accent px-2 py-1 text-[10px] font-medium text-bg-app disabled:opacity-40"
            >
              Add
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setAdding(true)}
          className="w-full rounded-lg border border-dashed border-border px-2.5 py-1.5 text-[11px] text-ink-faint transition hover:border-ink-faint hover:text-ink"
        >
          + Schedule a task
        </button>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Agents — ephemeral background sub-agents spawned via the model's own
// spawn_agents chat-belt tool (pleiades/agents.py + pleiades/tools/
// agents_tool.py). Unlike Scheduled tasks, agents aren't created from this
// panel -- the model spawns them; this panel is the toggle + model picker
// (gates both the chat tool and the toolsearch bridge's dispatch_subagent
// back door -- see Engine._bridge_harness_tools) plus live observability
// (task/status/history + a status light) and a stop button, polling
// GET /api/agents exactly like Progress polls GET /api/work.
// --------------------------------------------------------------------------

/** green = an in-flight tool_call with no result yet; red = the agent errored
 * out, OR any tool_result in its history had ok:false -- sticky, since once
 * something has gone wrong for this agent that stays the more important
 * signal even if it keeps working afterward; yellow = everything else
 * (thinking/planning between calls, queued, done clean). */
function agentLightClass(agent: AgentSummary, events: AgentEvent[]): string {
  if (agent.status === 'error') return 'bg-rose-400'
  if (events.some((e) => e.kind === 'tool_result' && e.ok === false)) return 'bg-rose-400'
  const last = events[events.length - 1]
  if (agent.status === 'running' && last?.kind === 'tool_call') return 'bg-emerald-400'
  return 'bg-amber-400'
}

function AgentsSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [profile, setProfile] = useState<ProfileDetail | null>(null)
  const [models, setModels] = useState<ModelEntry[]>([])
  const [agents, setAgents] = useState<AgentSummary[]>([])
  const [details, setDetails] = useState<Record<string, AgentEvent[]>>({})
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!active || !character) return
    let cancelled = false
    getProfile(base, character)
      .then((p) => !cancelled && setProfile(p))
      .catch((e) => !cancelled && setError((e as Error).message))
    listModels(base)
      .then((m) => !cancelled && setModels(m))
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [base, character, active])

  useEffect(() => {
    if (!active) return
    let cancelled = false

    const refresh = async (): Promise<void> => {
      try {
        const all = await listAgents(base)
        if (cancelled) return
        const mine = all.filter((a) => a.character === character)
        setAgents(mine)
        setError(null)
        const live = mine.filter((a) => a.status === 'running' || a.status === 'queued').slice(0, 6)
        const fetched = await Promise.all(live.map((a) => getAgent(base, a.id).catch(() => null)))
        if (cancelled) return
        setDetails((prev) => {
          const next = { ...prev }
          fetched.forEach((d, i) => {
            if (d) next[live[i].id] = d.events
          })
          return next
        })
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }

    refresh()
    const id = setInterval(refresh, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [base, character, active])

  const toggleEnabled = async (): Promise<void> => {
    if (!character) return
    setBusy(true)
    try {
      setProfile(await setAgentsSettings(base, character, { enabled: !profile?.agents_enabled }))
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const changeModel = async (model: string): Promise<void> => {
    if (!character) return
    try {
      setProfile(await setAgentsSettings(base, character, { model }))
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const stop = async (id: string): Promise<void> => {
    try {
      await stopAgent(base, id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const enabled = !!profile?.agents_enabled

  return (
    <div>
      {error && <div className="mb-2 text-xs text-rose-300">{error}</div>}

      <div className="mb-2 flex items-center justify-between gap-2 rounded-lg bg-bg-300/40 px-2.5 py-2">
        <div className="min-w-0">
          <div className="text-xs text-ink">Background agents</div>
          <div className="truncate text-[10px] text-ink-faint">
            Let {character || 'this character'} delegate tasks to ephemeral background agents.
          </div>
        </div>
        <button
          onClick={toggleEnabled}
          disabled={busy || !character}
          className={`flex-none rounded-full px-2.5 py-1 text-[10px] font-medium uppercase transition ${
            enabled ? 'bg-accent/20 text-accent' : 'bg-bg-400/40 text-ink-faint'
          } disabled:opacity-40`}
        >
          {enabled ? 'On' : 'Off'}
        </button>
      </div>

      {enabled && (
        <div className="mb-2">
          <div className="flex items-center gap-1.5">
            <span className="flex-none text-[10px] text-ink-faint">Model</span>
            <select
              value={profile?.agents_model || ''}
              onChange={(e) => changeModel(e.target.value)}
              className="min-w-0 flex-1 rounded-lg bg-bg-surface px-2 py-1 text-xs text-ink outline-none"
            >
              <option value="">Auto (currently running model)</option>
              {models.map((m) => (
                <option
                  key={m.name}
                  value={m.name}
                  disabled={!m.running}
                  title={m.running ? undefined : 'Start this model elsewhere first -- agents never auto-start a model'}
                >
                  {modelDisplayName(m)}
                  {m.running ? '' : ' (stopped -- start it first)'}
                </option>
              ))}
            </select>
          </div>
          {profile?.agents_model && !models.find((m) => m.name === profile.agents_model && m.running) && (
            <div className="mt-1 text-[10px] text-amber-300">
              &ldquo;{profile.agents_model}&rdquo; isn&apos;t running -- spawning an agent will fail until
              you start it.
            </div>
          )}
        </div>
      )}

      {agents.length === 0 ? (
        <p className="text-xs text-ink-faint">
          {enabled
            ? `No agents spawned yet for ${character || 'this character'}.`
            : `Agents are off for ${character || 'this character'}.`}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {agents.slice(0, 8).map((a) => {
            const events = details[a.id] || []
            const stoppable = a.status === 'running' || a.status === 'queued'
            return (
              <li key={a.id} className="rounded-lg bg-bg-300/50 px-2.5 py-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex min-w-0 items-start gap-1.5">
                    <span
                      className={`mt-1 h-1.5 w-1.5 flex-none rounded-full ${agentLightClass(a, events)}`}
                      aria-hidden
                      title={a.status}
                    />
                    <span className="truncate text-xs text-ink">{a.task}</span>
                  </div>
                  <div className="flex flex-none items-center gap-1.5">
                    <span className="text-[10px] uppercase text-ink-faint">{a.status}</span>
                    {stoppable && (
                      <button
                        onClick={() => stop(a.id)}
                        className="text-ink-faint transition hover:text-rose-300"
                        aria-label="Stop agent"
                        title="Stop this agent"
                      >
                        &#9632;
                      </button>
                    )}
                  </div>
                </div>
                <div className="mt-0.5 text-[10px] text-ink-faint">
                  {relativeTime(a.started)}
                  {a.model ? ` · ${a.model}` : ''}
                </div>
                {events.length > 0 && (
                  <ul className="mt-1.5 flex flex-col gap-0.5 border-t border-border pt-1.5">
                    {events.slice(-3).map((e, i) => (
                      <li key={i} className="truncate text-[10px] text-ink-faint">
                        {eventLine(e)}
                      </li>
                    ))}
                  </ul>
                )}
                {a.status === 'done' && a.result && (
                  <div className="mt-1.5 truncate text-[10px] text-ink-dim">{a.result}</div>
                )}
                {a.status === 'error' && a.error && (
                  <div className="mt-1.5 truncate text-[10px] text-rose-300">{a.error}</div>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Browser-use embedded view — live Playwright/Chromium CDP screencast.
//
// Owner brief items 5-6: this session is now headless by default (no OS
// window pops up), it's the SAME session the model's own `browser` tool
// drives in chat (see pleiades/tools/browser.py + pleiades/webui/
// browser_view.py), and it offers two explicit, human-triggered escape
// hatches this component wires up: "Pop out" (switches the running session
// to a real headed OS window in place -- backend endpoint/function names
// stayed `open-separate`/`browserViewOpenSeparate`, only the visible label
// changed) and "Expand" (a larger in-app view, requesting a bigger
// server-side viewport via resize() rather than just upscaling a blurry
// small image).
//
// fix/browser-popout-border pass (see the `expanded &&` block below):
// two owner complaints, addressed together since fixing #2 changes the
// shape of the container #1's fix has to fit inside.
//   1) "thick border only on top and bottom" -- the frame's own
//      aspect-ratio-preserving box (aspectRatio: viewport.width/height)
//      was never actually broken in isolation; the bug was the ancestor
//      box it had to fit inside. The old modal card was a *fixed*
//      rectangle (`max-w-5xl` x `max-h-[90vh]`, ~1024px x 90% of viewport
//      height) with no relationship to the frame's own ratio, so on any
//      reasonably wide window the 1024px width cap bit before the 90vh
//      height cap did, leaving the frame width-bound with real vertical
//      space unused inside that fixed card -- the asymmetric top/bottom
//      bars reported (thin left/right margin from the card padding,
//      thick top/bottom from the unused height). The fix removes the
//      fixed-size ancestor entirely: the card is now `w-fit`/content-
//      sized on both axes, so it always hugs exactly whatever size the
//      frame's own aspect-ratio box computes, no leftover space possible
//      inside the card on either axis. The frame's max-width/max-height
//      are expressed directly in vw/vh/rem (not percentages of the
//      card, which would be circular once the card shrinks to content),
//      so this holds for any real viewport.width/height the browser
//      session resizes to, not just today's 1600x1000/1280x800 values.
//   2) The expanded view previously was a `fixed inset-0` backdrop modal
//      covering the entire window, chat column included. Per owner
//      request ("pop out the live chat onto the side... so you can see
//      the model thing and do, and can interject if needed"), it's now
//      a right-anchored drawer: a `fixed`, `pointer-events-none`
//      positioner (inset-y-3/right-3, no backdrop, no click-to-close)
//      that vertically centers the actual card, which is
//      `pointer-events-auto`. The chat column ChatView renders to this
//      panel's left -- messages, streaming draft, and the live Composer
//      -- stays fully visible and clickable the entire time the drawer
//      is open, so the owner can watch the browser and interject in
//      chat at the same time instead of one eclipsing the other.
// --------------------------------------------------------------------------

/** Map a pointer/keyboard-capable client rect to the backend's fixed
 * viewport pixel space (frames are always rendered at that resolution
 * regardless of how large the <img> is drawn on screen). */
function toViewportCoords(
  e: { clientX: number; clientY: number },
  el: HTMLElement,
  viewport: { width: number; height: number }
): { x: number; y: number } {
  const rect = el.getBoundingClientRect()
  const scaleX = viewport.width / rect.width
  const scaleY = viewport.height / rect.height
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
}

const EXPANDED_VIEWPORT = { width: 1600, height: 1000 }
const DEFAULT_VIEWPORT = { width: 1280, height: 800 }

function BrowserViewSection({
  base,
  character,
  active
}: {
  base: string
  character: string
  active: boolean
}): React.JSX.Element {
  const [status, setStatus] = useState<BrowserViewStatus | null>(null)
  const [urlInput, setUrlInput] = useState('')
  const [interactive, setInteractive] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  const expandedImgRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const frameUrlRef = useRef<string | null>(null)
  const [frameSrc, setFrameSrc] = useState<string | null>(null)

  const closeSocket = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  // Poll status even when the socket isn't open, so a session started/
  // stopped from elsewhere (the model's own browser tool, or a backend
  // restart) is reflected here live.
  useEffect(() => {
    if (!active || !character) return
    let cancelled = false
    const poll = async (): Promise<void> => {
      try {
        const s = await browserViewStatus(base, character)
        if (!cancelled) setStatus(s)
      } catch {
        // backend not reachable / no session yet — leave last-known status
      }
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [base, character, active])

  // Own the WebSocket lifecycle: connect while a session is running/
  // starting, tear down on stop/unmount/character switch.
  useEffect(() => {
    if (!active || !character) return
    if (!status || status.status === 'stopped') return

    const ws = new WebSocket(browserViewWsUrl(base, character))
    ws.binaryType = 'blob'
    wsRef.current = ws
    ws.onmessage = (evt) => {
      if (typeof evt.data === 'string') {
        try {
          const msg = JSON.parse(evt.data)
          if (msg.type === 'status') setStatus(msg)
        } catch {
          // ignore
        }
        return
      }
      const blob = evt.data as Blob
      const url = URL.createObjectURL(blob)
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
      frameUrlRef.current = url
      setFrameSrc(url)
    }
    ws.onerror = () => setError('Live view connection error.')
    return () => {
      ws.close()
      if (wsRef.current === ws) wsRef.current = null
    }
    // Reconnect whenever the session's status transitions (e.g. stopped -> starting).
  }, [base, character, active, status?.status])

  useEffect(
    () => () => {
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
    },
    []
  )

  // Model calls the chat browser tool -> goto starts the session headless
  // -> this component's status poll picks it up within ~3s and the WS
  // effect above connects automatically, exactly like a human-triggered
  // start. Nothing special is needed here for that path to "just show up".

  const start = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      const s = await browserViewStart(base, character, urlInput || 'https://example.com')
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const stop = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      closeSocket()
      setFrameSrc(null)
      setInteractive(false)
      setExpanded(false)
      const s = await browserViewStop(base, character)
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const navigate = async (): Promise<void> => {
    if (!urlInput.trim()) return
    setBusy(true)
    setError(null)
    try {
      await browserViewNavigate(base, character, urlInput.trim())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const openSeparate = async (): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      const s = await browserViewOpenSeparate(base, character)
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const toggleExpand = async (): Promise<void> => {
    const next = !expanded
    setExpanded(next)
    if (status?.status !== 'running') return
    try {
      const target = next ? EXPANDED_VIEWPORT : DEFAULT_VIEWPORT
      const s = await browserViewResize(base, character, target.width, target.height)
      setStatus(s)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const sendInput = (evt: Record<string, unknown>): void => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(evt))
    }
  }

  const viewport = status?.viewport || DEFAULT_VIEWPORT

  const onMouseEvent = (
    ref: React.RefObject<HTMLImageElement | null>,
    type: 'mousePressed' | 'mouseMoved' | 'mouseReleased',
    e: React.MouseEvent<HTMLImageElement>
  ): void => {
    if (!interactive || !ref.current) return
    const { x, y } = toViewportCoords(e, ref.current, viewport)
    sendInput({ kind: 'mouse', type, x, y, button: 'left', clickCount: 1 })
  }

  const onWheel = (
    ref: React.RefObject<HTMLImageElement | null>,
    e: React.WheelEvent<HTMLImageElement>
  ): void => {
    if (!interactive || !ref.current) return
    const { x, y } = toViewportCoords(e, ref.current, viewport)
    sendInput({ kind: 'mouse', type: 'mouseWheel', x, y, deltaX: e.deltaX, deltaY: e.deltaY })
  }

  const onKey = (type: 'keyDown' | 'keyUp', e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (!interactive) return
    e.preventDefault()
    const printable = e.key.length === 1 ? e.key : undefined
    sendInput({
      kind: 'key',
      type,
      key: e.key,
      code: e.code,
      keyCode: e.keyCode,
      text: type === 'keyDown' ? printable : undefined
    })
  }

  const running = status?.status === 'running'
  const starting = status?.status === 'starting'

  const frameView = (
    imgRefToUse: React.RefObject<HTMLImageElement | null>,
    big: boolean
  ): React.JSX.Element => (
    <div
      tabIndex={interactive ? 0 : -1}
      onKeyDown={(e) => onKey('keyDown', e)}
      onKeyUp={(e) => onKey('keyUp', e)}
      className={`relative overflow-hidden rounded-lg bg-black/40 outline-none ${big ? 'h-full w-full' : ''}`}
    >
      {frameSrc ? (
        <img
          ref={imgRefToUse}
          src={frameSrc}
          alt="Live browser view"
          draggable={false}
          onMouseDown={(e) => onMouseEvent(imgRefToUse, 'mousePressed', e)}
          onMouseUp={(e) => onMouseEvent(imgRefToUse, 'mouseReleased', e)}
          onMouseMove={(e) => onMouseEvent(imgRefToUse, 'mouseMoved', e)}
          onWheel={(e) => onWheel(imgRefToUse, e)}
          className={`w-full select-none ${big ? 'h-full object-contain' : ''} ${
            interactive ? 'cursor-crosshair' : 'cursor-default'
          }`}
        />
      ) : (
        <div className="flex h-40 items-center justify-center text-xs text-ink-faint">
          {starting ? 'Starting browser…' : 'Waiting for the first frame…'}
        </div>
      )}
    </div>
  )

  // Shared control row -- rendered identically in the compact panel body
  // and inside the popped-out/expanded modal below. Previously the modal
  // only rendered a URL label plus a lone close (x) button around the
  // frame, so the interact toggle, the pop-out button, and Stop -- all
  // defined only in the compact JSX further down -- were unreachable once
  // expanded (the modal's opaque backdrop covers the compact view, and
  // clicking it just collapses the modal again). `inModal` only changes
  // the expand/collapse button's label; every control is the exact same
  // state/handler in both places, so toggling interaction (or anything
  // else) works identically whether popped out or not.
  const controlsRow = (inModal: boolean): React.JSX.Element => (
    <div className="flex flex-wrap items-center gap-1.5">
      <button
        onClick={() => setInteractive((v) => !v)}
        className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition ${
          interactive ? 'bg-accent text-white' : 'bg-bg-300 text-ink-dim hover:text-ink'
        }`}
        title="Forward clicks/keys from this view into the real page"
      >
        {interactive ? 'Interaction: ON' : 'Enter interaction mode'}
      </button>
      <button
        onClick={toggleExpand}
        disabled={!running}
        className="rounded-lg bg-bg-300 px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-bg-400 hover:text-ink disabled:opacity-40"
        title={
          inModal
            ? 'Close this pop-out and return to the small view'
            : 'Open a larger view within the app'
        }
      >
        {inModal ? 'Close pop-out' : 'Expand'}
      </button>
      <button
        onClick={openSeparate}
        disabled={busy || !running || !!status?.headed}
        className="rounded-lg bg-bg-300 px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-bg-400 hover:text-ink disabled:opacity-40"
        title="Switch this session to a real OS window (same session, same login state)"
      >
        {status?.headed ? 'Popped out: on-screen' : 'Pop out'}
      </button>
      <button
        onClick={stop}
        disabled={busy}
        className="ml-auto rounded-lg px-2.5 py-1 text-[11px] text-ink-dim transition hover:bg-rose-500/20 hover:text-rose-300"
      >
        Stop session
      </button>
    </div>
  )

  return (
    <div>
      {!running && !starting && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-ink-faint">
            Live, embeddable browser session for {character || 'this character'} — the same session
            the model&rsquo;s own browser tool drives in chat. Hidden by default (headless); use
            &ldquo;Pop out&rdquo; below to pop a real window once it&rsquo;s running.
          </p>
          <button
            onClick={start}
            disabled={busy || !character}
            className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? 'Starting…' : 'Start browser session'}
          </button>
        </div>
      )}

      {(running || starting) && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1.5">
            <input
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && navigate()}
              placeholder={status?.url || 'https://…'}
              className="min-w-0 flex-1 rounded-lg bg-bg-surface px-2 py-1 text-xs text-ink placeholder:text-ink-faint focus:outline-none"
            />
            <button
              onClick={navigate}
              disabled={busy}
              className="flex-none rounded-lg bg-bg-300 px-2 py-1 text-xs text-ink transition hover:bg-bg-400 disabled:opacity-40"
            >
              Go
            </button>
          </div>

          {frameView(imgRef, false)}

          {controlsRow(false)}
          {interactive && (
            <p className="text-[10px] text-ink-faint">
              Click the view to focus it, then your clicks/scroll/keys go to the real page.
            </p>
          )}
        </div>
      )}

      {status?.status === 'error' && (
        <div className="mt-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-2.5 py-2 text-xs text-rose-300">
          {status.error}
        </div>
      )}
      {error && <div className="mt-2 text-xs text-rose-300">{error}</div>}

      {expanded && (
        // Right-anchored drawer, not a full-bleed backdrop modal: chat
        // (rendered by ChatView to this panel's left -- messages,
        // streaming draft, and the live Composer) stays visible and
        // clickable the whole time this is open, so the owner can watch
        // the browser and interject in chat at the same time instead of
        // one eclipsing the other.
        //
        // This outer positioner is `fixed` with only inset-y-3/right-3
        // set (no `left`), which per CSS gives it auto width (shrinks to
        // its child's own width) while its height *does* get stretched to
        // fill top-to-bottom (both insets specified) -- that stretch is
        // intentional, it's just how `items-center` below gets something
        // to vertically center the actual card within. The stretched
        // region itself is invisible (no bg) and `pointer-events-none`,
        // so it never blocks clicks into chat behind it; only the visible
        // card (`pointer-events-auto`) is interactive.
        <div className="pointer-events-none fixed inset-y-3 right-3 z-40 flex items-center">
          {/* The card is `w-fit`/content-sized on both axes -- no
              max-w-5xl-style fixed box for the video to float inside with
              leftover background showing as a border. Its actual size
              comes entirely from the video wrapper below, which is given
              explicit vw/vh/rem-based max-width/max-height (not
              percentages of this card, which would be circular since the
              card's own size is "shrink to content"). The video wrapper
              picks the largest box matching the live frame's real
              viewport.width/height ratio that fits those absolute bounds,
              and the card then hugs exactly that -- header and
              controlsRow above/below, zero extra padding-as-border on any
              side. `max-h-[calc(100vh-1.5rem)]` is just a hard ceiling
              matching this positioner's own inset-y-3 margin, so a
              mis-estimate in the reserved-chrome math below can only
              clip/scroll, never overflow past the viewport. */}
          <div className="pointer-events-auto flex max-h-[calc(100vh-1.5rem)] w-fit max-w-full flex-col gap-3 rounded-2xl border border-border bg-bg-100 p-4 shadow-2xl">
            <div className="flex flex-none items-center justify-between gap-3">
              <span className="truncate text-sm text-ink-dim">{status?.url || character}</span>
              <button
                onClick={() => toggleExpand()}
                className="rounded-lg px-2 py-1 text-ink-dim transition hover:bg-bg-surface-hover hover:text-ink"
                aria-label="Close pop-out"
              >
                &#10005;
              </button>
            </div>
            {/* Reserved chrome subtracted from 100vh here: 1.5rem
                positioner margin (inset-y-3 x2) + 2rem card padding
                (p-4 x2) + ~2rem header row + ~2.25rem controlsRow row
                (buttons wrap onto one line at any width this drawer
                actually reaches) + 1.5rem for the two gap-3 gutters
                between header/video/controls ~= 9.25rem; rounded up to
                10rem so the video always comes in slightly under budget
                rather than right at the edge. Width budget mirrors the
                same min(64vw, 72rem) drawer-size intent from the layout
                pass, minus this card's own 2rem of horizontal p-4
                padding. Both are plain viewport-relative math, not
                today's specific window size, so this holds for any real
                page dimensions the browser session resizes to (the ratio
                itself -- viewport.width/height -- already comes from
                live state, not a hardcoded constant). */}
            <div className="flex min-h-0 items-center justify-center overflow-hidden">
              <div
                style={{
                  aspectRatio: `${viewport.width} / ${viewport.height}`,
                  maxWidth: 'calc(min(64vw, 72rem) - 2rem)',
                  maxHeight: 'calc(100vh - 10rem)'
                }}
              >
                {frameView(expandedImgRef, true)}
              </div>
            </div>
            {controlsRow(true)}
          </div>
        </div>
      )}
    </div>
  )
}

export default RightPanel
