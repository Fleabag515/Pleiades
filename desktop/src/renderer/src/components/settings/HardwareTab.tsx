import { useEffect, useState } from 'react'
import { getHardware } from '../../lib/api'
import type { HardwareInfo, ModelPlan } from '../../lib/types'
import { formatGiB } from '../../lib/format'

interface HardwareTabProps {
  base: string
}

const STRATEGY_LABEL: Record<string, string> = {
  full_gpu: 'fits fully on GPU',
  moe_cpu: 'experts → RAM, rest → GPU',
  moe_partial: 'MoE hybrid split',
  layers: 'partial offload',
  cpu: 'CPU only'
}

function Meter({ label, used, total }: { label: string; used: number; total: number }): React.JSX.Element {
  const pct = total ? Math.min(100, Math.round((100 * used) / total)) : 0
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-ink-dim">{label}</span>
        <span className="text-ink-dim">
          {formatGiB(used)} / {formatGiB(total)} GiB · {pct}%
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-surface-hover">
        <div
          className={`h-full rounded-full ${pct > 85 ? 'bg-rose-400' : 'bg-accent'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function PlanCard({ plan }: { plan: ModelPlan }): React.JSX.Element {
  const onGpu = plan.n_gpu_layers === -1 ? plan.n_layers : Number(plan.n_gpu_layers)
  const pct = plan.n_layers ? Math.round((100 * onGpu) / plan.n_layers) : 0
  const pillStyle = plan.fits_fully
    ? 'bg-emerald-500/15 text-emerald-300'
    : plan.strategy?.startsWith('moe')
      ? 'bg-violet-500/15 text-violet-300'
      : onGpu
        ? 'bg-amber-500/15 text-amber-300'
        : 'bg-bg-surface-hover text-ink-dim'
  return (
    <div className="rounded-xl border border-border bg-bg-app p-3">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-sm font-medium text-ink">{plan.model}</span>
          {plan.moe && (
            <span className="flex-none rounded-full bg-violet-500/15 px-1.5 py-0.5 text-[10px] text-violet-300">
              MoE
            </span>
          )}
        </div>
        <span className={`flex-none rounded-full px-2 py-0.5 text-[11px] ${pillStyle}`}>
          {STRATEGY_LABEL[plan.strategy] ?? 'plan'}
        </span>
      </div>
      <div className="text-xs text-ink-dim">
        {onGpu}/{plan.n_layers || '?'} layers on GPU ({pct}%)
        {plan.est_tps ? ` · ~${plan.est_tps.toFixed(1)} tok/s est.` : ''}
      </div>
      {plan.reason && <div className="mt-1 text-xs text-ink-dim/80">{plan.reason}</div>}
    </div>
  )
}

/** Hardware tab: this machine's detected GPU(s)/VRAM, RAM, CPU, native
 * runtime state, and the planned GPU/CPU layer split for each registered
 * model. All values come straight from GET /api/hardware — no placeholders. */
function HardwareTab({ base }: HardwareTabProps): React.JSX.Element {
  const [hw, setHw] = useState<HardwareInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getHardware(base)
      .then(setHw)
      .catch((e) => setError((e as Error).message))
  }, [base])

  if (error) {
    return (
      <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
        {error}
      </div>
    )
  }

  if (!hw) {
    return <p className="text-sm text-ink-dim">Detecting hardware…</p>
  }

  return (
    <div className="flex flex-col gap-4">
      <section className="rounded-xl border border-border bg-bg-app p-4">
        <h3 className="mb-3 text-sm font-semibold text-ink">This machine</h3>
        <div className="flex flex-col gap-3">
          {hw.gpus.length === 0 && (
            <p className="text-xs text-ink-dim">
              No GPU detected — models run on CPU.
            </p>
          )}
          {hw.gpus.map((g, i) => (
            <div key={i} className="flex flex-col gap-1">
              <div className="flex items-center gap-2 text-xs">
                <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-300">{g.vendor}</span>
                <span className="text-ink">{g.name}</span>
              </div>
              <Meter label="VRAM in use" used={g.vram_total - g.vram_free} total={g.vram_total} />
            </div>
          ))}
          <Meter label="RAM in use" used={hw.ram_total - hw.ram_available} total={hw.ram_total} />
          <div className="flex items-center gap-2 text-xs text-ink-dim">
            <span className="rounded-full bg-bg-surface-hover px-2 py-0.5">{hw.cpu_threads} CPU threads</span>
            {hw.unified_memory && <span className="rounded-full bg-bg-surface-hover px-2 py-0.5">unified memory</span>}
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-border bg-bg-app p-4">
        <h3 className="mb-3 text-sm font-semibold text-ink">Inference runtime</h3>
        {hw.engine ? (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-300">Pleiades engine (default)</span>
            {hw.engine.backend && (
              <span className="rounded-full bg-bg-surface-hover px-2 py-0.5 text-ink-dim">{hw.engine.backend}</span>
            )}
            <span
              className={`rounded-full px-2 py-0.5 ${
                hw.engine.moe_offload ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'
              }`}
            >
              {hw.engine.moe_offload ? 'MoE expert offload: on' : 'MoE offload unsupported by this build'}
            </span>
            {hw.engine.vision && (
              <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-300">vision</span>
            )}
          </div>
        ) : hw.runtime?.native ? (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-300">native llama-server</span>
            {hw.runtime.version && (
              <span className="rounded-full bg-bg-surface-hover px-2 py-0.5 text-ink-dim">{hw.runtime.version}</span>
            )}
            <span
              className={`rounded-full px-2 py-0.5 ${
                hw.runtime.moe_offload ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'
              }`}
            >
              {hw.runtime.moe_offload ? 'MoE expert offload: on' : 'MoE offload unsupported by this build'}
            </span>
          </div>
        ) : (
          <p className="text-xs text-ink-dim">Python fallback server (no native runtime found).</p>
        )}
      </section>

      <section className="rounded-xl border border-border bg-bg-app p-4">
        <h3 className="mb-3 text-sm font-semibold text-ink">Per-model offload plan</h3>
        {hw.plans.length === 0 ? (
          <p className="text-xs text-ink-dim">No models registered.</p>
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {hw.plans.map((p) => (
              <PlanCard key={p.model} plan={p} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

export default HardwareTab
