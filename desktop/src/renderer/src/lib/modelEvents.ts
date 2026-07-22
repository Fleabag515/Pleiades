type Listener = () => void

const listeners = new Set<Listener>()

/**
 * Rename-doesn't-refresh-live bug fix: root cause was that Composer's model
 * picker fetched `listModels()` exactly once per (base, character) via a
 * `useEffect` with no way to learn that ModelsTab (a totally separate,
 * simultaneously-mounted tree under SettingsPanel's overlay) had just
 * PUT a rename through `updateModel()`. Nothing ever told the already-fetched
 * `models` state in Composer to go stale, so it kept rendering the old name
 * until something remounted it (switching character re-runs that effect;
 * restarting the app obviously does too) -- exactly the symptom reported.
 *
 * Fix: a tiny module-level pub/sub. `api.ts` calls `notifyModelsChanged()`
 * right after any mutation that changes what a model displays as (rename,
 * delete, start, stop). Any mounted view that cares (right now: Composer)
 * subscribes here and refetches immediately, no matter where the mutation
 * came from or whether the two components share a parent below the app
 * root. Deliberately not a React context/store: the registry already has a
 * real source of truth on the backend (`GET /api/models`), so all the UI
 * needs is a "go re-ask" signal, not client-side state ownership.
 */
export function subscribeModelsChanged(fn: Listener): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

export function notifyModelsChanged(): void {
  for (const fn of listeners) fn()
}
