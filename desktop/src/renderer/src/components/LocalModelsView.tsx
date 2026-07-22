import ModelsTab from './settings/ModelsTab'

interface LocalModelsViewProps {
  base: string
}

/** Top-level "Local Models" section (sibling of Chats and Foundry in the
 * sidebar switch) — reuses the same start/stop/logs lifecycle list as
 * Settings' Models tab, but as its own full main-pane view instead of
 * being lumped into the model-download flow. Split out per the owner's
 * explicit ask: "already-registered/running models" needs to read as
 * organizationally separate from "download a new model", not just be a
 * subsection inside Model Foundry. */
function LocalModelsView({ base }: LocalModelsViewProps): React.JSX.Element {
  return (
    <div className="h-full w-full overflow-y-auto bg-bg-app px-6 py-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        <div>
          <h1 className="mb-1 text-lg font-semibold text-ink-bright">Local Models</h1>
          <p className="text-sm text-ink-dim">
            GGUF models already registered on this machine. Start or stop them and see their runtime state.
            Looking to add a new model? Switch to Foundry.
          </p>
        </div>
        <ModelsTab base={base} />
      </div>
    </div>
  )
}

export default LocalModelsView
