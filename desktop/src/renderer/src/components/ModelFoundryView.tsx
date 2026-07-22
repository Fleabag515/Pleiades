/** Placeholder for the Model Foundry section (Phase G builds the real
 * search/download/manage UI here). This phase only wires up the
 * navigation shell: a top-level section that sits alongside Chats,
 * mirroring where Claude Desktop puts its "Code" section. */
function ModelFoundryView(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-bg-app px-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/15 text-2xl text-accent">
        ⬡
      </div>
      <h1 className="text-xl font-semibold text-ink-bright">Model Foundry</h1>
      <p className="max-w-sm text-sm text-ink-dim">
        Search, download, and manage local GGUF models from here. This section is coming
        together in the next phase &mdash; for now it&apos;s just wired into the navigation.
      </p>
    </div>
  )
}

export default ModelFoundryView
