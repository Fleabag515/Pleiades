import { useState } from 'react'
import ModelsTab from './ModelsTab'
import DownloadTab from './DownloadTab'
import CloudModelsTab from './CloudModelsTab'

interface ModelsSectionProps {
  base: string
}

type ModelsInnerTab = 'models' | 'download' | 'ollama-cloud' | 'openrouter'

const INNER_TABS: { key: ModelsInnerTab; label: string }[] = [
  { key: 'models', label: 'Models' },
  { key: 'download', label: 'Download' },
  { key: 'ollama-cloud', label: 'Ollama Cloud' },
  { key: 'openrouter', label: 'OpenRouter' }
]

/** Models section of Settings' left nav. Owns its own internal top tab bar
 * (separate from the left nav) with four tabs:
 *   - Models: locally registered GGUFs — start/stop, logs, rename, delete.
 *   - Download: acquire a new local model (HF search/quant-plan/download) —
 *     this is the former standalone "Foundry" top-level Settings tab,
 *     relocated here now that everything it did lives under Models.
 *   - Ollama Cloud / OpenRouter: browsing hosted models by provider,
 *     split out of the old combined cloud-browsing tab.
 */
function ModelsSection({ base }: ModelsSectionProps): React.JSX.Element {
  const [tab, setTab] = useState<ModelsInnerTab>('models')

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-1 border-b border-border">
        {INNER_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`rounded-t-lg px-3 py-2 text-sm font-medium transition ${
              tab === t.key ? 'border-b-2 border-accent text-ink-bright' : 'text-ink-dim hover:text-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'models' && <ModelsTab base={base} />}
      {tab === 'download' && <DownloadTab base={base} />}
      {tab === 'ollama-cloud' && <CloudModelsTab base={base} source="ollama" />}
      {tab === 'openrouter' && <CloudModelsTab base={base} source="openrouter" />}
    </div>
  )
}

export default ModelsSection
