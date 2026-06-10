# Hardware autodetect, smart offload, smart fetch, and model-callable everything

Date: 2026-06-09 · Status: approved

## Problem

Pleiades builds llama-cpp-python with the right GPU backend at install time, but
runtime defaults to `n_gpu_layers=0` — every model runs CPU-only unless the user
hand-tunes layer counts. Picking a quantization is also manual, and several
capabilities (model management, profile setup) are user-only CLI/UI actions.
Pleiades is meant to be a self-building inference workstation: after the
installer, the only step should be "use it".

## Design

### 1. `pleiades/hardware.py` — detection + placement planner

- `detect() -> Hardware`: NVIDIA via `nvidia-smi`, AMD via `rocm-smi` or sysfs
  (`/sys/class/drm/card*/device/mem_info_vram_*`, no ROCm tools needed), Apple
  Silicon via platform + unified memory, plus system RAM and CPU cores.
- `read_gguf_meta(path) -> GGUFMeta`: pure-Python GGUF header parser (layer
  count, embedding size, KV heads, quant type). Falls back to size-based
  estimates on parse failure.
- `plan(meta, n_ctx, hw) -> Plan{n_gpu_layers, reason}`: full offload (-1) if
  model + KV cache + safety margin fits VRAM (unified memory budget on Apple);
  otherwise the max layer count that fits; otherwise CPU. Heuristic with
  margin; an explicit integer always overrides.
- `pick_quant(files, hw, n_ctx)`: choose the best GGUF quant that fits
  (prefer Q8_0 > Q6_K > Q5_K_M > Q4_K_M > … for full GPU offload, falling back
  to fits-in-RAM for CPU). Multi-part GGUFs are grouped and summed.

### 2. `n_gpu_layers="auto"` (new default)

Settings, the model registry, CLI, and web UI accept `"auto"` or an int.
Resolution happens at server launch so plans adapt to hardware changes.
`pleiades hw` and `GET /api/hardware` report what was detected and why.

### 3. Smart downloader — `pleiades model fetch <hf-repo>`

Lists GGUF files via the HuggingFace API (`/api/models/{repo}/tree/main`),
picks a quant with `pick_quant` (or `--quant` override), streams the file(s)
to `~/.pleiades/models/`, registers with auto layers. `pleiades model search
<query>` finds GGUF repos. Uses httpx; no new dependencies.

### 4. Model-callable tools

Shared implementations exposed both in the chat ToolBelt and the harness:
`hardware` (report + plan), `models` (list/start/stop/assign/search/fetch),
`profile` (read/update own email + persona + assigned model), `characters`
(list/create/adopt/delete — delete requires `confirm="delete <name>"`).
Harness side respects exec_policy for side-effecting actions.

### 5. Installer + UX

`ui` extra always installed; Vulkan build fallback for AMD without ROCm;
`pleiades` symlinked into `~/.local/bin`; closing message prints detected
hardware and says `pleiades ui`.

## Trade-offs

- Heuristic VRAM math over probe-by-loading: instant and dependency-free; the
  safety margin plus manual override covers estimation error.
- No local re-quantization: every useful quant already exists on HF.

## Testing

Unit tests with synthetic GGUF bytes, fake Hardware values, and fake HF file
lists; tool registration tests; no network in CI.
