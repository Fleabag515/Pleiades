#Requires -Version 5.1
<#
  Pleiades one-line installer  (Windows)

    irm https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.ps1 | iex

  With options, run the downloaded scriptblock yourself:
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.ps1))) -Gpu vulkan -Dir C:\Pleiades

  Parameters:
    -Dir DIR        install location          (default: $HOME\Pleiades)
    -Branch NAME    git branch               (default: main)
    -Gpu auto|cpu|gpu|cuda|vulkan   llama-cpp-python build (default: auto)
                    auto   detect: NVIDIA -> CUDA build; AMD/Intel -> CPU python
                           build + native Vulkan llama-server runtime (no SDK needed)
                    cuda   force the CUDA build ("gpu" is a legacy alias)
                    vulkan force a Vulkan build of llama-cpp-python (requires the
                           Vulkan SDK to compile; most AMD users want "auto" instead)
    -Core           core only (skip browser, SearXNG, Discord)
    -NoBrowser      skip Camoufox
    -NoSearxng      skip SearXNG (Docker)
    -NoDiscord      skip the Discord extra
    -NoNativeRuntime  skip 'pleiades runtime install' (native llama-server)
#>
[CmdletBinding()]
param(
  [string]$Dir    = "$HOME\Pleiades",
  [string]$Branch = "main",
  [string]$Repo   = "https://github.com/Fleabag515/Pleiades.git",
  [ValidateSet("auto","cpu","gpu","cuda","vulkan")][string]$Gpu = "auto",
  [switch]$Core,
  [switch]$NoBrowser,
  [switch]$NoSearxng,
  [switch]$NoDiscord,
  [switch]$NoNativeRuntime
)
$ErrorActionPreference = "Stop"

if ($Core) { $NoBrowser = $true; $NoSearxng = $true; $NoDiscord = $true }

function Say($m)  { Write-Host "==> $m" -ForegroundColor Green }
function Info($m) { Write-Host "    $m" -ForegroundColor Gray }
function Warn($m) { Write-Host "!!  $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "xx  $m" -ForegroundColor Red; exit 1 }
function Have($c) { $null = Get-Command $c -ErrorAction SilentlyContinue; return $? }

function Refresh-Path {
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
              [System.Environment]::GetEnvironmentVariable("Path","User")
}

function Winget-Install($id, $label) {
  if (-not (Have winget)) {
    Die "winget not found. Install '$label' manually (or get App Installer from the Microsoft Store), then re-run."
  }
  Say "Installing $label"
  winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements | Out-Null
  Refresh-Path
}

# ---- prerequisites (hybrid: auto-install runtimes, guide for Docker) -------
function Ensure-Git {
  if (-not (Have git)) { Winget-Install "Git.Git" "Git" }
}

function Get-PyBin {
  foreach ($c in @("python","python3","py")) {
    if (Have $c) {
      try {
        & $c -c "import sys; raise SystemExit(0 if sys.version_info[:2]>=(3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { return $c }
      } catch {}
    }
  }
  return $null
}

function Ensure-Python {
  $py = Get-PyBin
  if (-not $py) {
    Winget-Install "Python.Python.3.12" "Python 3.12"
    $py = Get-PyBin
  }
  if (-not $py) { Die "Python >=3.10 unavailable after install attempt." }
  Info ("Using " + (& $py --version 2>&1))
  return $py
}

function Ensure-Node {
  if (-not (Have npm)) { Winget-Install "OpenJS.NodeJS.LTS" "Node.js LTS" }
}

function Check-Docker {
  if ((Have docker)) {
    try { docker info *> $null; if ($LASTEXITCODE -eq 0) { return $true } } catch {}
    Warn "Docker is installed but not running. Start Docker Desktop, then 'pleiades search up'."
    return $false
  }
  Warn "Docker not found — it powers the built-in SearXNG web search."
  Info "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
  Info "SearXNG will be skipped now; bring it up later with 'pleiades search up'."
  return $false
}

# ---- GPU detection ---------------------------------------------------------
function Get-GpuVendor {
  # WMI sees every display adapter; nvidia-smi only exists for NVIDIA.
  try {
    $names = (Get-CimInstance Win32_VideoController -ErrorAction Stop |
              Select-Object -ExpandProperty Name) -join "; "
  } catch { $names = "" }
  if (Have nvidia-smi) {
    try { nvidia-smi -L *> $null; if ($LASTEXITCODE -eq 0) { return @{ Vendor = "nvidia"; Names = $names } } } catch {}
  }
  if ($names -match "NVIDIA|GeForce|Quadro")        { return @{ Vendor = "nvidia"; Names = $names } }
  if ($names -match "AMD|Radeon|ATI")               { return @{ Vendor = "amd";    Names = $names } }
  if ($names -match "Intel.*(Arc|Iris|Graphics)")   { return @{ Vendor = "intel";  Names = $names } }
  return @{ Vendor = "none"; Names = $names }
}

function Resolve-Gpu {
  # Returns @{ Args; Desc; NativeFirst } — Args drives the llama-cpp-python
  # (CMAKE) build; NativeFirst additionally installs the prebuilt native
  # llama-server runtime ('pleiades runtime install'), which is how AMD/Intel
  # GPUs get acceleration on Windows: the official Vulkan binaries need no
  # SDK and no ROCm — they just use the GPU driver. Autofit prefers the
  # native runtime automatically once it is installed.
  switch ($Gpu) {
    "cpu"    { return @{ Args = "";                Desc = "CPU (forced)";   NativeFirst = $false } }
    "gpu"    { return @{ Args = "-DGGML_CUDA=on";  Desc = "CUDA (forced)";  NativeFirst = $false } }
    "cuda"   { return @{ Args = "-DGGML_CUDA=on";  Desc = "CUDA (forced)";  NativeFirst = $false } }
    "vulkan" {
      if (-not $env:VULKAN_SDK) {
        Warn "-Gpu vulkan compiles llama-cpp-python against the Vulkan SDK, which wasn't found (VULKAN_SDK unset)."
        Info "Most AMD/Intel users should use -Gpu auto instead: it installs the prebuilt native"
        Info "Vulkan llama-server runtime, which needs only your normal GPU driver."
      }
      return @{ Args = "-DGGML_VULKAN=on"; Desc = "Vulkan (forced)"; NativeFirst = $true }
    }
    default {
      $det = Get-GpuVendor
      switch ($det.Vendor) {
        "nvidia" { return @{ Args = "-DGGML_CUDA=on"; Desc = "NVIDIA CUDA (auto)"; NativeFirst = $false } }
        "amd"    {
          Info "AMD GPU detected: $($det.Names)"
          Info "ROCm on Windows isn't a practical llama.cpp target — using the native Vulkan runtime instead."
          return @{ Args = ""; Desc = "AMD via native Vulkan runtime (python engine on CPU)"; NativeFirst = $true }
        }
        "intel"  {
          Info "Intel GPU detected: $($det.Names)"
          return @{ Args = ""; Desc = "Intel via native Vulkan runtime (python engine on CPU)"; NativeFirst = $true }
        }
        default  { return @{ Args = ""; Desc = "CPU (no GPU detected)"; NativeFirst = $false } }
      }
    }
  }
}

# ---- repo ------------------------------------------------------------------
function Clone-Repo {
  if (Test-Path (Join-Path $Dir ".git")) {
    Say "Updating existing checkout at $Dir"
    # 2>&1 | Out-Null: merges stderr into stdout then discards both; avoids
    # $ErrorActionPreference=Stop treating git's informational stderr as a
    # terminating error (the 2>$null form is unreliable in irm|iex context).
    git -C $Dir fetch origin $Branch 2>&1 | Out-Null
    # Hard-reset to the remote branch so shallow clones update cleanly; untracked
    # files (e.g. your .env) are preserved.
    git -C $Dir reset --hard "origin/$Branch" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { git -C $Dir pull --ff-only 2>&1 | Out-Null }
  } else {
    Say "Cloning Pleiades into $Dir"
    git clone --branch $Branch $Repo $Dir
  }
}

function Get-Extras {
  $e = @()
  $e += "ui"   # the control panel (pleiades ui) is always available
  if (-not $NoBrowser) { $e += "browser" }
  if (-not $NoDiscord) { $e += "discord" }
  if ($e.Count -eq 0) { return "" } else { return ($e -join ",") }
}

function Install-Pkg($py, $plan) {
  Say "Creating virtual environment (.venv)"
  & $py -m venv (Join-Path $Dir ".venv")
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  & $venvPy -m pip install --upgrade pip wheel setuptools | Out-Null

  $extras = Get-Extras
  $spec = if ($extras) { ".[$extras]" } else { "." }
  Say "Installing Pleiades — $($plan.Desc) build of llama-cpp-python (this can take several minutes)"
  Push-Location $Dir
  try {
    $env:CMAKE_ARGS = $plan.Args
    & $venvPy -m pip install -e $spec
    if ($LASTEXITCODE -ne 0) {
      if ($plan.Args) {
        Warn "GPU build failed (missing toolkit/SDK?). Retrying with a CPU build."
        $env:CMAKE_ARGS = ""
        & $venvPy -m pip install -e $spec
        if ($LASTEXITCODE -ne 0) { Die "pip install failed." }
        $plan.Desc = "CPU python engine (GPU build failed)"
        # The prebuilt native runtime can still give this machine GPU speed.
        $plan.NativeFirst = $true
      } else { Die "pip install failed." }
    }
  } finally { Pop-Location; $env:CMAKE_ARGS = "" }
}

function Install-NativeRuntime($plan) {
  if ($NoNativeRuntime -or -not $plan.NativeFirst) { return }
  Say "Installing the native llama-server runtime (prebuilt, GPU via Vulkan/CUDA — no SDK needed)"
  $venvPleiades = Join-Path $Dir ".venv\Scripts\pleiades.exe"
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  try {
    if (Test-Path $venvPleiades) { & $venvPleiades runtime install }
    else { & $venvPy -m pleiades.cli runtime install }
    if ($LASTEXITCODE -ne 0) { throw "runtime install exited $LASTEXITCODE" }
    Info "Models will run on the native runtime automatically (pleiades hw shows the plan)."
  } catch {
    Warn "Native runtime install failed: $($_.Exception.Message)"
    Info "Run it later from the venv:  pleiades runtime install"
  }
}

$AnamnesisSpec = "github:Fleabag515/anamnesis"   # the plain "anamnesis" name on the
# public npm registry is squatted by an unrelated, unmaintained package (last
# published 2022, no `anamnesis` binary at all) — `npm install -g anamnesis`
# silently installs the wrong thing on every platform. Installing straight
# from source is the only spec that actually resolves to Fleabag515/anamnesis.

function Install-Anamnesis {
  if (-not (Have npm)) { Warn "npm missing; skipping Anamnesis. Install Node, then 'npm i -g $AnamnesisSpec'."; return }
  $installed = ""
  $latest = ""
  try {
    $out = (npm ls -g anamnesis --depth=0 2>$null | Out-String)
    if ($out -match "anamnesis@([0-9][^\s]*)") { $installed = $Matches[1] }
  } catch {}
  try { $latest = (npm view $AnamnesisSpec version 2>$null | Out-String).Trim() } catch {}

  if ($installed -and $latest -and ($installed -eq $latest)) {
    Say "Anamnesis $installed already installed and current — skipping"
    return
  }
  if ($installed) {
    Say "Updating Anamnesis $installed -> $(if ($latest) { $latest } else { 'latest' })"
  } else {
    Say "Installing Anamnesis (memory proxy)"
  }
  Info "npm pulls large native deps here (node-llama-cpp, transformers) — a few minutes of progress output is normal."
  npm install -g $AnamnesisSpec --no-audit --no-fund --onnxruntime-node-install-cuda=skip
  if ($LASTEXITCODE -ne 0) { Warn "Could not install anamnesis globally; run 'npm i -g $AnamnesisSpec' yourself." }
}

function Fetch-Browser($py) {
  if ($NoBrowser) { return }
  Say "Fetching Camoufox browser"
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  & $venvPy -m camoufox fetch
  if ($LASTEXITCODE -ne 0) { Warn "camoufox fetch failed; run 'python -m camoufox fetch' later." }
}

function Make-Env {
  $envFile = Join-Path $Dir ".env"
  $example = Join-Path $Dir ".env.example"
  if ((-not (Test-Path $envFile)) -and (Test-Path $example)) {
    Copy-Item $example $envFile
    Say "Created .env — set PLEIADES_MODEL_PATH to your .gguf before chatting"
  }
}

function Start-Searxng($dockerOk) {
  if ($NoSearxng) { return }
  if (-not $dockerOk) { Warn "Skipping SearXNG (Docker not ready). Later: 'pleiades search up'."; return }
  Say "Starting SearXNG (docker compose)"
  Push-Location $Dir
  try { docker compose up -d searxng } catch { Warn "Could not start SearXNG." } finally { Pop-Location }
}

# ---- run -------------------------------------------------------------------
Say "Pleiades installer  (Windows)"
Ensure-Git
$py = Ensure-Python
Ensure-Node
$dockerOk = Check-Docker
# NOTE: deliberately NOT named $gpu — PowerShell variable names are case-
# insensitive, so $gpu IS the [ValidateSet(...)]$Gpu parameter. Validation
# attributes stay bound to the variable for the whole scope, so assigning
# Resolve-Gpu's hashtable to it re-triggers ValidateSet and aborts the install
# ("System.Collections.Hashtable is not a valid value for Gpu").
$gpuPlan = Resolve-Gpu
Clone-Repo
Install-Pkg $py $gpuPlan
Install-NativeRuntime $gpuPlan
Install-Anamnesis
Fetch-Browser $py
Make-Env
Start-Searxng $dockerOk

Write-Host ""
Say "Pleiades installed at $Dir  ($($gpuPlan.Desc))"
@"

  Next steps:
    cd "$Dir"
    notepad .env                       # set PLEIADES_MODEL_PATH to a .gguf model file
    .\.venv\Scripts\Activate.ps1
    pleiades new alice                 # create a character (memory + email + vault + browser)
    pleiades chat alice                # talk to it
    pleiades discord alice             # host it as a Discord bot

  Docs: README.md   .   Full build brief: CLAUDE.md
"@ | Write-Host
