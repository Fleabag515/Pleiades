#Requires -Version 5.1
<#
  Pleiades one-line installer  (Windows)

    irm https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.ps1 | iex

  With options, run the downloaded scriptblock yourself:
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.ps1))) -Gpu gpu -Dir C:\Pleiades

  Parameters:
    -Dir DIR        install location          (default: $HOME\Pleiades)
    -Branch NAME    git branch               (default: main)
    -Gpu auto|cpu|gpu   llama-cpp-python build (default: auto)
    -Core           core only (skip browser, SearXNG, Discord)
    -NoBrowser      skip Camoufox
    -NoSearxng      skip SearXNG (Docker)
    -NoDiscord      skip the Discord extra
#>
[CmdletBinding()]
param(
  [string]$Dir    = "$HOME\Pleiades",
  [string]$Branch = "main",
  [string]$Repo   = "https://github.com/Fleabag515/Pleiades.git",
  [ValidateSet("auto","cpu","gpu")][string]$Gpu = "auto",
  [switch]$Core,
  [switch]$NoBrowser,
  [switch]$NoSearxng,
  [switch]$NoDiscord
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
function Resolve-Gpu {
  switch ($Gpu) {
    "cpu" { return @{ Args = "";                 Desc = "CPU (forced)" } }
    "gpu" { return @{ Args = "-DGGML_CUDA=on";   Desc = "CUDA (forced)" } }
    default {
      if (Have nvidia-smi) {
        try { nvidia-smi -L *> $null; if ($LASTEXITCODE -eq 0) { return @{ Args = "-DGGML_CUDA=on"; Desc = "NVIDIA CUDA (auto)" } } } catch {}
      }
      return @{ Args = ""; Desc = "CPU (no GPU detected)" }
    }
  }
}

# ---- repo ------------------------------------------------------------------
function Clone-Repo {
  if (Test-Path (Join-Path $Dir ".git")) {
    Say "Updating existing checkout at $Dir"
    git -C $Dir fetch origin $Branch 2>$null
    # Hard-reset to the remote branch so shallow clones update cleanly; untracked
    # files (e.g. your .env) are preserved.
    git -C $Dir reset --hard "origin/$Branch" 2>$null
    if ($LASTEXITCODE -ne 0) { git -C $Dir pull --ff-only 2>$null }
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
        Warn "GPU build failed (missing CUDA toolkit?). Retrying with a CPU build."
        $env:CMAKE_ARGS = ""
        & $venvPy -m pip install -e $spec
        if ($LASTEXITCODE -ne 0) { Die "pip install failed." }
        $plan.Desc = "CPU (GPU build failed)"
      } else { Die "pip install failed." }
    }
  } finally { Pop-Location; $env:CMAKE_ARGS = "" }
}

function Install-Anamnesis {
  if (-not (Have npm)) { Warn "npm missing; skipping Anamnesis. Install Node, then 'npm i -g anamnesis'."; return }
  Say "Installing Anamnesis (memory proxy)"
  npm install -g anamnesis 2>$null
  if ($LASTEXITCODE -ne 0) { Warn "Could not install anamnesis globally; run 'npm i -g anamnesis' yourself." }
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
# insensitive, so $gpu IS the [ValidateSet("auto","cpu","gpu")]$Gpu parameter.
# Validation attributes stay bound to the variable for the whole scope, so
# assigning Resolve-Gpu's hashtable to it re-triggers ValidateSet and aborts
# the install ("System.Collections.Hashtable is not a valid value for Gpu").
$gpuPlan = Resolve-Gpu
Clone-Repo
Install-Pkg $py $gpuPlan
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
