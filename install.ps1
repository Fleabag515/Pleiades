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
    -NoBrowser      skip browser automation (Camoufox + desktop panel's Playwright/Chromium)
    -NoSearxng      skip SearXNG (local web search)
    -NoDiscord      skip the Discord extra
    -NoNativeRuntime  skip 'pleiades runtime install' (native llama-server)
    -NoDesktopApp   skip building the Electron desktop app
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
  [switch]$NoNativeRuntime,
  [switch]$NoDesktopApp
)
$ErrorActionPreference = "Continue"
# NOT "Stop" -- verified live on real Windows PowerShell 5.1 (the built-in
# powershell.exe every Windows machine has, and what `irm ... | iex` actually
# runs under): with "Stop", ANY native command that writes to stderr on a
# genuinely successful run -- git printing "Cloning into '...'", pip's own
# routine warnings, npm's routine notices, all completely normal -- gets
# converted into a terminating NativeCommandError and kills the whole
# installer before a single dependency is installed. This script already
# does its own explicit success/failure checking throughout (Die, `if
# ($LASTEXITCODE -ne 0)`, try/catch) for exactly this reason; "Stop" was
# fighting that design, not supporting it. Confirmed via PowerShell's own
# Language.Parser + a full real run on real Windows/AMD hardware: the first
# native call in the whole script (git clone) reliably killed the installer
# under "Stop" and the exact same run completes normally under "Continue".

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

# ---- prerequisites (auto-installed runtimes: Python, Node, native deps) ---
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
  # Returns @{ Args; Desc; NativeFirst } - Args drives the llama-cpp-python
  # (CMAKE) build; NativeFirst additionally installs the prebuilt native
  # llama-server runtime ('pleiades runtime install'), which is how AMD/Intel
  # GPUs get acceleration on Windows: the official Vulkan binaries need no
  # SDK and no ROCm - they just use the GPU driver. Autofit prefers the
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
          Info "ROCm on Windows isn't a practical llama.cpp target - using the native Vulkan runtime instead."
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
    git clone --branch $Branch $Repo $Dir 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "git clone failed (branch '$Branch' from $Repo)." }
  }
}

function Get-Extras {
  $e = @()
  $e += "ui"   # the control panel (pleiades ui) is always available
  # browser-view (Playwright+Chromium, the desktop app's embedded live
  # browser panel) rides the same -NoBrowser switch as Camoufox (the
  # harness's own agent-driven browsing tool) -- see install.sh's
  # build_extras() for the same reasoning, mirrored here.
  if (-not $NoBrowser) { $e += "browser"; $e += "browser-view" }
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
  Say "Installing Pleiades - $($plan.Desc) build of llama-cpp-python (this can take several minutes)"
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
  Say "Installing the native llama-server runtime (prebuilt, GPU via Vulkan/CUDA - no SDK needed)"
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

function Install-Anamnesis {
  # Anamnesis is vendored in-repo at $Dir\anamnesis (git subtree from
  # Fleabag515/anamnesis -- see docs/specs/2026-07-25-anamnesis-vendoring-
  # and-installer-plan.md). It used to be installed via `npm i -g anamnesis`,
  # which pulled the *unrelated upstream npm package* of the same name, not
  # this fork -- a real bug (every fresh install got the wrong Anamnesis).
  # That step is gone; this only installs the vendored copy's own deps
  # (mirrors install.sh's install_anamnesis()).
  if (-not (Have npm)) { Warn "npm missing; skipping Anamnesis. Install Node, then re-run this installer."; return }
  $anamnesisDir = Join-Path $Dir "anamnesis"
  if (-not (Test-Path (Join-Path $anamnesisDir "package.json"))) {
    Warn "No anamnesis\ found under $Dir -- skipping (expected the vendored copy to already be checked out)."
    return
  }
  Say "Installing Anamnesis's dependencies (vendored copy, not a separate package)"
  Info "npm pulls large native deps here (node-llama-cpp, transformers) - a few minutes of progress output is normal."
  Push-Location $anamnesisDir
  try {
    npm install --omit=dev --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) {
      Warn "Could not install Anamnesis's dependencies; run 'npm install --omit=dev' inside $anamnesisDir yourself."
      return
    }
  } finally { Pop-Location }
  Info "Anamnesis's deps are installed at $anamnesisDir. It starts on its own the first time you run 'pleiades ui' / the desktop app, or start it directly: pleiades anamnesis start"
}

function Fetch-Browser($py) {
  # One pass for both browser automation backends this install needs:
  # Camoufox (the harness's own agent-driven browsing tool) and
  # Playwright's Chromium (the desktop app's embedded live browser panel,
  # webui/browser_view.py). Playwright-Firefox (tor_browser.py's fallback
  # when Camoufox itself isn't preferred) is deliberately NOT fetched here
  # -- see install.sh's fetch_browser() for the same reasoning.
  if ($NoBrowser) { return }
  Say "Fetching browsers (Camoufox + Playwright Chromium)"
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  & $venvPy -m camoufox fetch
  if ($LASTEXITCODE -ne 0) { Warn "camoufox fetch failed; run 'python -m camoufox fetch' later." }
  & $venvPy -m playwright install chromium
  if ($LASTEXITCODE -ne 0) { Warn "playwright chromium fetch failed; run 'python -m playwright install chromium' later." }
}

function Make-Env {
  $envFile = Join-Path $Dir ".env"
  $example = Join-Path $Dir ".env.example"
  if ((-not (Test-Path $envFile)) -and (Test-Path $example)) {
    Copy-Item $example $envFile
    Say "Created .env - set PLEIADES_MODEL_PATH to your .gguf before chatting"
  }
}

$script:SearxngPin = "0909dbc9efb2c6e93e2ad51e60e66417ab291710"  # searxng/searxng@master, pinned 2026-07-25 (no git tags upstream)

function Install-Searxng {
  # SearXNG is fetched at install time (not vendored in-repo like Anamnesis --
  # it's a large, independently-developed third-party project, not a fork
  # Pleiades maintains). Gets its own dedicated venv, fully separate from
  # Pleiades' own .venv, to avoid any dependency-version collision between
  # SearXNG's Flask/lxml/babel stack and Pleiades' own FastAPI/webui stack.
  # See pleiades/searxng_runtime.py's module docstring for the full
  # rationale, and this repo's install.sh (the Linux/macOS equivalent, which
  # this mirrors) -- NOTE this Windows path is unverified, no Windows
  # machine available to test on when it was written (2026-07-25), same
  # caveat as this installer's other GPU/native-runtime branches.
  if ($NoSearxng) { return }
  $searxngDir = Join-Path $Dir "searxng-src"
  $venvPy = Join-Path $searxngDir ".venv\Scripts\python.exe"
  if (Test-Path $venvPy) {
    & $venvPy -c "import searx" *> $null
    if ($LASTEXITCODE -eq 0) { return }  # already fetched + installed
  }
  if (-not (Have git)) { Warn "git missing; skipping SearXNG."; return }
  Say "Fetching SearXNG (pinned commit, no Docker required)"
  if (Test-Path $searxngDir) { Remove-Item -Recurse -Force $searxngDir }
  New-Item -ItemType Directory -Path $searxngDir | Out-Null
  Push-Location $searxngDir
  try {
    git init -q
    git remote add origin https://github.com/searxng/searxng.git
    git fetch --depth 1 origin $script:SearxngPin
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }
    # SearXNG's utils/ tree (deployment configs/scripts for running it behind
    # httpd/nginx/uwsgi/Docker -- none of which Pleiades uses; searxng_runtime.py
    # launches it directly via `python -m searx.webapp`) contains filenames like
    # "searxng.conf:socket" -- a real Linux-only filename (a colon-suffixed
    # unit-style name). NTFS reserves ':' for alternate-data-stream syntax, and
    # `git checkout` refuses to materialize such a path on Windows ("invalid
    # path") -- confirmed live on real Windows/NTFS. Critically, `git sparse-
    # checkout` does NOT help here: sparse-checkout's exclusion is consulted
    # AFTER git validates every path in the tree for filesystem-safety, so
    # excluding utils/ via sparse-checkout still hits the same "invalid path"
    # error during a `git checkout FETCH_HEAD` of the whole tree (verified
    # live -- it fails identically with or without sparse-checkout enabled).
    # What actually works: checkout an explicit pathspec instead of the whole
    # tree. Given an explicit list of paths, git only ever looks at THOSE
    # paths -- utils/ (and its illegal filenames) is never enumerated, let
    # alone validated. This is everything setup.py below actually needs:
    # searx/ (the package + its package_data), and the four root files
    # setup.py's own `open(...)` calls read directly.
    #
    # No "git branch -f master HEAD; git checkout master" afterward (unlike
    # Clone-Repo's real, ongoing checkout below): a pathspec checkout never
    # moves HEAD or creates a commit -- there's nothing for HEAD to resolve
    # to yet on a fresh `git init`, so that would just fail ("not a valid
    # object name: 'HEAD'"), confirmed live. This directory only exists to
    # feed the pip install below and gets fully deleted and re-fetched from
    # scratch on the next run anyway (see the top of this function) -- it
    # was never a real ongoing checkout that needed a named branch.
    git checkout -q FETCH_HEAD -- searx setup.py README.rst LICENSE requirements.txt requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
  } catch {
    Warn "Could not fetch SearXNG: $($_.Exception.Message)"
    Pop-Location
    Remove-Item -Recurse -Force $searxngDir -ErrorAction SilentlyContinue
    return
  }
  try {
    Say "Installing SearXNG's dependencies into its own venv"
    & $py -m venv .venv
    # python.exe -m pip, not pip.exe directly: pip.exe can't overwrite its own
    # running executable on Windows ("To modify pip, please run ... python.exe
    # -m pip install ...", confirmed live) -- the exact pattern Install-Pkg
    # above this function already uses correctly for Pleiades' own venv; this
    # one just hadn't matched it.
    $venvPy2 = Join-Path $searxngDir ".venv\Scripts\python.exe"
    & $venvPy2 -m pip install -q -U pip setuptools wheel
    & $venvPy2 -m pip install -q -U pyyaml msgspec typing-extensions pybind11
    & $venvPy2 -m pip install -q --use-pep517 --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    Info "SearXNG installed at $searxngDir (own venv, no Docker)."
  } catch {
    Warn "Could not install SearXNG's dependencies: $($_.Exception.Message)"
  } finally { Pop-Location }
}

function Install-DesktopApp {
  # Windows counterpart of install.sh's install_desktop_app() -- builds the
  # Electron desktop app + its self-contained PyInstaller backend bundle.
  # Ported (2026-07-25) from the Linux-only version using real Windows 11/
  # AMD hardware access for the first time -- see anamnesis/build-native/
  # {build,fetch-node}.ps1 and desktop/backend-build/build.ps1, all new.
  #
  # Unlike Linux's .deb (which install_desktop_app() installs automatically
  # via dpkg/apt), the NSIS installer this produces is unsigned (no cert
  # available in this environment -- see electron-builder.yml's own note)
  # and Windows SmartScreen gates running an unsigned installer behind an
  # interactive "More info -> Run anyway" click that a script can't safely
  # automate (a scripted silent run risks hanging on a prompt nobody sees).
  # So this builds the installer and tells the user where it is, rather
  # than trying to silently run it the way the Linux path installs its .deb.
  if ($NoDesktopApp) { return }
  if (-not (Have npm)) { Warn "Skipping desktop app (no npm) - build it later: cd desktop && npm install && npm run dist:win"; return }
  if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    Info "Desktop app auto-build needs Windows x64 (anamnesis/build-native has no build for this platform yet); see desktop/README."
    return
  }

  Say "Building Anamnesis's pruned native payload for the desktop app"
  try {
    & (Join-Path $Dir "anamnesis\build-native\fetch-node.ps1")
    if ($LASTEXITCODE -ne 0) { throw "fetch-node.ps1 exited $LASTEXITCODE" }
    & (Join-Path $Dir "anamnesis\build-native\build.ps1")
    if ($LASTEXITCODE -ne 0) { throw "build.ps1 exited $LASTEXITCODE" }
  } catch {
    Warn "Anamnesis native build failed: $($_.Exception.Message) - skipping the desktop app. Build it later: see desktop/README."
    return
  }

  Say "Building the desktop app (Electron + bundled backend)"
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  Push-Location (Join-Path $Dir "desktop")
  try {
    & $venvPy -m pip show pyinstaller *> $null
    if ($LASTEXITCODE -ne 0) { & $venvPy -m pip install pyinstaller }
    $env:PLEIADES_BACKEND_PY = $venvPy
    & (Join-Path $Dir "desktop\backend-build\build.ps1")
    if ($LASTEXITCODE -ne 0) { throw "backend build.ps1 exited $LASTEXITCODE" }
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install exited $LASTEXITCODE" }
    npm run dist:win
    if ($LASTEXITCODE -ne 0) { throw "npm run dist:win exited $LASTEXITCODE" }
  } catch {
    Warn "Desktop app build failed: $($_.Exception.Message) - CLI/webui install is unaffected. Build it later: cd desktop && npm install && npm run dist:win"
    return
  } finally {
    $env:PLEIADES_BACKEND_PY = ""
    Pop-Location
  }

  $installer = Get-ChildItem (Join-Path $Dir "desktop\dist") -Filter "*-setup.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($installer) {
    Info "Built $($installer.FullName)"
    Info "This installer is unsigned -- Windows SmartScreen will warn on first run; click 'More info' -> 'Run anyway' to install."
  } else {
    Warn "Desktop app built but no installer found under desktop\dist -- check the build output above."
  }
}

function Start-Searxng {
  if ($NoSearxng) { return }
  Say "Starting SearXNG"
  $venvPleiades = Join-Path $Dir ".venv\Scripts\pleiades.exe"
  $venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
  Push-Location $Dir
  try {
    if (Test-Path $venvPleiades) { & $venvPleiades search up }
    else { & $venvPy -m pleiades.cli search up }
    if ($LASTEXITCODE -ne 0) { throw "search up exited $LASTEXITCODE" }
  } catch {
    Warn "Could not start SearXNG: $($_.Exception.Message)"
    Info "Later: pleiades search up"
  } finally { Pop-Location }
}

# ---- run -------------------------------------------------------------------
Say "Pleiades installer  (Windows)"
Ensure-Git
$py = Ensure-Python
Ensure-Node
# NOTE: deliberately NOT named $gpu - PowerShell variable names are case-
# insensitive, so $gpu IS the [ValidateSet(...)]$Gpu parameter. Validation
# attributes stay bound to the variable for the whole scope, so assigning
# Resolve-Gpu's hashtable to it re-triggers ValidateSet and aborts the install
# ("System.Collections.Hashtable is not a valid value for Gpu").
$gpuPlan = Resolve-Gpu
Clone-Repo
Install-Pkg $py $gpuPlan
Install-NativeRuntime $gpuPlan
Install-Anamnesis
Install-Searxng
Fetch-Browser $py
Make-Env
Start-Searxng
Install-DesktopApp

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
