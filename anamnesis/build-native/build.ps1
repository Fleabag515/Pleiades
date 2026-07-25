# Build a pruned, self-contained copy of Anamnesis's node_modules for
# bundling into the Pleiades desktop app. Windows counterpart of build.sh --
# see that file for the fuller rationale this one doesn't repeat (why a
# pruned copy at all, why CPU-only variants, etc).
#
# Reproduce with:
#   ~\Pleiades\anamnesis\build-native\build.ps1
#
# Output: ~\Pleiades\anamnesis\build-native\dist\anamnesis\
#   (a full copy of src/ + a pruned node_modules\, ready for extraResources)
$ErrorActionPreference = "Continue"

if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
  Write-Error "error: this script has only been built + verified for Windows x64. Got: $env:PROCESSOR_ARCHITECTURE."
  exit 1
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path   # ...\anamnesis
$BuildDir = Join-Path $RepoRoot "build-native"
$DistDir = Join-Path $BuildDir "dist\anamnesis"

Write-Host "==> Cleaning previous build output"
if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
New-Item -ItemType Directory -Path $DistDir | Out-Null

Write-Host "==> Copying Anamnesis source (src/, package.json, config.json) -- not node_modules, that gets a fresh install below"
# robocopy is the standard built-in Windows tool for this; exit codes 0-7 are
# all "success" (8+ is a real failure) -- see
# https://learn.microsoft.com/windows-server/administration/windows-commands/robocopy#exit-return-codes
robocopy $RepoRoot $DistDir /E /XD node_modules build-native .git /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) {
  Write-Error "error: robocopy failed copying Anamnesis source (exit $LASTEXITCODE)."
  exit 1
}

Write-Host "==> Installing production dependencies fresh into the build copy"
Push-Location $DistDir
try {
  npm ci --omit=dev --no-audit --no-fund
  if ($LASTEXITCODE -ne 0) {
    Write-Error "error: npm ci failed (exit $LASTEXITCODE)."
    exit 1
  }
} finally { Pop-Location }

$NodeModules = Join-Path $DistDir "node_modules"
$BeforeSize = "{0:N0} MB" -f ((Get-ChildItem $NodeModules -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)

Write-Host "==> Pruning @node-llama-cpp: keeping only win-x64 (CPU)"
# Same deliberate CPU-only choice as build.sh -- see that file's comment for
# the full reasoning (this powers Anamnesis's background extraction/
# foresight/persona-drift helper LLM, not the user-facing chat model).
$NlcDir = Join-Path $NodeModules "@node-llama-cpp"
if (Test-Path $NlcDir) {
  Get-ChildItem $NlcDir -Directory | Where-Object { $_.Name -ne "win-x64" } | Remove-Item -Recurse -Force
}

Write-Host "==> Pruning onnxruntime-node: keeping only win32/x64, dropping its GPU execution providers"
$OrtNapiDir = Join-Path $NodeModules "onnxruntime-node\bin\napi-v3"
if (Test-Path $OrtNapiDir) {
  Get-ChildItem $OrtNapiDir -Directory | Where-Object { $_.Name -ne "win32" } | Remove-Item -Recurse -Force
  $Win32Dir = Join-Path $OrtNapiDir "win32"
  if (Test-Path $Win32Dir) {
    Get-ChildItem $Win32Dir -Directory | Where-Object { $_.Name -ne "x64" } | Remove-Item -Recurse -Force
    $X64Dir = Join-Path $Win32Dir "x64"
    Remove-Item (Join-Path $X64Dir "onnxruntime_providers_cuda.dll") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $X64Dir "onnxruntime_providers_tensorrt.dll") -ErrorAction SilentlyContinue
  }
}

$AfterSize = "{0:N0} MB" -f ((Get-ChildItem $NodeModules -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)

Write-Host "==> Verifying the pruned copy still boots and answers a real health check"
$ScratchHome = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path (Join-Path $ScratchHome ".anamnesis") -Force | Out-Null
# Set-Content -Encoding UTF8 writes a BOM by default in Windows PowerShell
# -- confirmed live this breaks Node's JSON.parse() on this exact file,
# which daemon.js swallows silently and falls back to its default port
# (9000), colliding with any already-running real Anamnesis daemon rather
# than using this scratch verification's own 19898. WriteAllText with no
# explicit encoding defaults to UTF-8 *without* a BOM.
[System.IO.File]::WriteAllText((Join-Path $ScratchHome ".anamnesis\daemon.json"), '{ "controlPort": 19898 }')

$logPath = Join-Path $BuildDir "last-verify-boot.log"
$daemonJs = Join-Path $DistDir "src\daemon.js"
# Node's os.homedir() resolves from USERPROFILE on Windows (not HOME) --
# override just that for this child process (child processes inherit the
# parent's env block), matching build.sh's HOME= override for the same
# purpose. Start-Process has no -Environment parameter, so save/restore.
$origUserProfile = $env:USERPROFILE
$env:USERPROFILE = $ScratchHome
try {
  $proc = Start-Process -FilePath "node" -ArgumentList $daemonJs `
    -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" `
    -WindowStyle Hidden -PassThru
  Start-Sleep -Seconds 3
} finally { $env:USERPROFILE = $origUserProfile }
$health = $null
try { $health = (Invoke-WebRequest -Uri "http://127.0.0.1:19898/status" -TimeoutSec 5 -UseBasicParsing).Content } catch {}
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $ScratchHome -ErrorAction SilentlyContinue

if (-not $health -or $health -notmatch '"status":"ok"') {
  Write-Error "error: pruned build failed its own boot/health check. Log:"
  Get-Content $logPath -ErrorAction SilentlyContinue | Write-Host
  Get-Content "$logPath.err" -ErrorAction SilentlyContinue | Write-Host
  exit 1
}
Remove-Item $logPath, "$logPath.err" -ErrorAction SilentlyContinue

Write-Host "==> OK -- pruned copy boots and reports healthy: $health"
Write-Host "node_modules: $BeforeSize -> $AfterSize"
Write-Host "Built: $DistDir"
