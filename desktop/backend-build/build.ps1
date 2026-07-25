# Build the self-contained Pleiades backend bundle with PyInstaller (Windows).
# Mirrors build.sh -- see that file for the fuller rationale comments this
# one doesn't repeat.
#
# Reproduce with:
#   ~\Pleiades\desktop\backend-build\build.ps1
#
# Uses whichever python built ~\Pleiades\.venv (PLEIADES_BACKEND_PY overrides
# this -- install.ps1's Install-DesktopApp passes its own already-resolved
# interpreter so this script doesn't have to re-derive it). Requires
# PyInstaller there (one-time): <that python> -m pip install pyinstaller
#
# Output: ~\Pleiades\desktop\backend-build\dist\pleiades-backend\
#   (a folder containing the pleiades-backend.exe launcher + _internal\ deps
#   -- onedir, not onefile, same reasoning as the Linux build)
$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$VenvPy = $env:PLEIADES_BACKEND_PY
if (-not $VenvPy) {
  $VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $VenvPy)) {
  Write-Error "error: $VenvPy not found. Build the venv first (see repo CLAUDE.md)."
  exit 1
}

Push-Location (Join-Path $RepoRoot "desktop\backend-build")
try {
  & $VenvPy -m PyInstaller --noconfirm --clean backend.spec
  if ($LASTEXITCODE -ne 0) {
    Write-Error "error: PyInstaller build failed (exit $LASTEXITCODE)."
    exit 1
  }
} finally { Pop-Location }

$OutDir = Join-Path $RepoRoot "desktop\backend-build\dist\pleiades-backend"
Write-Host ""
Write-Host "Built: $OutDir\pleiades-backend.exe"
$size = (Get-ChildItem $OutDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("{0:N0} MB" -f $size)
