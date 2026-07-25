# Fetch a pinned, official Node runtime binary to bundle alongside Anamnesis's
# pruned node_modules (see build.ps1 + docs/specs/2026-07-25-anamnesis-
# vendoring-and-installer-plan.md Part 1b) -- so a packaged Pleiades install
# needs no system-installed Node at all. Windows counterpart of fetch-node.sh
# -- same pinned version, same "just the binary, not npm/corepack" scope, see
# that file for the fuller rationale this one doesn't repeat.
#
# Reproduce with:
#   ~\Pleiades\anamnesis\build-native\fetch-node.ps1
#
# Output: ~\Pleiades\anamnesis\build-native\dist\node-runtime\win32-x64\node.exe
$ErrorActionPreference = "Continue"

$NodeVersion = "24.18.0"  # keep in lockstep with fetch-node.sh's pin

if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
  Write-Error "error: this script has only been built + verified for Windows x64. Got: $env:PROCESSOR_ARCHITECTURE."
  exit 1
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path   # ...\anamnesis
$BuildDir = Join-Path $RepoRoot "build-native"
$Target = "win-x64"
$ArchiveName = "node-v$NodeVersion-$Target.zip"
$Url = "https://nodejs.org/dist/v$NodeVersion/$ArchiveName"
$ShasumsUrl = "https://nodejs.org/dist/v$NodeVersion/SHASUMS256.txt"

$OutDir = Join-Path $BuildDir "dist\node-runtime\win32-x64"
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $WorkDir | Out-Null

try {
  Write-Host "==> Fetching Node v$NodeVersion ($Target)"
  $archivePath = Join-Path $WorkDir $ArchiveName
  Invoke-WebRequest -Uri $Url -OutFile $archivePath
  $shasumsPath = Join-Path $WorkDir "SHASUMS256.txt"
  Invoke-WebRequest -Uri $ShasumsUrl -OutFile $shasumsPath

  Write-Host "==> Verifying checksum"
  $line = Select-String -Path $shasumsPath -Pattern "  $ArchiveName`$"
  if (-not $line) {
    Write-Error "error: no checksum entry found for $ArchiveName in SHASUMS256.txt"
    exit 1
  }
  $expected = ($line.Line -split "\s+")[0]
  $actual = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLower()
  if ($expected -ne $actual) {
    Write-Error "error: checksum mismatch for $ArchiveName`nexpected: $expected`nactual:   $actual"
    exit 1
  }
  Write-Host "Checksum OK ($actual)"

  Write-Host "==> Extracting just node.exe (not npm/corepack/docs/etc)"
  Expand-Archive -Path $archivePath -DestinationPath $WorkDir -Force
  $extractedExe = Join-Path $WorkDir "node-v$NodeVersion-$Target\node.exe"
  if (-not (Test-Path $extractedExe)) {
    Write-Error "error: node.exe not found in extracted archive at $extractedExe"
    exit 1
  }

  if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
  New-Item -ItemType Directory -Path $OutDir | Out-Null
  Copy-Item $extractedExe (Join-Path $OutDir "node.exe")

  Write-Host "==> Verifying the extracted binary actually runs"
  $actualVersion = & (Join-Path $OutDir "node.exe") --version
  if ($actualVersion -ne "v$NodeVersion") {
    Write-Error "error: extracted node reports '$actualVersion', expected 'v$NodeVersion'"
    exit 1
  }

  Write-Host "==> OK -- $actualVersion at $OutDir\node.exe"
  $size = (Get-Item (Join-Path $OutDir "node.exe")).Length / 1MB
  Write-Host ("{0:N1} MB" -f $size)
} finally {
  Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}
