# install.ps1 — install the anamnesis CLI on Windows.
# Run in PowerShell: irm https://raw.githubusercontent.com/Fleabag515/anamnesis/main/install.ps1 | iex

$ErrorActionPreference = 'Stop'
$RepoUrl    = 'https://github.com/Fleabag515/anamnesis/archive/refs/heads/main.zip'
$InstallDir = "$env:LOCALAPPDATA\anamnesis"

Write-Host "`n✨ Installing anamnesis...`n" -ForegroundColor Cyan

# ─── Check / install Node ─────────────────────────────────────────────────────
$NodeCmd = Get-Command node -ErrorAction SilentlyContinue
$NodeBin = if ($NodeCmd) { $NodeCmd.Source } else { $null }
if (-not $NodeBin) {
    Write-Host "Node.js not found — installing via winget..." -ForegroundColor Yellow
    winget install OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
    $NodeBin = (Get-Command node).Source
}

$NodeVersion = & node --version
$NodeMajor = [int]($NodeVersion -replace '^v(\d+).*','$1')
if ($NodeMajor -lt 18) {
    Write-Host "Node $NodeVersion found but Node 18+ required. Install Node 18+ and retry." -ForegroundColor Red
    exit 1
}
Write-Host "Node $NodeVersion found" -ForegroundColor Green

# ─── Download + extract ──────────────────────────────────────────────────────
$TmpZip = "$env:TEMP\anamnesis.zip"
Write-Host "Downloading..."
Invoke-WebRequest -Uri $RepoUrl -OutFile $TmpZip -UseBasicParsing

if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
Expand-Archive -Path $TmpZip -DestinationPath $env:TEMP -Force
Move-Item "$env:TEMP\anamnesis-main" $InstallDir -Force
Remove-Item $TmpZip

# ─── Install dependencies ─────────────────────────────────────────────────────
Set-Location $InstallDir
npm install --omit=dev --silent

# ─── Register anamnesis command globally ─────────────────────────────────────
npm link --silent

Write-Host "`n✓ anamnesis installed" -ForegroundColor Green
Write-Host "  Restart your terminal, then run: anamnesis new"
Write-Host "  To register as a Windows Service (run as Administrator): anamnesis install`n"
