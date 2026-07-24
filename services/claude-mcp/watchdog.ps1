# Claude MCP Server Watchdog
# Checks if server is alive on port 3456, starts it if not.

$serverDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logFile   = "$serverDir\server.log"
$errFile   = "$serverDir\server.err"

$env:COMSPEC = "C:\Windows\System32\cmd.exe"

function Is-ServerRunning {
  try {
    $r = Invoke-WebRequest -Uri "http://localhost:3456" -UseBasicParsing -TimeoutSec 3
    return $r.StatusCode -eq 200
  } catch { return $false }
}

if (Is-ServerRunning) {
  Write-Host "[$(Get-Date)] Claude MCP server already running."
} else {
  Write-Host "[$(Get-Date)] Server not detected — starting..."
  Start-Process -FilePath "node" `
    -ArgumentList "server.mjs" `
    -WorkingDirectory $serverDir `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError  $errFile `
    -WindowStyle Hidden
  Start-Sleep 3
  if (Is-ServerRunning) {
    Write-Host "[$(Get-Date)] Server started successfully."
  } else {
    Write-Host "[$(Get-Date)] ERROR: Server failed to start. Check $errFile"
  }
}
