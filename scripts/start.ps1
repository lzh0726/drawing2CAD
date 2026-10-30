param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$cadRoot = Split-Path -Parent $PSScriptRoot
$cadPython = Join-Path $cadRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $cadPython) -or !(Test-Path -LiteralPath (Join-Path $cadRoot 'frontend\dist\index.html'))) {
    throw 'Dependencies or frontend build missing. Run scripts\setup.ps1 first.'
}
$cadPidFile = Join-Path $cadRoot '.runtime.json'
if (Test-Path -LiteralPath $cadPidFile) {
    $cadRuntime = Get-Content -LiteralPath $cadPidFile -Raw | ConvertFrom-Json
    $cadExisting = Get-CimInstance Win32_Process -Filter "ProcessId = $($cadRuntime.pid)" -ErrorAction SilentlyContinue
    if ($cadExisting -and $cadExisting.ExecutablePath -eq $cadPython -and $cadExisting.CreationDate.ToUniversalTime().Ticks -eq ([datetime]$cadRuntime.created).ToUniversalTime().Ticks) {
        Write-Host 'Drawing2CAD is already running: http://127.0.0.1:8000'
        if (!$NoBrowser) { Start-Process 'http://127.0.0.1:8000' }
        exit 0
    }
}
$cadPort = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($cadPort) { throw 'Port 8000 is in use. Stop the existing service first; this script will not kill it.' }
$cadLogs = Join-Path $cadRoot 'storage\logs'
New-Item -ItemType Directory -Force -Path $cadLogs | Out-Null
$cadProcess = Start-Process -FilePath $cadPython -ArgumentList '-m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000' -WorkingDirectory $cadRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $cadLogs 'server.out.log') -RedirectStandardError (Join-Path $cadLogs 'server.err.log')
$cadIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($cadProcess.Id)"
@{pid=$cadProcess.Id;created=$cadIdentity.CreationDate.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $cadPidFile -Encoding UTF8
for ($cadAttempt=0; $cadAttempt -lt 40; $cadAttempt++) {
    try {
        $cadHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
        if ($cadHealth.status -eq 'ok') {
            Write-Host 'Drawing2CAD ready: http://127.0.0.1:8000'
            if (!$NoBrowser) { Start-Process 'http://127.0.0.1:8000' }
            exit 0
        }
    } catch { }
    if ($cadProcess.HasExited) { throw 'Backend exited. Check storage\logs\server.err.log locally.' }
    Start-Sleep -Milliseconds 500
}
throw 'Startup timed out. Check storage\logs locally.'
