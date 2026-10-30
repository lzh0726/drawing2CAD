$ErrorActionPreference = 'Stop'
$cadRoot = Split-Path -Parent $PSScriptRoot
$cadPidFile = Join-Path $cadRoot '.runtime.json'
if (!(Test-Path -LiteralPath $cadPidFile)) { Write-Host 'No managed service recorded.'; exit 0 }
$cadRuntime = Get-Content -LiteralPath $cadPidFile -Raw | ConvertFrom-Json
$cadProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($cadRuntime.pid)" -ErrorAction SilentlyContinue
$cadPython = Join-Path $cadRoot '.venv\Scripts\python.exe'
if (!$cadProcess) { Write-Host 'Service already stopped.'; exit 0 }
if ($cadProcess.ExecutablePath -ne $cadPython -or $cadProcess.CreationDate.ToUniversalTime().Ticks -ne ([datetime]$cadRuntime.created).ToUniversalTime().Ticks) {
    throw 'Process identity changed; refusing to stop an unrelated process.'
}
if ($cadProcess.CommandLine -notlike '*uvicorn backend.app.main:app*') { throw 'Unexpected command; no process stopped.' }
# Windows venv Python launches the base interpreter as a child process.
$cadBasePython = (& $cadPython -c 'import sys; print(sys._base_executable)').Trim()
function Stop-CadChildren([int]$CadParent) {
    $cadChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $CadParent"
    foreach ($cadChild in $cadChildren) {
        if ($cadChild.ExecutablePath -in @($cadPython,$cadBasePython) -and ($cadChild.CommandLine -like '*-m uvicorn backend.app.main:app*' -or $cadChild.CommandLine -like '*-m backend.app.cad_runner*')) {
            Stop-CadChildren $cadChild.ProcessId
            Stop-Process -Id $cadChild.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}
Stop-CadChildren $cadRuntime.pid
Stop-Process -Id $cadRuntime.pid -Force -ErrorAction SilentlyContinue
Write-Host 'Drawing2CAD stopped. Tasks and completed versions are preserved.'
