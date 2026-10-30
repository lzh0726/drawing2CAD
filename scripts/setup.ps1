$ErrorActionPreference = 'Stop'
$cadRoot = Split-Path -Parent $PSScriptRoot
$cadBundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
$cadPython = Join-Path $cadRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $cadPython)) {
    $cadBasePython = Join-Path $cadBundled 'python\python.exe'
    if (Test-Path -LiteralPath $cadBasePython) { & $cadBasePython -m venv (Join-Path $cadRoot '.venv') }
    else { py -3.12 -m venv (Join-Path $cadRoot '.venv') }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
}
$cadRequirements = Join-Path $cadRoot 'backend\requirements.lock.txt'
if (!(Test-Path -LiteralPath $cadRequirements)) { $cadRequirements = Join-Path $cadRoot 'backend\requirements.txt' }
& $cadPython -m pip install -r $cadRequirements --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
& (Join-Path $PSScriptRoot 'frontend.ps1') build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
Write-Host 'Setup complete. Run scripts\start.ps1.'
