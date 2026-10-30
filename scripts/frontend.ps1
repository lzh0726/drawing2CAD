param([ValidateSet('build','dev')][string]$Mode = 'dev')
$ErrorActionPreference = 'Stop'
$cadRoot = Split-Path -Parent $PSScriptRoot
$cadNode = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
if (!(Test-Path -LiteralPath $cadNode)) { $cadNode = (Get-Command node -ErrorAction Stop).Source }
$cadVersion = [version]((& $cadNode --version).Trim().TrimStart('v'))
if ($cadVersion.Major -lt 22 -or ($cadVersion.Major -eq 22 -and $cadVersion.Minor -lt 12)) { throw 'Install Node.js 22.12+ or Node.js 24 before continuing.' }
$env:PATH = (Split-Path -Parent $cadNode) + ';' + $env:PATH
$cadNpmCommand = Get-Command npm.cmd -ErrorAction Stop
$cadNpm = Join-Path (Split-Path -Parent $cadNpmCommand.Source) 'node_modules\npm\bin\npm-cli.js'
if (!(Test-Path -LiteralPath $cadNpm)) { throw 'npm-cli.js not found next to npm.cmd.' }
Push-Location (Join-Path $cadRoot 'frontend')
try {
    if (!(Test-Path 'node_modules')) {
        if (Test-Path 'package-lock.json') { & $cadNode $cadNpm ci --no-audit --no-fund }
        else { & $cadNode $cadNpm install --no-audit --no-fund }
        if ($LASTEXITCODE -ne 0) { throw 'npm installation failed.' }
    }
    & $cadNode $cadNpm run $Mode
    if ($LASTEXITCODE -ne 0) { throw 'Frontend command failed.' }
} finally { Pop-Location }
