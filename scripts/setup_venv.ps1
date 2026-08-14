param(
    [string]$Python312 = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
    $version = & $venvPython -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    if ($version -ne "3.12") {
        throw "Existing .venv uses Python $version; Python 3.12 is required."
    }
    Write-Host "Python 3.12 virtual environment already exists: $venvPython"
    exit 0
}

if ($Python312) {
    $basePython = (Resolve-Path -LiteralPath $Python312).Path
} else {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw @"
Python Launcher was not found. Install Python 3.12 side-by-side, without
replacing the global Python 3.14, then rerun one of:

  powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1

or pass its exact executable path:

  powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 `
    -Python312 "C:\path\to\Python312\python.exe"
"@
    }
    & $launcher.Source -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 is not registered with the Python Launcher."
    }
    $basePython = $launcher.Source
}

Push-Location $projectRoot
try {
    if ($Python312) {
        & $basePython -m venv .venv
    } else {
        & $basePython -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv."
    }
    & $venvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "The virtual environment was created, but pip bootstrap failed."
    }
    & $venvPython -c "import sys; print('Virtual environment ready:', sys.executable, sys.version)"
} finally {
    Pop-Location
}
