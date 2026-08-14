param(
    [switch]$SkipPatch
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainingRoot = Join-Path $projectRoot "training"
$paddleRoot = Join-Path $trainingRoot "PaddleOCR"
$patchPath = Join-Path $trainingRoot "patches\paddleocr-release-2.7.patch"
$repository = "https://github.com/PaddlePaddle/PaddleOCR.git"
$commit = "8cce9b6fd7ccb50226d0c38f94054d81c29b8184"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to restore PaddleOCR."
}

New-Item -ItemType Directory -Force -Path $trainingRoot | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $paddleRoot ".git"))) {
    if (Test-Path -LiteralPath $paddleRoot) {
        throw "training\PaddleOCR exists but is not a Git checkout."
    }
    & git clone --branch release/2.7 --single-branch $repository $paddleRoot
    if ($LASTEXITCODE -ne 0) {
        throw "PaddleOCR clone failed."
    }
}

$status = & git -c "safe.directory=$paddleRoot" -C $paddleRoot `
    status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the PaddleOCR checkout."
}
if ($status) {
    & git -c "safe.directory=$paddleRoot" -C $paddleRoot `
        apply --reverse --check `
        --ignore-space-change --ignore-whitespace $patchPath 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PaddleOCR patch is already applied."
        exit 0
    }
    throw "training\PaddleOCR contains uncommitted changes. Preserve them before setup."
}

& git -c "safe.directory=$paddleRoot" -C $paddleRoot `
    checkout --detach $commit
if ($LASTEXITCODE -ne 0) {
    throw "Unable to check out PaddleOCR commit $commit."
}

if ($SkipPatch) {
    Write-Host "PaddleOCR is ready at commit $commit without project patches."
    exit 0
}
if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) {
    throw "Missing PaddleOCR patch: $patchPath"
}

& git -c "safe.directory=$paddleRoot" -C $paddleRoot `
    apply --check `
    --ignore-space-change --ignore-whitespace $patchPath
if ($LASTEXITCODE -ne 0) {
    throw "The project patch is incompatible with PaddleOCR commit $commit."
}
& git -c "safe.directory=$paddleRoot" -C $paddleRoot `
    apply `
    --ignore-space-change --ignore-whitespace $patchPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply the project PaddleOCR patch."
}

Write-Host "PaddleOCR $commit is ready with the project training patch."
