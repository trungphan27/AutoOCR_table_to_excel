param(
    [switch]$Overwrite,
    [switch]$SkipRuntimeValidation,
    [switch]$SkipFp16
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$paddle2onnx = Join-Path $projectRoot ".venv\Scripts\paddle2onnx.exe"
$modelsRoot = Join-Path $projectRoot "models"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Missing project virtual environment: $venvPython"
}
$version = & $venvPython -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($version -ne "3.12") {
    throw "Deployment export requires Python 3.12 in .venv; found $version."
}
if (-not (Test-Path -LiteralPath $paddle2onnx -PathType Leaf)) {
    throw @"
paddle2onnx is not installed in .venv. Install export-only dependencies with:

  .\.venv\Scripts\python.exe -m pip install -r requirements-export.txt
"@
}

$exports = @(
    @{
        Name = "det"
        Component = "det"
        Config = "training\configs\pubtabnet_det.yml"
        RunName = "pubtabnet_det"
        Checkpoint = "../output/pubtabnet_det/best_accuracy"
    },
    @{
        Name = "rec"
        Component = "rec"
        Config = "training\configs\pubtabnet_rec_400k.yml"
        RunName = "pubtabnet_rec_400k"
        Checkpoint = "../output/pubtabnet_rec_400k/best_accuracy"
    },
    @{
        Name = "table"
        Component = "slanet"
        Config = "training\configs\pubtabnet_slanet.yml"
        RunName = "pubtabnet_slanet_30k"
        Checkpoint = "../output/pubtabnet_slanet_30k/best_structure_score"
    }
)

foreach ($entry in $exports) {
    $checkpointPrefix = Join-Path $projectRoot (
        "training\output\" + $entry.RunName + "\" +
        [System.IO.Path]::GetFileName($entry.Checkpoint)
    )
    if (-not (Test-Path -LiteralPath "$checkpointPrefix.pdparams" -PathType Leaf)) {
        throw "Missing checkpoint: $checkpointPrefix.pdparams"
    }
    & (Join-Path $PSScriptRoot "run_training.ps1") `
        -Component $entry.Component `
        -Action export `
        -Checkpoint $entry.Checkpoint `
        -ConfigFile $entry.Config `
        -RunName $entry.RunName
    if ($LASTEXITCODE -ne 0) {
        throw "Paddle export failed for $($entry.Name)."
    }

    $inferenceDir = Join-Path $projectRoot (
        "training\output\" + $entry.RunName + "\inference"
    )
    $modelName = @("inference.pdmodel", "model.pdmodel") |
        Where-Object { Test-Path -LiteralPath (Join-Path $inferenceDir $_) } |
        Select-Object -First 1
    $paramsName = @("inference.pdiparams", "model.pdiparams") |
        Where-Object { Test-Path -LiteralPath (Join-Path $inferenceDir $_) } |
        Select-Object -First 1
    if (-not $modelName -or -not $paramsName) {
        throw "Incomplete Paddle inference export in $inferenceDir"
    }

    $destinationDir = Join-Path $modelsRoot $entry.Name
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    $destination = Join-Path $destinationDir "model_fp32.onnx"
    if ((Test-Path -LiteralPath $destination) -and -not $Overwrite) {
        throw "$destination already exists. Pass -Overwrite to replace it."
    }
    & $paddle2onnx `
        --model_dir $inferenceDir `
        --model_filename $modelName `
        --params_filename $paramsName `
        --save_file $destination `
        --opset_version 11 `
        --enable_onnx_checker True
    if ($LASTEXITCODE -ne 0) {
        throw "Paddle2ONNX conversion failed for $($entry.Name)."
    }
}

$dictionaryDir = Join-Path $modelsRoot "dictionaries"
New-Item -ItemType Directory -Force -Path $dictionaryDir | Out-Null
$paddleDictDir = Join-Path $projectRoot "training\PaddleOCR\ppocr\utils\dict"
Copy-Item -LiteralPath (Join-Path $paddleDictDir "table_dict.txt") `
    -Destination (Join-Path $dictionaryDir "table_dict.txt") -Force
Copy-Item -LiteralPath (Join-Path $paddleDictDir "table_structure_dict.txt") `
    -Destination (Join-Path $dictionaryDir "table_structure_dict.txt") -Force

$validationArgs = @("scripts\validate_onnx_models.py", "--runs", "2")
if ($SkipRuntimeValidation) {
    $validationArgs += "--skip-run"
}
Push-Location $projectRoot
try {
    & $venvPython @validationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX validation failed."
    }
} finally {
    Pop-Location
}

if (-not $SkipFp16) {
    $fp32Models = @(
        (Join-Path $modelsRoot "det\model_fp32.onnx"),
        (Join-Path $modelsRoot "rec\model_fp32.onnx"),
        (Join-Path $modelsRoot "table\model_fp32.onnx")
    )
    $conversionArgs = @("scripts\convert_onnx_fp16.py") + $fp32Models
    if ($Overwrite) {
        $conversionArgs += "--overwrite"
    }
    Push-Location $projectRoot
    try {
        & $venvPython @conversionArgs
        if ($LASTEXITCODE -ne 0) {
            throw "FP16 conversion failed."
        }
        $fp16ValidationArgs = @(
            "scripts\validate_onnx_models.py",
            "--det-model", "models\det\model_fp16.onnx",
            "--rec-model", "models\rec\model_fp16.onnx",
            "--table-model", "models\table\model_fp16.onnx",
            "--manifest", "models\manifest_fp16.json",
            "--runs", "1"
        )
        if ($SkipRuntimeValidation) {
            $fp16ValidationArgs += "--skip-run"
        }
        & $venvPython @fp16ValidationArgs
        if ($LASTEXITCODE -ne 0) {
            throw "FP16 ONNX validation failed."
        }
    } finally {
        Pop-Location
    }
}

Write-Host "Deployment models are ready in $modelsRoot"
