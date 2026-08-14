param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("det", "rec", "slanet")]
    [string]$Component,

    [ValidateSet("train", "eval", "export")]
    [string]$Action = "train",

    [string]$Checkpoint = "",

    [string]$Subset = "",

    [string]$ConfigFile = "",

    [string]$RunName = "",

    [switch]$Resume,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$paddleRoot = Join-Path $projectRoot "training\PaddleOCR"

if ($ConfigFile) {
    $configCandidate = $ConfigFile
    if (-not [System.IO.Path]::IsPathRooted($configCandidate)) {
        $configCandidate = Join-Path $projectRoot $configCandidate
    }
    if (-not (Test-Path -LiteralPath $configCandidate -PathType Leaf)) {
        throw "Missing training config: $configCandidate"
    }
    $configPath = (Resolve-Path -LiteralPath $configCandidate).Path
} else {
    $configPath = Join-Path $projectRoot "training\configs\pubtabnet_$Component.yml"
}

if ($RunName) {
    if ([System.IO.Path]::GetFileName($RunName) -ne $RunName) {
        throw "RunName must be a directory name, not a path: $RunName"
    }
    $outputName = $RunName
} else {
    $outputName = "pubtabnet_$Component"
}
$relativeOutput = "../output/$outputName"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing project virtual environment: $venvPython"
}

$version = & $venvPython -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($version -ne "3.12") {
    throw "The project virtual environment must use Python 3.12; found $version."
}

$gpuDllDirectories = @(
    @{
        Name = "cuDNN 8"
        Directory = Join-Path $projectRoot ".vendor\cudnn-8.9.6-cuda11\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\bin"
        RequiredDll = "cudnn64_8.dll"
    },
    @{
        Name = "CUDA 11 runtime"
        Directory = Join-Path $projectRoot ".venv\Lib\site-packages\nvidia\cuda_runtime\bin"
        RequiredDll = "cudart64_110.dll"
    },
    @{
        Name = "cuBLAS 11"
        Directory = Join-Path $projectRoot ".venv\Lib\site-packages\nvidia\cublas\bin"
        RequiredDll = "cublasLt64_11.dll"
    }
)

foreach ($gpuDll in $gpuDllDirectories) {
    $dllPath = Join-Path $gpuDll.Directory $gpuDll.RequiredDll
    if (-not (Test-Path -LiteralPath $dllPath)) {
        throw "Missing $($gpuDll.Name) DLL: $dllPath. Repair the project-local GPU runtime before running $Action."
    }
}

foreach ($gpuDll in $gpuDllDirectories) {
    $env:PATH = "$($gpuDll.Directory);$env:PATH"
}

$subsetOverrideValues = @()
if ($Subset -and $Action -ne "export") {
    $subsetRoot = Join-Path $projectRoot "training\data\pubtabnet\subsets\$Subset"
    switch ($Component) {
        "det" {
            $trainLabel = Join-Path $subsetRoot "det_train.txt"
            $evalLabel = Join-Path $subsetRoot "det_val.txt"
            $relativeTrainLabel = "../data/pubtabnet/subsets/$Subset/det_train.txt"
            $relativeEvalLabel = "../data/pubtabnet/subsets/$Subset/det_val.txt"
        }
        "rec" {
            $trainLabel = Join-Path $subsetRoot "rec_train.txt"
            $evalLabel = Join-Path $subsetRoot "rec_val.txt"
            $relativeTrainLabel = "../data/pubtabnet/subsets/$Subset/rec_train.txt"
            $relativeEvalLabel = "../data/pubtabnet/subsets/$Subset/rec_val.txt"
        }
        "slanet" {
            $trainLabel = Join-Path $subsetRoot "PubTabNet_2.0.0_train.jsonl"
            $evalLabel = Join-Path $subsetRoot "PubTabNet_2.0.0_val.jsonl"
            $relativeTrainLabel = "../data/pubtabnet/subsets/$Subset/PubTabNet_2.0.0_train.jsonl"
            $relativeEvalLabel = "../data/pubtabnet/subsets/$Subset/PubTabNet_2.0.0_val.jsonl"
        }
    }
    foreach ($labelPath in @($trainLabel, $evalLabel)) {
        if (-not (Test-Path -LiteralPath $labelPath)) {
            throw "Missing $Component subset label: $labelPath"
        }
    }
    $subsetOverrideValues = @(
        "Train.dataset.label_file_list=[$relativeTrainLabel]",
        "Eval.dataset.label_file_list=[$relativeEvalLabel]",
        "Train.dataset.ratio_list=[1.0]",
        "Eval.dataset.ratio_list=[1.0]"
    )
}

$outputOverrideValues = @()
if ($RunName) {
    $outputOverrideValues = @(
        "Global.save_model_dir=$relativeOutput",
        "Global.save_inference_dir=$relativeOutput/inference",
        "Global.save_res_path=$relativeOutput/predicts.txt"
    )
}

$overrideValues = @($outputOverrideValues) + @($subsetOverrideValues) + @($Overrides | Where-Object { $_ })

if ($Resume -and $Action -ne "train") {
    throw "-Resume can only be used with -Action train."
}
if ($Resume -and $Checkpoint) {
    throw "Use either -Resume or -Checkpoint, not both."
}

if ($Resume) {
    $modelOutput = Join-Path $projectRoot "training\output\$outputName"
    $checkpointCandidates = @()
    foreach ($prefix in @("latest_step", "latest")) {
        $candidatePrefix = Join-Path $modelOutput $prefix
        $requiredFiles = @(
            "$candidatePrefix.pdparams",
            "$candidatePrefix.pdopt",
            "$candidatePrefix.states"
        )
        if (($requiredFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) {
            $checkpointCandidates += [PSCustomObject]@{
                Prefix = $prefix
                LastWriteTime = (Get-Item -LiteralPath "$candidatePrefix.states").LastWriteTime
            }
        }
    }
    if ($checkpointCandidates.Count -eq 0) {
        throw "No complete resumable checkpoint found in $modelOutput"
    }
    $resumePrefix = ($checkpointCandidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Prefix
    $resumeCheckpoint = "$relativeOutput/$resumePrefix"
    $overrideValues = @("Global.checkpoints=$resumeCheckpoint") + $overrideValues
    Write-Host "Resuming $Component from $resumeCheckpoint"
} elseif ($Action -eq "train" -and $Checkpoint) {
    $overrideValues = @("Global.checkpoints=$Checkpoint") + $overrideValues
}

switch ($Action) {
    "train" {
        $entrypoint = "tools\train.py"
        $arguments = @($entrypoint, "-c", $configPath)
        if ($overrideValues.Count -gt 0) {
            $arguments += @("-o") + $overrideValues
        }
    }
    "eval" {
        if (-not $Checkpoint) {
            throw "-Checkpoint is required for eval."
        }
        $entrypoint = "tools\eval.py"
        $arguments = @(
            $entrypoint,
            "-c", $configPath,
            "-o", "Global.checkpoints=$Checkpoint"
        ) + $overrideValues
    }
    "export" {
        if (-not $Checkpoint) {
            throw "-Checkpoint is required for export."
        }
        $entrypoint = "tools\export_model.py"
        $saveDirectory = "$relativeOutput/inference"
        $arguments = @(
            $entrypoint,
            "-c", $configPath,
            "-o", "Global.checkpoints=$Checkpoint",
            "Global.save_inference_dir=$saveDirectory"
        ) + $overrideValues
    }
}

Push-Location $paddleRoot
try {
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed for $Component with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
