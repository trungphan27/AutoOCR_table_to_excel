# End-to-End Setup, Training, Export, and Deployment

This runbook reproduces the English PubTabNet pipeline from a fresh clone. Run
PowerShell commands from the repository root. Every Python, pip, test, training,
export, and local deployment command uses the project Python 3.12 virtual
environment.

## 1. Prerequisites

Required:

- Git.
- Windows PowerShell 5.1 or PowerShell 7.
- Python 3.12 installed side-by-side with any global Python version.
- Sufficient disk space for PubTabNet, derived cell crops, checkpoints, and
  Docker images.

For Paddle GPU training:

- NVIDIA GPU and driver.
- PaddlePaddle GPU 2.6.2 for CUDA 11.8.
- cuDNN 8.9.6 for CUDA 11.

For Docker GPU deployment:

- Docker Desktop with the WSL2 backend.
- NVIDIA GPU access enabled in Docker.

## 2. Clone the repository

```powershell
git clone https://github.com/trungphan27/AutoOCR_table_to_excel.git
Set-Location AutoOCR_table_to_excel
```

Restore the pinned PaddleOCR source and apply this project's training patch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\setup_paddleocr.ps1
```

The script checks out PaddleOCR commit
`8cce9b6fd7ccb50226d0c38f94054d81c29b8184` and applies:

```text
training/patches/paddleocr-release-2.7.patch
```

The patch adds resumable mid-epoch checkpoints, training/evaluation progress
bars, SLANet token metrics, TEDS-Structure, and structure-aware best-checkpoint
selection.

## 3. Create the Python 3.12 environment

Recommended:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\setup_venv.ps1
```

If the Python launcher cannot locate 3.12, pass the executable explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\setup_venv.ps1 `
  -Python312 "C:\Path\To\Python312\python.exe"
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe --version
```

The reported version must start with `Python 3.12`.

## 4. Install training dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

Install exactly one PaddlePaddle build. CUDA 11.8 GPU build:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  paddlepaddle-gpu==2.6.2 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Install the project-local CUDA runtime and cuBLAS libraries:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  nvidia-cuda-runtime-cu11==11.8.89 `
  nvidia-cublas-cu11==11.11.3.6
```

Download cuDNN 8.9.6 for CUDA 11 from NVIDIA and place it at:

```text
.vendor/cudnn-8.9.6-cuda11/
└── cudnn-windows-x86_64-8.9.6.50_cuda11-archive/
    └── bin/
        └── cudnn64_8.dll
```

Do not install both `paddlepaddle` and `paddlepaddle-gpu` in the same venv.

Verify the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_training_env.py
.\.venv\Scripts\python.exe scripts\smoke_build_models.py
```

After the dataset is installed, use the stricter check:

```powershell
.\.venv\Scripts\python.exe scripts\check_training_env.py `
  --require-gpu --require-data
```

## 5. Download PubTabNet 2.0.0

Download the English PubTabNet 2.0.0 dataset from the
[official PubTabNet repository](https://github.com/ibm-aur-nlp/PubTabNet) or
its linked dataset host. Extract it to this exact layout:

```text
training/data/pubtabnet/
├── train/
│   └── *.png
├── val/
│   └── *.png
├── PubTabNet_2.0.0_train.jsonl
└── PubTabNet_2.0.0_val.jsonl
```

Check the layout:

```powershell
Test-Path training\data\pubtabnet\train
Test-Path training\data\pubtabnet\val
Test-Path training\data\pubtabnet\PubTabNet_2.0.0_train.jsonl
Test-Path training\data\pubtabnet\PubTabNet_2.0.0_val.jsonl
```

All four results must be `True`.

## 6. Preprocess the dataset

Run a 100-table smoke test first:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pubtabnet.py `
  --splits train val `
  --tasks validate det rec `
  --limit 100 `
  --overwrite
```

Then process the complete train/validation annotations:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pubtabnet.py `
  --splits train val `
  --tasks validate det rec `
  --overwrite
```

Generated artifacts:

```text
training/data/pubtabnet/derived/
├── det_train.txt
├── det_val.txt
├── rec_train.txt
├── rec_val.txt
└── rec/
    ├── train/
    └── val/
```

Warnings about empty cells, recognition strings longer than 100 characters, or
structures longer than 500 tokens are counters for excluded samples. They do
not indicate preprocessing failure unless the command exits non-zero.

## 7. Create the training subsets

Create the aligned 30,000-table training subset and 1,000-table validation
subset:

```powershell
.\.venv\Scripts\python.exe scripts\create_pubtabnet_subset.py `
  --name 30k `
  --train-size 30000 `
  --val-size 1000 `
  --seed 1024 `
  --overwrite
```

Create exactly 400,000 recognition crops from those tables. A validation size
of zero keeps every crop from the 30k validation subset:

```powershell
.\.venv\Scripts\python.exe scripts\create_rec_crop_subset.py `
  --source-subset 30k `
  --name rec400k `
  --train-size 400000 `
  --val-size 0 `
  --seed 1024 `
  --overwrite
```

## 8. Train the three models

### 8.1 DB detector

```powershell
.\scripts\run_training.ps1 `
  -Component det `
  -Action train `
  -Subset 30k
```

Output:

```text
training/output/pubtabnet_det/
```

Best checkpoint: `best_accuracy`, selected by detector hmean.

### 8.2 SVTR_LCNet recognizer

```powershell
.\scripts\run_training.ps1 `
  -Component rec `
  -Action train `
  -Subset rec400k `
  -ConfigFile training\configs\pubtabnet_rec_400k.yml `
  -RunName pubtabnet_rec_400k
```

Output:

```text
training/output/pubtabnet_rec_400k/
```

Best checkpoint: `best_accuracy`, selected by exact recognition accuracy.

### 8.3 SLANet

```powershell
.\scripts\run_training.ps1 `
  -Component slanet `
  -Action train `
  -Subset 30k `
  -RunName pubtabnet_slanet_30k
```

Output:

```text
training/output/pubtabnet_slanet_30k/
```

Best checkpoint: `best_structure_score`, selected from token accuracy,
normalized edit similarity, TEDS-Structure, and valid HTML rate.

## 9. Stop, resume, and extend training

Press `Ctrl+C` once. The current batch finishes and `latest_step` is saved.
Pressing it a second time forces termination without waiting.

Resume each run:

```powershell
.\scripts\run_training.ps1 -Component det -Action train `
  -Subset 30k -Resume

.\scripts\run_training.ps1 -Component rec -Action train `
  -Subset rec400k `
  -ConfigFile training\configs\pubtabnet_rec_400k.yml `
  -RunName pubtabnet_rec_400k `
  -Resume

.\scripts\run_training.ps1 -Component slanet -Action train `
  -Subset 30k `
  -RunName pubtabnet_slanet_30k `
  -Resume
```

To extend the recognizer from 10 to 20 total epochs:

```powershell
.\scripts\run_training.ps1 -Component rec -Action train `
  -Subset rec400k `
  -ConfigFile training\configs\pubtabnet_rec_400k.yml `
  -RunName pubtabnet_rec_400k `
  -Resume `
  -Overrides "Global.epoch_num=20"
```

Use the same `Global.epoch_num=<total>` override for detector or SLANet. The
number is the target total epoch count, not the number of extra epochs.

## 10. Evaluate checkpoints

```powershell
.\scripts\run_training.ps1 -Component det -Action eval `
  -Checkpoint "../output/pubtabnet_det/best_accuracy"

.\scripts\run_training.ps1 -Component rec -Action eval `
  -Checkpoint "../output/pubtabnet_rec_400k/best_accuracy" `
  -ConfigFile training\configs\pubtabnet_rec_400k.yml `
  -RunName pubtabnet_rec_400k

.\scripts\run_training.ps1 -Component slanet -Action eval `
  -Checkpoint "../output/pubtabnet_slanet_30k/best_structure_score" `
  -RunName pubtabnet_slanet_30k
```

## 11. Obtain deployment checkpoints

Large checkpoints and ONNX models are not committed to Git. The recommended
Google Drive release bundle contains deployment-ready FP32 models:

[Download the pretrained models and checkpoints from Google Drive](https://drive.google.com/drive/folders/1JH42pMtsKQ1tRaoEezmb3kAaKzxIAAwf?usp=drive_link).

Expected archive content:

```text
models/
├── det/model_fp32.onnx
├── rec/model_fp32.onnx
├── table/model_fp32.onnx
└── dictionaries/
    ├── table_dict.txt
    └── table_structure_dict.txt
```

Download the files manually from Drive and place them in the expected paths.
Alternatively, install `gdown` inside `.venv` and download the shared folder:

```powershell
.\.venv\Scripts\python.exe -m pip install gdown
.\.venv\Scripts\gdown.exe --folder `
  "https://drive.google.com/drive/folders/1JH42pMtsKQ1tRaoEezmb3kAaKzxIAAwf?usp=drive_link" `
  --output downloaded_models
```

Copy the downloaded files into `models/det`, `models/rec`, `models/table`, and
`models/dictionaries` according to the structure above.

Verify all deployment files:

```powershell
Test-Path models\det\model_fp32.onnx
Test-Path models\rec\model_fp32.onnx
Test-Path models\table\model_fp32.onnx
Test-Path models\dictionaries\table_dict.txt
Test-Path models\dictionaries\table_structure_dict.txt
```

### Export your own trained checkpoints

Install export dependencies in `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-export.txt
```

Export the three best checkpoints to Paddle inference format, convert to ONNX,
copy dictionaries, and validate the generated artifacts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\export_deployment_models.ps1 `
  -Overwrite
```

Use `-SkipFp16` when only production FP32 artifacts are needed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\export_deployment_models.ps1 `
  -Overwrite `
  -SkipFp16
```

Validate again if required:

```powershell
.\.venv\Scripts\python.exe scripts\validate_onnx_models.py `
  --providers CPUExecutionProvider `
  --runs 3
```

## 12. Configure deployment

```powershell
if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
}
```

The default model contract is:

```text
Detector input: dynamic NCHW, side limit 736/min
Recognizer input: N × 3 × 48 × 320
SLANet input: NCHW padded to 488 × 488
```

Do not commit `.env`; only `.env.example` belongs in Git.

## 13. Local CPU deployment

Install CPU deployment packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-deploy.txt
```

Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn deploy.app:app `
  --host 0.0.0.0 `
  --port 8000 `
  --workers 1 `
  --env-file .env
```

## 14. Docker GPU deployment

Verify that Docker sees the NVIDIA GPU:

```powershell
docker run --rm --gpus all `
  nvidia/cuda:12.8.1-base-ubuntu24.04 `
  nvidia-smi
```

Build and start the measured hybrid FP32 profile:

```powershell
docker compose --profile gpu-hybrid up -d --build
```

Check startup and providers:

```powershell
docker compose --profile gpu-hybrid ps
Invoke-RestMethod http://localhost:8000/ready |
  ConvertTo-Json -Depth 8
```

Expected providers:

```text
detector   -> CUDAExecutionProvider, CPUExecutionProvider
recognizer -> CUDAExecutionProvider, CPUExecutionProvider
SLANet     -> CPUExecutionProvider
```

Other profiles:

```powershell
docker compose --profile cpu up -d --build
docker compose --profile gpu up -d --build
docker compose --profile gpu-mixed up -d --build
```

Use only one profile at a time because every service binds host port `8000`.

## 15. Call the API and Gradio UI

Endpoints:

- Gradio: `http://localhost:8000/ui`
- Swagger: `http://localhost:8000/docs`
- Liveness: `http://localhost:8000/live`
- Readiness: `http://localhost:8000/ready`
- JSON/HTML: `POST /v1/table/recognize`
- Excel: `POST /v1/table/excel`

Export one image to Excel:

```powershell
curl.exe --fail `
  -F "file=@path\to\table.png;type=image/png" `
  http://localhost:8000/v1/table/excel `
  --output table_result.xlsx
```

Return structured JSON and HTML:

```powershell
curl.exe --fail `
  -F "file=@path\to\table.png;type=image/png" `
  http://localhost:8000/v1/table/recognize
```

Run deployment smoke tests against local artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_deployment.py `
  path\to\table.png --env-file .env

.\.venv\Scripts\python.exe scripts\smoke_api.py `
  path\to\table.png --env-file .env
```

## 16. Benchmark

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_deployment.py `
  training\data\pubtabnet\val `
  --env-file .env `
  --limit 100 `
  --iterations 2 `
  --output output\deploy\benchmark.json
```

The report includes mean, p50, p95, p99, maximum latency, throughput, and
per-stage timing.

## 17. Stop deployment

```powershell
docker compose --profile gpu-hybrid down
```

## 18. Final verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
docker compose --profile cpu --profile gpu --profile gpu-hybrid `
  --profile gpu-mixed config --quiet
```

If all commands pass, the repository can reproduce preprocessing, training,
checkpoint resume, ONNX export, API inference, Gradio inference, and Excel
generation without committing datasets or model binaries.

## 19. Publish with Git Bash

Open Git Bash in the project root and configure your identity if needed:

```bash
git config user.name "YOUR_GITHUB_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

Initialize the local repository and inspect the files selected by `.gitignore`:

```bash
git init
git branch -M main
git status --short
git add .
git status --short
```

The staged set must not contain `.venv`, `.python312`, `.vendor`, `.env`,
`training/data`, `training/output`, `training/PaddleOCR`, `output`, or ONNX and
Paddle checkpoint files.

Create the first commit and connect the GitHub repository:

```bash
git commit -m "Initial public release"
git remote add origin https://github.com/trungphan27/AutoOCR_table_to_excel.git
git remote -v
git push -u origin main
```

If `origin` was configured previously, update it instead of adding it again:

```bash
git remote set-url origin https://github.com/trungphan27/AutoOCR_table_to_excel.git
git push -u origin main
```

If the GitHub repository was initialized with a README, license, or `.gitignore`,
it already has a remote commit. Merge it once before pushing:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

Never use `git push --force` for this initial merge. Resolve any reported file
conflict, stage the resolved file, and commit the merge normally.
