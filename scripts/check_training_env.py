"""Check the isolated training environment without changing it."""

import argparse
import ctypes
import importlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIGS = ("det", "rec", "slanet")
EXPECTED_SPLITS = ("train", "val")
GPU_DLL_DIRECTORIES = (
    PROJECT_ROOT
    / ".vendor"
    / "cudnn-8.9.6-cuda11"
    / "cudnn-windows-x86_64-8.9.6.50_cuda11-archive"
    / "bin",
    PROJECT_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "nvidia"
    / "cuda_runtime"
    / "bin",
    PROJECT_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "nvidia"
    / "cublas"
    / "bin",
)
GPU_DLL_HANDLES = []


def configure_local_gpu_dlls():
    existing = [path for path in GPU_DLL_DIRECTORIES if path.is_dir()]
    if existing:
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in existing), os.environ.get("PATH", "")]
        )
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for path in existing:
            GPU_DLL_HANDLES.append(os.add_dll_directory(str(path)))
    return existing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def package_versions():
    versions = {}
    for name in (
        "cv2",
        "imgaug",
        "numpy",
        "onnx",
        "onnxruntime",
        "paddle",
        "yaml",
    ):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:  # Preserve the underlying import failure.
            versions[name] = "ERROR: {}".format(exc)
    return versions


def cudnn_available():
    if os.name != "nt":
        return None
    try:
        ctypes.WinDLL("cudnn64_8.dll")
        return True
    except OSError:
        return False


def dataset_status():
    root = PROJECT_ROOT / "training" / "data" / "pubtabnet"
    missing = []
    for split in EXPECTED_SPLITS:
        if not (root / split).is_dir():
            missing.append(str(root / split))
        annotation = root / "PubTabNet_2.0.0_{}.jsonl".format(split)
        if not annotation.is_file():
            missing.append(str(annotation))
    return root, missing


def main():
    args = parse_args()
    failures = []
    warnings = []
    gpu_dll_directories = configure_local_gpu_dlls()

    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 is required; found {}".format(sys.version))
    expected_venv = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_venv:
        failures.append("command is not running from {}".format(expected_venv))

    upstream = PROJECT_ROOT / "training" / "PaddleOCR" / "tools" / "train.py"
    if not upstream.is_file():
        failures.append("missing PaddleOCR training entrypoint: {}".format(upstream))
    for component in EXPECTED_CONFIGS:
        config = (
            PROJECT_ROOT
            / "training"
            / "configs"
            / "pubtabnet_{}.yml".format(component)
        )
        if not config.is_file():
            failures.append("missing config: {}".format(config))

    versions = package_versions()
    for package, version in versions.items():
        if str(version).startswith("ERROR:"):
            failures.append("{} {}".format(package, version))

    paddle = None
    if not str(versions.get("paddle", "")).startswith("ERROR:"):
        paddle = importlib.import_module("paddle")
    gpu_build = bool(paddle and paddle.device.is_compiled_with_cuda())
    if args.require_gpu and not gpu_build:
        failures.append("installed PaddlePaddle is not a CUDA build")
    if gpu_build and cudnn_available() is False:
        message = "cudnn64_8.dll is not discoverable on PATH"
        if args.require_gpu:
            failures.append(message)
        else:
            warnings.append(message)

    required_gpu_dlls = {
        "cudnn64_8.dll": GPU_DLL_DIRECTORIES[0],
        "cudart64_110.dll": GPU_DLL_DIRECTORIES[1],
        "cublasLt64_11.dll": GPU_DLL_DIRECTORIES[2],
        "cublas64_11.dll": GPU_DLL_DIRECTORIES[2],
    }
    missing_gpu_dlls = [
        str(directory / filename)
        for filename, directory in required_gpu_dlls.items()
        if not (directory / filename).is_file()
    ]
    if missing_gpu_dlls:
        message = "project-local CUDA runtime is missing: {}".format(
            ", ".join(missing_gpu_dlls)
        )
        if args.require_gpu:
            failures.append(message)
        else:
            warnings.append(message)

    dataset_root, missing_data = dataset_status()
    if missing_data:
        message = "PubTabNet is incomplete ({} paths missing)".format(
            len(missing_data)
        )
        if args.require_data:
            failures.append(message)
        else:
            warnings.append(message)

    report = {
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "packages": versions,
        "paddle_cuda_build": gpu_build,
        "cudnn64_8_available": cudnn_available(),
        "gpu_dll_directories": [str(path) for path in gpu_dll_directories],
        "dataset_root": str(dataset_root),
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
