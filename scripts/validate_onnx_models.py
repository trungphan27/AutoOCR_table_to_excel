"""Validate deployment ONNX contracts and write a reproducible manifest."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


MODEL_SPECS = {
    "det": {
        "algorithm": "DB",
        "default": Path("models/det/model_fp32.onnx"),
        "shape": (1, 3, 736, 736),
        "minimum_outputs": 1,
        "checkpoint": "training/output/pubtabnet_det/best_accuracy",
    },
    "rec": {
        "algorithm": "SVTR_LCNet",
        "default": Path("models/rec/model_fp32.onnx"),
        "shape": (1, 3, 48, 320),
        "minimum_outputs": 1,
        "checkpoint": "training/output/pubtabnet_rec_400k/best_accuracy",
    },
    "table": {
        "algorithm": "SLANet",
        "default": Path("models/table/model_fp32.onnx"),
        "shape": (1, 3, 488, 488),
        "minimum_outputs": 2,
        "checkpoint": (
            "training/output/pubtabnet_slanet_30k/best_structure_score"
        ),
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metadata(values):
    return [
        {"name": value.name, "shape": value.shape, "type": value.type}
        for value in values
    ]


def provider_list(raw):
    providers = [value.strip() for value in raw.split(",") if value.strip()]
    if not providers:
        raise ValueError("At least one execution provider is required.")
    available = ort.get_available_providers()
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise RuntimeError(
            "Providers unavailable: {}. Available: {}".format(missing, available)
        )
    return providers


def validate_model(name, path, providers, run_model=True, runs=1):
    spec = MODEL_SPECS[name]
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    onnx_model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        str(path), sess_options=session_options, providers=providers
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError("{} must expose exactly one input.".format(name))
    if len(inputs[0].shape) != 4:
        raise RuntimeError("{} input must be rank-4 NCHW.".format(name))
    if len(outputs) < spec["minimum_outputs"]:
        raise RuntimeError(
            "{} exposes {} outputs; expected at least {}.".format(
                name, len(outputs), spec["minimum_outputs"]
            )
        )

    elapsed_ms = None
    output_shapes = None
    if run_model:
        input_dtype = (
            np.float16 if inputs[0].type == "tensor(float16)" else np.float32
        )
        sample = np.zeros(spec["shape"], dtype=input_dtype)
        session.run(None, {inputs[0].name: sample})
        timings = []
        values = None
        for _ in range(runs):
            started = time.perf_counter()
            values = session.run(None, {inputs[0].name: sample})
            timings.append((time.perf_counter() - started) * 1000)
        elapsed_ms = round(float(np.median(timings)), 3)
        output_shapes = [list(value.shape) for value in values]
        if name == "table":
            has_locations = any(
                value.ndim >= 2 and value.shape[-1] == 4 for value in values
            )
            has_structure = any(
                value.ndim >= 2 and value.shape[-1] != 4 for value in values
            )
            if not has_locations or not has_structure:
                raise RuntimeError(
                    "SLANet outputs do not contain both locations and structure."
                )

    return {
        "algorithm": spec["algorithm"],
        "checkpoint": spec["checkpoint"],
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "opset": max(opset.version for opset in onnx_model.opset_import),
        "inputs": tensor_metadata(inputs),
        "outputs": tensor_metadata(outputs),
        "runtime_output_shapes": output_shapes,
        "median_raw_session_ms": elapsed_ms,
        "providers": session.get_providers(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name, spec in MODEL_SPECS.items():
        parser.add_argument("--{}-model".format(name), type=Path, default=spec["default"])
    parser.add_argument(
        "--rec-dict",
        type=Path,
        default=Path("models/dictionaries/table_dict.txt"),
    )
    parser.add_argument(
        "--table-dict",
        type=Path,
        default=Path("models/dictionaries/table_structure_dict.txt"),
    )
    parser.add_argument(
        "--providers", default="CPUExecutionProvider"
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--manifest", type=Path, default=Path("models/manifest.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least one.")
    providers = provider_list(args.providers)
    model_paths = {
        "det": args.det_model,
        "rec": args.rec_model,
        "table": args.table_model,
    }
    models = {
        name: validate_model(
            name,
            path,
            providers,
            run_model=not args.skip_run,
            runs=args.runs,
        )
        for name, path in model_paths.items()
    }

    dictionaries = {}
    for name, path in {
        "recognizer": args.rec_dict,
        "table": args.table_dict,
    }.items():
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        dictionaries[name] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing": {
            "det_limit_side_len": 736,
            "det_limit_type": "min",
            "rec_image_shape": [3, 48, 320],
            "table_max_len": 488,
        },
        "models": models,
        "dictionaries": dictionaries,
    }
    destination = args.manifest.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print("validated models and saved {}".format(destination))


if __name__ == "__main__":
    main()
