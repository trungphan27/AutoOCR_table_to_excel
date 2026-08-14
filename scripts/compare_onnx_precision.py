"""Compare FP16 end-to-end output and latency against the FP32 baseline."""

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from lxml import html as lxml_html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.config import DeploymentSettings
from deploy.envfile import load_env_file
from deploy.service import TableInferenceService
from table_metric import TEDS


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_images(path, limit):
    path = path.expanduser().resolve()
    if path.is_file():
        images = [path]
    elif path.is_dir():
        images = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(path)
    if limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError("No comparison images were found.")
    return images


def valid_html(value):
    try:
        tree = lxml_html.fromstring(value)
        return bool(tree.xpath("//table"))
    except Exception:
        return False


def run_service(settings, images):
    service = TableInferenceService(settings)
    results = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode {}".format(path))
        started = time.perf_counter()
        payload = service.predict_image(image)
        results.append(
            {
                "path": path.as_posix(),
                "html": payload["result"]["html"],
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--fp16-det", type=Path, default=Path("models/det/model_fp16.onnx"))
    parser.add_argument("--fp16-rec", type=Path, default=Path("models/rec/model_fp16.onnx"))
    parser.add_argument("--fp16-table", type=Path, default=Path("models/table/model_fp16.onnx"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_env_file(args.env_file)
    images = collect_images(args.images, args.limit)
    fp32_settings = DeploymentSettings.from_env()
    fp16_settings = replace(
        fp32_settings,
        det_model=args.fp16_det.expanduser().resolve(),
        rec_model=args.fp16_rec.expanduser().resolve(),
        table_model=args.fp16_table.expanduser().resolve(),
    )
    for path in (
        fp16_settings.det_model,
        fp16_settings.rec_model,
        fp16_settings.table_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    fp32 = run_service(fp32_settings, images)
    gc.collect()
    fp16 = run_service(fp16_settings, images)

    full_metric = TEDS(structure_only=False, n_jobs=1)
    structure_metric = TEDS(structure_only=True, n_jobs=1)
    rows = []
    for baseline, candidate in zip(fp32, fp16):
        rows.append(
            {
                "path": baseline["path"],
                "full_teds_vs_fp32": full_metric.evaluate(
                    candidate["html"], baseline["html"]
                ),
                "structure_teds_vs_fp32": structure_metric.evaluate(
                    candidate["html"], baseline["html"]
                ),
                "fp32_latency_ms": baseline["latency_ms"],
                "fp16_latency_ms": candidate["latency_ms"],
                "fp16_valid_html": valid_html(candidate["html"]),
                "exact_html": candidate["html"] == baseline["html"],
            }
        )

    report = {
        "providers": fp32_settings.providers,
        "images": len(rows),
        "mean_full_teds_vs_fp32": float(
            np.mean([row["full_teds_vs_fp32"] for row in rows])
        ),
        "mean_structure_teds_vs_fp32": float(
            np.mean([row["structure_teds_vs_fp32"] for row in rows])
        ),
        "minimum_structure_teds_vs_fp32": float(
            np.min([row["structure_teds_vs_fp32"] for row in rows])
        ),
        "exact_html_rate": float(np.mean([row["exact_html"] for row in rows])),
        "fp16_valid_html_rate": float(
            np.mean([row["fp16_valid_html"] for row in rows])
        ),
        "mean_fp32_latency_ms": float(
            np.mean([row["fp32_latency_ms"] for row in rows])
        ),
        "mean_fp16_latency_ms": float(
            np.mean([row["fp16_latency_ms"] for row in rows])
        ),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
