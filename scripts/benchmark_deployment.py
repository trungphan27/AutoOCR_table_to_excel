"""Benchmark warm end-to-end deployment latency on real table images."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.envfile import load_env_file
from deploy.service import TableInferenceService


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_images(path, limit):
    path = path.expanduser().resolve()
    if path.is_file():
        images = [path]
    elif path.is_dir():
        images = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(path)
    if limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError("No benchmark images were found.")
    return images


def percentile(values, quantile):
    return round(float(np.percentile(values, quantile)), 3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least one.")
    load_env_file(args.env_file)
    images = collect_images(args.images, args.limit)
    service = TableInferenceService.from_env()

    decoded = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode {}".format(path))
        decoded.append((path, image))

    totals = []
    stage_values = {name: [] for name in ("det", "rec", "table", "match", "all")}
    started_all = time.perf_counter()
    for _ in range(args.iterations):
        for _, image in decoded:
            started = time.perf_counter()
            payload = service.predict_image(image)
            totals.append((time.perf_counter() - started) * 1000)
            for name in stage_values:
                if name in payload["timing"]:
                    stage_values[name].append(payload["timing"][name] * 1000)
    wall_seconds = time.perf_counter() - started_all

    report = {
        "providers": service.provider_info(),
        "images": len(decoded),
        "requests": len(totals),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_requests_per_second": round(len(totals) / wall_seconds, 3),
        "latency_ms": {
            "mean": round(float(np.mean(totals)), 3),
            "p50": percentile(totals, 50),
            "p95": percentile(totals, 95),
            "p99": percentile(totals, 99),
            "max": round(float(np.max(totals)), 3),
        },
        "mean_stage_ms": {
            name: round(float(np.mean(values)), 3) if values else None
            for name, values in stage_values.items()
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
