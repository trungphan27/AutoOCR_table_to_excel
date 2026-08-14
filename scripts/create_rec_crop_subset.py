"""Create a reproducible recognition-crop subset without copying images.

This script samples recognition labels from an existing PubTabNet subset.  It
is intentionally separate from ``create_pubtabnet_subset.py`` because a table
subset can expand to millions of cell crops.  Detector and SLANet labels remain
untouched; only ``rec_train.txt`` and ``rec_val.txt`` are written.
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample an exact number of PubTabNet recognition crops."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training/data/pubtabnet"),
    )
    parser.add_argument(
        "--source-subset",
        default="30k",
        help="Existing subset containing rec_train.txt and rec_val.txt.",
    )
    parser.add_argument("--name", default="rec400k")
    parser.add_argument("--train-size", type=int, default=400_000)
    parser.add_argument(
        "--val-size",
        type=int,
        default=0,
        help="Number of validation crops; 0 keeps the complete source validation set.",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def count_lines(path, description, show_progress=True):
    total = 0
    file_size = path.stat().st_size
    with path.open("rb") as source, tqdm(
        total=file_size,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not show_progress,
    ) as progress:
        for line in source:
            progress.update(len(line))
            total += 1
    return total


def sample_label_file(
    source_path,
    output_path,
    sample_size,
    seed,
    description,
    show_progress=True,
):
    total = count_lines(
        source_path,
        "{}: indexing".format(description),
        show_progress=show_progress,
    )
    if sample_size < 0:
        raise ValueError("Subset size cannot be negative")
    if sample_size > total:
        raise ValueError(
            "Requested {} labels from {}, but only {} are available".format(
                sample_size, source_path, total
            )
        )

    keep_all = sample_size == 0 or sample_size == total
    target_size = total if sample_size == 0 else sample_size
    selected_indices = None
    if not keep_all:
        selected_indices = set(
            random.Random(seed).sample(range(total), target_size)
        )

    written = 0
    file_size = source_path.stat().st_size
    with source_path.open("rb") as source, output_path.open("wb") as output, tqdm(
        total=file_size,
        desc="{}: writing".format(description),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not show_progress,
    ) as progress:
        for index, line in enumerate(source):
            progress.update(len(line))
            if selected_indices is not None and index not in selected_indices:
                continue
            output.write(line if line.endswith(b"\n") else line + b"\n")
            written += 1

    if written != target_size:
        raise RuntimeError(
            "Expected {} sampled labels, wrote {}".format(target_size, written)
        )
    return {"source_crops": total, "crops": written, "seed": seed}


def main():
    args = parse_args()
    if args.train_size <= 0:
        raise ValueError("--train-size must be greater than zero")
    if args.val_size < 0:
        raise ValueError("--val-size cannot be negative")

    dataset_root = args.dataset_root.expanduser().resolve()
    source_dir = dataset_root / "subsets" / args.source_subset
    output_dir = dataset_root / "subsets" / args.name
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Source and output subsets must have different names")

    train_source = source_dir / "rec_train.txt"
    val_source = source_dir / "rec_val.txt"
    for source_path in (train_source, val_source):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "rec_train.txt"
    val_output = output_dir / "rec_val.txt"
    manifest_path = output_dir / "manifest.json"
    outputs = (train_output, val_output, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Subset files already exist; pass --overwrite: {}".format(existing)
        )

    train_tmp = train_output.with_suffix(".txt.tmp")
    val_tmp = val_output.with_suffix(".txt.tmp")
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    temporary_outputs = (train_tmp, val_tmp, manifest_tmp)
    for temporary_path in temporary_outputs:
        temporary_path.unlink(missing_ok=True)

    show_progress = not args.no_progress
    success = False
    try:
        train_stats = sample_label_file(
            train_source,
            train_tmp,
            args.train_size,
            args.seed,
            "train recognition labels",
            show_progress=show_progress,
        )
        val_stats = sample_label_file(
            val_source,
            val_tmp,
            args.val_size,
            args.seed + 1,
            "validation recognition labels",
            show_progress=show_progress,
        )
        manifest = {
            "name": args.name,
            "dataset_root": str(dataset_root),
            "source_subset": args.source_subset,
            "images_copied": False,
            "splits": {"train": train_stats, "val": val_stats},
        }
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        success = True
    finally:
        if success:
            for temporary_path, output_path in zip(temporary_outputs, outputs):
                temporary_path.replace(output_path)
        else:
            for temporary_path in temporary_outputs:
                temporary_path.unlink(missing_ok=True)

    print("train: {}".format(json.dumps(train_stats, sort_keys=True)))
    print("val: {}".format(json.dumps(val_stats, sort_keys=True)))
    print("subset: {}".format(output_dir))


if __name__ == "__main__":
    main()
