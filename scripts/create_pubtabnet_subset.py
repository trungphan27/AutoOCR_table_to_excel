"""Create a reproducible PubTabNet subset without copying image files.

The same sampled table records are used by the DB detector and SLANet.  The
recognition label subset contains every cell crop that belongs to those table
records, so the three training components remain aligned.
"""

import argparse
import json
import random
from pathlib import Path, PurePosixPath

from tqdm import tqdm

try:
    from scripts.prepare_pubtabnet import count_jsonl_lines
except ModuleNotFoundError:
    from prepare_pubtabnet import count_jsonl_lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create aligned DB, recognition and SLANet PubTabNet subsets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training/data/pubtabnet"),
    )
    parser.add_argument("--name", default="30k")
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--val-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def table_stem_from_rec_path(relative_path):
    crop_stem = Path(PurePosixPath(relative_path).name).stem
    table_stem, separator, cell_index = crop_stem.rpartition("_")
    if not separator or not cell_index.isdigit():
        return None
    return table_stem


def sample_annotation(
    source_path,
    output_path,
    sample_size,
    seed,
    show_progress=True,
):
    total = count_jsonl_lines(
        source_path,
        show_progress=show_progress,
        description="{}: indexing".format(source_path.stem),
    )
    if sample_size <= 0:
        raise ValueError("Subset size must be positive")
    if sample_size > total:
        raise ValueError(
            "Requested {} records from {}, but only {} are available".format(
                sample_size, source_path, total
            )
        )

    selected_indices = set(random.Random(seed).sample(range(total), sample_size))
    selected_filenames = set()
    selected_stems = set()
    written = 0
    file_size = source_path.stat().st_size

    with source_path.open("rb") as source, output_path.open("wb") as output, tqdm(
        total=file_size,
        desc="{}: SLANet labels".format(source_path.stem),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not show_progress,
    ) as progress:
        for index, line in enumerate(source):
            progress.update(len(line))
            if index not in selected_indices:
                continue
            record = json.loads(line)
            filename = record.get("filename")
            if not filename:
                raise ValueError(
                    "Missing filename at {} record {}".format(source_path, index + 1)
                )
            output.write(line if line.endswith(b"\n") else line + b"\n")
            selected_filenames.add(filename)
            selected_stems.add(Path(filename).stem)
            written += 1

    if written != sample_size:
        raise RuntimeError(
            "Expected {} sampled records, wrote {}".format(sample_size, written)
        )
    return total, selected_filenames, selected_stems


def filter_label_file(
    source_path,
    output_path,
    keep_line,
    description,
    show_progress=True,
):
    written = 0
    file_size = source_path.stat().st_size
    with source_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output, tqdm(
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
            progress.update(len(line.encode("utf-8")))
            relative_path = line.partition("\t")[0]
            if keep_line(relative_path):
                output.write(line if line.endswith("\n") else line + "\n")
                written += 1
    return written


def create_split_subset(
    dataset_root,
    output_dir,
    split,
    sample_size,
    seed,
    overwrite=False,
    show_progress=True,
):
    annotation_source = (
        dataset_root / "PubTabNet_2.0.0_{}.jsonl".format(split)
    )
    det_source = dataset_root / "derived" / "det_{}.txt".format(split)
    rec_source = dataset_root / "derived" / "rec_{}.txt".format(split)
    for source_path in (annotation_source, det_source, rec_source):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    annotation_output = (
        output_dir / "PubTabNet_2.0.0_{}.jsonl".format(split)
    )
    det_output = output_dir / "det_{}.txt".format(split)
    rec_output = output_dir / "rec_{}.txt".format(split)
    outputs = (annotation_output, det_output, rec_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Subset files already exist; pass --overwrite: {}".format(existing)
        )

    temporary_outputs = [path.with_suffix(path.suffix + ".tmp") for path in outputs]
    for temporary_path in temporary_outputs:
        temporary_path.unlink(missing_ok=True)

    annotation_tmp, det_tmp, rec_tmp = temporary_outputs
    success = False
    try:
        total, filenames, stems = sample_annotation(
            annotation_source,
            annotation_tmp,
            sample_size,
            seed,
            show_progress=show_progress,
        )
        det_count = filter_label_file(
            det_source,
            det_tmp,
            lambda relative_path: PurePosixPath(relative_path).name in filenames,
            "{}: DB labels".format(split),
            show_progress=show_progress,
        )
        rec_count = filter_label_file(
            rec_source,
            rec_tmp,
            lambda relative_path: table_stem_from_rec_path(relative_path) in stems,
            "{}: recognition labels".format(split),
            show_progress=show_progress,
        )
        success = True
    finally:
        if success:
            for temporary_path, output_path in zip(temporary_outputs, outputs):
                temporary_path.replace(output_path)
        else:
            for temporary_path in temporary_outputs:
                temporary_path.unlink(missing_ok=True)

    return {
        "source_tables": total,
        "tables": sample_size,
        "det_images": det_count,
        "rec_crops": rec_count,
        "seed": seed,
    }


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = dataset_root / "subsets" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress

    manifest = {
        "name": args.name,
        "dataset_root": str(dataset_root),
        "splits": {},
    }
    for offset, (split, sample_size) in enumerate(
        (("train", args.train_size), ("val", args.val_size))
    ):
        stats = create_split_subset(
            dataset_root,
            output_dir,
            split,
            sample_size,
            args.seed + offset,
            overwrite=args.overwrite,
            show_progress=show_progress,
        )
        manifest["splits"][split] = stats
        print("{}: {}".format(split, json.dumps(stats, sort_keys=True)))

    manifest_path = output_dir / "manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    print("subset: {}".format(output_dir))


if __name__ == "__main__":
    main()
