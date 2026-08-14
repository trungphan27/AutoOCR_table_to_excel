"""Validate PubTabNet and derive DB/recognition training labels.

SLANet consumes the original PubTabNet JSONL files directly. This script keeps
those annotations unchanged and creates only the additional artifacts required
by the OCR detector and recognizer.
"""

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
from tqdm import tqdm


TAG_PATTERN = re.compile(r"</?[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def cell_markup(tokens):
    value = "".join(str(token) for token in (tokens or []))
    value = value.replace("\u2028", " ").replace("\u2029", " ")
    value = html.unescape(value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def visible_text(tokens):
    value = cell_markup(tokens)
    value = TAG_PATTERN.sub("", value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def normalized_bbox(raw_bbox, width, height, padding=0):
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    left = max(0, int(math.floor(min(x1, x2))) - padding)
    top = max(0, int(math.floor(min(y1, y2))) - padding)
    right = min(width, int(math.ceil(max(x1, x2))) + padding)
    bottom = min(height, int(math.ceil(max(y1, y2))) + padding)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def posix_relative(*parts):
    return str(PurePosixPath(*(str(part) for part in parts)))


def count_jsonl_lines(path, show_progress=True, description="indexing"):
    """Count JSONL records without loading the annotation file into memory."""
    file_size = path.stat().st_size
    line_count = 0
    last_byte = b""
    with path.open("rb") as source, tqdm(
        total=file_size,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
            progress.update(len(chunk))
    if file_size and last_byte != b"\n":
        line_count += 1
    return line_count


def progress_postfix(stats):
    return {
        "images": stats["det_images"],
        "boxes": stats["det_boxes"],
        "crops": stats["rec_crops"],
        "empty": stats["empty_cells"],
        "long": stats["rec_too_long"],
        "structure": stats["invalid_structure_length"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare English PubTabNet labels for DB, OCR and SLANet."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training/data/pubtabnet"),
        help=(
            "Directory containing train/, val/ and "
            "PubTabNet_2.0.0_<split>.jsonl."
        ),
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val"]
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["validate", "det", "rec"],
        choices=["validate", "det", "rec"],
    )
    parser.add_argument("--crop-padding", type=int, default=2)
    parser.add_argument("--max-rec-length", type=int, default=100)
    parser.add_argument("--max-structure-length", type=int, default=500)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N rows per split."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (useful for CI or redirected logs).",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Complete with warnings instead of failing when images are missing.",
    )
    return parser.parse_args()


def prepare_split(args, dataset_root, split):
    annotation_path = dataset_root / "PubTabNet_2.0.0_{}.jsonl".format(split)
    image_dir = dataset_root / split
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    derived_dir = dataset_root / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    det_output = derived_dir / "det_{}.txt".format(split)
    rec_output = derived_dir / "rec_{}.txt".format(split)
    requested_outputs = []
    if "det" in args.tasks:
        requested_outputs.append(det_output)
    if "rec" in args.tasks:
        requested_outputs.append(rec_output)
    existing = [path for path in requested_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Derived labels already exist; pass --overwrite: {}".format(existing)
        )

    det_tmp = det_output.with_suffix(det_output.suffix + ".tmp")
    rec_tmp = rec_output.with_suffix(rec_output.suffix + ".tmp")
    det_handle = None
    rec_handle = None
    stats = Counter()
    success = False
    show_progress = not getattr(args, "no_progress", False)
    total_records = None
    if show_progress:
        if args.limit:
            total_records = args.limit
        else:
            total_records = count_jsonl_lines(
                annotation_path,
                show_progress=True,
                description="{}: indexing".format(split),
            )

    try:
        if "det" in args.tasks:
            det_handle = det_tmp.open("w", encoding="utf-8", newline="\n")
        if "rec" in args.tasks:
            rec_handle = rec_tmp.open("w", encoding="utf-8", newline="\n")

        with annotation_path.open("r", encoding="utf-8") as source, tqdm(
            total=total_records,
            desc="{}: preprocess".format(split),
            unit="table",
            dynamic_ncols=True,
            mininterval=0.5,
            disable=not show_progress,
        ) as progress:
            for line_number, line in enumerate(source, start=1):
                if args.limit and stats["rows"] >= args.limit:
                    break
                progress.update(1)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSON at {}:{}".format(annotation_path, line_number)
                    ) from exc

                filename = record.get("filename")
                cells = record.get("html", {}).get("cells", [])
                structure = record.get("html", {}).get("structure", {}).get(
                    "tokens", []
                )
                if not filename or not isinstance(cells, list):
                    stats["invalid_records"] += 1
                    continue
                stats["rows"] += 1
                if not structure or len(structure) > args.max_structure_length:
                    stats["invalid_structure_length"] += 1

                image_path = image_dir / filename
                if not image_path.is_file():
                    stats["missing_images"] += 1
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    stats["unreadable_images"] += 1
                    continue
                height, width = image.shape[:2]
                det_items = []

                for cell_index, cell in enumerate(cells):
                    tokens = cell.get("tokens", [])
                    det_text = visible_text(tokens)
                    rec_text = cell_markup(tokens)
                    if not det_text:
                        stats["empty_cells"] += 1
                        continue
                    bbox = normalized_bbox(cell.get("bbox"), width, height)
                    if bbox is None:
                        stats["invalid_bboxes"] += 1
                        continue
                    left, top, right, bottom = bbox
                    points = [
                        [left, top],
                        [right, top],
                        [right, bottom],
                        [left, bottom],
                    ]
                    det_items.append(
                        {"transcription": det_text, "points": points}
                    )
                    stats["det_boxes"] += 1

                    if rec_handle is None:
                        continue
                    if len(rec_text) > args.max_rec_length:
                        stats["rec_too_long"] += 1
                        continue
                    crop_bbox = normalized_bbox(
                        cell.get("bbox"),
                        width,
                        height,
                        padding=max(0, args.crop_padding),
                    )
                    if crop_bbox is None:
                        stats["invalid_rec_bboxes"] += 1
                        continue
                    crop_left, crop_top, crop_right, crop_bottom = crop_bbox
                    crop = image[crop_top:crop_bottom, crop_left:crop_right]
                    prefix = Path(filename).stem[:2] or "__"
                    crop_name = "{}_{:04d}.png".format(Path(filename).stem, cell_index)
                    crop_relative = Path("derived") / "rec" / split / prefix / crop_name
                    crop_path = dataset_root / crop_relative
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.overwrite or not crop_path.exists():
                        if not cv2.imwrite(str(crop_path), crop):
                            stats["crop_write_errors"] += 1
                            continue
                    rec_handle.write(
                        "{}\t{}\n".format(
                            posix_relative(*crop_relative.parts),
                            rec_text.replace("\t", " "),
                        )
                    )
                    stats["rec_crops"] += 1

                if det_handle is not None and det_items:
                    det_handle.write(
                        "{}\t{}\n".format(
                            posix_relative(split, filename),
                            json.dumps(det_items, ensure_ascii=False),
                        )
                    )
                    stats["det_images"] += 1

                if progress.n % 100 == 0:
                    progress.set_postfix(progress_postfix(stats), refresh=False)

            progress.set_postfix(progress_postfix(stats), refresh=False)

        success = True
    finally:
        if det_handle is not None:
            det_handle.close()
        if rec_handle is not None:
            rec_handle.close()
        if success:
            if det_handle is not None:
                det_tmp.replace(det_output)
            if rec_handle is not None:
                rec_tmp.replace(rec_output)
        else:
            det_tmp.unlink(missing_ok=True)
            rec_tmp.unlink(missing_ok=True)

    print("{}: {}".format(split, json.dumps(dict(stats), sort_keys=True)))
    missing = stats["missing_images"] + stats["unreadable_images"]
    if missing and not args.allow_missing_images:
        raise RuntimeError(
            "{} {} image(s) were missing or unreadable".format(split, missing)
        )
    return stats


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    totals = Counter()
    for split in args.splits:
        totals.update(prepare_split(args, dataset_root, split))
    print("total: {}".format(json.dumps(dict(totals), sort_keys=True)))
    if totals["invalid_records"] or totals["invalid_structure_length"]:
        print(
            "warning: some records are unsuitable for SLANet; inspect the counters above",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
