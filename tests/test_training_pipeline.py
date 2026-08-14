import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scripts.prepare_pubtabnet import (
    cell_markup,
    count_jsonl_lines,
    prepare_split,
    visible_text,
)
from scripts.create_pubtabnet_subset import (
    create_split_subset,
    table_stem_from_rec_path,
)
from table_metric import TEDS


class PreparePubTabNetTest(unittest.TestCase):
    def test_count_jsonl_lines_handles_missing_final_newline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation = Path(temporary_directory) / "sample.jsonl"
            annotation.write_bytes(b'{"row": 1}\n{"row": 2}')
            self.assertEqual(
                count_jsonl_lines(annotation, show_progress=False), 2
            )

    def test_cell_text_keeps_formatting_for_recognition(self):
        tokens = ["<b>", "Net", "&amp;", " Gross", "</b>"]
        self.assertEqual(cell_markup(tokens), "<b>Net& Gross</b>")
        self.assertEqual(visible_text(tokens), "Net& Gross")

    def test_prepare_split_writes_detector_and_recognizer_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "train").mkdir()
            image = np.full((60, 120, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "train" / "sample.png"), image))
            record = {
                "filename": "sample.png",
                "html": {
                    "cells": [
                        {"tokens": ["<b>", "ABC", "</b>"], "bbox": [5, 6, 80, 30]}
                    ],
                    "structure": {"tokens": ["<table>", "<tr>", "<td>", "</td>", "</tr>", "</table>"]},
                },
            }
            annotation = root / "PubTabNet_2.0.0_train.jsonl"
            annotation.write_text(json.dumps(record) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                tasks=["validate", "det", "rec"],
                overwrite=False,
                limit=0,
                max_structure_length=500,
                max_rec_length=100,
                crop_padding=2,
                allow_missing_images=False,
                no_progress=True,
            )

            stats = prepare_split(args, root, "train")

            self.assertEqual(stats["det_images"], 1)
            self.assertEqual(stats["rec_crops"], 1)
            det_label = (root / "derived" / "det_train.txt").read_text(
                encoding="utf-8"
            )
            rec_label = (root / "derived" / "rec_train.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn('"transcription": "ABC"', det_label)
            self.assertIn("\t<b>ABC</b>", rec_label)


class CreatePubTabNetSubsetTest(unittest.TestCase):
    def test_table_stem_from_rec_path(self):
        path = "derived/rec/train/PM/PMC4840965_004_00_0012.png"
        self.assertEqual(
            table_stem_from_rec_path(path), "PMC4840965_004_00"
        )

    def test_subset_keeps_aligned_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derived = root / "derived"
            output = root / "subsets" / "sample"
            derived.mkdir(parents=True)
            output.mkdir(parents=True)

            records = []
            det_lines = []
            rec_lines = []
            for index in range(4):
                filename = "table_{}.png".format(index)
                records.append(json.dumps({"filename": filename, "html": {}}))
                det_lines.append("train/{}\t[]".format(filename))
                for cell_index in range(2):
                    rec_lines.append(
                        "derived/rec/train/ta/table_{}_{:04d}.png\tcell".format(
                            index, cell_index
                        )
                    )

            (root / "PubTabNet_2.0.0_train.jsonl").write_text(
                "\n".join(records) + "\n", encoding="utf-8"
            )
            (derived / "det_train.txt").write_text(
                "\n".join(det_lines) + "\n", encoding="utf-8"
            )
            (derived / "rec_train.txt").write_text(
                "\n".join(rec_lines) + "\n", encoding="utf-8"
            )

            stats = create_split_subset(
                root,
                output,
                "train",
                sample_size=2,
                seed=7,
                show_progress=False,
            )

            self.assertEqual(stats["tables"], 2)
            self.assertEqual(stats["det_images"], 2)
            self.assertEqual(stats["rec_crops"], 4)
            selected = {
                json.loads(line)["filename"]
                for line in (output / "PubTabNet_2.0.0_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            det_selected = {
                Path(line.partition("\t")[0]).name
                for line in (output / "det_train.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            self.assertEqual(selected, det_selected)


class TedsTest(unittest.TestCase):
    def test_identical_tables_score_one(self):
        table = "<html><body><table><tr><td>A</td></tr></table></body></html>"
        self.assertEqual(TEDS().evaluate(table, table), 1.0)


if __name__ == "__main__":
    unittest.main()
