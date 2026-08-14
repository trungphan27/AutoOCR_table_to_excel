import sys
import unittest
from pathlib import Path


PADDLE_OCR_ROOT = (
    Path(__file__).resolve().parents[1] / "training" / "PaddleOCR"
)
for module_name in list(sys.modules):
    if module_name == "ppocr" or module_name.startswith("ppocr."):
        del sys.modules[module_name]
sys.path.insert(0, str(PADDLE_OCR_ROOT))

from ppocr.metrics.table_metric import TableStructureMetric


EPS = 1e-6


def metric_input(predictions, targets):
    return (
        {
            "structure_batch_list": [
                [tokens, 0.99] for tokens in predictions
            ]
        },
        {"structure_batch_list": targets},
    )


class TableStructureMetricTest(unittest.TestCase):
    def make_metric(self, **kwargs):
        defaults = {
            "compute_token_metrics": True,
            "compute_teds_structure": True,
            "show_progress_metrics": True,
            "teds_n_jobs": 1,
        }
        defaults.update(kwargs)
        return TableStructureMetric(**defaults)

    def test_identical_sequences_score_one_with_existing_epsilon_policy(self):
        tokens = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        metric = self.make_metric()
        metric(metric_input([tokens], [tokens]))

        result = metric.get_metric()

        expected = 1.0 / (1.0 + EPS)
        self.assertAlmostEqual(result["acc"], expected)
        self.assertAlmostEqual(result["token_acc"], 5.0 / (5.0 + EPS))
        self.assertAlmostEqual(result["norm_edit_dis"], expected)
        self.assertAlmostEqual(result["teds_structure"], expected)
        self.assertEqual(result["invalid_html"], 0)
        self.assertAlmostEqual(result["valid_html_rate"], 1.0)
        expected_score = (
            0.2 * result["token_acc"]
            + 0.2 * result["norm_edit_dis"]
            + 0.4 * result["teds_structure"]
            + 0.2 * result["valid_html_rate"]
        )
        self.assertAlmostEqual(result["structure_score"], expected_score)

    def test_one_replacement_reduces_every_near_match_metric(self):
        target = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        pred = ["<tbody>", "<tr>", '<td colspan="2"></td>', "</tr>", "</tbody>"]
        metric = self.make_metric()
        metric(metric_input([pred], [target]))

        result = metric.get_metric()

        self.assertEqual(result["acc"], 0.0)
        self.assertGreater(result["token_acc"], 0.0)
        self.assertLess(result["token_acc"], 1.0)
        self.assertGreater(result["norm_edit_dis"], 0.0)
        self.assertLess(result["norm_edit_dis"], 1.0)
        self.assertGreaterEqual(result["teds_structure"], 0.0)
        self.assertLess(result["teds_structure"], 1.0)

    def test_alignment_handles_insertion_without_shifting_the_suffix(self):
        target = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        pred = ["<tbody>", "<tr>", "<td></td>", "<td></td>", "</tr>", "</tbody>"]
        metric = self.make_metric(compute_teds_structure=False)
        metric(metric_input([pred], [target]))

        result = metric.get_metric()

        self.assertAlmostEqual(result["token_acc"], 5.0 / (6.0 + EPS))
        self.assertAlmostEqual(
            result["norm_edit_dis"], (1.0 - 1.0 / 6.0) / (1.0 + EPS)
        )

    def test_deletion_is_penalized(self):
        target = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        pred = ["<tbody>", "<tr>", "</tr>", "</tbody>"]
        metric = self.make_metric(compute_teds_structure=False)
        metric(metric_input([pred], [target]))

        result = metric.get_metric()

        self.assertAlmostEqual(result["token_acc"], 4.0 / (5.0 + EPS))
        self.assertAlmostEqual(
            result["norm_edit_dis"], (1.0 - 1.0 / 5.0) / (1.0 + EPS)
        )

    def test_rowspan_difference_reduces_teds_structure(self):
        target = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        pred = [
            "<tbody>", "<tr>", "<td", ' rowspan="2"', ">", "</td>",
            "</tr>", "</tbody>"
        ]
        metric = self.make_metric()
        metric(metric_input([pred], [target]))

        result = metric.get_metric()

        self.assertGreaterEqual(result["teds_structure"], 0.0)
        self.assertLess(result["teds_structure"], 1.0)

    def test_malformed_prediction_is_scored_zero_without_raising(self):
        target = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        pred = ["<tbody>", "<tr>", "<td></td>", "</tbody>"]
        metric = self.make_metric()

        metric(metric_input([pred], [target]))
        result = metric.get_metric()

        self.assertEqual(result["teds_structure"], 0.0)
        self.assertEqual(result["invalid_html"], 1)
        self.assertAlmostEqual(result["invalid_html_rate"], 1.0 / (1.0 + EPS))
        self.assertLess(result["valid_html_rate"], 1e-5)

    def test_structure_score_rewards_teds_and_valid_html(self):
        target = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        close_prediction = [
            "<tbody>", "<tr>", '<td colspan="2"></td>', "</tr>",
            "</tbody>"
        ]
        invalid_prediction = [
            "<tbody>", "<tr>", "<td></td>", "</tbody>"
        ]
        close_metric = self.make_metric()
        invalid_metric = self.make_metric()

        close_metric(metric_input([close_prediction], [target]))
        invalid_metric(metric_input([invalid_prediction], [target]))

        close_result = close_metric.get_metric()
        invalid_result = invalid_metric.get_metric()
        self.assertGreater(
            close_result["structure_score"],
            invalid_result["structure_score"],
        )

    def test_structure_score_weights_are_normalized(self):
        tokens = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        metric = self.make_metric(
            structure_score_weights={
                "token_acc": 1,
                "norm_edit_dis": 1,
                "teds_structure": 2,
                "valid_html_rate": 1,
            }
        )
        metric(metric_input([tokens], [tokens]))

        result = metric.get_metric()

        expected = (
            0.2 * result["token_acc"]
            + 0.2 * result["norm_edit_dis"]
            + 0.4 * result["teds_structure"]
            + 0.2 * result["valid_html_rate"]
        )
        self.assertAlmostEqual(result["structure_score"], expected)

    def test_invalid_structure_score_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_metric(
                structure_score_weights={"unsupported_metric": 1.0}
            )

    def test_structure_main_indicator_requires_all_source_metrics(self):
        from ppocr.metrics.table_metric import TableMetric

        with self.assertRaises(ValueError):
            TableMetric(
                main_indicator="structure_score",
                compute_token_metrics=True,
                compute_teds_structure=False,
            )

    def test_empty_sequence_edge_cases(self):
        metric = self.make_metric()
        metric(metric_input([[], []], [[], ["<tbody>", "</tbody>"]]))

        result = metric.get_metric()

        self.assertGreater(result["token_acc"], 0.0)
        self.assertGreater(result["norm_edit_dis"], 0.0)
        self.assertGreater(result["teds_structure"], 0.0)
        self.assertLess(result["acc"], 1.0)

    def test_multiple_batches_accumulate_before_final_reset(self):
        tokens = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        wrong = ["<tbody>", "<tr>", "</tr>", "</tbody>"]
        metric = self.make_metric(compute_teds_structure=False)
        metric(metric_input([tokens], [tokens]))
        metric(metric_input([wrong], [tokens]))

        progress = metric.get_progress_metrics()
        final = metric.get_metric()
        after_reset = metric.get_progress_metrics()

        self.assertAlmostEqual(progress["acc"], 1.0 / (2.0 + EPS))
        self.assertEqual(progress, final)
        self.assertEqual(after_reset["acc"], 0.0)

    def test_progress_read_does_not_reset_accumulators(self):
        tokens = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        metric = self.make_metric(compute_teds_structure=False)
        metric(metric_input([tokens], [tokens]))

        first = metric.get_progress_metrics()
        second = metric.get_progress_metrics()
        final = metric.get_metric()

        self.assertEqual(first, second)
        self.assertEqual(second, final)

    def test_del_thead_tbody_is_shared_by_all_sequence_metrics(self):
        pred = [
            "<thead>", "<tr>", "<td></td>", "</tr>", "</thead>",
            "<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"
        ]
        target = [
            "<tr>", "<td></td>", "</tr>", "<tr>", "<td></td>", "</tr>"
        ]
        metric = self.make_metric(del_thead_tbody=True)
        metric(metric_input([pred], [target]))

        result = metric.get_metric()

        self.assertAlmostEqual(result["acc"], 1.0 / (1.0 + EPS))
        self.assertAlmostEqual(result["token_acc"], 6.0 / (6.0 + EPS))
        self.assertAlmostEqual(result["norm_edit_dis"], 1.0 / (1.0 + EPS))

    def test_disabled_features_preserve_exact_accuracy_only_output(self):
        tokens = ["<tbody>", "<tr>", "</tr>", "</tbody>"]
        metric = TableStructureMetric()
        metric(metric_input([tokens], [tokens]))

        result = metric.get_metric()

        self.assertEqual(set(result), {"acc"})
        self.assertAlmostEqual(result["acc"], 1.0 / (1.0 + EPS))


if __name__ == "__main__":
    unittest.main()
