import logging
import sys
import tempfile
import unittest
from pathlib import Path

import paddle

paddle.set_device("cpu")


PADDLE_OCR_ROOT = (
    Path(__file__).resolve().parents[1] / "training" / "PaddleOCR"
)
# Isolate the training package from the inference compatibility namespace.
for module_name in list(sys.modules):
    if module_name == "ppocr" or module_name.startswith("ppocr."):
        del sys.modules[module_name]
sys.path.insert(0, str(PADDLE_OCR_ROOT))

from ppocr.utils.save_load import load_model, save_model


class ResumeCheckpointTest(unittest.TestCase):
    def make_model_and_optimizer(self):
        model = paddle.nn.Linear(2, 1)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters()
        )
        return model, optimizer

    def test_mid_epoch_checkpoint_restores_exact_position(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model, optimizer = self.make_model_and_optimizer()
            config = {
                "Global": {"distributed": False},
                "Architecture": {"model_type": "det"},
            }
            save_model(
                model,
                optimizer,
                temporary_directory,
                logging.getLogger(__name__),
                config,
                prefix="latest_step",
                best_model_dict={"hmean": 0.5},
                epoch=3,
                global_step=17,
                step_in_epoch=5,
                epoch_completed=False,
            )

            restored_model, restored_optimizer = self.make_model_and_optimizer()
            config["Global"]["checkpoints"] = str(
                Path(temporary_directory) / "latest_step"
            )
            state = load_model(
                config, restored_model, restored_optimizer, model_type="det"
            )

            self.assertEqual(state["start_epoch"], 3)
            self.assertEqual(state["resume_step_in_epoch"], 5)
            self.assertEqual(state["global_step"], 17)
            self.assertEqual(state["hmean"], 0.5)

    def test_completed_epoch_resumes_at_next_epoch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model, optimizer = self.make_model_and_optimizer()
            config = {
                "Global": {"distributed": False},
                "Architecture": {"model_type": "det"},
            }
            save_model(
                model,
                optimizer,
                temporary_directory,
                logging.getLogger(__name__),
                config,
                prefix="latest",
                best_model_dict={},
                epoch=10,
                global_step=100,
                step_in_epoch=0,
                epoch_completed=True,
            )

            restored_model, restored_optimizer = self.make_model_and_optimizer()
            config["Global"]["checkpoints"] = str(
                Path(temporary_directory) / "latest"
            )
            state = load_model(
                config, restored_model, restored_optimizer, model_type="det"
            )

            self.assertEqual(state["start_epoch"], 11)
            self.assertNotIn("resume_step_in_epoch", state)
            self.assertEqual(state["global_step"], 100)

    def test_custom_best_prefix_also_updates_best_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model, optimizer = self.make_model_and_optimizer()
            config = {
                "Global": {"distributed": False},
                "Architecture": {"model_type": "table"},
            }

            save_model(
                model,
                optimizer,
                temporary_directory,
                logging.getLogger(__name__),
                config,
                is_best=True,
                prefix="best_structure_score",
                best_model_dict={"structure_score": 0.9},
                epoch=11,
                global_step=110,
                step_in_epoch=1,
                epoch_completed=False,
            )

            root = Path(temporary_directory)
            self.assertTrue(
                (root / "best_structure_score.pdparams").is_file()
            )
            self.assertTrue(
                (root / "best_structure_score.states").is_file()
            )
            self.assertTrue((root / "best_model" / "model.pdparams").is_file())


if __name__ == "__main__":
    unittest.main()
