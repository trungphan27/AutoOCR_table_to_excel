import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from deploy.config import DeploymentSettings
from deploy.service import TableInferenceService, _json_safe
from infer.onnx_runtime import create_onnx_session


class DeploymentSettingsTest(unittest.TestCase):
    def make_env(self, root):
        paths = {}
        for name in ("det.onnx", "rec.onnx", "table.onnx", "rec.txt", "table.txt"):
            path = root / name
            path.write_bytes(b"test")
            paths[name] = path
        return {
            "OCR_DET_MODEL": str(paths["det.onnx"]),
            "OCR_REC_MODEL": str(paths["rec.onnx"]),
            "OCR_TABLE_MODEL": str(paths["table.onnx"]),
            "OCR_REC_DICT": str(paths["rec.txt"]),
            "OCR_TABLE_DICT": str(paths["table.txt"]),
            "OCR_OUTPUT_DIR": str(root / "output"),
        }

    def test_defaults_match_trained_model_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with mock.patch.dict(os.environ, self.make_env(root), clear=True):
                settings = DeploymentSettings.from_env()
        self.assertEqual(settings.rec_shape, (3, 48, 320))
        self.assertEqual(settings.rec_batch_num, 32)
        self.assertEqual(settings.table_max_len, 488)
        self.assertEqual(settings.det_limit_side_len, 736)
        self.assertEqual(settings.max_concurrency, 1)

    def test_custom_recognizer_shape_is_parsed_for_model_contract_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env = self.make_env(root)
            env["OCR_REC_IMAGE_SHAPE"] = "3,32,320"
            with mock.patch.dict(os.environ, env, clear=True):
                settings = DeploymentSettings.from_env()
        # Static model dimensions are validated when sessions are loaded.
        self.assertEqual(settings.rec_shape, (3, 32, 320))

    def test_bad_provider_and_boolean_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env = self.make_env(root)
            env["OCR_ONNX_ENABLE_MEM_PATTERN"] = "sometimes"
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    DeploymentSettings.from_env()

    def test_explicit_cuda_is_recorded_as_required(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env = self.make_env(root)
            env["OCR_ONNX_PROVIDERS"] = (
                "CUDAExecutionProvider,CPUExecutionProvider"
            )
            with mock.patch.dict(os.environ, env, clear=True):
                settings = DeploymentSettings.from_env()
        self.assertTrue(settings.requires_cuda)
        self.assertTrue(settings.use_gpu)

    def test_per_model_provider_override_supports_hybrid_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env = self.make_env(root)
            env["OCR_ONNX_PROVIDERS"] = (
                "CUDAExecutionProvider,CPUExecutionProvider"
            )
            env["OCR_TABLE_ONNX_PROVIDERS"] = "CPUExecutionProvider"
            with mock.patch.dict(os.environ, env, clear=True):
                settings = DeploymentSettings.from_env()
        self.assertEqual(
            settings.cuda_required_models, ("detector", "recognizer")
        )
        self.assertEqual(
            settings.model_providers["SLANet"], "CPUExecutionProvider"
        )


class FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.enable_mem_pattern = False
        self.enable_cpu_mem_arena = False
        self.log_severity_level = 0


class FakeSession:
    def __init__(self, _path, sess_options, providers):
        self.options = sess_options
        self.providers = [
            item[0] if isinstance(item, tuple) else item for item in providers
        ]
        self.provider_config = providers

    def get_providers(self):
        return self.providers


def fake_ort(available):
    module = ModuleType("onnxruntime")
    module.SessionOptions = FakeSessionOptions
    module.GraphOptimizationLevel = SimpleNamespace(
        ORT_DISABLE_ALL="disable",
        ORT_ENABLE_BASIC="basic",
        ORT_ENABLE_EXTENDED="extended",
        ORT_ENABLE_ALL="all",
    )
    module.ExecutionMode = SimpleNamespace(
        ORT_PARALLEL="parallel", ORT_SEQUENTIAL="sequential"
    )
    module.get_available_providers = lambda: list(available)
    module.InferenceSession = FakeSession
    return module


class OnnxRuntimeConfigurationTest(unittest.TestCase):
    def make_args(self, providers):
        return SimpleNamespace(
            onnx_providers=providers,
            use_gpu="CUDAExecutionProvider" in providers,
            onnx_intra_op_threads=4,
            onnx_inter_op_threads=1,
            onnx_graph_optimization="all",
            onnx_execution_mode="sequential",
            onnx_enable_mem_pattern=True,
            onnx_enable_cpu_mem_arena=True,
            onnx_log_severity=3,
            onnx_cuda_device_id=1,
            onnx_cuda_mem_limit_mb=1024,
            onnx_cuda_arena_extend_strategy="kNextPowerOfTwo",
            onnx_cuda_cudnn_conv_algo_search="EXHAUSTIVE",
            onnx_cuda_use_tf32=True,
        )

    def test_cuda_options_and_graph_optimization_are_applied(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory) / "model.onnx"
            model.write_bytes(b"model")
            module = fake_ort(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            with mock.patch.dict(sys.modules, {"onnxruntime": module}):
                session = create_onnx_session(
                    model,
                    self.make_args(
                        "CUDAExecutionProvider,CPUExecutionProvider"
                    ),
                )
        self.assertEqual(session.options.graph_optimization_level, "all")
        self.assertEqual(session.options.intra_op_num_threads, 4)
        cuda_options = session.provider_config[0][1]
        self.assertEqual(cuda_options["device_id"], "1")
        self.assertEqual(cuda_options["gpu_mem_limit"], str(1024 ** 3))
        self.assertEqual(cuda_options["use_tf32"], "1")

    def test_explicit_unavailable_cuda_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory) / "model.onnx"
            model.write_bytes(b"model")
            module = fake_ort(["CPUExecutionProvider"])
            with mock.patch.dict(sys.modules, {"onnxruntime": module}):
                with self.assertRaises(ValueError):
                    create_onnx_session(
                        model,
                        self.make_args(
                            "CUDAExecutionProvider,CPUExecutionProvider"
                        ),
                    )


class ServiceFacadeTest(unittest.TestCase):
    def test_numpy_payload_is_json_safe(self):
        payload = _json_safe(
            {"box": np.array([1, 2]), "score": np.float32(0.5)}
        )
        self.assertEqual(payload["box"], [1, 2])
        self.assertAlmostEqual(payload["score"], 0.5)

    def test_request_slot_is_released_after_inference(self):
        service = object.__new__(TableInferenceService)
        service.settings = SimpleNamespace(acquire_timeout_seconds=0.1)
        service._request_slots = threading.BoundedSemaphore(value=1)
        service._system = lambda _image, return_ocr_result_in_table: (
            {"html": "<table></table>", "cell_bbox": np.empty((0, 4))},
            {"all": np.float32(0.1)},
        )
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        first = service.predict_image(image)
        second = service.predict_image(image)
        self.assertEqual(first["result"]["html"], "<table></table>")
        self.assertEqual(second["result"]["html"], "<table></table>")


if __name__ == "__main__":
    unittest.main()
