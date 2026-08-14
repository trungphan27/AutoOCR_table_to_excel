import json
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from deploy.config import DeploymentSettings
from predict_table import TableSystem, to_excel
from utility import init_args


class ServiceBusyError(RuntimeError):
    """Raised when the bounded inference queue cannot accept another request."""


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class TableInferenceService:
    """Bounded, thread-safe facade shared by FastAPI and Gradio."""

    def __init__(self, settings):
        self.settings = settings
        self._ready = False
        self._started_at = time.time()
        self._request_slots = threading.BoundedSemaphore(
            value=settings.max_concurrency
        )

        args = init_args().parse_args([])
        args.use_onnx = True
        args.use_gpu = settings.use_gpu
        args.onnx_providers = settings.providers
        args.onnx_det_providers = settings.det_providers
        args.onnx_rec_providers = settings.rec_providers
        args.onnx_table_providers = settings.table_providers
        args.onnx_intra_op_threads = settings.intra_op_threads
        args.onnx_inter_op_threads = settings.inter_op_threads
        args.onnx_graph_optimization = settings.graph_optimization
        args.onnx_execution_mode = settings.execution_mode
        args.onnx_enable_mem_pattern = settings.enable_mem_pattern
        args.onnx_enable_cpu_mem_arena = settings.enable_cpu_mem_arena
        args.onnx_log_severity = settings.ort_log_severity
        args.onnx_cuda_device_id = settings.cuda_device_id
        args.onnx_cuda_mem_limit_mb = settings.cuda_mem_limit_mb
        args.onnx_cuda_arena_extend_strategy = (
            settings.cuda_arena_extend_strategy
        )
        args.onnx_cuda_cudnn_conv_algo_search = (
            settings.cuda_cudnn_conv_algo_search
        )
        args.onnx_cuda_use_tf32 = settings.cuda_use_tf32

        args.det_algorithm = "DB"
        args.rec_algorithm = "SVTR_LCNet"
        args.table_algorithm = "SLANet"
        args.det_model_dir = str(settings.det_model)
        args.rec_model_dir = str(settings.rec_model)
        args.table_model_dir = str(settings.table_model)
        args.rec_char_dict_path = str(settings.rec_char_dict)
        args.table_char_dict_path = str(settings.table_char_dict)
        args.det_limit_side_len = settings.det_limit_side_len
        args.det_limit_type = settings.det_limit_type
        args.rec_image_shape = settings.rec_image_shape
        args.rec_batch_num = settings.rec_batch_num
        args.table_max_len = settings.table_max_len
        args.show_log = False
        args.benchmark = False

        self._system = TableSystem(args)
        self._validate_model_contract()
        self._warmup(settings.warmup_runs)
        self._ready = True

    @classmethod
    def from_env(cls):
        return cls(DeploymentSettings.from_env())

    def _sessions(self):
        return {
            "detector": self._system.text_detector.predictor,
            "recognizer": self._system.text_recognizer.predictor,
            "SLANet": self._system.table_structurer.predictor,
        }

    def _validate_model_contract(self):
        sessions = self._sessions()
        minimum_outputs = {"detector": 1, "recognizer": 1, "SLANet": 2}
        for name, session in sessions.items():
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1:
                raise RuntimeError(
                    "{} ONNX model must have exactly one input; found {}."
                    .format(name, len(inputs))
                )
            model_input = inputs[0]
            if len(model_input.shape) != 4:
                raise RuntimeError(
                    "{} input must be rank 4 NCHW; found {}.".format(
                        name, model_input.shape
                    )
                )
            if model_input.type not in {"tensor(float)", "tensor(float16)"}:
                raise RuntimeError(
                    "{} input must be float32/float16; found {}.".format(
                        name, model_input.type
                    )
                )
            if len(outputs) < minimum_outputs[name]:
                raise RuntimeError(
                    "{} ONNX model must have at least {} output(s); found {}."
                    .format(name, minimum_outputs[name], len(outputs))
                )

        rec_input_shape = sessions["recognizer"].get_inputs()[0].shape
        expected_channels, expected_height, _ = self.settings.rec_shape
        static_channels = rec_input_shape[1]
        static_height = rec_input_shape[2]
        if isinstance(static_channels, int) and static_channels > 0:
            if static_channels != expected_channels:
                raise RuntimeError(
                    "Recognizer channel mismatch: model={}, config={}.".format(
                        static_channels, expected_channels
                    )
                )
        if isinstance(static_height, int) and static_height > 0:
            if static_height != expected_height:
                raise RuntimeError(
                    "Recognizer height mismatch: model={}, config={}.".format(
                        static_height, expected_height
                    )
                )

        if self.settings.requires_cuda:
            cpu_fallback = [
                name
                for name, session in sessions.items()
                if name in self.settings.cuda_required_models
                and "CUDAExecutionProvider" not in session.get_providers()
            ]
            if cpu_fallback:
                raise RuntimeError(
                    "CUDA was required but these sessions are not using the CUDA "
                    "provider: {}.".format(", ".join(cpu_fallback))
                )

    def _warmup(self, runs):
        if runs <= 0:
            return
        table_image = np.full((256, 512, 3), 255, dtype=np.uint8)
        cell_image = np.full((48, 160, 3), 255, dtype=np.uint8)
        cv2.putText(
            cell_image,
            "warmup",
            (4, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        for _ in range(runs):
            self._system.table_structurer(table_image.copy())
            self._system.text_detector(table_image.copy())
            self._system.text_recognizer([cell_image.copy()])

    @property
    def is_ready(self):
        return self._ready

    def provider_info(self):
        sessions = self._sessions()
        return {
            "requested": self.settings.model_providers,
            "active": {
                name: session.get_providers()
                for name, session in sessions.items()
            },
        }

    def health_info(self):
        return {
            "status": "ready" if self._ready else "starting",
            "uptime_seconds": round(time.time() - self._started_at, 3),
            "providers": self.provider_info(),
            "max_concurrency": self.settings.max_concurrency,
            "rec_batch_num": self.settings.rec_batch_num,
        }

    def decode_image(self, content):
        array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a valid image.")
        return image

    def predict_bytes(self, content):
        return self.predict_image(self.decode_image(content))

    def predict_image(self, image):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with three channels.")
        acquired = self._request_slots.acquire(
            timeout=self.settings.acquire_timeout_seconds
        )
        if not acquired:
            raise ServiceBusyError(
                "Inference capacity is full; retry the request later."
            )
        try:
            result, timing = self._system(
                image, return_ocr_result_in_table=True
            )
        finally:
            self._request_slots.release()
        result = _json_safe(result)
        timing = _json_safe(timing)
        return {"result": result, "timing": timing}

    def _cleanup_expired_outputs(self):
        ttl = self.settings.output_ttl_seconds
        if ttl <= 0:
            return
        cutoff = time.time() - ttl
        for path in self.settings.output_dir.glob("table-*.xlsx"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                # Cleanup races are harmless because output names are unique.
                continue

    def save_excel(self, payload):
        html = payload.get("result", {}).get("html")
        if not isinstance(html, str) or not html.strip():
            raise ValueError("Inference did not produce table HTML.")
        self._cleanup_expired_outputs()
        filename = "table-{}.xlsx".format(uuid.uuid4().hex)
        path = self.settings.output_dir / filename
        to_excel(html, str(path))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Excel conversion did not create a valid file.")
        return path

    def annotate_cells(self, image, cell_boxes):
        annotated = image.copy()
        for raw_box in cell_boxes:
            box = np.asarray(raw_box, dtype=np.int32).reshape(-1)
            if box.size == 4:
                x1, y1, x2, y2 = box.tolist()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif box.size == 8:
                points = box.reshape(4, 2)
                cv2.polylines(annotated, [points], True, (255, 0, 0), 2)
        return annotated

    def gradio_predict(self, rgb_image):
        if rgb_image is None:
            raise ValueError("Please upload an image.")
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        payload = self.predict_image(bgr_image)
        excel_path = self.save_excel(payload)
        annotated = self.annotate_cells(
            bgr_image, payload["result"].get("cell_bbox", [])
        )
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return (
            annotated,
            payload["result"]["html"],
            json.loads(json.dumps(payload, ensure_ascii=False)),
            str(excel_path),
        )
