import os
from dataclasses import dataclass
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _bool_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise RuntimeError(
        "{} must be one of: {}.".format(
            name, ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
        )
    )


def _int_env(name, default, minimum=None):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError("{} must be an integer.".format(name)) from exc
    if minimum is not None and value < minimum:
        raise RuntimeError("{} must be >= {}.".format(name, minimum))
    return value


def _float_env(name, default, minimum=None):
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError("{} must be a number.".format(name)) from exc
    if minimum is not None and value < minimum:
        raise RuntimeError("{} must be >= {}.".format(name, minimum))
    return value


def _required_path(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError("Missing required environment variable: {}".format(name))
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("{} does not point to a file: {}".format(name, path))
    return path


def _parse_image_shape(value):
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise RuntimeError(
            "OCR_REC_IMAGE_SHAPE must contain three comma-separated integers."
        ) from exc
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise RuntimeError(
            "OCR_REC_IMAGE_SHAPE must be a positive C,H,W shape."
        )
    if shape[0] != 3:
        raise RuntimeError("The recognizer expects a three-channel input.")
    return shape


@dataclass(frozen=True)
class DeploymentSettings:
    det_model: Path
    rec_model: Path
    table_model: Path
    rec_char_dict: Path
    table_char_dict: Path
    providers: str
    det_providers: str
    rec_providers: str
    table_providers: str
    intra_op_threads: int
    inter_op_threads: int
    graph_optimization: str
    execution_mode: str
    enable_mem_pattern: bool
    enable_cpu_mem_arena: bool
    ort_log_severity: int
    cuda_device_id: int
    cuda_mem_limit_mb: int
    cuda_arena_extend_strategy: str
    cuda_cudnn_conv_algo_search: str
    cuda_use_tf32: bool
    det_limit_side_len: float
    det_limit_type: str
    rec_image_shape: str
    rec_batch_num: int
    table_max_len: int
    output_dir: Path
    output_ttl_seconds: int
    max_upload_mb: int
    max_concurrency: int
    acquire_timeout_seconds: float
    warmup_runs: int

    @classmethod
    def from_env(cls):
        rec_image_shape = os.getenv("OCR_REC_IMAGE_SHAPE", "3,48,320")
        _parse_image_shape(rec_image_shape)

        det_limit_type = os.getenv("OCR_DET_LIMIT_TYPE", "min").strip().lower()
        if det_limit_type not in {"min", "max"}:
            raise RuntimeError("OCR_DET_LIMIT_TYPE must be 'min' or 'max'.")

        graph_optimization = os.getenv(
            "OCR_ONNX_GRAPH_OPTIMIZATION", "all"
        ).strip().lower()
        if graph_optimization not in {"disable", "basic", "extended", "all"}:
            raise RuntimeError(
                "OCR_ONNX_GRAPH_OPTIMIZATION must be disable, basic, extended, or all."
            )

        execution_mode = os.getenv(
            "OCR_ONNX_EXECUTION_MODE", "sequential"
        ).strip().lower()
        if execution_mode not in {"sequential", "parallel"}:
            raise RuntimeError(
                "OCR_ONNX_EXECUTION_MODE must be sequential or parallel."
            )

        cuda_search = os.getenv(
            "OCR_CUDA_CUDNN_CONV_ALGO_SEARCH", "EXHAUSTIVE"
        ).strip().upper()
        if cuda_search not in {"EXHAUSTIVE", "HEURISTIC", "DEFAULT"}:
            raise RuntimeError(
                "OCR_CUDA_CUDNN_CONV_ALGO_SEARCH must be EXHAUSTIVE, "
                "HEURISTIC, or DEFAULT."
            )

        cuda_arena = os.getenv(
            "OCR_CUDA_ARENA_EXTEND_STRATEGY", "kNextPowerOfTwo"
        ).strip()
        if cuda_arena not in {"kNextPowerOfTwo", "kSameAsRequested"}:
            raise RuntimeError(
                "OCR_CUDA_ARENA_EXTEND_STRATEGY must be kNextPowerOfTwo "
                "or kSameAsRequested."
            )

        providers = os.getenv("OCR_ONNX_PROVIDERS", "auto").strip()
        if not providers:
            raise RuntimeError("OCR_ONNX_PROVIDERS cannot be empty.")
        det_providers = os.getenv(
            "OCR_DET_ONNX_PROVIDERS", providers
        ).strip()
        rec_providers = os.getenv(
            "OCR_REC_ONNX_PROVIDERS", providers
        ).strip()
        table_providers = os.getenv(
            "OCR_TABLE_ONNX_PROVIDERS", providers
        ).strip()
        if not all((det_providers, rec_providers, table_providers)):
            raise RuntimeError(
                "Per-model ONNX provider settings cannot be empty."
            )

        output_dir = Path(
            os.getenv("OCR_OUTPUT_DIR", "./output/deploy")
        ).expanduser().resolve()

        settings = cls(
            det_model=_required_path("OCR_DET_MODEL"),
            rec_model=_required_path("OCR_REC_MODEL"),
            table_model=_required_path("OCR_TABLE_MODEL"),
            rec_char_dict=_required_path("OCR_REC_DICT"),
            table_char_dict=_required_path("OCR_TABLE_DICT"),
            providers=providers,
            det_providers=det_providers,
            rec_providers=rec_providers,
            table_providers=table_providers,
            intra_op_threads=_int_env("OCR_ONNX_THREADS", 0, 0),
            inter_op_threads=_int_env("OCR_ONNX_INTER_OP_THREADS", 0, 0),
            graph_optimization=graph_optimization,
            execution_mode=execution_mode,
            enable_mem_pattern=_bool_env("OCR_ONNX_ENABLE_MEM_PATTERN", True),
            enable_cpu_mem_arena=_bool_env(
                "OCR_ONNX_ENABLE_CPU_MEM_ARENA", True
            ),
            ort_log_severity=_int_env("OCR_ONNX_LOG_SEVERITY", 3, 0),
            cuda_device_id=_int_env("OCR_CUDA_DEVICE_ID", 0, 0),
            cuda_mem_limit_mb=_int_env("OCR_CUDA_MEM_LIMIT_MB", 0, 0),
            cuda_arena_extend_strategy=cuda_arena,
            cuda_cudnn_conv_algo_search=cuda_search,
            cuda_use_tf32=_bool_env("OCR_CUDA_USE_TF32", True),
            det_limit_side_len=_float_env(
                "OCR_DET_LIMIT_SIDE_LEN", 736, 1
            ),
            det_limit_type=det_limit_type,
            rec_image_shape=rec_image_shape,
            rec_batch_num=_int_env("OCR_REC_BATCH_NUM", 32, 1),
            table_max_len=_int_env("OCR_TABLE_MAX_LEN", 488, 32),
            output_dir=output_dir,
            output_ttl_seconds=_int_env("OCR_OUTPUT_TTL_SECONDS", 3600, 0),
            max_upload_mb=_int_env("OCR_MAX_UPLOAD_MB", 20, 1),
            max_concurrency=_int_env("OCR_MAX_CONCURRENCY", 1, 1),
            acquire_timeout_seconds=_float_env(
                "OCR_ACQUIRE_TIMEOUT_SECONDS", 30, 0
            ),
            warmup_runs=_int_env("OCR_WARMUP_RUNS", 1, 0),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return settings

    @property
    def rec_shape(self):
        return _parse_image_shape(self.rec_image_shape)

    @property
    def requested_providers(self):
        return self._parse_providers(self.providers)

    @staticmethod
    def _parse_providers(value):
        if value.lower() == "auto":
            return ()
        return tuple(
            provider.strip()
            for provider in value.split(",")
            if provider.strip()
        )

    @property
    def model_providers(self):
        return {
            "detector": self.det_providers,
            "recognizer": self.rec_providers,
            "SLANet": self.table_providers,
        }

    @property
    def cuda_required_models(self):
        return tuple(
            name
            for name, providers in self.model_providers.items()
            if "CUDAExecutionProvider" in self._parse_providers(providers)
        )

    @property
    def requires_cuda(self):
        return bool(self.cuda_required_models)

    @property
    def use_gpu(self):
        return any(
            providers.lower() == "auto"
            for providers in self.model_providers.values()
        ) or self.requires_cuda
