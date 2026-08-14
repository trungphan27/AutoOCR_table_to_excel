"""Centralized ONNX Runtime session configuration for inference."""

from pathlib import Path

import numpy as np


def _getattr(args, name, default):
    return getattr(args, name, default)


def _session_options(ort, args):
    options = ort.SessionOptions()

    intra_threads = int(_getattr(args, "onnx_intra_op_threads", 0))
    inter_threads = int(_getattr(args, "onnx_inter_op_threads", 0))
    if intra_threads > 0:
        options.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        options.inter_op_num_threads = inter_threads

    optimization = str(
        _getattr(args, "onnx_graph_optimization", "all")
    ).lower()
    optimization_levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if optimization not in optimization_levels:
        raise ValueError("Unsupported ONNX graph optimization: {}".format(optimization))
    options.graph_optimization_level = optimization_levels[optimization]

    execution_mode = str(
        _getattr(args, "onnx_execution_mode", "sequential")
    ).lower()
    if execution_mode == "parallel":
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    elif execution_mode == "sequential":
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        raise ValueError("Unsupported ONNX execution mode: {}".format(execution_mode))

    options.enable_mem_pattern = bool(
        _getattr(args, "onnx_enable_mem_pattern", True)
    )
    options.enable_cpu_mem_arena = bool(
        _getattr(args, "onnx_enable_cpu_mem_arena", True)
    )
    options.log_severity_level = int(
        _getattr(args, "onnx_log_severity", 3)
    )
    return options


def _cuda_options(args):
    options = {
        "device_id": str(int(_getattr(args, "onnx_cuda_device_id", 0))),
        "arena_extend_strategy": str(
            _getattr(
                args,
                "onnx_cuda_arena_extend_strategy",
                "kNextPowerOfTwo",
            )
        ),
        "cudnn_conv_algo_search": str(
            _getattr(args, "onnx_cuda_cudnn_conv_algo_search", "EXHAUSTIVE")
        ),
        "do_copy_in_default_stream": "1",
        "use_tf32": (
            "1" if bool(_getattr(args, "onnx_cuda_use_tf32", True)) else "0"
        ),
    }
    memory_limit_mb = int(
        _getattr(args, "onnx_cuda_mem_limit_mb", 0)
    )
    if memory_limit_mb > 0:
        options["gpu_mem_limit"] = str(memory_limit_mb * 1024 * 1024)
    return options


def create_onnx_session(model_path, args, providers_override=None):
    import onnxruntime as ort

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("ONNX model does not exist: {}".format(path))

    requested = str(
        providers_override
        if providers_override is not None
        else _getattr(args, "onnx_providers", "auto")
    ).strip()
    available = ort.get_available_providers()
    if requested.lower() == "auto":
        provider_names = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if bool(_getattr(args, "use_gpu", False))
            and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        explicit_cuda = False
    else:
        provider_names = [
            value.strip() for value in requested.split(",") if value.strip()
        ]
        if not provider_names:
            raise ValueError("At least one ONNX Runtime provider is required.")
        unavailable = [
            provider for provider in provider_names if provider not in available
        ]
        if unavailable:
            raise ValueError(
                "ONNX Runtime providers are unavailable: {}. Available: {}".format(
                    unavailable, available
                )
            )
        explicit_cuda = "CUDAExecutionProvider" in provider_names

    providers = []
    for provider in provider_names:
        if provider == "CUDAExecutionProvider":
            providers.append((provider, _cuda_options(args)))
        else:
            providers.append(provider)

    session = ort.InferenceSession(
        str(path),
        sess_options=_session_options(ort, args),
        providers=providers,
    )
    active = session.get_providers()
    if explicit_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but the session fell back to: {}"
            .format(active)
        )
    return session


def prepare_onnx_input(value, input_metadata):
    """Cast a NumPy input to the dtype declared by an ONNX model."""
    dtype_by_type = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    expected = dtype_by_type.get(input_metadata.type)
    if expected is None:
        raise TypeError(
            "Unsupported ONNX input type: {}".format(input_metadata.type)
        )
    if value.dtype == expected:
        return value
    return value.astype(expected, copy=False)


def prepare_onnx_outputs(values):
    """Return FP16 model outputs as FP32 for stable NumPy/OpenCV decoding."""
    return [
        value.astype(np.float32, copy=False)
        if isinstance(value, np.ndarray) and value.dtype == np.float16
        else value
        for value in values
    ]
