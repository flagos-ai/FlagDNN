"""Ascend dispatch selection implementation."""

from __future__ import annotations

import json
from ..compiler_identity import (
    COMPATIBILITY_ALIAS,
    MAXIMUM_AI_CORE_COUNT,
    RUNTIME_DETECTED_CODEGEN_ARCHES,
    SUPPORTED_CODEGEN_ARCHES,
)
from .common import (
    LAUNCH_ABI,
    PROVIDER_NAME,
    SUPPORTED_CONVOLUTION_OPERATIONS,
    SUPPORTED_LAYERNORM_OPERATIONS,
    SUPPORTED_LAYOUT_OPERATIONS,
    SUPPORTED_MATMUL_OPERATIONS,
    SUPPORTED_OPERATIONS,
    SUPPORTED_REDUCTION_OPERATIONS,
    SUPPORTED_RMSNORM_OPERATIONS,
    _MAX_JSON_RESOURCE_SIZE,
    require_list,
    require_object,
)
from flagdnn_codegen.kernel_registry import (
    KernelCandidate,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)


def _load_json_resource(path: Path, description: str) -> dict[str, Any]:
    data = path.read_bytes()
    if not data or len(data) > _MAX_JSON_RESOURCE_SIZE:
        raise ValueError(f"{description} is empty or exceeds its size limit")
    return require_object(json.loads(data), description)


def _load_capabilities() -> dict[str, Any]:
    document = _load_json_resource(
        Path(__file__).resolve().parents[1] / "capabilities.json", "Ascend capabilities"
    )
    expected_keys = {
        "schema_version",
        "backend",
        "launch_abi",
        "supported_operations",
        "data_types",
        "matmul",
        "convolution_fprop",
        "batchnorm_inference",
        "batchnorm",
        "rmsnorm",
        "layernorm",
        "codegen_arches",
        "target_policy",
        "max_rank",
        "workspace_alignment",
        "compile_options",
        "block_sizes",
        "persistent_launch",
        "grid",
    }
    if set(document) != expected_keys:
        raise ValueError("Ascend capability fields are invalid")
    if (
        document.get("schema_version") != 1
        or document.get("backend") != "ascend"
        or document.get("launch_abi") != LAUNCH_ABI
        or document.get("supported_operations") != list(SUPPORTED_OPERATIONS)
        or document.get("data_types") != ["float32", "float16", "bfloat16", "boolean"]
        or document.get("matmul")
        != {
            "data_types": ["float32", "float16", "bfloat16"],
            "compute_data_type": "float32",
            "minimum_rank": 2,
            "maximum_rank": 8,
            "maximum_batch_rank": 6,
            "block_sizes": [16, 32, 64, 128],
        }
        or document.get("convolution_fprop")
        != {
            "data_types": ["float32", "float16", "bfloat16"],
            "compute_data_type": "float32",
            "minimum_spatial_rank": 1,
            "maximum_spatial_rank": 3,
            "block_sizes": [256, 128],
        }
        or document.get("batchnorm_inference")
        != {
            "x_y_data_types": ["float32", "float16", "bfloat16"],
            "parameter_data_type": "float32",
            "parameter_shape": "contiguous_numel_channels",
            "compute_data_type": "float32",
            "minimum_rank": 2,
            "maximum_rank": 8,
            "channel_axis": 1,
        }
        or document.get("batchnorm")
        != {
            "x_y_scale_bias_data_types": [
                "float32",
                "float16",
                "bfloat16",
            ],
            "statistic_data_type": "float32",
            "parameter_shape": "contiguous_numel_channels",
            "compute_data_type": "float32",
            "minimum_rank": 2,
            "maximum_rank": 8,
            "channel_axis": 1,
            "input_count": 5,
            "output_count": 5,
        }
        or document.get("rmsnorm")
        != {
            "data_types": ["float32", "float16", "bfloat16"],
            "compute_data_type": "float32",
            "minimum_rank": 1,
            "maximum_rank": 8,
            "layout": "contiguous_suffix",
            "statistic_data_type": "float32",
        }
        or document.get("layernorm")
        != {
            "data_types": ["float32", "float16", "bfloat16"],
            "compute_data_type": "float32",
            "minimum_rank": 1,
            "maximum_rank": 8,
            "layout": "contiguous_suffix",
            "statistic_data_type": "float32",
            "statistic_count": 2,
        }
        or document.get("codegen_arches") != list(SUPPORTED_CODEGEN_ARCHES)
        or document.get("target_policy")
        != {
            "runtime_detected_codegen_arches": list(RUNTIME_DETECTED_CODEGEN_ARCHES),
            "compatibility_alias": COMPATIBILITY_ALIAS,
        }
        or document.get("max_rank") != 8
        or document.get("workspace_alignment") != 256
        or document.get("compile_options") != {"num_warps": [4], "num_stages": [1]}
        or document.get("block_sizes") != [4096, 2048, 1024, 256, 128]
        or document.get("persistent_launch")
        != {
            "minimum_ai_core_count": 1,
            "maximum_ai_core_count": MAXIMUM_AI_CORE_COUNT,
            "workers_per_ai_core": 2,
        }
    ):
        raise ValueError("Ascend capabilities do not match the kernel contract")
    return document


def _validate_build_options(value: object) -> bool:
    options = require_object(value, "build_options")
    modes = require_list(options.get("heuristic_modes"), "heuristic_modes")
    if (
        not modes
        or any(mode not in {"A", "FALLBACK"} for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("build_options.heuristic_modes is invalid")
    enable_autotune = options.get("autotune", False)
    if not isinstance(enable_autotune, bool):
        raise ValueError("build_options.autotune must be a boolean")
    return enable_autotune


def _validate_platform_candidate(operation: str, candidate: KernelCandidate) -> None:
    unary = operation in {
        "relu",
        "sqrt",
        "erf",
        "identity",
        "exp",
        "log",
        "neg",
        "abs",
        "ceil",
        "cos",
        "floor",
        "rsqrt",
        "sin",
        "tan",
        "reciprocal",
        "logical_not",
        "sigmoid",
        "tanh",
        "elu",
        "gelu",
        "softplus",
        "swish",
        "gelu_approx_tanh",
    }
    ternary = operation == "binary_select"
    layout = operation in SUPPORTED_LAYOUT_OPERATIONS
    reduction = operation in SUPPORTED_REDUCTION_OPERATIONS
    matmul = operation in SUPPORTED_MATMUL_OPERATIONS
    convolution_fprop = operation in SUPPORTED_CONVOLUTION_OPERATIONS
    batchnorm_inference = operation == "batchnorm_inference"
    batchnorm_training = operation == "batchnorm"
    rmsnorm = operation in SUPPORTED_RMSNORM_OPERATIONS
    layernorm = operation in SUPPORTED_LAYERNORM_OPERATIONS
    expected_source = (
        "convolution.py"
        if convolution_fprop
        else (
            "matmul.py"
            if matmul
            else (
                "normalization.py"
                if batchnorm_inference or batchnorm_training or rmsnorm or layernorm
                else (
                    "reduction.py"
                    if reduction
                    else (
                        "layout.py"
                        if layout
                        else (
                            "unary.py"
                            if unary
                            else "ternary.py" if ternary else "binary.py"
                        )
                    )
                )
            )
        )
    )
    expected_functions = (
        {
            "convolution_fprop_persistent_kernel",
            "convolution_fprop_im2col_kernel",
        }
        if convolution_fprop
        else (
            {"matmul_strided_kernel"}
            if matmul
            else (
                {"batchnorm_training_persistent_kernel"}
                if batchnorm_training
                else (
                    {"layernorm_persistent_kernel"}
                    if layernorm
                    else (
                        {"rmsnorm_persistent_kernel"}
                        if rmsnorm
                        else (
                            {
                                "batchnorm_inference_nchw_persistent_kernel",
                                "batchnorm_inference_strided_persistent_kernel",
                            }
                            if batchnorm_inference
                            else (
                                {
                                    "reduction_3d_persistent_kernel",
                                    "reduction_strided_persistent_kernel",
                                }
                                if reduction
                                else (
                                    {"layout_copy_kernel"}
                                    if layout
                                    else (
                                        {
                                            "unary_pointwise_contiguous_kernel",
                                            "unary_pointwise_strided_kernel",
                                        }
                                        if unary
                                        else (
                                            {
                                                "binary_select_contiguous_kernel",
                                                "binary_select_strided_kernel",
                                            }
                                            if ternary
                                            else {
                                                "binary_contiguous_kernel",
                                                "binary_strided_kernel",
                                            }
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    expected_table = (
        "convolution_fprop"
        if convolution_fprop
        else (
            "matmul"
            if matmul
            else (
                "batchnorm"
                if batchnorm_training
                else (
                    "layernorm"
                    if layernorm
                    else (
                        "rmsnorm"
                        if rmsnorm
                        else (
                            "batchnorm_inference"
                            if batchnorm_inference
                            else (
                                "reduction"
                                if reduction
                                else "unary" if unary else "binary"
                            )
                        )
                    )
                )
            )
        )
    )
    tuning = candidate.tuning
    if (
        candidate.backend != "ascend"
        or candidate.operation != operation
        or candidate.ownership != "platform"
        or candidate.source_layout != "platform"
        or candidate.provider != PROVIDER_NAME
        or candidate.source != expected_source
        or set(candidate.functions) != expected_functions
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != expected_table
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"Ascend {operation} platform kernel registry contract is invalid"
        )
