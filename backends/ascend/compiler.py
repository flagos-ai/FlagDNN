"""Ascend/NPU compiler provider for persistent pointwise kernels."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any

from .compiler_identity import (
    MAXIMUM_AI_CORE_COUNT,
    SUPPORTED_CODEGEN_ARCHES,
    aicore_count_from_target,
    build_compiler_identity,
    validate_target_name,
)

from .add_plan import (
    PointwiseStagePlan,
    GRAPH_WORKSPACE_ALIGNMENT,
    GRAPH_SCHEMA_VERSION,
    MAX_RANK,
    SUPPORTED_LAYOUT_OPERATIONS,
    SUPPORTED_LAYERNORM_OPERATIONS,
    SUPPORTED_MATMUL_OPERATIONS,
    SUPPORTED_CONVOLUTION_OPERATIONS,
    SUPPORTED_OPERATIONS,
    SUPPORTED_REDUCTION_OPERATIONS,
    SUPPORTED_RMSNORM_OPERATIONS,
    TensorPlan,
    plan_graph,
    require_list,
    require_object,
)
from flagdnn_codegen.kernel_registry import (
    KernelCandidate,
    materialize_kernel_source,
    resolve_kernel_source,
    resolve_tuning_source,
    select_kernel_candidate,
)
from .tuning_decoder import (
    TuningConfiguration,
    canonical_json_bytes,
    canonical_sha256,
    checked_grid,
    load_tuning_table,
    validate_add_configuration,
)


SCHEMA_VERSION = GRAPH_SCHEMA_VERSION
ARTIFACT_SCHEMA_VERSION = 5
EXECUTION_PROGRAM_VERSION = 3
LAUNCH_ABI = "ltj_npu_raw_v1"
LAUNCH_PAYLOAD_VERSION = 1
PROVIDER_NAME = "ascend_triton"
PROVIDER_VERSION = "1"

_POINTER_SIGNATURES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "boolean": "*i8",
}

_MAX_JSON_RESOURCE_SIZE = 1 << 20
_MAX_KERNEL_SOURCE_SIZE = 1 << 20
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CONTIGUOUS_PARAMETERS = (
    "x_ptr",
    "y_ptr",
    "out_ptr",
    "n_elements",
    "OP_KIND",
    "ALPHA",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)
_STRIDED_PARAMETERS = (
    ("x_ptr", "y_ptr", "out_ptr", "n_elements")
    + tuple(f"DIM_{axis}" for axis in range(8))
    + tuple(f"LEFT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"RIGHT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_STRIDE_{axis}" for axis in range(8))
    + ("OP_KIND", "ALPHA", "BLOCK_SIZE", "WORKER_COUNT")
)
_UNARY_CONTIGUOUS_PARAMETERS = (
    "in_ptr",
    "out_ptr",
    "n_elements",
    "OPERATION",
    "negative_slope",
    "lower_clip",
    "upper_clip",
    "HAS_UPPER_CLIP",
    "SWISH_BETA",
    "ELU_ALPHA",
    "SOFTPLUS_BETA",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)
_UNARY_STRIDED_PARAMETERS = (
    ("in_ptr", "out_ptr", "n_elements")
    + tuple(f"DIM_{axis}" for axis in range(8))
    + tuple(f"INPUT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_STRIDE_{axis}" for axis in range(8))
    + (
        "OPERATION",
        "negative_slope",
        "lower_clip",
        "upper_clip",
        "HAS_UPPER_CLIP",
        "SWISH_BETA",
        "ELU_ALPHA",
        "SOFTPLUS_BETA",
        "BLOCK_SIZE",
        "WORKER_COUNT",
    )
)
_TERNARY_CONTIGUOUS_PARAMETERS = (
    "x_ptr",
    "y_ptr",
    "t_ptr",
    "out_ptr",
    "n_elements",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)
_TERNARY_STRIDED_PARAMETERS = (
    ("x_ptr", "y_ptr", "t_ptr", "out_ptr", "n_elements")
    + tuple(f"DIM_{axis}" for axis in range(8))
    + tuple(f"LEFT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"RIGHT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"MASK_STRIDE_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_STRIDE_{axis}" for axis in range(8))
    + ("BLOCK_SIZE", "WORKER_COUNT")
)
_LAYOUT_PARAMETERS = (
    (
        "input_ptr",
        "output_ptr",
        "n_elements",
        "INPUT_BASE",
        "ELEMENT_SIZE_BYTES",
        "LAYOUT_MODE",
    )
    + tuple(f"INPUT_DIM_{axis}" for axis in range(8))
    + tuple(f"INPUT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_DIM_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_STRIDE_{axis}" for axis in range(8))
    + ("BLOCK_SIZE", "WORKER_COUNT")
)
_REDUCTION_3D_PARAMETERS = (
    "input_ptr",
    "output_ptr",
    "n_elements",
    "RANK",
    "OUTPUT_RANK",
    "AXIS",
    "KEEP_DIMENSIONS",
    "OUTER",
    "REDUCTION_SIZE",
    "INNER",
    "OUTPUT_ELEMENTS",
    "REDUCTION_MODE",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)
_REDUCTION_STRIDED_PARAMETERS = (
    (
        "input_ptr", "output_ptr", "n_elements", "RANK", "OUTPUT_RANK",
        "AXIS", "KEEP_DIMENSIONS", "OUTER", "REDUCTION_SIZE", "INNER",
        "OUTPUT_ELEMENTS", "REDUCTION_MODE",
    )
    + tuple(f"INPUT_DIM_{axis}" for axis in range(8))
    + tuple(f"INPUT_STRIDE_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_DIM_{axis}" for axis in range(8))
    + tuple(f"OUTPUT_STRIDE_{axis}" for axis in range(8))
    + ("BLOCK_SIZE", "WORKER_COUNT")
)
_MATMUL_PARAMETERS = (
    ("a_ptr", "b_ptr", "output_ptr", "n_elements", "BATCH", "M", "N", "K")
    + tuple(f"DIM_{axis}" for axis in range(6))
    + tuple(f"A_BATCH_STRIDE_{axis}" for axis in range(6))
    + tuple(f"B_BATCH_STRIDE_{axis}" for axis in range(6))
    + tuple(f"C_BATCH_STRIDE_{axis}" for axis in range(6))
    + (
        "A_STRIDE_M",
        "A_STRIDE_K",
        "B_STRIDE_K",
        "B_STRIDE_N",
        "C_STRIDE_M",
        "C_STRIDE_N",
        "INPUT_IS_FLOAT32",
        "GROUP_M",
        "BLOCK_SIZE",
        "WORKER_COUNT",
    )
)
_CONVOLUTION_FPROP_PARAMETERS = (
    (
        "input_ptr",
        "filter_ptr",
        "output_ptr",
        "n_elements",
        "SPATIAL_RANK",
        "GROUPS",
        "INPUT_CHANNELS",
        "OUTPUT_CHANNELS",
        "CHANNELS_PER_GROUP",
    )
    + tuple(f"INPUT_DIM_{axis}" for axis in range(5))
    + tuple(f"INPUT_STRIDE_{axis}" for axis in range(5))
    + tuple(f"FILTER_DIM_{axis}" for axis in range(5))
    + tuple(f"FILTER_STRIDE_{axis}" for axis in range(5))
    + tuple(f"OUTPUT_DIM_{axis}" for axis in range(5))
    + tuple(f"OUTPUT_STRIDE_{axis}" for axis in range(5))
    + tuple(f"PRE_PADDING_{axis}" for axis in range(3))
    + tuple(f"POST_PADDING_{axis}" for axis in range(3))
    + tuple(f"CONV_STRIDE_{axis}" for axis in range(3))
    + tuple(f"DILATION_{axis}" for axis in range(3))
    + ("BLOCK_SIZE", "WORKER_COUNT")
)
_CONVOLUTION_IM2COL_PARAMETERS = (
    (
        "input_ptr",
        "filter_ptr",
        "columns_ptr",
        "n_elements",
    )
    + _CONVOLUTION_FPROP_PARAMETERS[4:]
)
_BATCHNORM_TRAINING_PARAMETERS = (
    (
        "x_ptr",
        "scale_ptr",
        "bias_ptr",
        "previous_running_mean_ptr",
        "previous_running_variance_ptr",
        "y_ptr",
        "mean_ptr",
        "inv_variance_ptr",
        "next_running_mean_ptr",
        "next_running_variance_ptr",
        "n_elements",
        "RANK",
        "BATCH",
        "CHANNELS",
        "SPATIAL",
        "REDUCTION_ELEMENTS",
        "EPSILON",
        "MOMENTUM",
    )
    + tuple(f"DIM_{axis}" for axis in range(8))
    + tuple(f"X_STRIDE_{axis}" for axis in range(8))
    + tuple(f"Y_STRIDE_{axis}" for axis in range(8))
    + ("BLOCK_SIZE", "WORKER_COUNT")
)
_BATCHNORM_NCHW_PARAMETERS = (
    "x_ptr",
    "mean_ptr",
    "inv_variance_ptr",
    "scale_ptr",
    "bias_ptr",
    "y_ptr",
    "n_elements",
    "RANK",
    "CHANNELS",
    "SPATIAL",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)
_BATCHNORM_STRIDED_PARAMETERS = (
    (
        "x_ptr",
        "mean_ptr",
        "inv_variance_ptr",
        "scale_ptr",
        "bias_ptr",
        "y_ptr",
        "n_elements",
        "RANK",
        "CHANNELS",
        "SPATIAL",
    )
    + tuple(f"DIM_{axis}" for axis in range(8))
    + tuple(f"X_STRIDE_{axis}" for axis in range(8))
    + tuple(f"Y_STRIDE_{axis}" for axis in range(8))
    + ("BLOCK_SIZE", "WORKER_COUNT")
)
_RMSNORM_PARAMETERS = (
    "x_ptr",
    "scale_ptr",
    "bias_ptr",
    "y_ptr",
    "inv_variance_ptr",
    "n_elements",
    "ROWS",
    "NORMALIZED_ELEMENTS",
    "EPSILON",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)
_LAYERNORM_PARAMETERS = (
    "x_ptr",
    "scale_ptr",
    "bias_ptr",
    "y_ptr",
    "mean_ptr",
    "inv_variance_ptr",
    "n_elements",
    "ROWS",
    "NORMALIZED_ELEMENTS",
    "EPSILON",
    "BLOCK_SIZE",
    "WORKER_COUNT",
)


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def compiler_identity(
    target_name: str,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    return build_compiler_identity(
        target_name,
        execution_engine,
        provider_path=Path(__file__),
        compiler_entry=_compiler_entry_path(),
        provider_name=PROVIDER_NAME,
        provider_version=PROVIDER_VERSION,
        graph_schema_version=SCHEMA_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        execution_program_version=EXECUTION_PROGRAM_VERSION,
        launch_abi=LAUNCH_ABI,
    )


def _load_json_resource(path: Path, description: str) -> dict[str, Any]:
    data = path.read_bytes()
    if not data or len(data) > _MAX_JSON_RESOURCE_SIZE:
        raise ValueError(f"{description} is empty or exceeds its size limit")
    return require_object(json.loads(data), description)


def _load_capabilities() -> dict[str, Any]:
    document = _load_json_resource(
        Path(__file__).with_name("capabilities.json"), "Ascend capabilities"
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
        or document.get("data_types")
        != ["float32", "float16", "bfloat16", "boolean"]
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
        or document.get("max_rank") != 8
        or document.get("workspace_alignment") != 256
        or document.get("compile_options")
        != {"num_warps": [4], "num_stages": [1]}
        or document.get("block_sizes") != [4096, 2048, 1024, 256, 128]
        or document.get("persistent_launch")
        != {
            "minimum_ai_core_count": 1,
            "maximum_ai_core_count": MAXIMUM_AI_CORE_COUNT,
            "workers_per_ai_core": 2,
        }
    ):
        raise ValueError(
            "Ascend capabilities do not match the kernel contract"
        )
    return document


def _validate_kernel_function(
    source_bytes: bytes, source_name: str, function_name: str
) -> None:
    try:
        source = source_bytes.decode("utf-8")
        module = ast.parse(source, filename=source_name)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            "Ascend pointwise kernel source is not valid UTF-8 Python"
        ) from error
    definitions = {
        node.name: tuple(argument.arg for argument in node.args.args)
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected = (
        _CONVOLUTION_IM2COL_PARAMETERS
        if function_name == "convolution_fprop_im2col_kernel"
        else _CONVOLUTION_FPROP_PARAMETERS
        if function_name == "convolution_fprop_persistent_kernel"
        else _MATMUL_PARAMETERS
        if function_name == "matmul_strided_kernel"
        else _BATCHNORM_TRAINING_PARAMETERS
        if function_name == "batchnorm_training_persistent_kernel"
        else _LAYERNORM_PARAMETERS
        if function_name == "layernorm_persistent_kernel"
        else _RMSNORM_PARAMETERS
        if function_name == "rmsnorm_persistent_kernel"
        else _BATCHNORM_NCHW_PARAMETERS
        if function_name == "batchnorm_inference_nchw_persistent_kernel"
        else _BATCHNORM_STRIDED_PARAMETERS
        if function_name == "batchnorm_inference_strided_persistent_kernel"
        else _CONTIGUOUS_PARAMETERS
        if function_name == "binary_contiguous_kernel"
        else (
            _STRIDED_PARAMETERS
            if function_name == "binary_strided_kernel"
            else (
                _UNARY_CONTIGUOUS_PARAMETERS
                if function_name == "unary_pointwise_contiguous_kernel"
                else (
                    _UNARY_STRIDED_PARAMETERS
                    if function_name == "unary_pointwise_strided_kernel"
                    else (
                        _TERNARY_CONTIGUOUS_PARAMETERS
                        if function_name
                        == "binary_select_contiguous_kernel"
                        else (
                            _TERNARY_STRIDED_PARAMETERS
                            if function_name
                            == "binary_select_strided_kernel"
                            else (
                                _LAYOUT_PARAMETERS
                                if function_name == "layout_copy_kernel"
                                else (
                                    _REDUCTION_3D_PARAMETERS
                                    if function_name
                                    == "reduction_3d_persistent_kernel"
                                    else (
                                        _REDUCTION_STRIDED_PARAMETERS
                                        if function_name
                                        == "reduction_strided_persistent_kernel"
                                        else None
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    if expected is None or definitions.get(function_name) != expected:
        raise ValueError(
            "Ascend kernel declaration does not match its raw ABI "
            "contract"
        )


def _ascend_full_signature(
    stage: PointwiseStagePlan, meta: dict[str, int | float]
) -> str:
    def pointer_signature(
        tensor: TensorPlan, token_override: str | None = None
    ) -> str:
        token = (
            token_override
            if token_override is not None
            else _POINTER_SIGNATURES[tensor.data_type]
        )
        effective_alignment = (
            GRAPH_WORKSPACE_ALIGNMENT if tensor.virtual else tensor.alignment
        )
        return f"{token}:16" if effective_alignment >= 16 else token

    layout_pointer_token: str | None = None
    if stage.kernel_family == "layout":
        element_size = int(meta["ELEMENT_SIZE_BYTES"])
        layout_pointer_token = "*i16" if element_size == 2 else "*i32"
    tokens = [
        pointer_signature(tensor, layout_pointer_token)
        for tensor in stage.tensors
    ]
    tokens.append("i32")
    if stage.function_name == "binary_strided_kernel":
        names = (
            [f"DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"LEFT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"RIGHT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "unary_pointwise_strided_kernel":
        names = (
            [f"DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "binary_select_strided_kernel":
        names = (
            [f"DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"LEFT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"RIGHT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"MASK_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "layout_copy_kernel":
        names = (
            ["INPUT_BASE", "ELEMENT_SIZE_BYTES", "LAYOUT_MODE"]
            + [f"INPUT_DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "matmul_strided_kernel":
        names = (
            ["BATCH", "M", "N", "K"]
            + [f"DIM_{axis}" for axis in range(6)]
            + [f"A_BATCH_STRIDE_{axis}" for axis in range(6)]
            + [f"B_BATCH_STRIDE_{axis}" for axis in range(6)]
            + [f"C_BATCH_STRIDE_{axis}" for axis in range(6)]
            + [
                "A_STRIDE_M",
                "A_STRIDE_K",
                "B_STRIDE_K",
                "B_STRIDE_N",
                "C_STRIDE_M",
                "C_STRIDE_N",
                "INPUT_IS_FLOAT32",
                "GROUP_M",
            ]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name in {
        "convolution_fprop_persistent_kernel",
        "convolution_fprop_im2col_kernel",
    }:
        names = (
            [
                "SPATIAL_RANK",
                "GROUPS",
                "INPUT_CHANNELS",
                "OUTPUT_CHANNELS",
                "CHANNELS_PER_GROUP",
            ]
            + [f"INPUT_DIM_{axis}" for axis in range(5)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(5)]
            + [f"FILTER_DIM_{axis}" for axis in range(5)]
            + [f"FILTER_STRIDE_{axis}" for axis in range(5)]
            + [f"OUTPUT_DIM_{axis}" for axis in range(5)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(5)]
            + [f"PRE_PADDING_{axis}" for axis in range(3)]
            + [f"POST_PADDING_{axis}" for axis in range(3)]
            + [f"CONV_STRIDE_{axis}" for axis in range(3)]
            + [f"DILATION_{axis}" for axis in range(3)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "reduction_3d_persistent_kernel":
        names = (
            "RANK",
            "OUTPUT_RANK",
            "AXIS",
            "KEEP_DIMENSIONS",
            "OUTER",
            "REDUCTION_SIZE",
            "INNER",
            "OUTPUT_ELEMENTS",
            "REDUCTION_MODE",
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "reduction_strided_persistent_kernel":
        names = (
            [
                "RANK", "OUTPUT_RANK", "AXIS", "KEEP_DIMENSIONS", "OUTER",
                "REDUCTION_SIZE", "INNER", "OUTPUT_ELEMENTS", "REDUCTION_MODE",
            ]
            + [f"INPUT_DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "batchnorm_inference_nchw_persistent_kernel":
        tokens.extend(
            str(meta[name]) for name in ("RANK", "CHANNELS", "SPATIAL")
        )
    elif stage.function_name == "batchnorm_inference_strided_persistent_kernel":
        names = (
            ["RANK", "CHANNELS", "SPATIAL"]
            + [f"DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"X_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"Y_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "batchnorm_training_persistent_kernel":
        names = (
            [
                "RANK",
                "BATCH",
                "CHANNELS",
                "SPATIAL",
                "REDUCTION_ELEMENTS",
                "EPSILON",
                "MOMENTUM",
            ]
            + [f"DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"X_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"Y_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name in {
        "rmsnorm_persistent_kernel",
        "layernorm_persistent_kernel",
    }:
        tokens.extend(
            str(meta[name])
            for name in ("ROWS", "NORMALIZED_ELEMENTS", "EPSILON")
        )
    if stage.kernel_family == "binary":
        tokens.extend(
            [
                str(meta["OP_KIND"]),
                repr(float(meta["ALPHA"])),
            ]
        )
    elif stage.kernel_family == "unary":
        tokens.extend(
            [
                str(meta["OPERATION"]),
                repr(float(meta["negative_slope"])),
                repr(float(meta["lower_clip"])),
                repr(float(meta["upper_clip"])),
                str(meta["HAS_UPPER_CLIP"]),
                repr(float(meta["SWISH_BETA"])),
                repr(float(meta["ELU_ALPHA"])),
                repr(float(meta["SOFTPLUS_BETA"])),
            ]
        )
    tokens.extend([str(meta["BLOCK_SIZE"]), str(meta["WORKER_COUNT"])])
    return ",".join(tokens)


def _runtime_argument_abi(
    stage: PointwiseStagePlan,
) -> list[dict[str, int | str]]:
    if stage.kernel_family == "binary":
        pointer_names = ("x_ptr", "y_ptr", "out_ptr")
    elif stage.kernel_family == "unary":
        pointer_names = ("in_ptr", "out_ptr")
    elif stage.kernel_family == "ternary":
        pointer_names = ("x_ptr", "y_ptr", "t_ptr", "out_ptr")
    elif stage.kernel_family == "layout":
        pointer_names = ("input_ptr", "output_ptr")
    elif stage.kernel_family == "reduction":
        pointer_names = ("input_ptr", "output_ptr")
    elif stage.kernel_family == "matmul":
        pointer_names = ("a_ptr", "b_ptr", "output_ptr")
    elif stage.kernel_family == "convolution_fprop":
        pointer_names = (
            (
                "input_ptr",
                "filter_ptr",
                "columns_ptr",
            )
            if stage.function_name == "convolution_fprop_im2col_kernel"
            else ("input_ptr", "filter_ptr", "output_ptr")
        )
    elif stage.kernel_family == "batchnorm_inference":
        pointer_names = (
            "x_ptr",
            "mean_ptr",
            "inv_variance_ptr",
            "scale_ptr",
            "bias_ptr",
            "y_ptr",
        )
    elif stage.kernel_family == "batchnorm":
        pointer_names = (
            "x_ptr",
            "scale_ptr",
            "bias_ptr",
            "previous_running_mean_ptr",
            "previous_running_variance_ptr",
            "y_ptr",
            "mean_ptr",
            "inv_variance_ptr",
            "next_running_mean_ptr",
            "next_running_variance_ptr",
        )
    elif stage.kernel_family == "rmsnorm":
        pointer_names = (
            "x_ptr",
            "scale_ptr",
            "bias_ptr",
            "y_ptr",
            "inv_variance_ptr",
        )
    elif stage.kernel_family == "layernorm":
        pointer_names = (
            "x_ptr",
            "scale_ptr",
            "bias_ptr",
            "y_ptr",
            "mean_ptr",
            "inv_variance_ptr",
        )
    else:
        raise ValueError("Ascend stage kernel family is unsupported")
    result: list[dict[str, int | str]] = [
        {"index": index, "name": name, "type": "pointer"}
        for index, name in enumerate(pointer_names)
    ]
    result.append(
        {
            "index": len(pointer_names),
            "name": "n_elements",
            "type": "i32",
        }
    )
    return result


def _write_immutable(path: Path, data: bytes, description: str) -> None:
    if path.exists():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != data
        ):
            raise ValueError(
                f"existing {description} violates content identity"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != data
        ):
            raise ValueError(f"concurrent {description} publication disagrees")


def _materialize_source(
    *,
    output_directory: Path,
    source_bytes: bytes,
    compiler_identity_sha256: str,
) -> tuple[str, str]:
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if _SHA256.fullmatch(compiler_identity_sha256) is None:
        raise ValueError("Ascend compiler identity is not canonical SHA-256")
    filename = f"source-{compiler_identity_sha256}-{source_sha256}.py"
    _write_immutable(
        output_directory / filename, source_bytes, "materialized kernel source"
    )
    return filename, source_sha256


def _batchnorm_inference_grid(
    stage: PointwiseStagePlan,
    block_size: int,
    capabilities: dict[str, Any],
) -> tuple[int, int, int]:
    if (
        stage.function_name == "batchnorm_inference_strided_persistent_kernel"
        and int(stage.meta["RANK"]) == 5
    ):
        channels = int(stage.meta["CHANNELS"])
        spatial = int(stage.meta["SPATIAL"])
        batch_elements = channels * spatial
        if batch_elements <= 0 or stage.n_elements % batch_elements != 0:
            raise ValueError("BatchNorm inference dimensions are inconsistent")
        batches = stage.n_elements // batch_elements
        block_spatial = block_size // 128
        if block_spatial <= 0:
            raise ValueError("BatchNorm inference block size is invalid")
        work_items = (
            batches
            * ((channels + 1) // 2)
            * ((spatial + block_spatial - 1) // block_spatial)
        )
        return checked_grid(work_items, 1, capabilities)
    return checked_grid(stage.n_elements, block_size, capabilities)


def _candidate_payload(
    *,
    stage: PointwiseStagePlan,
    configuration: TuningConfiguration,
    capabilities: dict[str, Any],
    source_path: str,
    source_sha256: str,
    worker_count: int,
) -> dict[str, Any]:
    validate_add_configuration(
        configuration, capabilities, kernel_family=stage.kernel_family
    )
    meta = dict(stage.meta)
    meta.update(configuration.meta)
    meta["WORKER_COUNT"] = worker_count
    if stage.kernel_family in {
        "reduction",
        "batchnorm",
        "rmsnorm",
        "layernorm",
    }:
        work_items = (
            int(meta["ROWS"])
            if stage.kernel_family in {"rmsnorm", "layernorm"}
            else int(meta["CHANNELS"])
            if stage.kernel_family == "batchnorm"
            else stage.n_elements
        )
        grid = checked_grid(work_items, 1, capabilities)
    elif stage.kernel_family == "batchnorm_inference":
        grid = _batchnorm_inference_grid(
            stage, int(meta["BLOCK_SIZE"]), capabilities
        )
    elif stage.kernel_family == "matmul":
        block_size = int(meta["BLOCK_SIZE"])
        tiles = (
            (int(meta["M"]) + block_size - 1) // block_size
        ) * ((int(meta["N"]) + block_size - 1) // block_size)
        batch = int(meta["BATCH"])
        grid = checked_grid(tiles * batch, 1, capabilities)
    elif stage.kernel_family == "layout" and int(meta["LAYOUT_MODE"]) == 2:
        inner_elements = int(meta["OUTPUT_DIM_7"])
        if inner_elements <= 0 or stage.n_elements % inner_elements != 0:
            raise ValueError("shared-row layout dimensions are inconsistent")
        grid = checked_grid(
            stage.n_elements // inner_elements, 1, capabilities
        )
    else:
        grid = checked_grid(
            stage.n_elements, int(meta["BLOCK_SIZE"]), capabilities
        )
    grid = (min(grid[0], worker_count), grid[1], grid[2])
    return {
        "schema_version": LAUNCH_PAYLOAD_VERSION,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "entry_point": stage.function_name,
        "full_signature": _ascend_full_signature(stage, meta),
        "grid": list(grid),
        "compile_options": {
            "num_warps": configuration.num_warps,
            "num_stages": configuration.num_stages,
        },
        "meta": meta,
        "argument_abi": _runtime_argument_abi(stage),
    }


def _autotune_candidate_id(
    operation: str, configuration: TuningConfiguration
) -> str:
    return "config_" + canonical_sha256(
        {
            "operation": operation,
            "configuration": configuration.as_dict(),
        },
        "Ascend binary tuning configuration",
    )


def _autotune_candidate_descriptor(
    candidate_id: str, configuration: TuningConfiguration
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "meta": dict(configuration.meta),
        "num_warps": configuration.num_warps,
        "num_stages": configuration.num_stages,
    }


def _f64_identity(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _stage_semantic_attributes(
    plan: PointwiseStagePlan,
) -> dict[str, Any]:
    if plan.kernel_family == "binary":
        return {"alpha": _f64_identity(plan.alpha)}
    if plan.kernel_family == "unary":
        return {
            "elu_alpha": _f64_identity(plan.elu_alpha),
            "has_upper_clip": plan.has_upper_clip,
            "lower_clip": _f64_identity(plan.lower_clip),
            "negative_slope": _f64_identity(plan.negative_slope),
            "softplus_beta": _f64_identity(plan.softplus_beta),
            "swish_beta": _f64_identity(plan.swish_beta),
            "upper_clip": _f64_identity(plan.upper_clip),
        }
    if plan.kernel_family == "reduction":
        return {
            name.lower(): int(plan.meta[name])
            for name in (
                "AXIS",
                "KEEP_DIMENSIONS",
                "OUTER",
                "REDUCTION_SIZE",
                "INNER",
                "OUTPUT_ELEMENTS",
                "REDUCTION_MODE",
            )
        }
    if plan.kernel_family == "matmul":
        roles = ("a", "b", "output")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "batch": int(plan.meta["BATCH"]),
            "m": int(plan.meta["M"]),
            "n": int(plan.meta["N"]),
            "k": int(plan.meta["K"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "convolution_fprop":
        roles = ("input", "filter", "output")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "meta": {
                name.lower(): int(value)
                for name, value in plan.meta.items()
                if name != "BLOCK_SIZE"
            },
            "n_elements": plan.n_elements,
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "batchnorm_inference":
        roles = ("x", "mean", "inv_variance", "scale", "bias", "y")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "channels": int(plan.meta["CHANNELS"]),
            "n_elements": plan.n_elements,
            "rank": int(plan.meta["RANK"]),
            "spatial": int(plan.meta["SPATIAL"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "batchnorm":
        roles = (
            "x",
            "scale",
            "bias",
            "previous_running_mean",
            "previous_running_variance",
            "y",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        )
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "batch": int(plan.meta["BATCH"]),
            "channels": int(plan.meta["CHANNELS"]),
            "epsilon": _f64_identity(float(plan.meta["EPSILON"])),
            "momentum": _f64_identity(float(plan.meta["MOMENTUM"])),
            "n_elements": plan.n_elements,
            "rank": int(plan.meta["RANK"]),
            "reduction_elements": int(plan.meta["REDUCTION_ELEMENTS"]),
            "spatial": int(plan.meta["SPATIAL"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "rmsnorm":
        roles = ("x", "scale", "bias", "y", "inv_variance")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "epsilon": _f64_identity(float(plan.meta["EPSILON"])),
            "normalized_elements": int(plan.meta["NORMALIZED_ELEMENTS"]),
            "rows": int(plan.meta["ROWS"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "layernorm":
        roles = (
            "x",
            "scale",
            "bias",
            "y",
            "mean",
            "inv_variance",
        )
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "epsilon": _f64_identity(float(plan.meta["EPSILON"])),
            "normalized_elements": int(plan.meta["NORMALIZED_ELEMENTS"]),
            "rows": int(plan.meta["ROWS"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    return {}


def _order_stage_configurations(
    plan: PointwiseStagePlan,
    configurations: tuple[TuningConfiguration, ...],
) -> tuple[TuningConfiguration, ...]:
    if (
        plan.kernel_family == "unary"
        and plan.operation in {"log", "rsqrt"}
        and plan.function_name == "unary_pointwise_contiguous_kernel"
        and plan.n_elements < 1024
    ):
        preferred = next(
            (
                configuration
                for configuration in configurations
                if int(configuration.meta["BLOCK_SIZE"]) == 256
            ),
            configurations[0],
        )
        return (preferred,) + tuple(
            configuration
            for configuration in configurations
            if configuration is not preferred
        )
    if plan.kernel_family not in {"rmsnorm", "layernorm"}:
        return configurations
    normalized_elements = int(plan.meta["NORMALIZED_ELEMENTS"])
    covering = [
        configuration
        for configuration in configurations
        if int(configuration.meta["BLOCK_SIZE"]) >= normalized_elements
    ]
    preferred = (
        min(covering, key=lambda value: int(value.meta["BLOCK_SIZE"]))
        if covering
        else max(
            configurations, key=lambda value: int(value.meta["BLOCK_SIZE"])
        )
    )
    return (preferred,) + tuple(
        configuration
        for configuration in configurations
        if configuration is not preferred
    )


def _build_stage(
    *,
    plan: PointwiseStagePlan,
    candidate: KernelCandidate,
    configurations: tuple[TuningConfiguration, ...],
    tuning_source_sha256: str,
    enable_autotune: bool,
    capabilities: dict[str, Any],
    source_path: str,
    source_sha256: str,
    source_size: int,
    worker_count: int,
) -> dict[str, Any]:
    if not configurations:
        raise ValueError(
            f"Ascend {plan.operation} tuning produced no candidates"
        )
    if not enable_autotune:
        configurations = _order_stage_configurations(plan, configurations)
    effective_configurations = (
        configurations if enable_autotune else configurations[:1]
    )
    tuning_enabled = enable_autotune and len(effective_configurations) > 1
    candidate_entries: list[dict[str, Any]] = []
    tuning_candidate_descriptors: list[dict[str, Any]] = []
    for configuration in effective_configurations:
        candidate_id = (
            _autotune_candidate_id(plan.operation, configuration)
            if tuning_enabled
            else "default"
        )
        candidate_entries.append(
            {
                "candidate_id": candidate_id,
                "launch_abi": LAUNCH_ABI,
                "payload": _candidate_payload(
                    stage=plan,
                    configuration=configuration,
                    capabilities=capabilities,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    worker_count=worker_count,
                ),
            }
        )
        if tuning_enabled:
            descriptor = _autotune_candidate_descriptor(
                candidate_id, configuration
            )
            if plan.kernel_family in {
                "matmul",
                "convolution_fprop",
                "batchnorm",
                "batchnorm_inference",
                "rmsnorm",
                "layernorm",
            }:
                work_items = (
                    int(plan.meta["ROWS"])
                    if plan.kernel_family in {"rmsnorm", "layernorm"}
                    else int(plan.meta["CHANNELS"])
                    if plan.kernel_family == "batchnorm"
                    else plan.n_elements
                )
                if plan.kernel_family == "matmul":
                    block_size = int(configuration.meta["BLOCK_SIZE"])
                    tiles = (
                        (int(plan.meta["M"]) + block_size - 1) // block_size
                    ) * (
                        (int(plan.meta["N"]) + block_size - 1) // block_size
                    )
                    descriptor["grid"] = [
                        min(tiles * int(plan.meta["BATCH"]), worker_count),
                        1,
                        1,
                    ]
                elif plan.kernel_family == "convolution_fprop":
                    grid = checked_grid(
                        work_items,
                        int(configuration.meta["BLOCK_SIZE"]),
                        capabilities,
                    )
                    descriptor["grid"] = [
                        min(grid[0], worker_count),
                        grid[1],
                        grid[2],
                    ]
                elif plan.kernel_family == "batchnorm_inference":
                    grid = _batchnorm_inference_grid(
                        plan,
                        int(configuration.meta["BLOCK_SIZE"]),
                        capabilities,
                    )
                    descriptor["grid"] = [
                        min(grid[0], worker_count),
                        grid[1],
                        grid[2],
                    ]
                else:
                    grid = checked_grid(work_items, 1, capabilities)
                    descriptor["grid"] = [
                        min(grid[0], worker_count),
                        grid[1],
                        grid[2],
                    ]
            tuning_candidate_descriptors.append(descriptor)

    if len({entry["candidate_id"] for entry in candidate_entries}) != len(
        candidate_entries
    ):
        raise ValueError(
            f"Ascend {plan.operation} tuning candidate identity is duplicated"
        )

    selection = {
        "state": "pending" if tuning_enabled else "fixed",
        "candidate_id": "" if tuning_enabled else "default",
    }
    autotune: dict[str, Any] = {
        "schema_version": 1,
        "enabled": tuning_enabled,
        "selection": selection,
    }
    if tuning_enabled:
        tuning = candidate.tuning
        if tuning is None:
            raise RuntimeError(
                f"Ascend {plan.operation} candidate lost its tuning metadata"
            )
        identity_payload = {
            "schema_version": 1,
            "backend": "ascend",
            "launch_abi": LAUNCH_ABI,
            "operation": plan.operation,
            "attributes": _stage_semantic_attributes(plan),
            "function": plan.function_name,
            "source_sha256": tuning_source_sha256,
            "key": tuning.key,
            "key_value": plan.n_elements,
            "strategy": tuning.strategy,
            "candidates": (
                sorted(
                    tuning_candidate_descriptors,
                    key=lambda descriptor: str(descriptor["candidate_id"]),
                )
                if plan.kernel_family
                in {
                    "matmul",
                    "convolution_fprop",
                    "batchnorm",
                    "batchnorm_inference",
                    "rmsnorm",
                    "layernorm",
                }
                else tuning_candidate_descriptors
            ),
        }
        if plan.kernel_family in {
            "matmul",
            "convolution_fprop",
            "batchnorm",
            "batchnorm_inference",
            "rmsnorm",
            "layernorm",
        }:
            identity_payload["kernel_source_sha256"] = source_sha256
        autotune.update(
            {
                "source": tuning.source,
                "source_sha256": tuning_source_sha256,
                "table": tuning.table,
                "key": tuning.key,
                "strategy": tuning.strategy,
                "warmup": tuning.warmup,
                "repetitions": tuning.repetitions,
                "candidate_identity": canonical_sha256(
                    identity_payload,
                    f"Ascend {plan.operation} tuning identity",
                ),
            }
        )

    return {
        "stage_id": plan.stage_id,
        "kind": "kernel",
        "source_node_ids": list(plan.source_node_ids),
        "dependencies": list(plan.dependencies),
        "operation": plan.operation,
        "kernel_family": plan.kernel_family,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "source_sha256": source_sha256,
            "entry_point": plan.function_name,
            "materialized_source": {
                "file": source_path,
                "size": source_size,
                "sha256": source_sha256,
            },
        },
        "workspace": dict(plan.workspace),
        "argument_sources": [dict(value) for value in plan.argument_sources],
        "candidates": candidate_entries,
        "autotune": autotune,
    }


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


def _validate_platform_candidate(
    operation: str, candidate: KernelCandidate
) -> None:
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
        else "matmul.py"
        if matmul
        else "normalization.py"
        if batchnorm_inference or batchnorm_training or rmsnorm or layernorm
        else "reduction.py"
        if reduction
        else "layout.py"
        if layout
        else "unary.py"
        if unary
        else "ternary.py"
        if ternary
        else "binary.py"
    )
    expected_functions = (
        {
            "convolution_fprop_persistent_kernel",
            "convolution_fprop_im2col_kernel",
        }
        if convolution_fprop
        else {"matmul_strided_kernel"}
        if matmul
        else {"batchnorm_training_persistent_kernel"}
        if batchnorm_training
        else {"layernorm_persistent_kernel"}
        if layernorm
        else {"rmsnorm_persistent_kernel"}
        if rmsnorm
        else {
            "batchnorm_inference_nchw_persistent_kernel",
            "batchnorm_inference_strided_persistent_kernel",
        }
        if batchnorm_inference
        else {
            "reduction_3d_persistent_kernel",
            "reduction_strided_persistent_kernel",
        }
        if reduction
        else {"layout_copy_kernel"}
        if layout
        else {
            "unary_pointwise_contiguous_kernel",
            "unary_pointwise_strided_kernel",
        }
        if unary
        else {
            "binary_select_contiguous_kernel",
            "binary_select_strided_kernel",
        }
        if ternary
        else {"binary_contiguous_kernel", "binary_strided_kernel"}
    )
    expected_table = (
        "convolution_fprop"
        if convolution_fprop
        else "matmul"
        if matmul
        else "batchnorm"
        if batchnorm_training
        else "layernorm"
        if layernorm
        else "rmsnorm"
        if rmsnorm
        else "batchnorm_inference"
        if batchnorm_inference
        else "reduction"
        if reduction
        else "unary"
        if unary
        else "binary"
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


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("Ascend supports only the libtriton_jit engine")
    request_bytes = request_path.read_bytes()
    request = require_object(json.loads(request_bytes), "request")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Ascend compiler requires Graph IR schema v3")
    flagdnn_version = request.get("flagdnn_version")
    if (
        not isinstance(flagdnn_version, str)
        or _SEMANTIC_VERSION.fullmatch(flagdnn_version) is None
    ):
        raise ValueError("request FlagDNN version is invalid")
    if request.get("backend") != "ascend":
        raise ValueError("Ascend provider received another backend")
    target_name = validate_target_name(request.get("target"))
    ai_core_count = aicore_count_from_target(target_name)
    identity = compiler_identity(target_name, execution_engine)
    if request.get("compiler_identity") != identity["identity_sha256"]:
        raise ValueError("request compiler identity does not match provider")

    enable_autotune = _validate_build_options(request.get("build_options"))
    graph_plan = plan_graph(request.get("graph"))
    capabilities = _load_capabilities()
    persistent_launch = require_object(
        capabilities["persistent_launch"], "persistent_launch"
    )
    if not (
        int(persistent_launch["minimum_ai_core_count"])
        <= ai_core_count
        <= int(persistent_launch["maximum_ai_core_count"])
    ):
        raise ValueError("Ascend target AI core count exceeds capability")
    worker_count = ai_core_count * int(
        persistent_launch["workers_per_ai_core"]
    )
    candidate_by_operation: dict[str, KernelCandidate] = {}
    for operation in sorted({stage.operation for stage in graph_plan.stages}):
        candidate = select_kernel_candidate("ascend", operation)
        _validate_platform_candidate(operation, candidate)
        candidate_by_operation[operation] = candidate

    compiler_path = _compiler_entry_path()
    output_directory.mkdir(parents=True, exist_ok=True)
    stage_assets: dict[
        str,
        tuple[tuple[TuningConfiguration, ...], str, str, str, int],
    ] = {}
    for operation, candidate in candidate_by_operation.items():
        kernel_source_path = resolve_kernel_source(compiler_path, candidate)
        source_bytes = materialize_kernel_source(kernel_source_path, candidate)
        if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_SIZE:
            raise ValueError("Ascend pointwise kernel source size is invalid")
        for stage in graph_plan.stages:
            if stage.operation == operation:
                _validate_kernel_function(
                    source_bytes, candidate.source, stage.function_name
                )

        tuning = candidate.tuning
        if tuning is None:
            raise RuntimeError(
                "validated Ascend pointwise candidate lost tuning"
            )
        tuning_path = resolve_tuning_source(compiler_path, candidate)
        configurations, tuning_source_sha256 = load_tuning_table(
            tuning_path, tuning.table
        )
        for configuration in configurations:
            validate_add_configuration(
                configuration,
                capabilities,
                kernel_family=(
                    next(
                        stage.kernel_family
                        for stage in graph_plan.stages
                        if stage.operation == operation
                    )
                ),
            )
        source_path, source_sha256 = _materialize_source(
            output_directory=output_directory,
            source_bytes=source_bytes,
            compiler_identity_sha256=identity["identity_sha256"],
        )
        stage_assets[operation] = (
            configurations,
            tuning_source_sha256,
            source_path,
            source_sha256,
            len(source_bytes),
        )

    stages = []
    for stage in graph_plan.stages:
        (
            configurations,
            tuning_source_sha256,
            source_path,
            source_sha256,
            source_size,
        ) = stage_assets[stage.operation]
        stages.append(
            _build_stage(
                plan=stage,
                candidate=candidate_by_operation[stage.operation],
                configurations=configurations,
                tuning_source_sha256=tuning_source_sha256,
                enable_autotune=(
                    enable_autotune
                    and stage.function_name
                    != "convolution_fprop_im2col_kernel"
                ),
                capabilities=capabilities,
                source_path=source_path,
                source_sha256=source_sha256,
                source_size=source_size,
                worker_count=worker_count,
            )
        )

    combined_source_sha256 = hashlib.sha256(
        canonical_json_bytes(
            [stage["kernel"]["source_sha256"] for stage in stages],
            "Ascend stage source hashes",
        )
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "backend": "ascend",
        "target": target_name,
        "graph_node_count": graph_plan.node_count,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": combined_source_sha256,
        "compiler": identity,
        "workspace_size": graph_plan.workspace_size,
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_immutable(
        output_directory / "manifest.json", manifest_bytes, "Ascend manifest"
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "ascend",
        "provider": PROVIDER_NAME,
        "node_count": graph_plan.node_count,
        "stage_count": len(stages),
        "target": target_name,
        "artifact_directory": str(output_directory),
        "workspace_size": graph_plan.workspace_size,
        "execution_engine": execution_engine,
        "launch_abi": LAUNCH_ABI,
    }
