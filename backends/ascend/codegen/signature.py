"""Ascend codegen signature implementation."""

from __future__ import annotations

import ast
from ..dispatch.common import (
    GRAPH_WORKSPACE_ALIGNMENT,
    MAX_RANK,
    PointwiseStagePlan,
    TensorPlan,
)

_POINTER_SIGNATURES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "boolean": "*i8",
}

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
    "input_ptr",
    "filter_ptr",
    "columns_ptr",
    "n_elements",
) + _CONVOLUTION_FPROP_PARAMETERS[4:]

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
        else (
            _CONVOLUTION_FPROP_PARAMETERS
            if function_name == "convolution_fprop_persistent_kernel"
            else (
                _MATMUL_PARAMETERS
                if function_name == "matmul_strided_kernel"
                else (
                    _BATCHNORM_TRAINING_PARAMETERS
                    if function_name == "batchnorm_training_persistent_kernel"
                    else (
                        _LAYERNORM_PARAMETERS
                        if function_name == "layernorm_persistent_kernel"
                        else (
                            _RMSNORM_PARAMETERS
                            if function_name == "rmsnorm_persistent_kernel"
                            else (
                                _BATCHNORM_NCHW_PARAMETERS
                                if function_name
                                == "batchnorm_inference_nchw_persistent_kernel"
                                else (
                                    _BATCHNORM_STRIDED_PARAMETERS
                                    if function_name
                                    == "batchnorm_inference_strided_persistent_kernel"
                                    else (
                                        _CONTIGUOUS_PARAMETERS
                                        if function_name == "binary_contiguous_kernel"
                                        else (
                                            _STRIDED_PARAMETERS
                                            if function_name == "binary_strided_kernel"
                                            else (
                                                _UNARY_CONTIGUOUS_PARAMETERS
                                                if function_name
                                                == "unary_pointwise_contiguous_kernel"
                                                else (
                                                    _UNARY_STRIDED_PARAMETERS
                                                    if function_name
                                                    == "unary_pointwise_strided_kernel"
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
                                                                if function_name
                                                                == "layout_copy_kernel"
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
            "Ascend kernel declaration does not match its raw ABI " "contract"
        )


def _ascend_full_signature(
    stage: PointwiseStagePlan, meta: dict[str, int | float]
) -> str:
    def pointer_signature(tensor: TensorPlan, token_override: str | None = None) -> str:
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
        pointer_signature(tensor, layout_pointer_token) for tensor in stage.tensors
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
                "RANK",
                "OUTPUT_RANK",
                "AXIS",
                "KEEP_DIMENSIONS",
                "OUTER",
                "REDUCTION_SIZE",
                "INNER",
                "OUTPUT_ELEMENTS",
                "REDUCTION_MODE",
            ]
            + [f"INPUT_DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_DIM_{axis}" for axis in range(MAX_RANK)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(MAX_RANK)]
        )
        tokens.extend(str(meta[name]) for name in names)
    elif stage.function_name == "batchnorm_inference_nchw_persistent_kernel":
        tokens.extend(str(meta[name]) for name in ("RANK", "CHANNELS", "SPATIAL"))
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
            str(meta[name]) for name in ("ROWS", "NORMALIZED_ELEMENTS", "EPSILON")
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
