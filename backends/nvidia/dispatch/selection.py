"""Operation-to-kernel routing; every configuration returns a KernelPlan."""

from __future__ import annotations

from typing import Any
import math

from flagdnn_codegen.kernel_registry import BINARY_POINTWISE_OPERATIONS
from flagdnn_codegen.kernel_registry import TERNARY_POINTWISE_OPERATIONS
from flagdnn_codegen.kernel_registry import UNARY_POINTWISE_OPERATIONS

from .attention_gradients import partial_gradient_reduction
from .attention_backward import (
    _sdpa_backward_kernel_configuration,
    _sdpa_fp8_backward_kernel_configuration,
)
from .attention_forward import (
    _sdpa_forward_kernel_configuration,
    _sdpa_fp8_forward_kernel_configuration,
)
from .common import (
    NUMERIC_DATA_TYPES,
    KernelPlan,
    REDUCTION_OPERATIONS,
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _next_power_of_two,
    _reduction_tensor_constants,
    _require_integer,
)
from .conv_backward import (
    _convolution_backward_kernel_configuration,
    _convolution_dgrad_3d_pipeline_kernel_configuration,
    _convolution_dgrad_stride2_pipeline_kernel_configuration,
)
from .conv_fprop import (
    _convolution_general_im2col_kernel_configuration,
    _convolution_im2col_kernel_configuration,
    _convolution_kernel_configuration,
)
from .conv_wgrad import (
    _convolution_wgrad_1x1_pipeline_kernel_configuration,
    _convolution_wgrad_batched_pipeline_kernel_configuration,
    _convolution_wgrad_p5_pipeline_kernel_configuration,
    _convolution_wgrad_pipeline_kernel_configuration,
    _convolution_wgrad_stride2_pipeline_kernel_configuration,
)
from .normalization_extended import _extended_normalization_configuration
from .position_embedding import _rope_kernel_configuration
from .random import _rng_kernel_configuration
from .resample import _resample_kernel_configuration
from .moe_matmul import _moe_matmul_configuration
from .fp8_matmul import _fp8_matmul_configuration
from .precision import _explicit_precision_configuration
from .causal_convolution import _causal_conv1d_kernel_configuration
from .statistics import (
    _genstats_kernel_configuration,
    _bn_finalize_kernel_configuration,
)
from .pointwise import _pointwise_kernel_configuration
from .index import _index_kernel_configuration
from .layout import _layout_kernel_configuration
from .matmul import (
    _matmul_kernel_configuration,
    _matmul_p5_pipeline_kernel_configuration,
)
from .normalization import (
    _batchnorm_inference_kernel_configuration,
    _batchnorm_kernel_configuration,
    _normalization_forward_kernel_configuration,
)


def _kernel_configuration_impl(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if parameters.get("_sdpa_reduce_partials") is True:
        return partial_gradient_reduction(operation, parameters, tensors)
    tensor_data_types = [tensor["data_type"] for tensor in tensors]
    if operation in {"moe_grouped_matmul", "moe_grouped_matmul_bwd"}:
        return _moe_matmul_configuration(
            operation, parameters, tensors, architecture
        )
    if operation == "matmul_fp8":
        return _fp8_matmul_configuration(parameters, tensors, architecture)
    if parameters.get("input_precision", 0) != 0 and operation in {
        "matmul",
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    }:
        return _explicit_precision_configuration(
            operation, parameters, tensors, architecture
        )
    if operation in {
        "instancenorm",
        "instancenorm_backward",
        "adalayernorm",
        "batchnorm_backward",
        "rmsnorm_backward",
        "layernorm_backward",
        "adalayernorm_backward",
    }:
        return _extended_normalization_configuration(
            operation, parameters, tensors
        )
    if operation == "bn_finalize":
        return _bn_finalize_kernel_configuration(parameters, tensors)
    if operation in {"rope", "rope_backward"}:
        return _rope_kernel_configuration(operation, parameters, tensors)
    if operation == "rng":
        return _rng_kernel_configuration(parameters, tensors)
    if operation == "resample":
        return _resample_kernel_configuration(parameters, tensors)
    if operation == "causal_conv1d":
        return _causal_conv1d_kernel_configuration(parameters, tensors)
    if operation == "genstats":
        return _genstats_kernel_configuration(parameters, tensors)
    if operation in {"gen_index", "concatenate"}:
        return _index_kernel_configuration(operation, parameters, tensors)
    if operation == "sdpa":
        return _sdpa_forward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_backward":
        return _sdpa_backward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_fp8":
        return _sdpa_fp8_forward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_fp8_backward":
        return _sdpa_fp8_backward_kernel_configuration(parameters, tensors)
    if operation == "layernorm":
        return _normalization_forward_kernel_configuration(
            parameters, tensors, rmsnorm=False
        )
    if operation == "rmsnorm":
        return _normalization_forward_kernel_configuration(
            parameters, tensors, rmsnorm=True
        )
    if operation == "batchnorm":
        return _batchnorm_kernel_configuration(parameters, tensors)
    if operation == "batchnorm_inference":
        return _batchnorm_inference_kernel_configuration(parameters, tensors)
    if operation in {"reshape", "transpose", "slice"}:
        return _layout_kernel_configuration(operation, parameters, tensors)
    if operation == "matmul" and "_fprop_p5_matmul_stage" in parameters:
        return _matmul_p5_pipeline_kernel_configuration(parameters, tensors)
    if operation == "matmul":
        return _matmul_kernel_configuration(parameters, tensors, architecture)
    if (
        operation == "convolution_fprop"
        and parameters.get("_fprop_pipeline_stage") == "im2col"
    ):
        if parameters.get("_fprop_pipeline_algorithm") == "general":
            return _convolution_general_im2col_kernel_configuration(
                parameters, tensors
            )
        return _convolution_im2col_kernel_configuration(parameters, tensors)
    if operation == "convolution_fprop":
        return _convolution_kernel_configuration(parameters, tensors)
    if (
        operation == "convolution_wgrad"
        and "_wgrad_pipeline_stage" in parameters
    ):
        if parameters.get("_wgrad_pipeline_algorithm") == "p5":
            return _convolution_wgrad_p5_pipeline_kernel_configuration(
                parameters, tensors
            )
        if parameters.get("_wgrad_pipeline_algorithm") == "1x1":
            return _convolution_wgrad_1x1_pipeline_kernel_configuration(
                parameters, tensors
            )
        if parameters.get("_wgrad_pipeline_algorithm") in {
            "1x1_split",
            "stride2_im2col",
        }:
            if parameters.get("_wgrad_pipeline_stage") == "im2col":
                return _convolution_im2col_kernel_configuration(
                    parameters, tensors
                )
            return _convolution_wgrad_batched_pipeline_kernel_configuration(
                parameters, tensors
            )
        if parameters.get("_wgrad_pipeline_algorithm") == "stride2_row4":
            return _convolution_wgrad_stride2_pipeline_kernel_configuration(
                parameters, tensors
            )
        return _convolution_wgrad_pipeline_kernel_configuration(
            parameters, tensors
        )
    if (
        operation == "convolution_dgrad"
        and "_dgrad_3d_pipeline_stage" in parameters
    ):
        return _convolution_dgrad_3d_pipeline_kernel_configuration(
            parameters, tensors
        )
    if (
        operation == "convolution_dgrad"
        and "_dgrad_pipeline_stage" in parameters
    ):
        return _convolution_dgrad_stride2_pipeline_kernel_configuration(
            parameters, tensors
        )
    if operation in {"convolution_dgrad", "convolution_wgrad"}:
        return _convolution_backward_kernel_configuration(
            operation, parameters, tensors
        )
    if (
        operation in {"relu", "add"}
        or operation in UNARY_POINTWISE_OPERATIONS
        or operation in BINARY_POINTWISE_OPERATIONS
        or operation in TERNARY_POINTWISE_OPERATIONS
    ):
        return _pointwise_kernel_configuration(operation, parameters, tensors)
    if operation in REDUCTION_OPERATIONS:
        if (
            len(tensor_data_types) != 2
            or tensor_data_types[0] not in NUMERIC_DATA_TYPES
            or (
                tensor_data_types[1] != "float32"
                and (
                    tensor_data_types[0] == "int32"
                    or tensor_data_types[0] != tensor_data_types[1]
                )
            )
        ):
            raise ValueError(
                "Reduction requires numeric input and FP32 "
                "or matching floating output"
            )
        pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        if pointer_type is None:
            raise ValueError(
                "unsupported Reduction data type: " f"{tensor_data_types[0]!r}"
            )
        outer = _require_integer(parameters, "outer")
        extent = _require_integer(parameters, "reduction", maximum=65536)
        inner = _require_integer(parameters, "inner")
        output_elements = _require_integer(parameters, "output_elements")
        input_rank = len(tensors[0]["dimensions"])
        if input_rank == 0:
            raise ValueError("Reduction input must have positive rank")
        axis = _require_integer(
            parameters,
            "axis",
            minimum=0,
            maximum=input_rank - 1,
        )
        keep_dimensions_value = _require_integer(
            parameters, "keep_dimensions", minimum=0, maximum=1
        )
        keep_dimensions = keep_dimensions_value == 1
        if math.prod(tensors[0]["dimensions"]) != outer * extent * inner:
            raise ValueError(
                "Reduction parameters are inconsistent with input shape"
            )
        if (
            math.prod(tensors[1]["dimensions"]) != output_elements
            or output_elements != outer * inner
        ):
            raise ValueError(
                "Reduction parameters are inconsistent with output shape"
            )
        strided_constants = _reduction_tensor_constants(
            tensors, axis, keep_dimensions
        )

        block_n = _next_power_of_two(extent)
        constants: dict[str, int | str] = {
            "N": extent,
            "OP": REDUCTION_OPERATIONS[operation],
            "BLOCK_M": 1,
            "BLOCK_N": block_n,
        }
        signature = {
            "x_ptr": pointer_type,
            "out_ptr": TRITON_POINTER_TYPES[tensor_data_types[1]],
            "M": "i32",
        }
        tensors_are_contiguous = all(
            _is_row_major_contiguous(tensor) for tensor in tensors
        )
        if tensors_are_contiguous and inner == 1:
            constants.update({"stride_xm": extent, "stride_xn": 1})
            return (
                "reduction_2d_kernel",
                signature,
                constants,
                (outer, 1, 1),
                [
                    ("tensor", None),
                    ("tensor", None),
                    ("scalar_i32", "outer"),
                ],
            )

        if tensors_are_contiguous:
            constants.update(
                {
                    "I": inner,
                    "stride_xo": extent * inner,
                    "stride_xr": inner,
                    "stride_xi": 1,
                }
            )
            return (
                "reduction_3d_kernel",
                signature,
                constants,
                (output_elements, 1, 1),
                [
                    ("tensor", None),
                    ("tensor", None),
                    ("scalar_i32", "output_elements"),
                ],
            )

        block_m = min(16, max(1, 65536 // block_n))
        constants.update(strided_constants)
        constants["BLOCK_M"] = block_m
        return (
            "reduction_strided_kernel",
            signature,
            constants,
            ((output_elements + block_m - 1) // block_m, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "output_elements"),
            ],
        )
    # Compatibility parser for schema-v2 requests emitted by the original
    # Conv2D-only Core. New requests use the N-D branch above.
    if operation == "conv2d_fprop":
        if len(tensor_data_types) != 3 or len(set(tensor_data_types)) != 1:
            raise ValueError("Conv2D FProp tensor data types must match")
        pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        if pointer_type is None:
            raise ValueError(
                "unsupported Conv2D FProp data type: "
                f"{tensor_data_types[0]!r}"
            )
        names = ("n", "c", "h", "w", "k", "r", "s", "oh", "ow")
        dimensions = {
            name: _require_integer(parameters, name) for name in names
        }
        pad_top = _require_integer(parameters, "pad_top", minimum=0)
        pad_bottom = _require_integer(parameters, "pad_bottom", minimum=0)
        pad_left = _require_integer(parameters, "pad_left", minimum=0)
        pad_right = _require_integer(parameters, "pad_right", minimum=0)
        stride_h = _require_integer(parameters, "stride_h")
        stride_w = _require_integer(parameters, "stride_w")
        dilation_h = _require_integer(parameters, "dilation_h")
        dilation_w = _require_integer(parameters, "dilation_w")
        groups = _require_integer(parameters, "groups")
        outputs = _require_integer(parameters, "n_outputs")
        if any(len(tensor["dimensions"]) != 4 for tensor in tensors):
            raise ValueError("Conv2D FProp requires rank-4 tensors")
        if any(
            not _has_non_overlapping_strides(
                tensor["dimensions"], tensor["strides"]
            )
            for tensor in tensors
        ):
            raise ValueError(
                "Conv2D FProp tensors must have non-overlapping strides"
            )
        if dimensions["c"] % groups != 0 or dimensions["k"] % groups != 0:
            raise ValueError("Conv2D FProp channels must divide groups")
        channels_per_group = dimensions["c"] // groups
        outputs_per_group = dimensions["k"] // groups
        expected_input = [
            dimensions["n"],
            dimensions["c"],
            dimensions["h"],
            dimensions["w"],
        ]
        expected_filter = [
            dimensions["k"],
            channels_per_group,
            dimensions["r"],
            dimensions["s"],
        ]
        expected_output = [
            dimensions["n"],
            dimensions["k"],
            dimensions["oh"],
            dimensions["ow"],
        ]
        if tensors[0]["dimensions"] != expected_input:
            raise ValueError("Conv2D FProp input metadata is inconsistent")
        if tensors[1]["dimensions"] != expected_filter:
            raise ValueError("Conv2D FProp filter metadata is inconsistent")
        if tensors[2]["dimensions"] != expected_output:
            raise ValueError("Conv2D FProp output metadata is inconsistent")
        expected_oh = (
            dimensions["h"]
            + pad_top
            + pad_bottom
            - dilation_h * (dimensions["r"] - 1)
            - 1
        ) // stride_h + 1
        expected_ow = (
            dimensions["w"]
            + pad_left
            + pad_right
            - dilation_w * (dimensions["s"] - 1)
            - 1
        ) // stride_w + 1
        if expected_oh != dimensions["oh"] or expected_ow != dimensions["ow"]:
            raise ValueError("Conv2D FProp output dimensions are inconsistent")
        if outputs != (
            dimensions["n"]
            * dimensions["k"]
            * dimensions["oh"]
            * dimensions["ow"]
        ):
            raise ValueError("parameters.n_outputs is inconsistent with shape")
        block_oc = 16
        block_hw = 16
        block_k = 16
        return (
            "conv2d_spatial_nchw_kernel",
            {
                "x_ptr": pointer_type,
                "w_ptr": pointer_type,
                "bias_ptr": pointer_type,
                "y_ptr": pointer_type,
            },
            {
                "XH": dimensions["h"],
                "XW": dimensions["w"],
                "OH": dimensions["oh"],
                "OW": dimensions["ow"],
                "C_IN": dimensions["c"],
                "C_OUT": dimensions["k"],
                "CIN_PER_GROUP": channels_per_group,
                "COUT_PER_GROUP": outputs_per_group,
                "GROUPS": groups,
                "STRIDE_H": stride_h,
                "STRIDE_W": stride_w,
                "PAD_TOP": pad_top,
                "PAD_LEFT": pad_left,
                "DIL_H": dilation_h,
                "DIL_W": dilation_w,
                "KH": dimensions["r"],
                "KW": dimensions["s"],
                "HAS_BIAS": False,
                "BLOCK_OC": block_oc,
                "BLOCK_HW": block_hw,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
                "DTYPE_ID": {
                    "float16": 0,
                    "bfloat16": 1,
                    "float32": 2,
                }[tensor_data_types[0]],
                "INPUT_PRECISION": 0,
                "X_STRIDE_N": tensors[0]["strides"][0],
                "X_STRIDE_C": tensors[0]["strides"][1],
                "X_STRIDE_H": tensors[0]["strides"][2],
                "X_STRIDE_W": tensors[0]["strides"][3],
                "W_STRIDE_K": tensors[1]["strides"][0],
                "W_STRIDE_C": tensors[1]["strides"][1],
                "W_STRIDE_R": tensors[1]["strides"][2],
                "W_STRIDE_S": tensors[1]["strides"][3],
                "Y_STRIDE_N": tensors[2]["strides"][0],
                "Y_STRIDE_C": tensors[2]["strides"][1],
                "Y_STRIDE_H": tensors[2]["strides"][2],
                "Y_STRIDE_W": tensors[2]["strides"][3],
            },
            (
                (
                    (dimensions["oh"] * dimensions["ow"] + block_hw - 1)
                    // block_hw
                )
                * ((outputs_per_group + block_oc - 1) // block_oc),
                dimensions["n"] * groups,
                1,
            ),
            [
                ("tensor", None),
                ("tensor", None),
                ("tensor_alias", -1),
                ("tensor", None),
            ],
        )
    raise ValueError(f"unsupported operation: {operation!r}")


def _kernel_configuration(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> KernelPlan:
    """Typed boundary between operator configuration and artifact emission."""
    plan = KernelPlan(
        *_kernel_configuration_impl(
            operation, parameters, tensors, architecture
        )
    )
    if plan.function_name in {
        "conv_dgrad_nd_kernel",
        "conv_dgrad2d_stride1_kernel",
        "conv_wgrad_nd_kernel",
    }:
        plan = plan._replace(
            constants=dict(plan.constants, NATIVE_TF32_RNE=architecture >= 90)
        )
    if plan.function_name == "conv2d_spatial_nchw_kernel":
        constants = dict(plan.constants, NATIVE_TF32_RNE=architecture >= 90)
        plan = plan._replace(constants=constants)
        if (
            constants["INPUT_PRECISION"] == 2
            and constants["X_STRIDE_C"] == 1
            and constants["W_STRIDE_C"] == 1
            and int(constants["COUT_PER_GROUP"]) <= 64
        ):
            constants.update(BLOCK_OC=32, BLOCK_HW=16, BLOCK_K=128)
            plan = plan._replace(
                grid=(
                    ((int(constants["OH"]) * int(constants["OW"]) + 15) // 16)
                    * ((int(constants["COUT_PER_GROUP"]) + 31) // 32),
                    plan.grid[1],
                    1,
                )
            )
    if operation in {"sdpa", "sdpa_backward", "sdpa_fp8", "sdpa_fp8_backward"}:
        # Graph shapes, strides and mask bounds are fixed at build time.
        # Specialization exposes contiguous accesses and removes integer
        # division from the attention inner loops.
        signature = dict(plan.signature)
        constants = dict(plan.constants)
        arguments = []
        for name, argument in zip(
            plan.signature, plan.argument_layout, strict=True
        ):
            kind, source = argument
            if kind == "scalar_i32":
                constants[name] = parameters[str(source)]
                del signature[name]
            else:
                arguments.append(argument)
        plan = plan._replace(
            signature=signature, constants=constants, argument_layout=arguments
        )
    return plan
