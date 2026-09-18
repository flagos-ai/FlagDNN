"""
Pointwise dtype, broadcast, activation attribute, and kernel selection rules.
"""

from __future__ import annotations

import math

from flagdnn_codegen.kernel_registry import (
    BINARY_POINTWISE_OPERATIONS,
    TERNARY_POINTWISE_OPERATIONS,
    UNARY_POINTWISE_OPERATIONS,
)

from .metadata import (
    BINARY_POINTWISE_MODES,
    COMPARISON_POINTWISE_OPERATIONS,
    FLOAT_DATA_TYPES,
    LOGICAL_BINARY_POINTWISE_OPERATIONS,
    NUMERIC_DATA_TYPES,
    TRITON_POINTER_TYPES,
    UNARY_POINTWISE_MODES,
    _binary_pointwise_tensor_constants,
    _can_use_dense_binary_kernel,
    _can_use_dense_ternary_kernel,
    _require_integer,
    _require_number,
    _ternary_pointwise_tensor_constants,
    _unary_pointwise_tensor_constants,
)


def _pointwise_kernel_configuration(operation, parameters, tensors):
    tensor_data_types = [tensor["data_type"] for tensor in tensors]
    if operation == "relu" or operation in UNARY_POINTWISE_OPERATIONS:
        if len(tensor_data_types) != 2:
            raise ValueError("unary pointwise tensor count is invalid")
        if operation == "logical_not":
            if tensor_data_types != ["boolean", "boolean"]:
                raise ValueError(
                    "logical_not input/output data types must be boolean"
                )
        elif len(set(tensor_data_types)) != 1 or (
            operation != "identity"
            and tensor_data_types[0] not in FLOAT_DATA_TYPES
        ):
            raise ValueError(
                "numeric unary pointwise tensor data types must match and "
                "be floating"
            )
        input_pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        output_pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[1])
        if input_pointer_type is None or output_pointer_type is None:
            raise ValueError(
                "unsupported unary pointwise data type: "
                f"{tensor_data_types[0]!r}"
            )
        elements = _require_integer(parameters, "n_elements")
        expected_elements = math.prod(tensors[0]["dimensions"])
        if elements != expected_elements:
            raise ValueError(
                "parameters.n_elements is inconsistent with unary "
                "pointwise input"
            )
        has_upper_clip_value = parameters.get("has_upper_clip", 0)
        if (
            isinstance(has_upper_clip_value, bool)
            or not isinstance(has_upper_clip_value, int)
            or has_upper_clip_value < 0
            or has_upper_clip_value > 1
        ):
            raise ValueError(
                "parameters.has_upper_clip must be either zero or one"
            )
        has_upper_clip = has_upper_clip_value
        negative_slope = _require_number(
            parameters, "negative_slope", default=0.0
        )
        lower_clip = _require_number(parameters, "lower_clip", default=0.0)
        upper_clip = _require_number(parameters, "upper_clip", default=0.0)
        swish_beta = _require_number(parameters, "swish_beta", default=1.0)
        elu_alpha = _require_number(parameters, "elu_alpha", default=1.0)
        softplus_beta = _require_number(
            parameters, "softplus_beta", default=1.0
        )
        if softplus_beta <= 0.0:
            raise ValueError("parameters.softplus_beta must be positive")
        if has_upper_clip and upper_clip < lower_clip:
            raise ValueError(
                "parameters.upper_clip must not be less than lower_clip"
            )
        block = 256
        constants: dict[str, int | float | str | bool] = {
            "OPERATION": UNARY_POINTWISE_MODES[operation],
            "negative_slope": negative_slope,
            "lower_clip": lower_clip,
            "upper_clip": upper_clip,
            "HAS_UPPER_CLIP": bool(has_upper_clip),
            "SWISH_BETA": swish_beta,
            "ELU_ALPHA": elu_alpha,
            "SOFTPLUS_BETA": softplus_beta,
            "TILES_PER_PROGRAM": 1,
            "BLOCK_SIZE": block,
        }
        tensor_constants = _unary_pointwise_tensor_constants(tensors)
        function_name = "unary_pointwise_contiguous_kernel"
        if bool(tensor_constants["STRIDED"]):
            constants.update(tensor_constants)
            function_name = "unary_pointwise_strided_kernel"
        return (
            function_name,
            {
                "in_ptr": input_pointer_type,
                "out_ptr": output_pointer_type,
                "n_elements": "i32",
            },
            constants,
            ((elements + block - 1) // block, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "n_elements"),
            ],
        )
    if operation == "add" or operation in BINARY_POINTWISE_OPERATIONS:
        if len(tensor_data_types) != 3:
            raise ValueError("binary pointwise tensor count is invalid")
        if operation in COMPARISON_POINTWISE_OPERATIONS:
            if (
                tensor_data_types[0] != tensor_data_types[1]
                or tensor_data_types[0] not in NUMERIC_DATA_TYPES
                or tensor_data_types[2] != "boolean"
            ):
                raise ValueError(
                    "comparison pointwise requires matching floating inputs "
                    "and boolean output"
                )
        elif operation in LOGICAL_BINARY_POINTWISE_OPERATIONS:
            if tensor_data_types != ["boolean", "boolean", "boolean"]:
                raise ValueError(
                    "logical pointwise input/output data types must be boolean"
                )
        elif (
            len(set(tensor_data_types)) != 1
            or tensor_data_types[0] not in NUMERIC_DATA_TYPES
        ):
            raise ValueError(
                "numeric binary pointwise tensor data types must match and "
                "be floating"
            )
        if (
            operation.endswith("_backward")
            and tensor_data_types[0] not in FLOAT_DATA_TYPES
        ):
            raise ValueError("activation gradients must be floating")
        pointer_types = [
            TRITON_POINTER_TYPES.get(data_type)
            for data_type in tensor_data_types
        ]
        if any(pointer_type is None for pointer_type in pointer_types):
            raise ValueError(
                "unsupported binary pointwise data type: "
                f"{tensor_data_types[0]!r}"
            )
        elements = _require_integer(parameters, "n_elements")
        expected_elements = math.prod(tensors[2]["dimensions"])
        if elements != expected_elements:
            raise ValueError(
                "parameters.n_elements is inconsistent with binary "
                "pointwise output"
            )
        pointwise_mode = _require_integer(
            parameters, "pointwise_mode", minimum=1, maximum=48
        )
        if BINARY_POINTWISE_MODES.get(operation) != pointwise_mode:
            raise ValueError(
                "parameters.pointwise_mode is inconsistent with "
                "binary operation"
            )
        block = 256
        strided_constants = _binary_pointwise_tensor_constants(tensors)
        alpha = _require_number(parameters, "alpha", default=1.0)
        if operation not in ("add", "sub") and alpha != 1.0:
            raise ValueError(
                "pointwise alpha is only supported by add and sub"
            )

        if tensor_data_types[0] == "int32":
            if alpha != math.trunc(alpha) or not -(2**31) <= alpha < 2**31:
                raise ValueError("INT32 alpha must be representable as INT32")
            alpha = int(alpha)
        constants: dict[str, int | float] = {
            "OP_KIND": pointwise_mode,
            "ALPHA": alpha,
            "BLOCK_SIZE": block,
        }
        if operation.endswith("_backward"):
            if (
                tensors[0]["dimensions"] != tensors[1]["dimensions"]
                or tensors[0]["dimensions"] != tensors[2]["dimensions"]
            ):
                raise ValueError(
                    "activation gradient tensor shapes must match"
                )
            constants.update(
                {
                    "NEGATIVE_SLOPE": _require_number(
                        parameters, "relu_lower_clip_slope", default=0.0
                    ),
                    "LOWER_CLIP": _require_number(
                        parameters, "relu_lower_clip", default=0.0
                    ),
                    "UPPER_CLIP": _require_number(
                        parameters, "relu_upper_clip", default=0.0
                    ),
                    "HAS_UPPER_CLIP": bool(
                        _require_integer(
                            parameters, "has_upper_clip", minimum=0, maximum=1
                        )
                    ),
                    "SWISH_BETA": _require_number(
                        parameters, "swish_beta", default=1.0
                    ),
                    "ELU_ALPHA": _require_number(
                        parameters, "elu_alpha", default=1.0
                    ),
                    "SOFTPLUS_BETA": _require_number(
                        parameters, "softplus_beta", default=1.0
                    ),
                }
            )
            if constants["SOFTPLUS_BETA"] <= 0 or (
                constants["HAS_UPPER_CLIP"]
                and constants["UPPER_CLIP"] < constants["LOWER_CLIP"]
            ):
                raise ValueError("invalid activation gradient attributes")
        function_name = "binary_contiguous_kernel"
        if not _can_use_dense_binary_kernel(tensors):
            constants.update(strided_constants)
            function_name = "binary_strided_kernel"
        if operation.endswith("_backward"):
            function_name = function_name.replace(
                "binary_", "activation_backward_", 1
            )
            if function_name == "activation_backward_contiguous_kernel":
                constants["MASK_TAIL"] = elements % block != 0
        return (
            function_name,
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[1],
                "out_ptr": pointer_types[2],
                "n_elements": "i32",
            },
            constants,
            ((elements + block - 1) // block, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "n_elements"),
            ],
        )
    if operation in TERNARY_POINTWISE_OPERATIONS:
        if len(tensor_data_types) != 4:
            raise ValueError("ternary pointwise tensor count is invalid")
        if (
            tensor_data_types[0] != tensor_data_types[1]
            or tensor_data_types[0] != tensor_data_types[3]
            or tensor_data_types[0] not in NUMERIC_DATA_TYPES
            or tensor_data_types[2] != "boolean"
        ):
            raise ValueError(
                "binary_select requires matching floating A/B/output "
                "and a boolean T predicate"
            )
        pointer_types = [
            TRITON_POINTER_TYPES.get(data_type)
            for data_type in tensor_data_types
        ]
        if any(pointer_type is None for pointer_type in pointer_types):
            raise ValueError("unsupported binary_select data type")
        elements = _require_integer(parameters, "n_elements")
        if elements != math.prod(tensors[3]["dimensions"]):
            raise ValueError(
                "parameters.n_elements is inconsistent with binary_select "
                "output"
            )
        block = 256
        constants: dict[str, int] = {"BLOCK_SIZE": block}
        function_name = "binary_select_tensor_kernel"
        signature = {
            "input0_ptr": pointer_types[0],
            "input1_ptr": pointer_types[1],
            "mask_ptr": pointer_types[2],
            "out_ptr": pointer_types[3],
            "n_elements": "i32",
        }
        if not _can_use_dense_ternary_kernel(tensors):
            constants.update(_ternary_pointwise_tensor_constants(tensors))
            function_name = "binary_select_strided_kernel"
            signature = {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[1],
                "t_ptr": pointer_types[2],
                "out_ptr": pointer_types[3],
                "n_elements": "i32",
            }
        return (
            function_name,
            signature,
            constants,
            ((elements + block - 1) // block, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "n_elements"),
            ],
        )

    raise ValueError(f"unsupported pointwise operation: {operation!r}")
