# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Route validated operation families to their artifact generators."""

from pathlib import Path
from typing import Any

from .attention import _compile_attention
from .batchnorm import _compile_batchnorm
from .convolution import _compile_convolution
from .extended import _compile_extended
from .layout import _compile_layout
from .matmul import _compile_matmul
from .normalization import _compile_normalization
from .pointwise import (
    _compile_add_square,
    _compile_binary_pointwise,
    _compile_unary_pointwise,
)
from .reduction import _compile_reduction
from ..dispatch.common import (
    _BATCHNORM_OPERATIONS,
    _BINARY_POINTWISE_MODES,
    _CONVOLUTION_OPERATIONS,
    _CONV_BIAS_RELU_OPERATION_TYPES,
    _LAYOUT_OPERATIONS,
    _MATMUL_OPERATIONS,
    _NORMALIZATION_OPERATIONS,
    _REDUCTION_OPERATIONS,
    _UNARY_POINTWISE_MODES,
)
from ..dispatch.extended import SUPPORTED_OPERATIONS


_OPERATION_GENERATORS = {
    **dict.fromkeys(SUPPORTED_OPERATIONS, _compile_extended),
    **dict.fromkeys(
        ("sdpa", "sdpa_backward", "sdpa_fp8", "sdpa_fp8_backward"),
        _compile_attention,
    ),
    **dict.fromkeys(_BINARY_POINTWISE_MODES, _compile_binary_pointwise),
    "binary_select": _compile_binary_pointwise,
    **dict.fromkeys(_UNARY_POINTWISE_MODES, _compile_unary_pointwise),
    **dict.fromkeys(_LAYOUT_OPERATIONS, _compile_layout),
    **dict.fromkeys(_REDUCTION_OPERATIONS, _compile_reduction),
    **dict.fromkeys(_NORMALIZATION_OPERATIONS, _compile_normalization),
    **dict.fromkeys(_BATCHNORM_OPERATIONS, _compile_batchnorm),
    **dict.fromkeys(_CONVOLUTION_OPERATIONS, _compile_convolution),
}


def _compile_graph_operation(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation_types: list[str],
    execution_engine: str,
) -> dict[str, Any]:
    arguments: dict[str, Any] = dict(
        request=request,
        request_bytes=request_bytes,
        identity=identity,
        target=target,
        output_directory=output_directory,
        enable_autotune=enable_autotune,
    )
    if operation_types == _CONV_BIAS_RELU_OPERATION_TYPES:
        return _compile_convolution(**arguments, operation="conv_bias_relu")
    if operation_types == ["add", "mul"]:
        return _compile_add_square(**arguments)
    if len(operation_types) == 1:
        operation = operation_types[0]
        if operation in _MATMUL_OPERATIONS:
            return _compile_matmul(**arguments)
        generator = _OPERATION_GENERATORS.get(operation)
        if generator is not None:
            return generator(**arguments, operation=operation)
    return {
        "schema_version": 1,
        "status": "unsupported",
        "reason_code": "operation_family_not_implemented",
        "backend": "thead",
        "target": target,
        "execution_engine": execution_engine,
        "operation_types": operation_types,
    }
