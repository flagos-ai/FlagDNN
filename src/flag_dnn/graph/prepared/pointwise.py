from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from flag_dnn.graph.prepared import RunFn, register_generic_prepared_run_fn
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _require_runtime_backend,
    _unsupported_triton_path,
)
from flag_dnn.graph.tensor import TensorSpec

# Pointwise prepared paths

_POINTWISE_BINARY_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}

_POINTWISE_CMP_REVERSE = {
    "eq": "eq",
    "ne": "ne",
    "lt": "gt",
    "le": "ge",
    "gt": "lt",
    "ge": "le",
}

_POINTWISE_CMP_ALIASES = {
    "cmp_eq": "eq",
    "cmp_neq": "ne",
    "cmp_lt": "lt",
    "cmp_le": "le",
    "cmp_gt": "gt",
    "cmp_ge": "ge",
}


@register_generic_prepared_run_fn
def _prepare_pointwise(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if not input_specs or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None

    actual_op_type = _POINTWISE_CMP_ALIASES.get(
        op_type, attrs.get("op_type", op_type)
    )
    if actual_op_type in _POINTWISE_BINARY_OPS:
        return _prepare_binary_pointwise(actual_op_type, attrs, default_run_fn)
    if actual_op_type == "pow":
        return _prepare_pow_pointwise(attrs, default_run_fn)
    if op_type == "abs":
        from flag_dnn.ops.abs import abs as abs_op

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "abs")
            return abs_op(inputs[0])

        return run
    if op_type == "sigmoid":
        from flag_dnn.ops.sigmoid import sigmoid

        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "sigmoid")
            return sigmoid(
                inputs[0], compute_data_type=compute_data_type, name=name
            )

        return run
    if op_type == "sigmoid_backward":
        from flag_dnn.ops.sigmoid_backward import sigmoid_backward

        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "sigmoid_backward")
            return sigmoid_backward(
                inputs[0],
                inputs[1],
                compute_data_type=compute_data_type,
                name=name,
            )

        return run
    return None


def _pointwise_operands(
    inputs: Sequence[Any], attrs: dict[str, Any]
) -> tuple[Any, Any]:
    left = inputs[0]
    if len(inputs) > 1:
        right = inputs[1]
    else:
        right = attrs["other"]
    if attrs.get("reverse"):
        return right, left
    return left, right


def _prepare_binary_pointwise(
    op_type: str, attrs: dict[str, Any], default_run_fn: RunFn
) -> RunFn:
    from flag_dnn.ops.binary import binary

    alpha = attrs.get("alpha", 1)
    rounding_mode = attrs.get("rounding_mode")

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, op_type)
        left, right = _pointwise_operands(inputs, attrs)
        if op_type == "add":
            if isinstance(left, torch.Tensor):
                return binary(left, right, alpha=alpha, op_type="add")
            if isinstance(right, torch.Tensor):
                return binary(right, left, alpha=alpha, op_type="add")
        elif op_type == "sub":
            if isinstance(left, torch.Tensor):
                return binary(left, right, alpha=alpha, op_type="sub")
        elif op_type == "mul":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="mul")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="mul")
        elif op_type == "div":
            if isinstance(left, torch.Tensor):
                return binary(
                    left,
                    right,
                    rounding_mode=rounding_mode,
                    op_type="div",
                )
        elif op_type == "max":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="max")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="max")
        else:
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type=op_type)
            if isinstance(right, torch.Tensor):
                return binary(
                    right,
                    left,
                    op_type=_POINTWISE_CMP_REVERSE[op_type],
                )
        _unsupported_triton_path(op_type, "operand combination")

    return run


def _prepare_pow_pointwise(
    attrs: dict[str, Any], default_run_fn: RunFn
) -> RunFn:
    from flag_dnn.ops.pow import pow as pow_op

    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "pow")
        left, right = _pointwise_operands(inputs, attrs)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return pow_op(
                left, right, compute_data_type=compute_data_type, name=name
            )
        _unsupported_triton_path("pow", "two scalar operands")

    return run
