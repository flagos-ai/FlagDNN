# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.registry.core import NormalizeFn, OpDef, register_op_def
from flag_dnn.graph.registry.schemas._run_common import (
    _format_bias,
    _require_runtime_backend,
    _unsupported_triton_path,
)
from flag_dnn.graph.registry.schemas.common import (
    _pop_operand,
    _shape_like_first,
)
from flag_dnn.graph.tensor import TensorSpec, canonical_dtype, torch_dtype


def _broadcast_shape(
    a: tuple[Any, ...], b: tuple[Any, ...]
) -> tuple[Any, ...]:
    try:
        if all(isinstance(v, int) for v in a + b):
            return tuple(torch.broadcast_shapes(tuple(a), tuple(b)))
    except RuntimeError as exc:
        raise RuntimeError(
            f"graph broadcast shape mismatch: {a} and {b}"
        ) from exc

    result = []
    ra, rb = list(reversed(a)), list(reversed(b))
    for idx in range(max(len(ra), len(rb))):
        da = ra[idx] if idx < len(ra) else 1
        db = rb[idx] if idx < len(rb) else 1
        if da == 1:
            result.append(db)
        elif db == 1 or da == db:
            result.append(da)
        else:
            raise NotImplementedError(
                "graph symbolic broadcast only supports equal or unit dims"
            )
    return tuple(reversed(result))


def _binary_result_dtype(
    op_type: str,
    input_specs: list[TensorSpec],
    attrs: dict[str, Any],
) -> str:
    if op_type in (
        "eq",
        "ne",
        "lt",
        "le",
        "ge",
        "gt",
        "logical_and",
        "logical_or",
        "logical_not",
    ):
        return "bool"
    if len(input_specs) > 1:
        left = torch.empty((), dtype=torch_dtype(input_specs[0].dtype))
        right = torch.empty((), dtype=torch_dtype(input_specs[1].dtype))
        return canonical_dtype(torch.result_type(left, right))

    other = attrs.get("other")
    left = torch.empty((), dtype=torch_dtype(input_specs[0].dtype))
    return canonical_dtype(torch.result_type(left, other))


def _binary_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    op_type = attrs["op_type"]
    left = input_specs[0]
    if len(input_specs) > 1:
        shape = _broadcast_shape(left.shape, input_specs[1].shape)
    else:
        shape = left.shape
    return [
        TensorSpec(
            name="",
            shape=shape,
            dtype=_binary_result_dtype(op_type, input_specs, attrs),
            layout=left.layout,
            device=left.device,
        )
    ]


def _binary_select_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    del attrs
    input0, input1, mask = input_specs
    shape = _broadcast_shape(
        _broadcast_shape(input0.shape, input1.shape), mask.shape
    )
    out_dtype = canonical_dtype(
        torch.result_type(
            torch.empty((), dtype=torch_dtype(input0.dtype)),
            torch.empty((), dtype=torch_dtype(input1.dtype)),
        )
    )
    return [
        TensorSpec(
            name="",
            shape=shape,
            dtype=out_dtype,
            layout=input0.layout,
            device=input0.device,
        )
    ]


def _bias_add_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    x, bias = input_specs[0], input_specs[1]
    if len(bias.shape) != 1:
        raise RuntimeError("graph bias_add expects a 1D bias")
    if len(x.shape) >= 2 and isinstance(x.shape[1], int):
        if bias.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"graph bias_add expected bias size {x.shape[1]}, "
                f"got {bias.shape[0]}"
            )
    return [
        TensorSpec(
            name="",
            shape=x.shape,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
    ]


def _normalize_unary(
    op_type: str,
    allowed_attrs: tuple[str, ...] = (),
) -> NormalizeFn:
    def normalize(ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) > 1:
            raise TypeError(f"{op_type} expects one positional tensor")
        params = dict(kwargs)
        if args:
            x = args[0]
        elif "input" in params:
            x = params.pop("input")
        else:
            x = params.pop("x", None)
        if x is None:
            raise TypeError(f"{op_type} missing input tensor")
        for key in list(params):
            if key not in allowed_attrs:
                raise TypeError(f"{op_type} got unsupported graph attr {key}")
        if params.get("inplace"):
            raise NotImplementedError(
                "FlagDNN graph does not support inplace ops"
            )
        return [ctx.as_value(x, name_hint="input")], params

    return normalize


def _normalize_activation_backward(op_type: str) -> NormalizeFn:
    def normalize(ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) > 2:
            raise TypeError(f"{op_type} expects at most two positional args")
        params = dict(kwargs)
        if "out" in params and params["out"] is not None:
            raise NotImplementedError(
                "FlagDNN graph does not support out tensors"
            )
        params.pop("out", None)
        loss = args[0] if args else params.pop("loss", None)
        if loss is None:
            raise TypeError(f"{op_type} missing loss tensor")
        if len(args) >= 2:
            x = args[1]
        else:
            x = params.pop("input", None)
        if x is None:
            raise TypeError(f"{op_type} missing input tensor")
        attrs = {
            "compute_data_type": params.pop("compute_data_type", None),
            "name": params.pop("name", ""),
        }
        if params:
            raise TypeError(
                f"{op_type} got unsupported graph attrs: {sorted(params)}"
            )
        return [
            ctx.as_value(loss, name_hint="loss"),
            ctx.as_value(x, name_hint="input"),
        ], attrs

    return normalize


def _activation_backward_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    loss, x = input_specs[0], input_specs[1]
    if loss.shape != x.shape:
        raise RuntimeError(
            "graph activation backward expects loss and input to have the "
            f"same shape, got {loss.shape} and {x.shape}"
        )
    return [
        TensorSpec(
            name="",
            shape=x.shape,
            dtype=canonical_dtype(
                torch.result_type(
                    torch.empty((), dtype=torch_dtype(loss.dtype)),
                    torch.empty((), dtype=torch_dtype(x.dtype)),
                )
            ),
            layout=x.layout,
            device=x.device,
        )
    ]


_CMP_ALIAS_TO_OP = {
    "cmp_eq": "eq",
    "cmp_neq": "ne",
    "cmp_lt": "lt",
    "cmp_le": "le",
    "cmp_gt": "gt",
    "cmp_ge": "ge",
}


def _normalize_binary(op_type: str) -> NormalizeFn:
    def normalize(ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) > 2:
            raise TypeError(f"{op_type} expects at most two positional args")
        params = dict(kwargs)
        if "out" in params and params["out"] is not None:
            raise NotImplementedError(
                "FlagDNN graph does not support out tensors"
            )
        params.pop("out", None)
        left = (
            args[0] if args else _pop_operand(params, ("input", "a", "input0"))
        )
        if left is None:
            raise TypeError(f"{op_type} missing input tensor")
        if len(args) >= 2:
            right = args[1]
        else:
            right = _pop_operand(params, ("other", "b", "input1"))
        if right is None:
            raise TypeError(f"{op_type} missing other operand")
        attrs = {
            "op_type": op_type,
            "alpha": params.pop("alpha", 1),
            "rounding_mode": params.pop("rounding_mode", None),
        }
        attrs.update(params)
        input_ids = [ctx.as_value(left, name_hint="input")]
        if ctx.is_tensor_like(right):
            input_ids.append(ctx.as_value(right, name_hint="other"))
        else:
            attrs["other"] = right
        return input_ids, attrs

    return normalize


def _normalize_binary_select(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 3:
        raise TypeError("binary_select expects at most three positional args")
    params = dict(kwargs)
    if "out" in params and params["out"] is not None:
        raise NotImplementedError("FlagDNN graph does not support out tensors")
    params.pop("out", None)

    input0 = args[0] if args else _pop_operand(params, ("input0", "a"))
    if input0 is None:
        raise TypeError("binary_select missing input0 tensor")
    if len(args) >= 2:
        input1 = args[1]
    else:
        input1 = _pop_operand(params, ("input1", "b"))
    if input1 is None:
        raise TypeError("binary_select missing input1 tensor")
    if len(args) >= 3:
        mask = args[2]
    else:
        mask = _pop_operand(params, ("mask", "condition"))
    if mask is None:
        raise TypeError("binary_select missing mask tensor")

    attrs = {
        "op_type": "binary_select",
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            f"binary_select got unsupported graph attrs: {sorted(params)}"
        )
    return [
        ctx.as_value(input0, "input0"),
        ctx.as_value(input1, "input1"),
        ctx.as_value(mask, "mask"),
    ], attrs


def _normalize_scale(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("scale expects at most two positional args")
    params = dict(kwargs)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("scale missing input tensor")
    if len(args) >= 2:
        scale_value = args[1]
    else:
        scale_value = params.pop("scale", None)
    if scale_value is None:
        raise TypeError("scale missing scale tensor")
    attrs = {
        "op_type": "mul",
        "alpha": 1,
        "rounding_mode": None,
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"scale got unsupported graph attrs: {sorted(params)}")
    return [
        ctx.as_value(x, name_hint="input"),
        ctx.as_value(scale_value, name_hint="scale"),
    ], attrs


def _normalize_pow(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("pow expects at most two positional args")
    params = dict(kwargs)
    if "out" in params and params["out"] is not None:
        raise NotImplementedError("FlagDNN graph does not support out tensors")
    params.pop("out", None)
    left = args[0] if args else _pop_operand(params, ("input", "input0", "a"))
    if left is None:
        raise TypeError("pow missing input tensor")
    if len(args) >= 2:
        right = args[1]
    else:
        right = _pop_operand(params, ("exponent", "input1", "other", "b"))
    if right is None:
        raise TypeError("pow missing exponent operand")
    attrs = {
        "op_type": "pow",
        "alpha": 1,
        "rounding_mode": None,
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"pow got unsupported graph attrs: {sorted(params)}")
    input_ids = [ctx.as_value(left, name_hint="input")]
    if ctx.is_tensor_like(right):
        input_ids.append(ctx.as_value(right, name_hint="exponent"))
    else:
        attrs["other"] = right
    return input_ids, attrs


def _normalize_cmp_alias(alias_name: str, op_type: str) -> NormalizeFn:
    def normalize(ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) > 2:
            raise TypeError(
                f"{alias_name} expects at most two positional args"
            )
        params = dict(kwargs)
        if "out" in params and params["out"] is not None:
            raise NotImplementedError(
                "FlagDNN graph does not support out tensors"
            )
        params.pop("out", None)
        left = (
            args[0] if args else _pop_operand(params, ("input", "a", "input0"))
        )
        if left is None:
            raise TypeError(f"{alias_name} missing input tensor")
        if len(args) >= 2:
            right = args[1]
        else:
            right = _pop_operand(
                params, ("comparison", "other", "b", "input1")
            )
        if right is None:
            raise TypeError(f"{alias_name} missing comparison operand")
        attrs = {
            "op_type": op_type,
            "alpha": 1,
            "rounding_mode": None,
            "compute_data_type": params.pop("compute_data_type", None),
            "name": params.pop("name", ""),
        }
        if params:
            raise TypeError(
                f"{alias_name} got unsupported graph attrs: {sorted(params)}"
            )
        input_ids = [ctx.as_value(left, name_hint="input")]
        if ctx.is_tensor_like(right):
            input_ids.append(ctx.as_value(right, name_hint="comparison"))
        else:
            attrs["other"] = right
        return input_ids, attrs

    return normalize


def _normalize_bias_add(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("bias_add expects input and bias")
    params = dict(kwargs)
    x = args[0] if args else params.pop("input", None)
    bias = args[1] if len(args) > 1 else params.pop("bias", None)
    if x is None or bias is None:
        raise TypeError("bias_add missing input or bias")
    if params:
        raise TypeError(f"bias_add got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input"), ctx.as_value(bias, "bias")], {}


# --- eager-fallback run functions ---


def _run_bias_add(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "bias_add")
        x, bias = inputs
        return flag_ops.add(x, _format_bias(x, bias))

    return run


def _binary_operands(
    inputs: list[Any], attrs: dict[str, Any]
) -> tuple[Any, Any]:
    left = inputs[0]
    if len(inputs) > 1:
        right = inputs[1]
    else:
        right = attrs["other"]
    if attrs.get("reverse"):
        return right, left
    return left, right


def _run_binary(flag_ops: Any, op_type: str) -> Any:
    if op_type == "add":
        return _run_binary_add(flag_ops)
    if op_type == "sub":
        return _run_binary_sub(flag_ops)
    if op_type == "mul":
        return _run_binary_mul(flag_ops)
    if op_type == "div":
        return _run_binary_div(flag_ops)
    if op_type == "mod":
        return _run_binary_mod(flag_ops)
    if op_type == "pow":
        return _run_binary_pow(flag_ops)
    if op_type == "max":
        return _run_binary_max(flag_ops)
    if op_type in ("min", "minimum"):
        return _run_binary_min(flag_ops)
    if op_type == "maximum":
        return _run_binary_maximum(flag_ops)
    if op_type in ("logical_and", "logical_or"):
        return _run_binary_logical(flag_ops, op_type)
    if op_type == "add_square":
        return _run_add_square(flag_ops)
    if op_type in ("eq", "ne", "lt", "le", "gt", "ge"):
        return _run_binary_cmp(flag_ops, op_type)
    raise RuntimeError(f"unsupported graph binary op: {op_type}")


def _run_binary_add(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "add")
        left, right = _binary_operands(inputs, attrs)
        alpha = attrs.get("alpha", 1)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if not isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        ):
            return flag_ops.add(
                right,
                left,
                alpha=alpha,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(left, torch.Tensor):
            return flag_ops.add(
                left,
                right,
                alpha=alpha,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("add", "two scalar operands")

    return run


def _run_binary_sub(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "sub")
        left, right = _binary_operands(inputs, attrs)
        alpha = attrs.get("alpha", 1)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if isinstance(left, torch.Tensor):
            return flag_ops.sub(
                left,
                right,
                alpha=alpha,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("sub", "scalar left operand")

    return run


def _run_binary_mul(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "mul")
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if not isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        ):
            return flag_ops.mul(
                right,
                left,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(left, torch.Tensor):
            return flag_ops.mul(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("mul", "two scalar operands")

    return run


def _run_binary_div(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "div")
        left, right = _binary_operands(inputs, attrs)
        rounding_mode = attrs.get("rounding_mode")
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if isinstance(left, torch.Tensor):
            return flag_ops.div(
                left,
                right,
                rounding_mode=rounding_mode,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("div", "scalar left operand")

    return run


def _run_binary_mod(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "mod")
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if isinstance(left, torch.Tensor):
            return flag_ops.mod(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("mod", "scalar left operand")

    return run


def _run_binary_pow(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "pow")
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return flag_ops.pow(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("pow", "two scalar operands")

    return run


def _run_binary_max(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "max")
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if not isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        ):
            return flag_ops.max(
                right,
                left,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(left, torch.Tensor):
            return flag_ops.max(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("max", "two scalar operands")

    return run


def _run_binary_min(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "min")
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if not isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        ):
            return flag_ops.minimum(
                right,
                left,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(left, torch.Tensor):
            return flag_ops.minimum(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("min", "two scalar operands")

    return run


def _run_binary_maximum(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "maximum")
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if not isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        ):
            return flag_ops.maximum(
                right,
                left,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(left, torch.Tensor):
            return flag_ops.maximum(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path("maximum", "two scalar operands")

    return run


def _run_binary_logical(flag_ops: Any, op_type: str) -> Any:
    flag_fn = getattr(flag_ops, op_type)

    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, op_type)
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if not isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        ):
            return flag_fn(
                right,
                left,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(left, torch.Tensor):
            return flag_fn(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path(op_type, "two scalar operands")

    return run


def _run_add_square(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "add_square")
        return flag_ops.add_square(
            inputs[0],
            inputs[1],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_binary_cmp(flag_ops: Any, op_type: str) -> Any:
    flag_fn = getattr(flag_ops, op_type)
    reverse_op = {
        "eq": "eq",
        "ne": "ne",
        "lt": "gt",
        "le": "ge",
        "gt": "lt",
        "ge": "le",
    }[op_type]
    reverse_flag_fn = getattr(flag_ops, reverse_op)

    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, op_type)
        left, right = _binary_operands(inputs, attrs)
        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")
        if isinstance(left, torch.Tensor):
            return flag_fn(
                left,
                right,
                compute_data_type=compute_data_type,
                name=name,
            )
        if isinstance(right, torch.Tensor):
            return reverse_flag_fn(
                right,
                left,
                compute_data_type=compute_data_type,
                name=name,
            )
        _unsupported_triton_path(op_type, "two scalar operands")

    return run


def _run_binary_select(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "binary_select")
        return flag_ops.binary_select(
            inputs[0],
            inputs[1],
            inputs[2],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_relu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "relu")
        return flag_ops.relu(
            inputs[0],
            inplace=attrs.get("inplace", False),
            negative_slope=attrs.get("negative_slope"),
            lower_clip=attrs.get("lower_clip"),
            upper_clip=attrs.get("upper_clip"),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_swish(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "swish")
        return flag_ops.swish(
            inputs[0],
            swish_beta=attrs.get("swish_beta"),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_gelu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gelu")
        return flag_ops.gelu(
            inputs[0], approximate=attrs.get("approximate", "none")
        )

    return run


def _run_gelu_approx_tanh(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gelu_approx_tanh")
        return flag_ops.gelu(inputs[0], approximate="tanh")

    return run


def _run_leaky_relu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "leaky_relu")
        negative_slope = attrs.get("negative_slope", 0.01)
        if negative_slope is None:
            negative_slope = 0.01
        return flag_ops.leaky_relu(
            inputs[0],
            negative_slope=float(negative_slope),
            inplace=False,
        )

    return run


def _run_elu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "elu")
        alpha = attrs.get("alpha", 1.0)
        if alpha is None:
            alpha = 1.0
        return flag_ops.elu(inputs[0], alpha=float(alpha), inplace=False)

    return run


def _run_softplus(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "softplus")
        beta = attrs.get("beta", 1.0)
        threshold = attrs.get("threshold", 20.0)
        if beta is None:
            beta = 1.0
        if threshold is None:
            threshold = 20.0
        return flag_ops.softplus(
            inputs[0],
            beta=float(beta),
            threshold=float(threshold),
        )

    return run


def _run_unary_flag(flag_ops: Any, op_type: str) -> Any:
    flag_fn = getattr(flag_ops, op_type)

    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, op_type)
        return flag_fn(inputs[0])

    return run


def _run_logical_not(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "logical_not")
        return flag_ops.logical_not(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_abs(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "abs")
        return flag_ops.abs(inputs[0])

    return run


def _run_sigmoid(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "sigmoid")
        return flag_ops.sigmoid(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_sigmoid_backward(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "sigmoid_backward")
        return flag_ops.sigmoid_backward(
            inputs[0],
            inputs[1],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


# --- op specs and registration ---

_BINARY_OP_TYPES = (
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "max",
    "min",
    "minimum",
    "maximum",
    "logical_and",
    "logical_or",
    "add_square",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
)

_ACTIVATION_OP_SPECS = (
    (
        "relu",
        (
            "inplace",
            "negative_slope",
            "lower_clip",
            "upper_clip",
            "compute_data_type",
            "name",
        ),
        _run_relu,
    ),
    ("swish", ("swish_beta", "compute_data_type", "name"), _run_swish),
    ("gelu", ("approximate",), _run_gelu),
    (
        "gelu_approx_tanh",
        ("compute_data_type", "name"),
        _run_gelu_approx_tanh,
    ),
    (
        "leaky_relu",
        ("negative_slope", "compute_data_type", "name", "inplace"),
        _run_leaky_relu,
    ),
    ("elu", ("alpha", "compute_data_type", "name", "inplace"), _run_elu),
    (
        "softplus",
        ("beta", "threshold", "compute_data_type", "name"),
        _run_softplus,
    ),
)

_UNARY_FLAG_OP_TYPES = (
    "sqrt",
    "square",
    "rsqrt",
    "exp",
    "log",
    "reciprocal",
    "ceil",
    "floor",
    "erf",
    "sin",
    "cos",
    "tan",
    "neg",
    "tanh",
    "silu",
)

_POINTWISE_UNARY_OP_SPECS = (
    ("logical_not", _run_logical_not),
    ("sigmoid", _run_sigmoid),
    ("abs", _run_abs),
)


def register(flag_ops: Any) -> None:
    """Register the pointwise op family: binary arithmetic/compare/logical,
    scale/pow/bias_add, activations and simple unary ops."""
    for op_type in _BINARY_OP_TYPES:
        register_op_def(
            OpDef(
                name=op_type,
                normalize=_normalize_binary(op_type),
                shape=_binary_shape,
                run=_run_binary(flag_ops, op_type),
            )
        )
    register_op_def(
        OpDef(
            name="binary_select",
            normalize=_normalize_binary_select,
            shape=_binary_select_shape,
            run=_run_binary_select(flag_ops),
        )
    )
    register_op_def(
        OpDef(
            name="scale",
            normalize=_normalize_scale,
            shape=_binary_shape,
            run=_run_binary(flag_ops, "mul"),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="pow",
            normalize=_normalize_pow,
            shape=_binary_shape,
            run=_run_binary(flag_ops, "pow"),
        )
    )
    for alias_name, op_type in _CMP_ALIAS_TO_OP.items():
        register_op_def(
            OpDef(
                name=alias_name,
                normalize=_normalize_cmp_alias(alias_name, op_type),
                shape=_binary_shape,
                run=_run_binary(flag_ops, op_type),
            )
        )
    register_op_def(
        OpDef(
            name="bias_add",
            normalize=_normalize_bias_add,
            shape=_bias_add_shape,
            run=_run_bias_add(flag_ops),
            fusible=True,
        )
    )
    for name, attrs, run_factory in _ACTIVATION_OP_SPECS:
        register_op_def(
            OpDef(
                name=name,
                normalize=_normalize_unary(name, attrs),
                shape=_shape_like_first,
                run=run_factory(flag_ops),
                fusible=True,
            )
        )
    for name in _UNARY_FLAG_OP_TYPES:
        register_op_def(
            OpDef(
                name=name,
                normalize=_normalize_unary(
                    name, ("compute_data_type", "name")
                ),
                shape=_shape_like_first,
                run=_run_unary_flag(flag_ops, name),
                fusible=True,
            )
        )
    for name, run_factory in _POINTWISE_UNARY_OP_SPECS:
        register_op_def(
            OpDef(
                name=name,
                normalize=_normalize_unary(
                    name, ("compute_data_type", "name")
                ),
                shape=_shape_like_first,
                run=run_factory(flag_ops),
                fusible=True,
            )
        )
    register_op_def(
        OpDef(
            name="sigmoid_backward",
            normalize=_normalize_activation_backward("sigmoid_backward"),
            shape=_activation_backward_shape,
            run=_run_sigmoid_backward(flag_ops),
        )
    )


__all__ = (
    "register",
    "_broadcast_shape",
    "_binary_result_dtype",
    "_binary_shape",
    "_binary_select_shape",
    "_bias_add_shape",
    "_normalize_unary",
    "_normalize_activation_backward",
    "_activation_backward_shape",
    "_CMP_ALIAS_TO_OP",
    "_normalize_binary",
    "_normalize_binary_select",
    "_normalize_scale",
    "_normalize_pow",
    "_normalize_cmp_alias",
    "_normalize_bias_add",
)
