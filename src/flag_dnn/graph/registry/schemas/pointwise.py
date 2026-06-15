from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.registry.core import NormalizeFn
from flag_dnn.graph.registry.schemas.common import _pop_operand
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


__all__ = (
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
