from __future__ import annotations

import functools
from typing import Any, Callable

import torch

from flag_dnn.graph.capture import current_capture, is_capturing
from flag_dnn.graph.tensor import GraphTensor

GRAPH_AWARE_OPS = (
    "identity",
    "reshape",
    "transpose",
    "slice",
    "concatenate",
    "gen_index",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "cmp_eq",
    "cmp_neq",
    "cmp_lt",
    "cmp_le",
    "cmp_gt",
    "cmp_ge",
    "bias_add",
    "relu",
    "gelu",
    "conv2d",
    "conv_fprop",
    "mm",
    "sqrt",
    "abs",
    "neg",
    "tanh",
    "silu",
)


def install_graph_wrappers(namespace: dict[str, Any]) -> None:
    for op_type in GRAPH_AWARE_OPS:
        eager_fn = namespace.get(op_type)
        if eager_fn is None and op_type == "bias_add":
            eager_fn = eager_bias_add
        if eager_fn is None:
            continue
        if getattr(eager_fn, "__flagdnn_graph_wrapped__", False):
            continue
        namespace[op_type] = make_graph_wrapper(op_type, eager_fn)


def make_graph_wrapper(op_type: str, eager_fn: Callable[..., Any]):
    @functools.wraps(eager_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if (
            is_capturing()
            or _contains_graph_tensor(args)
            or _contains_graph_tensor(kwargs)
        ):
            ctx = current_capture()
            if ctx is None:
                raise RuntimeError(
                    f"FlagDNN graph op {op_type} used outside graph capture"
                )
            return ctx.add_op_call(op_type, args, kwargs)
        return eager_fn(*args, **kwargs)

    wrapper.__flagdnn_graph_wrapped__ = True
    wrapper.__flagdnn_eager_fn__ = eager_fn
    return wrapper


def eager_bias_add(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if bias.dim() == 1 and input.dim() >= 2:
        shape = [1] * input.dim()
        shape[1] = bias.numel()
        bias = bias.reshape(shape)
    return input + bias


def _contains_graph_tensor(value: Any) -> bool:
    if isinstance(value, GraphTensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_graph_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_graph_tensor(item) for item in value.values())
    return False
