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
    "scale",
    "div",
    "mod",
    "pow",
    "max",
    "min",
    "minimum",
    "maximum",
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
    "add_square",
    "relu",
    "swish",
    "gelu",
    "gelu_approx_tanh",
    "leaky_relu",
    "elu",
    "softplus",
    "conv2d",
    "conv_fprop",
    "conv_dgrad",
    "causal_conv1d",
    "mm",
    "matmul",
    "batchnorm",
    "batchnorm_inference",
    "layernorm",
    "rmsnorm",
    "rmsnorm_rht_amax_wrapper_sm100",
    "reduction",
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
    "abs",
    "neg",
    "tanh",
    "sigmoid",
    "sigmoid_backward",
    "silu",
    "logical_and",
    "logical_or",
    "logical_not",
    "binary_select",
)


_DICT_OUTPUT_OPS = {
    "rmsnorm_rht_amax_wrapper_sm100": ("o_tensor", "amax_tensor"),
}


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
            outputs = ctx.add_op_call(op_type, args, kwargs)
            output_keys = _DICT_OUTPUT_OPS.get(op_type)
            if output_keys is not None:
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                if len(outputs) != len(output_keys):
                    raise RuntimeError(
                        f"FlagDNN graph op {op_type} returned "
                        f"{len(outputs)} outputs, expected {len(output_keys)}"
                    )
                return dict(zip(output_keys, outputs))
            return outputs
        return eager_fn(*args, **kwargs)

    setattr(wrapper, "__flagdnn_graph_wrapped__", True)
    setattr(wrapper, "__flagdnn_eager_fn__", eager_fn)
    return wrapper


def eager_bias_add(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if bias.dim() == 1 and input.dim() >= 2:
        shape = [1] * input.dim()
        shape[1] = bias.numel()
        bias = bias.reshape(shape)
    from flag_dnn.ops.add import add

    return add(input, bias)


def _contains_graph_tensor(value: Any) -> bool:
    if isinstance(value, GraphTensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_graph_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_graph_tensor(item) for item in value.values())
    return False
