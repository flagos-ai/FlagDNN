from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.graph.registry.core import (
    OpSchema,
    RunFn,
    get_registered_op,
    register_op,
    registered_raw_ops,
)
from flag_dnn.graph.tensor import torch_dtype
from flag_dnn.graph.registry.schemas.common import (
    _shape_like_first,
)
from flag_dnn.graph.registry.schemas.utility import (
    _concatenate_shape,
    _gen_index_shape,
    _normalize_concatenate,
    _normalize_gen_index,
    _normalize_reshape,
    _normalize_slice,
    _normalize_transpose,
    _reshape_shape,
    _slice_shape,
    _transpose_shape,
)
from flag_dnn.graph.registry.schemas.conv import (
    _conv2d_shape,
    _conv_dgrad_shape,
    _conv_fprop_shape,
    _conv_wgrad_shape,
    _normalize_causal_conv1d,
    _normalize_conv2d,
    _normalize_conv_dgrad,
    _normalize_conv_fprop,
    _normalize_conv_wgrad,
)
from flag_dnn.graph.registry.schemas.matmul_attention import (
    _matmul_shape,
    _mm_shape,
    _normalize_matmul,
    _normalize_mm,
    _normalize_sdpa,
    _normalize_sdpa_backward,
    _sdpa_backward_shape,
    _sdpa_shape,
)
from flag_dnn.graph.registry.schemas.pointwise import (
    _CMP_ALIAS_TO_OP,
    _activation_backward_shape,
    _bias_add_shape,
    _binary_select_shape,
    _binary_shape,
    _normalize_activation_backward,
    _normalize_bias_add,
    _normalize_binary,
    _normalize_binary_select,
    _normalize_cmp_alias,
    _normalize_pow,
    _normalize_scale,
    _normalize_unary,
)
from flag_dnn.graph.registry.schemas.norm_reduction import (
    _batchnorm_shape,
    _layernorm_shape,
    _normalize_batchnorm,
    _normalize_batchnorm_inference,
    _normalize_layernorm,
    _normalize_reduction,
    _normalize_rmsnorm,
    _normalize_rmsnorm_rht_amax,
    _reduction_shape,
    _rmsnorm_rht_amax_shape,
    _rmsnorm_shape,
)

_DEFAULTS_REGISTERED = False


def get_op_schema(name: str) -> OpSchema:
    register_default_ops()
    schema = get_registered_op(name)
    if schema is None:
        raise KeyError(f"FlagDNN graph op is not registered: {name}")
    return schema


def registered_ops() -> dict[str, OpSchema]:
    register_default_ops()
    return registered_raw_ops()


def _format_bias(
    input_tensor: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if bias.dim() == 1 and input_tensor.dim() >= 2:
        shape = [1] * input_tensor.dim()
        shape[1] = bias.numel()
        return bias.reshape(shape)
    return bias


def _runtime_backend_available(inputs: list[Any]) -> bool:
    tensor_inputs = [
        value for value in inputs if isinstance(value, torch.Tensor)
    ]
    return bool(tensor_inputs) and all(
        is_runtime_device_tensor(value) for value in tensor_inputs
    )


def _require_runtime_backend(inputs: list[Any], op_type: str) -> None:
    if not _runtime_backend_available(inputs):
        raise NotImplementedError(
            f"FlagDNN graph {op_type} requires runtime device tensors; "
            "torch fallback is disabled"
        )


def _unsupported_triton_path(op_type: str, detail: str) -> None:
    raise NotImplementedError(
        f"FlagDNN graph {op_type} has no Triton path for {detail}; "
        "torch fallback is disabled"
    )


def _run_bias_add(flag_ops: Any) -> RunFn:
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


def _run_binary(flag_ops: Any, op_type: str) -> RunFn:
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


def _run_binary_add(flag_ops: Any) -> RunFn:
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


def _run_binary_sub(flag_ops: Any) -> RunFn:
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


def _run_binary_mul(flag_ops: Any) -> RunFn:
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


def _run_binary_div(flag_ops: Any) -> RunFn:
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


def _run_binary_mod(flag_ops: Any) -> RunFn:
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


def _run_binary_pow(flag_ops: Any) -> RunFn:
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


def _run_binary_max(flag_ops: Any) -> RunFn:
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


def _run_binary_min(flag_ops: Any) -> RunFn:
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


def _run_binary_maximum(flag_ops: Any) -> RunFn:
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


def _run_binary_logical(flag_ops: Any, op_type: str) -> RunFn:
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


def _run_add_square(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "add_square")
        return flag_ops.add_square(
            inputs[0],
            inputs[1],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_binary_cmp(flag_ops: Any, op_type: str) -> RunFn:
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


def _run_binary_select(flag_ops: Any) -> RunFn:
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


def _run_relu(flag_ops: Any) -> RunFn:
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


def _run_swish(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "swish")
        return flag_ops.swish(
            inputs[0],
            swish_beta=attrs.get("swish_beta"),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_gelu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gelu")
        return flag_ops.gelu(
            inputs[0], approximate=attrs.get("approximate", "none")
        )

    return run


def _run_gelu_approx_tanh(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gelu_approx_tanh")
        return flag_ops.gelu(inputs[0], approximate="tanh")

    return run


def _run_leaky_relu(flag_ops: Any) -> RunFn:
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


def _run_elu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "elu")
        alpha = attrs.get("alpha", 1.0)
        if alpha is None:
            alpha = 1.0
        return flag_ops.elu(inputs[0], alpha=float(alpha), inplace=False)

    return run


def _run_sdpa_backward(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs[:6], "sdpa_backward")
        idx = 6
        bias = None
        if attrs.get("has_bias"):
            bias = inputs[idx]
            idx += 1
        dbias = None
        if attrs.get("has_dbias"):
            dbias = inputs[idx]
        return flag_ops.sdpa_backward(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            attn_scale=attrs.get("attn_scale"),
            bias=bias,
            dBias=dbias,
            diagonal_alignment=attrs.get("diagonal_alignment"),
            diagonal_band_left_bound=attrs.get("diagonal_band_left_bound"),
            diagonal_band_right_bound=attrs.get("diagonal_band_right_bound"),
            use_deterministic_algorithm=attrs.get(
                "use_deterministic_algorithm", False
            ),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_softplus(flag_ops: Any) -> RunFn:
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


def _run_unary_flag(flag_ops: Any, op_type: str) -> RunFn:
    flag_fn = getattr(flag_ops, op_type)

    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, op_type)
        return flag_fn(inputs[0])

    return run


def _run_logical_not(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "logical_not")
        return flag_ops.logical_not(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_abs(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "abs")
        return flag_ops.abs(inputs[0])

    return run


def _run_sigmoid(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "sigmoid")
        return flag_ops.sigmoid(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_sigmoid_backward(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "sigmoid_backward")
        return flag_ops.sigmoid_backward(
            inputs[0],
            inputs[1],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_identity(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "identity")
        return flag_ops.identity(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_reshape(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "reshape")
        return flag_ops.reshape(
            inputs[0],
            attrs["shape"],
            name=attrs.get("name", ""),
            reshape_mode=attrs.get("reshape_mode", "VIEW_ONLY"),
        )

    return run


def _run_transpose(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "transpose")
        return flag_ops.transpose(
            inputs[0],
            attrs["permutation"],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_slice(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "slice")
        return flag_ops.slice(
            inputs[0],
            attrs.get("slices", ()),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_concatenate(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "concatenate")
        return flag_ops.concatenate(
            inputs,
            attrs["axis"],
            in_place_index=attrs.get("in_place_index"),
            name=attrs.get("name", ""),
        )

    return run


def _run_gen_index(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gen_index")
        return flag_ops.gen_index(
            inputs[0],
            attrs["axis"],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_conv2d(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "conv2d")
        bias = inputs[2] if len(inputs) > 2 else None
        op_attrs = _public_attrs(attrs)
        return flag_ops.conv2d(
            inputs[0],
            inputs[1],
            bias=bias,
            stride=op_attrs.get("stride", 1),
            padding=op_attrs.get("padding", 0),
            dilation=op_attrs.get("dilation", 1),
            groups=op_attrs.get("groups", 1),
        )

    return run


def _run_conv_fprop(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "conv_fprop")
        op_attrs = _public_attrs(attrs)
        return flag_ops.conv_fprop(
            inputs[0],
            inputs[1],
            padding=op_attrs.get("padding"),
            pre_padding=op_attrs.get("pre_padding"),
            post_padding=op_attrs.get("post_padding"),
            stride=op_attrs.get("stride", 1),
            dilation=op_attrs.get("dilation", 1),
            convolution_mode=op_attrs.get(
                "convolution_mode", "CROSS_CORRELATION"
            ),
            compute_data_type=op_attrs.get("compute_data_type"),
            name=op_attrs.get("name", ""),
            groups=op_attrs.get("groups", 1),
        )

    return run


def _run_conv_dgrad(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "conv_dgrad")
        op_attrs = _public_attrs(attrs)
        return flag_ops.conv_dgrad(
            inputs[0],
            inputs[1],
            input_size=op_attrs["input_size"],
            padding=op_attrs.get("padding"),
            pre_padding=op_attrs.get("pre_padding"),
            post_padding=op_attrs.get("post_padding"),
            stride=op_attrs.get("stride", 1),
            dilation=op_attrs.get("dilation", 1),
            convolution_mode=op_attrs.get(
                "convolution_mode", "CROSS_CORRELATION"
            ),
            compute_data_type=op_attrs.get("compute_data_type"),
            name=op_attrs.get("name", ""),
            groups=op_attrs.get("groups", 1),
        )

    return run


def _run_conv_wgrad(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "conv_wgrad")
        op_attrs = _public_attrs(attrs)
        return flag_ops.conv_wgrad(
            inputs[0],
            inputs[1],
            filter_size=op_attrs["filter_size"],
            padding=op_attrs.get("padding"),
            pre_padding=op_attrs.get("pre_padding"),
            post_padding=op_attrs.get("post_padding"),
            stride=op_attrs.get("stride", 1),
            dilation=op_attrs.get("dilation", 1),
            convolution_mode=op_attrs.get(
                "convolution_mode", "CROSS_CORRELATION"
            ),
            compute_data_type=op_attrs.get("compute_data_type"),
            name=op_attrs.get("name", ""),
            groups=op_attrs.get("groups", 1),
        )

    return run


def _run_mm(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "mm")
        out_dtype = attrs.get("out_dtype")
        return flag_ops.mm(
            inputs[0],
            inputs[1],
            out_dtype=torch_dtype(out_dtype) if out_dtype else None,
        )

    return run


def _run_reduction(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "reduction")
        return flag_ops.reduction(
            inputs[0],
            attrs.get("mode"),
            dim=attrs.get("dim"),
            keepdim=bool(attrs.get("keepdim", True)),
            dtype=attrs.get("dtype"),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_batchnorm_inference(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "batchnorm_inference")
        return flag_ops.batchnorm_inference(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_batchnorm(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs[:5], "batchnorm")
        peer_count = int(attrs.get("peer_stats_count", 0))
        return flag_ops.batchnorm(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            peer_stats=inputs[7 : 7 + peer_count],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_layernorm(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs[:3], "layernorm")
        return flag_ops.layernorm(
            attrs.get("norm_forward_phase"),
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_rmsnorm(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        has_bias = bool(attrs.get("has_bias"))
        _require_runtime_backend(
            inputs[:3] if has_bias else inputs[:2], "rmsnorm"
        )
        bias = inputs[2] if has_bias else None
        epsilon = inputs[3] if has_bias else inputs[2]
        return flag_ops.rmsnorm(
            attrs.get("norm_forward_phase"),
            inputs[0],
            inputs[1],
            bias,
            epsilon,
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_rmsnorm_rht_amax(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs[:2], "rmsnorm_rht_amax_wrapper_sm100")
        result = flag_ops.rmsnorm_rht_amax_wrapper_sm100(
            inputs[0],
            inputs[1],
            eps=attrs.get("eps", 1e-5),
            num_threads=attrs.get("num_threads"),
            rows_per_cta=attrs.get("rows_per_cta"),
        )
        return result["o_tensor"], result["amax_tensor"]

    return run


def _run_causal_conv1d(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "causal_conv1d")
        bias = inputs[2] if attrs.get("has_bias") else None
        return flag_ops.causal_conv1d(
            inputs[0],
            inputs[1],
            bias=bias,
            activation=attrs.get("activation", "identity"),
        )

    return run


def _run_matmul(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "matmul")
        return flag_ops.matmul(
            inputs[0],
            inputs[1],
            compute_data_type=attrs.get("compute_data_type"),
            padding=float(attrs.get("padding", 0.0)),
            name=attrs.get("name", ""),
        )

    return run


def _run_sdpa(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs[:3], "sdpa")
        bias = inputs[3] if attrs.get("has_bias") else None
        return flag_ops.sdpa(
            inputs[0],
            inputs[1],
            inputs[2],
            attn_scale=attrs.get("attn_scale"),
            bias=bias,
            diagonal_alignment=attrs.get("diagonal_alignment"),
            diagonal_band_left_bound=attrs.get("diagonal_band_left_bound"),
            diagonal_band_right_bound=attrs.get("diagonal_band_right_bound"),
            generate_stats=attrs.get("generate_stats", False),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_fused_bias_relu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "fused_bias_relu")
        x, bias = inputs
        y = flag_ops.add(x, _format_bias(x, bias))
        return flag_ops.relu(y)

    return run


def _run_fused_bias_gelu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "fused_bias_gelu")
        x, bias = inputs
        y = flag_ops.add(x, _format_bias(x, bias))
        return flag_ops.gelu(y, approximate=attrs.get("approximate", "none"))

    return run


def _run_fused_conv2d_bias_relu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "fused_conv2d_bias_relu")
        implementation = attrs.get("_implementation", "triton_fused")
        if implementation != "triton_fused":
            _unsupported_triton_path(
                "fused_conv2d_bias_relu", f"implementation={implementation}"
            )
        op_attrs = _public_attrs(attrs)
        from flag_dnn.graph.kernels import fused_conv2d_bias_relu

        return fused_conv2d_bias_relu(
            inputs[0],
            inputs[1],
            inputs[2],
            stride=op_attrs.get("stride", 1),
            padding=op_attrs.get("padding", 0),
            dilation=op_attrs.get("dilation", 1),
            groups=op_attrs.get("groups", 1),
            config=attrs.get("_kernel_config"),
        )

    return run


def _public_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in attrs.items() if not key.startswith("_")
    }


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

_UTILITY_OP_SPECS = (
    ("reshape", _normalize_reshape, _reshape_shape, _run_reshape),
    ("transpose", _normalize_transpose, _transpose_shape, _run_transpose),
    ("slice", _normalize_slice, _slice_shape, _run_slice),
    (
        "concatenate",
        _normalize_concatenate,
        _concatenate_shape,
        _run_concatenate,
    ),
    ("gen_index", _normalize_gen_index, _gen_index_shape, _run_gen_index),
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


def _register_schema(
    name: str,
    normalize_fn: Any,
    shape_fn: Any,
    run_fn: RunFn,
    *,
    num_outputs: int | tuple[int, ...] | None = 1,
    fusible: bool = False,
) -> None:
    register_op(
        OpSchema(
            name=name,
            normalize_fn=normalize_fn,
            shape_fn=shape_fn,
            run_fn=run_fn,
            num_outputs=num_outputs,
            fusible=fusible,
        )
    )


def _register_utility_ops(flag_ops: Any) -> None:
    _register_schema(
        "identity",
        _normalize_unary("identity", ("compute_data_type", "name")),
        _shape_like_first,
        _run_identity(flag_ops),
    )
    for name, normalize_fn, shape_fn, run_factory in _UTILITY_OP_SPECS:
        _register_schema(name, normalize_fn, shape_fn, run_factory(flag_ops))


def _register_binary_ops(flag_ops: Any) -> None:
    for op_type in _BINARY_OP_TYPES:
        _register_schema(
            op_type,
            _normalize_binary(op_type),
            _binary_shape,
            _run_binary(flag_ops, op_type),
        )

    _register_schema(
        "binary_select",
        _normalize_binary_select,
        _binary_select_shape,
        _run_binary_select(flag_ops),
    )
    _register_schema(
        "scale",
        _normalize_scale,
        _binary_shape,
        _run_binary(flag_ops, "mul"),
        fusible=True,
    )
    _register_schema(
        "pow",
        _normalize_pow,
        _binary_shape,
        _run_binary(flag_ops, "pow"),
    )
    for alias_name, op_type in _CMP_ALIAS_TO_OP.items():
        _register_schema(
            alias_name,
            _normalize_cmp_alias(alias_name, op_type),
            _binary_shape,
            _run_binary(flag_ops, op_type),
        )
    _register_schema(
        "bias_add",
        _normalize_bias_add,
        _bias_add_shape,
        _run_bias_add(flag_ops),
        fusible=True,
    )


def _register_activation_ops(flag_ops: Any) -> None:
    for name, attrs, run_factory in _ACTIVATION_OP_SPECS:
        _register_schema(
            name,
            _normalize_unary(name, attrs),
            _shape_like_first,
            run_factory(flag_ops),
            fusible=True,
        )


def _register_simple_unary_ops(flag_ops: Any) -> None:
    for name in _UNARY_FLAG_OP_TYPES:
        _register_schema(
            name,
            _normalize_unary(name, ("compute_data_type", "name")),
            _shape_like_first,
            _run_unary_flag(flag_ops, name),
            fusible=True,
        )
    for name, run_factory in _POINTWISE_UNARY_OP_SPECS:
        _register_schema(
            name,
            _normalize_unary(name, ("compute_data_type", "name")),
            _shape_like_first,
            run_factory(flag_ops),
            fusible=True,
        )
    _register_schema(
        "sigmoid_backward",
        _normalize_activation_backward("sigmoid_backward"),
        _activation_backward_shape,
        _run_sigmoid_backward(flag_ops),
    )


def _register_conv_ops(flag_ops: Any) -> None:
    _register_schema(
        "conv2d",
        _normalize_conv2d,
        _conv2d_shape,
        _run_conv2d(flag_ops),
        fusible=True,
    )
    _register_schema(
        "conv_fprop",
        _normalize_conv_fprop,
        _conv_fprop_shape,
        _run_conv_fprop(flag_ops),
        fusible=True,
    )
    _register_schema(
        "conv_dgrad",
        _normalize_conv_dgrad,
        _conv_dgrad_shape,
        _run_conv_dgrad(flag_ops),
        fusible=True,
    )
    _register_schema(
        "conv_wgrad",
        _normalize_conv_wgrad,
        _conv_wgrad_shape,
        _run_conv_wgrad(flag_ops),
        fusible=True,
    )
    _register_schema(
        "causal_conv1d",
        _normalize_causal_conv1d,
        _shape_like_first,
        _run_causal_conv1d(flag_ops),
        fusible=True,
    )
    _register_schema(
        "fused_conv2d_bias_relu",
        _normalize_conv2d,
        _conv2d_shape,
        _run_fused_conv2d_bias_relu(flag_ops),
        fusible=True,
    )


def _register_matmul_attention_ops(flag_ops: Any) -> None:
    _register_schema(
        "mm",
        _normalize_mm,
        _mm_shape,
        _run_mm(flag_ops),
        fusible=True,
    )
    _register_schema(
        "matmul",
        _normalize_matmul,
        _matmul_shape,
        _run_matmul(flag_ops),
        fusible=True,
    )
    _register_schema(
        "sdpa",
        _normalize_sdpa,
        _sdpa_shape,
        _run_sdpa(flag_ops),
        num_outputs=(1, 2),
        fusible=True,
    )
    _register_schema(
        "sdpa_backward",
        _normalize_sdpa_backward,
        _sdpa_backward_shape,
        _run_sdpa_backward(flag_ops),
        num_outputs=3,
    )


def _register_norm_reduction_ops(flag_ops: Any) -> None:
    _register_schema(
        "reduction",
        _normalize_reduction,
        _reduction_shape,
        _run_reduction(flag_ops),
        fusible=True,
    )
    _register_schema(
        "batchnorm_inference",
        _normalize_batchnorm_inference,
        _shape_like_first,
        _run_batchnorm_inference(flag_ops),
        fusible=True,
    )
    _register_schema(
        "batchnorm",
        _normalize_batchnorm,
        _batchnorm_shape,
        _run_batchnorm(flag_ops),
        num_outputs=5,
        fusible=True,
    )
    _register_schema(
        "layernorm",
        _normalize_layernorm,
        _layernorm_shape,
        _run_layernorm(flag_ops),
        num_outputs=3,
        fusible=True,
    )
    _register_schema(
        "rmsnorm",
        _normalize_rmsnorm,
        _rmsnorm_shape,
        _run_rmsnorm(flag_ops),
        num_outputs=2,
        fusible=True,
    )
    _register_schema(
        "rmsnorm_rht_amax_wrapper_sm100",
        _normalize_rmsnorm_rht_amax,
        _rmsnorm_rht_amax_shape,
        _run_rmsnorm_rht_amax(flag_ops),
        num_outputs=2,
        fusible=True,
    )


def _register_fused_ops(flag_ops: Any) -> None:
    _register_schema(
        "fused_bias_relu",
        _normalize_bias_add,
        _bias_add_shape,
        _run_fused_bias_relu(flag_ops),
        fusible=True,
    )
    _register_schema(
        "fused_bias_gelu",
        _normalize_bias_add,
        _bias_add_shape,
        _run_fused_bias_gelu(flag_ops),
        fusible=True,
    )


def register_default_ops() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    import flag_dnn.ops as flag_ops

    _register_utility_ops(flag_ops)
    _register_binary_ops(flag_ops)
    _register_activation_ops(flag_ops)
    _register_conv_ops(flag_ops)
    _register_matmul_attention_ops(flag_ops)
    _register_norm_reduction_ops(flag_ops)
    _register_simple_unary_ops(flag_ops)
    _register_fused_ops(flag_ops)

    _DEFAULTS_REGISTERED = True
