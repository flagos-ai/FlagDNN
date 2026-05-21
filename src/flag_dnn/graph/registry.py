from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.graph.tensor import (
    TensorSpec,
    canonical_dtype,
    torch_dtype,
)

NormalizeFn = Callable[
    [Any, tuple[Any, ...], dict[str, Any]], tuple[list[int], dict]
]
ShapeFn = Callable[[list[TensorSpec], dict[str, Any]], list[TensorSpec]]
RunFn = Callable[[list[Any], dict[str, Any]], Any]


@dataclass
class OpSchema:
    name: str
    normalize_fn: NormalizeFn
    shape_fn: ShapeFn
    run_fn: RunFn
    num_outputs: int = 1
    attrs_schema: dict[str, Any] = field(default_factory=dict)
    fusible: bool = False

    def normalize(
        self, ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[int], dict]:
        return self.normalize_fn(ctx, args, kwargs)

    def infer_outputs(
        self, input_specs: list[TensorSpec], attrs: dict[str, Any]
    ) -> list[TensorSpec]:
        return self.shape_fn(input_specs, attrs)

    def run(self, inputs: list[Any], attrs: dict[str, Any]) -> Any:
        return self.run_fn(inputs, attrs)


_REGISTRY: dict[str, OpSchema] = {}
_DEFAULTS_REGISTERED = False


def register_op(schema: OpSchema) -> OpSchema:
    _REGISTRY[schema.name] = schema
    return schema


def get_op_schema(name: str) -> OpSchema:
    register_default_ops()
    if name not in _REGISTRY:
        raise KeyError(f"FlagDNN graph op is not registered: {name}")
    return _REGISTRY[name]


def registered_ops() -> dict[str, OpSchema]:
    register_default_ops()
    return dict(_REGISTRY)


def _shape_like_first(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=inp.dtype,
            stride=None,
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        )
    ]


def _normalize_axis(axis: Any, rank: int, op_type: str = "op") -> int:
    axis = int(axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise IndexError(
            f"graph {op_type} axis out of range for rank {rank}: {axis}"
        )
    return axis


def _rank_of(value: Any) -> int:
    if hasattr(value, "dim"):
        return int(value.dim())
    if hasattr(value, "shape"):
        return len(tuple(value.shape))
    raise TypeError(f"cannot infer rank from {type(value)!r}")


def _normalize_shape_arg(shape: Any) -> tuple[int, ...]:
    if shape is None:
        raise TypeError("reshape missing required argument: shape")
    if isinstance(shape, torch.Size):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _normalize_permutation_arg(permutation: Any, rank: int) -> tuple[int, ...]:
    permutation = tuple(int(dim) for dim in permutation)
    if len(permutation) != rank:
        raise RuntimeError(
            f"graph transpose permutation length {len(permutation)} does not "
            f"match input rank {rank}"
        )
    normalized = tuple(
        _normalize_axis(dim, rank, "transpose") for dim in permutation
    )
    if len(set(normalized)) != rank:
        raise RuntimeError(
            f"graph transpose permutation must be unique, got {permutation}"
        )
    return normalized


def _swap_permutation(rank: int, dim0: Any, dim1: Any) -> tuple[int, ...]:
    dim0 = _normalize_axis(dim0, rank, "transpose")
    dim1 = _normalize_axis(dim1, rank, "transpose")
    permutation = list(range(rank))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return tuple(permutation)


def _numel(shape: tuple[Any, ...]) -> Optional[int]:
    if any(not isinstance(dim, int) for dim in shape):
        return None
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _infer_reshape_shape(
    input_shape: tuple[Any, ...], requested_shape: tuple[int, ...]
) -> tuple[int, ...]:
    unknown_dims = [
        idx for idx, dim in enumerate(requested_shape) if dim == -1
    ]
    if len(unknown_dims) > 1:
        raise RuntimeError("graph reshape can only infer one dimension")
    for dim in requested_shape:
        if dim < -1:
            raise RuntimeError(f"graph reshape invalid dimension {dim}")

    input_numel = _numel(input_shape)
    known_product = 1
    for dim in requested_shape:
        if dim != -1:
            known_product *= dim

    if unknown_dims:
        if input_numel is None:
            raise NotImplementedError(
                "graph reshape cannot infer symbolic -1 dimensions"
            )
        if known_product == 0:
            raise RuntimeError(
                "graph reshape cannot infer -1 with zero known product"
            )
        if input_numel % known_product != 0:
            raise RuntimeError(
                f"graph reshape shape {requested_shape} is invalid for input "
                f"shape {input_shape}"
            )
        result = list(requested_shape)
        result[unknown_dims[0]] = input_numel // known_product
        return tuple(result)

    output_numel = _numel(requested_shape)
    if input_numel is not None and output_numel != input_numel:
        raise RuntimeError(
            f"graph reshape shape {requested_shape} is invalid for input "
            f"shape {input_shape}"
        )
    return requested_shape


def _reshape_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    out_shape = _infer_reshape_shape(inp.shape, tuple(attrs["shape"]))
    return [
        TensorSpec(
            name="",
            shape=out_shape,
            dtype=inp.dtype,
            layout="contiguous" if inp.contiguous else inp.layout,
            device=inp.device,
            contiguous=True if inp.contiguous else None,
        )
    ]


def _transpose_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    permutation = tuple(attrs["permutation"])
    identity = permutation == tuple(range(len(inp.shape)))
    return [
        TensorSpec(
            name="",
            shape=tuple(inp.shape[dim] for dim in permutation),
            dtype=inp.dtype,
            layout=inp.layout if identity else "strided",
            device=inp.device,
            contiguous=inp.contiguous if identity else False,
        )
    ]


def _normalize_slice_specs(
    slices: Any, rank: int
) -> tuple[tuple[Any, Any, Any], ...]:
    import builtins

    if slices is None:
        raw = ()
    elif isinstance(slices, builtins.slice):
        raw = (slices,)
    else:
        raw = tuple(slices)
    if len(raw) > rank:
        raise IndexError(
            f"graph slice got {len(raw)} slice specs for rank {rank}"
        )

    specs: list[tuple[Any, Any, Any]] = []
    for item in raw:
        if isinstance(item, builtins.slice):
            specs.append((item.start, item.stop, item.step))
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            specs.append((item[0], item[1], item[2]))
        elif item is None:
            specs.append((None, None, None))
        else:
            raise TypeError(
                f"graph slice expects slice specs, got {type(item)!r}"
            )
    while len(specs) < rank:
        specs.append((None, None, None))
    for _, _, step in specs:
        step_value = 1 if step is None else int(step)
        if step_value <= 0:
            raise ValueError("graph slice step must be greater than zero")
    return tuple(specs)


def _resolve_slice_dim(dim: Any, spec: tuple[Any, Any, Any]) -> Any:
    start, stop, step = spec
    step_value = 1 if step is None else int(step)
    if not isinstance(dim, int):
        if start is None and stop is None and step_value == 1:
            return dim
        raise NotImplementedError(
            "graph slice symbolic dimensions only support full slices"
        )
    import builtins

    normalized = builtins.slice(start, stop, step).indices(dim)
    return len(range(*normalized))


def _slice_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    specs = tuple(attrs["slices"])
    return [
        TensorSpec(
            name="",
            shape=tuple(
                _resolve_slice_dim(dim, spec)
                for dim, spec in zip(inp.shape, specs)
            ),
            dtype=inp.dtype,
            layout="strided",
            device=inp.device,
            contiguous=False,
        )
    ]


def _concatenate_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    if not input_specs:
        raise RuntimeError("graph concatenate expects at least one input")
    first = input_specs[0]
    rank = len(first.shape)
    axis = _normalize_axis(attrs["axis"], rank, "concatenate")
    out_shape = list(first.shape)
    axis_size = first.shape[axis]

    for spec in input_specs[1:]:
        if len(spec.shape) != rank:
            raise RuntimeError("graph concatenate inputs must have same rank")
        if spec.dtype != first.dtype:
            raise RuntimeError("graph concatenate inputs must have same dtype")
        for dim_index, (expected, actual) in enumerate(
            zip(first.shape, spec.shape)
        ):
            if dim_index == axis:
                continue
            if expected != actual:
                raise RuntimeError(
                    "graph concatenate non-axis dimensions must match: "
                    f"{first.shape} vs {spec.shape}"
                )
        if not isinstance(axis_size, int) or not isinstance(
            spec.shape[axis], int
        ):
            raise NotImplementedError(
                "graph concatenate symbolic axis dimensions are not enabled"
            )
        axis_size += spec.shape[axis]

    out_shape[axis] = axis_size
    return [
        TensorSpec(
            name="",
            shape=tuple(out_shape),
            dtype=first.dtype,
            layout="contiguous",
            device=first.device,
            contiguous=True,
        )
    ]


def _compute_dtype_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return canonical_dtype(value)
    key = str(value).lower()
    if key in ("none", "not_set", "data_type.not_set"):
        return default
    aliases = {
        "boolean": "bool",
        "data_type.boolean": "bool",
        "data_type.bfloat16": "bfloat16",
        "data_type.double": "float64",
        "data_type.float": "float32",
        "data_type.float16": "float16",
        "data_type.int32": "int32",
        "data_type.int64": "int64",
        "double": "float64",
        "float": "float32",
    }
    key = aliases.get(key, key)
    tail = key.rsplit(".", 1)[-1]
    tail = aliases.get(tail, tail)
    return canonical_dtype(tail)


def _gen_index_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=_compute_dtype_or_default(
                attrs.get("compute_data_type"), "int32"
            ),
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        )
    ]


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise RuntimeError(f"expected length 2, got {value}")
    return int(value[0]), int(value[1])


def _conv_out_dim(
    input_size: int,
    pad_before: int,
    pad_after: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return (
        input_size + pad_before + pad_after - dilation * (kernel - 1) - 1
    ) // stride + 1


def _tuple_n(value: Any, rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _normalize_convolution_mode(convolution_mode: Any) -> None:
    if convolution_mode is None:
        return
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    if mode == "CROSS_CORRELATION":
        return
    if mode == "CONVOLUTION":
        raise NotImplementedError(
            "graph conv_fprop currently supports CROSS_CORRELATION only"
        )
    raise RuntimeError(
        "convolution_mode must be CROSS_CORRELATION or CONVOLUTION"
    )


def _conv_spatial_rank_from_values(
    image: Any, weight: Any, op_type: str
) -> int:
    image_rank = _rank_of(image)
    weight_rank = _rank_of(weight)
    if image_rank == 2 and weight_rank == 3:
        return 1
    if image_rank >= 3 and image_rank == weight_rank:
        return image_rank - 2
    raise RuntimeError(
        f"graph {op_type} expects matching 1D/2D/3D convolution shapes, "
        f"got image rank {image_rank} and weight rank {weight_rank}"
    )


def _normalize_conv_padding_from_attrs(
    weight: TensorSpec,
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    attrs: dict[str, Any],
    op_type: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rank = len(stride)
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    padding = attrs.get("padding")

    if pre_padding is not None or post_padding is not None:
        if padding is not None:
            raise TypeError(
                f"graph {op_type} accepts either padding or "
                "pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                f"graph {op_type} requires both pre_padding and post_padding"
            )
        return (
            _tuple_n(pre_padding, rank, "pre_padding"),
            _tuple_n(post_padding, rank, "post_padding"),
        )

    if padding is None:
        padding = 0
    if isinstance(padding, str):
        if padding == "valid":
            zeros = (0,) * rank
            return zeros, zeros
        if padding == "same":
            if any(dim != 1 for dim in stride):
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            pre = []
            post = []
            for axis in range(rank):
                kernel = int(weight.shape[2 + axis])
                effective_kernel = dilation[axis] * (kernel - 1) + 1
                total_pad = max(effective_kernel - 1, 0)
                before = total_pad // 2
                pre.append(before)
                post.append(total_pad - before)
            return tuple(pre), tuple(post)
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")

    if isinstance(padding, int):
        values = (int(padding),) * rank
        return values, values

    values = tuple(int(v) for v in padding)
    if len(values) == rank:
        return values, values
    if len(values) == 2 * rank:
        pre = tuple(values[2 * axis] for axis in range(rank))
        post = tuple(values[2 * axis + 1] for axis in range(rank))
        return pre, post
    raise RuntimeError(
        f"padding must have length {rank} or {2 * rank}, got {padding}"
    )



def _normalize_padding_from_spec(
    weight: TensorSpec,
    stride: tuple[int, int],
    padding: Any,
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    pre, post = _normalize_conv_padding_from_attrs(
        weight,
        stride,
        dilation,
        {"padding": padding},
        "conv2d",
    )
    if len(pre) != 2:
        raise RuntimeError("graph conv2d expects 2D padding")
    return pre[0], post[0], pre[1], post[1]


def _conv2d_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    x, weight = input_specs[0], input_specs[1]
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise RuntimeError("graph conv2d expects 4D input and weight")

    stride = _pair(attrs.get("stride", 1))
    dilation = _pair(attrs.get("dilation", 1))
    padding_2d = _normalize_padding_from_spec(
        weight, stride, attrs.get("padding", 0), dilation
    )
    groups = int(attrs.get("groups", 1))

    n, c_in, h, w = x.shape
    c_out, c_per_group, kh, kw = weight.shape
    if not all(isinstance(v, int) for v in (n, c_in, h, w, c_out, kh, kw)):
        raise NotImplementedError(
            "graph conv2d dynamic shapes are not enabled"
        )
    if int(c_in) % groups != 0:
        raise RuntimeError("graph conv2d input channels must divide groups")
    if int(c_per_group) != int(c_in) // groups:
        raise RuntimeError("graph conv2d weight channel mismatch")

    pad_top, pad_bottom, pad_left, pad_right = padding_2d
    oh = _conv_out_dim(
        int(h), pad_top, pad_bottom, dilation[0], int(kh), stride[0]
    )
    ow = _conv_out_dim(
        int(w), pad_left, pad_right, dilation[1], int(kw), stride[1]
    )
    if oh < 0 or ow < 0:
        raise RuntimeError("computed graph conv2d output size is negative")

    return [
        TensorSpec(
            name="",
            shape=(n, c_out, max(oh, 0), max(ow, 0)),
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
    ]


def _conv_fprop_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    x, weight = input_specs[0], input_specs[1]
    rank = _conv_spatial_rank_from_values(x, weight, "conv_fprop")

    stride = _tuple_n(attrs.get("stride", 1), rank, "stride")
    dilation = _tuple_n(attrs.get("dilation", 1), rank, "dilation")
    _normalize_convolution_mode(attrs.get("convolution_mode"))
    pre_padding, post_padding = _normalize_conv_padding_from_attrs(
        weight, stride, dilation, attrs, "conv_fprop"
    )
    groups = int(attrs.get("groups", 1))

    if rank == 1 and len(x.shape) == 2:
        leading_shape: tuple[Any, ...] = ()
        c_in = x.shape[0]
        spatial_shape = x.shape[1:]
    else:
        leading_shape = (x.shape[0],)
        c_in = x.shape[1]
        spatial_shape = x.shape[2:]

    c_out = weight.shape[0]
    c_per_group = weight.shape[1]
    kernel_shape = weight.shape[2:]
    static_values = (c_in, c_out, c_per_group) + spatial_shape + kernel_shape
    if not all(isinstance(v, int) for v in static_values):
        raise NotImplementedError(
            "graph conv_fprop dynamic shapes are not enabled"
        )
    if int(c_in) % groups != 0:
        raise RuntimeError("graph conv_fprop input channels must divide groups")
    if int(c_per_group) != int(c_in) // groups:
        raise RuntimeError("graph conv_fprop weight channel mismatch")

    out_spatial = []
    for axis in range(rank):
        out_dim = _conv_out_dim(
            int(spatial_shape[axis]),
            pre_padding[axis],
            post_padding[axis],
            dilation[axis],
            int(kernel_shape[axis]),
            stride[axis],
        )
        if out_dim < 0:
            raise RuntimeError(
                "computed graph conv_fprop output size is negative"
            )
        out_spatial.append(max(out_dim, 0))

    return [
        TensorSpec(
            name="",
            shape=leading_shape + (c_out,) + tuple(out_spatial),
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
    ]


def _mm_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    a, b = input_specs[0], input_specs[1]
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise RuntimeError("graph mm expects 2D inputs")
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(
            f"graph mm shape mismatch: {a.shape} cannot multiply {b.shape}"
        )
    out_dtype = attrs.get("out_dtype")
    return [
        TensorSpec(
            name="",
            shape=(a.shape[0], b.shape[1]),
            dtype=canonical_dtype(out_dtype) if out_dtype else a.dtype,
            layout="contiguous",
            device=a.device,
        )
    ]


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
    if op_type in ("eq", "ne", "lt", "le", "ge", "gt"):
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


def _pop_operand(params: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in params:
            return params.pop(name)
    return None


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
            args[0]
            if args
            else _pop_operand(params, ("input", "a", "input0"))
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


def _normalize_pow(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("pow expects at most two positional args")
    params = dict(kwargs)
    if "out" in params and params["out"] is not None:
        raise NotImplementedError("FlagDNN graph does not support out tensors")
    params.pop("out", None)
    left = (
        args[0]
        if args
        else _pop_operand(params, ("input", "input0", "a"))
    )
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
            args[0]
            if args
            else _pop_operand(params, ("input", "a", "input0"))
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


def _normalize_conv2d(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 7:
        raise TypeError("conv2d got too many positional args")
    names = [
        "input",
        "weight",
        "bias",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]
    defaults = {
        "bias": None,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
    }
    params = defaults.copy()
    params.update(kwargs)
    for idx, arg in enumerate(args):
        name = names[idx]
        if name in kwargs:
            raise TypeError(f"conv2d got multiple values for {name}")
        params[name] = arg
    input_ids = [
        ctx.as_value(params.pop("input"), "input"),
        ctx.as_value(params.pop("weight"), "weight"),
    ]
    bias = params.pop("bias")
    if bias is not None:
        input_ids.append(ctx.as_value(bias, "bias"))
    return input_ids, params


def _normalize_conv_fprop(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 3:
        raise TypeError("conv_fprop got too many positional args")
    names = ["image", "weight", "padding"]
    defaults = {
        "padding": None,
        "pre_padding": None,
        "post_padding": None,
        "stride": 1,
        "dilation": 1,
        "convolution_mode": "CROSS_CORRELATION",
        "compute_data_type": None,
        "name": "",
        "groups": 1,
    }
    params = defaults.copy()
    params.update(kwargs)
    for idx, arg in enumerate(args):
        name = names[idx]
        if name in kwargs:
            raise TypeError(f"conv_fprop got multiple values for {name}")
        params[name] = arg

    image = params.pop("image", None)
    weight = params.pop("weight", None)
    if image is None or weight is None:
        raise TypeError("conv_fprop missing image or weight")

    rank = _conv_spatial_rank_from_values(image, weight, "conv_fprop")
    params["stride"] = _tuple_n(params["stride"], rank, "stride")
    params["dilation"] = _tuple_n(params["dilation"], rank, "dilation")
    _normalize_convolution_mode(params.get("convolution_mode"))

    padding = params.get("padding")
    pre_padding = params.get("pre_padding")
    post_padding = params.get("post_padding")
    if pre_padding is not None or post_padding is not None:
        if padding is not None:
            raise TypeError(
                "conv_fprop accepts either padding or pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                "conv_fprop requires both pre_padding and post_padding"
            )
        params["pre_padding"] = _tuple_n(pre_padding, rank, "pre_padding")
        params["post_padding"] = _tuple_n(post_padding, rank, "post_padding")
    elif padding is not None and not isinstance(padding, str):
        params["padding"] = _tuple_n(padding, rank, "padding")

    return [
        ctx.as_value(image, "image"),
        ctx.as_value(weight, "weight"),
    ], params


def _normalize_mm(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 3:
        raise TypeError("mm got too many positional args")
    params = {"out_dtype": None, "out": None}
    params.update(kwargs)
    if args:
        params["input"] = args[0]
    if len(args) > 1:
        params["mat2"] = args[1]
    if len(args) > 2:
        params["out_dtype"] = args[2]
    if params.get("out") is not None:
        raise NotImplementedError("FlagDNN graph does not support mm out")
    out_dtype = params.get("out_dtype")
    attrs = {"out_dtype": canonical_dtype(out_dtype) if out_dtype else None}
    return [
        ctx.as_value(params["input"], "input"),
        ctx.as_value(params["mat2"], "mat2"),
    ], attrs


def _normalize_reshape(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("reshape expects input and shape")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError("FlagDNN graph does not support reshape out")
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("reshape missing input tensor")
    if len(args) > 1:
        shape = args[1]
    else:
        shape = params.pop("shape", params.pop("size", None))
    attrs = {
        "shape": _normalize_shape_arg(shape),
        "name": params.pop("name", ""),
        "reshape_mode": params.pop("reshape_mode", "VIEW_ONLY"),
    }
    if params:
        raise TypeError(f"reshape got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _normalize_transpose(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 3:
        raise TypeError("transpose got too many positional args")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError(
            "FlagDNN graph does not support transpose out"
        )
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("transpose missing input tensor")
    rank = _rank_of(x)

    if len(args) == 3:
        permutation = _swap_permutation(rank, args[1], args[2])
    elif "dim0" in params or "dim1" in params:
        if len(args) > 1:
            dim0 = args[1]
        else:
            dim0 = params.pop("dim0")
        dim1 = params.pop("dim1")
        permutation = _swap_permutation(rank, dim0, dim1)
    else:
        if len(args) > 1:
            raw_permutation = args[1]
        else:
            raw_permutation = params.pop("permutation", None)
        if raw_permutation is None:
            raise TypeError("transpose missing permutation")
        permutation = _normalize_permutation_arg(raw_permutation, rank)

    attrs = {
        "permutation": permutation,
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"transpose got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _normalize_slice(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError("FlagDNN graph does not support slice out")
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("slice missing input tensor")
    if len(args) > 2:
        raw_slices = args[1:]
    elif len(args) == 2:
        raw_slices = args[1]
    else:
        raw_slices = params.pop("slices", ())
    attrs = {
        "slices": _normalize_slice_specs(raw_slices, _rank_of(x)),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"slice got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _normalize_concatenate(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("concatenate expects inputs and axis")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError(
            "FlagDNN graph does not support concatenate out"
        )
    params.pop("out", None)
    inputs = args[0] if args else params.pop("inputs", None)
    if inputs is None:
        raise TypeError("concatenate missing inputs")
    inputs = tuple(inputs)
    if not inputs:
        raise RuntimeError("concatenate expects a non-empty input sequence")
    if len(args) > 1:
        axis = args[1]
    else:
        axis = params.pop("axis", None)
    if axis is None:
        raise TypeError("concatenate missing axis")
    attrs = {
        "axis": int(axis),
        "in_place_index": params.pop("in_place_index", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"concatenate got unsupported attrs: {sorted(params)}")
    return [
        ctx.as_value(value, f"input{index}")
        for index, value in enumerate(inputs)
    ], attrs


def _normalize_gen_index(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("gen_index expects input and axis")
    params = dict(kwargs)
    if params.get("out") is not None:
        raise NotImplementedError(
            "FlagDNN graph does not support gen_index out"
        )
    params.pop("out", None)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("gen_index missing input tensor")
    axis = args[1] if len(args) > 1 else params.pop("axis", None)
    if axis is None:
        raise TypeError("gen_index missing axis")
    attrs = {
        "axis": _normalize_axis(axis, _rank_of(x), "gen_index"),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"gen_index got unsupported attrs: {sorted(params)}")
    return [ctx.as_value(x, "input")], attrs


def _format_bias(
    input_tensor: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if bias.dim() == 1 and input_tensor.dim() >= 2:
        shape = [1] * input_tensor.dim()
        shape[1] = bias.numel()
        return bias.reshape(shape)
    return bias


def _runtime_backend_available(inputs: list[Any]) -> bool:
    tensor_inputs = [value for value in inputs if isinstance(value, torch.Tensor)]
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
    if op_type == "pow":
        return _run_binary_pow(flag_ops)
    if op_type == "max":
        return _run_binary_max(flag_ops)
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


def _run_relu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "relu")
        return flag_ops.relu(inputs[0], inplace=attrs.get("inplace", False))

    return run


def _run_gelu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "gelu")
        return flag_ops.gelu(
            inputs[0], approximate=attrs.get("approximate", "none")
        )

    return run


def _run_unary_flag(flag_ops: Any, op_type: str) -> RunFn:
    flag_fn = getattr(flag_ops, op_type)

    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, op_type)
        return flag_fn(inputs[0])

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
        return flag_ops.gelu(
            y, approximate=attrs.get("approximate", "none")
        )

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


def register_default_ops() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    import flag_dnn.ops as flag_ops

    register_op(
        OpSchema(
            name="identity",
            normalize_fn=_normalize_unary(
                "identity", ("compute_data_type", "name")
            ),
            shape_fn=_shape_like_first,
            run_fn=_run_identity(flag_ops),
        )
    )
    register_op(
        OpSchema(
            name="reshape",
            normalize_fn=_normalize_reshape,
            shape_fn=_reshape_shape,
            run_fn=_run_reshape(flag_ops),
        )
    )
    register_op(
        OpSchema(
            name="transpose",
            normalize_fn=_normalize_transpose,
            shape_fn=_transpose_shape,
            run_fn=_run_transpose(flag_ops),
        )
    )
    register_op(
        OpSchema(
            name="slice",
            normalize_fn=_normalize_slice,
            shape_fn=_slice_shape,
            run_fn=_run_slice(flag_ops),
        )
    )
    register_op(
        OpSchema(
            name="concatenate",
            normalize_fn=_normalize_concatenate,
            shape_fn=_concatenate_shape,
            run_fn=_run_concatenate(flag_ops),
        )
    )
    register_op(
        OpSchema(
            name="gen_index",
            normalize_fn=_normalize_gen_index,
            shape_fn=_gen_index_shape,
            run_fn=_run_gen_index(flag_ops),
        )
    )

    for op_type in (
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
    ):
        register_op(
            OpSchema(
                name=op_type,
                normalize_fn=_normalize_binary(op_type),
                shape_fn=_binary_shape,
                run_fn=_run_binary(flag_ops, op_type),
            )
        )

    register_op(
        OpSchema(
            name="pow",
            normalize_fn=_normalize_pow,
            shape_fn=_binary_shape,
            run_fn=_run_binary(flag_ops, "pow"),
        )
    )

    for alias_name, op_type in _CMP_ALIAS_TO_OP.items():
        register_op(
            OpSchema(
                name=alias_name,
                normalize_fn=_normalize_cmp_alias(alias_name, op_type),
                shape_fn=_binary_shape,
                run_fn=_run_binary(flag_ops, op_type),
            )
        )

    register_op(
        OpSchema(
            name="bias_add",
            normalize_fn=_normalize_bias_add,
            shape_fn=_bias_add_shape,
            run_fn=_run_bias_add(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="relu",
            normalize_fn=_normalize_unary("relu", ("inplace",)),
            shape_fn=_shape_like_first,
            run_fn=_run_relu(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="gelu",
            normalize_fn=_normalize_unary("gelu", ("approximate",)),
            shape_fn=_shape_like_first,
            run_fn=_run_gelu(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="conv2d",
            normalize_fn=_normalize_conv2d,
            shape_fn=_conv2d_shape,
            run_fn=_run_conv2d(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="conv_fprop",
            normalize_fn=_normalize_conv_fprop,
            shape_fn=_conv_fprop_shape,
            run_fn=_run_conv_fprop(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="fused_conv2d_bias_relu",
            normalize_fn=_normalize_conv2d,
            shape_fn=_conv2d_shape,
            run_fn=_run_fused_conv2d_bias_relu(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="mm",
            normalize_fn=_normalize_mm,
            shape_fn=_mm_shape,
            run_fn=_run_mm(flag_ops),
            fusible=True,
        )
    )

    for name in ("sqrt", "neg", "tanh", "silu"):
        register_op(
            OpSchema(
                name=name,
                normalize_fn=_normalize_unary(name),
                shape_fn=_shape_like_first,
                run_fn=_run_unary_flag(flag_ops, name),
                fusible=True,
            )
        )

    register_op(
        OpSchema(
            name="sigmoid",
            normalize_fn=_normalize_unary(
                "sigmoid", ("compute_data_type", "name")
            ),
            shape_fn=_shape_like_first,
            run_fn=_run_sigmoid(flag_ops),
            fusible=True,
        )
    )

    register_op(
        OpSchema(
            name="sigmoid_backward",
            normalize_fn=_normalize_activation_backward("sigmoid_backward"),
            shape_fn=_activation_backward_shape,
            run_fn=_run_sigmoid_backward(flag_ops),
        )
    )

    register_op(
        OpSchema(
            name="abs",
            normalize_fn=_normalize_unary(
                "abs", ("compute_data_type", "name")
            ),
            shape_fn=_shape_like_first,
            run_fn=_run_abs(flag_ops),
            fusible=True,
        )
    )

    register_op(
        OpSchema(
            name="fused_bias_relu",
            normalize_fn=_normalize_bias_add,
            shape_fn=_bias_add_shape,
            run_fn=_run_fused_bias_relu(flag_ops),
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="fused_bias_gelu",
            normalize_fn=_normalize_bias_add,
            shape_fn=_bias_add_shape,
            run_fn=_run_fused_bias_gelu(flag_ops),
            fusible=True,
        )
    )

    _DEFAULTS_REGISTERED = True
