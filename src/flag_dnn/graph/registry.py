from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

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


def _normalize_padding_from_spec(
    weight: TensorSpec,
    stride: tuple[int, int],
    padding: Any,
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0, 0, 0
        if padding == "same":
            if stride != (1, 1):
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            kh, kw = int(weight.shape[2]), int(weight.shape[3])
            eff_kh = dilation[0] * (kh - 1) + 1
            eff_kw = dilation[1] * (kw - 1) + 1
            pad_h, pad_w = max(eff_kh - 1, 0), max(eff_kw - 1, 0)
            pad_top, pad_left = pad_h // 2, pad_w // 2
            return pad_top, pad_h - pad_top, pad_left, pad_w - pad_left
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")
    pad_h, pad_w = _pair(padding)
    return pad_h, pad_h, pad_w, pad_w


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
        left = args[0] if args else params.pop("input", None)
        if left is None:
            raise TypeError(f"{op_type} missing input tensor")
        if len(args) >= 2:
            right = args[1]
        else:
            right = params.pop("other", None)
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


def _run_bias_add(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
    x, bias = inputs
    return x + _format_bias(x, bias)


def _run_binary(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        op_type = attrs["op_type"]
        left = inputs[0]
        if len(inputs) > 1:
            right = inputs[1]
        else:
            right = attrs["other"]
        if attrs.get("reverse"):
            left, right = right, left
        alpha = attrs.get("alpha", 1)
        rounding_mode = attrs.get("rounding_mode")
        if op_type == "add":
            if _cuda_available(inputs):
                if not isinstance(left, torch.Tensor) and isinstance(
                    right, torch.Tensor
                ):
                    return flag_ops.add(
                        right,
                        left,
                        alpha=alpha,
                        compute_data_type=attrs.get("compute_data_type"),
                        name=attrs.get("name", ""),
                    )
                return flag_ops.add(
                    left,
                    right,
                    alpha=alpha,
                    compute_data_type=attrs.get("compute_data_type"),
                    name=attrs.get("name", ""),
                )
            return torch.add(left, right, alpha=alpha)
        if op_type == "sub":
            return torch.sub(left, right, alpha=alpha)
        if op_type == "mul":
            return torch.mul(left, right)
        if op_type == "div":
            return torch.div(left, right, rounding_mode=rounding_mode)
        if op_type == "eq":
            return torch.eq(left, right)
        if op_type == "ne":
            return torch.ne(left, right)
        if op_type == "lt":
            return torch.lt(left, right)
        if op_type == "le":
            return torch.le(left, right)
        if op_type == "gt":
            return torch.gt(left, right)
        if op_type == "ge":
            return torch.ge(left, right)
        raise RuntimeError(f"unsupported graph binary op: {op_type}")

    return run


def _cuda_available(inputs: list[Any]) -> bool:
    return any(
        isinstance(value, torch.Tensor) and value.is_cuda for value in inputs
    )


def _run_relu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        x = inputs[0]
        if _cuda_available(inputs):
            return flag_ops.relu(x, inplace=attrs.get("inplace", False))
        return torch.nn.functional.relu(x, inplace=attrs.get("inplace", False))

    return run


def _run_gelu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        x = inputs[0]
        approximate = attrs.get("approximate", "none")
        if _cuda_available(inputs):
            return flag_ops.gelu(x, approximate=approximate)
        return torch.nn.functional.gelu(x, approximate=approximate)

    return run


def _run_unary_torch(torch_fn: Callable[[Any], Any]) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        return torch_fn(inputs[0])

    return run


def _run_identity(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        return flag_ops.identity(
            inputs[0],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_reshape(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        return flag_ops.reshape(
            inputs[0],
            attrs["shape"],
            name=attrs.get("name", ""),
            reshape_mode=attrs.get("reshape_mode", "VIEW_ONLY"),
        )

    return run


def _run_transpose(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        return flag_ops.transpose(
            inputs[0],
            attrs["permutation"],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_slice(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        return flag_ops.slice(
            inputs[0],
            attrs.get("slices", ()),
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_concatenate(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        return flag_ops.concatenate(
            inputs,
            attrs["axis"],
            in_place_index=attrs.get("in_place_index"),
            name=attrs.get("name", ""),
        )

    return run


def _run_gen_index(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        return flag_ops.gen_index(
            inputs[0],
            attrs["axis"],
            compute_data_type=attrs.get("compute_data_type"),
            name=attrs.get("name", ""),
        )

    return run


def _run_conv2d(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        bias = inputs[2] if len(inputs) > 2 else None
        op_attrs = _public_attrs(attrs)
        if _cuda_available(inputs):
            return flag_ops.conv2d(
                inputs[0],
                inputs[1],
                bias=bias,
                stride=op_attrs.get("stride", 1),
                padding=op_attrs.get("padding", 0),
                dilation=op_attrs.get("dilation", 1),
                groups=op_attrs.get("groups", 1),
            )
        return torch.nn.functional.conv2d(
            inputs[0],
            inputs[1],
            bias=bias,
            stride=op_attrs.get("stride", 1),
            padding=op_attrs.get("padding", 0),
            dilation=op_attrs.get("dilation", 1),
            groups=op_attrs.get("groups", 1),
        )

    return run


def _run_mm(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        if _cuda_available(inputs):
            out_dtype = attrs.get("out_dtype")
            return flag_ops.mm(
                inputs[0],
                inputs[1],
                out_dtype=torch_dtype(out_dtype) if out_dtype else None,
            )
        out = torch.mm(inputs[0], inputs[1])
        out_dtype = attrs.get("out_dtype")
        if out_dtype:
            out = out.to(torch_dtype(out_dtype))
        return out

    return run


def _run_fused_bias_relu(
    inputs: list[Any], attrs: dict[str, Any]
) -> torch.Tensor:
    return torch.nn.functional.relu(_run_bias_add(inputs, attrs))


def _run_fused_bias_gelu(
    inputs: list[Any], attrs: dict[str, Any]
) -> torch.Tensor:
    approximate = attrs.get("approximate", "none")
    return torch.nn.functional.gelu(
        _run_bias_add(inputs, attrs), approximate=approximate
    )


def _run_fused_conv2d_bias_relu(flag_ops: Any) -> RunFn:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        op_attrs = _public_attrs(attrs)
        implementation = attrs.get("_implementation", "triton_fused")
        if _cuda_available(inputs) and implementation == "triton_fused":
            try:
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
            except NotImplementedError:
                pass
        y = _run_conv2d(flag_ops)(inputs, op_attrs)
        if _cuda_available(inputs):
            return flag_ops.relu(y)
        return torch.nn.functional.relu(y)

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
                run_fn=_run_binary(flag_ops),
            )
        )

    register_op(
        OpSchema(
            name="bias_add",
            normalize_fn=_normalize_bias_add,
            shape_fn=_bias_add_shape,
            run_fn=_run_bias_add,
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

    for name, torch_fn in (
        ("sqrt", torch.sqrt),
        ("abs", torch.abs),
        ("neg", torch.neg),
        ("tanh", torch.tanh),
        ("silu", torch.nn.functional.silu),
    ):
        register_op(
            OpSchema(
                name=name,
                normalize_fn=_normalize_unary(name),
                shape_fn=_shape_like_first,
                run_fn=_run_unary_torch(torch_fn),
                fusible=True,
            )
        )

    register_op(
        OpSchema(
            name="fused_bias_relu",
            normalize_fn=_normalize_bias_add,
            shape_fn=_bias_add_shape,
            run_fn=_run_fused_bias_relu,
            fusible=True,
        )
    )
    register_op(
        OpSchema(
            name="fused_bias_gelu",
            normalize_fn=_normalize_bias_add,
            shape_fn=_bias_add_shape,
            run_fn=_run_fused_bias_gelu,
            fusible=True,
        )
    )

    _DEFAULTS_REGISTERED = True
