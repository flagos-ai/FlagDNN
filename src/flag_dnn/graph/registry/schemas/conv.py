from __future__ import annotations

from typing import Any

from flag_dnn.graph.registry.core import OpDef, register_op_def
from flag_dnn.graph.registry.schemas._run_common import (
    _public_attrs,
    _require_runtime_backend,
)
from flag_dnn.graph.registry.schemas.common import (
    _rank_of,
    _shape_like_first,
    _tuple_n,
)
from flag_dnn.graph.tensor import TensorSpec


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


def _normalize_convolution_mode(convolution_mode: Any) -> str:
    if convolution_mode is None:
        return "CROSS_CORRELATION"
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    if mode in ("CROSS_CORRELATION", "CONVOLUTION"):
        return mode
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
            pre_values: list[int] = []
            post_values: list[int] = []
            for axis in range(rank):
                kernel = int(weight.shape[2 + axis])
                effective_kernel = dilation[axis] * (kernel - 1) + 1
                total_pad = max(effective_kernel - 1, 0)
                before = total_pad // 2
                pre_values.append(before)
                post_values.append(total_pad - before)
            return tuple(pre_values), tuple(post_values)
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")

    if isinstance(padding, int):
        values = (int(padding),) * rank
        return values, values

    values = tuple(int(v) for v in padding)
    if len(values) == rank:
        return values, values
    if len(values) == 2 * rank:
        pre_tuple = tuple(values[2 * axis] for axis in range(rank))
        post_tuple = tuple(values[2 * axis + 1] for axis in range(rank))
        return pre_tuple, post_tuple
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
    spatial_shape: tuple[Any, ...]

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
        raise RuntimeError(
            "graph conv_fprop input channels must divide groups"
        )
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


def _conv_dgrad_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    loss, weight = input_specs[0], input_specs[1]
    input_size = tuple(attrs["input_size"])
    rank = _conv_spatial_rank_from_values(loss, weight, "conv_dgrad")
    is_unbatched_1d = rank == 1 and len(input_size) == 2
    expected_input_rank = rank + 1 if is_unbatched_1d else rank + 2
    if len(input_size) != expected_input_rank:
        raise RuntimeError(
            f"graph conv_dgrad input_size must have length "
            f"{expected_input_rank}, got {input_size}"
        )

    stride = _tuple_n(attrs.get("stride", 1), rank, "stride")
    dilation = _tuple_n(attrs.get("dilation", 1), rank, "dilation")
    _normalize_convolution_mode(attrs.get("convolution_mode"))
    pre_padding, post_padding = _normalize_conv_padding_from_attrs(
        weight, stride, dilation, attrs, "conv_dgrad"
    )
    groups = int(attrs.get("groups", 1))

    if is_unbatched_1d:
        leading_shape: tuple[Any, ...] = ()
        c_in = input_size[0]
        dx_spatial = input_size[1:]
        loss_leading: tuple[Any, ...] = ()
        loss_channels = loss.shape[0]
        loss_spatial = loss.shape[1:]
    else:
        leading_shape = (input_size[0],)
        c_in = input_size[1]
        dx_spatial = input_size[2:]
        loss_leading = (loss.shape[0],)
        loss_channels = loss.shape[1]
        loss_spatial = loss.shape[2:]

    c_out = weight.shape[0]
    c_per_group = weight.shape[1]
    kernel_shape = weight.shape[2:]
    static_values = (
        (c_in, c_out, c_per_group)
        + tuple(dx_spatial)
        + tuple(kernel_shape)
        + tuple(loss_spatial)
    )
    if not all(isinstance(v, int) for v in static_values):
        raise NotImplementedError(
            "graph conv_dgrad dynamic shapes are not enabled"
        )
    if any(int(v) < 0 for v in input_size):
        raise RuntimeError(
            "graph conv_dgrad input_size dimensions must be non-negative"
        )
    if int(c_in) % groups != 0:
        raise RuntimeError(
            "graph conv_dgrad input channels must divide groups"
        )
    if int(c_out) % groups != 0:
        raise RuntimeError(
            "graph conv_dgrad output channels must divide groups"
        )
    if int(c_per_group) != int(c_in) // groups:
        raise RuntimeError("graph conv_dgrad filter channel mismatch")
    if tuple(loss_leading) != tuple(leading_shape):
        raise RuntimeError("graph conv_dgrad loss batch size mismatch")
    if int(loss_channels) != int(c_out):
        raise RuntimeError("graph conv_dgrad loss channel mismatch")

    expected_loss_spatial = []
    for axis in range(rank):
        out_dim = _conv_out_dim(
            int(dx_spatial[axis]),
            pre_padding[axis],
            post_padding[axis],
            dilation[axis],
            int(kernel_shape[axis]),
            stride[axis],
        )
        if out_dim < 0:
            raise RuntimeError(
                "computed graph conv_dgrad loss size is negative"
            )
        expected_loss_spatial.append(max(out_dim, 0))
    if tuple(loss_spatial) != tuple(expected_loss_spatial):
        raise RuntimeError(
            "graph conv_dgrad loss spatial shape does not match input_size"
        )

    return [
        TensorSpec(
            name="",
            shape=input_size,
            dtype=loss.dtype,
            layout="contiguous",
            device=loss.device,
            contiguous=True,
        )
    ]


def _conv_wgrad_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    image, loss = input_specs[0], input_specs[1]
    filter_size = tuple(int(dim) for dim in attrs["filter_size"])
    if len(filter_size) not in (3, 4, 5):
        raise RuntimeError(
            "graph conv_wgrad filter_size must describe a 1D/2D/3D filter"
        )
    rank = len(filter_size) - 2
    image_rank = _rank_of(image)
    loss_rank = _rank_of(loss)
    is_unbatched_1d = rank == 1 and image_rank == 2 and loss_rank == 2
    expected_rank = rank + 1 if is_unbatched_1d else rank + 2
    if image_rank != expected_rank or loss_rank != expected_rank:
        raise RuntimeError(
            f"graph conv_wgrad expected image/loss rank {expected_rank}, "
            f"got {image_rank}/{loss_rank}"
        )

    stride = _tuple_n(attrs.get("stride", 1), rank, "stride")
    dilation = _tuple_n(attrs.get("dilation", 1), rank, "dilation")
    _normalize_convolution_mode(attrs.get("convolution_mode"))
    filter_spec = TensorSpec(
        name="",
        shape=filter_size,
        dtype=image.dtype,
        layout="contiguous",
        device=image.device,
        contiguous=True,
    )
    pre_padding, post_padding = _normalize_conv_padding_from_attrs(
        filter_spec, stride, dilation, attrs, "conv_wgrad"
    )
    groups = int(attrs.get("groups", 1))

    if is_unbatched_1d:
        image_leading: tuple[Any, ...] = ()
        c_in = image.shape[0]
        image_spatial = image.shape[1:]
        loss_leading: tuple[Any, ...] = ()
        loss_channels = loss.shape[0]
        loss_spatial = loss.shape[1:]
    else:
        image_leading = (image.shape[0],)
        c_in = image.shape[1]
        image_spatial = image.shape[2:]
        loss_leading = (loss.shape[0],)
        loss_channels = loss.shape[1]
        loss_spatial = loss.shape[2:]

    c_out = filter_size[0]
    c_per_group = filter_size[1]
    kernel_shape = filter_size[2:]
    static_values = (
        (c_in, c_out, c_per_group)
        + tuple(image_spatial)
        + tuple(kernel_shape)
        + tuple(loss_spatial)
    )
    if not all(isinstance(v, int) for v in static_values):
        raise NotImplementedError(
            "graph conv_wgrad dynamic shapes are not enabled"
        )
    if any(int(v) < 0 for v in filter_size):
        raise RuntimeError(
            "graph conv_wgrad filter_size dimensions must be non-negative"
        )
    if int(c_in) % groups != 0:
        raise RuntimeError(
            "graph conv_wgrad input channels must divide groups"
        )
    if int(c_out) % groups != 0:
        raise RuntimeError(
            "graph conv_wgrad output channels must divide groups"
        )
    if int(c_per_group) != int(c_in) // groups:
        raise RuntimeError("graph conv_wgrad filter channel mismatch")
    if tuple(loss_leading) != tuple(image_leading):
        raise RuntimeError("graph conv_wgrad loss batch size mismatch")
    if int(loss_channels) != int(c_out):
        raise RuntimeError("graph conv_wgrad loss channel mismatch")

    expected_loss_spatial = []
    for axis in range(rank):
        out_dim = _conv_out_dim(
            int(image_spatial[axis]),
            pre_padding[axis],
            post_padding[axis],
            dilation[axis],
            int(kernel_shape[axis]),
            stride[axis],
        )
        if out_dim < 0:
            raise RuntimeError(
                "computed graph conv_wgrad loss size is negative"
            )
        expected_loss_spatial.append(max(out_dim, 0))
    if tuple(loss_spatial) != tuple(expected_loss_spatial):
        raise RuntimeError(
            "graph conv_wgrad loss spatial shape does not match filter_size"
        )

    return [
        TensorSpec(
            name="",
            shape=filter_size,
            dtype=image.dtype,
            layout="contiguous",
            device=image.device,
            contiguous=True,
        )
    ]


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


def _normalize_conv_dgrad(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 4:
        raise TypeError("conv_dgrad got too many positional args")
    names = ["loss", "filter", "input_size", "padding"]
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
    params: dict[str, Any] = defaults.copy()
    params.update(kwargs)
    for idx, arg in enumerate(args):
        name = names[idx]
        if name in kwargs:
            raise TypeError(f"conv_dgrad got multiple values for {name}")
        params[name] = arg

    loss = params.pop("loss", None)
    filter_value = params.pop("filter", None)
    input_size = params.get("input_size")
    if loss is None or filter_value is None:
        raise TypeError("conv_dgrad missing loss or filter")
    if input_size is None:
        raise TypeError("conv_dgrad missing input_size")
    input_size = tuple(int(dim) for dim in input_size)
    params["input_size"] = input_size

    rank = _conv_spatial_rank_from_values(loss, filter_value, "conv_dgrad")
    if rank == 1 and len(input_size) == 2:
        expected_input_rank = 2
    else:
        expected_input_rank = rank + 2
    if len(input_size) != expected_input_rank:
        raise RuntimeError(
            f"conv_dgrad input_size must have length {expected_input_rank}, "
            f"got {input_size}"
        )

    params["stride"] = _tuple_n(params["stride"], rank, "stride")
    params["dilation"] = _tuple_n(params["dilation"], rank, "dilation")
    _normalize_convolution_mode(params.get("convolution_mode"))

    padding = params.get("padding")
    pre_padding = params.get("pre_padding")
    post_padding = params.get("post_padding")
    if pre_padding is not None or post_padding is not None:
        if padding is not None:
            raise TypeError(
                "conv_dgrad accepts either padding or "
                "pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                "conv_dgrad requires both pre_padding and post_padding"
            )
        params["pre_padding"] = _tuple_n(pre_padding, rank, "pre_padding")
        params["post_padding"] = _tuple_n(post_padding, rank, "post_padding")
    elif padding is not None and not isinstance(padding, str):
        params["padding"] = _tuple_n(padding, rank, "padding")

    return [
        ctx.as_value(loss, "loss"),
        ctx.as_value(filter_value, "filter"),
    ], params


def _normalize_conv_wgrad(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 4:
        raise TypeError("conv_wgrad got too many positional args")
    names = ["image", "loss", "filter_size", "padding"]
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
    params: dict[str, Any] = defaults.copy()
    params.update(kwargs)
    for idx, arg in enumerate(args):
        name = names[idx]
        if name in kwargs:
            raise TypeError(f"conv_wgrad got multiple values for {name}")
        params[name] = arg

    image = params.pop("image", None)
    loss = params.pop("loss", None)
    filter_size = params.get("filter_size")
    if image is None or loss is None:
        raise TypeError("conv_wgrad missing image or loss")
    if filter_size is None:
        raise TypeError("conv_wgrad missing filter_size")
    filter_size = tuple(int(dim) for dim in filter_size)
    if len(filter_size) not in (3, 4, 5):
        raise RuntimeError(
            "conv_wgrad filter_size must describe a 1D/2D/3D filter"
        )
    params["filter_size"] = filter_size

    rank = len(filter_size) - 2
    image_rank = _rank_of(image)
    loss_rank = _rank_of(loss)
    is_unbatched_1d = rank == 1 and image_rank == 2 and loss_rank == 2
    expected_rank = rank + 1 if is_unbatched_1d else rank + 2
    if image_rank != expected_rank or loss_rank != expected_rank:
        raise RuntimeError(
            f"conv_wgrad expected image/loss rank {expected_rank}, "
            f"got {image_rank}/{loss_rank}"
        )

    params["stride"] = _tuple_n(params["stride"], rank, "stride")
    params["dilation"] = _tuple_n(params["dilation"], rank, "dilation")
    _normalize_convolution_mode(params.get("convolution_mode"))

    padding = params.get("padding")
    pre_padding = params.get("pre_padding")
    post_padding = params.get("post_padding")
    if pre_padding is not None or post_padding is not None:
        if padding is not None:
            raise TypeError(
                "conv_wgrad accepts either padding or "
                "pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                "conv_wgrad requires both pre_padding and post_padding"
            )
        params["pre_padding"] = _tuple_n(pre_padding, rank, "pre_padding")
        params["post_padding"] = _tuple_n(post_padding, rank, "post_padding")
    elif padding is not None and not isinstance(padding, str):
        params["padding"] = _tuple_n(padding, rank, "padding")

    return [
        ctx.as_value(image, "image"),
        ctx.as_value(loss, "loss"),
    ], params


def _normalize_causal_conv1d(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    names = ("x", "weight", "bias", "activation")
    if len(args) > len(names):
        raise TypeError("causal_conv1d got too many positional args")
    params = dict(kwargs)
    values = {"bias": None, "activation": "identity"}
    for name, value in zip(names, args):
        values[name] = value
    if "input" in params and "x" not in params:
        params["x"] = params.pop("input")
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    if values.get("x") is None or values.get("weight") is None:
        raise TypeError("causal_conv1d missing x/input or weight")
    if params:
        raise TypeError(
            f"causal_conv1d got unsupported graph attrs: {sorted(params)}"
        )
    attrs = {
        "activation": str(values["activation"]).lower(),
        "has_bias": values.get("bias") is not None,
    }
    input_ids = [
        ctx.as_value(values["x"], "x"),
        ctx.as_value(values["weight"], "weight"),
    ]
    if values.get("bias") is not None:
        input_ids.append(ctx.as_value(values["bias"], "bias"))
    return input_ids, attrs


# --- eager-fallback run functions ---


def _run_conv2d(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
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


def _run_conv_fprop(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
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


def _run_conv_dgrad(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
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


def _run_conv_wgrad(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
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


def _run_causal_conv1d(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "causal_conv1d")
        bias = inputs[2] if attrs.get("has_bias") else None
        return flag_ops.causal_conv1d(
            inputs[0],
            inputs[1],
            bias=bias,
            activation=attrs.get("activation", "identity"),
        )

    return run


def register(flag_ops: Any) -> None:
    """Register the convolution op family (conv2d / conv_fprop / conv_dgrad /
    conv_wgrad / causal_conv1d)."""
    register_op_def(
        OpDef(
            name="conv2d",
            normalize=_normalize_conv2d,
            shape=_conv2d_shape,
            run=_run_conv2d(flag_ops),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="conv_fprop",
            normalize=_normalize_conv_fprop,
            shape=_conv_fprop_shape,
            run=_run_conv_fprop(flag_ops),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="conv_dgrad",
            normalize=_normalize_conv_dgrad,
            shape=_conv_dgrad_shape,
            run=_run_conv_dgrad(flag_ops),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="conv_wgrad",
            normalize=_normalize_conv_wgrad,
            shape=_conv_wgrad_shape,
            run=_run_conv_wgrad(flag_ops),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="causal_conv1d",
            normalize=_normalize_causal_conv1d,
            shape=_shape_like_first,
            run=_run_causal_conv1d(flag_ops),
            fusible=True,
        )
    )


__all__ = (
    "register",
    "_pair",
    "_conv_out_dim",
    "_normalize_convolution_mode",
    "_conv_spatial_rank_from_values",
    "_normalize_conv_padding_from_attrs",
    "_normalize_padding_from_spec",
    "_conv2d_shape",
    "_conv_fprop_shape",
    "_conv_dgrad_shape",
    "_conv_wgrad_shape",
    "_normalize_conv2d",
    "_normalize_conv_fprop",
    "_normalize_conv_dgrad",
    "_normalize_conv_wgrad",
    "_normalize_causal_conv1d",
)
