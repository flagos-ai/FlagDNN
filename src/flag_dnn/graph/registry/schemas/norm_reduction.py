from __future__ import annotations

from typing import Any

from flag_dnn.graph.registry.schemas.common import (
    _float32_spec,
    _normalize_axis,
)
from flag_dnn.graph.tensor import TensorSpec, canonical_dtype


def _normalize_reduction(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("reduction expects at most two positional args")
    params = dict(kwargs)
    x = args[0] if args else params.pop("input", None)
    if x is None:
        raise TypeError("reduction missing input tensor")
    mode = args[1] if len(args) >= 2 else params.pop("mode", None)
    if mode is None:
        raise TypeError("reduction missing mode")
    attrs = {
        "mode": mode,
        "dim": params.pop("dim", None),
        "keepdim": bool(params.pop("keepdim", True)),
        "dtype": params.pop("dtype", None),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            f"reduction got unsupported graph attrs: {sorted(params)}"
        )
    return [ctx.as_value(x, "input")], attrs


def _reduction_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    rank = len(inp.shape)
    dim = attrs.get("dim")
    if dim is None:
        dims = list(range(rank))
    elif isinstance(dim, int):
        dims = [_normalize_axis(dim, rank, "reduction")]
    else:
        dims = [_normalize_axis(item, rank, "reduction") for item in dim]
    dims = sorted(set(dims))
    keepdim = bool(attrs.get("keepdim", True))
    out_shape: list[Any] = []
    for index, size in enumerate(inp.shape):
        if index in dims:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(size)
    dtype = attrs.get("dtype")
    if dtype is not None:
        out_dtype = canonical_dtype(dtype)
    else:
        out_dtype = inp.dtype
    return [
        TensorSpec(
            name="",
            shape=tuple(out_shape),
            dtype=out_dtype,
            device=inp.device,
        )
    ]


def _norm_axes_from_scale(
    input_spec: TensorSpec, scale_spec: TensorSpec
) -> tuple[int, ...]:
    rank = len(input_spec.shape)
    scale_shape = tuple(scale_spec.shape)
    if len(scale_shape) > rank:
        raise RuntimeError("norm scale rank cannot exceed input rank")
    aligned = (1,) * (rank - len(scale_shape)) + scale_shape
    axes = tuple(index for index, size in enumerate(aligned) if size != 1)
    if not axes:
        axes = (rank - 1,)
    return axes


def _norm_stats_shape(
    input_spec: TensorSpec, scale_spec: TensorSpec
) -> tuple[Any, ...]:
    axes = set(_norm_axes_from_scale(input_spec, scale_spec))
    return tuple(
        1 if index in axes else size
        for index, size in enumerate(input_spec.shape)
    )


def _layernorm_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    del attrs
    inp = input_specs[0]
    stat_shape = _norm_stats_shape(inp, input_specs[1])
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=inp.dtype,
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        ),
        _float32_spec(stat_shape, inp.device),
        _float32_spec(stat_shape, inp.device),
    ]


_RMSNORM_RHT_AMAX_RPC_CANDIDATES = (2, 4, 8)
_RMSNORM_RHT_AMAX_TARGET_MIN_CTAS = 148


def _rmsnorm_rht_amax_pick_rows_per_cta(m: int) -> int:
    for rows_per_cta in reversed(_RMSNORM_RHT_AMAX_RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        if m // rows_per_cta >= _RMSNORM_RHT_AMAX_TARGET_MIN_CTAS:
            return rows_per_cta
    return _RMSNORM_RHT_AMAX_RPC_CANDIDATES[0]


def _squeeze_trailing_unit_spec_shape(
    shape: tuple[Any, ...], expected_rank: int
) -> tuple[Any, ...]:
    if len(shape) == expected_rank + 1 and shape[-1] == 1:
        return shape[:-1]
    return shape


def _rmsnorm_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    del attrs
    inp = input_specs[0]
    stat_shape = _norm_stats_shape(inp, input_specs[1])
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=inp.dtype,
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        ),
        _float32_spec(stat_shape, inp.device),
    ]


def _rmsnorm_rht_amax_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    weight = input_specs[1]
    x_shape = _squeeze_trailing_unit_spec_shape(tuple(inp.shape), 2)
    w_shape = _squeeze_trailing_unit_spec_shape(tuple(weight.shape), 1)
    if len(x_shape) != 2:
        raise RuntimeError(
            "rmsnorm_rht_amax_wrapper_sm100 x_tensor must be 2D"
        )
    if len(w_shape) != 1:
        raise RuntimeError(
            "rmsnorm_rht_amax_wrapper_sm100 w_tensor must be 1D"
        )
    m, n = x_shape
    if isinstance(n, int) and isinstance(w_shape[0], int) and w_shape[0] != n:
        raise RuntimeError(
            "rmsnorm_rht_amax_wrapper_sm100 w_tensor length must match "
            "x hidden dimension"
        )
    if isinstance(n, int) and n % 16 != 0:
        raise RuntimeError(
            "rmsnorm_rht_amax_wrapper_sm100 N must be divisible by 16"
        )

    rows_per_cta = attrs.get("rows_per_cta")
    if rows_per_cta is None:
        if not isinstance(m, int):
            raise RuntimeError(
                "rmsnorm_rht_amax_wrapper_sm100 requires concrete M when "
                "rows_per_cta is omitted"
            )
        rows_per_cta = _rmsnorm_rht_amax_pick_rows_per_cta(m)
    rows_per_cta = int(rows_per_cta)
    if rows_per_cta <= 0:
        raise RuntimeError(
            "rmsnorm_rht_amax_wrapper_sm100 rows_per_cta must be positive"
        )
    if isinstance(m, int):
        if m % rows_per_cta != 0:
            raise RuntimeError(
                "rmsnorm_rht_amax_wrapper_sm100 M must be divisible by "
                "rows_per_cta"
            )
        amax_shape = (m // rows_per_cta,)
    else:
        amax_shape = (m,)

    return [
        TensorSpec(
            name="o_tensor",
            shape=x_shape,
            dtype=inp.dtype,
            layout="contiguous",
            device=inp.device,
            contiguous=True,
        ),
        _float32_spec(amax_shape, inp.device).with_name("amax_tensor"),
    ]


def _batchnorm_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    del attrs
    inp = input_specs[0]
    running_mean = input_specs[3]
    running_var = input_specs[4]
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=inp.dtype,
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        ),
        _float32_spec(running_mean.shape, inp.device),
        _float32_spec(running_var.shape, inp.device),
        _float32_spec(running_mean.shape, inp.device),
        _float32_spec(running_var.shape, inp.device),
    ]


def _normalize_batchnorm_inference(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    names = ("input", "mean", "inv_variance", "scale", "bias")
    if len(args) > len(names):
        raise TypeError("batchnorm_inference got too many positional args")
    params = dict(kwargs)
    values = {}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"batchnorm_inference missing {missing[0]} tensor")
    attrs = {
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            "batchnorm_inference got unsupported graph attrs: "
            f"{sorted(params)}"
        )
    return [ctx.as_value(values[name], name) for name in names], attrs


def _normalize_layernorm(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    names = ("norm_forward_phase", "input", "scale", "bias", "epsilon")
    if len(args) > len(names):
        raise TypeError("layernorm got too many positional args")
    params = dict(kwargs)
    values = {}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"layernorm missing {missing[0]}")
    attrs = {
        "norm_forward_phase": values.pop("norm_forward_phase"),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            f"layernorm got unsupported graph attrs: {sorted(params)}"
        )
    return [
        ctx.as_value(values[name], name)
        for name in ("input", "scale", "bias", "epsilon")
    ], attrs


def _normalize_rmsnorm(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    names = ("norm_forward_phase", "input", "scale", "bias", "epsilon")
    if len(args) > len(names):
        raise TypeError("rmsnorm got too many positional args")
    params = dict(kwargs)
    values = {"bias": None, "epsilon": 1e-5}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    for name in ("norm_forward_phase", "input", "scale"):
        if name not in values:
            raise TypeError(f"rmsnorm missing {name}")
    attrs = {
        "norm_forward_phase": values.pop("norm_forward_phase"),
        "has_bias": values.get("bias") is not None,
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            f"rmsnorm got unsupported graph attrs: {sorted(params)}"
        )
    input_ids = [
        ctx.as_value(values["input"], "input"),
        ctx.as_value(values["scale"], "scale"),
    ]
    if values.get("bias") is not None:
        input_ids.append(ctx.as_value(values["bias"], "bias"))
    input_ids.append(ctx.as_value(values["epsilon"], "epsilon"))
    return input_ids, attrs


def _normalize_rmsnorm_rht_amax(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    names = ("x_tensor", "w_tensor")
    if len(args) > len(names):
        raise TypeError(
            "rmsnorm_rht_amax_wrapper_sm100 got too many positional args"
        )
    params = dict(kwargs)
    values = {}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"rmsnorm_rht_amax_wrapper_sm100 missing {missing[0]}")
    current_stream = params.pop("current_stream", None)
    if current_stream is not None:
        raise TypeError(
            "rmsnorm_rht_amax_wrapper_sm100 current_stream is not supported "
            "in graph capture"
        )
    attrs = {
        "eps": float(params.pop("eps", 1e-5)),
        "num_threads": params.pop("num_threads", None),
        "rows_per_cta": params.pop("rows_per_cta", None),
        "name": params.pop("name", ""),
    }
    if attrs["num_threads"] is not None:
        attrs["num_threads"] = int(attrs["num_threads"])
    if attrs["rows_per_cta"] is not None:
        attrs["rows_per_cta"] = int(attrs["rows_per_cta"])
    if params:
        raise TypeError(
            "rmsnorm_rht_amax_wrapper_sm100 got unsupported graph attrs: "
            f"{sorted(params)}"
        )
    return [
        ctx.as_value(values["x_tensor"], "x_tensor"),
        ctx.as_value(values["w_tensor"], "w_tensor"),
    ], attrs


def _normalize_batchnorm(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    names = (
        "input",
        "scale",
        "bias",
        "in_running_mean",
        "in_running_var",
        "epsilon",
        "momentum",
    )
    if len(args) > len(names):
        raise TypeError("batchnorm got too many positional args")
    params = dict(kwargs)
    values = {}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"batchnorm missing {missing[0]}")
    peer_stats = params.pop("peer_stats", [])
    if peer_stats is None:
        peer_stats = []
    if not isinstance(peer_stats, (list, tuple)):
        raise TypeError("batchnorm peer_stats must be a list or tuple")
    attrs = {
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
        "peer_stats_count": len(peer_stats),
    }
    if params:
        raise TypeError(
            f"batchnorm got unsupported graph attrs: {sorted(params)}"
        )
    input_ids = [ctx.as_value(values[name], name) for name in names]
    input_ids.extend(
        ctx.as_value(peer_stat, f"peer_stats_{index}")
        for index, peer_stat in enumerate(peer_stats)
    )
    return input_ids, attrs


__all__ = (
    "_normalize_reduction",
    "_reduction_shape",
    "_norm_axes_from_scale",
    "_norm_stats_shape",
    "_layernorm_shape",
    "_rmsnorm_rht_amax_pick_rows_per_cta",
    "_squeeze_trailing_unit_spec_shape",
    "_rmsnorm_shape",
    "_rmsnorm_rht_amax_shape",
    "_batchnorm_shape",
    "_normalize_batchnorm_inference",
    "_normalize_layernorm",
    "_normalize_rmsnorm",
    "_normalize_rmsnorm_rht_amax",
    "_normalize_batchnorm",
)
