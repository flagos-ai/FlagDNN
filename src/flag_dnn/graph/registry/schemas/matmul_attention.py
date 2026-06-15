from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.registry.core import OpDef, register_op_def
from flag_dnn.graph.registry.schemas._run_common import (
    _require_runtime_backend,
)
from flag_dnn.graph.registry.schemas.common import (
    _float32_spec,
    _pop_operand,
)
from flag_dnn.graph.tensor import TensorSpec, canonical_dtype, torch_dtype


def _matmul_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    a, b = input_specs[0], input_specs[1]
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise RuntimeError("graph matmul expects rank >= 2 inputs")
    if a.shape[-1] != b.shape[-2]:
        raise RuntimeError(
            f"graph matmul shape mismatch: {a.shape} cannot multiply {b.shape}"
        )
    batch_shape = torch.broadcast_shapes(
        tuple(a.shape[:-2]), tuple(b.shape[:-2])
    )
    out_dtype = attrs.get("out_dtype") or canonical_dtype(
        torch.result_type(
            torch.empty((), dtype=torch_dtype(a.dtype)),
            torch.empty((), dtype=torch_dtype(b.dtype)),
        )
    )
    return [
        TensorSpec(
            name="",
            shape=tuple(batch_shape) + (a.shape[-2], b.shape[-1]),
            dtype=out_dtype,
            device=a.device or b.device,
        )
    ]


def _sdpa_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    q, k, v = input_specs[0], input_specs[1], input_specs[2]
    for name, spec in (("q", q), ("k", k), ("v", v)):
        if len(spec.shape) != 4:
            raise RuntimeError(
                f"graph sdpa {name} must be a 4D (B, H, S, D) tensor"
            )
    if q.shape[3] != k.shape[3]:
        raise RuntimeError(
            "graph sdpa q and k head dimensions must match: "
            f"{q.shape} vs {k.shape}"
        )
    if k.shape[2] != v.shape[2]:
        raise RuntimeError(
            "graph sdpa k and v sequence lengths must match: "
            f"{k.shape} vs {v.shape}"
        )
    out = TensorSpec(
        name="",
        shape=(q.shape[0], q.shape[1], q.shape[2], v.shape[3]),
        dtype=q.dtype,
        device=q.device,
    )
    if attrs.get("generate_stats"):
        stats_shape = (q.shape[0], q.shape[1], q.shape[2], 1)
        return [out, _float32_spec(stats_shape, q.device)]
    return [out]


def _sdpa_backward_shape(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    q, k, v, o, dO, stats = input_specs[:6]
    for name, spec in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("o", o),
        ("dO", dO),
    ):
        if len(spec.shape) != 4:
            raise RuntimeError(
                f"graph sdpa_backward {name} must be a 4D tensor"
            )
    if q.shape[3] != k.shape[3]:
        raise RuntimeError(
            "graph sdpa_backward q and k head dimensions must match: "
            f"{q.shape} vs {k.shape}"
        )
    if k.shape[1] != v.shape[1]:
        raise RuntimeError(
            "graph sdpa_backward currently requires k and v to have the "
            f"same head count: {k.shape} vs {v.shape}"
        )
    if k.shape[2] != v.shape[2]:
        raise RuntimeError(
            "graph sdpa_backward k and v sequence lengths must match: "
            f"{k.shape} vs {v.shape}"
        )
    expected_o = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
    if o.shape != expected_o or dO.shape != expected_o:
        raise RuntimeError(
            "graph sdpa_backward expects o and dO shape "
            f"{expected_o}, got {o.shape} and {dO.shape}"
        )
    expected_stats = (q.shape[0], q.shape[1], q.shape[2], 1)
    if len(stats.shape) != 4 or stats.shape != expected_stats:
        raise RuntimeError(
            "graph sdpa_backward expects stats shape "
            f"{expected_stats}, got {stats.shape}"
        )
    return [
        TensorSpec(name="", shape=q.shape, dtype=q.dtype, device=q.device),
        TensorSpec(name="", shape=k.shape, dtype=k.dtype, device=k.device),
        TensorSpec(name="", shape=v.shape, dtype=v.dtype, device=v.device),
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


def _normalize_matmul(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    if len(args) > 2:
        raise TypeError("matmul expects at most two positional args")
    params = dict(kwargs)
    a = args[0] if args else _pop_operand(params, ("A", "a", "input"))
    if a is None:
        raise TypeError("matmul missing A tensor")
    if len(args) >= 2:
        b = args[1]
    else:
        b = _pop_operand(params, ("B", "b", "other", "mat2"))
    if b is None:
        raise TypeError("matmul missing B tensor")
    attrs = {
        "out_dtype": None,
        "compute_data_type": params.pop("compute_data_type", None),
        "padding": params.pop("padding", 0.0),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            f"matmul got unsupported graph attrs: {sorted(params)}"
        )
    return [ctx.as_value(a, "A"), ctx.as_value(b, "B")], attrs


def _normalize_sdpa(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    from flag_dnn.ops.sdpa import (
        _resolve_band,
        _resolve_generate_stats,
        _validate_dropout,
    )

    names = ("q", "k", "v", "is_inference")
    if len(args) > len(names):
        raise TypeError("sdpa got too many positional args")
    params = dict(kwargs)
    values: dict[str, Any] = {"is_inference": None}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    for name in ("q", "k", "v"):
        if name not in values:
            raise TypeError(f"sdpa missing {name}")

    bias = params.pop("bias", None)
    attn_scale = params.pop("attn_scale", None)
    if attn_scale is not None:
        attn_scale = float(attn_scale)
    alignment, left, right = _resolve_band(
        bool(params.pop("use_causal_mask", False)),
        bool(params.pop("use_causal_mask_bottom_right", False)),
        params.pop("sliding_window_length", None),
        params.pop("diagonal_alignment", None),
        params.pop("diagonal_band_left_bound", None),
        params.pop("diagonal_band_right_bound", None),
    )
    _validate_dropout(params.pop("dropout", None))
    generate_stats = _resolve_generate_stats(
        params.pop("generate_stats", None), values.pop("is_inference")
    )
    attrs = {
        "attn_scale": attn_scale,
        "diagonal_alignment": alignment,
        "diagonal_band_left_bound": left,
        "diagonal_band_right_bound": right,
        "generate_stats": generate_stats,
        "has_bias": bias is not None,
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(f"sdpa got unsupported graph attrs: {sorted(params)}")
    input_ids = [
        ctx.as_value(values["q"], "q"),
        ctx.as_value(values["k"], "k"),
        ctx.as_value(values["v"], "v"),
    ]
    if bias is not None:
        input_ids.append(ctx.as_value(bias, "bias"))
    return input_ids, attrs


def _normalize_sdpa_backward(
    ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[int], dict]:
    from flag_dnn.ops.sdpa import _resolve_band, _validate_dropout
    from flag_dnn.ops.sdpa_backward import _reject_unsupported

    names = ("q", "k", "v", "o", "dO", "stats")
    if len(args) > len(names):
        raise TypeError("sdpa_backward got too many positional args")
    params = dict(kwargs)
    values: dict[str, Any] = {}
    for name, value in zip(names, args):
        values[name] = value
    for name in names[len(args) :]:
        if name in params:
            values[name] = params.pop(name)
    for name in names:
        if name not in values:
            raise TypeError(f"sdpa_backward missing {name}")

    bias = params.pop("bias", None)
    dbias = params.pop("dBias", None)
    attn_scale = params.pop("attn_scale", None)
    if attn_scale is not None:
        if not isinstance(attn_scale, (int, float)):
            raise NotImplementedError(
                "sdpa_backward graph only supports float attn_scale"
            )
        attn_scale = float(attn_scale)

    use_alibi_mask = bool(params.pop("use_alibi_mask", False))
    use_padding_mask = bool(params.pop("use_padding_mask", False))
    seq_len_q = params.pop("seq_len_q", None)
    seq_len_kv = params.pop("seq_len_kv", None)
    max_total_seq_len_q = params.pop("max_total_seq_len_q", None)
    max_total_seq_len_kv = params.pop("max_total_seq_len_kv", None)
    dropout = params.pop("dropout", None)
    rng_dump = params.pop("rng_dump", None)
    score_mod = params.pop("score_mod", None)
    score_mod_bprop = params.pop("score_mod_bprop", None)
    sink_token = params.pop("sink_token", None)
    dsink_token = params.pop("dSink_token", None)
    _reject_unsupported(
        use_alibi_mask,
        use_padding_mask,
        seq_len_q,
        seq_len_kv,
        max_total_seq_len_q,
        max_total_seq_len_kv,
        rng_dump,
        score_mod,
        score_mod_bprop,
        sink_token,
        dsink_token,
    )
    _validate_dropout(dropout)

    alignment, left, right = _resolve_band(
        bool(params.pop("use_causal_mask", False)),
        bool(params.pop("use_causal_mask_bottom_right", False)),
        params.pop("sliding_window_length", None),
        params.pop("diagonal_alignment", None),
        params.pop("diagonal_band_left_bound", None),
        params.pop("diagonal_band_right_bound", None),
    )

    attrs = {
        "attn_scale": attn_scale,
        "diagonal_alignment": alignment,
        "diagonal_band_left_bound": left,
        "diagonal_band_right_bound": right,
        "has_bias": bias is not None,
        "has_dbias": dbias is not None,
        "use_deterministic_algorithm": bool(
            params.pop("use_deterministic_algorithm", False)
        ),
        "compute_data_type": params.pop("compute_data_type", None),
        "name": params.pop("name", ""),
    }
    if params:
        raise TypeError(
            f"sdpa_backward got unsupported graph attrs: {sorted(params)}"
        )

    input_ids = [ctx.as_value(values[name], name) for name in names]
    if bias is not None:
        input_ids.append(ctx.as_value(bias, "bias"))
    if dbias is not None:
        input_ids.append(ctx.as_value(dbias, "dBias"))
    return input_ids, attrs


# --- eager-fallback run functions ---


def _run_mm(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> torch.Tensor:
        _require_runtime_backend(inputs, "mm")
        out_dtype = attrs.get("out_dtype")
        return flag_ops.mm(
            inputs[0],
            inputs[1],
            out_dtype=torch_dtype(out_dtype) if out_dtype else None,
        )

    return run


def _run_matmul(flag_ops: Any) -> Any:
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


def _run_sdpa(flag_ops: Any) -> Any:
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


def _run_sdpa_backward(flag_ops: Any) -> Any:
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


def register(flag_ops: Any) -> None:
    """Register the matmul / attention op family (mm / matmul / sdpa /
    sdpa_backward)."""
    register_op_def(
        OpDef(
            name="mm",
            normalize=_normalize_mm,
            shape=_mm_shape,
            run=_run_mm(flag_ops),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="matmul",
            normalize=_normalize_matmul,
            shape=_matmul_shape,
            run=_run_matmul(flag_ops),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="sdpa",
            normalize=_normalize_sdpa,
            shape=_sdpa_shape,
            run=_run_sdpa(flag_ops),
            num_outputs=(1, 2),
            fusible=True,
        )
    )
    register_op_def(
        OpDef(
            name="sdpa_backward",
            normalize=_normalize_sdpa_backward,
            shape=_sdpa_backward_shape,
            run=_run_sdpa_backward(flag_ops),
            num_outputs=3,
        )
    )


__all__ = (
    "register",
    "_matmul_shape",
    "_sdpa_shape",
    "_sdpa_backward_shape",
    "_mm_shape",
    "_normalize_mm",
    "_normalize_matmul",
    "_normalize_sdpa",
    "_normalize_sdpa_backward",
)
