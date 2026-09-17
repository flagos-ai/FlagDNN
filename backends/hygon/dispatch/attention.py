# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon dispatch attention."""

from __future__ import annotations

from ..dispatch.nn_common import (
    ScalarArgument,
    ELEMENT_SIZES,
    FLOAT_TYPES,
    FP8_TYPES,
    GridSpec,
    INTERNAL_TUNING,
    KernelStagePlan,
    MAX_I32,
    MAX_I64,
    NodePlan,
    SDPA_BACKWARD_TUNING,
    SDPA_FP8_BACKWARD_TUNING,
    SDPA_FP8_TUNING,
    SDPA_TUNING,
    UNBOUNDED_DIAGONAL,
    WORKSPACE_ALIGNMENT,
    WorkspaceTensor,
    _ceil_div,
    _checked_product,
    _is_contiguous,
    _make_stage,
    _next_power_of_two,
    _require_flag,
    _require_integer,
    _require_number,
    _require_object,
    _same_dtype,
    _tensor_pointer,
    _workspace_pointer,
)
from typing import Any
from typing import Mapping
from typing import Sequence


def _attention_full_dimension_block(value: int) -> int:
    """Return a tl.dot-compatible power-of-two block for a full D/V axis."""
    return max(16, _next_power_of_two(value))


def _attention_output_dimension_block(value: int) -> int:
    """Return the bounded output-axis block used by backward attention."""
    return min(32, _attention_full_dimension_block(value))


def _validate_qkv(
    q: Mapping[str, Any],
    k: Mapping[str, Any],
    v: Mapping[str, Any],
    *,
    fp8: bool,
) -> dict[str, int | str]:
    allowed = FP8_TYPES if fp8 else FLOAT_TYPES
    data_type = _same_dtype((q, k, v), allowed, "SDPA Q/K/V")
    if any(len(tensor["dimensions"]) != 4 for tensor in (q, k, v)):
        raise ValueError("SDPA Q/K/V must be rank-4 BHSD tensors")
    qd, kd, vd = (
        list(q["dimensions"]),
        list(k["dimensions"]),
        list(v["dimensions"]),
    )
    if qd[0] != kd[0] or qd[0] != vd[0]:
        raise ValueError("SDPA Q/K/V batch dimensions must match")
    if qd[3] != kd[3]:
        raise ValueError("SDPA Q/K head dimensions must match")
    if kd[2] != vd[2]:
        raise ValueError("SDPA K/V sequence dimensions must match")
    if qd[1] % kd[1] or qd[1] % vd[1]:
        raise ValueError(
            "SDPA query heads must be divisible by key/value heads"
        )
    if qd[3] > 256 or vd[3] > 256:
        raise ValueError(
            "SDPA head dimensions greater than 256 are unsupported"
        )
    return {
        "data_type": data_type,
        "batch": qd[0],
        "heads": qd[1],
        "key_heads": kd[1],
        "value_heads": vd[1],
        "sequence_q": qd[2],
        "sequence_kv": kd[2],
        "head_dimension": qd[3],
        "value_dimension": vd[3],
        "q_per_k": qd[1] // kd[1],
        "q_per_v": qd[1] // vd[1],
    }


def _validate_attention_tensor(
    tensor: Mapping[str, Any],
    expected_shape: Sequence[int],
    expected_type: str,
    name: str,
) -> None:
    if (
        tensor["data_type"] != expected_type
        or list(tensor["dimensions"]) != list(expected_shape)
        or len(tensor["dimensions"]) != 4
    ):
        raise ValueError(f"{name} tensor metadata is incorrect")


def _validate_bias(
    tensor: Mapping[str, Any],
    shape: Mapping[str, int | str],
    data_type: str,
    name: str,
) -> None:
    dims = list(tensor["dimensions"])
    if (
        tensor["data_type"] != data_type
        or len(dims) != 4
        or dims[0] not in (1, shape["batch"])
        or dims[1] not in (1, shape["heads"])
        or dims[2] != shape["sequence_q"]
        or dims[3] != shape["sequence_kv"]
    ):
        raise ValueError(
            f"{name} must broadcast over B/H and match Q/KV sequences"
        )


def _validate_fp32_scalar(tensor: Mapping[str, Any], name: str) -> None:
    if (
        tensor["data_type"] != "float32"
        or _checked_product(list(tensor["dimensions"]), name) != 1
    ):
        raise ValueError(f"{name} must be a one-element float32 tensor")


def _validate_attention(node: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(node["operation"])
    p = _require_object(node["parameters"], "node.parameters")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    fp8 = operation in ("sdpa_fp8", "sdpa_fp8_backward")
    shape = _validate_qkv(ports["q"], ports["k"], ports["v"], fp8=fp8)
    if operation == "sdpa_backward" and (
        shape["key_heads"] != shape["value_heads"]
    ):
        raise ValueError("SDPA backward requires matching K/V head counts")
    data_type = str(shape["data_type"])
    b = int(shape["batch"])
    h = int(shape["heads"])
    sq = int(shape["sequence_q"])
    d = int(shape["head_dimension"])
    dv = int(shape["value_dimension"])
    output_shape = [b, h, sq, dv]
    stats_shape = [b, h, sq, 1]

    _validate_attention_tensor(ports["o"], output_shape, data_type, "SDPA O")
    if operation in ("sdpa_backward", "sdpa_fp8_backward"):
        _validate_attention_tensor(
            ports["do"], output_shape, data_type, "SDPA dO"
        )
        for primal, gradient in (("q", "dq"), ("k", "dk"), ("v", "dv")):
            _validate_attention_tensor(
                ports[gradient],
                list(ports[primal]["dimensions"]),
                data_type,
                f"SDPA {gradient}",
            )
    stats = ports["stats"]
    if (
        stats["data_type"] != "float32"
        or list(stats["dimensions"]) != stats_shape
    ):
        raise ValueError("SDPA stats must be float32 [B,H,SQ,1]")

    has_bias = _require_flag(p, "has_bias")
    has_dbias = _require_flag(p, "has_dbias")
    if has_bias:
        _validate_bias(ports["bias"], shape, data_type, "SDPA bias")
    if has_dbias:
        _validate_bias(ports["dbias"], shape, data_type, "SDPA dbias")
    if operation != "sdpa_backward" and has_dbias:
        raise ValueError(f"{operation} cannot produce dbias")

    for name in (
        "batch",
        "heads",
        "key_heads",
        "value_heads",
        "sequence_q",
        "sequence_kv",
        "head_dimension",
        "value_dimension",
        "q_per_k",
        "q_per_v",
    ):
        if _require_integer(p, name, maximum=MAX_I64) != int(shape[name]):
            raise ValueError(
                f"parameters.{name} is inconsistent with Q/K/V metadata"
            )
    min_diag = _require_integer(
        p, "min_diag", minimum=-MAX_I32, maximum=MAX_I32
    )
    max_diag = _require_integer(
        p, "max_diag", minimum=-MAX_I32, maximum=MAX_I32
    )
    if min_diag > max_diag:
        raise ValueError("SDPA diagonal interval is empty")
    banded = _require_flag(p, "banded")
    if banded != (
        min_diag != -UNBOUNDED_DIAGONAL or max_diag != UNBOUNDED_DIAGONAL
    ):
        raise ValueError("parameters.banded disagrees with diagonal bounds")
    causal_top_left = _require_flag(p, "causal_top_left")
    reverse_causal = _require_flag(p, "reverse_causal")
    generate_stats = _require_flag(p, "generate_stats")
    if (
        operation in ("sdpa_backward", "sdpa_fp8_backward")
        and not generate_stats
    ):
        raise ValueError("SDPA backward requires generated forward stats")
    attn_scale = _require_number(p, "attn_scale")

    if fp8:
        scalar_names = (
            (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_s",
                "scale_s",
                "scale_o",
            )
            if operation == "sdpa_fp8"
            else (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_o",
                "descale_do",
                "descale_s",
                "descale_dp",
                "scale_s",
                "scale_dq",
                "scale_dk",
                "scale_dv",
                "scale_dp",
            )
        )
        for name in scalar_names:
            _validate_fp32_scalar(ports[name], f"FP8 SDPA {name}")
        amax_names = (
            ("amax_s", "amax_o")
            if operation == "sdpa_fp8"
            else ("amax_dq", "amax_dk", "amax_dv", "amax_dp")
        )
        for name in amax_names:
            _validate_fp32_scalar(ports[name], f"FP8 SDPA {name}")
    if operation == "sdpa_fp8_backward" and (
        shape["key_heads"] != shape["value_heads"] or d != dv or d > 128
    ):
        raise ValueError(
            "FP8 SDPA backward requires matching K/V heads and D == V <= 128"
        )
    return {
        **shape,
        "has_bias": has_bias,
        "has_dbias": has_dbias,
        "min_diag": min_diag,
        "max_diag": max_diag,
        "banded": banded,
        "causal_top_left": causal_top_left,
        "reverse_causal": reverse_causal,
        "generate_stats": generate_stats,
        "attn_scale": attn_scale,
    }


_ATTENTION_BACKWARD_RUNTIME_SCALARS = (
    ("attn_scale", "fp32"),
    ("SQ", "i32"),
    ("SKV", "i32"),
    ("min_diag", "i32"),
    ("max_diag", "i32"),
)


def _attention_forward_scalar_arguments(
    *, fp8: bool
) -> tuple[ScalarArgument, ...]:
    scale_name = "attn_scale" if fp8 else "qk_scale"
    return ((scale_name, "fp32"),) + tuple(
        (name, "i32") for name in _ATTENTION_FORWARD_I32_SCALARS
    )


def _attention_bias_strides(
    node: Mapping[str, Any], d: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    if not bool(d["has_bias"]):
        return (0, 0, 0, 0)
    bias = node["port_tensors"]["bias"]
    dims = list(bias["dimensions"])
    strides = list(bias["strides"])
    return (
        0 if dims[0] == 1 else strides[0],
        0 if dims[1] == 1 else strides[1],
        strides[2],
        strides[3],
    )


def _attention_forward_constants(
    node: Mapping[str, Any], *, fp8: bool
) -> dict[str, int | float | bool]:
    d = _require_object(node["derived"], "node.derived")
    q = node["port_tensors"]["q"]
    k = node["port_tensors"]["k"]
    v = node["port_tensors"]["v"]
    o = node["port_tensors"]["o"]
    stats = node["port_tensors"]["stats"]
    bias_strides = _attention_bias_strides(node, d)
    head_dimension = int(d["head_dimension"])
    value_dimension = int(d["value_dimension"])
    constants: dict[str, int | float | bool] = {
        ("attn_scale" if fp8 else "qk_scale"): (
            float(d["attn_scale"]) if fp8 else float(d["attn_scale"]) * LOG2_E
        ),
        "HQ": int(d["heads"]),
        "SQ": int(d["sequence_q"]),
        "SKV": int(d["sequence_kv"]),
        "q_per_k": int(d["q_per_k"]),
        "q_per_v": int(d["q_per_v"]),
        "min_diag": int(d["min_diag"]),
        "max_diag": int(d["max_diag"]),
        "stride_qb": q["strides"][0],
        "stride_qh": q["strides"][1],
        "stride_qm": q["strides"][2],
        "stride_qd": q["strides"][3],
        "stride_kb": k["strides"][0],
        "stride_kh": k["strides"][1],
        "stride_kn": k["strides"][2],
        "stride_kd": k["strides"][3],
        "stride_vb": v["strides"][0],
        "stride_vh": v["strides"][1],
        "stride_vn": v["strides"][2],
        "stride_vd": v["strides"][3],
        "stride_bias_b": bias_strides[0],
        "stride_bias_h": bias_strides[1],
        "stride_bias_m": bias_strides[2],
        "stride_bias_n": bias_strides[3],
        "stride_ob": o["strides"][0],
        "stride_oh": o["strides"][1],
        "stride_om": o["strides"][2],
        "stride_od": o["strides"][3],
        "stride_sb": stats["strides"][0],
        "stride_sh": stats["strides"][1],
        "stride_sm": stats["strides"][2],
        "HEAD_DIM": head_dimension,
        "V_DIM": value_dimension,
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_D": _attention_full_dimension_block(head_dimension),
        "BLOCK_DV": _attention_full_dimension_block(value_dimension),
        "HAS_BIAS": bool(d["has_bias"]),
        "BANDED": bool(d["banded"]),
        "GENERATE_STATS": bool(d["generate_stats"]),
        "REVERSE_CAUSAL": bool(d["reverse_causal"]),
    }
    if not fp8:
        constants["ELEM_SIZE"] = ELEMENT_SIZES[str(d["data_type"])]
    return constants


def _attention_forward_stage(
    node: Mapping[str, Any],
    *,
    dependencies: tuple[str, ...] = (),
) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "stats_ptr", "stats"),
    )
    return _make_stage(
        operation=operation,
        stage_name="forward",
        function_name="_sdpa_fwd_kernel",
        pointer_arguments=pointers,
        constants=_attention_forward_constants(node, fp8=False),
        scalar_arguments=_attention_forward_scalar_arguments(fp8=False),
        tuning=SDPA_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fwd",
            (
                int(d["sequence_q"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=dependencies,
        num_stages=2,
    )


def _fp8_forward_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = "sdpa_fp8"
    d = _require_object(node["derived"], "node.derived")
    zero = _make_stage(
        operation=operation,
        stage_name="zero_amax",
        function_name="_zero_sdpa_fp8_fwd_amax_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "amax_s_ptr", "amax_s"),
            _tensor_pointer(node, "amax_o_ptr", "amax_o"),
        ),
        constants={},
        tuning=INTERNAL_TUNING,
        tuning_key_value=1,
        grid_spec=GridSpec("fixed", (1, 1, 1)),
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "amax_s_ptr", "amax_s"),
        _tensor_pointer(node, "amax_o_ptr", "amax_o"),
        _tensor_pointer(node, "descale_q_ptr", "descale_q"),
        _tensor_pointer(node, "descale_k_ptr", "descale_k"),
        _tensor_pointer(node, "descale_v_ptr", "descale_v"),
        _tensor_pointer(node, "descale_s_ptr", "descale_s"),
        _tensor_pointer(node, "scale_s_ptr", "scale_s"),
        _tensor_pointer(node, "scale_o_ptr", "scale_o"),
    )
    forward = _make_stage(
        operation=operation,
        stage_name="forward",
        function_name="_sdpa_fp8_fwd_kernel",
        pointer_arguments=pointers,
        constants=_attention_forward_constants(node, fp8=True),
        scalar_arguments=_attention_forward_scalar_arguments(fp8=True),
        tuning=SDPA_FP8_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fp8_fwd",
            (
                int(d["sequence_q"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=("zero_amax",),
        num_stages=2,
    )
    plan = NodePlan(operation, (zero, forward))
    plan.validate_dependencies()
    return plan


def _attention_backward_base(
    node: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    d = _require_object(node["derived"], "node.derived")
    q = node["port_tensors"]["q"]
    k = node["port_tensors"]["k"]
    v = node["port_tensors"]["v"]
    do = node["port_tensors"]["do"]
    stats = node["port_tensors"]["stats"]
    bias_strides = _attention_bias_strides(node, d)
    return {
        "SQ": int(d["sequence_q"]),
        "SKV": int(d["sequence_kv"]),
        "min_diag": int(d["min_diag"]),
        "max_diag": int(d["max_diag"]),
        "stride_qb": q["strides"][0],
        "stride_qh": q["strides"][1],
        "stride_qm": q["strides"][2],
        "stride_qd": q["strides"][3],
        "stride_kb": k["strides"][0],
        "stride_kh": k["strides"][1],
        "stride_kn": k["strides"][2],
        "stride_kd": k["strides"][3],
        "stride_vb": v["strides"][0],
        "stride_vh": v["strides"][1],
        "stride_vn": v["strides"][2],
        "stride_vd": v["strides"][3],
        "stride_bias_b": bias_strides[0],
        "stride_bias_h": bias_strides[1],
        "stride_bias_m": bias_strides[2],
        "stride_bias_n": bias_strides[3],
        "stride_dob": do["strides"][0],
        "stride_doh": do["strides"][1],
        "stride_dom": do["strides"][2],
        "stride_dod": do["strides"][3],
        "stride_sb": stats["strides"][0],
        "stride_sh": stats["strides"][1],
        "stride_sm": stats["strides"][2],
        "HEAD_DIM": int(d["head_dimension"]),
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_D_FULL": _attention_full_dimension_block(
            int(d["head_dimension"])
        ),
        # The common backward kernels iterate a full BLOCK_M query tile and
        # require masking for tail query blocks, even for non-banded attention.
        "FULL_ATTENTION": False,
        "HAS_BIAS": bool(d["has_bias"]),
        "BANDED": bool(d["banded"]),
        "CAUSAL_TOP_LEFT": bool(d["causal_top_left"]),
    }


def _attention_backward_dq_stage(
    node: Mapping[str, Any],
    dependencies: tuple[str, ...],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    o = node["port_tensors"]["o"]
    dq = node["port_tensors"]["dq"]
    delta_strides = (
        int(d["heads"]) * int(d["sequence_q"]),
        int(d["sequence_q"]),
        1,
    )
    has_dbias = bool(d["has_dbias"])
    dbias = node["port_tensors"].get("dbias")
    dbias_reduce = bool(
        has_dbias
        and (
            dbias["dimensions"][0] != d["batch"]
            or dbias["dimensions"][1] != d["heads"]
        )
    )
    dbias_strides = list(dbias["strides"]) if has_dbias else [0, 0, 0, 0]
    constants = _attention_backward_base(node)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HQ": int(d["heads"]),
            "q_per_k": int(d["q_per_k"]),
            "q_per_v": int(d["q_per_v"]),
            "stride_ob": o["strides"][0],
            "stride_oh": o["strides"][1],
            "stride_om": o["strides"][2],
            "stride_od": o["strides"][3],
            "stride_delta_b": delta_strides[0],
            "stride_delta_h": delta_strides[1],
            "stride_delta_m": delta_strides[2],
            "stride_dqb": dq["strides"][0],
            "stride_dqh": dq["strides"][1],
            "stride_dqm": dq["strides"][2],
            "stride_dqd": dq["strides"][3],
            "stride_dbias_b": dbias_strides[0],
            "stride_dbias_h": dbias_strides[1],
            "stride_dbias_m": dbias_strides[2],
            "stride_dbias_n": dbias_strides[3],
            "V_DIM": int(d["value_dimension"]),
            "DBIAS_BATCHES": dbias["dimensions"][0] if has_dbias else 1,
            "DBIAS_HEADS": dbias["dimensions"][1] if has_dbias else 1,
            "BLOCK_D_OUT": _attention_output_dimension_block(
                int(d["head_dimension"])
            ),
            "BLOCK_DV": _attention_full_dimension_block(
                int(d["value_dimension"])
            ),
            "HAS_DBIAS": has_dbias,
            "DBIAS_REDUCE": dbias_reduce,
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    dbias_port = "dbias" if has_dbias else "dq"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _workspace_pointer("delta_ptr", "delta"),
        _tensor_pointer(node, "dq_ptr", "dq"),
        _tensor_pointer(node, "dbias_ptr", dbias_port),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dq_delta_dbias",
        function_name="_sdpa_bwd_dq_dbias_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dq",
            (
                int(d["sequence_q"]),
                int(d["head_dimension"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=dependencies,
        num_stages=2,
    )


def _attention_backward_dkdv_stage(
    node: Mapping[str, Any],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dk = node["port_tensors"]["dk"]
    dv = node["port_tensors"]["dv"]
    delta_strides = (
        int(d["heads"]) * int(d["sequence_q"]),
        int(d["sequence_q"]),
        1,
    )
    constants = _attention_backward_base(node)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HKV": int(d["key_heads"]),
            "stride_delta_b": delta_strides[0],
            "stride_delta_h": delta_strides[1],
            "stride_delta_m": delta_strides[2],
            "stride_dkb": dk["strides"][0],
            "stride_dkh": dk["strides"][1],
            "stride_dkn": dk["strides"][2],
            "stride_dkd": dk["strides"][3],
            "stride_dvb": dv["strides"][0],
            "stride_dvh": dv["strides"][1],
            "stride_dvn": dv["strides"][2],
            "stride_dvd": dv["strides"][3],
            "Q_PER": int(d["q_per_k"]),
            "BLOCK_D_OUT": _attention_output_dimension_block(
                int(d["head_dimension"])
            ),
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _workspace_pointer("delta_ptr", "delta"),
        _tensor_pointer(node, "dk_ptr", "dk"),
        _tensor_pointer(node, "dv_ptr", "dv"),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dk_dv",
        function_name="_sdpa_bwd_dkdv_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dk",
            (
                int(d["sequence_kv"]),
                int(d["head_dimension"]),
                int(d["batch"]) * int(d["key_heads"]),
            ),
        ),
        dependencies=("dq_delta_dbias",),
        num_stages=2,
    )


def _attention_backward_dk_stage(
    node: Mapping[str, Any],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dk = node["port_tensors"]["dk"]
    delta_strides = (
        int(d["heads"]) * int(d["sequence_q"]),
        int(d["sequence_q"]),
        1,
    )
    constants = _attention_backward_base(node)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HKV": int(d["key_heads"]),
            "stride_delta_b": delta_strides[0],
            "stride_delta_h": delta_strides[1],
            "stride_delta_m": delta_strides[2],
            "stride_dkb": dk["strides"][0],
            "stride_dkh": dk["strides"][1],
            "stride_dkn": dk["strides"][2],
            "stride_dkd": dk["strides"][3],
            "V_DIM": int(d["value_dimension"]),
            "Q_PER": int(d["q_per_k"]),
            "BLOCK_D_OUT": _attention_output_dimension_block(
                int(d["head_dimension"])
            ),
            "BLOCK_DV": _attention_full_dimension_block(
                int(d["value_dimension"])
            ),
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _workspace_pointer("delta_ptr", "delta"),
        _tensor_pointer(node, "dk_ptr", "dk"),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dk",
        function_name="_sdpa_bwd_dk_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dk",
            (
                int(d["sequence_kv"]),
                int(d["head_dimension"]),
                int(d["batch"]) * int(d["key_heads"]),
            ),
        ),
        dependencies=("dq_delta_dbias",),
        num_stages=2,
    )


def _attention_backward_dv_stage(
    node: Mapping[str, Any],
) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dv = node["port_tensors"]["dv"]
    constants = _attention_backward_base(node)
    # The DV-only registry entry neither receives V nor the hidden delta.
    for name in (
        "stride_vb",
        "stride_vh",
        "stride_vn",
        "stride_vd",
        "BLOCK_D_FULL",
    ):
        constants.pop(name)
    constants.update(
        {
            "attn_scale": float(d["attn_scale"]),
            "HKV": int(d["value_heads"]),
            "stride_dvb": dv["strides"][0],
            "stride_dvh": dv["strides"][1],
            "stride_dvn": dv["strides"][2],
            "stride_dvd": dv["strides"][3],
            "V_DIM": int(d["value_dimension"]),
            "Q_PER": int(d["q_per_v"]),
            "BLOCK_D_FULL": _attention_full_dimension_block(
                int(d["head_dimension"])
            ),
            "BLOCK_DV_OUT": _attention_output_dimension_block(
                int(d["value_dimension"])
            ),
        }
    )
    bias_port = "bias" if bool(d["has_bias"]) else "q"
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "bias_ptr", bias_port),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "dv_ptr", "dv"),
    )
    return _make_stage(
        operation="sdpa_backward",
        stage_name="dv",
        function_name="_sdpa_bwd_dv_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_dv",
            (
                int(d["sequence_kv"]),
                int(d["value_dimension"]),
                int(d["batch"]) * int(d["value_heads"]),
            ),
        ),
        dependencies=("dk",),
        num_stages=2,
    )


def _attention_backward_plan(node: Mapping[str, Any]) -> NodePlan:
    d = _require_object(node["derived"], "node.derived")
    stages: list[KernelStagePlan] = []
    dq_dependencies: tuple[str, ...] = ()
    if bool(d["has_dbias"]):
        dbias = node["port_tensors"]["dbias"]
        reduces = (
            dbias["dimensions"][0] != d["batch"]
            or dbias["dimensions"][1] != d["heads"]
        )
        if reduces:
            if not _is_contiguous(dbias):
                raise ValueError(
                    "broadcast dbias must be contiguous because the registry "
                    "only declares _zero_contiguous_kernel"
                )
            dbias_elements = _checked_product(
                list(dbias["dimensions"]), "SDPA dbias elements"
            )
            stages.append(
                _make_stage(
                    operation="sdpa_backward",
                    stage_name="zero_dbias",
                    function_name="_zero_contiguous_kernel",
                    pointer_arguments=(_tensor_pointer(node, "ptr", "dbias"),),
                    constants={"n_elements": dbias_elements, "BLOCK": 256},
                    scalar_arguments=(("n_elements", "i32"),),
                    tuning=INTERNAL_TUNING,
                    tuning_key_value=dbias_elements,
                    grid_spec=GridSpec("zero", (dbias_elements,)),
                )
            )
            dq_dependencies = ("zero_dbias",)
    stages.append(_attention_backward_dq_stage(node, dq_dependencies))
    if (
        d["key_heads"] == d["value_heads"]
        and d["head_dimension"] == d["value_dimension"]
    ):
        stages.append(_attention_backward_dkdv_stage(node))
    else:
        stages.append(_attention_backward_dk_stage(node))
        stages.append(_attention_backward_dv_stage(node))

    delta_dimensions = (
        int(d["batch"]),
        int(d["heads"]),
        int(d["sequence_q"]),
    )
    delta_strides = (
        delta_dimensions[1] * delta_dimensions[2],
        delta_dimensions[2],
        1,
    )
    raw_size = 4 * _checked_product(delta_dimensions, "SDPA delta elements")
    workspace_size = (
        _ceil_div(raw_size, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT
    )
    workspace = WorkspaceTensor(
        name="delta",
        data_type="float32",
        dimensions=delta_dimensions,
        strides=delta_strides,
        offset=0,
        size=raw_size,
    )
    plan = NodePlan(
        "sdpa_backward", tuple(stages), (workspace,), workspace_size
    )
    plan.validate_dependencies()
    return plan


def _fp8_backward_base(
    node: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    d = _require_object(node["derived"], "node.derived")
    q = node["port_tensors"]["q"]
    k = node["port_tensors"]["k"]
    v = node["port_tensors"]["v"]
    o = node["port_tensors"]["o"]
    do = node["port_tensors"]["do"]
    stats = node["port_tensors"]["stats"]
    return {
        "attn_scale": float(d["attn_scale"]),
        "SQ": int(d["sequence_q"]),
        "SKV": int(d["sequence_kv"]),
        "min_diag": int(d["min_diag"]),
        "max_diag": int(d["max_diag"]),
        "stride_qb": q["strides"][0],
        "stride_qh": q["strides"][1],
        "stride_qm": q["strides"][2],
        "stride_qd": q["strides"][3],
        "stride_kb": k["strides"][0],
        "stride_kh": k["strides"][1],
        "stride_kn": k["strides"][2],
        "stride_kd": k["strides"][3],
        "stride_vb": v["strides"][0],
        "stride_vh": v["strides"][1],
        "stride_vn": v["strides"][2],
        "stride_vd": v["strides"][3],
        "stride_ob": o["strides"][0],
        "stride_oh": o["strides"][1],
        "stride_om": o["strides"][2],
        "stride_od": o["strides"][3],
        "stride_dob": do["strides"][0],
        "stride_doh": do["strides"][1],
        "stride_dom": do["strides"][2],
        "stride_dod": do["strides"][3],
        "stride_sb": stats["strides"][0],
        "stride_sh": stats["strides"][1],
        "stride_sm": stats["strides"][2],
        "HEAD_DIM": int(d["head_dimension"]),
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_D": _attention_full_dimension_block(int(d["head_dimension"])),
        "BANDED": bool(d["banded"]),
        # BLOCK_M/BLOCK_N are autotuned and Q/KV/D may have tail tiles.  The
        # common kernel's FULL_BLOCKS path performs unmasked memory accesses,
        # so it is legal only with a proof the planner cannot provide here.
        "FULL_BLOCKS": False,
        "CAUSAL_TOP_LEFT": bool(d["causal_top_left"]),
    }


def _fp8_backward_dq_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dq = node["port_tensors"]["dq"]
    constants = _fp8_backward_base(node)
    constants.update(
        {
            "HQ": int(d["heads"]),
            "q_per_k": int(d["q_per_k"]),
            "q_per_v": int(d["q_per_v"]),
            "stride_dqb": dq["strides"][0],
            "stride_dqh": dq["strides"][1],
            "stride_dqm": dq["strides"][2],
            "stride_dqd": dq["strides"][3],
        }
    )
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "dq_ptr", "dq"),
        _tensor_pointer(node, "amax_dq_ptr", "amax_dq"),
        _tensor_pointer(node, "descale_q_ptr", "descale_q"),
        _tensor_pointer(node, "descale_k_ptr", "descale_k"),
        _tensor_pointer(node, "descale_v_ptr", "descale_v"),
        _tensor_pointer(node, "descale_o_ptr", "descale_o"),
        _tensor_pointer(node, "descale_do_ptr", "descale_do"),
        _tensor_pointer(node, "descale_dp_ptr", "descale_dp"),
        _tensor_pointer(node, "scale_dq_ptr", "scale_dq"),
        _tensor_pointer(node, "scale_dp_ptr", "scale_dp"),
    )
    return _make_stage(
        operation="sdpa_fp8_backward",
        stage_name="dq",
        function_name="_sdpa_fp8_bwd_dq_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=(
            ("attn_scale", "fp32"),
            ("HQ", "i32"),
            ("SQ", "i32"),
            ("SKV", "i32"),
            ("min_diag", "i32"),
            ("max_diag", "i32"),
        ),
        tuning=SDPA_FP8_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fp8_dq",
            (
                int(d["sequence_q"]),
                int(d["batch"]) * int(d["heads"]),
            ),
        ),
        dependencies=("zero_amax",),
        num_stages=2,
    )


def _fp8_backward_dkdv_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    d = _require_object(node["derived"], "node.derived")
    dk = node["port_tensors"]["dk"]
    dv = node["port_tensors"]["dv"]
    constants = _fp8_backward_base(node)
    constants.update(
        {
            "HKV": int(d["key_heads"]),
            "stride_dkb": dk["strides"][0],
            "stride_dkh": dk["strides"][1],
            "stride_dkn": dk["strides"][2],
            "stride_dkd": dk["strides"][3],
            "stride_dvb": dv["strides"][0],
            "stride_dvh": dv["strides"][1],
            "stride_dvn": dv["strides"][2],
            "stride_dvd": dv["strides"][3],
            "Q_PER": int(d["q_per_k"]),
        }
    )
    pointers = (
        _tensor_pointer(node, "q_ptr", "q"),
        _tensor_pointer(node, "k_ptr", "k"),
        _tensor_pointer(node, "v_ptr", "v"),
        _tensor_pointer(node, "o_ptr", "o"),
        _tensor_pointer(node, "do_ptr", "do"),
        _tensor_pointer(node, "stats_ptr", "stats"),
        _tensor_pointer(node, "dk_ptr", "dk"),
        _tensor_pointer(node, "dv_ptr", "dv"),
        _tensor_pointer(node, "amax_dk_ptr", "amax_dk"),
        _tensor_pointer(node, "amax_dv_ptr", "amax_dv"),
        _tensor_pointer(node, "amax_dp_ptr", "amax_dp"),
        _tensor_pointer(node, "descale_q_ptr", "descale_q"),
        _tensor_pointer(node, "descale_k_ptr", "descale_k"),
        _tensor_pointer(node, "descale_v_ptr", "descale_v"),
        _tensor_pointer(node, "descale_o_ptr", "descale_o"),
        _tensor_pointer(node, "descale_do_ptr", "descale_do"),
        _tensor_pointer(node, "descale_s_ptr", "descale_s"),
        _tensor_pointer(node, "descale_dp_ptr", "descale_dp"),
        _tensor_pointer(node, "scale_s_ptr", "scale_s"),
        _tensor_pointer(node, "scale_dk_ptr", "scale_dk"),
        _tensor_pointer(node, "scale_dv_ptr", "scale_dv"),
        _tensor_pointer(node, "scale_dp_ptr", "scale_dp"),
    )
    return _make_stage(
        operation="sdpa_fp8_backward",
        stage_name="dk_dv",
        function_name="_sdpa_fp8_bwd_dkdv_kernel",
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=_ATTENTION_BACKWARD_RUNTIME_SCALARS,
        tuning=SDPA_FP8_BACKWARD_TUNING,
        tuning_key_value=int(d["sequence_q"]),
        grid_spec=GridSpec(
            "sdpa_fp8_dkdv",
            (
                int(d["sequence_kv"]),
                int(d["batch"]) * int(d["key_heads"]),
            ),
        ),
        dependencies=("dq",),
        num_stages=2,
    )


def _fp8_backward_plan(node: Mapping[str, Any]) -> NodePlan:
    zero = _make_stage(
        operation="sdpa_fp8_backward",
        stage_name="zero_amax",
        function_name="_zero_sdpa_fp8_bwd_amax_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "amax_dq_ptr", "amax_dq"),
            _tensor_pointer(node, "amax_dk_ptr", "amax_dk"),
            _tensor_pointer(node, "amax_dv_ptr", "amax_dv"),
            _tensor_pointer(node, "amax_dp_ptr", "amax_dp"),
        ),
        constants={},
        tuning=INTERNAL_TUNING,
        tuning_key_value=1,
        grid_spec=GridSpec("fixed", (1, 1, 1)),
    )
    plan = NodePlan(
        "sdpa_fp8_backward",
        (zero, _fp8_backward_dq_stage(node), _fp8_backward_dkdv_stage(node)),
    )
    plan.validate_dependencies()
    return plan


LOG2_E = 1.4426950408889634


_ATTENTION_FORWARD_I32_SCALARS = (
    "HQ",
    "SQ",
    "SKV",
    "q_per_k",
    "q_per_v",
    "min_diag",
    "max_diag",
    "stride_qb",
    "stride_qh",
    "stride_qm",
    "stride_qd",
    "stride_kb",
    "stride_kh",
    "stride_kn",
    "stride_kd",
    "stride_vb",
    "stride_vh",
    "stride_vn",
    "stride_vd",
    "stride_bias_b",
    "stride_bias_h",
    "stride_bias_m",
    "stride_bias_n",
    "stride_ob",
    "stride_oh",
    "stride_om",
    "stride_od",
    "stride_sb",
    "stride_sh",
    "stride_sm",
)
