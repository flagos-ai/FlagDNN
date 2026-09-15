"""Forward attention and FP8 attention kernel configurations."""

from __future__ import annotations

from typing import Any

from .attention_common import (
    _attention_broadcast_strides,
    _attention_flag,
    _attention_runtime_f32,
    _attention_runtime_i32,
    _attention_strides,
    _require_fp8_scale_tensor,
    _validate_attention_base,
)
from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _ceil_div,
    _next_power_of_two,
    _require_integer,
    _require_number,
)


def _sdpa_forward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    has_bias = _attention_flag(parameters, "has_bias")
    if len(tensors) != (6 if has_bias else 5):
        raise ValueError("SDPA forward tensor count is invalid")
    (
        batch,
        heads,
        _,
        _,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors)
    q, k, v = tensors[:3]
    bias_index = 3 if has_bias else None
    output_index = 4 if has_bias else 3
    stats_index = output_index + 1
    output = tensors[output_index]
    stats = tensors[stats_index]
    if (
        output["data_type"] != q["data_type"]
        or output["dimensions"] != [batch, heads, sequence_q, value_dimension]
        or stats["data_type"] != "float32"
        or stats["dimensions"] != [batch, heads, sequence_q, 1]
    ):
        raise ValueError("SDPA forward output metadata is invalid")
    if has_bias:
        bias = tensors[bias_index]
        if (
            bias["data_type"] != q["data_type"]
            or bias["dimensions"][0] not in {1, batch}
            or bias["dimensions"][1] not in {1, heads}
            or bias["dimensions"][2:] != [sequence_q, sequence_kv]
        ):
            raise ValueError("SDPA bias metadata is invalid")
        bias_strides = _attention_broadcast_strides(bias, "bias_")
    else:
        bias_strides = {
            "stride_bias_b": 0,
            "stride_bias_h": 0,
            "stride_bias_m": 0,
            "stride_bias_n": 0,
        }

    runtime_values: dict[str, int | float] = {
        "qk_scale": _require_number(parameters, "attn_scale")
        * 1.4426950408889634,
        "HQ": heads,
        "SQ": sequence_q,
        "SKV": sequence_kv,
        "q_per_k": _require_integer(parameters, "q_per_k"),
        "q_per_v": _require_integer(parameters, "q_per_v"),
        "min_diag": _require_integer(
            parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        "max_diag": _require_integer(
            parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        **_attention_strides(q, "q", "bhmd"),
        **_attention_strides(k, "k", "bhnd"),
        **_attention_strides(v, "v", "bhnd"),
        **bias_strides,
        **_attention_strides(output, "o", "bhmd"),
        **_attention_strides(stats, "s", "bhm"),
    }
    for name, value in runtime_values.items():
        if isinstance(value, float):
            _attention_runtime_f32(parameters, name, value)
        else:
            _attention_runtime_i32(parameters, name, value)

    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    runtime_signature: dict[str, str] = {
        "q_ptr": pointer_type,
        "k_ptr": pointer_type,
        "v_ptr": pointer_type,
        "bias_ptr": pointer_type,
        "o_ptr": pointer_type,
        "stats_ptr": TRITON_POINTER_TYPES["float32"],
        "qk_scale": "fp32",
    }
    for name in (
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
    ):
        runtime_signature[name] = "i32"

    block_m = 64
    constants: dict[str, int | float | str | bool] = {
        "HEAD_DIM": head_dimension,
        "V_DIM": value_dimension,
        "ELEM_SIZE": 4 if q["data_type"] == "float32" else 2,
        "BLOCK_M": block_m,
        "BLOCK_N": 64,
        "BLOCK_D": max(16, _next_power_of_two(head_dimension)),
        "BLOCK_DV": max(16, _next_power_of_two(value_dimension)),
        "HAS_BIAS": has_bias,
        "BANDED": _attention_flag(parameters, "banded"),
        "GENERATE_STATS": _attention_flag(parameters, "generate_stats"),
        "REVERSE_CAUSAL": _attention_flag(parameters, "reverse_causal"),
    }
    pointer_layout = [
        ("tensor_alias", 0),
        ("tensor_alias", 1),
        ("tensor_alias", 2),
        ("tensor_alias", bias_index if bias_index is not None else 0),
        ("tensor_alias", output_index),
        ("tensor_alias", stats_index),
        ("scalar_f32", "qk_scale"),
    ]
    scalar_layout = [
        ("scalar_i32", name) for name in list(runtime_signature)[7:]
    ]
    return (
        "_sdpa_fwd_kernel",
        runtime_signature,
        constants,
        (_ceil_div(sequence_q, block_m), batch * heads, 1),
        pointer_layout + scalar_layout,
    )


def _sdpa_fp8_forward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_sdpa_fp8_stage")
    if stage == "zero_amax":
        if len(tensors) != 2:
            raise ValueError(
                "FP8 SDPA amax zero stage tensor count is invalid"
            )
        _require_fp8_scale_tensor(tensors[0], "FP8 SDPA amax S")
        _require_fp8_scale_tensor(tensors[1], "FP8 SDPA amax O")
        return (
            "_zero_sdpa_fp8_fwd_amax_kernel",
            {
                "amax_s_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_o_ptr": TRITON_POINTER_TYPES["float32"],
            },
            {},
            (1, 1, 1),
            [("tensor_alias", 0), ("tensor_alias", 1)],
        )

    if stage != "forward":
        raise ValueError("FP8 SDPA forward pipeline stage is invalid")
    has_bias = _attention_flag(parameters, "has_bias")
    if len(tensors) != 13 + int(has_bias):
        raise ValueError("FP8 SDPA forward tensor count is invalid")
    (
        batch,
        heads,
        _,
        _,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors, fp8=True)
    q, k, v = tensors[:3]
    for index, name in enumerate(
        (
            "descale Q",
            "descale K",
            "descale V",
            "descale S",
            "scale S",
            "scale O",
        ),
        start=3,
    ):
        _require_fp8_scale_tensor(tensors[index], f"FP8 SDPA {name}")

    bias_index = 9 if has_bias else None
    output_index = 10 if has_bias else 9
    stats_index = output_index + 1
    amax_s_index = output_index + 2
    amax_o_index = output_index + 3
    output = tensors[output_index]
    stats = tensors[stats_index]
    if (
        output["data_type"] != q["data_type"]
        or output["dimensions"] != [batch, heads, sequence_q, value_dimension]
        or stats["data_type"] != "float32"
        or stats["dimensions"] != [batch, heads, sequence_q, 1]
    ):
        raise ValueError("FP8 SDPA forward output metadata is invalid")
    _require_fp8_scale_tensor(tensors[amax_s_index], "FP8 SDPA amax S")
    _require_fp8_scale_tensor(tensors[amax_o_index], "FP8 SDPA amax O")

    if has_bias:
        bias = tensors[bias_index]
        if (
            bias["data_type"] not in FLOAT_DATA_TYPES
            or len(bias["dimensions"]) != 4
            or bias["dimensions"][0] not in {1, batch}
            or bias["dimensions"][1] not in {1, heads}
            or bias["dimensions"][2:] != [sequence_q, sequence_kv]
        ):
            raise ValueError("FP8 SDPA bias metadata is invalid")
        bias_strides = _attention_broadcast_strides(bias, "bias_")
        bias_pointer_type = TRITON_POINTER_TYPES[bias["data_type"]]
    else:
        bias_strides = {
            "stride_bias_b": 0,
            "stride_bias_h": 0,
            "stride_bias_m": 0,
            "stride_bias_n": 0,
        }
        bias_pointer_type = TRITON_POINTER_TYPES[q["data_type"]]

    runtime_values: dict[str, int | float] = {
        "attn_scale": _require_number(parameters, "attn_scale"),
        "HQ": heads,
        "SQ": sequence_q,
        "SKV": sequence_kv,
        "q_per_k": _require_integer(parameters, "q_per_k"),
        "q_per_v": _require_integer(parameters, "q_per_v"),
        "min_diag": _require_integer(
            parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        "max_diag": _require_integer(
            parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        **_attention_strides(q, "q", "bhmd"),
        **_attention_strides(k, "k", "bhnd"),
        **_attention_strides(v, "v", "bhnd"),
        **bias_strides,
        **_attention_strides(output, "o", "bhmd"),
        **_attention_strides(stats, "s", "bhm"),
    }
    for name, value in runtime_values.items():
        if isinstance(value, float):
            _attention_runtime_f32(parameters, name, value)
        else:
            _attention_runtime_i32(parameters, name, value)

    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    runtime_signature: dict[str, str] = {
        "q_ptr": pointer_type,
        "k_ptr": pointer_type,
        "v_ptr": pointer_type,
        "bias_ptr": bias_pointer_type,
        "o_ptr": pointer_type,
        "stats_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_s_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_o_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_q_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_k_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_v_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_o_ptr": TRITON_POINTER_TYPES["float32"],
        "attn_scale": "fp32",
    }
    integer_runtime_names = (
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
    for name in integer_runtime_names:
        runtime_signature[name] = "i32"

    block_m = 64
    constants: dict[str, int | float | str | bool] = {
        "HEAD_DIM": head_dimension,
        "V_DIM": value_dimension,
        "BLOCK_M": block_m,
        "BLOCK_N": 128,
        "BLOCK_D": max(16, _next_power_of_two(head_dimension)),
        "BLOCK_DV": max(16, _next_power_of_two(value_dimension)),
        "HAS_BIAS": has_bias,
        "BANDED": _attention_flag(parameters, "banded"),
        "GENERATE_STATS": _attention_flag(parameters, "generate_stats"),
        "REVERSE_CAUSAL": _attention_flag(parameters, "reverse_causal"),
    }
    pointer_layout: list[tuple[str, str | int | None]] = [
        ("tensor_alias", 0),
        ("tensor_alias", 1),
        ("tensor_alias", 2),
        ("tensor_alias", bias_index if bias_index is not None else 0),
        ("tensor_alias", output_index),
        ("tensor_alias", stats_index),
        ("tensor_alias", amax_s_index),
        ("tensor_alias", amax_o_index),
        *[("tensor_alias", index) for index in range(3, 9)],
        ("scalar_f32", "attn_scale"),
    ]
    scalar_layout = [("scalar_i32", name) for name in integer_runtime_names]
    return (
        "_sdpa_fp8_fwd_kernel",
        runtime_signature,
        constants,
        (_ceil_div(sequence_q, block_m), batch * heads, 1),
        pointer_layout + scalar_layout,
    )
