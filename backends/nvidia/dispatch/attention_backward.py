"""Backward attention and FP8 attention kernel configurations."""

from __future__ import annotations

from typing import Any
import math

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


def _sdpa_fp8_backward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_sdpa_fp8_bwd_stage")
    if stage == "zero_amax":
        if len(tensors) != 4:
            raise ValueError(
                "FP8 SDPA backward amax zero stage tensor count is invalid"
            )
        for tensor, name in zip(
            tensors, ("amax dQ", "amax dK", "amax dV", "amax dP")
        ):
            _require_fp8_scale_tensor(tensor, f"FP8 SDPA backward {name}")
        return (
            "_zero_sdpa_fp8_bwd_amax_kernel",
            {
                "amax_dq_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_dk_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_dv_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_dp_ptr": TRITON_POINTER_TYPES["float32"],
            },
            {},
            (1, 1, 1),
            [("tensor_alias", index) for index in range(4)],
        )

    if stage not in {"dq", "dkdv"}:
        raise ValueError("FP8 SDPA backward pipeline stage is invalid")
    expected_count = 16 if stage == "dq" else 22
    if len(tensors) != expected_count:
        raise ValueError(
            f"FP8 SDPA backward {stage} stage tensor count is invalid"
        )
    (
        batch,
        heads,
        key_heads,
        value_heads,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors, fp8=True)
    if key_heads != value_heads or head_dimension != value_dimension:
        raise ValueError("FP8 SDPA backward requires matching K/V heads and D")
    if head_dimension > 128:
        raise ValueError("FP8 SDPA backward head dimension exceeds 128")

    q, k, v, output, doutput, stats = tensors[:6]
    expected_output = [batch, heads, sequence_q, value_dimension]
    if (
        output["dimensions"] != expected_output
        or doutput["dimensions"] != expected_output
    ):
        raise ValueError("FP8 SDPA backward O/dO metadata is invalid")
    if (
        output["data_type"] != q["data_type"]
        or doutput["data_type"] != q["data_type"]
        or stats["data_type"] != "float32"
        or stats["dimensions"] != [batch, heads, sequence_q, 1]
    ):
        raise ValueError("FP8 SDPA backward primal metadata is invalid")

    if stage == "dq":
        dq = tensors[14]
        amax_dq = tensors[15]
        if (
            dq["data_type"] != q["data_type"]
            or dq["dimensions"] != q["dimensions"]
        ):
            raise ValueError("FP8 SDPA backward dQ metadata is invalid")
        scale_names = (
            "descale Q",
            "descale K",
            "descale V",
            "descale O",
            "descale dO",
            "descale dP",
            "scale dQ",
            "scale dP",
        )
        scale_tensors = tensors[6:14]
        amax_tensors = ((amax_dq, "amax dQ"),)
    else:
        dk, dv = tensors[17:19]
        partial = parameters.get("_sdpa_partial_count", 1) > 1
        output_type = "float32" if partial else q["data_type"]
        output_shape = [
            batch,
            key_heads * parameters.get("_sdpa_partial_count", 1),
            sequence_kv,
            head_dimension,
        ]
        if (
            dk["data_type"] != output_type
            or dv["data_type"] != output_type
            or dk["dimensions"] != output_shape
            or dv["dimensions"] != output_shape
        ):
            raise ValueError("FP8 SDPA backward dK/dV metadata is invalid")
        scale_names = (
            "descale Q",
            "descale K",
            "descale V",
            "descale O",
            "descale dO",
            "descale S",
            "descale dP",
            "scale S",
            "scale dK",
            "scale dV",
            "scale dP",
        )
        scale_tensors = tensors[6:17]
        amax_tensors = tuple(
            zip(tensors[19:22], ("amax dK", "amax dV", "amax dP"))
        )
    for tensor, name in zip(scale_tensors, scale_names):
        _require_fp8_scale_tensor(tensor, f"FP8 SDPA backward {name}")
    for tensor, name in amax_tensors:
        _require_fp8_scale_tensor(tensor, f"FP8 SDPA backward {name}")

    attn_scale = _attention_runtime_f32(
        parameters, "attn_scale", _require_number(parameters, "attn_scale")
    )
    del attn_scale
    for name, value in (
        ("HQ", heads),
        ("SQ", sequence_q),
        ("SKV", sequence_kv),
        (
            "min_diag",
            _require_integer(
                parameters,
                "min_diag",
                minimum=-(2**31),
                maximum=2**31 - 1,
            ),
        ),
        (
            "max_diag",
            _require_integer(
                parameters,
                "max_diag",
                minimum=-(2**31),
                maximum=2**31 - 1,
            ),
        ),
    ):
        _attention_runtime_i32(parameters, name, value)

    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    common_constants: dict[str, int | float | str | bool] = {
        **_attention_strides(q, "q", "bhmd"),
        **_attention_strides(k, "k", "bhnd"),
        **_attention_strides(v, "v", "bhnd"),
        **_attention_strides(output, "o", "bhmd"),
        **_attention_strides(doutput, "do", "bhmd"),
        **_attention_strides(stats, "s", "bhm"),
        "HEAD_DIM": head_dimension,
        "BLOCK_M": 16 if stage == "dq" else 64,
        "BLOCK_N": 64 if stage == "dq" else 16,
        "BLOCK_D": max(16, _next_power_of_two(head_dimension)),
        "BANDED": _attention_flag(parameters, "banded"),
        "FULL_BLOCKS": False,
        "CAUSAL_TOP_LEFT": _attention_flag(parameters, "causal_top_left"),
    }
    runtime_scalars = {
        "attn_scale": "fp32",
        "HQ": "i32",
        "SQ": "i32",
        "SKV": "i32",
        "min_diag": "i32",
        "max_diag": "i32",
    }

    if stage == "dq":
        constants = {
            **common_constants,
            **_attention_strides(tensors[14], "dq", "bhmd"),
            "q_per_k": heads // key_heads,
            "q_per_v": heads // value_heads,
        }
        signature: dict[str, str] = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "o_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "dq_ptr": pointer_type,
            "amax_dq_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_q_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_k_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_v_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_o_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_do_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_dp_ptr": TRITON_POINTER_TYPES["float32"],
            "scale_dq_ptr": TRITON_POINTER_TYPES["float32"],
            "scale_dp_ptr": TRITON_POINTER_TYPES["float32"],
            **runtime_scalars,
        }
        layout: list[tuple[str, str | int | None]] = [
            *[("tensor_alias", index) for index in (0, 1, 2, 3, 4, 5)],
            ("tensor_alias", 14),
            ("tensor_alias", 15),
            *[("tensor_alias", index) for index in range(6, 14)],
            ("scalar_f32", "attn_scale"),
            ("scalar_i32", "HQ"),
            ("scalar_i32", "SQ"),
            ("scalar_i32", "SKV"),
            ("scalar_i32", "min_diag"),
            ("scalar_i32", "max_diag"),
        ]
        return (
            "_sdpa_fp8_bwd_dq_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_q, int(constants["BLOCK_M"])),
                batch * heads,
                1,
            ),
            layout,
        )

    constants = {
        **common_constants,
        **_attention_strides(tensors[17], "dk", "bhnd"),
        **_attention_strides(tensors[18], "dv", "bhnd"),
        "HKV": key_heads,
        "Q_PER": heads // key_heads,
        "PARTIAL": parameters.get("_sdpa_partial_count", 1) > 1,
        "QUERY_CHUNK": parameters.get("_sdpa_query_chunk", sequence_q),
    }
    signature = {
        "q_ptr": pointer_type,
        "k_ptr": pointer_type,
        "v_ptr": pointer_type,
        "o_ptr": pointer_type,
        "do_ptr": pointer_type,
        "stats_ptr": TRITON_POINTER_TYPES["float32"],
        "dk_ptr": TRITON_POINTER_TYPES[tensors[17]["data_type"]],
        "dv_ptr": TRITON_POINTER_TYPES[tensors[18]["data_type"]],
        "amax_dk_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_dv_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_dp_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_q_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_k_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_v_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_o_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_do_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_dp_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_dk_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_dv_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_dp_ptr": TRITON_POINTER_TYPES["float32"],
        "attn_scale": "fp32",
        "SQ": "i32",
        "SKV": "i32",
        "min_diag": "i32",
        "max_diag": "i32",
    }
    layout = [
        *[("tensor_alias", index) for index in (0, 1, 2, 3, 4, 5)],
        ("tensor_alias", 17),
        ("tensor_alias", 18),
        ("tensor_alias", 19),
        ("tensor_alias", 20),
        ("tensor_alias", 21),
        *[("tensor_alias", index) for index in range(6, 17)],
        ("scalar_f32", "attn_scale"),
        ("scalar_i32", "SQ"),
        ("scalar_i32", "SKV"),
        ("scalar_i32", "min_diag"),
        ("scalar_i32", "max_diag"),
    ]
    return (
        "_sdpa_fp8_bwd_dkdv_kernel",
        signature,
        constants,
        (
            _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
            batch * key_heads * int(parameters.get("_sdpa_partial_count", 1)),
            1,
        ),
        layout,
    )


def _sdpa_backward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_sdpa_bwd_stage")
    # FP32 dot operands need smaller tiles to fit the shared-memory budget.
    fp32 = bool(tensors) and tensors[0]["data_type"] == "float32"
    parameters["_sdpa_bwd_fp32"] = fp32
    block_m = 32 if fp32 else 64
    block_d_out = 64 if fp32 else 128
    # Short FP16/BF16 MMA dimensions need padded shared-memory layouts.
    minimum_dot_dimension = 16 if fp32 else 64
    if stage == "zero_dbias":
        if len(tensors) != 1:
            raise ValueError("SDPA dBias zero stage tensor count is invalid")
        elements = math.prod(tensors[0]["dimensions"])
        _attention_runtime_i32(parameters, "dbias_elements", elements)
        return (
            "_zero_contiguous_kernel",
            {
                "ptr": TRITON_POINTER_TYPES[tensors[0]["data_type"]],
                "n_elements": "i32",
            },
            {"BLOCK": 1024},
            (_ceil_div(elements, 1024), 1, 1),
            [
                ("tensor_alias", 0),
                ("scalar_i32", "dbias_elements"),
            ],
        )

    if stage == "dv":
        has_bias = _attention_flag(parameters, "has_bias")
        if len(tensors) != 5 + int(has_bias):
            raise ValueError("SDPA backward dV stage tensor count is invalid")
        q, k = tensors[:2]
        if (
            len(q["dimensions"]) != 4
            or len(k["dimensions"]) != 4
            or q["data_type"] not in FLOAT_DATA_TYPES
            or k["data_type"] != q["data_type"]
        ):
            raise ValueError("SDPA backward dV Q/K metadata is invalid")
        batch, heads, sequence_q, head_dimension = q["dimensions"]
        key_heads = k["dimensions"][1]
        sequence_kv = k["dimensions"][2]
        value_heads = _require_integer(parameters, "value_heads")
        value_dimension = _require_integer(parameters, "value_dimension")
        if (
            k["dimensions"][0] != batch
            or k["dimensions"][3] != head_dimension
            or key_heads != value_heads
            or heads % value_heads != 0
        ):
            raise ValueError("SDPA backward dV Q/K shapes are inconsistent")
        bias_index = 2 if has_bias else None
        offset = 2 + int(has_bias)
        doutput_index = offset
        stats_index = offset + 1
        dv_index = offset + 2
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        dv = tensors[dv_index]
        if dv["dimensions"] != [
            batch,
            value_heads,
            sequence_kv,
            value_dimension,
        ]:
            raise ValueError("SDPA backward dV output metadata is invalid")
        attn_scale = _require_number(parameters, "attn_scale")
        min_diag = _require_integer(
            parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
        )
        max_diag = _require_integer(
            parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
        )
        for name, value in (
            ("attn_scale", attn_scale),
            ("SQ", sequence_q),
            ("SKV", sequence_kv),
            ("min_diag", min_diag),
            ("max_diag", max_diag),
        ):
            if isinstance(value, float):
                _attention_runtime_f32(parameters, name, value)
            else:
                _attention_runtime_i32(parameters, name, value)
        if has_bias:
            bias_strides = _attention_broadcast_strides(
                tensors[bias_index], "bias_"
            )
        else:
            bias_strides = {
                "stride_bias_b": 0,
                "stride_bias_h": 0,
                "stride_bias_m": 0,
                "stride_bias_n": 0,
            }
        pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
        constants: dict[str, int | float | str | bool] = {
            "HKV": value_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **bias_strides,
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(dv, "dv", "bhnd"),
            "HEAD_DIM": head_dimension,
            "V_DIM": value_dimension,
            "Q_PER": heads // value_heads,
            "BLOCK_M": block_m,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": max(
                minimum_dot_dimension, _next_power_of_two(head_dimension)
            ),
            "BLOCK_DV_OUT": min(
                block_d_out,
                max(
                    minimum_dot_dimension, _next_power_of_two(value_dimension)
                ),
            ),
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "BANDED": _attention_flag(parameters, "banded"),
            "CAUSAL_TOP_LEFT": _attention_flag(parameters, "causal_top_left"),
        }
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "dv_ptr": pointer_type,
            "attn_scale": "fp32",
            "SQ": "i32",
            "SKV": "i32",
            "min_diag": "i32",
            "max_diag": "i32",
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", dv_index),
            ("scalar_f32", "attn_scale"),
            ("scalar_i32", "SQ"),
            ("scalar_i32", "SKV"),
            ("scalar_i32", "min_diag"),
            ("scalar_i32", "max_diag"),
        ]
        return (
            "_sdpa_bwd_dv_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
                _ceil_div(value_dimension, int(constants["BLOCK_DV_OUT"])),
                batch * value_heads,
            ),
            layout,
        )

    has_bias = _attention_flag(parameters, "has_bias")
    has_dbias = _attention_flag(parameters, "has_dbias")
    (
        batch,
        heads,
        key_heads,
        value_heads,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors)
    if key_heads != value_heads:
        raise ValueError("SDPA backward requires matching K/V head counts")
    q, k, v = tensors[:3]
    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    attn_scale = _require_number(parameters, "attn_scale")
    min_diag = _require_integer(
        parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
    )
    max_diag = _require_integer(
        parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
    )
    for name, value in (
        ("attn_scale", attn_scale),
        ("SQ", sequence_q),
        ("SKV", sequence_kv),
        ("min_diag", min_diag),
        ("max_diag", max_diag),
    ):
        if isinstance(value, float):
            _attention_runtime_f32(parameters, name, value)
        else:
            _attention_runtime_i32(parameters, name, value)

    has_banded = _attention_flag(parameters, "banded")
    causal_top_left = _attention_flag(parameters, "causal_top_left")
    block_d_full = max(
        minimum_dot_dimension, _next_power_of_two(head_dimension)
    )
    block_dv = max(minimum_dot_dimension, _next_power_of_two(value_dimension))

    def bias_constants(bias: dict[str, Any] | None) -> dict[str, int]:
        if bias is None:
            return {
                "stride_bias_b": 0,
                "stride_bias_h": 0,
                "stride_bias_m": 0,
                "stride_bias_n": 0,
            }
        return _attention_broadcast_strides(bias, "bias_")

    runtime_scalars = {
        "attn_scale": "fp32",
        "SQ": "i32",
        "SKV": "i32",
        "min_diag": "i32",
        "max_diag": "i32",
    }
    scalar_layout: list[tuple[str, str | int | None]] = [
        ("scalar_f32", "attn_scale"),
        ("scalar_i32", "SQ"),
        ("scalar_i32", "SKV"),
        ("scalar_i32", "min_diag"),
        ("scalar_i32", "max_diag"),
    ]

    if stage == "dq":
        expected_count = 8 + int(has_bias) + int(has_dbias)
        if len(tensors) != expected_count:
            raise ValueError("SDPA backward dQ stage tensor count is invalid")
        bias_index = 3 if has_bias else None
        offset = 3 + int(has_bias)
        output_index = offset
        doutput_index = offset + 1
        stats_index = offset + 2
        delta_index = offset + 3
        dq_index = offset + 4
        dbias_index = offset + 5 if has_dbias else None
        output = tensors[output_index]
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        delta = tensors[delta_index]
        dq = tensors[dq_index]
        dbias = tensors[dbias_index] if dbias_index is not None else None
        constants: dict[str, int | float | str | bool] = {
            "HQ": heads,
            "q_per_k": heads // key_heads,
            "q_per_v": heads // value_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **_attention_strides(v, "v", "bhnd"),
            **bias_constants(tensors[bias_index] if has_bias else None),
            **_attention_strides(output, "o", "bhmd"),
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(delta, "delta_", "bhm"),
            **_attention_strides(dq, "dq", "bhmd"),
            "HEAD_DIM": head_dimension,
            "V_DIM": value_dimension,
            "DBIAS_BATCHES": dbias["dimensions"][0] if dbias else 1,
            "DBIAS_HEADS": dbias["dimensions"][1] if dbias else 1,
            "BLOCK_M": block_m,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": block_d_full,
            "BLOCK_D_OUT": min(
                block_d_out,
                max(minimum_dot_dimension, _next_power_of_two(head_dimension)),
            ),
            "BLOCK_DV": block_dv,
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "HAS_DBIAS": has_dbias,
            "DBIAS_REDUCE": _attention_flag(parameters, "dbias_reduce"),
            "BANDED": has_banded,
            "CAUSAL_TOP_LEFT": causal_top_left,
        }
        if dbias is None:
            constants.update(
                {
                    "stride_dbias_b": 0,
                    "stride_dbias_h": 0,
                    "stride_dbias_m": 0,
                    "stride_dbias_n": 0,
                }
            )
        else:
            constants.update(_attention_strides(dbias, "dbias_", "bhmn"))
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "o_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "delta_ptr": TRITON_POINTER_TYPES["float32"],
            "dq_ptr": pointer_type,
            "dbias_ptr": pointer_type,
            **runtime_scalars,
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", output_index),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", delta_index),
            ("tensor_alias", dq_index),
            ("tensor_alias", dbias_index if dbias_index is not None else 0),
            *scalar_layout,
        ]
        return (
            "_sdpa_bwd_dq_dbias_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_q, int(constants["BLOCK_M"])),
                _ceil_div(head_dimension, int(constants["BLOCK_D_OUT"])),
                batch * heads,
            ),
            layout,
        )

    bias_index = 3 if has_bias else None
    offset = 3 + int(has_bias)
    if stage == "dkdv":
        if len(tensors) != 8 + int(has_bias):
            raise ValueError(
                "SDPA backward dK/dV stage tensor count is invalid"
            )
        doutput_index = offset
        stats_index = offset + 1
        delta_index = offset + 2
        dk_index = offset + 3
        dv_index = offset + 4
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        delta = tensors[delta_index]
        dk = tensors[dk_index]
        dv = tensors[dv_index]
        constants = {
            "HKV": key_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **_attention_strides(v, "v", "bhnd"),
            **bias_constants(tensors[bias_index] if has_bias else None),
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(delta, "delta_", "bhm"),
            **_attention_strides(dk, "dk", "bhnd"),
            **_attention_strides(dv, "dv", "bhnd"),
            "HEAD_DIM": head_dimension,
            "Q_PER": heads // key_heads,
            "PARTIAL": parameters.get("_sdpa_partial_count", 1) > 1,
            "QUERY_CHUNK": parameters.get("_sdpa_query_chunk", sequence_q),
            "BLOCK_M": block_m,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": block_d_full,
            "BLOCK_D_OUT": min(
                block_d_out,
                max(minimum_dot_dimension, _next_power_of_two(head_dimension)),
            ),
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "BANDED": has_banded,
            "CAUSAL_TOP_LEFT": causal_top_left,
        }
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "delta_ptr": TRITON_POINTER_TYPES["float32"],
            "dk_ptr": TRITON_POINTER_TYPES[dk["data_type"]],
            "dv_ptr": TRITON_POINTER_TYPES[dv["data_type"]],
            **runtime_scalars,
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", delta_index),
            ("tensor_alias", dk_index),
            ("tensor_alias", dv_index),
            *scalar_layout,
        ]
        return (
            "_sdpa_bwd_dkdv_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
                _ceil_div(head_dimension, int(constants["BLOCK_D_OUT"])),
                batch
                * key_heads
                * int(parameters.get("_sdpa_partial_count", 1)),
            ),
            layout,
        )

    if stage == "dk":
        if len(tensors) != 7 + int(has_bias):
            raise ValueError("SDPA backward dK stage tensor count is invalid")
        doutput_index = offset
        stats_index = offset + 1
        delta_index = offset + 2
        dk_index = offset + 3
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        delta = tensors[delta_index]
        dk = tensors[dk_index]
        constants = {
            "HKV": key_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **_attention_strides(v, "v", "bhnd"),
            **bias_constants(tensors[bias_index] if has_bias else None),
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(delta, "delta_", "bhm"),
            **_attention_strides(dk, "dk", "bhnd"),
            "HEAD_DIM": head_dimension,
            "V_DIM": value_dimension,
            "Q_PER": heads // key_heads,
            "BLOCK_M": block_m,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": block_d_full,
            "BLOCK_D_OUT": min(
                block_d_out,
                max(minimum_dot_dimension, _next_power_of_two(head_dimension)),
            ),
            "BLOCK_DV": block_dv,
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "BANDED": has_banded,
            "CAUSAL_TOP_LEFT": causal_top_left,
        }
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "delta_ptr": TRITON_POINTER_TYPES["float32"],
            "dk_ptr": pointer_type,
            **runtime_scalars,
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", delta_index),
            ("tensor_alias", dk_index),
            *scalar_layout,
        ]
        return (
            "_sdpa_bwd_dk_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
                _ceil_div(head_dimension, int(constants["BLOCK_D_OUT"])),
                batch * key_heads,
            ),
            layout,
        )

    raise ValueError("SDPA backward pipeline stage is invalid")
