# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch attention lowering."""

from __future__ import annotations

from typing import Any
import math
from ..dispatch.common import (
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _dense_strides,
    _named_port_uids,
)


def _validate_attention_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    fp8 = operation in {"sdpa_fp8", "sdpa_fp8_backward"}
    backward = operation in {"sdpa_backward", "sdpa_fp8_backward"}
    nodes = _require_list(graph["nodes"], "graph.nodes")
    tensors = _require_list(graph["tensors"], "graph.tensors")
    if len(nodes) != 1 or nodes[0]["id"] != 0 or nodes[0]["type"] != operation:
        raise ValueError("THead attention requires one canonical graph node")
    node = nodes[0]
    if node["compute_data_type"] != "float32":
        raise ValueError("THead attention requires float32 accumulation")
    attributes = _require_object(node["attributes"], "attention attributes")
    fields = {
        "attn_scale",
        "attn_scale_set",
        "banded",
        "batch",
        "causal_top_left",
        "diagonal_alignment",
        "diagonal_band_left_bound",
        "diagonal_band_right_bound",
        "generate_stats",
        "has_bias",
        "has_dbias",
        "head_dimension",
        "heads",
        "key_heads",
        "left_bound_set",
        "max_diag",
        "min_diag",
        "q_per_k",
        "q_per_v",
        "reverse_causal",
        "right_bound_set",
        "sequence_kv",
        "sequence_q",
        "value_dimension",
        "value_heads",
    }
    _require_exact_fields(attributes, fields, set(), "attention attributes")
    for name in ("attn_scale_set", "left_bound_set", "right_bound_set"):
        if not isinstance(attributes[name], bool):
            raise ValueError(f"THead attention {name} must be boolean")
    integers = {
        name: _integer(attributes[name], f"attention {name}")
        for name in fields
        - {"attn_scale", "attn_scale_set", "left_bound_set", "right_bound_set"}
    }
    for name in (
        "has_bias",
        "has_dbias",
        "generate_stats",
        "banded",
        "causal_top_left",
        "reverse_causal",
    ):
        if integers[name] not in (0, 1):
            raise ValueError(f"THead attention {name} must be zero or one")
    if (
        attributes["left_bound_set"]
        or integers["diagonal_alignment"] != 0
        or integers["diagonal_band_left_bound"] != 0
        or integers["diagonal_band_right_bound"] != 0
    ):
        raise ValueError(
            "THead attention supports dense and top-left causal masks"
        )
    causal = attributes["right_bound_set"]
    has_bias, has_dbias = bool(integers["has_bias"]), bool(
        integers["has_dbias"]
    )
    input_names = (
        ("q", "k", "v", "o", "do", "stats") if backward else ("q", "k", "v")
    )
    if fp8 and (has_bias or has_dbias):
        raise ValueError("THead FP8 attention bias is not qualified")
    scalar_names: tuple[str, ...] = ()
    amax_names: tuple[str, ...] = ()
    if fp8:
        scalar_names = (
            (
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
            if backward
            else (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_s",
                "scale_s",
                "scale_o",
            )
        )
        amax_names = (
            ("amax_dq", "amax_dk", "amax_dv", "amax_dp")
            if backward
            else ("amax_s", "amax_o")
        )
        input_names += scalar_names
    if has_bias:
        input_names += ("bias",)
    output_names = ("dq", "dk", "dv") if backward else ("o", "stats")
    if has_dbias:
        if not backward or not has_bias:
            raise ValueError(
                "THead attention bias gradient requires a backward bias input"
            )
        output_names += ("dbias",)
    output_names += amax_names
    input_uids = _named_port_uids(node, "inputs", input_names, "attention")
    output_uids = _named_port_uids(node, "outputs", output_names, "attention")
    uids = input_uids + output_uids
    if len(set(uids)) != len(uids) or len(uids) != len(tensors):
        raise ValueError(
            "THead attention requires distinct complete tensor ports"
        )
    by_uid = {int(t["uid"]): t for t in tensors}
    named = {
        name: by_uid[uid]
        for name, uid in zip(input_names + output_names, uids)
    }
    q, k, v = (named[name] for name in ("q", "k", "v"))
    if any(
        len(t["dimensions"]) != 4
        or t["strides"] != _dense_strides(t["dimensions"])
        or int(t["alignment"]) < 16
        for t in tensors
    ):
        raise ValueError(
            "THead attention requires dense rank-four tensors with 16-byte"
            " alignment"
        )
    dtype = q["data_type"]
    allowed_types = (
        {"fp8_e4m3", "fp8_e5m2"} if fp8 else {"float32", "float16", "bfloat16"}
    )
    float_names = {"stats", *scalar_names, *amax_names}
    if dtype not in allowed_types or any(
        t["data_type"] != ("float32" if name in float_names else dtype)
        for name, t in named.items()
    ):
        raise ValueError(
            "THead attention tensor types differ from the operation dtype and"
            " fp32 metadata"
        )
    batch, heads, sq, dimension = q["dimensions"]
    kb, key_heads, sk, kd = k["dimensions"]
    vb, value_heads, vs, value_dimension = v["dimensions"]
    if (
        kb != batch
        or vb != batch
        or kd != dimension
        or vs != sk
        or key_heads != value_heads
        or heads % key_heads
    ):
        raise ValueError("THead attention Q/K/V geometry is inconsistent")
    if batch > 2 or heads > 8 or heads // key_heads > 4:
        raise ValueError("THead attention batch/head profile is not qualified")
    profile = (sq, sk, dimension, value_dimension)
    if not (
        1 <= sq <= 512
        and 1 <= sk <= 512
        and dimension in {16, 32, 64, 128}
        and value_dimension in {16, 32, 64, 128}
    ):
        raise ValueError(
            "THead attention geometry exceeds the supported tile bounds"
        )
    expected_shapes = {
        "o": [batch, heads, sq, value_dimension],
        "stats": [batch, heads, sq, 1],
    }
    if backward:
        expected_shapes.update(
            do=expected_shapes["o"],
            dq=q["dimensions"],
            dk=k["dimensions"],
            dv=v["dimensions"],
        )
    expected_shapes.update(
        {name: [1, 1, 1, 1] for name in scalar_names + amax_names}
    )
    for name, shape in expected_shapes.items():
        if named[name]["dimensions"] != shape:
            raise ValueError(f"THead attention {name} shape mismatch")
    if has_bias:
        bias_shape = named["bias"]["dimensions"]
        if bias_shape not in ([1, heads, sq, sk], [batch, heads, sq, sk]):
            raise ValueError(
                "THead attention supports batch-broadcast bias with matching"
                " heads"
            )
        if has_dbias and named["dbias"]["dimensions"] != bias_shape:
            raise ValueError("THead attention bias gradient shape mismatch")
    for name, tensor in named.items():
        expected_virtual = (
            name == "stats" and not backward and not integers["generate_stats"]
        )
        if tensor["virtual"] != expected_virtual:
            raise ValueError(
                "THead attention tensor storage differs from stats generation"
            )
    expected = {
        "batch": batch,
        "heads": heads,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "sequence_q": sq,
        "sequence_kv": sk,
        "head_dimension": dimension,
        "value_dimension": value_dimension,
        "q_per_k": heads // key_heads,
        "q_per_v": heads // value_heads,
        "banded": int(causal),
        "causal_top_left": int(causal and sq == sk),
        "reverse_causal": int(causal),
        "min_diag": -(1 << 30),
        "max_diag": 0 if causal else 1 << 30,
    }
    if any(integers[name] != value for name, value in expected.items()):
        raise ValueError(
            "THead attention lowered metadata differs from tensors or mask"
        )
    if backward and integers["generate_stats"] != 1:
        raise ValueError("THead attention backward requires stats")
    scale = attributes["attn_scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
    ):
        raise ValueError("THead attention scale must be finite")
    if not attributes["attn_scale_set"] and not math.isclose(
        scale, 1 / math.sqrt(dimension), rel_tol=1e-12
    ):
        raise ValueError("THead attention default scale is inconsistent")
    return {
        "tensors": tensors,
        "named": named,
        "attributes": attributes,
        "profile": profile,
        "backward": backward,
        "fp8": fp8,
        "backward_shared": (0),
    }
