# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Graph port contracts for the extended THead operator families."""
from .extended_common import _require_integer


def tensor_roles(operation, attributes):
    if operation in {"instancenorm", "adalayernorm"}:
        return ("x", "scale", "bias"), ("y", "mean", "inv_variance")
    if operation in {
        "instancenorm_backward",
        "adalayernorm_backward",
        "layernorm_backward",
        "batchnorm_backward",
    }:
        return ("dy", "x", "scale", "mean", "inv_variance"), (
            "dx",
            "dscale",
            "dbias",
        )
    if operation == "rmsnorm_backward":
        return ("dy", "x", "scale", "inv_variance"), ("dx", "dscale", "dbias")
    if operation == "genstats":
        return ("x",), ("sum", "sq_sum")
    if operation in {"gen_index", "rng"}:
        return (), ("output",)
    if operation == "concatenate":
        count = _require_integer(attributes, "input_count", maximum=65536)
        return tuple(f"input_{i}" for i in range(count)), ("output",)
    if operation == "rope":
        return ("input", "freqs"), ("output",)
    if operation == "rope_backward":
        return ("dy", "freqs"), ("dx",)
    if operation == "matmul":
        return ("a", "b"), ("output",)
    if operation == "matmul_fp8":
        mode = _require_integer(attributes, "scale_mode", minimum=0, maximum=2)
        return ("a", "b") + (("descale_a", "descale_b") if mode else ()), (
            "output",
        )
    if operation == "causal_conv1d":
        bias = _require_integer(attributes, "has_bias", minimum=0, maximum=1)
        return ("input", "weight") + (("bias",) if bias else ()), ("output",)
    if operation == "resample":
        index = _require_integer(
            attributes, "generate_index", minimum=0, maximum=1
        )
        return ("input",), ("output",) + (("index",) if index else ())
    if operation == "bn_finalize":
        running = _require_integer(
            attributes, "has_running", minimum=0, maximum=1
        )
        return (
            ("sum", "sq_sum", "scale", "bias")
            + (
                ("previous_running_mean", "previous_running_variance")
                if running
                else ()
            ),
            ("eq_scale", "eq_bias", "mean", "inv_variance")
            + (
                ("next_running_mean", "next_running_variance")
                if running
                else ()
            ),
        )
    if operation == "moe_grouped_matmul":
        mode = _require_integer(attributes, "mode", minimum=0, maximum=2)
        return (
            ("token", "weight", "first_token_offset")
            + (("token_index",) if mode else ())
            + (("token_ks",) if mode == 2 else ()),
            ("output",),
        )
    if operation == "moe_grouped_matmul_bwd":
        return ("doutput", "token", "first_token_offset"), ("dweight",)
    raise ValueError(f"unsupported extended operation {operation!r}")
