"""Fusion-internal ops produced by graph fusion passes.

These ops are registered (they need shape inference and an eager fallback) but
are *not* graph_aware: users never call them directly, so no capture wrapper is
installed. Their normalize/shape are reused from the pointwise and conv
families.
"""

from __future__ import annotations

from typing import Any

from flag_dnn.graph.registry.core import OpDef, register_op_def
from flag_dnn.graph.registry.schemas._run_common import (
    _format_bias,
    _public_attrs,
    _require_runtime_backend,
    _unsupported_triton_path,
)
from flag_dnn.graph.registry.schemas.conv import (
    _conv2d_shape,
    _normalize_conv2d,
)
from flag_dnn.graph.registry.schemas.pointwise import (
    _bias_add_shape,
    _normalize_bias_add,
)


def _run_fused_bias_relu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "fused_bias_relu")
        x, bias = inputs
        y = flag_ops.add(x, _format_bias(x, bias))
        return flag_ops.relu(y)

    return run


def _run_fused_bias_gelu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "fused_bias_gelu")
        x, bias = inputs
        y = flag_ops.add(x, _format_bias(x, bias))
        return flag_ops.gelu(y, approximate=attrs.get("approximate", "none"))

    return run


def _run_fused_conv2d_bias_relu(flag_ops: Any) -> Any:
    def run(inputs: list[Any], attrs: dict[str, Any]) -> Any:
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


def register(flag_ops: Any) -> None:
    """Register fusion-internal ops (not graph_aware, no capture wrapper)."""
    register_op_def(
        OpDef(
            name="fused_bias_relu",
            normalize=_normalize_bias_add,
            shape=_bias_add_shape,
            run=_run_fused_bias_relu(flag_ops),
            fusible=True,
            graph_aware=False,
        )
    )
    register_op_def(
        OpDef(
            name="fused_bias_gelu",
            normalize=_normalize_bias_add,
            shape=_bias_add_shape,
            run=_run_fused_bias_gelu(flag_ops),
            fusible=True,
            graph_aware=False,
        )
    )
    register_op_def(
        OpDef(
            name="fused_conv2d_bias_relu",
            normalize=_normalize_conv2d,
            shape=_conv2d_shape,
            run=_run_fused_conv2d_bias_relu(flag_ops),
            fusible=True,
            graph_aware=False,
        )
    )


__all__ = ("register",)
