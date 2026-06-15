"""Default graph op registration.

Each op family declares its own ops (normalize + shape inference + eager
fallback run + registration) in ``flag_dnn.graph.registry.schemas.<family>``.
This module just wires those families together: to add a new op, edit the
relevant family module -- nothing here needs to change unless you add a whole
new family (then add one ``register`` call below).
"""

from __future__ import annotations

from flag_dnn.graph.registry.core import (
    OpSchema,
    get_registered_op,
    registered_raw_ops,
)

_DEFAULTS_REGISTERED = False


def get_op_schema(name: str) -> OpSchema:
    register_default_ops()
    schema = get_registered_op(name)
    if schema is None:
        raise KeyError(f"FlagDNN graph op is not registered: {name}")
    return schema


def registered_ops() -> dict[str, OpSchema]:
    register_default_ops()
    return registered_raw_ops()


def register_default_ops() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    import flag_dnn.ops as flag_ops
    from flag_dnn.graph.registry.schemas import conv as _conv
    from flag_dnn.graph.registry.schemas import fused as _fused
    from flag_dnn.graph.registry.schemas import (
        matmul_attention as _matmul_attention,
    )
    from flag_dnn.graph.registry.schemas import (
        norm_reduction as _norm_reduction,
    )
    from flag_dnn.graph.registry.schemas import pointwise as _pointwise
    from flag_dnn.graph.registry.schemas import utility as _utility

    _pointwise.register(flag_ops)
    _utility.register(flag_ops)
    _conv.register(flag_ops)
    _matmul_attention.register(flag_ops)
    _norm_reduction.register(flag_ops)
    _fused.register(flag_ops)

    _DEFAULTS_REGISTERED = True
