from __future__ import annotations

from typing import Any

from flag_dnn.graph.registry.core import (
    GraphWrapperSpec,
    NormalizeFn,
    OpDef,
    OpSchema,
    OutputArity,
    RunFn,
    ShapeFn,
    graph_aware_op_names,
    graph_output_keys,
    graph_wrapper_specs,
    register_op,
    register_op_def,
)
from flag_dnn.graph.registry.ops import (
    get_op_schema,
    register_default_ops,
    registered_ops,
)
import flag_dnn.graph.registry.ops as _registry_ops

__all__ = [
    "GraphWrapperSpec",
    "NormalizeFn",
    "OpDef",
    "OpSchema",
    "OutputArity",
    "RunFn",
    "ShapeFn",
    "get_op_schema",
    "graph_aware_op_names",
    "graph_output_keys",
    "graph_wrapper_specs",
    "register_default_ops",
    "register_op",
    "register_op_def",
    "registered_ops",
]


def __getattr__(name: str) -> Any:
    return getattr(_registry_ops, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_registry_ops)))
