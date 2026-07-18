# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


class CaptureContext(Protocol):
    """Capture-time context passed to an op's ``normalize`` function.

    A normalize function turns user args/kwargs into graph input value ids plus
    immutable attrs; it uses this context to register tensor/scalar operands as
    graph values. ``flag_dnn.graph.capture.GraphCapture`` implements it.
    """

    def as_value(self, value: Any, name_hint: str = ...) -> int:
        """Register an operand and return its graph value id."""
        ...

    def is_tensor_like(self, value: Any) -> bool:
        """Whether value is tensor-like (vs. a scalar folded into attrs)."""
        ...


NormalizeFn = Callable[
    [CaptureContext, tuple[Any, ...], dict[str, Any]], tuple[list[int], dict]
]
ShapeFn = Callable[[list[Any], dict[str, Any]], list[Any]]
RunFn = Callable[[list[Any], dict[str, Any]], Any]
OutputArity = int | tuple[int, ...] | None


@dataclass
class OpSchema:
    """Runtime contract for one graph op type.

    ``normalize_fn`` runs during capture (args/kwargs -> input ids + attrs),
    ``shape_fn`` runs during graph build (output ``TensorSpec`` inference) and
    ``run_fn`` is the eager fallback executed at replay when no prepared fast
    path applies. ``graph_aware``/``eager_name``/``output_keys`` drive capture
    wrapper installation and are the single source of truth for it -- there is
    no separate metadata table to keep in sync.
    """

    name: str
    normalize_fn: NormalizeFn
    shape_fn: ShapeFn
    run_fn: RunFn
    num_outputs: OutputArity = 1
    attrs_schema: dict[str, Any] = field(default_factory=dict)
    fusible: bool = False
    graph_aware: bool = False
    eager_name: Optional[str] = None
    output_keys: tuple[str, ...] | None = None

    def normalize(
        self, ctx: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[int], dict]:
        return self.normalize_fn(ctx, args, kwargs)

    def infer_outputs(
        self, input_specs: list[Any], attrs: dict[str, Any]
    ) -> list[Any]:
        outputs = list(self.shape_fn(input_specs, attrs))
        self.validate_output_count(len(outputs))
        return outputs

    def validate_output_count(self, count: int) -> None:
        arity = self.num_outputs
        if arity is None:
            return
        if isinstance(arity, tuple):
            if count in arity:
                return
            expected = "/".join(str(item) for item in arity)
        else:
            if count == arity:
                return
            expected = str(arity)
        raise RuntimeError(
            f"graph op {self.name} inferred {count} outputs, "
            f"expected {expected}"
        )

    def run(self, inputs: list[Any], attrs: dict[str, Any]) -> Any:
        return self.run_fn(inputs, attrs)


@dataclass(frozen=True)
class OpDef:
    """Declarative definition of one graph op, the place a new operator is
    added. ``register_op_def`` turns it into an ``OpSchema`` and, when
    ``graph_aware`` is set, records the capture-wrapper spec -- both derived
    from this single declaration.

    Fields:
        name: graph op type (matches the eager op name unless ``eager_name``).
        normalize: capture-time arg/kwarg -> (input ids, attrs).
        shape: build-time output ``TensorSpec`` inference.
        run: replay fallback; call the eager op for correctness.
        num_outputs: output arity (int, tuple of allowed counts, or None).
        fusible: whether fusion passes may absorb this op.
        graph_aware: install a capture wrapper so users can call it inside
            ``@flag_dnn.graph`` (False for fusion-internal ops like fused_*).
        eager_name: eager namespace function to wrap, when it differs from
            ``name``.
        output_keys: keys when the eager op returns a dict instead of tensors.
    """

    name: str
    normalize: NormalizeFn
    shape: ShapeFn
    run: RunFn
    num_outputs: OutputArity = 1
    fusible: bool = False
    graph_aware: bool = True
    eager_name: Optional[str] = None
    output_keys: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GraphWrapperSpec:
    """How to install one capture wrapper, derived from a graph-aware op."""

    op_type: str
    eager_name: str
    output_keys: tuple[str, ...] | None = None


_REGISTRY: dict[str, OpSchema] = {}
_WRAPPER_SPECS: dict[str, GraphWrapperSpec] = {}


def register_op(schema: OpSchema) -> OpSchema:
    """Register an ``OpSchema`` and derive its capture-wrapper spec."""
    if schema.output_keys is not None:
        schema.validate_output_count(len(schema.output_keys))
    _REGISTRY[schema.name] = schema
    if schema.graph_aware:
        _WRAPPER_SPECS[schema.name] = GraphWrapperSpec(
            op_type=schema.name,
            eager_name=schema.eager_name or schema.name,
            output_keys=schema.output_keys,
        )
    else:
        _WRAPPER_SPECS.pop(schema.name, None)
    return schema


def register_op_def(op: OpDef) -> OpSchema:
    """Register an operator from its declarative ``OpDef``."""
    return register_op(
        OpSchema(
            name=op.name,
            normalize_fn=op.normalize,
            shape_fn=op.shape,
            run_fn=op.run,
            num_outputs=op.num_outputs,
            fusible=op.fusible,
            graph_aware=op.graph_aware,
            eager_name=op.eager_name,
            output_keys=op.output_keys,
        )
    )


def get_registered_op(name: str) -> OpSchema | None:
    return _REGISTRY.get(name)


def registered_raw_ops() -> dict[str, OpSchema]:
    return dict(_REGISTRY)


def graph_wrapper_specs() -> dict[str, GraphWrapperSpec]:
    """Capture-wrapper specs for every graph-aware op, derived from the
    registry -- the single source of truth consumed by ``wrappers.py``."""
    return dict(_WRAPPER_SPECS)


def graph_aware_op_names() -> tuple[str, ...]:
    return tuple(_WRAPPER_SPECS)


def graph_output_keys(name: str) -> tuple[str, ...] | None:
    spec = _WRAPPER_SPECS.get(name)
    return None if spec is None else spec.output_keys
