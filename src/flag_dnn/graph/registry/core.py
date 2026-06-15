from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

NormalizeFn = Callable[
    [Any, tuple[Any, ...], dict[str, Any]], tuple[list[int], dict]
]
ShapeFn = Callable[[list[Any], dict[str, Any]], list[Any]]
RunFn = Callable[[list[Any], dict[str, Any]], Any]
OutputArity = int | tuple[int, ...] | None


@dataclass(frozen=True)
class GraphOpMetadata:
    graph_aware: bool = True
    output_keys: tuple[str, ...] | None = None


GRAPH_OP_METADATA: dict[str, GraphOpMetadata] = {
    name: GraphOpMetadata()
    for name in (
        "identity",
        "reshape",
        "transpose",
        "slice",
        "concatenate",
        "gen_index",
        "add",
        "sub",
        "mul",
        "scale",
        "div",
        "mod",
        "pow",
        "max",
        "min",
        "minimum",
        "maximum",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "cmp_eq",
        "cmp_neq",
        "cmp_lt",
        "cmp_le",
        "cmp_gt",
        "cmp_ge",
        "bias_add",
        "add_square",
        "relu",
        "swish",
        "gelu",
        "gelu_approx_tanh",
        "leaky_relu",
        "elu",
        "softplus",
        "conv2d",
        "conv_fprop",
        "conv_dgrad",
        "conv_wgrad",
        "causal_conv1d",
        "mm",
        "matmul",
        "sdpa",
        "sdpa_backward",
        "batchnorm",
        "batchnorm_inference",
        "layernorm",
        "rmsnorm",
        "reduction",
        "sqrt",
        "square",
        "rsqrt",
        "exp",
        "log",
        "reciprocal",
        "ceil",
        "floor",
        "erf",
        "sin",
        "cos",
        "tan",
        "abs",
        "neg",
        "tanh",
        "sigmoid",
        "sigmoid_backward",
        "silu",
        "logical_and",
        "logical_or",
        "logical_not",
        "binary_select",
    )
}
GRAPH_OP_METADATA["rmsnorm_rht_amax_wrapper_sm100"] = GraphOpMetadata(
    output_keys=("o_tensor", "amax_tensor"),
)


@dataclass
class OpSchema:
    name: str
    normalize_fn: NormalizeFn
    shape_fn: ShapeFn
    run_fn: RunFn
    num_outputs: OutputArity = 1
    attrs_schema: dict[str, Any] = field(default_factory=dict)
    fusible: bool = False
    graph_aware: bool = False
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


_REGISTRY: dict[str, OpSchema] = {}


def register_op(schema: OpSchema) -> OpSchema:
    metadata = GRAPH_OP_METADATA.get(schema.name)
    if metadata is not None:
        schema = replace(
            schema,
            graph_aware=metadata.graph_aware,
            output_keys=(
                schema.output_keys
                if schema.output_keys is not None
                else metadata.output_keys
            ),
        )
    if schema.output_keys is not None:
        schema.validate_output_count(len(schema.output_keys))
    _REGISTRY[schema.name] = schema
    return schema


def get_registered_op(name: str) -> OpSchema | None:
    return _REGISTRY.get(name)


def registered_raw_ops() -> dict[str, OpSchema]:
    return dict(_REGISTRY)


def graph_aware_op_names() -> tuple[str, ...]:
    return tuple(
        name
        for name, metadata in GRAPH_OP_METADATA.items()
        if metadata.graph_aware
    )


def graph_output_keys(name: str) -> tuple[str, ...] | None:
    metadata = GRAPH_OP_METADATA.get(name)
    return None if metadata is None else metadata.output_keys
