from __future__ import annotations

from flag_dnn.graph import wrappers
from flag_dnn.graph.registry import get_op_schema, registered_ops
from flag_dnn.graph.registry.core import (
    GRAPH_OP_METADATA,
    graph_aware_op_names,
)


def test_graph_metadata_ops_are_registered_and_wrapped() -> None:
    schemas = registered_ops()
    graph_aware = graph_aware_op_names()

    assert wrappers.GRAPH_AWARE_OPS == graph_aware
    missing = [name for name in graph_aware if name not in schemas]
    assert missing == []


def test_output_key_metadata_matches_schema_arity() -> None:
    schemas = registered_ops()
    for name, metadata in GRAPH_OP_METADATA.items():
        if metadata.output_keys is None:
            continue
        schema = schemas[name]
        assert schema.output_keys == metadata.output_keys
        schema.validate_output_count(len(metadata.output_keys))


def test_variable_output_ops_are_explicit() -> None:
    assert get_op_schema("sdpa").num_outputs == (1, 2)

    for schema in registered_ops().values():
        if isinstance(schema.num_outputs, tuple):
            assert schema.num_outputs
            assert all(isinstance(count, int) for count in schema.num_outputs)
            assert min(schema.num_outputs) > 0
