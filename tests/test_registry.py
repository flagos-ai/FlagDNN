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

import flag_dnn  # noqa: F401  (ensures capture wrappers are installed)
from flag_dnn.graph import wrappers
from flag_dnn.graph.registry import (
    get_op_schema,
    graph_wrapper_specs,
    registered_ops,
)


def test_wrapper_specs_are_derived_from_registry() -> None:
    # The capture-wrapper set is derived from the registry, so a graph-aware
    # schema and its wrapper can never drift apart (no parallel metadata list).
    schemas = registered_ops()
    specs = graph_wrapper_specs()
    for name, spec in specs.items():
        assert name in schemas, name
        assert schemas[name].graph_aware
        assert spec.eager_name
    for name, schema in schemas.items():
        assert (name in specs) == schema.graph_aware


def test_installed_wrappers_match_specs() -> None:
    # GRAPH_AWARE_OPS is populated at install time straight from the registry.
    assert set(wrappers.GRAPH_AWARE_OPS) == set(graph_wrapper_specs())


def test_fusion_internal_ops_have_no_wrapper() -> None:
    # Fusion-internal ops are registered (need shape/run) but are never user
    # callable, so they must not get a capture wrapper.
    schemas = registered_ops()
    specs = graph_wrapper_specs()
    for name in (
        "fused_bias_relu",
        "fused_bias_gelu",
        "fused_conv2d_bias_relu",
    ):
        assert name in schemas
        assert name not in specs


def test_dict_output_op_keys() -> None:
    schema = get_op_schema("rmsnorm_rht_amax_wrapper_sm100")
    assert schema.output_keys == ("o_tensor", "amax_tensor")
    schema.validate_output_count(len(schema.output_keys))
    spec = graph_wrapper_specs()["rmsnorm_rht_amax_wrapper_sm100"]
    assert spec.output_keys == ("o_tensor", "amax_tensor")


def test_output_keys_match_schema_arity() -> None:
    for schema in registered_ops().values():
        if schema.output_keys is None:
            continue
        schema.validate_output_count(len(schema.output_keys))


def test_variable_output_ops_are_explicit() -> None:
    assert get_op_schema("sdpa").num_outputs == (1, 2)

    for schema in registered_ops().values():
        if isinstance(schema.num_outputs, tuple):
            assert schema.num_outputs
            assert all(isinstance(count, int) for count in schema.num_outputs)
            assert min(schema.num_outputs) > 0
