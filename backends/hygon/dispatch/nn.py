# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon dispatch nn."""

from __future__ import annotations

from .nn_common import (
    ALLOW_TF32,
    ATTENTION_OPERATIONS,
    CONVOLUTION_OPERATIONS,
    CONV_FPROP_STANDARD_3X3_TUNING,
    CONV_FPROP_YOLO_X_P5_GEMM_TUNING,
    DGRAD_EXACT_3X3_S1_WORKSPACE_CAP,
    GridSpec,
    KernelStagePlan,
    NORMALIZATION_OPERATIONS,
    NodePlan,
    OPERATIONS,
    OPERATION_SCHEMAS,
    OperationSchema,
    REGISTRY_FUNCTIONS,
    STRICT_DOT_INPUT_PRECISION,
    SUPPORTED_OPERATIONS,
    TuningMetadata,
    WORKSPACE_ALIGNMENT,
    WorkspaceTensor,
    _parse_port,
    _require_object,
    _require_sequence,
    _resolved_ports,
    tensor_storage_size,
)

from ..dispatch.attention import (
    _attention_backward_plan,
    _attention_forward_stage,
    _fp8_backward_plan,
    _fp8_forward_plan,
    _validate_attention,
)
from ..dispatch.convolution import (
    _convolution_plan,
    _validate_convolution,
)
from ..dispatch.normalization import (
    _normalization_stage,
    _validate_normalization,
)
from typing import Any
from typing import Mapping


def parse_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Parse and semantically validate one schema-v3 DNN node."""

    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
        or isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count <= 0
        or position >= node_count
    ):
        raise ValueError("graph node position/count is invalid")
    node = _require_object(node_value, f"graph.nodes[{position}]")
    node_id = node.get("id")
    if (
        isinstance(node_id, bool)
        or not isinstance(node_id, int)
        or node_id < 0
        or node_id >= node_count
    ):
        raise ValueError("graph node ID is invalid")
    operation = node.get("type")
    if not isinstance(operation, str) or operation not in OPERATION_SCHEMAS:
        raise ValueError(
            f"Hygon NN compiler does not support operation {operation!r}"
        )
    if node.get("compute_data_type") != "float32":
        raise ValueError(f"{operation} requires float32 compute_data_type")
    parameters = dict(
        _require_object(
            node.get("attributes"), f"graph.nodes[{position}].attributes"
        )
    )
    schema = OPERATION_SCHEMAS[operation]
    input_ports, output_ports = _resolved_ports(operation, schema, parameters)
    inputs = _require_sequence(node.get("inputs"), "node.inputs")
    outputs = _require_sequence(node.get("outputs"), "node.outputs")
    if len(inputs) != len(input_ports) or len(outputs) != len(output_ports):
        raise ValueError(f"{operation} node port count is invalid")

    input_uids: list[int] = []
    output_uids: list[int] = []
    tensors: list[Mapping[str, Any]] = []
    port_tensors: dict[str, Mapping[str, Any]] = {}
    port_indices: dict[str, int] = {}
    for index, name in enumerate(input_ports):
        uid, tensor = _parse_port(
            inputs[index], name, f"input[{index}]", tensor_registry
        )
        input_uids.append(uid)
        port_indices[name] = len(tensors)
        port_tensors[name] = tensor
        tensors.append(tensor)
    for index, name in enumerate(output_ports):
        uid, tensor = _parse_port(
            outputs[index], name, f"output[{index}]", tensor_registry
        )
        output_uids.append(uid)
        port_indices[name] = len(tensors)
        port_tensors[name] = tensor
        tensors.append(tensor)
    if len(set(output_uids)) != len(output_uids):
        raise ValueError(f"{operation} output tensors must be distinct")
    if set(input_uids).intersection(output_uids):
        raise ValueError(f"{operation} does not support in-place outputs")

    parsed = {
        "id": node_id,
        "operation": operation,
        "schema": schema,
        "compute_data_type": "float32",
        "parameters": parameters,
        "tensors": tensors,
        "tensor_roles": [*input_ports, *output_ports],
        "port_tensors": port_tensors,
        "port_indices": port_indices,
        "input_uids": input_uids,
        "output_uids": output_uids,
    }
    if operation in CONVOLUTION_OPERATIONS:
        derived = _validate_convolution(parsed)
    elif operation in NORMALIZATION_OPERATIONS:
        derived = _validate_normalization(parsed)
    else:
        derived = _validate_attention(parsed)
    parsed["derived"] = derived
    return parsed


def plan_kernel_stages(node: Mapping[str, Any]) -> NodePlan:
    """Plan all registry-declared stages for a parsed DNN node."""

    operation = node.get("operation")
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Hygon NN operation {operation!r}")
    if node.get("compute_data_type") != "float32":
        raise ValueError(f"{operation} requires float32 compute_data_type")
    if not isinstance(node.get("derived"), Mapping):
        raise ValueError("node must be produced by dispatch.nn.parse_node")
    if operation in CONVOLUTION_OPERATIONS:
        plan = _convolution_plan(node)
    elif operation in NORMALIZATION_OPERATIONS:
        plan = NodePlan(operation, (_normalization_stage(node),))
    elif operation == "sdpa":
        plan = NodePlan(operation, (_attention_forward_stage(node),))
    elif operation == "sdpa_backward":
        plan = _attention_backward_plan(node)
    elif operation == "sdpa_fp8":
        plan = _fp8_forward_plan(node)
    else:
        plan = _fp8_backward_plan(node)
    plan.validate_dependencies()
    return plan


kernel_stage_plan = plan_kernel_stages


def kernel_configurations(
    node: Mapping[str, Any],
) -> tuple[KernelStagePlan, ...]:
    """Return every stage while preserving dependency order."""

    return plan_kernel_stages(node).stages


def kernel_configuration(node: Mapping[str, Any]) -> KernelStagePlan:
    """Return the only stage for a single-stage operation.

    Multi-stage attention must use :func:`plan_kernel_stages`; silently
    dropping zeroing or dependency stages would be a correctness bug.
    """

    plan = plan_kernel_stages(node)
    if len(plan.stages) != 1:
        raise ValueError(
            f"{plan.operation} is multi-stage; use plan_kernel_stages"
        )
    return plan.stages[0]


# Public contracts shared by graph lowering and stage emission.

__all__ = [
    "ATTENTION_OPERATIONS",
    "ALLOW_TF32",
    "CONVOLUTION_OPERATIONS",
    "CONV_FPROP_YOLO_X_P5_GEMM_TUNING",
    "CONV_FPROP_STANDARD_3X3_TUNING",
    "DGRAD_EXACT_3X3_S1_WORKSPACE_CAP",
    "GridSpec",
    "KernelStagePlan",
    "NORMALIZATION_OPERATIONS",
    "NodePlan",
    "OPERATIONS",
    "OPERATION_SCHEMAS",
    "OperationSchema",
    "REGISTRY_FUNCTIONS",
    "SUPPORTED_OPERATIONS",
    "TuningMetadata",
    "WORKSPACE_ALIGNMENT",
    "STRICT_DOT_INPUT_PRECISION",
    "WorkspaceTensor",
    "kernel_configuration",
    "kernel_configurations",
    "kernel_stage_plan",
    "parse_node",
    "plan_kernel_stages",
    "tensor_storage_size",
]
