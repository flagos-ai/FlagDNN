from __future__ import annotations

from typing import Any, Optional

import torch

from flag_dnn.graph.plan import ExecutionPlan
from flag_dnn.graph.registry import get_op_schema
from flag_dnn.graph.tensor import canonical_dtype


def execute_plan(plan: ExecutionPlan, inputs: tuple[Any, ...]) -> Any:
    if len(inputs) != len(plan.graph.inputs):
        raise RuntimeError(
            f"compiled graph expected {len(plan.graph.inputs)} inputs, "
            f"got {len(inputs)}"
        )

    runtime_values: dict[int, Any] = {}
    for value_id, actual, spec in zip(
        plan.graph.inputs, inputs, plan.input_specs
    ):
        _validate_input(value_id, actual, spec)
        runtime_values[value_id] = actual

    for value_id, value in plan.graph.values.items():
        if value.is_constant:
            runtime_values[value_id] = value.const_value

    workspace = _allocate_workspace(plan, inputs)
    for step in plan.steps:
        schema = get_op_schema(step.op_type)
        step_inputs = [runtime_values[value_id] for value_id in step.inputs]
        attrs = dict(step.attrs)
        if workspace is not None:
            attrs["_workspace"] = workspace
            attrs["_memory_plan"] = plan.memory_plan
        result = schema.run(step_inputs, attrs)
        if len(step.outputs) == 1:
            runtime_values[step.outputs[0]] = result
        else:
            for output_id, output in zip(step.outputs, result):
                runtime_values[output_id] = output

    flat_outputs = [
        runtime_values[value_id] for value_id in plan.graph.outputs
    ]
    structure = plan.debug_info.get("output_structure")
    if structure is None:
        return (
            flat_outputs[0] if len(flat_outputs) == 1 else tuple(flat_outputs)
        )
    return unflatten_output(structure, flat_outputs)


def _allocate_workspace(
    plan: ExecutionPlan, inputs: tuple[Any, ...]
) -> Optional[torch.Tensor]:
    if plan.workspace_size <= 0:
        return None
    device = None
    for value in inputs:
        if isinstance(value, torch.Tensor):
            device = value.device
            break
    if device is None:
        return None
    return torch.empty(plan.workspace_size, dtype=torch.uint8, device=device)


def _validate_input(value_id: int, actual: Any, spec: Any) -> None:
    if not isinstance(actual, torch.Tensor):
        raise TypeError(f"graph input {value_id} must be a torch.Tensor")
    if tuple(actual.shape) != tuple(spec.shape):
        raise RuntimeError(
            f"graph input {spec.name or value_id} shape mismatch: "
            f"expected {spec.shape}, got {tuple(actual.shape)}"
        )
    actual_dtype = canonical_dtype(actual.dtype)
    if actual_dtype != spec.dtype:
        raise RuntimeError(
            f"graph input {spec.name or value_id} dtype mismatch: "
            f"expected {spec.dtype}, got {actual_dtype}"
        )
    if spec.stride is not None and tuple(actual.stride()) != tuple(
        spec.stride
    ):
        raise RuntimeError(
            f"graph input {spec.name or value_id} stride mismatch: "
            f"expected {spec.stride}, got {tuple(actual.stride())}"
        )
    if spec.device is not None:
        actual_device = str(actual.device)
        if actual_device != spec.device and not actual_device.startswith(
            spec.device + ":"
        ):
            raise RuntimeError(
                f"graph input {spec.name or value_id} device mismatch: "
                f"expected {spec.device}, got {actual_device}"
            )


def unflatten_output(structure: Any, flat_outputs: list[Any]) -> Any:
    index = 0

    def build(node: Any) -> Any:
        nonlocal index
        kind = node[0]
        if kind == "leaf":
            value = flat_outputs[index]
            index += 1
            return value
        if kind == "tuple":
            return tuple(build(child) for child in node[1])
        if kind == "list":
            return [build(child) for child in node[1]]
        if kind == "dict":
            return {key: build(child) for key, child in node[1]}
        raise RuntimeError(f"unknown graph output structure node: {kind}")

    result = build(structure)
    if index != len(flat_outputs):
        raise RuntimeError(
            "graph output structure did not consume all outputs"
        )
    return result
