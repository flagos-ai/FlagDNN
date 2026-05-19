from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from flag_dnn.graph.plan import ExecutionPlan
from flag_dnn.graph.registry import get_op_schema
from flag_dnn.graph.tensor import canonical_dtype


RunFn = Callable[[list[Any], dict[str, Any]], Any]


@dataclass(frozen=True)
class _InputCheck:
    value_id: int
    label: Any
    shape: tuple[Any, ...]
    dtype: str
    stride: Optional[tuple[int, ...]]
    device: Optional[str]


@dataclass(frozen=True)
class _PreparedStep:
    run_fn: RunFn
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class _InputSource:
    input_index: Optional[int] = None
    constant_value: Any = None

    def resolve(self, inputs: tuple[Any, ...]) -> Any:
        if self.input_index is not None:
            return inputs[self.input_index]
        return self.constant_value


@dataclass(frozen=True)
class _SingleStepFastPath:
    step: _PreparedStep
    input_sources: tuple[_InputSource, ...]

    def run(self, inputs: tuple[Any, ...]) -> Any:
        step_inputs = [
            source.resolve(inputs) for source in self.input_sources
        ]
        return self.step.run_fn(step_inputs, self.step.attrs)


@dataclass(frozen=True)
class _PreparedPlan:
    input_count: int
    input_checks: tuple[_InputCheck, ...]
    constants: dict[int, Any]
    steps: tuple[_PreparedStep, ...]
    flat_output_ids: tuple[int, ...]
    output_structure: Any
    fast_path: Optional[_SingleStepFastPath]


def execute_plan(plan: ExecutionPlan, inputs: tuple[Any, ...]) -> Any:
    prepared = _get_prepared_plan(plan)
    if len(inputs) != prepared.input_count:
        raise RuntimeError(
            f"compiled graph expected {prepared.input_count} inputs, "
            f"got {len(inputs)}"
        )

    for check, actual in zip(prepared.input_checks, inputs):
        _validate_prepared_input(check, actual)

    if prepared.fast_path is not None:
        return prepared.fast_path.run(inputs)

    runtime_values: dict[int, Any] = {
        check.value_id: actual
        for check, actual in zip(prepared.input_checks, inputs)
    }
    runtime_values.update(prepared.constants)

    workspace = _allocate_workspace(plan, inputs)
    for step in prepared.steps:
        step_inputs = [runtime_values[value_id] for value_id in step.inputs]
        attrs = step.attrs
        if workspace is not None:
            attrs = dict(attrs)
            attrs["_workspace"] = workspace
            attrs["_memory_plan"] = plan.memory_plan
        result = step.run_fn(step_inputs, attrs)
        if len(step.outputs) == 1:
            runtime_values[step.outputs[0]] = result
        else:
            for output_id, output in zip(step.outputs, result):
                runtime_values[output_id] = output

    flat_outputs = [
        runtime_values[value_id] for value_id in prepared.flat_output_ids
    ]
    return _format_outputs(prepared.output_structure, flat_outputs)


def _get_prepared_plan(plan: ExecutionPlan) -> _PreparedPlan:
    output_structure = plan.debug_info.get("output_structure")
    prepared = getattr(plan, "_prepared_executor", None)
    if prepared is None or prepared.output_structure != output_structure:
        prepared = _prepare_plan(plan, output_structure)
        setattr(plan, "_prepared_executor", prepared)
    return prepared


def _prepare_plan(
    plan: ExecutionPlan, output_structure: Any
) -> _PreparedPlan:
    input_checks = tuple(
        _make_input_check(value_id, spec)
        for value_id, spec in zip(plan.graph.inputs, plan.input_specs)
    )
    constants = {
        value_id: value.const_value
        for value_id, value in plan.graph.values.items()
        if value.is_constant
    }
    steps = tuple(
        _PreparedStep(
            run_fn=get_op_schema(step.op_type).run_fn,
            inputs=tuple(step.inputs),
            outputs=tuple(step.outputs),
            attrs=dict(step.attrs),
        )
        for step in plan.steps
    )
    flat_output_ids = tuple(plan.graph.outputs)
    fast_path = _make_single_step_fast_path(
        plan,
        steps,
        constants,
        output_structure,
    )
    return _PreparedPlan(
        input_count=len(plan.graph.inputs),
        input_checks=input_checks,
        constants=constants,
        steps=steps,
        flat_output_ids=flat_output_ids,
        output_structure=output_structure,
        fast_path=fast_path,
    )


def _make_input_check(value_id: int, spec: Any) -> _InputCheck:
    return _InputCheck(
        value_id=value_id,
        label=spec.name or value_id,
        shape=tuple(spec.shape),
        dtype=spec.dtype,
        stride=None if spec.stride is None else tuple(spec.stride),
        device=spec.device,
    )


def _make_single_step_fast_path(
    plan: ExecutionPlan,
    steps: tuple[_PreparedStep, ...],
    constants: dict[int, Any],
    output_structure: Any,
) -> Optional[_SingleStepFastPath]:
    if (
        len(steps) != 1
        or plan.workspace_size > 0
        or len(plan.graph.outputs) != 1
        or not _is_leaf_output(output_structure)
    ):
        return None

    step = steps[0]
    if len(step.outputs) != 1 or step.outputs[0] != plan.graph.outputs[0]:
        return None

    input_positions = {
        value_id: index for index, value_id in enumerate(plan.graph.inputs)
    }
    input_sources = []
    for value_id in step.inputs:
        if value_id in input_positions:
            input_sources.append(
                _InputSource(input_index=input_positions[value_id])
            )
        elif value_id in constants:
            input_sources.append(
                _InputSource(constant_value=constants[value_id])
            )
        else:
            return None
    return _SingleStepFastPath(step=step, input_sources=tuple(input_sources))


def _is_leaf_output(output_structure: Any) -> bool:
    return output_structure is None or output_structure == ("leaf",)


def _format_outputs(output_structure: Any, flat_outputs: list[Any]) -> Any:
    if output_structure is None:
        return (
            flat_outputs[0] if len(flat_outputs) == 1 else tuple(flat_outputs)
        )
    return unflatten_output(output_structure, flat_outputs)


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
    _validate_prepared_input(_make_input_check(value_id, spec), actual)


def _validate_prepared_input(check: _InputCheck, actual: Any) -> None:
    if not isinstance(actual, torch.Tensor):
        raise TypeError(f"graph input {check.value_id} must be a torch.Tensor")
    if tuple(actual.shape) != check.shape:
        raise RuntimeError(
            f"graph input {check.label} shape mismatch: "
            f"expected {check.shape}, got {tuple(actual.shape)}"
        )
    actual_dtype = canonical_dtype(actual.dtype)
    if actual_dtype != check.dtype:
        raise RuntimeError(
            f"graph input {check.label} dtype mismatch: "
            f"expected {check.dtype}, got {actual_dtype}"
        )
    if check.stride is not None and tuple(actual.stride()) != check.stride:
        raise RuntimeError(
            f"graph input {check.label} stride mismatch: "
            f"expected {check.stride}, got {tuple(actual.stride())}"
        )
    if check.device is not None:
        actual_device = str(actual.device)
        if actual_device != check.device and not actual_device.startswith(
            check.device + ":"
        ):
            raise RuntimeError(
                f"graph input {check.label} device mismatch: "
                f"expected {check.device}, got {actual_device}"
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
