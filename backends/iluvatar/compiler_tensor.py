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

"""Graph-IR planning for Iluvatar tensor, reduction, and MatMul kernels.

This module is intentionally independent from :mod:`backends.iluvatar.compiler`.
The provider can import it without creating a circular dependency, while pure
Python schema/configuration tests can import it without Triton or a GPU.

The returned :class:`KernelConfiguration` describes only visible kernel
arguments.  The Iluvatar provider must append its global/profile scratch pointer
arguments when it renders the libtriton_jit ABI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ILUVATAR_WARP_SIZE = 64
MAX_WORKGROUP_SIZE = 1024
WORKSPACE_ALIGNMENT = 256
MAX_RANK = 8
MAX_BATCH_RANK = 6
MAX_I32 = 2**31 - 1
MAX_U32 = 2**32 - 1
MAX_I64 = 2**63 - 1

POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
    "boolean": "*i8",
}
FLOAT_TYPES = frozenset(("float32", "float16", "bfloat16"))
ELEMENT_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "boolean": 1,
}

LAYOUT_OPERATIONS = frozenset(("reshape", "transpose", "slice"))
REDUCTION_OPERATIONS = {
    "reduction_sum": 1,
    "reduction_avg": 2,
    "reduction_mul": 3,
}
MATMUL_OPERATIONS = frozenset(("matmul",))
SUPPORTED_OPERATIONS = frozenset(
    (*LAYOUT_OPERATIONS, *REDUCTION_OPERATIONS, *MATMUL_OPERATIONS)
)
# Alias kept deliberately simple for provider-side set unions.
OPERATIONS = SUPPORTED_OPERATIONS

ArgumentLayout = tuple[tuple[str, str | int | None], ...]
Grid = tuple[int, int, int]


@dataclass(frozen=True)
class TuningMetadata:
    """Registry-compatible autotuning metadata for one kernel family."""

    source: str
    table: str
    key: str
    strategy: str
    warmup: int = 5
    repetitions: int = 10
    meta_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationSchema:
    """Graph IR port and tuning contract for one lowered operation."""

    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    tuning: TuningMetadata


LAYOUT_TUNING = TuningMetadata(
    source="common.yaml",
    table="binary",
    key="n_elements",
    strategy="align32",
    meta_keys=("BLOCK_SIZE",),
)
REDUCTION_TUNING = TuningMetadata(
    source="common.yaml",
    table="reduction",
    key="output_elements",
    strategy="reduction",
    meta_keys=("BLOCK_M", "BLOCK_N"),
)
MATMUL_TUNING = TuningMetadata(
    source="common.yaml",
    table="matmul",
    key="m",
    strategy="matmul",
    meta_keys=("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"),
)

OPERATION_SCHEMAS: dict[str, OperationSchema] = {
    **{
        operation: OperationSchema(
            input_ports=("input",),
            output_ports=("output",),
            tuning=LAYOUT_TUNING,
        )
        for operation in LAYOUT_OPERATIONS
    },
    **{
        operation: OperationSchema(
            input_ports=("input",),
            output_ports=("output",),
            tuning=REDUCTION_TUNING,
        )
        for operation in REDUCTION_OPERATIONS
    },
    "matmul": OperationSchema(
        input_ports=("a", "b"),
        output_ports=("output",),
        tuning=MATMUL_TUNING,
    ),
}


@dataclass(frozen=True)
class GridSpec:
    """A tuning-aware launch-grid formula."""

    kind: str
    extents: tuple[int, ...]

    def evaluate(self, constants: Mapping[str, int | bool]) -> Grid:
        if self.kind == "linear":
            block = _positive_meta_integer(constants, "BLOCK_SIZE")
            result = (_ceil_div(self.extents[0], block), 1, 1)
        elif self.kind == "reduction_rows":
            block = _positive_meta_integer(constants, "BLOCK_M")
            result = (_ceil_div(self.extents[0], block), 1, 1)
        elif self.kind == "matmul":
            block_m = _positive_meta_integer(constants, "BLOCK_M")
            block_n = _positive_meta_integer(constants, "BLOCK_N")
            result = (
                _ceil_div(self.extents[0], block_m)
                * _ceil_div(self.extents[1], block_n),
                self.extents[2],
                1,
            )
        else:
            raise ValueError(f"unknown Iluvatar grid kind {self.kind!r}")
        _validate_grid(result)
        return result


@dataclass(frozen=True)
class KernelConfiguration:
    """One common-Triton stage ready for provider-side materialization."""

    operation: str
    function_name: str
    runtime_signature: dict[str, str]
    constants: dict[str, int | float | bool]
    default_grid: Grid
    argument_layout: ArgumentLayout
    tuning: TuningMetadata
    tuning_key_value: int
    default_num_warps: int
    default_num_stages: int
    workspace_alignment: int
    grid_spec: GridSpec

    @property
    def hidden_argument_layout(self) -> ArgumentLayout:
        """libtriton_jit scratch arguments appended by the provider."""

        return (
            ("global_scratch_pointer", None),
            ("profile_scratch_pointer", None),
        )

    @property
    def full_argument_layout(self) -> ArgumentLayout:
        return self.argument_layout + self.hidden_argument_layout

    def validate_launch(
        self,
        *,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ) -> None:
        """Validate a corex_71/wave64 Triton launch configuration."""

        _validate_launch(
            self.default_num_warps if num_warps is None else num_warps,
            self.default_num_stages if num_stages is None else num_stages,
        )

    def variant(
        self,
        meta: Mapping[str, object],
        *,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ) -> tuple[dict[str, int | float | bool], Grid]:
        """Apply tuning META and recompute the launch grid.

        The provider should use this for every autotune variant instead of
        reusing ``default_grid`` after BLOCK_* values change.
        """

        unknown = set(meta).difference(self.tuning.meta_keys)
        if unknown:
            raise ValueError(
                "tuning META contains unsupported keys: "
                + ", ".join(sorted(unknown))
            )
        constants = dict(self.constants)
        for name, value in meta.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"tuning META.{name} must be an integer")
            if value <= 0:
                raise ValueError(f"tuning META.{name} must be positive")
            if name.startswith("BLOCK_") and value & (value - 1):
                raise ValueError(f"tuning META.{name} must be a power of two")
            constants[name] = value
        self.validate_launch(num_warps=num_warps, num_stages=num_stages)
        return constants, self.grid_spec.evaluate(constants)

    def as_legacy_tuple(
        self,
    ) -> tuple[
        str,
        dict[str, str],
        dict[str, int | float | bool],
        Grid,
        list[tuple[str, str | int | None]],
    ]:
        """Return the tuple shape used by the existing NVIDIA planner."""

        return (
            self.function_name,
            dict(self.runtime_signature),
            dict(self.constants),
            self.default_grid,
            list(self.argument_layout),
        )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_integer(
    values: Mapping[str, Any],
    name: str,
    *,
    minimum: int = 1,
    maximum: int = MAX_I32,
) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"parameters.{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(
            f"parameters.{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _require_integer_list(
    values: Mapping[str, Any],
    name: str,
    length: int,
    *,
    minimum: int,
    maximum: int = MAX_I64,
) -> list[int]:
    raw = _require_sequence(values.get(name), f"parameters.{name}")
    if len(raw) != length:
        raise ValueError(f"parameters.{name} must contain {length} integers")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
        for value in raw
    ):
        raise ValueError(
            f"parameters.{name} values must be integers in "
            f"[{minimum}, {maximum}]"
        )
    return list(raw)


def _positive_meta_integer(values: Mapping[str, int | bool], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"kernel constant {name} must be a positive integer")
    return value


def _validate_launch(num_warps: int, num_stages: int) -> None:
    if (
        isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps <= 0
        or (num_warps & (num_warps - 1)) != 0
        or num_warps * ILUVATAR_WARP_SIZE > MAX_WORKGROUP_SIZE
    ):
        raise ValueError(
            "Iluvatar num_warps must be a positive power of two and "
            "num_warps * 64 <= 1024"
        )
    if (
        isinstance(num_stages, bool)
        or not isinstance(num_stages, int)
        or not 1 <= num_stages <= 32
    ):
        raise ValueError("Iluvatar num_stages must be in [1, 32]")


def _validate_grid(grid: Grid) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_U32
        for value in grid
    ):
        raise ValueError("Iluvatar launch grid dimensions must fit uint32")


def _checked_product(values: Sequence[int], name: str) -> int:
    result = math.prod(values)
    if result <= 0 or result > MAX_I64:
        raise ValueError(f"{name} is invalid or overflows int64")
    return result


def _has_non_overlapping_strides(
    dimensions: Sequence[int], strides: Sequence[int]
) -> bool:
    axes = sorted(
        (stride, dimension)
        for dimension, stride in zip(dimensions, strides, strict=True)
        if dimension > 1
    )
    required_span = 1
    for stride, dimension in axes:
        if stride < required_span:
            return False
        required_span += (dimension - 1) * stride
        if required_span > MAX_I64:
            return False
    return True


def _is_dense_layout(
    dimensions: Sequence[int], strides: Sequence[int]
) -> bool:
    """Return whether logical elements cover one dense physical allocation."""

    axes = sorted(
        (stride, dimension)
        for dimension, stride in zip(dimensions, strides, strict=True)
        if dimension > 1
    )
    expected_stride = 1
    for stride, dimension in axes:
        if stride != expected_stride:
            return False
        expected_stride *= dimension
    return True


def _is_row_major_contiguous(tensor: Mapping[str, Any]) -> bool:
    expected = 1
    for dimension, stride in zip(
        reversed(tensor["dimensions"]), reversed(tensor["strides"])
    ):
        if stride != expected:
            return False
        expected *= dimension
    return True


def _validate_tensor(tensor_value: object, name: str) -> Mapping[str, Any]:
    tensor = _require_object(tensor_value, name)
    uid = tensor.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"{name} UID is invalid")
    data_type = tensor.get("data_type")
    if not isinstance(data_type, str) or data_type not in POINTER_TYPES:
        raise ValueError(f"{name} data type is unsupported: {data_type!r}")
    dimensions = _require_sequence(
        tensor.get("dimensions"), f"{name}.dimensions"
    )
    strides = _require_sequence(tensor.get("strides"), f"{name}.strides")
    if len(dimensions) != len(strides) or len(dimensions) > MAX_RANK:
        raise ValueError(f"{name} rank must be in [0, {MAX_RANK}]")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_I32
        for value in dimensions
    ):
        raise ValueError(f"{name} dimensions must be positive int32 values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_I64
        for value in strides
    ):
        raise ValueError(f"{name} strides must be positive int64 values")
    if not _has_non_overlapping_strides(dimensions, strides):
        raise ValueError(f"{name} strides overlap")
    alignment = tensor.get("alignment", 16)
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
        or alignment > MAX_I32
        or alignment & (alignment - 1)
    ):
        raise ValueError(f"{name} alignment must be a positive power of two")
    is_virtual = tensor.get("virtual", False)
    if not isinstance(is_virtual, bool):
        raise ValueError(f"{name} virtual flag must be boolean")
    tensor_storage_size(tensor)
    return tensor


def tensor_storage_size(tensor: Mapping[str, Any]) -> int:
    """Return the positive-stride storage span in bytes."""

    data_type = tensor.get("data_type")
    if not isinstance(data_type, str) or data_type not in ELEMENT_SIZES:
        raise ValueError("tensor data type has no storage size")
    dimensions = tensor.get("dimensions")
    strides = tensor.get("strides")
    if not isinstance(dimensions, (list, tuple)) or not isinstance(
        strides, (list, tuple)
    ):
        raise ValueError("tensor dimensions and strides must be arrays")
    if len(dimensions) != len(strides):
        raise ValueError("tensor dimensions and strides have different ranks")
    elements = 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(dimensions, strides, strict=True)
    )
    size = elements * ELEMENT_SIZES[data_type]
    if size <= 0 or size > MAX_I64:
        raise ValueError("tensor storage size is invalid or overflows int64")
    return size


def workspace_layout(
    tensor_registry: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, tuple[int, int, int]], int]:
    """Pack graph virtual tensors with the Iluvatar 256-byte minimum alignment.

    The returned packed size excludes the provider-owned global scratch tail.
    Each entry is ``(offset, size, effective_alignment)``.
    """

    offset = 0
    result: dict[int, tuple[int, int, int]] = {}
    for uid, tensor_value in tensor_registry.items():
        tensor = _validate_tensor(tensor_value, f"tensor {uid}")
        if tensor["uid"] != uid:
            raise ValueError("tensor registry key does not match tensor UID")
        if not tensor.get("virtual", False):
            continue
        alignment = max(WORKSPACE_ALIGNMENT, int(tensor.get("alignment", 16)))
        offset = _ceil_div(offset, alignment) * alignment
        size = tensor_storage_size(tensor)
        if offset > MAX_I64 - size:
            raise ValueError("virtual tensor workspace overflows int64")
        result[uid] = (offset, size, alignment)
        offset += size
    if offset:
        offset = _ceil_div(offset, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT
    return result, offset


def _parse_port(
    port_value: object,
    expected_name: str,
    direction: str,
    tensor_registry: Mapping[int, Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    port = _require_object(port_value, f"node.{direction}")
    if port.get("name") != expected_name:
        raise ValueError(
            f"node {direction} port must be named {expected_name!r}"
        )
    optional = port.get("optional", False)
    if not isinstance(optional, bool) or optional:
        raise ValueError(
            "Iluvatar tensor operations require every tensor port"
        )
    uid = port.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"node {direction} UID is invalid")
    try:
        tensor = tensor_registry[uid]
    except KeyError as error:
        raise ValueError(
            f"node references unknown tensor UID {uid}"
        ) from error
    return uid, _validate_tensor(tensor, f"tensor {uid}")


def parse_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Parse one schema-v3 Graph IR node supported by this module."""

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
            f"Iluvatar tensor compiler does not support operation {operation!r}"
        )
    schema = OPERATION_SCHEMAS[operation]
    inputs = _require_sequence(node.get("inputs"), "node.inputs")
    outputs = _require_sequence(node.get("outputs"), "node.outputs")
    if len(inputs) != len(schema.input_ports) or len(outputs) != len(
        schema.output_ports
    ):
        raise ValueError(f"{operation} node port count is invalid")

    input_uids: list[int] = []
    input_tensors: list[Mapping[str, Any]] = []
    for index, expected_name in enumerate(schema.input_ports):
        uid, tensor = _parse_port(
            inputs[index],
            expected_name,
            f"input[{index}]",
            tensor_registry,
        )
        input_uids.append(uid)
        input_tensors.append(tensor)
    output_uids: list[int] = []
    output_tensors: list[Mapping[str, Any]] = []
    for index, expected_name in enumerate(schema.output_ports):
        uid, tensor = _parse_port(
            outputs[index],
            expected_name,
            f"output[{index}]",
            tensor_registry,
        )
        output_uids.append(uid)
        output_tensors.append(tensor)
    if set(input_uids).intersection(output_uids):
        raise ValueError(
            f"{operation} does not support in-place output aliasing"
        )

    compute_data_type = node.get("compute_data_type")
    if compute_data_type != "float32":
        raise ValueError(
            f"{operation} requires float32 compute_data_type on Iluvatar"
        )
    attributes = dict(
        _require_object(
            node.get("attributes"), f"graph.nodes[{position}].attributes"
        )
    )
    return {
        "id": node_id,
        "operation": operation,
        "schema": schema,
        "compute_data_type": compute_data_type,
        "parameters": attributes,
        "tensors": [*input_tensors, *output_tensors],
        "tensor_roles": [*schema.input_ports, *schema.output_ports],
        "input_uids": input_uids,
        "output_uids": output_uids,
    }


def _metadata_array(
    parameters: Mapping[str, Any], name: str, expected: Sequence[int]
) -> list[int]:
    result = _require_integer_list(
        parameters, name, len(expected), minimum=1, maximum=MAX_I64
    )
    if result != list(expected):
        raise ValueError(
            f"parameters.{name} is inconsistent with tensor metadata"
        )
    return result


def _layout_configuration(
    operation: str,
    parameters: Mapping[str, Any],
    tensors: Sequence[Mapping[str, Any]],
) -> KernelConfiguration:
    if len(tensors) != 2:
        raise ValueError("layout operation tensor count is invalid")
    input_tensor, output_tensor = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1:
        raise ValueError("layout input/output data types must match")
    pointer_type = POINTER_TYPES[data_types[0]]
    input_dimensions = list(input_tensor["dimensions"])
    input_strides = list(input_tensor["strides"])
    output_dimensions = list(output_tensor["dimensions"])
    output_strides = list(output_tensor["strides"])
    input_rank = len(input_dimensions)
    output_rank = len(output_dimensions)
    elements = _require_integer(parameters, "n_elements")
    if elements != _checked_product(output_dimensions, "layout output size"):
        raise ValueError(
            "parameters.n_elements is inconsistent with layout output"
        )
    _metadata_array(parameters, "input_dimensions", input_dimensions)
    _metadata_array(parameters, "input_strides", input_strides)
    _metadata_array(parameters, "output_dimensions", output_dimensions)
    _metadata_array(parameters, "output_strides", output_strides)

    input_base = 0
    logical_input_dimensions = input_dimensions
    logical_input_strides = input_strides
    if operation == "reshape":
        if (
            _require_integer(
                parameters, "input_rank", minimum=0, maximum=MAX_RANK
            )
            != input_rank
            or _require_integer(
                parameters, "output_rank", minimum=0, maximum=MAX_RANK
            )
            != output_rank
        ):
            raise ValueError("reshape rank parameters are inconsistent")
        _require_integer(parameters, "reshape_mode", minimum=2, maximum=2)
        if (
            _checked_product(input_dimensions, "reshape input size")
            != elements
        ):
            raise ValueError("reshape input/output element counts must match")
    elif operation == "transpose":
        if input_rank == 0 or input_rank != output_rank:
            raise ValueError("transpose ranks must match in [1, 8]")
        if (
            _require_integer(parameters, "rank", minimum=1, maximum=MAX_RANK)
            != input_rank
        ):
            raise ValueError("transpose rank parameter is inconsistent")
        permutation = _require_integer_list(
            parameters,
            "permutation",
            input_rank,
            minimum=0,
            maximum=input_rank - 1,
        )
        if sorted(permutation) != list(range(input_rank)):
            raise ValueError(
                "transpose permutation must contain each axis once"
            )
        if output_dimensions != [
            input_dimensions[axis] for axis in permutation
        ]:
            raise ValueError(
                "transpose output shape does not match permutation"
            )
        logical_input_dimensions = output_dimensions
        logical_input_strides = [input_strides[axis] for axis in permutation]
    elif operation == "slice":
        if input_rank == 0 or input_rank != output_rank:
            raise ValueError("slice ranks must match in [1, 8]")
        if (
            _require_integer(parameters, "rank", minimum=1, maximum=MAX_RANK)
            != input_rank
        ):
            raise ValueError("slice rank parameter is inconsistent")
        starts = _require_integer_list(
            parameters, "starts", input_rank, minimum=0
        )
        limits = _require_integer_list(
            parameters, "limits", input_rank, minimum=1
        )
        steps = _require_integer_list(
            parameters, "slice_strides", input_rank, minimum=1
        )
        expected_output: list[int] = []
        for axis in range(input_rank):
            if (
                starts[axis] >= limits[axis]
                or limits[axis] > input_dimensions[axis]
            ):
                raise ValueError("slice range is outside input shape")
            expected_output.append(
                _ceil_div(limits[axis] - starts[axis], steps[axis])
            )
        if output_dimensions != expected_output:
            raise ValueError("slice output shape does not match attributes")
        input_base = sum(
            start * stride
            for start, stride in zip(starts, input_strides, strict=True)
        )
        if input_base > MAX_I64:
            raise ValueError("slice input base overflows int64")
        logical_input_dimensions = output_dimensions
        logical_input_strides = [
            stride * step
            for stride, step in zip(input_strides, steps, strict=True)
        ]
        if any(value > MAX_I64 for value in logical_input_strides):
            raise ValueError("slice logical stride overflows int64")
    else:
        raise ValueError(f"unknown layout operation {operation!r}")

    block = 256
    function_name = "dense_copy_kernel"
    constants: dict[str, int | float | bool] = {"BLOCK_SIZE": block}
    if not (
        input_base == 0
        and logical_input_dimensions == output_dimensions
        and logical_input_strides == output_strides
        and _is_dense_layout(output_dimensions, output_strides)
    ):
        function_name = "layout_copy_kernel"
        input_leading = MAX_RANK - len(logical_input_dimensions)
        output_leading = MAX_RANK - output_rank
        padded_input_dimensions = [
            1
        ] * input_leading + logical_input_dimensions
        padded_input_strides = [0] * input_leading + logical_input_strides
        padded_output_dimensions = [1] * output_leading + output_dimensions
        padded_output_strides = [0] * output_leading + output_strides
        constants["INPUT_BASE"] = input_base
        for axis in range(MAX_RANK):
            constants[f"INPUT_DIM_{axis}"] = padded_input_dimensions[axis]
            constants[f"INPUT_STRIDE_{axis}"] = padded_input_strides[axis]
            constants[f"OUTPUT_DIM_{axis}"] = padded_output_dimensions[axis]
            constants[f"OUTPUT_STRIDE_{axis}"] = padded_output_strides[axis]
    return _configuration(
        operation=operation,
        function_name=function_name,
        runtime_signature={
            "input_ptr": pointer_type,
            "output_ptr": pointer_type,
            "n_elements": "i32",
        },
        constants=constants,
        argument_layout=(
            ("tensor", None),
            ("tensor", None),
            ("scalar_i32", "n_elements"),
        ),
        tuning=LAYOUT_TUNING,
        tuning_key_value=elements,
        default_num_warps=4,
        default_num_stages=1,
        grid_spec=GridSpec("linear", (elements,)),
    )


def _reduction_tensor_constants(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
    axis: int,
    keep_dimensions: bool,
) -> dict[str, int]:
    input_dimensions = list(input_tensor["dimensions"])
    rank = len(input_dimensions)
    logical_dimensions = list(input_dimensions)
    logical_dimensions[axis] = 1
    input_strides = list(input_tensor["strides"])
    reduction_stride = input_strides[axis]
    input_strides[axis] = 0
    if keep_dimensions:
        output_strides = list(output_tensor["strides"])
    else:
        output_strides = []
        output_axis = 0
        for input_axis in range(rank):
            if input_axis == axis:
                output_strides.append(0)
            else:
                output_strides.append(output_tensor["strides"][output_axis])
                output_axis += 1
    output_strides[axis] = 0
    leading = MAX_RANK - rank
    dimensions = [1] * leading + logical_dimensions
    input_strides = [0] * leading + input_strides
    output_strides = [0] * leading + output_strides
    constants: dict[str, int] = {"REDUCTION_STRIDE": reduction_stride}
    for padded_axis in range(MAX_RANK):
        constants[f"DIM_{padded_axis}"] = dimensions[padded_axis]
        constants[f"INPUT_STRIDE_{padded_axis}"] = input_strides[padded_axis]
        constants[f"OUTPUT_STRIDE_{padded_axis}"] = output_strides[padded_axis]
    return constants


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _reduction_configuration(
    operation: str,
    parameters: Mapping[str, Any],
    tensors: Sequence[Mapping[str, Any]],
) -> KernelConfiguration:
    if len(tensors) != 2:
        raise ValueError("reduction tensor count is invalid")
    input_tensor, output_tensor = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1 or data_types[0] not in FLOAT_TYPES:
        raise ValueError(
            "reduction tensors must use one matching floating data type"
        )
    input_dimensions = list(input_tensor["dimensions"])
    rank = len(input_dimensions)
    if rank == 0:
        raise ValueError("reduction input must have positive rank")
    axis = _require_integer(parameters, "axis", minimum=0, maximum=rank - 1)
    keep_value = _require_integer(
        parameters, "keep_dimensions", minimum=0, maximum=1
    )
    keep_dimensions = keep_value == 1
    expected_output = list(input_dimensions)
    if keep_dimensions:
        expected_output[axis] = 1
    else:
        del expected_output[axis]
    if list(output_tensor["dimensions"]) != expected_output:
        raise ValueError("reduction output shape is incorrect")

    expected_outer = _checked_product(
        input_dimensions[:axis], "reduction outer extent"
    )
    expected_extent = input_dimensions[axis]
    expected_inner = _checked_product(
        input_dimensions[axis + 1 :], "reduction inner extent"
    )
    expected_output_elements = expected_outer * expected_inner
    outer = _require_integer(parameters, "outer")
    extent = _require_integer(parameters, "reduction", maximum=65536)
    inner = _require_integer(parameters, "inner")
    output_elements = _require_integer(parameters, "output_elements")
    if (outer, extent, inner, output_elements) != (
        expected_outer,
        expected_extent,
        expected_inner,
        expected_output_elements,
    ):
        raise ValueError("reduction parameters are inconsistent with shape")
    if (
        _checked_product(expected_output, "reduction output size")
        != output_elements
    ):
        raise ValueError("reduction output element count is inconsistent")

    block_n = _next_power_of_two(extent)
    constants: dict[str, int | float | bool] = {
        "N": extent,
        "OP": REDUCTION_OPERATIONS[operation],
        "BLOCK_M": 1,
        "BLOCK_N": block_n,
    }
    signature = {
        "x_ptr": POINTER_TYPES[data_types[0]],
        "out_ptr": POINTER_TYPES[data_types[0]],
        "M": "i32",
    }
    contiguous = _is_row_major_contiguous(
        input_tensor
    ) and _is_row_major_contiguous(output_tensor)
    if contiguous and inner == 1:
        constants.update({"stride_xm": extent, "stride_xn": 1})
        function_name = "reduction_2d_kernel"
        rows = outer
        scalar_name = "outer"
    elif contiguous:
        constants.update(
            {
                "I": inner,
                "stride_xo": extent * inner,
                "stride_xr": inner,
                "stride_xi": 1,
            }
        )
        function_name = "reduction_3d_kernel"
        rows = output_elements
        scalar_name = "output_elements"
    else:
        block_m = min(16, max(1, 65536 // block_n))
        constants.update(
            _reduction_tensor_constants(
                input_tensor, output_tensor, axis, keep_dimensions
            )
        )
        constants["BLOCK_M"] = block_m
        function_name = "reduction_strided_kernel"
        rows = output_elements
        scalar_name = "output_elements"
    return _configuration(
        operation=operation,
        function_name=function_name,
        runtime_signature=signature,
        constants=constants,
        argument_layout=(
            ("tensor", None),
            ("tensor", None),
            ("scalar_i32", scalar_name),
        ),
        tuning=REDUCTION_TUNING,
        tuning_key_value=output_elements,
        default_num_warps=4,
        default_num_stages=1,
        grid_spec=GridSpec("reduction_rows", (rows,)),
    )


def _matmul_configuration(
    parameters: Mapping[str, Any],
    tensors: Sequence[Mapping[str, Any]],
) -> KernelConfiguration:
    if len(tensors) != 3:
        raise ValueError("matmul tensor count is invalid")
    a, b, output = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1 or data_types[0] not in FLOAT_TYPES:
        raise ValueError("matmul tensors must use one matching floating type")
    if any(
        not 2 <= len(tensor["dimensions"]) <= MAX_RANK for tensor in tensors
    ):
        raise ValueError("matmul tensor ranks must be in [2, 8]")
    m = a["dimensions"][-2]
    k = a["dimensions"][-1]
    if b["dimensions"][-2] != k:
        raise ValueError("matmul contraction dimensions do not match")
    n = b["dimensions"][-1]
    a_batch = list(a["dimensions"][:-2])
    b_batch = list(b["dimensions"][:-2])
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > MAX_BATCH_RANK:
        raise ValueError("matmul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        b_dimension = b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            raise ValueError("matmul batch dimensions are not broadcastable")
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    if list(output["dimensions"]) != [*batch_dimensions, m, n]:
        raise ValueError("matmul output shape is incorrect")
    batch = _checked_product(batch_dimensions, "matmul batch extent")
    if batch > MAX_I32:
        raise ValueError("matmul batch extent exceeds int32")
    for name, expected in (("batch", batch), ("m", m), ("n", n), ("k", k)):
        if _require_integer(parameters, name) != expected:
            raise ValueError(f"parameters.{name} is inconsistent with matmul")

    def batch_strides(tensor: Mapping[str, Any]) -> list[int]:
        tensor_batch = list(tensor["dimensions"][:-2])
        leading = batch_rank - len(tensor_batch)
        dimensions = [1] * leading + tensor_batch
        strides = [0] * leading + list(tensor["strides"][:-2])
        effective = [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        ]
        return [0] * (MAX_BATCH_RANK - batch_rank) + effective

    padded_dimensions = [1] * (MAX_BATCH_RANK - batch_rank) + batch_dimensions
    a_batch_strides = batch_strides(a)
    b_batch_strides = batch_strides(b)
    c_batch_strides = [0] * (MAX_BATCH_RANK - batch_rank) + list(
        output["strides"][:-2]
    )
    block_m = 32 if m < 64 else 64
    block_n = 32 if n < 64 else 64
    block_k = 32
    constants: dict[str, int | float | bool] = {
        "M": m,
        "N": n,
        "K": k,
        "A_STRIDE_M": a["strides"][-2],
        "A_STRIDE_K": a["strides"][-1],
        "B_STRIDE_K": b["strides"][-2],
        "B_STRIDE_N": b["strides"][-1],
        "C_STRIDE_M": output["strides"][-2],
        "C_STRIDE_N": output["strides"][-1],
        "INPUT_IS_FLOAT32": data_types[0] == "float32",
        # Iluvatar must not inherit CUDA TF32 semantics.
        "USE_TF32": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": 8,
    }
    for axis in range(MAX_BATCH_RANK):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        constants[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        constants[f"C_BATCH_STRIDE_{axis}"] = c_batch_strides[axis]
    pointer_type = POINTER_TYPES[data_types[0]]
    return _configuration(
        operation="matmul",
        function_name="matmul_strided_kernel",
        runtime_signature={
            "a_ptr": pointer_type,
            "b_ptr": pointer_type,
            "c_ptr": pointer_type,
        },
        constants=constants,
        argument_layout=(
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
        ),
        tuning=MATMUL_TUNING,
        tuning_key_value=m,
        default_num_warps=4,
        default_num_stages=2,
        grid_spec=GridSpec("matmul", (m, n, batch)),
    )


def _configuration(
    *,
    operation: str,
    function_name: str,
    runtime_signature: dict[str, str],
    constants: dict[str, int | float | bool],
    argument_layout: ArgumentLayout,
    tuning: TuningMetadata,
    tuning_key_value: int,
    default_num_warps: int,
    default_num_stages: int,
    grid_spec: GridSpec,
) -> KernelConfiguration:
    _validate_launch(default_num_warps, default_num_stages)
    default_grid = grid_spec.evaluate(constants)
    return KernelConfiguration(
        operation=operation,
        function_name=function_name,
        runtime_signature=runtime_signature,
        constants=constants,
        default_grid=default_grid,
        argument_layout=argument_layout,
        tuning=tuning,
        tuning_key_value=tuning_key_value,
        default_num_warps=default_num_warps,
        default_num_stages=default_num_stages,
        workspace_alignment=WORKSPACE_ALIGNMENT,
        grid_spec=grid_spec,
    )


def kernel_configuration(node: Mapping[str, Any]) -> KernelConfiguration:
    """Create a common-kernel configuration from :func:`parse_node` output."""

    operation = node.get("operation")
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"unsupported Iluvatar tensor operation {operation!r}"
        )
    if node.get("compute_data_type") != "float32":
        raise ValueError(f"{operation} requires float32 compute_data_type")
    parameters = _require_object(node.get("parameters"), "node.parameters")
    raw_tensors = _require_sequence(node.get("tensors"), "node.tensors")
    tensors = [
        _validate_tensor(tensor, f"node tensor[{index}]")
        for index, tensor in enumerate(raw_tensors)
    ]
    if operation in LAYOUT_OPERATIONS:
        return _layout_configuration(operation, parameters, tensors)
    if operation in REDUCTION_OPERATIONS:
        return _reduction_configuration(operation, parameters, tensors)
    if operation == "matmul":
        return _matmul_configuration(parameters, tensors)
    raise ValueError(f"unsupported Iluvatar tensor operation {operation!r}")


__all__ = [
    "ILUVATAR_WARP_SIZE",
    "KernelConfiguration",
    "LAYOUT_OPERATIONS",
    "MATMUL_OPERATIONS",
    "OPERATIONS",
    "OPERATION_SCHEMAS",
    "OperationSchema",
    "REDUCTION_OPERATIONS",
    "SUPPORTED_OPERATIONS",
    "TuningMetadata",
    "WORKSPACE_ALIGNMENT",
    "kernel_configuration",
    "parse_node",
    "tensor_storage_size",
    "workspace_layout",
]
