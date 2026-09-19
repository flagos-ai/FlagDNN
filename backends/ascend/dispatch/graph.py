"""Ascend dispatch graph implementation."""

from __future__ import annotations

import math
from .common import (
    BINARY_POINTWISE_MODES,
    BOOLEAN_COMPUTE_DATA_TYPE,
    COMPARISON_POINTWISE_OPERATIONS,
    ELEMENT_SIZES,
    GRAPH_WORKSPACE_ALIGNMENT,
    LOGICAL_POINTWISE_OPERATIONS,
    MAX_GRAPH_NODES,
    MAX_I32,
    MAX_I64,
    MAX_RANK,
    NUMERIC_COMPUTE_DATA_TYPE,
    NUMERIC_STORAGE_DATA_TYPES,
    PointwiseGraphPlan,
    PointwiseStagePlan,
    REDUCTION_MODES,
    SUPPORTED_LAYOUT_OPERATIONS,
    TERNARY_POINTWISE_MODES,
    TensorPlan,
    UNARY_POINTWISE_MODES,
    require_f32,
    require_integer,
    require_integer_list,
    require_list,
    require_object,
)
from .convolution import (
    _convolution_fprop_meta,
)
from .layout import (
    _layout_meta,
)
from .matmul import (
    _matmul_meta,
)
from .normalization import (
    _batchnorm_inference_meta,
    _batchnorm_training_meta,
)
from .pointwise import (
    _broadcast_dimensions,
    _can_use_contiguous_kernel,
    _can_use_ternary_contiguous_kernel,
    _can_use_unary_contiguous_kernel,
    _optional_mode,
    _parse_unary_attributes,
    _strided_meta,
    _ternary_strided_meta,
    _unary_strided_meta,
)
from .reduction import (
    _reduction_meta,
)
from .tensor import (
    _is_physically_dense,
    _is_row_major_tensor,
    _parse_named_ports,
    _parse_port,
    _parse_tensor_table,
)
from typing import (
    Any,
)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _workspace_layout(
    tensors: dict[int, TensorPlan],
) -> tuple[dict[int, tuple[int, int]], int]:
    result: dict[int, tuple[int, int]] = {}
    offset = 0
    for uid in sorted(tensors):
        tensor = tensors[uid]
        if not tensor.virtual:
            continue
        offset = _align_up(offset, GRAPH_WORKSPACE_ALIGNMENT)
        if tensor.storage_size > MAX_I64 - offset:
            raise ValueError("Graph workspace size exceeds int64 range")
        result[uid] = (offset, tensor.storage_size)
        offset += tensor.storage_size
    return result, (_align_up(offset, GRAPH_WORKSPACE_ALIGNMENT) if offset else 0)


def _argument_source(
    index: int,
    name: str,
    tensor: TensorPlan,
    workspace_layout: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    if tensor.virtual:
        offset, size = workspace_layout[tensor.uid]
        return {
            "index": index,
            "name": name,
            "source": "graph_workspace",
            "uid": tensor.uid,
            "offset": offset,
            "size": size,
            "alignment": GRAPH_WORKSPACE_ALIGNMENT,
        }
    return {
        "index": index,
        "name": name,
        "source": "binding",
        "uid": tensor.uid,
        "size": tensor.storage_size,
        "alignment": tensor.alignment,
    }


def _stage_workspace(
    stage_tensors: tuple[TensorPlan, ...],
    workspace_layout: dict[int, tuple[int, int]],
) -> dict[str, int]:
    ranges = [
        workspace_layout[tensor.uid] for tensor in stage_tensors if tensor.virtual
    ]
    if not ranges:
        return {"offset": 0, "size": 0, "alignment": 1}
    start = min(offset for offset, _ in ranges)
    end = max(offset + size for offset, size in ranges)
    return {
        "offset": start,
        "size": end - start,
        "alignment": GRAPH_WORKSPACE_ALIGNMENT,
    }


def plan_graph(graph_value: object) -> PointwiseGraphPlan:
    graph = require_object(graph_value, "graph")
    tensors = _parse_tensor_table(graph)
    nodes = require_list(graph.get("nodes"), "graph.nodes")
    node_count = graph.get("node_count")
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count != len(nodes)
        or node_count < 1
        or node_count > MAX_GRAPH_NODES
    ):
        raise ValueError("graph.node_count is invalid")

    node_positions: set[int] = set()
    producer_stages: dict[int, int] = {}
    parsed: list[dict[str, Any]] = []
    has_external_output = False

    for position, raw_node in enumerate(nodes):
        node = require_object(raw_node, f"graph.nodes[{position}]")
        node_id = node.get("id")
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
            or node_id >= node_count
            or node_id in node_positions
        ):
            raise ValueError("pointwise graph node IDs must be unique positions")
        node_positions.add(node_id)
        operation = node.get("type")
        if operation in BINARY_POINTWISE_MODES:
            kernel_family = "binary"
            pointwise_mode = BINARY_POINTWISE_MODES[operation]
            input_names = ("left", "right")
        elif operation in UNARY_POINTWISE_MODES:
            kernel_family = "unary"
            pointwise_mode = UNARY_POINTWISE_MODES[operation]
            input_names = ("input",)
        elif operation in TERNARY_POINTWISE_MODES:
            kernel_family = "ternary"
            pointwise_mode = TERNARY_POINTWISE_MODES[operation]
            input_names = ("a", "b", "t")
        elif operation in SUPPORTED_LAYOUT_OPERATIONS:
            kernel_family = "layout"
            pointwise_mode = 0
            input_names = ("input",)
        elif operation in REDUCTION_MODES:
            kernel_family = "reduction"
            pointwise_mode = REDUCTION_MODES[operation]
            input_names = ("input",)
        elif operation == "matmul":
            kernel_family = "matmul"
            pointwise_mode = 0
            input_names = ("a", "b")
        elif operation == "convolution_fprop":
            kernel_family = "convolution_fprop"
            pointwise_mode = 0
            input_names = ("input", "filter")
        elif operation == "batchnorm_inference":
            kernel_family = "batchnorm_inference"
            pointwise_mode = 0
            input_names = ("x", "mean", "inv_variance", "scale", "bias")
        elif operation == "batchnorm":
            kernel_family = "batchnorm"
            pointwise_mode = 0
            input_names = (
                "x",
                "scale",
                "bias",
                "previous_running_mean",
                "previous_running_variance",
            )
        elif operation == "rmsnorm":
            kernel_family = "rmsnorm"
            pointwise_mode = 0
            input_names = ("x", "scale", "bias")
        elif operation == "layernorm":
            kernel_family = "layernorm"
            pointwise_mode = 0
            input_names = ("x", "scale", "bias")
        else:
            raise ValueError(
                "Ascend compiler does not support operation " f"{operation!r}"
            )

        inputs = require_list(node.get("inputs"), f"graph.nodes[{position}].inputs")
        outputs = require_list(node.get("outputs"), f"graph.nodes[{position}].outputs")
        output_names = (
            ("y", "mean", "inv_variance")
            if kernel_family == "layernorm"
            else (
                (
                    "y",
                    "mean",
                    "inv_variance",
                    "next_running_mean",
                    "next_running_variance",
                )
                if kernel_family == "batchnorm"
                else (
                    ("y", "inv_variance")
                    if kernel_family == "rmsnorm"
                    else (
                        ("y",)
                        if kernel_family == "batchnorm_inference"
                        else ("output",)
                    )
                )
            )
        )
        if len(inputs) != len(input_names) or len(outputs) != len(output_names):
            raise ValueError(f"{operation} has invalid arity")
        if kernel_family == "batchnorm_inference":
            input_tensors = _parse_named_ports(
                inputs,
                description=f"graph.nodes[{position}].inputs",
                expected_names=input_names,
                tensors=tensors,
            )
        else:
            input_tensors = tuple(
                _parse_port(
                    inputs[index],
                    description=f"graph.nodes[{position}].inputs[{index}]",
                    expected_name=name,
                    tensors=tensors,
                )
                for index, name in enumerate(input_names)
            )
        output_tensors = tuple(
            _parse_port(
                outputs[index],
                description=f"graph.nodes[{position}].outputs[{index}]",
                expected_name=name,
                tensors=tensors,
            )
            for index, name in enumerate(output_names)
        )
        output = output_tensors[0]
        if kernel_family in {
            "matmul",
            "convolution_fprop",
            "batchnorm",
            "batchnorm_inference",
            "rmsnorm",
            "layernorm",
        }:
            input_uids = {tensor.uid for tensor in input_tensors}
            if len(input_uids) != len(input_tensors):
                raise ValueError(f"{operation} requires distinct input tensor UIDs")
            output_uids = {tensor.uid for tensor in output_tensors}
            if len(output_uids) != len(output_tensors) or input_uids & output_uids:
                raise ValueError(
                    f"{operation} requires distinct input/output tensor UIDs"
                )
        logical = operation in LOGICAL_POINTWISE_OPERATIONS
        comparison = operation in COMPARISON_POINTWISE_OPERATIONS
        ternary = kernel_family == "ternary"
        layout = kernel_family == "layout"
        reduction_family = kernel_family == "reduction"
        matmul_family = kernel_family == "matmul"
        convolution_family = kernel_family == "convolution_fprop"
        batchnorm_family = kernel_family == "batchnorm_inference"
        batchnorm_training_family = kernel_family == "batchnorm"
        rmsnorm_family = kernel_family == "rmsnorm"
        layernorm_family = kernel_family == "layernorm"
        if matmul_family:
            a, b = input_tensors
            if (
                a.data_type not in NUMERIC_STORAGE_DATA_TYPES
                or b.data_type != a.data_type
                or output.data_type != a.data_type
            ):
                raise ValueError(
                    "matmul A/B/output storage data types must match and be floating"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("matmul requires float32 compute")
        elif convolution_family:
            input_tensor, filter_tensor = input_tensors
            if (
                input_tensor.data_type not in NUMERIC_STORAGE_DATA_TYPES
                or filter_tensor.data_type != input_tensor.data_type
                or output.data_type != input_tensor.data_type
            ):
                raise ValueError(
                    "convolution input/filter/output storage data types must "
                    "match and be floating"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("convolution_fprop requires float32 compute")
        elif batchnorm_training_family:
            (
                x,
                scale,
                bias,
                previous_running_mean,
                previous_running_variance,
            ) = input_tensors
            (
                y,
                mean,
                inv_variance,
                next_running_mean,
                next_running_variance,
            ) = output_tensors
            if (
                x.data_type not in NUMERIC_STORAGE_DATA_TYPES
                or scale.data_type != x.data_type
                or bias.data_type != x.data_type
                or y.data_type != x.data_type
            ):
                raise ValueError(
                    "batchnorm X/scale/bias/Y storage data types must match "
                    "and be floating"
                )
            if any(
                statistic.data_type != "float32"
                for statistic in (
                    previous_running_mean,
                    previous_running_variance,
                    mean,
                    inv_variance,
                    next_running_mean,
                    next_running_variance,
                )
            ):
                raise ValueError(
                    "batchnorm running and saved statistics must use " "float32 storage"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("batchnorm requires float32 compute")
        elif batchnorm_family:
            x, mean, inv_variance, scale, bias = input_tensors
            if (
                x.data_type not in NUMERIC_STORAGE_DATA_TYPES
                or output.data_type != x.data_type
            ):
                raise ValueError(
                    "batchnorm_inference X/Y storage data types must match "
                    "and be floating"
                )
            if any(
                parameter.data_type != "float32"
                for parameter in (mean, inv_variance, scale, bias)
            ):
                raise ValueError(
                    "Ascend batchnorm_inference parameters must use float32 " "storage"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("batchnorm_inference requires float32 compute")
        elif rmsnorm_family:
            x, scale, bias = input_tensors
            inv_variance = output_tensors[1]
            if (
                x.data_type not in NUMERIC_STORAGE_DATA_TYPES
                or scale.data_type != x.data_type
                or bias.data_type != x.data_type
                or output.data_type != x.data_type
            ):
                raise ValueError(
                    "rmsnorm X/scale/bias/Y storage data types must match "
                    "and be floating"
                )
            if inv_variance.data_type != "float32":
                raise ValueError("rmsnorm inverse variance must use float32 storage")
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("rmsnorm requires float32 compute")
        elif layernorm_family:
            x, scale, bias = input_tensors
            mean, inv_variance = output_tensors[1:]
            if (
                x.data_type not in NUMERIC_STORAGE_DATA_TYPES
                or scale.data_type != x.data_type
                or bias.data_type != x.data_type
                or output.data_type != x.data_type
            ):
                raise ValueError(
                    "layernorm X/scale/bias/Y storage data types must match "
                    "and be floating"
                )
            if mean.data_type != "float32" or inv_variance.data_type != "float32":
                raise ValueError("layernorm statistics must use float32 storage")
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("layernorm requires float32 compute")
        elif layout:
            if input_tensors[0].data_type != output.data_type:
                raise ValueError("layout operation input/output data types must match")
            if node.get("compute_data_type") not in ELEMENT_SIZES:
                raise ValueError("layout compute data type is unsupported")
        elif reduction_family:
            if (
                input_tensors[0].data_type not in NUMERIC_STORAGE_DATA_TYPES
                or output.data_type != input_tensors[0].data_type
            ):
                raise ValueError(
                    "reduction input/output storage data types must match "
                    "and be floating"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("reduction requires float32 compute")
        elif ternary:
            if (
                input_tensors[0].data_type not in NUMERIC_STORAGE_DATA_TYPES
                or input_tensors[1].data_type != input_tensors[0].data_type
                or output.data_type != input_tensors[0].data_type
            ):
                raise ValueError(
                    "binary_select requires matching floating A/B/output "
                    "storage data types"
                )
            if input_tensors[2].data_type != "boolean":
                raise ValueError(
                    "binary_select requires a BOOLEAN predicate storage " "data type"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("binary_select requires float32 compute")
        elif comparison:
            if (
                any(
                    tensor.data_type not in NUMERIC_STORAGE_DATA_TYPES
                    for tensor in input_tensors
                )
                or input_tensors[0].data_type != input_tensors[1].data_type
            ):
                raise ValueError(
                    "comparison pointwise requires same floating input "
                    "storage data types"
                )
            if output.data_type != "boolean":
                raise ValueError("comparison pointwise requires BOOLEAN output storage")
            if node.get("compute_data_type") != BOOLEAN_COMPUTE_DATA_TYPE:
                raise ValueError(
                    "comparison pointwise requires BOOLEAN compute data type"
                )
        elif logical:
            if any(tensor.data_type != output.data_type for tensor in input_tensors):
                raise ValueError("pointwise input/output data types must match")
            if output.data_type != "boolean":
                raise ValueError(
                    "logical pointwise requires BOOLEAN storage data types"
                )
            if node.get("compute_data_type") != BOOLEAN_COMPUTE_DATA_TYPE:
                raise ValueError("logical pointwise requires BOOLEAN compute data type")
        else:
            if any(tensor.data_type != output.data_type for tensor in input_tensors):
                raise ValueError("pointwise input/output data types must match")
            if output.data_type not in NUMERIC_STORAGE_DATA_TYPES:
                raise ValueError(
                    "numeric pointwise storage data types must be floating"
                )
            if node.get("compute_data_type") != NUMERIC_COMPUTE_DATA_TYPE:
                raise ValueError("pointwise compute data type must be float32")
        reduction_meta: dict[str, int] = {}
        matmul_meta: dict[str, int] = {}
        convolution_meta: dict[str, int] = {}
        batchnorm_meta: dict[str, int] = {}
        batchnorm_training_meta: dict[str, int | float] = {}
        rmsnorm_meta: dict[str, int | float] = {}
        layernorm_meta: dict[str, int | float] = {}
        if matmul_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            if (
                set(attributes) - {"input_precision"} != {"batch", "m", "n", "k"}
                or attributes.get("input_precision", 0) != 0
            ):
                raise ValueError("matmul attributes are invalid")
            a, b = input_tensors
            if not (2 <= len(a.dimensions) <= MAX_RANK) or not (
                2 <= len(b.dimensions) <= MAX_RANK
            ):
                raise ValueError("matmul input ranks must be in [2, 8]")
            m = a.dimensions[-2]
            k = a.dimensions[-1]
            if b.dimensions[-2] != k:
                raise ValueError("matmul contraction dimensions do not match")
            n = b.dimensions[-1]
            a_batch = a.dimensions[:-2]
            b_batch = b.dimensions[:-2]
            batch_rank = max(len(a_batch), len(b_batch))
            batch_dimensions: list[int] = [1] * batch_rank
            for trailing in range(batch_rank):
                a_dimension = (
                    a_batch[len(a_batch) - 1 - trailing]
                    if trailing < len(a_batch)
                    else 1
                )
                b_dimension = (
                    b_batch[len(b_batch) - 1 - trailing]
                    if trailing < len(b_batch)
                    else 1
                )
                if a_dimension != b_dimension and a_dimension != 1 and b_dimension != 1:
                    raise ValueError(
                        "matmul batch dimensions are not broadcast-compatible"
                    )
                batch_dimensions[batch_rank - 1 - trailing] = max(
                    a_dimension, b_dimension
                )
            expected_output_dimensions = tuple(batch_dimensions) + (m, n)
            if output.dimensions != expected_output_dimensions:
                raise ValueError("matmul output shape is inconsistent")
            batch = math.prod(batch_dimensions)
            encoded = {
                name: require_integer(attributes, name)
                for name in ("batch", "m", "n", "k")
            }
            if encoded != {"batch": batch, "m": m, "n": n, "k": k}:
                raise ValueError("matmul attributes disagree with Graph tensors")
            matmul_meta = _matmul_meta(a, b, output, batch=batch, m=m, n=n, k=k)
            expected_dimensions = output.dimensions
        elif convolution_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            expected_attribute_names = {
                "spatial_rank",
                "groups",
                "n_outputs",
                "pre_padding",
                "post_padding",
                "stride",
                "dilation",
            }
            if (
                set(attributes) - {"input_precision"} != expected_attribute_names
                or attributes.get("input_precision", 0) != 0
            ):
                raise ValueError("convolution_fprop attributes are invalid")
            spatial_rank = require_integer(
                attributes, "spatial_rank", minimum=1, maximum=3
            )
            groups = require_integer(attributes, "groups")
            n_outputs = require_integer(attributes, "n_outputs")
            pre_padding = require_integer_list(
                attributes,
                "pre_padding",
                spatial_rank,
                maximum=MAX_I32,
            )
            post_padding = require_integer_list(
                attributes,
                "post_padding",
                spatial_rank,
                maximum=MAX_I32,
            )
            convolution_stride = require_integer_list(
                attributes,
                "stride",
                spatial_rank,
                minimum=1,
                maximum=MAX_I32,
            )
            dilation = require_integer_list(
                attributes,
                "dilation",
                spatial_rank,
                minimum=1,
                maximum=MAX_I32,
            )
            input_tensor, filter_tensor = input_tensors
            tensor_rank = spatial_rank + 2
            if any(
                len(tensor.dimensions) != tensor_rank
                for tensor in (input_tensor, filter_tensor, output)
            ):
                raise ValueError(
                    "convolution_fprop tensor ranks must equal spatial_rank + 2"
                )
            input_channels = input_tensor.dimensions[1]
            output_channels = filter_tensor.dimensions[0]
            if (
                input_channels % groups != 0
                or output_channels % groups != 0
                or filter_tensor.dimensions[1] != input_channels // groups
            ):
                raise ValueError("convolution_fprop channel metadata is invalid")
            expected_output = [input_tensor.dimensions[0], output_channels]
            for axis in range(spatial_rank):
                effective_filter = (
                    dilation[axis] * (filter_tensor.dimensions[axis + 2] - 1) + 1
                )
                numerator = (
                    input_tensor.dimensions[axis + 2]
                    + pre_padding[axis]
                    + post_padding[axis]
                    - effective_filter
                )
                if numerator < 0:
                    raise ValueError(
                        "convolution_fprop filter is larger than padded input"
                    )
                expected_output.append(numerator // convolution_stride[axis] + 1)
            if output.dimensions != tuple(expected_output):
                raise ValueError("convolution_fprop output shape is inconsistent")
            if n_outputs != math.prod(output.dimensions):
                raise ValueError(
                    "convolution_fprop n_outputs disagrees with Graph output"
                )
            convolution_meta = _convolution_fprop_meta(
                input_tensor,
                filter_tensor,
                output,
                spatial_rank=spatial_rank,
                groups=groups,
                pre_padding=pre_padding,
                post_padding=post_padding,
                stride=convolution_stride,
                dilation=dilation,
            )
            expected_dimensions = output.dimensions
        elif batchnorm_training_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            expected_attribute_names = {
                "n_elements",
                "batch",
                "channels",
                "spatial",
                "rank",
                "epsilon",
                "momentum",
                "dimensions",
                "x_strides",
                "y_strides",
            }
            if set(attributes) != expected_attribute_names:
                raise ValueError("batchnorm attributes are invalid")
            x, scale, bias, previous_mean, previous_variance = input_tensors
            y, mean, inv_variance, next_mean, next_variance = output_tensors
            rank = len(x.dimensions)
            if rank < 2 or rank > MAX_RANK:
                raise ValueError("batchnorm X rank must be in [2, 8]")
            batch = require_integer(attributes, "batch")
            channels = require_integer(attributes, "channels")
            spatial = require_integer(attributes, "spatial")
            encoded_rank = require_integer(
                attributes, "rank", minimum=2, maximum=MAX_RANK
            )
            n_elements = require_integer(attributes, "n_elements")
            if y.dimensions != x.dimensions:
                raise ValueError("batchnorm Y shape must match X")
            if batch != x.dimensions[0]:
                raise ValueError("batchnorm batch attribute disagrees with X")
            if channels != x.dimensions[1]:
                raise ValueError("batchnorm channels attribute disagrees with X")
            if spatial != math.prod(x.dimensions[2:]):
                raise ValueError("batchnorm spatial attribute disagrees with X")
            if encoded_rank != rank:
                raise ValueError("batchnorm rank attribute disagrees with X")
            if n_elements != math.prod(x.dimensions):
                raise ValueError("batchnorm n_elements attribute disagrees with X")
            if (
                tuple(
                    require_integer_list(
                        attributes, "dimensions", rank, minimum=1, maximum=MAX_I32
                    )
                )
                != x.dimensions
            ):
                raise ValueError("batchnorm dimensions attribute disagrees with X")
            if tuple(require_integer_list(attributes, "x_strides", rank)) != x.strides:
                raise ValueError("batchnorm x_strides attribute disagrees with X")
            if tuple(require_integer_list(attributes, "y_strides", rank)) != y.strides:
                raise ValueError("batchnorm y_strides attribute disagrees with Y")
            for parameter in (
                scale,
                bias,
                previous_mean,
                previous_variance,
                mean,
                inv_variance,
                next_mean,
                next_variance,
            ):
                if math.prod(parameter.dimensions) != channels:
                    raise ValueError(
                        "batchnorm parameters/statistics must contain "
                        "exactly channels elements"
                    )
                if not _is_row_major_tensor(parameter):
                    raise ValueError(
                        "batchnorm parameters/statistics must be contiguous"
                    )
            epsilon = require_f32(attributes, "epsilon")
            momentum = require_f32(attributes, "momentum")
            if epsilon <= 0.0:
                raise ValueError("batchnorm epsilon must be positive")
            if momentum < 0.0 or momentum > 1.0:
                raise ValueError("batchnorm momentum must be in [0, 1]")
            batchnorm_training_meta = _batchnorm_training_meta(
                x,
                y,
                batch=batch,
                channels=channels,
                spatial=spatial,
                epsilon=epsilon,
                momentum=momentum,
            )
            expected_dimensions = x.dimensions
        elif layernorm_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            normalized_attribute_names = {
                "rows",
                "normalized_elements",
                "epsilon",
            }
            attribute_names = set(attributes)
            if attribute_names not in (
                normalized_attribute_names,
                normalized_attribute_names | {"forward_phase"},
            ):
                raise ValueError("layernorm attributes are invalid")
            if (
                "forward_phase" in attributes
                and require_integer(attributes, "forward_phase") != 2
            ):
                raise ValueError("layernorm forward phase must be TRAINING")
            x, scale, bias = input_tensors
            y, mean, inv_variance = output_tensors
            rank = len(x.dimensions)
            if rank < 1 or rank > MAX_RANK:
                raise ValueError("layernorm X rank must be in [1, 8]")
            if y.dimensions != x.dimensions:
                raise ValueError("layernorm Y shape must match X")
            if not _is_row_major_tensor(x) or not _is_row_major_tensor(y):
                raise ValueError("layernorm X/Y must be contiguous")
            if (
                scale.dimensions != bias.dimensions
                or not _is_row_major_tensor(scale)
                or not _is_row_major_tensor(bias)
                or not scale.dimensions
                or len(scale.dimensions) > rank
            ):
                raise ValueError(
                    "layernorm scale/bias must be matching contiguous suffix tensors"
                )
            leading = rank - len(scale.dimensions)
            normalized_start: int | None = None
            normalized_elements = 1
            statistic_dimensions = list(x.dimensions)
            for axis, dimension in enumerate(x.dimensions):
                parameter_dimension = (
                    1 if axis < leading else scale.dimensions[axis - leading]
                )
                if parameter_dimension != 1:
                    if parameter_dimension != dimension:
                        raise ValueError("layernorm scale shape does not match X")
                    if normalized_start is None:
                        normalized_start = axis
                elif normalized_start is not None and dimension != 1:
                    raise ValueError(
                        "layernorm scale must describe a contiguous suffix"
                    )
                if normalized_start is not None:
                    normalized_elements *= dimension
                    statistic_dimensions[axis] = 1
            if (
                normalized_start is None
                or math.prod(scale.dimensions) != normalized_elements
                or math.prod(bias.dimensions) != normalized_elements
            ):
                raise ValueError("layernorm scale/bias size is invalid")
            rows = math.prod(x.dimensions) // normalized_elements
            if require_integer(attributes, "rows") != rows:
                raise ValueError("layernorm rows attribute disagrees with X")
            if (
                require_integer(attributes, "normalized_elements")
                != normalized_elements
            ):
                raise ValueError(
                    "layernorm normalized_elements attribute disagrees with X"
                )
            epsilon = require_f32(attributes, "epsilon")
            if epsilon <= 0.0:
                raise ValueError("layernorm epsilon must be positive")
            if (
                mean.dimensions != tuple(statistic_dimensions)
                or inv_variance.dimensions != tuple(statistic_dimensions)
                or not _is_row_major_tensor(mean)
                or not _is_row_major_tensor(inv_variance)
            ):
                raise ValueError("layernorm statistic metadata is invalid")
            layernorm_meta = {
                "ROWS": rows,
                "NORMALIZED_ELEMENTS": normalized_elements,
                "EPSILON": epsilon,
            }
            expected_dimensions = x.dimensions
        elif rmsnorm_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            normalized_attribute_names = {
                "rows",
                "normalized_elements",
                "epsilon",
            }
            attribute_names = set(attributes)
            if attribute_names not in (
                normalized_attribute_names,
                normalized_attribute_names | {"forward_phase"},
            ):
                raise ValueError("rmsnorm attributes are invalid")
            if (
                "forward_phase" in attributes
                and require_integer(attributes, "forward_phase") != 2
            ):
                raise ValueError("rmsnorm forward phase must be TRAINING")
            x, scale, bias = input_tensors
            y, inv_variance = output_tensors
            rank = len(x.dimensions)
            if rank < 1 or rank > MAX_RANK:
                raise ValueError("rmsnorm X rank must be in [1, 8]")
            if y.dimensions != x.dimensions:
                raise ValueError("rmsnorm Y shape must match X")
            if not _is_row_major_tensor(x) or not _is_row_major_tensor(y):
                raise ValueError("rmsnorm X/Y must be contiguous")
            if (
                scale.dimensions != bias.dimensions
                or not _is_row_major_tensor(scale)
                or not _is_row_major_tensor(bias)
                or not scale.dimensions
                or len(scale.dimensions) > rank
            ):
                raise ValueError(
                    "rmsnorm scale/bias must be matching contiguous suffix tensors"
                )
            leading = rank - len(scale.dimensions)
            normalized_start: int | None = None
            normalized_elements = 1
            statistic_dimensions = list(x.dimensions)
            for axis, dimension in enumerate(x.dimensions):
                parameter_dimension = (
                    1 if axis < leading else scale.dimensions[axis - leading]
                )
                if parameter_dimension != 1:
                    if parameter_dimension != dimension:
                        raise ValueError("rmsnorm scale shape does not match X")
                    if normalized_start is None:
                        normalized_start = axis
                elif normalized_start is not None and dimension != 1:
                    raise ValueError("rmsnorm scale must describe a contiguous suffix")
                if normalized_start is not None:
                    normalized_elements *= dimension
                    statistic_dimensions[axis] = 1
            if (
                normalized_start is None
                or math.prod(scale.dimensions) != normalized_elements
                or math.prod(bias.dimensions) != normalized_elements
            ):
                raise ValueError("rmsnorm scale/bias size is invalid")
            rows = math.prod(x.dimensions) // normalized_elements
            if require_integer(attributes, "rows") != rows:
                raise ValueError("rmsnorm rows attribute disagrees with X")
            if (
                require_integer(attributes, "normalized_elements")
                != normalized_elements
            ):
                raise ValueError(
                    "rmsnorm normalized_elements attribute disagrees with X"
                )
            epsilon = require_f32(attributes, "epsilon")
            if epsilon <= 0.0:
                raise ValueError("rmsnorm epsilon must be positive")
            if inv_variance.dimensions != tuple(
                statistic_dimensions
            ) or not _is_row_major_tensor(inv_variance):
                raise ValueError("rmsnorm inverse variance metadata is invalid")
            rmsnorm_meta = {
                "ROWS": rows,
                "NORMALIZED_ELEMENTS": normalized_elements,
                "EPSILON": epsilon,
            }
            expected_dimensions = x.dimensions
        elif batchnorm_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            expected_attribute_names = {
                "n_elements",
                "channels",
                "spatial",
                "rank",
                "dimensions",
                "x_strides",
                "y_strides",
            }
            if set(attributes) != expected_attribute_names:
                raise ValueError("batchnorm_inference attributes are invalid")
            x = input_tensors[0]
            rank = len(x.dimensions)
            if rank < 2 or rank > MAX_RANK:
                raise ValueError("batchnorm_inference X rank must be in [2, 8]")
            channels = require_integer(attributes, "channels")
            spatial = require_integer(attributes, "spatial")
            encoded_rank = require_integer(
                attributes, "rank", minimum=2, maximum=MAX_RANK
            )
            if output.dimensions != x.dimensions:
                raise ValueError("batchnorm_inference Y shape must match X")
            if channels != x.dimensions[1]:
                raise ValueError(
                    "batchnorm_inference channels attribute disagrees with X"
                )
            if spatial != math.prod(x.dimensions[2:]):
                raise ValueError(
                    "batchnorm_inference spatial attribute disagrees with X"
                )
            if encoded_rank != rank:
                raise ValueError("batchnorm_inference rank attribute disagrees with X")
            if (
                tuple(
                    require_integer_list(
                        attributes, "dimensions", rank, minimum=1, maximum=MAX_I32
                    )
                )
                != x.dimensions
            ):
                raise ValueError(
                    "batchnorm_inference dimensions attribute disagrees with X"
                )
            if tuple(require_integer_list(attributes, "x_strides", rank)) != x.strides:
                raise ValueError(
                    "batchnorm_inference x_strides attribute disagrees with X"
                )
            if (
                tuple(require_integer_list(attributes, "y_strides", rank))
                != output.strides
            ):
                raise ValueError(
                    "batchnorm_inference y_strides attribute disagrees with Y"
                )
            for parameter in input_tensors[1:]:
                if math.prod(parameter.dimensions) != channels:
                    raise ValueError(
                        "batchnorm_inference parameters must contain exactly "
                        "channels elements"
                    )
                if not _is_row_major_tensor(parameter):
                    raise ValueError(
                        "batchnorm_inference parameters must be contiguous"
                    )
            batchnorm_meta = _batchnorm_inference_meta(
                x, output, channels=channels, spatial=spatial
            )
            expected_dimensions = x.dimensions
        elif reduction_family:
            attributes = require_object(
                node.get("attributes"),
                f"graph.nodes[{position}].attributes",
            )
            expected_attribute_names = {
                "mode",
                "axis",
                "keep_dimensions",
                "outer",
                "reduction",
                "inner",
                "output_elements",
            }
            if set(attributes) != expected_attribute_names:
                raise ValueError("reduction attributes are invalid")
            mode = require_integer(attributes, "mode", minimum=0, maximum=2)
            if mode != REDUCTION_MODES[operation]:
                raise ValueError("reduction mode attribute disagrees with operation")
            rank = len(input_tensors[0].dimensions)
            if rank < 1:
                raise ValueError("reduction input rank must be in [1, 8]")
            raw_axis = attributes.get("axis")
            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                raise ValueError("parameters.axis must be an integer")
            if raw_axis < -rank or raw_axis >= rank:
                raise ValueError("parameters.axis is outside input rank")
            axis = raw_axis + rank if raw_axis < 0 else raw_axis
            keep_dimensions = require_integer(
                attributes, "keep_dimensions", minimum=0, maximum=1
            )
            outer = require_integer(attributes, "outer")
            reduction = require_integer(attributes, "reduction")
            inner = require_integer(attributes, "inner")
            output_elements = require_integer(attributes, "output_elements")
            expected_outer = math.prod(input_tensors[0].dimensions[:axis])
            expected_reduction = input_tensors[0].dimensions[axis]
            expected_inner = math.prod(input_tensors[0].dimensions[axis + 1 :])
            expected_output_elements = expected_outer * expected_inner
            if (
                outer != expected_outer
                or reduction != expected_reduction
                or inner != expected_inner
                or output_elements != expected_output_elements
            ):
                raise ValueError("reduction decomposition attributes disagree")
            expected_output_dimensions = list(input_tensors[0].dimensions)
            if keep_dimensions:
                expected_output_dimensions[axis] = 1
            else:
                del expected_output_dimensions[axis]
            if output.dimensions != tuple(expected_output_dimensions):
                raise ValueError("reduction output shape is inconsistent")
            reduction_meta = _reduction_meta(
                input_tensors[0],
                output,
                axis=axis,
                keep_dimensions=keep_dimensions,
                outer=outer,
                reduction=reduction,
                inner=inner,
                output_elements=output_elements,
                mode=pointwise_mode,
            )
            expected_dimensions = output.dimensions
        elif layout:
            expected_dimensions = output.dimensions
        elif ternary:
            expected_dimensions = _broadcast_dimensions(
                input_tensors[0], input_tensors[1]
            )
            partial = TensorPlan(
                uid=0,
                data_type=input_tensors[0].data_type,
                dimensions=expected_dimensions,
                strides=tuple(0 for _ in expected_dimensions),
                alignment=1,
                virtual=True,
                storage_size=0,
            )
            expected_dimensions = _broadcast_dimensions(partial, input_tensors[2])
        elif operation == "sigmoid_backward":
            if not (
                input_tensors[0].dimensions
                == input_tensors[1].dimensions
                == output.dimensions
            ):
                raise ValueError(
                    "sigmoid_backward requires left, right, and output "
                    "exactly equal dimensions"
                )
            expected_dimensions = output.dimensions
        elif kernel_family == "binary":
            expected_dimensions = _broadcast_dimensions(
                input_tensors[0], input_tensors[1]
            )
        else:
            expected_dimensions = input_tensors[0].dimensions
        if output.dimensions != expected_dimensions:
            raise ValueError("pointwise output shape is inconsistent")

        attributes = require_object(
            node.get("attributes"), f"graph.nodes[{position}].attributes"
        )
        if matmul_family:
            n_elements = int(matmul_meta["BATCH"] * matmul_meta["M"] * matmul_meta["N"])
        elif convolution_family:
            n_elements = math.prod(output.dimensions)
        elif reduction_family:
            n_elements = int(reduction_meta["OUTPUT_ELEMENTS"])
        elif rmsnorm_family or layernorm_family:
            n_elements = math.prod(output.dimensions)
        else:
            n_elements = require_integer(attributes, "n_elements")
            if n_elements != math.prod(output.dimensions):
                raise ValueError("parameters.n_elements does not match output")
        if n_elements < 1 or n_elements > MAX_I32:
            raise ValueError("operation element count must fit the LTJ i32 ABI")
        alpha = 1.0
        negative_slope = 0.0
        lower_clip = 0.0
        upper_clip = 0.0
        has_upper_clip = 0
        swish_beta = 1.0
        elu_alpha = 1.0
        softplus_beta = 1.0
        layout_meta: dict[str, int] = {}
        if (
            matmul_family
            or convolution_family
            or reduction_family
            or batchnorm_training_family
            or batchnorm_family
            or rmsnorm_family
            or layernorm_family
        ):
            pass
        elif layout:
            layout_meta = _layout_meta(operation, attributes, input_tensors[0], output)
        elif kernel_family == "binary":
            encoded_mode = require_integer(
                attributes, "pointwise_mode", minimum=1, maximum=40
            )
            if encoded_mode != pointwise_mode:
                raise ValueError(
                    "parameters.pointwise_mode is inconsistent with binary " "operation"
                )
            _optional_mode(attributes, "mode", pointwise_mode)
            alpha = require_f32(attributes, "alpha", default=1.0)
            if operation not in {"add", "sub"} and alpha != 1.0:
                raise ValueError("pointwise alpha is only supported by add and sub")
        elif kernel_family == "unary":
            (
                negative_slope,
                lower_clip,
                upper_clip,
                has_upper_clip,
                swish_beta,
                elu_alpha,
                softplus_beta,
            ) = _parse_unary_attributes(operation, attributes, pointwise_mode)
        else:
            _optional_mode(attributes, "mode", pointwise_mode)
            _optional_mode(attributes, "pointwise_mode", pointwise_mode)

        for produced in output_tensors:
            if produced.uid in producer_stages:
                raise ValueError("graph tensor has more than one producer")
            producer_stages[produced.uid] = position
            has_external_output = has_external_output or not produced.virtual
        parsed.append(
            {
                "node_id": node_id,
                "operation": operation,
                "kernel_family": kernel_family,
                "pointwise_mode": pointwise_mode,
                "inputs": input_tensors,
                "output": output,
                "outputs": output_tensors,
                "n_elements": n_elements,
                "alpha": alpha,
                "negative_slope": negative_slope,
                "lower_clip": lower_clip,
                "upper_clip": upper_clip,
                "has_upper_clip": has_upper_clip,
                "swish_beta": swish_beta,
                "elu_alpha": elu_alpha,
                "softplus_beta": softplus_beta,
                "layout_meta": layout_meta,
                "matmul_meta": matmul_meta,
                "convolution_meta": convolution_meta,
                "reduction_meta": reduction_meta,
                "batchnorm_meta": batchnorm_meta,
                "batchnorm_training_meta": batchnorm_training_meta,
                "rmsnorm_meta": rmsnorm_meta,
                "layernorm_meta": layernorm_meta,
            }
        )

    for position, node in enumerate(parsed):
        for tensor in node["inputs"]:
            producer = producer_stages.get(tensor.uid)
            if tensor.virtual and producer is None:
                raise ValueError("virtual pointwise input has no producer")
            if producer is not None and producer >= position:
                raise ValueError("pointwise graph is not in topological order")
    if not any(not tensor.virtual for tensor in tensors.values()):
        raise ValueError("pointwise graph has no external bindings")
    if not has_external_output:
        raise ValueError("pointwise graph has no external output")

    next_internal_uid = max(tensors) + 1
    execution_nodes: list[dict[str, Any]] = []
    for node in parsed:
        inputs = node["inputs"]
        output = node["output"]
        convolution_meta = node["convolution_meta"]
        im2col_fits_i32 = False
        im2col_block_offsets_fit_i32 = False
        if node["kernel_family"] == "convolution_fprop":
            im2col_elements = (
                inputs[0].dimensions[0]
                * math.prod(output.dimensions[2:])
                * int(convolution_meta["CHANNELS_PER_GROUP"])
                * int(convolution_meta["FILTER_DIM_2"])
                * int(convolution_meta["FILTER_DIM_3"])
                * int(convolution_meta["FILTER_DIM_4"])
            )
            # The raw LTJ ABI carries n_elements as i32. A direct convolution
            # can still represent a graph whose expanded columns tensor cannot,
            # so keep that graph valid and decline only the im2col rewrite.
            im2col_fits_i32 = im2col_elements <= MAX_I32
            # Ascend Triton's block-pointer offsets are signed i32 even though
            # task arithmetic is intentionally i64. Include the largest fixed
            # block overread used by this kernel and keep extreme, but valid,
            # convolutions on the direct implementation.
            width_extent = (
                (int(convolution_meta["OUTPUT_DIM_4"]) - 1)
                * int(convolution_meta["CONV_STRIDE_2"])
                + (int(convolution_meta["FILTER_DIM_4"]) - 1)
                * int(convolution_meta["DILATION_2"])
                + 255
            )
            im2col_block_offsets_fit_i32 = width_extent <= (
                MAX_I32 + int(convolution_meta["PRE_PADDING_2"])
            )
        use_1d_im2col = (
            node["kernel_family"] == "convolution_fprop"
            and im2col_fits_i32
            and im2col_block_offsets_fit_i32
            and next_internal_uid <= MAX_I64
            and int(convolution_meta["SPATIAL_RANK"]) == 1
            and int(convolution_meta["GROUPS"]) == 1
            and _is_row_major_tensor(inputs[0])
            and _is_row_major_tensor(inputs[1])
            and len(output.dimensions) == 3
            and _is_physically_dense(output)
            and output.strides
            == (
                output.dimensions[1] * output.dimensions[2],
                1,
                output.dimensions[1],
            )
        )
        use_2d_im2col = (
            node["kernel_family"] == "convolution_fprop"
            and im2col_fits_i32
            and im2col_block_offsets_fit_i32
            and next_internal_uid <= MAX_I64
            and int(convolution_meta["SPATIAL_RANK"]) == 2
            and int(convolution_meta["GROUPS"]) == 1
            and (output.dimensions[-1] % 32 == 0 or int(node["n_elements"]) >= 65536)
            and _is_row_major_tensor(inputs[0])
            and _is_row_major_tensor(inputs[1])
            and _is_row_major_tensor(output)
        )
        use_3d_im2col = (
            node["kernel_family"] == "convolution_fprop"
            and im2col_fits_i32
            and im2col_block_offsets_fit_i32
            and next_internal_uid <= MAX_I64
            and int(convolution_meta["SPATIAL_RANK"]) == 3
            and int(convolution_meta["GROUPS"]) == 1
            and _is_row_major_tensor(inputs[0])
            and _is_row_major_tensor(inputs[1])
            and _is_row_major_tensor(output)
        )
        if not (use_1d_im2col or use_2d_im2col or use_3d_im2col):
            execution_nodes.append({**node, "convolution_im2col": False})
            continue

        input_tensor = inputs[0]
        filter_tensor = inputs[1]
        batch = input_tensor.dimensions[0]
        output_spatial = math.prod(output.dimensions[2:])
        output_channels = output.dimensions[1]

        # A dense group-one 1x1 convolution with unit stride and no padding is
        # already an [O, C] x [N, C, HW] matrix multiplication.  Do not copy
        # the input into an identical im2col workspace before invoking the
        # dedicated Ascend MatMul kernel.  Keep small convolutions on the
        # direct convolution kernel; this rewrite is intentionally limited to
        # shapes which would otherwise enter the 2D im2col pipeline.
        use_2d_pointwise_matmul = (
            use_2d_im2col
            and int(convolution_meta["FILTER_DIM_3"]) == 1
            and int(convolution_meta["FILTER_DIM_4"]) == 1
            and int(convolution_meta["PRE_PADDING_1"]) == 0
            and int(convolution_meta["PRE_PADDING_2"]) == 0
            and int(convolution_meta["POST_PADDING_1"]) == 0
            and int(convolution_meta["POST_PADDING_2"]) == 0
            and int(convolution_meta["CONV_STRIDE_1"]) == 1
            and int(convolution_meta["CONV_STRIDE_2"]) == 1
            and int(convolution_meta["DILATION_1"]) == 1
            and int(convolution_meta["DILATION_2"]) == 1
        )
        if use_2d_pointwise_matmul:
            left_view = TensorPlan(
                uid=filter_tensor.uid,
                data_type=filter_tensor.data_type,
                dimensions=(output_channels, input_tensor.dimensions[1]),
                strides=(filter_tensor.strides[0], filter_tensor.strides[1]),
                alignment=filter_tensor.alignment,
                virtual=filter_tensor.virtual,
                storage_size=filter_tensor.storage_size,
            )
            right_view = TensorPlan(
                uid=input_tensor.uid,
                data_type=input_tensor.data_type,
                dimensions=(
                    batch,
                    input_tensor.dimensions[1],
                    output_spatial,
                ),
                strides=(
                    input_tensor.strides[0],
                    input_tensor.strides[1],
                    1,
                ),
                alignment=input_tensor.alignment,
                virtual=input_tensor.virtual,
                storage_size=input_tensor.storage_size,
            )
            output_view = TensorPlan(
                uid=output.uid,
                data_type=output.data_type,
                dimensions=(batch, output_channels, output_spatial),
                strides=(output.strides[0], output.strides[1], 1),
                alignment=output.alignment,
                virtual=output.virtual,
                storage_size=output.storage_size,
            )
            execution_nodes.append(
                {
                    **node,
                    "operation": "matmul",
                    "kernel_family": "matmul",
                    "pointwise_mode": 0,
                    "inputs": (left_view, right_view),
                    "outputs": (output_view,),
                    "output": output_view,
                    "n_elements": math.prod(output_view.dimensions),
                    "matmul_meta": _matmul_meta(
                        left_view,
                        right_view,
                        output_view,
                        batch=batch,
                        m=output_channels,
                        n=output_spatial,
                        k=input_tensor.dimensions[1],
                    ),
                    "convolution_im2col": False,
                }
            )
            continue

        def dense_virtual_tensor(uid: int, dimensions: tuple[int, ...]) -> TensorPlan:
            strides = [1] * len(dimensions)
            for axis in range(len(dimensions) - 2, -1, -1):
                strides[axis] = strides[axis + 1] * dimensions[axis + 1]
            elements = math.prod(dimensions)
            element_size = ELEMENT_SIZES[input_tensor.data_type]
            if elements > MAX_I64 // element_size:
                raise ValueError("internal convolution workspace exceeds int64")
            return TensorPlan(
                uid=uid,
                data_type=input_tensor.data_type,
                dimensions=dimensions,
                strides=tuple(strides),
                alignment=16,
                virtual=True,
                storage_size=elements * element_size,
            )

        reduction_extent = (
            int(convolution_meta["CHANNELS_PER_GROUP"])
            * int(convolution_meta["FILTER_DIM_2"])
            * int(convolution_meta["FILTER_DIM_3"])
            * int(convolution_meta["FILTER_DIM_4"])
        )
        dense_columns = dense_virtual_tensor(
            next_internal_uid,
            (batch, output_spatial, reduction_extent),
        )
        columns = TensorPlan(
            uid=dense_columns.uid,
            data_type=dense_columns.data_type,
            dimensions=dense_columns.dimensions,
            strides=(
                output_spatial * reduction_extent,
                1,
                output_spatial,
            ),
            alignment=dense_columns.alignment,
            virtual=True,
            storage_size=dense_columns.storage_size,
        )
        tensors[columns.uid] = columns
        next_internal_uid += 1

        # The physical columns layout is [batch, K, spatial].  For dense NCHW
        # outputs, compute [O, K] x [K, spatial] so MatMul writes the output
        # contiguously.  The 1D channels-last path keeps [spatial, K] x [K, O].
        if use_2d_im2col or use_3d_im2col:
            left_view = TensorPlan(
                uid=filter_tensor.uid,
                data_type=filter_tensor.data_type,
                dimensions=(output_channels, reduction_extent),
                strides=(reduction_extent, 1),
                alignment=filter_tensor.alignment,
                virtual=filter_tensor.virtual,
                storage_size=filter_tensor.storage_size,
            )
            right_view = TensorPlan(
                uid=columns.uid,
                data_type=columns.data_type,
                dimensions=(batch, reduction_extent, output_spatial),
                strides=(
                    output_spatial * reduction_extent,
                    output_spatial,
                    1,
                ),
                alignment=columns.alignment,
                virtual=True,
                storage_size=columns.storage_size,
            )
            matmul_m = output_channels
            matmul_n = output_spatial
            output_view = TensorPlan(
                uid=output.uid,
                data_type=output.data_type,
                dimensions=(batch, output_channels, output_spatial),
                strides=(output.strides[0], output.strides[1], 1),
                alignment=output.alignment,
                virtual=output.virtual,
                storage_size=output.storage_size,
            )
        else:
            left_view = columns
            right_view = TensorPlan(
                uid=filter_tensor.uid,
                data_type=filter_tensor.data_type,
                dimensions=(reduction_extent, output_channels),
                strides=(1, reduction_extent),
                alignment=filter_tensor.alignment,
                virtual=filter_tensor.virtual,
                storage_size=filter_tensor.storage_size,
            )
            matmul_m = output_spatial
            matmul_n = output_channels
            output_view = TensorPlan(
                uid=output.uid,
                data_type=output.data_type,
                dimensions=(batch, output_spatial, output_channels),
                strides=(
                    output.strides[0],
                    output.strides[-1],
                    output.strides[1],
                ),
                alignment=output.alignment,
                virtual=output.virtual,
                storage_size=output.storage_size,
            )
        im2col_node = {
            **node,
            "outputs": (columns,),
            "output": columns,
            "n_elements": math.prod(columns.dimensions),
            "convolution_im2col": True,
        }
        matmul_node = {
            **node,
            "operation": "matmul",
            "kernel_family": "matmul",
            "pointwise_mode": 0,
            "inputs": (left_view, right_view),
            "outputs": (output_view,),
            "output": output_view,
            "n_elements": math.prod(output_view.dimensions),
            "matmul_meta": _matmul_meta(
                left_view,
                right_view,
                output_view,
                batch=batch,
                m=matmul_m,
                n=matmul_n,
                k=reduction_extent,
            ),
            "convolution_im2col": False,
        }
        execution_nodes.extend((im2col_node, matmul_node))

    workspace_layout, workspace_size = _workspace_layout(tensors)

    stages: list[PointwiseStagePlan] = []
    tensor_to_stage: dict[int, int] = {}
    for node in execution_nodes:
        stage_id = len(stages)
        inputs = node["inputs"]
        output = node["output"]
        outputs = node["outputs"]
        stage_tensors = (*inputs, *outputs)
        dependencies = tuple(
            sorted(
                {
                    tensor_to_stage[tensor.uid]
                    for tensor in inputs
                    if tensor.uid in tensor_to_stage
                }
            )
        )
        if node["kernel_family"] == "matmul":
            function_name = "matmul_strided_kernel"
            meta = {**node["matmul_meta"], "BLOCK_SIZE": 16}
            argument_names = ("a_ptr", "b_ptr", "output_ptr")
        elif node["kernel_family"] == "convolution_fprop":
            function_name = (
                "convolution_fprop_im2col_kernel"
                if node["convolution_im2col"]
                else "convolution_fprop_persistent_kernel"
            )
            meta = {**node["convolution_meta"], "BLOCK_SIZE": 256}
            argument_names = (
                (
                    "input_ptr",
                    "filter_ptr",
                    "columns_ptr",
                )
                if node["convolution_im2col"]
                else ("input_ptr", "filter_ptr", "output_ptr")
            )
        elif node["kernel_family"] == "batchnorm":
            function_name = "batchnorm_training_persistent_kernel"
            meta = {**node["batchnorm_training_meta"], "BLOCK_SIZE": 256}
            argument_names = (
                "x_ptr",
                "scale_ptr",
                "bias_ptr",
                "previous_running_mean_ptr",
                "previous_running_variance_ptr",
                "y_ptr",
                "mean_ptr",
                "inv_variance_ptr",
                "next_running_mean_ptr",
                "next_running_variance_ptr",
            )
        elif node["kernel_family"] == "layernorm":
            function_name = "layernorm_persistent_kernel"
            meta = {**node["layernorm_meta"], "BLOCK_SIZE": 256}
            argument_names = (
                "x_ptr",
                "scale_ptr",
                "bias_ptr",
                "y_ptr",
                "mean_ptr",
                "inv_variance_ptr",
            )
        elif node["kernel_family"] == "rmsnorm":
            function_name = "rmsnorm_persistent_kernel"
            meta = {**node["rmsnorm_meta"], "BLOCK_SIZE": 256}
            argument_names = (
                "x_ptr",
                "scale_ptr",
                "bias_ptr",
                "y_ptr",
                "inv_variance_ptr",
            )
        elif node["kernel_family"] == "batchnorm_inference":
            contiguous = all(
                _is_row_major_tensor(tensor) for tensor in (inputs[0], output)
            )
            function_name = (
                "batchnorm_inference_nchw_persistent_kernel"
                if contiguous
                else "batchnorm_inference_strided_persistent_kernel"
            )
            batchnorm_meta = node["batchnorm_meta"]
            meta = {
                name: batchnorm_meta[name] for name in ("RANK", "CHANNELS", "SPATIAL")
            }
            if not contiguous:
                meta.update(
                    {
                        name: value
                        for name, value in batchnorm_meta.items()
                        if name.startswith("DIM_")
                        or name.startswith("X_STRIDE_")
                        or name.startswith("Y_STRIDE_")
                    }
                )
            meta["BLOCK_SIZE"] = 256
            argument_names = (
                "x_ptr",
                "mean_ptr",
                "inv_variance_ptr",
                "scale_ptr",
                "bias_ptr",
                "y_ptr",
            )
        elif node["kernel_family"] == "layout":
            function_name = "layout_copy_kernel"
            meta = {**node["layout_meta"], "BLOCK_SIZE": 256}
            argument_names = ("input_ptr", "output_ptr")
        elif node["kernel_family"] == "reduction":
            contiguous = all(
                _is_row_major_tensor(tensor) for tensor in (inputs[0], output)
            )
            function_name = (
                "reduction_3d_persistent_kernel"
                if contiguous
                else "reduction_strided_persistent_kernel"
            )
            reduction_meta = node["reduction_meta"]
            meta = {
                name: reduction_meta[name]
                for name in (
                    "RANK",
                    "OUTPUT_RANK",
                    "AXIS",
                    "KEEP_DIMENSIONS",
                    "OUTER",
                    "REDUCTION_SIZE",
                    "INNER",
                    "OUTPUT_ELEMENTS",
                    "REDUCTION_MODE",
                )
            }
            if not contiguous:
                meta.update(
                    {
                        name: value
                        for name, value in reduction_meta.items()
                        if name.startswith("INPUT_")
                        or name.startswith("OUTPUT_DIM_")
                        or name.startswith("OUTPUT_STRIDE_")
                    }
                )
            meta["BLOCK_SIZE"] = 256
            argument_names = ("input_ptr", "output_ptr")
        elif node["kernel_family"] == "binary":
            contiguous = _can_use_contiguous_kernel(inputs[0], inputs[1], output)
            function_name = (
                "binary_contiguous_kernel" if contiguous else "binary_strided_kernel"
            )
            meta: dict[str, int | float] = {
                "OP_KIND": node["pointwise_mode"],
                "ALPHA": node["alpha"],
                "BLOCK_SIZE": 256,
            }
            if not contiguous:
                meta = {
                    **_strided_meta(inputs[0], inputs[1], output),
                    **meta,
                }
            argument_names = ("x_ptr", "y_ptr", "out_ptr")
        elif node["kernel_family"] == "unary":
            contiguous = _can_use_unary_contiguous_kernel(inputs[0], output)
            function_name = (
                "unary_pointwise_contiguous_kernel"
                if contiguous
                else "unary_pointwise_strided_kernel"
            )
            meta = {
                "OPERATION": node["pointwise_mode"],
                "negative_slope": node["negative_slope"],
                "lower_clip": node["lower_clip"],
                "upper_clip": node["upper_clip"],
                "HAS_UPPER_CLIP": node["has_upper_clip"],
                "SWISH_BETA": node["swish_beta"],
                "ELU_ALPHA": node["elu_alpha"],
                "SOFTPLUS_BETA": node["softplus_beta"],
                "BLOCK_SIZE": 256,
            }
            if not contiguous:
                meta = {**_unary_strided_meta(inputs[0], output), **meta}
            argument_names = ("in_ptr", "out_ptr")
        elif node["kernel_family"] == "ternary":
            contiguous = _can_use_ternary_contiguous_kernel(
                inputs[0], inputs[1], inputs[2], output
            )
            function_name = (
                "binary_select_contiguous_kernel"
                if contiguous
                else "binary_select_strided_kernel"
            )
            meta = {"BLOCK_SIZE": 256}
            if not contiguous:
                meta = {
                    **_ternary_strided_meta(inputs[0], inputs[1], inputs[2], output),
                    **meta,
                }
            argument_names = ("x_ptr", "y_ptr", "t_ptr", "out_ptr")
        else:
            raise ValueError("Ascend kernel family is unsupported")
        arguments = tuple(
            _argument_source(index, argument_names[index], tensor, workspace_layout)
            for index, tensor in enumerate(stage_tensors)
        ) + (
            {
                "index": len(stage_tensors),
                "name": "n_elements",
                "source": "scalar",
                "type": "i32",
                "value": node["n_elements"],
            },
        )
        stages.append(
            PointwiseStagePlan(
                stage_id=stage_id,
                kernel_family=node["kernel_family"],
                operation=node["operation"],
                pointwise_mode=node["pointwise_mode"],
                source_node_ids=(node["node_id"],),
                dependencies=dependencies,
                function_name=function_name,
                n_elements=node["n_elements"],
                alpha=node["alpha"],
                negative_slope=node["negative_slope"],
                lower_clip=node["lower_clip"],
                upper_clip=node["upper_clip"],
                has_upper_clip=node["has_upper_clip"],
                swish_beta=node["swish_beta"],
                elu_alpha=node["elu_alpha"],
                softplus_beta=node["softplus_beta"],
                tensors=stage_tensors,
                meta=meta,
                argument_sources=arguments,
                workspace=_stage_workspace(stage_tensors, workspace_layout),
            )
        )
        for produced in outputs:
            tensor_to_stage[produced.uid] = stage_id
    return PointwiseGraphPlan(tuple(stages), workspace_size, node_count)


plan_pointwise_graph = plan_graph
plan_binary_graph = plan_graph
