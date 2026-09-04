"""Closed-schema parser for supported mthreads Graph requests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any

from .compiler_tensor import (
    MAX_I32,
    MAX_I64,
    TensorSpec,
    broadcast_dimensions,
    element_count,
    is_physically_dense,
    is_row_major_contiguous,
    parse_tensor_table,
    require_exact_keys,
    require_integer,
    require_list,
    require_number,
    require_object,
)


GRAPH_SCHEMA_VERSION = 3
TARGET_PATTERN = re.compile(
    r"^musa-[a-z0-9_.+-]+-cc([0-9]+)-w([0-9]+)$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_ROOT_KEYS = {
    "schema_version",
    "flagdnn_version",
    "backend",
    "target",
    "build_options",
    "graph",
    "compiler_identity",
}
_BUILD_OPTION_KEYS = {"heuristic_modes", "autotune"}
_GRAPH_KEYS = {"name", "tensor_count", "tensors", "node_count", "nodes"}
_NODE_KEYS = {
    "id",
    "type",
    "name",
    "compute_data_type",
    "inputs",
    "outputs",
    "attributes",
}
_BINARY_ATTRIBUTE_KEYS = {
    "mode",
    "alpha",
    "n_elements",
    "pointwise_mode",
}
_BINARY_OPERATIONS = {
    1: "add",
    17: "sub",
    18: "mul",
    19: "div",
    20: "min",
    21: "max",
    22: "mod",
    23: "pow",
    25: "cmp_eq",
    26: "cmp_neq",
    27: "cmp_gt",
    28: "cmp_ge",
    29: "cmp_lt",
    30: "cmp_le",
    31: "logical_and",
    32: "logical_or",
    40: "sigmoid_backward",
}
_UNARY_ATTRIBUTE_KEYS = {
    "mode",
    "relu_lower_clip",
    "relu_upper_clip",
    "relu_lower_clip_slope",
    "relu_upper_clip_set",
    "swish_beta",
    "elu_alpha",
    "softplus_beta",
    "n_elements",
    "has_upper_clip",
    "negative_slope",
    "lower_clip",
    "upper_clip",
}
_UNARY_OPERATIONS = {
    2: "relu",
    3: "sqrt",
    4: "erf",
    5: "identity",
    6: "exp",
    7: "log",
    8: "neg",
    9: "abs",
    10: "ceil",
    11: "cos",
    12: "floor",
    13: "rsqrt",
    14: "sin",
    15: "tan",
    16: "reciprocal",
    24: "logical_not",
    33: "sigmoid",
    34: "tanh",
    35: "elu",
    36: "gelu",
    37: "softplus",
    38: "swish",
    39: "gelu_approx_tanh",
}
_TERNARY_ATTRIBUTE_KEYS = {"mode", "n_elements"}
_TERNARY_MODE = 41
_LAYOUT_OPERATIONS = frozenset({"reshape", "transpose", "slice"})
_RESHAPE_ATTRIBUTE_KEYS = {
    "n_elements",
    "input_rank",
    "output_rank",
    "reshape_mode",
    "input_dimensions",
    "input_strides",
    "output_dimensions",
    "output_strides",
}
_TRANSPOSE_ATTRIBUTE_KEYS = {
    "n_elements",
    "rank",
    "permutation",
    "input_dimensions",
    "input_strides",
    "output_dimensions",
    "output_strides",
}
_SLICE_ATTRIBUTE_KEYS = {
    "n_elements",
    "rank",
    "starts",
    "limits",
    "slice_strides",
    "input_dimensions",
    "input_strides",
    "output_dimensions",
    "output_strides",
}
_REDUCTION_ATTRIBUTE_KEYS = {
    "mode",
    "axis",
    "keep_dimensions",
    "outer",
    "reduction",
    "inner",
    "output_elements",
}
_REDUCTION_OPERATIONS = {
    0: "reduction_sum",
    1: "reduction_avg",
    2: "reduction_mul",
}
_MATMUL_ATTRIBUTE_KEYS = {"batch", "m", "n", "k"}
_CONVOLUTION_OPERATIONS = frozenset(
    {
        "conv2d_fprop",
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    }
)
_CONVOLUTION_FPROP_ATTRIBUTE_KEYS = {
    "spatial_rank",
    "groups",
    "n_outputs",
    "pre_padding",
    "post_padding",
    "stride",
    "dilation",
}
_CONVOLUTION_BACKWARD_ATTRIBUTE_KEYS = (
    _CONVOLUTION_FPROP_ATTRIBUTE_KEYS | {"convolution_mode"}
)
_NORMALIZATION_ATTRIBUTE_KEYS = {
    "epsilon",
    "forward_phase",
    "normalized_elements",
    "rows",
}
_BATCHNORM_ATTRIBUTE_KEYS = {
    "batch",
    "channels",
    "dimensions",
    "epsilon",
    "momentum",
    "n_elements",
    "rank",
    "spatial",
    "x_strides",
    "y_strides",
}
_BATCHNORM_INFERENCE_ATTRIBUTE_KEYS = {
    "channels",
    "dimensions",
    "n_elements",
    "rank",
    "spatial",
    "x_strides",
    "y_strides",
}
_ATTENTION_OPERATIONS = frozenset(
    {"sdpa", "sdpa_backward", "sdpa_fp8", "sdpa_fp8_backward"}
)
_ATTENTION_ATTRIBUTE_KEYS = {
    "attn_scale",
    "attn_scale_set",
    "banded",
    "batch",
    "causal_top_left",
    "diagonal_alignment",
    "diagonal_band_left_bound",
    "diagonal_band_right_bound",
    "generate_stats",
    "has_bias",
    "has_dbias",
    "head_dimension",
    "heads",
    "key_heads",
    "left_bound_set",
    "max_diag",
    "min_diag",
    "q_per_k",
    "q_per_v",
    "reverse_causal",
    "right_bound_set",
    "sequence_kv",
    "sequence_q",
    "value_dimension",
    "value_heads",
}
_COMPARISON_MODES = frozenset(range(25, 31))
_LOGICAL_BINARY_MODES = frozenset({31, 32})
_FLOATING_DATA_TYPES = frozenset({"float32", "float16", "bfloat16"})
_FP8_DATA_TYPES = frozenset({"fp8_e4m3", "fp8_e5m2"})
_UNBOUNDED_DIAGONAL = 1 << 30


@dataclass(frozen=True)
class ParsedBinaryRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    pointwise_mode: int
    left: TensorSpec
    right: TensorSpec
    output: TensorSpec
    alpha: float
    alpha_bits: str
    n_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedUnaryRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    pointwise_mode: int
    input: TensorSpec
    output: TensorSpec
    negative_slope: float
    lower_clip: float
    upper_clip: float
    has_upper_clip: int
    swish_beta: float
    elu_alpha: float
    softplus_beta: float
    n_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedTernaryRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    pointwise_mode: int
    a: TensorSpec
    b: TensorSpec
    predicate: TensorSpec
    output: TensorSpec
    n_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedLayoutRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    input: TensorSpec
    output: TensorSpec
    logical_input_dimensions: tuple[int, ...]
    logical_input_strides: tuple[int, ...]
    input_base: int
    n_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedReductionRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    reduction_mode: int
    input: TensorSpec
    output: TensorSpec
    axis: int
    keep_dimensions: bool
    outer: int
    extent: int
    inner: int
    output_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedMatmulRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    a: TensorSpec
    b: TensorSpec
    output: TensorSpec
    batch: int
    m: int
    n: int
    k: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedConvolutionRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    image: TensorSpec
    filter: TensorSpec
    result: TensorSpec
    spatial_rank: int
    groups: int
    convolution_mode: int
    pre_padding: tuple[int, ...]
    post_padding: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    batch: int
    in_channels: int
    out_channels: int
    in_per_group: int
    out_per_group: int
    n_outputs: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedAddSquareRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    left: TensorSpec
    right: TensorSpec
    output: TensorSpec
    n_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedConvBiasReluRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    image: TensorSpec
    filter: TensorSpec
    bias: TensorSpec
    output: TensorSpec
    spatial_rank: int
    groups: int
    pre_padding: tuple[int, ...]
    post_padding: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    batch: int
    in_channels: int
    out_channels: int
    in_per_group: int
    out_per_group: int
    n_outputs: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedNormalizationRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    x: TensorSpec
    scale: TensorSpec
    bias: TensorSpec
    y: TensorSpec
    mean: TensorSpec | None
    inv_variance: TensorSpec
    rows: int
    normalized_elements: int
    epsilon: float
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedBatchnormRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    x: TensorSpec
    scale: TensorSpec
    bias: TensorSpec
    previous_running_mean: TensorSpec
    previous_running_variance: TensorSpec
    y: TensorSpec
    mean: TensorSpec
    inv_variance: TensorSpec
    next_running_mean: TensorSpec
    next_running_variance: TensorSpec
    batch: int
    channels: int
    spatial: int
    n_elements: int
    epsilon: float
    momentum: float
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedBatchnormInferenceRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    x: TensorSpec
    mean: TensorSpec
    inv_variance: TensorSpec
    scale: TensorSpec
    bias: TensorSpec
    y: TensorSpec
    batch: int
    channels: int
    spatial: int
    n_elements: int
    external_binding_uids: tuple[int, ...]
    request_sha256: str


@dataclass(frozen=True)
class ParsedAttentionRequest:
    flagdnn_version: str
    target: str
    compiler_identity: str
    autotune: bool
    node_id: int
    operation: str
    port_tensors: tuple[tuple[str, TensorSpec], ...]
    batch: int
    heads: int
    key_heads: int
    value_heads: int
    sequence_q: int
    sequence_kv: int
    head_dimension: int
    value_dimension: int
    q_per_k: int
    q_per_v: int
    min_diag: int
    max_diag: int
    has_bias: bool
    has_dbias: bool
    banded: bool
    causal_top_left: bool
    reverse_causal: bool
    generate_stats: bool
    attn_scale: float
    external_binding_uids: tuple[int, ...]
    request_sha256: str

    def tensor(self, name: str) -> TensorSpec:
        for semantic_name, tensor in self.port_tensors:
            if semantic_name == name:
                return tensor
        raise ValueError(f"Attention request has no {name!r} tensor")

    def optional_tensor(self, name: str) -> TensorSpec | None:
        for semantic_name, tensor in self.port_tensors:
            if semantic_name == name:
                return tensor
        return None


ParsedPointwiseRequest = (
    ParsedBinaryRequest | ParsedUnaryRequest | ParsedTernaryRequest
)
ParsedCompilerRequest = (
    ParsedPointwiseRequest
    | ParsedLayoutRequest
    | ParsedReductionRequest
    | ParsedMatmulRequest
    | ParsedConvolutionRequest
    | ParsedAddSquareRequest
    | ParsedConvBiasReluRequest
    | ParsedNormalizationRequest
    | ParsedBatchnormRequest
    | ParsedBatchnormInferenceRequest
    | ParsedAttentionRequest
)


# Compatibility alias for existing Phase 1 contract imports.
ParsedAddRequest = ParsedBinaryRequest


@dataclass(frozen=True)
class _Node:
    id: int
    operation: str
    compute_data_type: str
    inputs: dict[str, int]
    outputs: dict[str, int]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class _Envelope:
    version: str
    target: str
    identity: str
    autotune: bool
    graph: dict[str, Any]
    ordered_tensors: tuple[TensorSpec, ...]
    registry: dict[int, TensorSpec]
    nodes: tuple[_Node, ...]

    @property
    def node(self) -> _Node:
        if len(self.nodes) != 1:
            raise ValueError("mthreads operation requires exactly one node")
        return self.nodes[0]


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object key is duplicated: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"JSON nonfinite number is forbidden: {value}")


def load_request(request_bytes: bytes) -> dict[str, Any]:
    if not request_bytes or len(request_bytes) > (16 << 20):
        raise ValueError("compiler request size is invalid")
    try:
        decoded = request_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("compiler request is not UTF-8") from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"compiler request JSON is invalid: {error}") from error
    return require_object(value, "request")


def validate_target(target: object) -> str:
    if not isinstance(target, str):
        raise ValueError("mthreads target fingerprint must be a string")
    match = TARGET_PATTERN.fullmatch(target)
    if match is None:
        raise ValueError("mthreads target fingerprint is invalid")
    architecture = int(match.group(1))
    warp_size = int(match.group(2))
    if architecture <= 0 or architecture > 999 or warp_size not in {32, 64}:
        raise ValueError("mthreads target attributes are unsupported")
    return target


def _parse_port_array(
    raw: object,
    context: str,
    registry: dict[int, TensorSpec],
) -> dict[str, int]:
    values = require_list(raw, context)
    result: dict[str, int] = {}
    for index, raw_value in enumerate(values):
        item_context = f"{context}[{index}]"
        value = require_object(raw_value, item_context)
        keys = set(value)
        if keys not in ({"name", "uid"}, {"name", "uid", "optional"}):
            raise ValueError(f"{item_context} keys are invalid")
        name = value["name"]
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or "\x00" in name
        ):
            raise ValueError(f"{item_context}.name is invalid")
        if name in result:
            raise ValueError(f"{context} role is duplicated")
        uid = require_integer(
            value["uid"],
            f"{item_context}.uid",
            minimum=0,
            maximum=MAX_I64,
        )
        if uid not in registry:
            raise ValueError(f"{item_context} references an unknown tensor")
        if "optional" in value:
            optional = value["optional"]
            if optional is not True:
                raise ValueError(
                    f"{item_context}.optional, when present, must be true"
                )
        result[name] = uid
    return result


def _parse_nodes(
    graph: dict[str, Any],
    registry: dict[int, TensorSpec],
) -> tuple[_Node, ...]:
    values = require_list(graph.get("nodes"), "graph.nodes")
    node_count = require_integer(
        graph.get("node_count"),
        "graph.node_count",
        minimum=1,
        maximum=1024,
    )
    if node_count != len(values):
        raise ValueError("graph.node_count does not match graph.nodes")

    nodes: list[_Node] = []
    node_positions: dict[int, int] = {}
    producer_nodes: dict[int, int] = {}
    for position, raw_value in enumerate(values):
        context = f"graph.nodes[{position}]"
        value = require_object(raw_value, context)
        require_exact_keys(value, _NODE_KEYS, context)
        node_id = require_integer(
            value["id"], f"{context}.id", minimum=0, maximum=MAX_I64
        )
        if node_id in node_positions:
            raise ValueError("graph node ID is duplicated")
        node_positions[node_id] = position
        operation = value["type"]
        if (
            not isinstance(operation, str)
            or not operation
            or len(operation) > 128
        ):
            raise ValueError(f"{context}.type is invalid")
        name = value["name"]
        if (
            not isinstance(name, str)
            or len(name) > 4096
            or "\x00" in name
        ):
            raise ValueError(f"{context}.name is invalid")
        compute_data_type = value["compute_data_type"]
        if not isinstance(compute_data_type, str):
            raise ValueError(f"{context}.compute_data_type is invalid")
        inputs = _parse_port_array(
            value["inputs"], f"{context}.inputs", registry
        )
        outputs = _parse_port_array(
            value["outputs"], f"{context}.outputs", registry
        )
        attributes = require_object(
            value["attributes"], f"{context}.attributes"
        )
        for uid in outputs.values():
            if uid in producer_nodes:
                raise ValueError("graph tensor has more than one producer")
            producer_nodes[uid] = node_id
        nodes.append(
            _Node(
                id=node_id,
                operation=operation,
                compute_data_type=compute_data_type,
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        )

    has_external_output = False
    for node in nodes:
        position = node_positions[node.id]
        for uid in node.inputs.values():
            producer = producer_nodes.get(uid)
            if registry[uid].virtual and producer is None:
                raise ValueError("virtual tensor input has no producer")
            if (
                producer is not None
                and node_positions[producer] >= position
            ):
                raise ValueError(
                    "graph nodes are not in topological execution order"
                )
        for uid in node.outputs.values():
            if not registry[uid].virtual:
                has_external_output = True
    if not has_external_output:
        raise ValueError("graph has no non-virtual output tensor")
    return tuple(nodes)


def _float32(value: float, context: str) -> tuple[float, str]:
    try:
        encoded = struct.pack("<f", value)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{context} is not representable as float32") from error
    result = struct.unpack("<f", encoded)[0]
    if not isinstance(result, float) or not (abs(result) <= 3.4028235e38):
        raise ValueError(f"{context} is not representable as float32")
    return result, encoded.hex()


def _parse_envelope(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> _Envelope:
    root = load_request(request_bytes)
    require_exact_keys(root, _ROOT_KEYS, "request")
    if root["schema_version"] != GRAPH_SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    version = root["flagdnn_version"]
    if (
        not isinstance(version, str)
        or _VERSION_PATTERN.fullmatch(version) is None
    ):
        raise ValueError("request FlagDNN version is invalid")
    if root["backend"] != "mthreads":
        raise ValueError("mthreads provider received another backend")
    target = validate_target(root["target"])
    if target != expected_target:
        raise ValueError("request target does not match compiler target")
    identity = root["compiler_identity"]
    if (
        not isinstance(identity, str)
        or _SHA256_PATTERN.fullmatch(identity) is None
        or identity != expected_identity
    ):
        raise ValueError("request compiler identity does not match provider")

    build_options = require_object(
        root["build_options"], "request.build_options"
    )
    require_exact_keys(
        build_options, _BUILD_OPTION_KEYS, "request.build_options"
    )
    heuristic_modes = require_list(
        build_options["heuristic_modes"],
        "request.build_options.heuristic_modes",
    )
    if (
        not heuristic_modes
        or any(mode not in {"A", "FALLBACK"} for mode in heuristic_modes)
        or len(set(heuristic_modes)) != len(heuristic_modes)
    ):
        raise ValueError("request heuristic modes are invalid")
    autotune = build_options["autotune"]
    if not isinstance(autotune, bool):
        raise ValueError("request build_options.autotune must be a boolean")

    graph = require_object(root["graph"], "request.graph")
    require_exact_keys(graph, _GRAPH_KEYS, "request.graph")
    graph_name = graph["name"]
    if (
        not isinstance(graph_name, str)
        or len(graph_name) > (1 << 20)
        or "\x00" in graph_name
    ):
        raise ValueError("graph.name is invalid")
    ordered_tensors, registry = parse_tensor_table(graph)
    nodes = _parse_nodes(graph, registry)
    return _Envelope(
        version=version,
        target=target,
        identity=identity,
        autotune=autotune,
        graph=graph,
        ordered_tensors=ordered_tensors,
        registry=registry,
        nodes=nodes,
    )


def _require_nonoptional_ports(
    graph: dict[str, Any], context: str
) -> None:
    if any(
        "optional" in port
        for port in (
            *require_list(graph["nodes"][0]["inputs"], f"{context} inputs"),
            *require_list(
                graph["nodes"][0]["outputs"], f"{context} outputs"
            ),
        )
    ):
        raise ValueError(f"{context} ports must not be optional")


def _float32_attribute(
    attributes: dict[str, Any], name: str
) -> float:
    value = require_number(attributes[name], f"pointwise attributes.{name}")
    return _float32(value, f"pointwise attributes.{name}")[0]


def parse_binary_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedBinaryRequest:
    root = load_request(request_bytes)
    require_exact_keys(root, _ROOT_KEYS, "request")
    if root["schema_version"] != GRAPH_SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    version = root["flagdnn_version"]
    if (
        not isinstance(version, str)
        or _VERSION_PATTERN.fullmatch(version) is None
    ):
        raise ValueError("request FlagDNN version is invalid")
    if root["backend"] != "mthreads":
        raise ValueError("mthreads provider received another backend")
    target = validate_target(root["target"])
    if target != expected_target:
        raise ValueError("request target does not match compiler target")
    identity = root["compiler_identity"]
    if (
        not isinstance(identity, str)
        or _SHA256_PATTERN.fullmatch(identity) is None
        or identity != expected_identity
    ):
        raise ValueError("request compiler identity does not match provider")

    build_options = require_object(
        root["build_options"], "request.build_options"
    )
    require_exact_keys(
        build_options, _BUILD_OPTION_KEYS, "request.build_options"
    )
    heuristic_modes = require_list(
        build_options["heuristic_modes"],
        "request.build_options.heuristic_modes",
    )
    if (
        not heuristic_modes
        or any(mode not in {"A", "FALLBACK"} for mode in heuristic_modes)
        or len(set(heuristic_modes)) != len(heuristic_modes)
    ):
        raise ValueError("request heuristic modes are invalid")
    autotune = build_options["autotune"]
    if not isinstance(autotune, bool):
        raise ValueError("request build_options.autotune must be a boolean")

    graph = require_object(root["graph"], "request.graph")
    require_exact_keys(graph, _GRAPH_KEYS, "request.graph")
    graph_name = graph["name"]
    if (
        not isinstance(graph_name, str)
        or len(graph_name) > (1 << 20)
        or "\x00" in graph_name
    ):
        raise ValueError("graph.name is invalid")
    ordered_tensors, registry = parse_tensor_table(graph)
    nodes = _parse_nodes(graph, registry)
    if len(nodes) != 1:
        raise ValueError(
            "mthreads pointwise compiler supports exactly one node"
        )
    node = nodes[0]
    if set(node.inputs) != {"left", "right"}:
        raise ValueError(
            "binary pointwise inputs must have roles left and right"
        )
    if set(node.outputs) != {"output"}:
        raise ValueError("binary pointwise output must have role output")
    if any(
        "optional" in port
        for port in (
            *require_list(
                graph["nodes"][0]["inputs"], "pointwise inputs"
            ),
            *require_list(
                graph["nodes"][0]["outputs"], "pointwise outputs"
            ),
        )
    ):
        raise ValueError("binary pointwise ports must not be optional")

    left = registry[node.inputs["left"]]
    right = registry[node.inputs["right"]]
    output = registry[node.outputs["output"]]
    referenced_uids = {left.uid, right.uid, output.uid}
    if len(referenced_uids) != 3 or len(ordered_tensors) != 3:
        raise ValueError(
            "binary pointwise requires exactly three distinct tensors"
        )
    if left.virtual or right.virtual or output.virtual:
        raise ValueError("binary pointwise tensors must be externally bound")
    if left.data_type != right.data_type:
        raise ValueError("binary pointwise input data types must match")
    expected_output = broadcast_dimensions(left, right)
    if output.dimensions != expected_output:
        raise ValueError(
            "binary pointwise output shape does not match broadcast result"
        )
    count = element_count(output)

    require_exact_keys(
        node.attributes, _BINARY_ATTRIBUTE_KEYS, "pointwise attributes"
    )
    mode = require_integer(
        node.attributes["mode"], "pointwise attributes.mode"
    )
    pointwise_mode = require_integer(
        node.attributes["pointwise_mode"],
        "pointwise attributes.pointwise_mode",
    )
    operation = _BINARY_OPERATIONS.get(mode)
    if (
        operation is None
        or pointwise_mode != mode
        or node.operation != operation
    ):
        raise ValueError(
            "binary pointwise node type and mode do not match"
        )
    if mode in _COMPARISON_MODES:
        if (
            left.data_type not in _FLOATING_DATA_TYPES
            or output.data_type != "boolean"
            or node.compute_data_type != "boolean"
        ):
            raise ValueError(
                "comparison pointwise requires floating inputs and "
                "BOOLEAN output/compute"
            )
    elif mode in _LOGICAL_BINARY_MODES:
        if (
            left.data_type != "boolean"
            or output.data_type != "boolean"
            or node.compute_data_type != "boolean"
        ):
            raise ValueError(
                "logical pointwise requires BOOLEAN storage/compute"
            )
    elif (
        left.data_type not in _FLOATING_DATA_TYPES
        or output.data_type != left.data_type
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "numeric binary pointwise requires matching floating storage "
            "and float32 compute"
        )
    if mode == 40 and (
        left.dimensions != right.dimensions
        or output.dimensions != left.dimensions
    ):
        raise ValueError("sigmoid backward tensors must have equal shapes")
    requested_elements = require_integer(
        node.attributes["n_elements"],
        "pointwise attributes.n_elements",
        minimum=1,
    )
    if requested_elements != count:
        raise ValueError(
            "pointwise attributes.n_elements does not match output shape"
        )
    alpha_value = require_number(
        node.attributes["alpha"], "pointwise attributes.alpha"
    )
    alpha, alpha_bits = _float32(
        alpha_value, "pointwise attributes.alpha"
    )
    if mode not in {1, 17} and alpha != 1.0:
        raise ValueError("pointwise alpha only applies to ADD or SUB")
    external_uids = tuple(tensor.uid for tensor in ordered_tensors)
    return ParsedBinaryRequest(
        flagdnn_version=version,
        target=target,
        compiler_identity=identity,
        autotune=autotune,
        node_id=node.id,
        operation=operation,
        pointwise_mode=mode,
        left=left,
        right=right,
        output=output,
        alpha=alpha,
        alpha_bits=alpha_bits,
        n_elements=count,
        external_binding_uids=external_uids,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_unary_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedUnaryRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if set(node.inputs) != {"input"} or set(node.outputs) != {"output"}:
        raise ValueError("unary pointwise port roles are invalid")
    _require_nonoptional_ports(envelope.graph, "unary pointwise")

    input_tensor = envelope.registry[node.inputs["input"]]
    output = envelope.registry[node.outputs["output"]]
    if (
        input_tensor.uid == output.uid
        or len(envelope.ordered_tensors) != 2
        or input_tensor.virtual
        or output.virtual
    ):
        raise ValueError(
            "unary pointwise requires two distinct externally bound tensors"
        )
    if input_tensor.dimensions != output.dimensions:
        raise ValueError("unary pointwise input/output shapes must match")

    attributes = node.attributes
    require_exact_keys(
        attributes, _UNARY_ATTRIBUTE_KEYS, "unary pointwise attributes"
    )
    mode = require_integer(attributes["mode"], "pointwise attributes.mode")
    operation = _UNARY_OPERATIONS.get(mode)
    if operation is None or node.operation != operation:
        raise ValueError("unary pointwise node type and mode do not match")
    if mode == 24:
        if (
            input_tensor.data_type != "boolean"
            or output.data_type != "boolean"
            or node.compute_data_type != "boolean"
        ):
            raise ValueError(
                "logical NOT requires BOOLEAN storage and compute"
            )
    elif (
        input_tensor.data_type not in _FLOATING_DATA_TYPES
        or output.data_type != input_tensor.data_type
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "numeric unary pointwise requires matching floating storage "
            "and float32 compute"
        )

    count = element_count(output)
    requested_elements = require_integer(
        attributes["n_elements"],
        "pointwise attributes.n_elements",
        minimum=1,
    )
    if requested_elements != count:
        raise ValueError(
            "pointwise attributes.n_elements does not match output shape"
        )
    relu_upper_clip_set = attributes["relu_upper_clip_set"]
    if not isinstance(relu_upper_clip_set, bool):
        raise ValueError(
            "pointwise attributes.relu_upper_clip_set must be a boolean"
        )
    has_upper_clip = require_integer(
        attributes["has_upper_clip"],
        "pointwise attributes.has_upper_clip",
        minimum=0,
        maximum=1,
    )
    negative_slope = _float32_attribute(attributes, "negative_slope")
    lower_clip = _float32_attribute(attributes, "lower_clip")
    upper_clip = _float32_attribute(attributes, "upper_clip")
    relu_negative_slope = _float32_attribute(
        attributes, "relu_lower_clip_slope"
    )
    relu_lower_clip = _float32_attribute(attributes, "relu_lower_clip")
    relu_upper_clip = _float32_attribute(attributes, "relu_upper_clip")
    swish_beta = _float32_attribute(attributes, "swish_beta")
    elu_alpha = _float32_attribute(attributes, "elu_alpha")
    softplus_beta = _float32_attribute(attributes, "softplus_beta")
    if (
        has_upper_clip != int(relu_upper_clip_set)
        or negative_slope != relu_negative_slope
        or lower_clip != relu_lower_clip
        or upper_clip != relu_upper_clip
    ):
        raise ValueError(
            "unary normalized ReLU attributes differ from descriptor values"
        )
    if has_upper_clip and upper_clip < lower_clip:
        raise ValueError("ReLU upper clip is below its lower clip")
    if softplus_beta <= 0.0:
        raise ValueError("softplus beta must be positive")

    relu_defaults = (
        negative_slope == 0.0
        and lower_clip == 0.0
        and upper_clip == 0.0
        and has_upper_clip == 0
    )
    if mode != 2 and not relu_defaults:
        raise ValueError("ReLU attributes are set for another unary mode")
    if mode != 38 and swish_beta != 1.0:
        raise ValueError("swish beta is set for another unary mode")
    if mode != 35 and elu_alpha != 1.0:
        raise ValueError("ELU alpha is set for another unary mode")
    if mode != 37 and softplus_beta != 1.0:
        raise ValueError("softplus beta is set for another unary mode")

    return ParsedUnaryRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=operation,
        pointwise_mode=mode,
        input=input_tensor,
        output=output,
        negative_slope=negative_slope,
        lower_clip=lower_clip,
        upper_clip=upper_clip,
        has_upper_clip=has_upper_clip,
        swish_beta=swish_beta,
        elu_alpha=elu_alpha,
        softplus_beta=softplus_beta,
        n_elements=count,
        external_binding_uids=tuple(
            tensor.uid for tensor in envelope.ordered_tensors
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def _broadcast_three(
    a: TensorSpec, b: TensorSpec, predicate: TensorSpec
) -> tuple[int, ...]:
    shapes = (a.dimensions, b.dimensions, predicate.dimensions)
    rank = max(len(shape) for shape in shapes)
    result = [1] * rank
    for shape in shapes:
        leading = rank - len(shape)
        for index, dimension in enumerate(shape):
            output_index = leading + index
            current = result[output_index]
            if dimension != current and dimension != 1 and current != 1:
                raise ValueError(
                    "ternary pointwise inputs are not broadcast-compatible"
                )
            result[output_index] = max(current, dimension)
    return tuple(result)


def parse_ternary_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedTernaryRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if set(node.inputs) != {"a", "b", "t"}:
        raise ValueError("ternary pointwise input roles must be a, b, and t")
    if set(node.outputs) != {"output"}:
        raise ValueError("ternary pointwise output role must be output")
    _require_nonoptional_ports(envelope.graph, "ternary pointwise")

    a = envelope.registry[node.inputs["a"]]
    b = envelope.registry[node.inputs["b"]]
    predicate = envelope.registry[node.inputs["t"]]
    output = envelope.registry[node.outputs["output"]]
    if (
        len({a.uid, b.uid, predicate.uid, output.uid}) != 4
        or len(envelope.ordered_tensors) != 4
        or any(tensor.virtual for tensor in (a, b, predicate, output))
    ):
        raise ValueError(
            "ternary pointwise requires four distinct externally bound "
            "tensors"
        )
    if (
        a.data_type not in _FLOATING_DATA_TYPES
        or b.data_type != a.data_type
        or output.data_type != a.data_type
        or predicate.data_type != "boolean"
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "binary_select requires matching floating A/B/output, "
            "BOOLEAN T, and float32 compute"
        )
    if output.dimensions != _broadcast_three(a, b, predicate):
        raise ValueError(
            "ternary pointwise output shape does not match broadcast result"
        )

    attributes = node.attributes
    require_exact_keys(
        attributes, _TERNARY_ATTRIBUTE_KEYS, "ternary pointwise attributes"
    )
    mode = require_integer(attributes["mode"], "pointwise attributes.mode")
    if mode != _TERNARY_MODE or node.operation != "binary_select":
        raise ValueError("ternary pointwise node type and mode do not match")
    count = element_count(output)
    requested_elements = require_integer(
        attributes["n_elements"],
        "pointwise attributes.n_elements",
        minimum=1,
    )
    if requested_elements != count:
        raise ValueError(
            "pointwise attributes.n_elements does not match output shape"
        )
    return ParsedTernaryRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation="binary_select",
        pointwise_mode=mode,
        a=a,
        b=b,
        predicate=predicate,
        output=output,
        n_elements=count,
        external_binding_uids=tuple(
            tensor.uid for tensor in envelope.ordered_tensors
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def _layout_integer_array(
    attributes: dict[str, Any],
    name: str,
    *,
    length: int,
    minimum: int,
) -> tuple[int, ...]:
    values = require_list(attributes[name], f"layout attributes.{name}")
    if len(values) != length:
        raise ValueError(f"layout attributes.{name} has the wrong length")
    return tuple(
        require_integer(
            value,
            f"layout attributes.{name}[{index}]",
            minimum=minimum,
            maximum=MAX_I64,
        )
        for index, value in enumerate(values)
    )


def parse_layout_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedLayoutRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if node.operation not in _LAYOUT_OPERATIONS:
        raise ValueError("mthreads layout operation is unsupported")
    if set(node.inputs) != {"input"} or set(node.outputs) != {"output"}:
        raise ValueError("layout ports must be input and output")
    _require_nonoptional_ports(envelope.graph, "layout")

    input_tensor = envelope.registry[node.inputs["input"]]
    output = envelope.registry[node.outputs["output"]]
    if (
        input_tensor.uid == output.uid
        or len(envelope.ordered_tensors) != 2
        or input_tensor.virtual
        or output.virtual
    ):
        raise ValueError(
            "layout requires two distinct externally bound tensors"
        )
    if (
        input_tensor.data_type != output.data_type
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "layout input/output types must match with float32 compute"
        )

    attributes = node.attributes
    input_rank = len(input_tensor.dimensions)
    output_rank = len(output.dimensions)
    logical_dimensions = input_tensor.dimensions
    logical_strides = input_tensor.strides
    input_base = 0
    if node.operation == "reshape":
        require_exact_keys(
            attributes, _RESHAPE_ATTRIBUTE_KEYS, "reshape attributes"
        )
        if (
            require_integer(
                attributes["input_rank"], "reshape attributes.input_rank"
            )
            != input_rank
            or require_integer(
                attributes["output_rank"],
                "reshape attributes.output_rank",
            )
            != output_rank
            or require_integer(
                attributes["reshape_mode"],
                "reshape attributes.reshape_mode",
            )
            != 2
        ):
            raise ValueError("reshape rank or mode metadata is invalid")
        if element_count(input_tensor) != element_count(output):
            raise ValueError(
                "reshape input/output element counts must match"
            )
    elif node.operation == "transpose":
        require_exact_keys(
            attributes,
            _TRANSPOSE_ATTRIBUTE_KEYS,
            "transpose attributes",
        )
        rank = require_integer(
            attributes["rank"], "transpose attributes.rank"
        )
        if rank != input_rank or output_rank != input_rank:
            raise ValueError("transpose ranks must match")
        permutation = _layout_integer_array(
            attributes,
            "permutation",
            length=rank,
            minimum=0,
        )
        if set(permutation) != set(range(rank)):
            raise ValueError("transpose permutation is invalid")
        expected_output = tuple(
            input_tensor.dimensions[axis] for axis in permutation
        )
        if output.dimensions != expected_output:
            raise ValueError(
                "transpose output shape does not match permutation"
            )
        logical_dimensions = output.dimensions
        logical_strides = tuple(
            input_tensor.strides[axis] for axis in permutation
        )
    else:
        require_exact_keys(
            attributes, _SLICE_ATTRIBUTE_KEYS, "slice attributes"
        )
        rank = require_integer(attributes["rank"], "slice attributes.rank")
        if rank != input_rank or output_rank != input_rank:
            raise ValueError("slice ranks must match")
        starts = _layout_integer_array(
            attributes, "starts", length=rank, minimum=0
        )
        limits = _layout_integer_array(
            attributes, "limits", length=rank, minimum=1
        )
        slice_strides = _layout_integer_array(
            attributes, "slice_strides", length=rank, minimum=1
        )
        logical_dimensions = output.dimensions
        logical_stride_values: list[int] = []
        for axis, (start, limit, step) in enumerate(
            zip(starts, limits, slice_strides, strict=True)
        ):
            if (
                start >= limit
                or limit > input_tensor.dimensions[axis]
                or output.dimensions[axis]
                != 1 + (limit - start - 1) // step
            ):
                raise ValueError("slice range or output shape is invalid")
            contribution = start * input_tensor.strides[axis]
            if contribution > MAX_I64 - input_base:
                raise ValueError("slice input base overflows int64")
            input_base += contribution
            effective_stride = step * input_tensor.strides[axis]
            if effective_stride > MAX_I64:
                raise ValueError("slice effective stride overflows int64")
            logical_stride_values.append(effective_stride)
        logical_strides = tuple(logical_stride_values)

    expected_arrays = {
        "input_dimensions": input_tensor.dimensions,
        "input_strides": input_tensor.strides,
        "output_dimensions": output.dimensions,
        "output_strides": output.strides,
    }
    for name, expected in expected_arrays.items():
        actual = _layout_integer_array(
            attributes, name, length=len(expected), minimum=1
        )
        if actual != expected:
            raise ValueError(
                f"layout attributes.{name} differs from tensor metadata"
            )

    count = element_count(output)
    requested_elements = require_integer(
        attributes["n_elements"],
        "layout attributes.n_elements",
        minimum=1,
    )
    if requested_elements != count:
        raise ValueError(
            "layout attributes.n_elements does not match output shape"
        )
    return ParsedLayoutRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=node.operation,
        input=input_tensor,
        output=output,
        logical_input_dimensions=logical_dimensions,
        logical_input_strides=logical_strides,
        input_base=input_base,
        n_elements=count,
        external_binding_uids=tuple(
            tensor.uid for tensor in envelope.ordered_tensors
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_reduction_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedReductionRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if set(node.inputs) != {"input"} or set(node.outputs) != {"output"}:
        raise ValueError("reduction ports must be input and output")
    _require_nonoptional_ports(envelope.graph, "reduction")

    input_tensor = envelope.registry[node.inputs["input"]]
    output = envelope.registry[node.outputs["output"]]
    if (
        input_tensor.uid == output.uid
        or len(envelope.ordered_tensors) != 2
        or input_tensor.virtual
        or output.virtual
    ):
        raise ValueError(
            "reduction requires two distinct externally bound tensors"
        )
    if (
        input_tensor.data_type != output.data_type
        or input_tensor.data_type not in _FLOATING_DATA_TYPES
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "reduction requires matching floating storage and float32 compute"
        )
    rank = len(input_tensor.dimensions)
    if not 1 <= rank <= 8:
        raise ValueError("reduction input rank must be in [1, 8]")

    attributes = node.attributes
    require_exact_keys(
        attributes, _REDUCTION_ATTRIBUTE_KEYS, "reduction attributes"
    )
    mode = require_integer(
        attributes["mode"],
        "reduction attributes.mode",
        minimum=0,
        maximum=2,
    )
    operation = _REDUCTION_OPERATIONS.get(mode)
    if operation is None or node.operation != operation:
        raise ValueError("reduction node type and mode do not match")
    axis = require_integer(
        attributes["axis"],
        "reduction attributes.axis",
        minimum=0,
        maximum=rank - 1,
    )
    keep_value = require_integer(
        attributes["keep_dimensions"],
        "reduction attributes.keep_dimensions",
        minimum=0,
        maximum=1,
    )
    keep_dimensions = keep_value == 1
    expected_output = list(input_tensor.dimensions)
    if keep_dimensions:
        expected_output[axis] = 1
    else:
        del expected_output[axis]
    if output.dimensions != tuple(expected_output):
        raise ValueError("reduction output shape is invalid")

    expected_outer = math.prod(input_tensor.dimensions[:axis])
    expected_extent = input_tensor.dimensions[axis]
    expected_inner = math.prod(input_tensor.dimensions[axis + 1 :])
    expected_output_elements = expected_outer * expected_inner
    outer = require_integer(
        attributes["outer"],
        "reduction attributes.outer",
        minimum=1,
        maximum=MAX_I32,
    )
    extent = require_integer(
        attributes["reduction"],
        "reduction attributes.reduction",
        minimum=1,
        maximum=65536,
    )
    inner = require_integer(
        attributes["inner"],
        "reduction attributes.inner",
        minimum=1,
        maximum=MAX_I32,
    )
    output_elements = require_integer(
        attributes["output_elements"],
        "reduction attributes.output_elements",
        minimum=1,
        maximum=MAX_I32,
    )
    if (
        (outer, extent, inner, output_elements)
        != (
            expected_outer,
            expected_extent,
            expected_inner,
            expected_output_elements,
        )
        or element_count(output) != output_elements
    ):
        raise ValueError(
            "reduction lowered parameters are inconsistent with tensors"
        )
    return ParsedReductionRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=operation,
        reduction_mode=mode,
        input=input_tensor,
        output=output,
        axis=axis,
        keep_dimensions=keep_dimensions,
        outer=outer,
        extent=extent,
        inner=inner,
        output_elements=output_elements,
        external_binding_uids=tuple(
            tensor.uid for tensor in envelope.ordered_tensors
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_matmul_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedMatmulRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if node.operation != "matmul":
        raise ValueError("Matmul node type is invalid")
    if set(node.inputs) != {"a", "b"} or set(node.outputs) != {"output"}:
        raise ValueError("Matmul ports must be a, b, and output")
    _require_nonoptional_ports(envelope.graph, "Matmul")

    a = envelope.registry[node.inputs["a"]]
    b = envelope.registry[node.inputs["b"]]
    output = envelope.registry[node.outputs["output"]]
    if (
        len({a.uid, b.uid, output.uid}) != 3
        or len(envelope.ordered_tensors) != 3
        or a.virtual
        or b.virtual
        or output.virtual
    ):
        raise ValueError(
            "Matmul requires three distinct externally bound tensors"
        )
    if (
        a.data_type != b.data_type
        or a.data_type != output.data_type
        or a.data_type not in _FLOATING_DATA_TYPES
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "Matmul requires matching floating storage and float32 compute"
        )
    if any(not 2 <= len(tensor.dimensions) <= 8 for tensor in (a, b, output)):
        raise ValueError("Matmul tensor ranks must be in [2, 8]")

    m = a.dimensions[-2]
    k = a.dimensions[-1]
    if b.dimensions[-2] != k:
        raise ValueError("Matmul contraction dimensions do not match")
    n = b.dimensions[-1]
    a_batch = a.dimensions[:-2]
    b_batch = b.dimensions[:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("Matmul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = (
            a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        )
        b_dimension = (
            b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        )
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            raise ValueError("Matmul batch dimensions are not broadcastable")
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    if output.dimensions != tuple((*batch_dimensions, m, n)):
        raise ValueError("Matmul output shape is invalid")
    batch = math.prod(batch_dimensions)
    if batch > MAX_I32 or element_count(output) > MAX_I32:
        raise ValueError("Matmul launch extent exceeds int32")

    attributes = node.attributes
    require_exact_keys(attributes, _MATMUL_ATTRIBUTE_KEYS, "Matmul attributes")
    lowered = {
        name: require_integer(
            attributes[name],
            f"Matmul attributes.{name}",
            minimum=1,
            maximum=MAX_I32,
        )
        for name in ("batch", "m", "n", "k")
    }
    if lowered != {"batch": batch, "m": m, "n": n, "k": k}:
        raise ValueError(
            "Matmul lowered parameters are inconsistent with tensors"
        )
    return ParsedMatmulRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation="matmul",
        a=a,
        b=b,
        output=output,
        batch=batch,
        m=m,
        n=n,
        k=k,
        external_binding_uids=tuple(
            tensor.uid for tensor in envelope.ordered_tensors
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def _convolution_integer_array(
    attributes: dict[str, Any],
    name: str,
    *,
    length: int,
    minimum: int,
) -> tuple[int, ...]:
    values = require_list(
        attributes[name], f"convolution attributes.{name}"
    )
    if len(values) != length:
        raise ValueError(
            f"convolution attributes.{name} length must equal spatial_rank"
        )
    return tuple(
        require_integer(
            value,
            f"convolution attributes.{name}[{index}]",
            minimum=minimum,
            maximum=MAX_I32,
        )
        for index, value in enumerate(values)
    )


def _convolution_output_dimension(
    input_size: int,
    filter_size: int,
    pre_padding: int,
    post_padding: int,
    stride: int,
    dilation: int,
) -> int:
    effective = (filter_size - 1) * dilation + 1
    padded = input_size + pre_padding + post_padding
    if effective > MAX_I64 or padded > MAX_I64:
        raise ValueError("convolution spatial geometry overflows int64")
    if padded < effective:
        raise ValueError("convolution filter is larger than padded input")
    result = (padded - effective) // stride + 1
    if result > MAX_I32:
        raise ValueError("convolution spatial result exceeds int32")
    return result


def parse_convolution_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedConvolutionRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    operation = node.operation
    if operation not in _CONVOLUTION_OPERATIONS:
        raise ValueError("convolution node type is invalid")

    fprop = operation in {"conv2d_fprop", "convolution_fprop"}
    if fprop:
        if set(node.inputs) != {"input", "filter"} or set(
            node.outputs
        ) != {"output"}:
            raise ValueError(
                "convolution FProp ports must be input, filter, and output"
            )
        image = envelope.registry[node.inputs["input"]]
        filter_tensor = envelope.registry[node.inputs["filter"]]
        result = envelope.registry[node.outputs["output"]]
        output_tensor = result
    elif operation == "convolution_dgrad":
        if set(node.inputs) != {"dy", "w"} or set(node.outputs) != {"dx"}:
            raise ValueError(
                "convolution DGrad ports must be dy, w, and dx"
            )
        result = envelope.registry[node.inputs["dy"]]
        filter_tensor = envelope.registry[node.inputs["w"]]
        image = envelope.registry[node.outputs["dx"]]
        output_tensor = image
    else:
        if set(node.inputs) != {"dy", "x"} or set(node.outputs) != {"dw"}:
            raise ValueError(
                "convolution WGrad ports must be dy, x, and dw"
            )
        result = envelope.registry[node.inputs["dy"]]
        image = envelope.registry[node.inputs["x"]]
        filter_tensor = envelope.registry[node.outputs["dw"]]
        output_tensor = filter_tensor
    _require_nonoptional_ports(envelope.graph, "convolution")

    tensors = (image, filter_tensor, result)
    if (
        len({tensor.uid for tensor in tensors}) != 3
        or len(envelope.ordered_tensors) != 3
        or any(tensor.virtual for tensor in tensors)
    ):
        raise ValueError(
            "convolution requires three distinct externally bound tensors"
        )
    if (
        len({tensor.data_type for tensor in tensors}) != 1
        or image.data_type not in _FLOATING_DATA_TYPES
        or node.compute_data_type != "float32"
    ):
        raise ValueError(
            "convolution requires matching floating storage and float32 compute"
        )

    attributes = node.attributes
    require_exact_keys(
        attributes,
        _CONVOLUTION_FPROP_ATTRIBUTE_KEYS
        if fprop
        else _CONVOLUTION_BACKWARD_ATTRIBUTE_KEYS,
        "convolution attributes",
    )
    spatial_rank = require_integer(
        attributes["spatial_rank"],
        "convolution attributes.spatial_rank",
        minimum=1,
        maximum=3,
    )
    rank = spatial_rank + 2
    if any(len(tensor.dimensions) != rank for tensor in tensors):
        raise ValueError(
            "convolution tensor ranks must equal spatial_rank + 2"
        )
    groups = require_integer(
        attributes["groups"],
        "convolution attributes.groups",
        minimum=1,
        maximum=MAX_I32,
    )
    convolution_mode = (
        0
        if fprop
        else require_integer(
            attributes["convolution_mode"],
            "convolution attributes.convolution_mode",
            minimum=0,
            maximum=1,
        )
    )
    pre_padding = _convolution_integer_array(
        attributes, "pre_padding", length=spatial_rank, minimum=0
    )
    post_padding = _convolution_integer_array(
        attributes, "post_padding", length=spatial_rank, minimum=0
    )
    stride = _convolution_integer_array(
        attributes, "stride", length=spatial_rank, minimum=1
    )
    dilation = _convolution_integer_array(
        attributes, "dilation", length=spatial_rank, minimum=1
    )

    batch, in_channels = image.dimensions[:2]
    out_channels = filter_tensor.dimensions[0]
    if (
        result.dimensions[0] != batch
        or result.dimensions[1] != out_channels
    ):
        raise ValueError(
            "convolution loss batch or channel dimension is incorrect"
        )
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError(
            "convolution channels must be divisible by groups"
        )
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    if filter_tensor.dimensions[1] != in_per_group:
        raise ValueError(
            "convolution filter channels do not match image tensor"
        )
    expected_result = [batch, out_channels]
    for axis in range(spatial_rank):
        expected_result.append(
            _convolution_output_dimension(
                image.dimensions[axis + 2],
                filter_tensor.dimensions[axis + 2],
                pre_padding[axis],
                post_padding[axis],
                stride[axis],
                dilation[axis],
            )
        )
    if result.dimensions != tuple(expected_result):
        raise ValueError("convolution output/loss shape is incorrect")

    n_outputs = require_integer(
        attributes["n_outputs"],
        "convolution attributes.n_outputs",
        minimum=1,
        maximum=MAX_I32,
    )
    if n_outputs != element_count(output_tensor):
        raise ValueError(
            "convolution n_outputs differs from the graph output tensor"
        )

    image_rows = batch * math.prod(image.dimensions[2:])
    result_rows = batch * math.prod(result.dimensions[2:])
    kernel_volume = math.prod(filter_tensor.dimensions[2:])
    if (
        image_rows > MAX_I32
        or result_rows > MAX_I32
        or kernel_volume > MAX_I32
        or batch * groups > MAX_I32
    ):
        raise ValueError("convolution launch extent exceeds int32")

    return ParsedConvolutionRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=operation,
        image=image,
        filter=filter_tensor,
        result=result,
        spatial_rank=spatial_rank,
        groups=groups,
        convolution_mode=convolution_mode,
        pre_padding=pre_padding,
        post_padding=post_padding,
        stride=stride,
        dilation=dilation,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_per_group=in_per_group,
        out_per_group=out_per_group,
        n_outputs=n_outputs,
        external_binding_uids=tuple(
            tensor.uid for tensor in envelope.ordered_tensors
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_add_square_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedAddSquareRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    if len(envelope.nodes) != 2:
        raise ValueError("AddSquare requires exactly two Graph nodes")
    square_node, add_node = envelope.nodes
    if (
        square_node.operation != "mul"
        or set(square_node.inputs) != {"left", "right"}
        or set(square_node.outputs) != {"output"}
        or square_node.inputs["left"] != square_node.inputs["right"]
        or add_node.operation != "add"
        or set(add_node.inputs) != {"left", "right"}
        or set(add_node.outputs) != {"output"}
        or add_node.inputs["right"] != square_node.outputs["output"]
    ):
        raise ValueError("AddSquare Graph topology is invalid")
    for index, raw_node in enumerate(
        require_list(envelope.graph["nodes"], "graph.nodes")
    ):
        node_value = require_object(raw_node, f"graph.nodes[{index}]")
        for port in (
            *require_list(
                node_value["inputs"], f"graph.nodes[{index}].inputs"
            ),
            *require_list(
                node_value["outputs"], f"graph.nodes[{index}].outputs"
            ),
        ):
            if "optional" in require_object(port, "AddSquare port"):
                raise ValueError("AddSquare ports must not be optional")

    right = envelope.registry[square_node.inputs["left"]]
    square = envelope.registry[square_node.outputs["output"]]
    left = envelope.registry[add_node.inputs["left"]]
    output = envelope.registry[add_node.outputs["output"]]
    tensors = (left, right, square, output)
    if (
        len(envelope.ordered_tensors) != 4
        or len({tensor.uid for tensor in tensors}) != 4
        or left.virtual
        or right.virtual
        or not square.virtual
        or output.virtual
    ):
        raise ValueError(
            "AddSquare requires three external tensors and one virtual tensor"
        )
    if (
        len({tensor.data_type for tensor in tensors}) != 1
        or left.data_type not in _FLOATING_DATA_TYPES
        or square_node.compute_data_type != "float32"
        or add_node.compute_data_type != "float32"
        or any(
            tensor.dimensions != output.dimensions
            or tensor.strides != output.strides
            or not is_physically_dense(tensor)
            for tensor in tensors
        )
    ):
        raise ValueError(
            "AddSquare tensors must use one floating, dense shape/layout"
        )
    count = element_count(output)
    for node, expected_mode, context in (
        (square_node, 18, "AddSquare square"),
        (add_node, 1, "AddSquare add"),
    ):
        require_exact_keys(node.attributes, _BINARY_ATTRIBUTE_KEYS, context)
        mode = require_integer(node.attributes["mode"], f"{context}.mode")
        pointwise_mode = require_integer(
            node.attributes["pointwise_mode"],
            f"{context}.pointwise_mode",
        )
        requested_elements = require_integer(
            node.attributes["n_elements"],
            f"{context}.n_elements",
            minimum=1,
        )
        alpha, _ = _float32(
            require_number(node.attributes["alpha"], f"{context}.alpha"),
            f"{context}.alpha",
        )
        if (
            mode != expected_mode
            or pointwise_mode != expected_mode
            or requested_elements != count
            or alpha != 1.0
        ):
            raise ValueError(f"{context} attributes are invalid")
    referenced_uids = {tensor.uid for tensor in tensors}
    if referenced_uids != {
        tensor.uid for tensor in envelope.ordered_tensors
    }:
        raise ValueError("AddSquare contains an unreferenced tensor")
    return ParsedAddSquareRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=add_node.id,
        operation="add_square",
        left=left,
        right=right,
        output=output,
        n_elements=count,
        external_binding_uids=tuple(
            tensor.uid
            for tensor in envelope.ordered_tensors
            if not tensor.virtual
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_conv_bias_relu_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedConvBiasReluRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    if len(envelope.nodes) != 3:
        raise ValueError("ConvBiasRelu requires exactly three Graph nodes")
    convolution_node, bias_node, relu_node = envelope.nodes
    if (
        convolution_node.operation != "convolution_fprop"
        or set(convolution_node.inputs) != {"input", "filter"}
        or set(convolution_node.outputs) != {"output"}
        or bias_node.operation != "add"
        or set(bias_node.inputs) != {"left", "right"}
        or set(bias_node.outputs) != {"output"}
        or bias_node.inputs["left"]
        != convolution_node.outputs["output"]
        or relu_node.operation != "relu"
        or set(relu_node.inputs) != {"input"}
        or set(relu_node.outputs) != {"output"}
        or relu_node.inputs["input"] != bias_node.outputs["output"]
    ):
        raise ValueError("ConvBiasRelu Graph topology is invalid")
    for index, raw_node in enumerate(
        require_list(envelope.graph["nodes"], "graph.nodes")
    ):
        node_value = require_object(raw_node, f"graph.nodes[{index}]")
        for port in (
            *require_list(
                node_value["inputs"], f"graph.nodes[{index}].inputs"
            ),
            *require_list(
                node_value["outputs"], f"graph.nodes[{index}].outputs"
            ),
        ):
            if "optional" in require_object(port, "ConvBiasRelu port"):
                raise ValueError("ConvBiasRelu ports must not be optional")

    image = envelope.registry[convolution_node.inputs["input"]]
    filter_tensor = envelope.registry[convolution_node.inputs["filter"]]
    convolution = envelope.registry[convolution_node.outputs["output"]]
    bias = envelope.registry[bias_node.inputs["right"]]
    biased = envelope.registry[bias_node.outputs["output"]]
    output = envelope.registry[relu_node.outputs["output"]]
    tensors = (image, filter_tensor, convolution, bias, biased, output)
    if (
        len(envelope.ordered_tensors) != 6
        or len({tensor.uid for tensor in tensors}) != 6
        or image.virtual
        or filter_tensor.virtual
        or not convolution.virtual
        or bias.virtual
        or not biased.virtual
        or output.virtual
    ):
        raise ValueError(
            "ConvBiasRelu requires four external and two virtual tensors"
        )
    if (
        len({tensor.data_type for tensor in tensors}) != 1
        or image.data_type not in _FLOATING_DATA_TYPES
        or any(len(tensor.dimensions) != 4 for tensor in tensors)
        or any(
            not is_physically_dense(tensor)
            for tensor in tensors
        )
        or convolution_node.compute_data_type != "float32"
        or bias_node.compute_data_type != "float32"
        or relu_node.compute_data_type != "float32"
    ):
        raise ValueError(
            "ConvBiasRelu requires dense rank-4 floating tensors and "
            "float32 compute"
        )

    attributes = convolution_node.attributes
    require_exact_keys(
        attributes,
        _CONVOLUTION_FPROP_ATTRIBUTE_KEYS,
        "ConvBiasRelu convolution attributes",
    )
    spatial_rank = require_integer(
        attributes["spatial_rank"],
        "ConvBiasRelu convolution spatial_rank",
        minimum=2,
        maximum=2,
    )
    groups = require_integer(
        attributes["groups"],
        "ConvBiasRelu convolution groups",
        minimum=1,
        maximum=MAX_I32,
    )
    pre_padding = _convolution_integer_array(
        attributes, "pre_padding", length=2, minimum=0
    )
    post_padding = _convolution_integer_array(
        attributes, "post_padding", length=2, minimum=0
    )
    stride = _convolution_integer_array(
        attributes, "stride", length=2, minimum=1
    )
    dilation = _convolution_integer_array(
        attributes, "dilation", length=2, minimum=1
    )
    batch, in_channels, input_height, input_width = image.dimensions
    out_channels, filter_channels, filter_height, filter_width = (
        filter_tensor.dimensions
    )
    if (
        in_channels % groups != 0
        or out_channels % groups != 0
        or filter_channels != in_channels // groups
    ):
        raise ValueError("ConvBiasRelu grouped channel geometry is invalid")
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    expected_output = (
        batch,
        out_channels,
        _convolution_output_dimension(
            input_height,
            filter_height,
            pre_padding[0],
            post_padding[0],
            stride[0],
            dilation[0],
        ),
        _convolution_output_dimension(
            input_width,
            filter_width,
            pre_padding[1],
            post_padding[1],
            stride[1],
            dilation[1],
        ),
    )
    if (
        convolution.dimensions != expected_output
        or biased.dimensions != expected_output
        or output.dimensions != expected_output
        or convolution.strides != biased.strides
        or convolution.strides != output.strides
        or bias.dimensions != (1, out_channels, 1, 1)
        or bias.strides[1] != 1
        or broadcast_dimensions(convolution, bias) != biased.dimensions
    ):
        raise ValueError("ConvBiasRelu tensor geometry or layout is invalid")
    count = element_count(output)
    n_outputs = require_integer(
        attributes["n_outputs"],
        "ConvBiasRelu convolution n_outputs",
        minimum=1,
        maximum=MAX_I32,
    )
    if n_outputs != count or element_count(convolution) != count:
        raise ValueError("ConvBiasRelu convolution n_outputs is invalid")

    require_exact_keys(
        bias_node.attributes,
        _BINARY_ATTRIBUTE_KEYS,
        "ConvBiasRelu bias attributes",
    )
    bias_alpha, _ = _float32(
        require_number(
            bias_node.attributes["alpha"], "ConvBiasRelu bias alpha"
        ),
        "ConvBiasRelu bias alpha",
    )
    if (
        require_integer(
            bias_node.attributes["mode"], "ConvBiasRelu bias mode"
        )
        != 1
        or require_integer(
            bias_node.attributes["pointwise_mode"],
            "ConvBiasRelu bias pointwise_mode",
        )
        != 1
        or require_integer(
            bias_node.attributes["n_elements"],
            "ConvBiasRelu bias n_elements",
            minimum=1,
        )
        != count
        or bias_alpha != 1.0
    ):
        raise ValueError("ConvBiasRelu bias attributes are invalid")

    require_exact_keys(
        relu_node.attributes,
        _UNARY_ATTRIBUTE_KEYS,
        "ConvBiasRelu ReLU attributes",
    )
    if (
        require_integer(
            relu_node.attributes["mode"], "ConvBiasRelu ReLU mode"
        )
        != 2
        or require_integer(
            relu_node.attributes["n_elements"],
            "ConvBiasRelu ReLU n_elements",
            minimum=1,
        )
        != count
        or require_integer(
            relu_node.attributes["has_upper_clip"],
            "ConvBiasRelu ReLU has_upper_clip",
            minimum=0,
            maximum=0,
        )
        != 0
        or relu_node.attributes["relu_upper_clip_set"] is not False
    ):
        raise ValueError("ConvBiasRelu ReLU attributes are invalid")
    expected_relu_scalars = {
        "relu_lower_clip": 0.0,
        "relu_upper_clip": 0.0,
        "relu_lower_clip_slope": 0.0,
        "swish_beta": 1.0,
        "elu_alpha": 1.0,
        "softplus_beta": 1.0,
        "negative_slope": 0.0,
        "lower_clip": 0.0,
        "upper_clip": 0.0,
    }
    for name, expected in expected_relu_scalars.items():
        actual, _ = _float32(
            require_number(
                relu_node.attributes[name], f"ConvBiasRelu ReLU {name}"
            ),
            f"ConvBiasRelu ReLU {name}",
        )
        if actual != expected:
            raise ValueError("ConvBiasRelu ReLU attributes are invalid")

    if {tensor.uid for tensor in tensors} != {
        tensor.uid for tensor in envelope.ordered_tensors
    }:
        raise ValueError("ConvBiasRelu contains an unreferenced tensor")
    if (
        batch * groups > MAX_I32
        or batch * expected_output[2] * expected_output[3] > MAX_I32
        or filter_height * filter_width * in_per_group > MAX_I32
    ):
        raise ValueError("ConvBiasRelu launch extent exceeds int32")
    return ParsedConvBiasReluRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=relu_node.id,
        operation="conv_bias_relu",
        image=image,
        filter=filter_tensor,
        bias=bias,
        output=output,
        spatial_rank=spatial_rank,
        groups=groups,
        pre_padding=pre_padding,
        post_padding=post_padding,
        stride=stride,
        dilation=dilation,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_per_group=in_per_group,
        out_per_group=out_per_group,
        n_outputs=count,
        external_binding_uids=tuple(
            tensor.uid
            for tensor in envelope.ordered_tensors
            if not tensor.virtual
        ),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def _normalization_integer_array(
    attributes: dict[str, Any],
    name: str,
    *,
    length: int,
) -> tuple[int, ...]:
    values = require_list(
        attributes[name], f"normalization attributes.{name}"
    )
    if len(values) != length:
        raise ValueError(
            f"normalization attributes.{name} length is invalid"
        )
    return tuple(
        require_integer(
            value,
            f"normalization attributes.{name}[{index}]",
            minimum=1,
            maximum=MAX_I32,
        )
        for index, value in enumerate(values)
    )


def _normalization_external_uids(
    envelope: _Envelope,
    tensors: tuple[TensorSpec, ...],
    context: str,
) -> tuple[int, ...]:
    if (
        len(envelope.ordered_tensors) != len(tensors)
        or len({tensor.uid for tensor in tensors}) != len(tensors)
        or any(tensor.virtual for tensor in tensors)
        or {tensor.uid for tensor in tensors}
        != {tensor.uid for tensor in envelope.ordered_tensors}
    ):
        raise ValueError(
            f"{context} requires distinct external tensors without extras"
        )
    return tuple(tensor.uid for tensor in envelope.ordered_tensors)


def parse_normalization_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedNormalizationRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if node.operation not in {"layernorm", "rmsnorm"}:
        raise ValueError("normalization parser received another operation")
    expected_outputs = (
        {"y", "mean", "inv_variance"}
        if node.operation == "layernorm"
        else {"y", "inv_variance"}
    )
    if (
        set(node.inputs) != {"x", "scale", "bias"}
        or set(node.outputs) != expected_outputs
    ):
        raise ValueError("normalization Graph ports are invalid")
    _require_nonoptional_ports(envelope.graph, node.operation)

    x = envelope.registry[node.inputs["x"]]
    scale = envelope.registry[node.inputs["scale"]]
    bias = envelope.registry[node.inputs["bias"]]
    y = envelope.registry[node.outputs["y"]]
    mean = (
        envelope.registry[node.outputs["mean"]]
        if node.operation == "layernorm"
        else None
    )
    inv_variance = envelope.registry[node.outputs["inv_variance"]]
    tensors = (
        (x, scale, bias, y, mean, inv_variance)
        if mean is not None
        else (x, scale, bias, y, inv_variance)
    )
    external_uids = _normalization_external_uids(
        envelope, tensors, node.operation
    )
    if (
        node.compute_data_type != "float32"
        or x.data_type not in _FLOATING_DATA_TYPES
        or y.data_type != x.data_type
        or scale.data_type != x.data_type
        or bias.data_type != x.data_type
        or x.dimensions != y.dimensions
        or x.strides != y.strides
        or not 1 <= len(x.dimensions) <= 8
        or not is_row_major_contiguous(x)
        or not is_row_major_contiguous(y)
        or not is_row_major_contiguous(scale)
        or not is_row_major_contiguous(bias)
        or scale.dimensions != bias.dimensions
        or not 1 <= len(scale.dimensions) <= len(x.dimensions)
        or inv_variance.data_type != "float32"
        or not is_row_major_contiguous(inv_variance)
        or (
            mean is not None
            and (
                mean.data_type != "float32"
                or not is_row_major_contiguous(mean)
            )
        )
    ):
        raise ValueError("normalization tensor metadata is invalid")

    leading = len(x.dimensions) - len(scale.dimensions)
    padded_scale = (1,) * leading + scale.dimensions
    normalized_start: int | None = None
    for axis, (input_dimension, scale_dimension) in enumerate(
        zip(x.dimensions, padded_scale, strict=True)
    ):
        if scale_dimension != 1:
            if scale_dimension != input_dimension:
                raise ValueError(
                    "normalization scale does not match the input suffix"
                )
            if normalized_start is None:
                normalized_start = axis
        elif normalized_start is not None and input_dimension != 1:
            raise ValueError(
                "normalization scale does not describe a suffix"
            )
    if normalized_start is None:
        normalized_start = len(x.dimensions) - 1
    normalized_elements = math.prod(x.dimensions[normalized_start:])
    rows = math.prod(x.dimensions[:normalized_start])
    expected_statistics = (
        x.dimensions[:normalized_start]
        + (1,) * (len(x.dimensions) - normalized_start)
    )
    statistics = (
        (mean, inv_variance) if mean is not None else (inv_variance,)
    )
    if (
        math.prod(scale.dimensions) != normalized_elements
        or any(
            statistic is None
            or statistic.dimensions != expected_statistics
            or element_count(statistic) != rows
            for statistic in statistics
        )
        or rows <= 0
        or normalized_elements <= 0
        or rows > MAX_I32
        or normalized_elements > MAX_I32
        or rows * normalized_elements != element_count(x)
    ):
        raise ValueError("normalization row geometry is invalid")

    require_exact_keys(
        node.attributes,
        _NORMALIZATION_ATTRIBUTE_KEYS,
        f"{node.operation} attributes",
    )
    epsilon, _ = _float32(
        require_number(
            node.attributes["epsilon"], f"{node.operation} epsilon"
        ),
        f"{node.operation} epsilon",
    )
    if (
        epsilon <= 0.0
        or require_integer(
            node.attributes["forward_phase"],
            f"{node.operation} forward_phase",
        )
        != 2
        or require_integer(
            node.attributes["rows"],
            f"{node.operation} rows",
            minimum=1,
            maximum=MAX_I32,
        )
        != rows
        or require_integer(
            node.attributes["normalized_elements"],
            f"{node.operation} normalized_elements",
            minimum=1,
            maximum=MAX_I32,
        )
        != normalized_elements
    ):
        raise ValueError("normalization attributes are inconsistent")
    return ParsedNormalizationRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=node.operation,
        x=x,
        scale=scale,
        bias=bias,
        y=y,
        mean=mean,
        inv_variance=inv_variance,
        rows=rows,
        normalized_elements=normalized_elements,
        epsilon=epsilon,
        external_binding_uids=external_uids,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def _validate_batchnorm_parameters(
    parameters: tuple[tuple[str, TensorSpec], ...],
    *,
    channels: int,
    data_type: str | None,
) -> None:
    for name, tensor in parameters:
        if (
            (data_type is not None and tensor.data_type != data_type)
            or (data_type is None and tensor.data_type != "float32")
            or element_count(tensor) != channels
            or not is_row_major_contiguous(tensor)
        ):
            raise ValueError(f"BatchNorm {name} metadata is invalid")


def _validate_batchnorm_common_attributes(
    node: _Node,
    x: TensorSpec,
    y: TensorSpec,
    *,
    expected_keys: set[str],
) -> tuple[int, int, int, int]:
    require_exact_keys(node.attributes, expected_keys, "BatchNorm attributes")
    if (
        node.compute_data_type != "float32"
        or x.data_type not in _FLOATING_DATA_TYPES
        or y.data_type != x.data_type
        or x.dimensions != y.dimensions
        or not 2 <= len(x.dimensions) <= 8
    ):
        raise ValueError("BatchNorm X/Y metadata is invalid")
    rank = len(x.dimensions)
    batch = x.dimensions[0]
    channels = x.dimensions[1]
    spatial = math.prod(x.dimensions[2:])
    n_elements = element_count(x)
    dimensions = _normalization_integer_array(
        node.attributes, "dimensions", length=rank
    )
    x_strides = _normalization_integer_array(
        node.attributes, "x_strides", length=rank
    )
    y_strides = _normalization_integer_array(
        node.attributes, "y_strides", length=rank
    )
    if (
        dimensions != x.dimensions
        or x_strides != x.strides
        or y_strides != y.strides
        or require_integer(
            node.attributes["rank"], "BatchNorm rank", minimum=2, maximum=8
        )
        != rank
        or require_integer(
            node.attributes["channels"],
            "BatchNorm channels",
            minimum=1,
            maximum=MAX_I32,
        )
        != channels
        or require_integer(
            node.attributes["spatial"],
            "BatchNorm spatial",
            minimum=1,
            maximum=MAX_I32,
        )
        != spatial
        or require_integer(
            node.attributes["n_elements"],
            "BatchNorm n_elements",
            minimum=1,
            maximum=MAX_I32,
        )
        != n_elements
    ):
        raise ValueError("BatchNorm derived attributes are inconsistent")
    return batch, channels, spatial, n_elements


def parse_batchnorm_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedBatchnormRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if (
        node.operation != "batchnorm"
        or set(node.inputs)
        != {
            "x",
            "scale",
            "bias",
            "previous_running_mean",
            "previous_running_variance",
        }
        or set(node.outputs)
        != {
            "y",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        }
    ):
        raise ValueError("BatchNorm Graph ports are invalid")
    _require_nonoptional_ports(envelope.graph, "BatchNorm")
    x = envelope.registry[node.inputs["x"]]
    scale = envelope.registry[node.inputs["scale"]]
    bias = envelope.registry[node.inputs["bias"]]
    previous_mean = envelope.registry[node.inputs["previous_running_mean"]]
    previous_variance = envelope.registry[
        node.inputs["previous_running_variance"]
    ]
    y = envelope.registry[node.outputs["y"]]
    mean = envelope.registry[node.outputs["mean"]]
    inv_variance = envelope.registry[node.outputs["inv_variance"]]
    next_mean = envelope.registry[node.outputs["next_running_mean"]]
    next_variance = envelope.registry[
        node.outputs["next_running_variance"]
    ]
    tensors = (
        x,
        scale,
        bias,
        previous_mean,
        previous_variance,
        y,
        mean,
        inv_variance,
        next_mean,
        next_variance,
    )
    external_uids = _normalization_external_uids(
        envelope, tensors, "BatchNorm"
    )
    batch, channels, spatial, n_elements = (
        _validate_batchnorm_common_attributes(
            node,
            x,
            y,
            expected_keys=_BATCHNORM_ATTRIBUTE_KEYS,
        )
    )
    _validate_batchnorm_parameters(
        (("scale", scale), ("bias", bias)),
        channels=channels,
        data_type=x.data_type,
    )
    _validate_batchnorm_parameters(
        (
            ("previous running mean", previous_mean),
            ("previous running variance", previous_variance),
            ("mean", mean),
            ("inverse variance", inv_variance),
            ("next running mean", next_mean),
            ("next running variance", next_variance),
        ),
        channels=channels,
        data_type=None,
    )
    configured_batch = require_integer(
        node.attributes["batch"],
        "BatchNorm batch",
        minimum=1,
        maximum=MAX_I32,
    )
    epsilon, _ = _float32(
        require_number(node.attributes["epsilon"], "BatchNorm epsilon"),
        "BatchNorm epsilon",
    )
    momentum, _ = _float32(
        require_number(node.attributes["momentum"], "BatchNorm momentum"),
        "BatchNorm momentum",
    )
    if (
        configured_batch != batch
        or epsilon <= 0.0
        or not 0.0 <= momentum <= 1.0
    ):
        raise ValueError("BatchNorm scalar attributes are inconsistent")
    return ParsedBatchnormRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=node.operation,
        x=x,
        scale=scale,
        bias=bias,
        previous_running_mean=previous_mean,
        previous_running_variance=previous_variance,
        y=y,
        mean=mean,
        inv_variance=inv_variance,
        next_running_mean=next_mean,
        next_running_variance=next_variance,
        batch=batch,
        channels=channels,
        spatial=spatial,
        n_elements=n_elements,
        epsilon=epsilon,
        momentum=momentum,
        external_binding_uids=external_uids,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_batchnorm_inference_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedBatchnormInferenceRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    if (
        node.operation != "batchnorm_inference"
        or set(node.inputs)
        != {"x", "mean", "inv_variance", "scale", "bias"}
        or set(node.outputs) != {"y"}
    ):
        raise ValueError("BatchNorm inference Graph ports are invalid")
    _require_nonoptional_ports(envelope.graph, "BatchNorm inference")
    x = envelope.registry[node.inputs["x"]]
    mean = envelope.registry[node.inputs["mean"]]
    inv_variance = envelope.registry[node.inputs["inv_variance"]]
    scale = envelope.registry[node.inputs["scale"]]
    bias = envelope.registry[node.inputs["bias"]]
    y = envelope.registry[node.outputs["y"]]
    tensors = (x, mean, inv_variance, scale, bias, y)
    external_uids = _normalization_external_uids(
        envelope, tensors, "BatchNorm inference"
    )
    batch, channels, spatial, n_elements = (
        _validate_batchnorm_common_attributes(
            node,
            x,
            y,
            expected_keys=_BATCHNORM_INFERENCE_ATTRIBUTE_KEYS,
        )
    )
    _validate_batchnorm_parameters(
        (
            ("mean", mean),
            ("inverse variance", inv_variance),
            ("scale", scale),
            ("bias", bias),
        ),
        channels=channels,
        data_type=None,
    )
    return ParsedBatchnormInferenceRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=node.operation,
        x=x,
        mean=mean,
        inv_variance=inv_variance,
        scale=scale,
        bias=bias,
        y=y,
        batch=batch,
        channels=channels,
        spatial=spatial,
        n_elements=n_elements,
        external_binding_uids=external_uids,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def _attention_flag(attributes: dict[str, Any], name: str) -> bool:
    value = require_integer(
        attributes[name],
        f"Attention attributes.{name}",
        minimum=0,
        maximum=1,
    )
    return value == 1


def _validate_attention_tensor(
    tensor: TensorSpec,
    *,
    dimensions: tuple[int, ...],
    data_type: str,
    name: str,
    allow_virtual: bool = False,
) -> None:
    if (
        tensor.dimensions != dimensions
        or tensor.data_type != data_type
        or (tensor.virtual and not allow_virtual)
    ):
        raise ValueError(f"Attention {name} tensor metadata is invalid")


def _validate_attention_scalar(tensor: TensorSpec, name: str) -> None:
    if (
        tensor.data_type != "float32"
        or math.prod(tensor.dimensions) != 1
        or tensor.virtual
    ):
        raise ValueError(
            f"FP8 Attention {name} must be an external float32 scalar"
        )


def parse_attention_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedAttentionRequest:
    envelope = _parse_envelope(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    node = envelope.node
    operation = node.operation
    if operation not in _ATTENTION_OPERATIONS:
        raise ValueError("Attention node type is invalid")
    _require_nonoptional_ports(envelope.graph, "Attention")
    if node.compute_data_type != "float32":
        raise ValueError("Attention requires float32 compute data type")
    attributes = node.attributes
    require_exact_keys(
        attributes, _ATTENTION_ATTRIBUTE_KEYS, "Attention attributes"
    )

    has_bias = _attention_flag(attributes, "has_bias")
    has_dbias = _attention_flag(attributes, "has_dbias")
    if operation == "sdpa":
        input_names = ("q", "k", "v") + (("bias",) if has_bias else ())
        output_names = ("o", "stats")
    elif operation == "sdpa_backward":
        input_names = (
            "q",
            "k",
            "v",
            "o",
            "do",
            "stats",
        ) + (("bias",) if has_bias else ())
        output_names = ("dq", "dk", "dv") + (
            ("dbias",) if has_dbias else ()
        )
    elif operation == "sdpa_fp8":
        if has_bias or has_dbias:
            raise ValueError("FP8 SDPA does not support bias or dbias")
        input_names = (
            "q",
            "k",
            "v",
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_s",
            "scale_s",
            "scale_o",
        )
        output_names = ("o", "stats", "amax_s", "amax_o")
    else:
        if has_bias or has_dbias:
            raise ValueError("FP8 SDPA backward does not support bias")
        input_names = (
            "q",
            "k",
            "v",
            "o",
            "do",
            "stats",
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_o",
            "descale_do",
            "descale_s",
            "descale_dp",
            "scale_s",
            "scale_dq",
            "scale_dk",
            "scale_dv",
            "scale_dp",
        )
        output_names = (
            "dq",
            "dk",
            "dv",
            "amax_dq",
            "amax_dk",
            "amax_dv",
            "amax_dp",
        )
    if set(node.inputs) != set(input_names) or set(node.outputs) != set(
        output_names
    ):
        raise ValueError("Attention Graph ports are invalid")

    ports = {
        name: envelope.registry[
            node.inputs[name] if name in node.inputs else node.outputs[name]
        ]
        for name in (*input_names, *output_names)
    }
    if (
        len({tensor.uid for tensor in ports.values()}) != len(ports)
        or len(envelope.ordered_tensors) != len(ports)
        or {tensor.uid for tensor in envelope.ordered_tensors}
        != {tensor.uid for tensor in ports.values()}
    ):
        raise ValueError(
            "Attention requires distinct tensors and no unused descriptors"
        )

    q, k, v = ports["q"], ports["k"], ports["v"]
    fp8 = operation in {"sdpa_fp8", "sdpa_fp8_backward"}
    allowed_types = _FP8_DATA_TYPES if fp8 else _FLOATING_DATA_TYPES
    if (
        q.data_type not in allowed_types
        or k.data_type != q.data_type
        or v.data_type != q.data_type
        or any(tensor.virtual for tensor in (q, k, v))
        or any(len(tensor.dimensions) != 4 for tensor in (q, k, v))
    ):
        raise ValueError("Attention Q/K/V metadata is invalid")
    qd, kd, vd = q.dimensions, k.dimensions, v.dimensions
    if (
        qd[0] != kd[0]
        or qd[0] != vd[0]
        or qd[3] != kd[3]
        or kd[2] != vd[2]
        or qd[1] % kd[1] != 0
        or qd[1] % vd[1] != 0
    ):
        raise ValueError("Attention Q/K/V shapes are inconsistent")
    batch, heads, sequence_q, head_dimension = qd
    key_heads = kd[1]
    value_heads = vd[1]
    sequence_kv = kd[2]
    value_dimension = vd[3]
    if head_dimension > 256 or value_dimension > 256:
        raise ValueError("Attention head dimensions above 256 are unsupported")
    if operation == "sdpa_backward" and key_heads != value_heads:
        raise ValueError("SDPA backward requires matching K/V head counts")
    if operation == "sdpa_fp8_backward" and (
        key_heads != value_heads
        or head_dimension != value_dimension
        or head_dimension > 128
    ):
        raise ValueError(
            "FP8 SDPA backward requires matching K/V heads and D == V <= 128"
        )

    output_dimensions = (batch, heads, sequence_q, value_dimension)
    stats_dimensions = (batch, heads, sequence_q, 1)
    _validate_attention_tensor(
        ports["o"],
        dimensions=output_dimensions,
        data_type=q.data_type,
        name="O",
    )
    generate_stats = _attention_flag(attributes, "generate_stats")
    backward = operation in {"sdpa_backward", "sdpa_fp8_backward"}
    if backward and not generate_stats:
        raise ValueError("Attention backward requires forward stats")
    stats = ports["stats"]
    _validate_attention_tensor(
        stats,
        dimensions=stats_dimensions,
        data_type="float32",
        name="stats",
        allow_virtual=not generate_stats,
    )
    if stats.virtual == generate_stats:
        raise ValueError(
            "Attention stats virtual state disagrees with generate_stats"
        )
    if backward:
        _validate_attention_tensor(
            ports["do"],
            dimensions=output_dimensions,
            data_type=q.data_type,
            name="dO",
        )
        for primal_name, gradient_name in (
            ("q", "dq"),
            ("k", "dk"),
            ("v", "dv"),
        ):
            primal = ports[primal_name]
            _validate_attention_tensor(
                ports[gradient_name],
                dimensions=primal.dimensions,
                data_type=q.data_type,
                name=gradient_name,
            )
    if has_bias:
        bias = ports["bias"]
        bias_dimensions = bias.dimensions
        if (
            bias.data_type != q.data_type
            or bias.virtual
            or len(bias_dimensions) != 4
            or bias_dimensions[0] not in {1, batch}
            or bias_dimensions[1] not in {1, heads}
            or bias_dimensions[2:] != (sequence_q, sequence_kv)
        ):
            raise ValueError("Attention bias metadata is invalid")
    if has_dbias:
        dbias = ports["dbias"]
        bias = ports.get("bias")
        if bias is None or dbias.dimensions != bias.dimensions:
            raise ValueError("Attention dbias must match bias")
        _validate_attention_tensor(
            dbias,
            dimensions=bias.dimensions,
            data_type=q.data_type,
            name="dbias",
        )

    if fp8:
        scalar_names = (
            (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_s",
                "scale_s",
                "scale_o",
                "amax_s",
                "amax_o",
            )
            if operation == "sdpa_fp8"
            else (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_o",
                "descale_do",
                "descale_s",
                "descale_dp",
                "scale_s",
                "scale_dq",
                "scale_dk",
                "scale_dv",
                "scale_dp",
                "amax_dq",
                "amax_dk",
                "amax_dv",
                "amax_dp",
            )
        )
        for name in scalar_names:
            _validate_attention_scalar(ports[name], name)

    integer_shape = {
        "batch": batch,
        "heads": heads,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "sequence_q": sequence_q,
        "sequence_kv": sequence_kv,
        "head_dimension": head_dimension,
        "value_dimension": value_dimension,
        "q_per_k": heads // key_heads,
        "q_per_v": heads // value_heads,
    }
    for name, expected in integer_shape.items():
        actual = require_integer(
            attributes[name],
            f"Attention attributes.{name}",
            minimum=1,
            maximum=MAX_I32,
        )
        if actual != expected:
            raise ValueError(
                f"Attention attributes.{name} disagrees with Q/K/V"
            )

    scale_set = attributes["attn_scale_set"]
    if not isinstance(scale_set, bool):
        raise ValueError("Attention attn_scale_set must be a boolean")
    attn_scale = _float32(
        require_number(
            attributes["attn_scale"], "Attention attributes.attn_scale"
        ),
        "Attention attributes.attn_scale",
    )[0]
    if attn_scale <= 0.0:
        raise ValueError("Attention scale must be positive")
    if not scale_set:
        default_scale = _float32(
            1.0 / math.sqrt(head_dimension), "default Attention scale"
        )[0]
        if attn_scale != default_scale:
            raise ValueError("Attention default scale is inconsistent")

    left_set = attributes["left_bound_set"]
    right_set = attributes["right_bound_set"]
    if not isinstance(left_set, bool) or not isinstance(right_set, bool):
        raise ValueError("Attention diagonal set flags must be booleans")
    left = require_integer(
        attributes["diagonal_band_left_bound"],
        "Attention left diagonal bound",
        minimum=0,
        maximum=MAX_I32,
    )
    right = require_integer(
        attributes["diagonal_band_right_bound"],
        "Attention right diagonal bound",
        minimum=0,
        maximum=MAX_I32,
    )
    if left_set and left < 1:
        raise ValueError("Attention left diagonal bound must be positive")
    alignment = require_integer(
        attributes["diagonal_alignment"],
        "Attention diagonal alignment",
        minimum=0,
        maximum=1,
    )
    shift = sequence_kv - sequence_q if alignment == 1 else 0
    expected_min = 1 - left + shift if left_set else -_UNBOUNDED_DIAGONAL
    expected_max = right + shift if right_set else _UNBOUNDED_DIAGONAL
    min_diag = require_integer(
        attributes["min_diag"],
        "Attention min_diag",
        minimum=-MAX_I32,
        maximum=MAX_I32,
    )
    max_diag = require_integer(
        attributes["max_diag"],
        "Attention max_diag",
        minimum=-MAX_I32,
        maximum=MAX_I32,
    )
    banded = _attention_flag(attributes, "banded")
    causal_top_left = _attention_flag(attributes, "causal_top_left")
    reverse_causal = _attention_flag(attributes, "reverse_causal")
    expected_causal = (
        alignment == 0
        and not left_set
        and right_set
        and right == 0
        and sequence_q == sequence_kv
    )
    expected_reverse = (
        alignment == 0 and not left_set and right_set and right == 0
    )
    if (
        min_diag != expected_min
        or max_diag != expected_max
        or min_diag > max_diag
        or banded != (left_set or right_set)
        or causal_top_left != expected_causal
        or reverse_causal != expected_reverse
    ):
        raise ValueError("Attention lowered diagonal metadata is inconsistent")

    external_uids = tuple(
        tensor.uid for tensor in envelope.ordered_tensors if not tensor.virtual
    )
    return ParsedAttentionRequest(
        flagdnn_version=envelope.version,
        target=envelope.target,
        compiler_identity=envelope.identity,
        autotune=envelope.autotune,
        node_id=node.id,
        operation=operation,
        port_tensors=tuple((name, ports[name]) for name in ports),
        batch=batch,
        heads=heads,
        key_heads=key_heads,
        value_heads=value_heads,
        sequence_q=sequence_q,
        sequence_kv=sequence_kv,
        head_dimension=head_dimension,
        value_dimension=value_dimension,
        q_per_k=heads // key_heads,
        q_per_v=heads // value_heads,
        min_diag=min_diag,
        max_diag=max_diag,
        has_bias=has_bias,
        has_dbias=has_dbias,
        banded=banded,
        causal_top_left=causal_top_left,
        reverse_causal=reverse_causal,
        generate_stats=generate_stats,
        attn_scale=attn_scale,
        external_binding_uids=external_uids,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )


def parse_pointwise_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedPointwiseRequest:
    preliminary = load_request(request_bytes)
    graph = require_object(preliminary.get("graph"), "request.graph")
    nodes = require_list(graph.get("nodes"), "graph.nodes")
    if len(nodes) != 1:
        raise ValueError("mthreads pointwise compiler requires one node")
    node = require_object(nodes[0], "graph.nodes[0]")
    attributes = require_object(
        node.get("attributes"), "graph.nodes[0].attributes"
    )
    mode = require_integer(attributes.get("mode"), "pointwise attributes.mode")
    if mode in _UNARY_OPERATIONS:
        return parse_unary_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if mode in _BINARY_OPERATIONS:
        return parse_binary_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if mode == _TERNARY_MODE:
        return parse_ternary_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    raise ValueError("mthreads pointwise mode is unsupported")


def parse_compiler_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedCompilerRequest:
    preliminary = load_request(request_bytes)
    graph = require_object(preliminary.get("graph"), "request.graph")
    nodes = require_list(graph.get("nodes"), "graph.nodes")
    if len(nodes) == 3:
        return parse_conv_bias_relu_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if len(nodes) == 2:
        return parse_add_square_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if len(nodes) != 1:
        raise ValueError("mthreads compiler requires one node")
    node = require_object(nodes[0], "graph.nodes[0]")
    if node.get("type") in _LAYOUT_OPERATIONS:
        return parse_layout_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") in _REDUCTION_OPERATIONS.values():
        return parse_reduction_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") == "matmul":
        return parse_matmul_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") in _CONVOLUTION_OPERATIONS:
        return parse_convolution_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") in {"layernorm", "rmsnorm"}:
        return parse_normalization_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") == "batchnorm":
        return parse_batchnorm_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") == "batchnorm_inference":
        return parse_batchnorm_inference_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    if node.get("type") in _ATTENTION_OPERATIONS:
        return parse_attention_request(
            request_bytes,
            expected_target=expected_target,
            expected_identity=expected_identity,
        )
    return parse_pointwise_request(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )


def parse_add_request(
    request_bytes: bytes,
    *,
    expected_target: str,
    expected_identity: str,
) -> ParsedAddRequest:
    """Compatibility entry point retaining the strict Add-only contract."""

    request = parse_binary_request(
        request_bytes,
        expected_target=expected_target,
        expected_identity=expected_identity,
    )
    if request.operation != "add":
        raise ValueError("Add parser received another pointwise operation")
    return request
