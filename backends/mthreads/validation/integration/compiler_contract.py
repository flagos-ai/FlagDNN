#!/usr/bin/env python3
"""Strict provider and binary pointwise Graph contract for mthreads."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Callable


TARGET = "musa-mtgpu-cc31-w32"
MAX_LINEAR_ELEMENTS = (1 << 31) - 1 - ((1 << 16) - 1)
_MATMUL_TLE_CONFIGS = {
    (32, 512, 512, 512): (128, 128, 32, 3, 2),
    (16, 1024, 1024, 1024): (256, 256, 32, 3, 2),
    (16, 2048, 2048, 512): (256, 256, 32, 3, 2),
    (8, 2048, 2048, 2048): (256, 256, 64, 3, 2),
    (32, 1024, 1024, 4096): (256, 256, 64, 3, 4),
    (4, 4096, 4096, 4096): (256, 256, 64, 3, 2),
}
ROOT_KEYS = {
    "schema_version",
    "artifact_kind",
    "flagdnn_version",
    "backend",
    "target",
    "engine",
    "request_sha256",
    "compiler_identity",
    "source_sha256",
    "workspace_size",
    "workspace_alignment",
    "external_binding_uids",
    "program",
    "files",
}
STAGE_KEYS = {
    "id",
    "node_id",
    "operation",
    "dependencies",
    "source",
    "function",
    "variants",
    "autotune",
}
VARIANT_KEYS = {
    "variant_id",
    "source",
    "source_sha256",
    "function",
    "full_signature",
    "grid",
    "num_warps",
    "num_stages",
    "arguments",
}
ARGUMENT_KEYS = {"kind", "semantic_name", "uid", "scalar_bits"}
UNARY_OPERATIONS = {
    "relu",
    "sqrt",
    "erf",
    "identity",
    "exp",
    "log",
    "neg",
    "abs",
    "ceil",
    "cos",
    "floor",
    "rsqrt",
    "sin",
    "tan",
    "reciprocal",
    "logical_not",
    "sigmoid",
    "tanh",
    "elu",
    "gelu",
    "softplus",
    "swish",
    "gelu_approx_tanh",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(
    command: list[str],
    *,
    environment: dict[str, str],
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if expect_success != (result.returncode == 0):
        fail(
            "command result differed:\n"
            + " ".join(command)
            + f"\nreturncode: {result.returncode}"
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def write_request(path: Path, value: object) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def dense_strides(dimensions: list[int]) -> list[int]:
    result = [1] * len(dimensions)
    running = 1
    for index in range(len(dimensions) - 1, -1, -1):
        result[index] = running
        running *= dimensions[index]
    return result


def uses_matmul_descriptor(request: dict[str, Any]) -> bool:
    tensors = request["graph"]["tensors"]
    a, b, output = tensors
    batch_dimensions = output["dimensions"][:-2]
    m, k = a["dimensions"][-2:]
    n = b["dimensions"][-1]
    batch = math.prod(batch_dimensions)
    return (
        a["data_type"] in {"float16", "bfloat16"}
        and a["dimensions"][:-2] == batch_dimensions
        and b["dimensions"][:-2] == batch_dimensions
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in tensors
        )
        and min(tensor["alignment"] for tensor in tensors) >= 16
        and m >= 128
        and n >= 128
        and k >= 64
        and m % 128 == 0
        and n % 128 == 0
        and k % 64 == 0
        and batch * m <= (1 << 31) - 1
        and batch * k <= (1 << 31) - 1
    )


def matmul_tle_config(
    request: dict[str, Any],
) -> tuple[int, int, int, int, int] | None:
    if not uses_matmul_descriptor(request):
        return None
    a, b, output = request["graph"]["tensors"]
    return _MATMUL_TLE_CONFIGS.get(
        (
            math.prod(output["dimensions"][:-2]),
            a["dimensions"][-2],
            b["dimensions"][-1],
            a["dimensions"][-1],
        )
    )


def uses_stride2_tile4_dgrad(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] != "convolution_dgrad":
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[outputs["dx"]]
    loss = tensor_by_uid[inputs["dy"]]
    filter_tensor = tensor_by_uid[inputs["w"]]
    attributes = node["attributes"]
    return (
        image["data_type"] in {"float32", "float16", "bfloat16"}
        and loss["data_type"] == image["data_type"]
        and filter_tensor["data_type"] == image["data_type"]
        and filter_tensor["dimensions"][2:] == [3, 3]
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, loss, filter_tensor)
        )
        and attributes["spatial_rank"] == 2
        and attributes["stride"] == [2, 2]
        and attributes["pre_padding"] == [1, 1]
        and attributes["post_padding"] == [1, 1]
        and attributes["dilation"] == [1, 1]
        and attributes["convolution_mode"] == 0
    )


def uses_stride2_packed2_1d_dgrad(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] != "convolution_dgrad":
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[outputs["dx"]]
    loss = tensor_by_uid[inputs["dy"]]
    filter_tensor = tensor_by_uid[inputs["w"]]
    attributes = node["attributes"]
    return (
        image["data_type"] in {"float32", "float16", "bfloat16"}
        and loss["data_type"] == image["data_type"]
        and filter_tensor["data_type"] == image["data_type"]
        and filter_tensor["dimensions"][2:] == [5]
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, loss, filter_tensor)
        )
        and attributes["spatial_rank"] == 1
        and attributes["stride"] == [2]
        and attributes["pre_padding"] == [2]
        and attributes["post_padding"] == [1]
        and attributes["dilation"] == [1]
        and attributes["convolution_mode"] == 0
    )


def uses_stride2_packed4_dgrad(request: dict[str, Any]) -> bool:
    if not uses_stride2_tile4_dgrad(request):
        return False
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[outputs["dx"]]
    return image["dimensions"][1] // node["attributes"]["groups"] <= 4


def uses_dense_stride2_dgrad(request: dict[str, Any]) -> bool:
    if not uses_stride2_tile4_dgrad(request):
        return False
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[outputs["dx"]]
    groups = node["attributes"]["groups"]
    return groups == 1 and image["dimensions"][1] // groups > 4


def uses_im2col_fprop(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] not in {"conv2d_fprop", "convolution_fprop"}:
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[inputs["input"]]
    weight = tensor_by_uid[inputs["filter"]]
    result = tensor_by_uid[outputs["output"]]
    attributes = node["attributes"]
    groups = attributes["groups"]
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }.get(image["data_type"])
    if element_size is None:
        return False
    output_area = math.prod(result["dimensions"][2:])
    reduction_extent = (
        image["dimensions"][1] // groups * math.prod(weight["dimensions"][2:])
    )
    filter_area = math.prod(weight["dimensions"][2:])
    standard_stride2_3x3 = (
        weight["dimensions"][2:] == [3, 3]
        and attributes["stride"] == [2, 2]
        and attributes["pre_padding"] == [1, 1]
        and attributes["post_padding"] == [1, 1]
        and attributes["dilation"] == [1, 1]
    )
    stride2_3x3 = (
        standard_stride2_3x3 and image["dimensions"][1] // groups >= 64
    )
    fp32_stem = (
        standard_stride2_3x3
        and image["data_type"] == "float32"
        and image["dimensions"] == [1, 3, 640, 640]
        and weight["dimensions"][1:] == [3, 3, 3]
        and weight["dimensions"][0] in {16, 32, 64, 96}
    )
    medium_batched = (
        image["dimensions"][0] >= 4
        and image["dimensions"][1] // groups >= 32
        and 2 <= filter_area <= 15
    )
    workspace = (
        image["dimensions"][0] * reduction_extent * output_area * element_size
    )
    workspace = max(4096, (workspace + 255) // 256 * 256)
    return (
        weight["data_type"] == image["data_type"]
        and result["data_type"] == image["data_type"]
        and attributes["spatial_rank"] == 2
        and groups == 1
        and attributes.get("convolution_mode", 0) == 0
        and (stride2_3x3 or fp32_stem or medium_batched)
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, weight, result)
        )
        and workspace <= 512 * 1024 * 1024
    )


def uses_p5_wgrad(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] != "convolution_wgrad":
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[inputs["x"]]
    loss = tensor_by_uid[inputs["dy"]]
    output = tensor_by_uid[outputs["dw"]]
    attributes = node["attributes"]
    return (
        image["data_type"] in {"float32", "float16", "bfloat16"}
        and loss["data_type"] == image["data_type"]
        and output["data_type"] == image["data_type"]
        and image["dimensions"][0] == 1
        and image["dimensions"][2:] == [40, 40]
        and loss["dimensions"][0] == 1
        and loss["dimensions"][2:] == [20, 20]
        and output["dimensions"][2:] == [3, 3]
        and image["dimensions"][1] == output["dimensions"][1]
        and loss["dimensions"][1] == output["dimensions"][0]
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, loss, output)
        )
        and attributes["spatial_rank"] == 2
        and attributes["groups"] == 1
        and attributes["convolution_mode"] == 0
        and attributes["stride"] == [2, 2]
        and attributes["pre_padding"] == [1, 1]
        and attributes["post_padding"] == [1, 1]
        and attributes["dilation"] == [1, 1]
    )


def uses_stem_wgrad(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] != "convolution_wgrad":
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[inputs["x"]]
    loss = tensor_by_uid[inputs["dy"]]
    output = tensor_by_uid[outputs["dw"]]
    attributes = node["attributes"]
    return (
        image["data_type"] in {"float32", "float16", "bfloat16"}
        and loss["data_type"] == image["data_type"]
        and output["data_type"] == image["data_type"]
        and image["dimensions"] == [1, 3, 640, 640]
        and loss["dimensions"][0] == 1
        and loss["dimensions"][1] in {16, 32, 64, 96}
        and loss["dimensions"][2:] == [320, 320]
        and output["dimensions"] == [loss["dimensions"][1], 3, 3, 3]
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, loss, output)
        )
        and attributes["spatial_rank"] == 2
        and attributes["groups"] == 1
        and attributes["convolution_mode"] == 0
        and attributes["stride"] == [2, 2]
        and attributes["pre_padding"] == [1, 1]
        and attributes["post_padding"] == [1, 1]
        and attributes["dilation"] == [1, 1]
    )


def uses_standard_wgrad(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] != "convolution_wgrad":
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[inputs["x"]]
    loss = tensor_by_uid[inputs["dy"]]
    output = tensor_by_uid[outputs["dw"]]
    attributes = node["attributes"]
    shape_key = (
        tuple(image["dimensions"]),
        tuple(loss["dimensions"]),
        tuple(output["dimensions"]),
        tuple(attributes["stride"]),
        tuple(attributes["pre_padding"]),
        tuple(attributes["post_padding"]),
    )
    supported_shapes = {
        (
            (8, 64, 56, 56),
            (8, 128, 28, 28),
            (128, 64, 3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
        ),
        (
            (8, 32, 32, 32),
            (8, 64, 32, 32),
            (64, 32, 3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
        ),
        (
            (8, 64, 28, 28),
            (8, 128, 28, 28),
            (128, 64, 1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
        ),
    }
    return (
        image["data_type"] in {"float32", "float16", "bfloat16"}
        and loss["data_type"] == image["data_type"]
        and output["data_type"] == image["data_type"]
        and shape_key in supported_shapes
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, loss, output)
        )
        and attributes["spatial_rank"] == 2
        and attributes["groups"] == 1
        and attributes["convolution_mode"] == 0
        and attributes["dilation"] == [1, 1]
    )


def uses_nd_packed_wgrad(request: dict[str, Any]) -> bool:
    graph = request["graph"]
    if graph["node_count"] != 1:
        return False
    node = graph["nodes"][0]
    if node["type"] != "convolution_wgrad":
        return False
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[inputs["x"]]
    loss = tensor_by_uid[inputs["dy"]]
    output = tensor_by_uid[outputs["dw"]]
    attributes = node["attributes"]
    shape_key = (
        tuple(image["dimensions"]),
        tuple(loss["dimensions"]),
        tuple(output["dimensions"]),
        tuple(attributes["stride"]),
        tuple(attributes["pre_padding"]),
        tuple(attributes["post_padding"]),
        tuple(attributes["dilation"]),
    )
    supported_shapes = {
        (
            (16, 32, 256),
            (16, 64, 256),
            (64, 32, 3),
            (1,),
            (1,),
            (1,),
            (1,),
        ),
        (
            (2, 8, 8, 16, 16),
            (2, 16, 8, 16, 16),
            (16, 8, 3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        (
            (1, 8, 10, 12, 14),
            (1, 12, 10, 11, 15),
            (12, 8, 2, 3, 3),
            (1, 1, 1),
            (1, 0, 1),
            (0, 1, 2),
            (1, 1, 1),
        ),
    }
    return (
        image["data_type"] in {"float32", "float16", "bfloat16"}
        and loss["data_type"] == image["data_type"]
        and output["data_type"] == image["data_type"]
        and shape_key in supported_shapes
        and all(
            tensor["strides"] == dense_strides(tensor["dimensions"])
            for tensor in (image, loss, output)
        )
        and attributes["spatial_rank"] in {1, 3}
        and attributes["groups"] == 1
        and attributes["convolution_mode"] == 0
    )


def captured_request(
    *,
    executable: Path,
    capture_compiler: Path,
    python: Path,
    environment: dict[str, str],
    root: Path,
    graph_kind: str = "add",
) -> dict[str, Any]:
    if graph_kind not in {
        "add",
        "add_square",
        "conv_bias_relu",
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "batchnorm_inference",
        "sdpa",
        "sdpa_backward",
        "sdpa_fp8",
        "sdpa_fp8_backward",
    }:
        fail(f"unknown public Graph capture kind: {graph_kind}")
    capture = root / f"captured-{graph_kind}-request.json"
    capture_environment = dict(environment)
    capture_environment["FLAGDNN_MTHREADS_CAPTURE_REQUEST"] = str(capture)
    capture_environment["FLAGDNN_MTHREADS_CAPTURE_GRAPH"] = graph_kind
    run(
        [
            str(executable),
            str(python),
            str(capture_compiler),
            str(root / f"capture-{graph_kind}-cache"),
        ],
        environment=capture_environment,
        expect_success=True,
    )
    if not capture.is_file():
        fail("public frontend capture did not produce a request")
    value = json.loads(capture.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 3
        or value.get("backend") != "mthreads"
        or value.get("target") != TARGET
    ):
        fail("captured public Graph request metadata is invalid")
    graph = value.get("graph")
    if not isinstance(graph, dict):
        fail("captured public Graph request has no Graph object")
    if graph_kind == "add":
        if graph.get("node_count") != 1:
            fail("captured public Graph request has no single Add node")
        node = graph["nodes"][0]
        if node.get("type") != "add" or {
            key: node.get("attributes", {}).get(key)
            for key in ("alpha", "mode", "n_elements", "pointwise_mode")
        } != {
            "alpha": -0.75,
            "mode": 1,
            "n_elements": 24,
            "pointwise_mode": 1,
        }:
            fail("captured public Add lowering fields differ")
    elif graph_kind == "add_square":
        nodes = graph.get("nodes")
        tensors = graph.get("tensors")
        if (
            graph.get("node_count") != 2
            or not isinstance(nodes, list)
            or [node.get("type") for node in nodes] != ["mul", "add"]
            or nodes[0].get("inputs")
            != [
                {"name": "left", "uid": 101},
                {"name": "right", "uid": 101},
            ]
            or nodes[0].get("outputs") != [{"name": "output", "uid": 103}]
            or nodes[1].get("inputs")
            != [
                {"name": "left", "uid": 100},
                {"name": "right", "uid": 103},
            ]
            or nodes[1].get("outputs") != [{"name": "output", "uid": 102}]
            or not isinstance(tensors, list)
            or [
                tensor["uid"]
                for tensor in tensors
                if tensor.get("virtual") is True
            ]
            != [103]
        ):
            fail("captured public AddSquare lowering fields differ")
    elif graph_kind == "conv_bias_relu":
        nodes = graph.get("nodes")
        tensors = graph.get("tensors")
        if (
            graph.get("node_count") != 3
            or graph.get("tensor_count") != 6
            or not isinstance(nodes, list)
            or [node.get("type") for node in nodes]
            != ["convolution_fprop", "add", "relu"]
            or nodes[0].get("inputs")
            != [
                {"name": "input", "uid": 200},
                {"name": "filter", "uid": 201},
            ]
            or nodes[0].get("outputs") != [{"name": "output", "uid": 204}]
            or nodes[0].get("attributes")
            != {
                "dilation": [1, 1],
                "groups": 1,
                "n_outputs": 180,
                "input_precision": 0,
                "post_padding": [1, 1],
                "pre_padding": [1, 1],
                "spatial_rank": 2,
                "stride": [1, 1],
            }
            or nodes[1].get("inputs")
            != [
                {"name": "left", "uid": 204},
                {"name": "right", "uid": 202},
            ]
            or nodes[1].get("outputs") != [{"name": "output", "uid": 205}]
            or nodes[2].get("inputs") != [{"name": "input", "uid": 205}]
            or nodes[2].get("outputs") != [{"name": "output", "uid": 203}]
            or not isinstance(tensors, list)
            or [
                tensor["uid"]
                for tensor in tensors
                if tensor.get("virtual") is True
            ]
            != [204, 205]
        ):
            fail("captured public ConvBiasRelu lowering fields differ")
    elif graph_kind in {
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "batchnorm_inference",
    }:
        nodes = graph.get("nodes")
        tensors = graph.get("tensors")
        expected_tensor_count = {
            "layernorm": 6,
            "rmsnorm": 5,
            "batchnorm": 10,
            "batchnorm_inference": 6,
        }[graph_kind]
        if (
            graph.get("node_count") != 1
            or graph.get("tensor_count") != expected_tensor_count
            or not isinstance(nodes, list)
            or len(nodes) != 1
            or nodes[0].get("type") != graph_kind
            or not isinstance(tensors, list)
            or len(tensors) != expected_tensor_count
            or any(tensor.get("virtual") is not False for tensor in tensors)
        ):
            fail(f"captured public {graph_kind} lowering fields differ")
    else:
        nodes = graph.get("nodes")
        tensors = graph.get("tensors")
        expected_tensor_count = {
            "sdpa": 6,
            "sdpa_backward": 9,
            "sdpa_fp8": 13,
            "sdpa_fp8_backward": 25,
        }[graph_kind]
        if (
            graph.get("node_count") != 1
            or graph.get("tensor_count") != expected_tensor_count
            or not isinstance(nodes, list)
            or len(nodes) != 1
            or nodes[0].get("type") != graph_kind
            or not isinstance(tensors, list)
            or len(tensors) != expected_tensor_count
        ):
            fail(f"captured public {graph_kind} lowering fields differ")
    return value


def identify(
    *,
    compiler: Path,
    python: Path,
    target: str,
    engine: str,
    output: Path,
    environment: dict[str, str],
    expect_success: bool,
) -> str | None:
    result = run(
        [
            str(python),
            str(compiler),
            "--identify",
            "--backend",
            "mthreads",
            "--target",
            target,
            "--execution-engine",
            engine,
            "--identity-output",
            str(output),
            "--quiet",
        ],
        environment=environment,
        expect_success=expect_success,
    )
    if not expect_success:
        return None
    lines = output.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or len(lines[0]) != 64:
        fail("mthreads compiler identity output is malformed")
    if any(character not in "0123456789abcdef" for character in lines[0]):
        fail("mthreads compiler identity is not lowercase SHA-256")
    metadata = json.loads(lines[1])
    if (
        metadata.get("schema_version") != 1
        or metadata.get("snapshot_schema_version") != 1
        or metadata.get("dependencies_complete") is not True
        or not metadata.get("files")
        or len(metadata["files"]) != len(metadata.get("snapshots", []))
    ):
        fail("mthreads compiler identity dependency closure is incomplete")
    return lines[0]


def add_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    data_type: str = "float32",
    alpha: float = 1.0,
    dimensions: list[int] | None = None,
    strides: list[int] | None = None,
    right_dimensions: list[int] | None = None,
    right_strides: list[int] | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    return binary_case(
        fixture,
        identity,
        operation="add",
        mode=1,
        data_type=data_type,
        alpha=alpha,
        dimensions=dimensions,
        strides=strides,
        right_dimensions=right_dimensions,
        right_strides=right_strides,
        autotune=autotune,
    )


def binary_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    operation: str,
    mode: int,
    data_type: str = "float32",
    output_data_type: str | None = None,
    compute_data_type: str | None = None,
    alpha: float = 1.0,
    dimensions: list[int] | None = None,
    strides: list[int] | None = None,
    right_dimensions: list[int] | None = None,
    right_strides: list[int] | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    dimensions = dimensions or [2, 3, 4]
    strides = strides or dense_strides(dimensions)
    right_dimensions = right_dimensions or list(dimensions)
    right_strides = right_strides or list(strides)
    tensors = value["graph"]["tensors"]
    tensors[0].update(
        data_type=data_type,
        dimensions=list(dimensions),
        strides=list(strides),
    )
    tensors[1].update(
        data_type=data_type,
        dimensions=list(right_dimensions),
        strides=list(right_strides),
    )
    tensors[2].update(
        data_type=output_data_type or data_type,
        dimensions=list(dimensions),
        strides=list(strides),
    )
    node = value["graph"]["nodes"][0]
    node["type"] = operation
    node["name"] = operation
    node["compute_data_type"] = compute_data_type or "float32"
    node["attributes"].update(
        alpha=alpha,
        mode=mode,
        n_elements=math.prod(dimensions),
        pointwise_mode=mode,
    )
    return value


def unary_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    operation: str,
    mode: int,
    data_type: str = "float32",
    compute_data_type: str = "float32",
    dimensions: list[int] | None = None,
    strides: list[int] | None = None,
    autotune: bool = False,
    negative_slope: float = 0.0,
    lower_clip: float = 0.0,
    upper_clip: float = 0.0,
    has_upper_clip: bool = False,
    swish_beta: float = 1.0,
    elu_alpha: float = 1.0,
    softplus_beta: float = 1.0,
) -> dict[str, Any]:
    if operation not in UNARY_OPERATIONS:
        fail(f"unknown unary contract operation: {operation}")
    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    dimensions = dimensions or [2, 3, 4]
    strides = strides or dense_strides(dimensions)
    input_tensor = value["graph"]["tensors"][0]
    output_tensor = value["graph"]["tensors"][2]
    for tensor in (input_tensor, output_tensor):
        tensor.update(
            data_type=data_type,
            dimensions=list(dimensions),
            strides=list(strides),
        )
    value["graph"]["tensors"] = [input_tensor, output_tensor]
    value["graph"]["tensor_count"] = 2
    node = value["graph"]["nodes"][0]
    node.update(
        type=operation,
        name=operation,
        compute_data_type=compute_data_type,
        inputs=[{"name": "input", "uid": input_tensor["uid"]}],
        outputs=[{"name": "output", "uid": output_tensor["uid"]}],
        attributes={
            "mode": mode,
            "relu_lower_clip": lower_clip,
            "relu_upper_clip": upper_clip,
            "relu_lower_clip_slope": negative_slope,
            "relu_upper_clip_set": has_upper_clip,
            "swish_beta": swish_beta,
            "elu_alpha": elu_alpha,
            "softplus_beta": softplus_beta,
            "n_elements": math.prod(dimensions),
            "has_upper_clip": int(has_upper_clip),
            "negative_slope": negative_slope,
            "lower_clip": lower_clip,
            "upper_clip": upper_clip,
        },
    )
    return value


def ternary_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    data_type: str = "float32",
    a_dimensions: list[int] | None = None,
    b_dimensions: list[int] | None = None,
    predicate_dimensions: list[int] | None = None,
    output_dimensions: list[int] | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    a_dimensions = a_dimensions or [2, 3, 4]
    b_dimensions = b_dimensions or list(a_dimensions)
    predicate_dimensions = predicate_dimensions or list(a_dimensions)
    output_dimensions = output_dimensions or list(a_dimensions)
    original = value["graph"]["tensors"]
    a = copy.deepcopy(original[0])
    b = copy.deepcopy(original[1])
    predicate = copy.deepcopy(original[2])
    output = copy.deepcopy(original[2])
    output["uid"] = 103
    for tensor, dimensions, tensor_type in (
        (a, a_dimensions, data_type),
        (b, b_dimensions, data_type),
        (predicate, predicate_dimensions, "boolean"),
        (output, output_dimensions, data_type),
    ):
        tensor.update(
            data_type=tensor_type,
            dimensions=list(dimensions),
            strides=dense_strides(dimensions),
        )
    value["graph"]["tensors"] = [a, b, predicate, output]
    value["graph"]["tensor_count"] = 4
    node = value["graph"]["nodes"][0]
    node.update(
        type="binary_select",
        name="binary_select",
        compute_data_type="float32",
        inputs=[
            {"name": "a", "uid": a["uid"]},
            {"name": "b", "uid": b["uid"]},
            {"name": "t", "uid": predicate["uid"]},
        ],
        outputs=[{"name": "output", "uid": output["uid"]}],
        attributes={
            "mode": 41,
            "n_elements": math.prod(output_dimensions),
        },
    )
    return value


def layout_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    operation: str,
    input_dimensions: list[int],
    output_dimensions: list[int],
    permutation: list[int] | None = None,
    starts: list[int] | None = None,
    limits: list[int] | None = None,
    slice_strides: list[int] | None = None,
    data_type: str = "float32",
    autotune: bool = False,
) -> dict[str, Any]:
    if operation not in {"reshape", "transpose", "slice"}:
        fail(f"unknown layout contract operation: {operation}")
    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    original = value["graph"]["tensors"]
    input_tensor = copy.deepcopy(original[0])
    output = copy.deepcopy(original[2])
    input_strides = dense_strides(input_dimensions)
    if operation == "transpose":
        if permutation is None:
            fail("transpose contract requires a permutation")
        output_strides = [input_strides[axis] for axis in permutation]
    elif operation == "slice":
        if starts is None or limits is None or slice_strides is None:
            fail("slice contract requires starts, limits, and strides")
        output_strides = [
            stride * step
            for stride, step in zip(input_strides, slice_strides, strict=True)
        ]
    else:
        output_strides = dense_strides(output_dimensions)
    input_tensor.update(
        data_type=data_type,
        dimensions=list(input_dimensions),
        strides=input_strides,
    )
    output.update(
        data_type=data_type,
        dimensions=list(output_dimensions),
        strides=output_strides,
    )
    value["graph"]["tensors"] = [input_tensor, output]
    value["graph"]["tensor_count"] = 2
    node = value["graph"]["nodes"][0]
    attributes: dict[str, Any] = {
        "n_elements": math.prod(output_dimensions),
        "input_dimensions": list(input_dimensions),
        "input_strides": input_strides,
        "output_dimensions": list(output_dimensions),
        "output_strides": output_strides,
    }
    if operation == "reshape":
        attributes.update(
            input_rank=len(input_dimensions),
            output_rank=len(output_dimensions),
            reshape_mode=2,
        )
    elif operation == "transpose":
        attributes.update(
            rank=len(input_dimensions),
            permutation=list(permutation or []),
        )
    else:
        attributes.update(
            rank=len(input_dimensions),
            starts=list(starts or []),
            limits=list(limits or []),
            slice_strides=list(slice_strides or []),
        )
    node.update(
        type=operation,
        name=operation,
        compute_data_type="float32",
        inputs=[{"name": "input", "uid": input_tensor["uid"]}],
        outputs=[{"name": "output", "uid": output["uid"]}],
        attributes=attributes,
    )
    return value


def reduction_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    operation: str,
    mode: int,
    input_dimensions: list[int],
    axis: int,
    keep_dimensions: bool,
    data_type: str = "float32",
    input_strides: list[int] | None = None,
    output_strides: list[int] | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    if {0: "reduction_sum", 1: "reduction_avg", 2: "reduction_mul"}.get(
        mode
    ) != operation:
        fail("reduction contract operation and mode differ")
    if not 0 <= axis < len(input_dimensions):
        fail("reduction contract axis is invalid")
    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    input_tensor = copy.deepcopy(value["graph"]["tensors"][0])
    output = copy.deepcopy(value["graph"]["tensors"][2])
    input_strides = input_strides or dense_strides(input_dimensions)
    output_dimensions = list(input_dimensions)
    if keep_dimensions:
        output_dimensions[axis] = 1
    else:
        del output_dimensions[axis]
    output_strides = (
        output_strides
        if output_strides is not None
        else dense_strides(output_dimensions)
    )
    input_tensor.update(
        data_type=data_type,
        dimensions=list(input_dimensions),
        strides=list(input_strides),
    )
    output.update(
        data_type=data_type,
        dimensions=output_dimensions,
        strides=list(output_strides),
    )
    value["graph"]["tensors"] = [input_tensor, output]
    value["graph"]["tensor_count"] = 2
    outer = math.prod(input_dimensions[:axis])
    extent = input_dimensions[axis]
    inner = math.prod(input_dimensions[axis + 1 :])
    node = value["graph"]["nodes"][0]
    node.update(
        type=operation,
        name=operation,
        compute_data_type="float32",
        inputs=[{"name": "input", "uid": input_tensor["uid"]}],
        outputs=[{"name": "output", "uid": output["uid"]}],
        attributes={
            "mode": mode,
            "axis": axis,
            "keep_dimensions": 1 if keep_dimensions else 0,
            "outer": outer,
            "reduction": extent,
            "inner": inner,
            "output_elements": outer * inner,
        },
    )
    return value


def matmul_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    a_dimensions: list[int],
    b_dimensions: list[int],
    data_type: str = "float32",
    a_strides: list[int] | None = None,
    b_strides: list[int] | None = None,
    output_strides: list[int] | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    if len(a_dimensions) < 2 or len(b_dimensions) < 2:
        fail("Matmul contract tensors must have rank at least two")
    if a_dimensions[-1] != b_dimensions[-2]:
        fail("Matmul contract dimensions do not contract")
    a_batch = a_dimensions[:-2]
    b_batch = b_dimensions[:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        b_dimension = b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            fail("Matmul contract batch dimensions do not broadcast")
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    output_dimensions = [
        *batch_dimensions,
        a_dimensions[-2],
        b_dimensions[-1],
    ]

    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    original = value["graph"]["tensors"]
    a = copy.deepcopy(original[0])
    b = copy.deepcopy(original[1])
    output = copy.deepcopy(original[2])
    for tensor, dimensions, strides in (
        (a, a_dimensions, a_strides),
        (b, b_dimensions, b_strides),
        (output, output_dimensions, output_strides),
    ):
        tensor.update(
            data_type=data_type,
            dimensions=list(dimensions),
            strides=(
                list(strides)
                if strides is not None
                else dense_strides(dimensions)
            ),
        )
    value["graph"]["tensors"] = [a, b, output]
    value["graph"]["tensor_count"] = 3
    node = value["graph"]["nodes"][0]
    node.update(
        type="matmul",
        name="matmul",
        compute_data_type="float32",
        inputs=[
            {"name": "a", "uid": a["uid"]},
            {"name": "b", "uid": b["uid"]},
        ],
        outputs=[{"name": "output", "uid": output["uid"]}],
        attributes={
            "batch": math.prod(batch_dimensions),
            "m": a_dimensions[-2],
            "n": b_dimensions[-1],
            "k": a_dimensions[-1],
        },
    )
    return value


def convolution_case(
    fixture: dict[str, Any],
    identity: str,
    *,
    operation: str,
    image_dimensions: list[int],
    filter_dimensions: list[int],
    pre_padding: list[int],
    post_padding: list[int],
    stride: list[int],
    dilation: list[int],
    groups: int = 1,
    convolution_mode: int = 0,
    data_type: str = "float32",
    image_strides: list[int] | None = None,
    filter_strides: list[int] | None = None,
    result_strides: list[int] | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    if operation not in {
        "conv2d_fprop",
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    }:
        fail("unknown convolution contract operation")
    spatial_rank = len(image_dimensions) - 2
    if (
        spatial_rank not in {1, 2, 3}
        or len(filter_dimensions) != spatial_rank + 2
        or any(
            len(values) != spatial_rank
            for values in (pre_padding, post_padding, stride, dilation)
        )
    ):
        fail("convolution contract spatial metadata is invalid")
    result_dimensions = [image_dimensions[0], filter_dimensions[0]]
    for axis in range(spatial_rank):
        effective = (filter_dimensions[axis + 2] - 1) * dilation[axis] + 1
        padded = (
            image_dimensions[axis + 2] + pre_padding[axis] + post_padding[axis]
        )
        if padded < effective:
            fail("convolution contract filter exceeds padded input")
        result_dimensions.append((padded - effective) // stride[axis] + 1)

    value = copy.deepcopy(fixture)
    value["compiler_identity"] = identity
    value["build_options"]["autotune"] = autotune
    original = value["graph"]["tensors"]
    image = copy.deepcopy(original[0])
    filter_tensor = copy.deepcopy(original[1])
    result = copy.deepcopy(original[2])
    for tensor, dimensions, strides in (
        (image, image_dimensions, image_strides),
        (filter_tensor, filter_dimensions, filter_strides),
        (result, result_dimensions, result_strides),
    ):
        tensor.update(
            data_type=data_type,
            dimensions=list(dimensions),
            strides=(
                list(strides)
                if strides is not None
                else dense_strides(dimensions)
            ),
        )
    value["graph"]["tensors"] = [image, filter_tensor, result]
    value["graph"]["tensor_count"] = 3
    node = value["graph"]["nodes"][0]
    attributes: dict[str, Any] = {
        "spatial_rank": spatial_rank,
        "groups": groups,
        "pre_padding": list(pre_padding),
        "post_padding": list(post_padding),
        "stride": list(stride),
        "dilation": list(dilation),
    }
    if operation in {"conv2d_fprop", "convolution_fprop"}:
        inputs = [
            {"name": "input", "uid": image["uid"]},
            {"name": "filter", "uid": filter_tensor["uid"]},
        ]
        outputs = [{"name": "output", "uid": result["uid"]}]
        attributes["n_outputs"] = math.prod(result_dimensions)
    elif operation == "convolution_dgrad":
        inputs = [
            {"name": "dy", "uid": result["uid"]},
            {"name": "w", "uid": filter_tensor["uid"]},
        ]
        outputs = [{"name": "dx", "uid": image["uid"]}]
        attributes["convolution_mode"] = convolution_mode
        attributes["n_outputs"] = math.prod(image_dimensions)
    else:
        inputs = [
            {"name": "dy", "uid": result["uid"]},
            {"name": "x", "uid": image["uid"]},
        ]
        outputs = [{"name": "dw", "uid": filter_tensor["uid"]}]
        attributes["convolution_mode"] = convolution_mode
        attributes["n_outputs"] = math.prod(filter_dimensions)
    node.update(
        type=operation,
        name=operation,
        compute_data_type="float32",
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
    )
    return value


def attention_expected_functions(request: dict[str, Any]) -> list[str]:
    operation = request["graph"]["nodes"][0]["type"]
    if operation == "sdpa":
        return ["_sdpa_fwd_kernel"]
    if operation == "sdpa_fp8":
        return [
            "_zero_sdpa_fp8_fwd_amax_kernel",
            "_sdpa_fp8_fwd_kernel",
        ]
    if operation == "sdpa_fp8_backward":
        return [
            "_zero_sdpa_fp8_bwd_amax_kernel",
            "_sdpa_fp8_bwd_dq_kernel",
            "_sdpa_fp8_bwd_dkdv_kernel",
        ]
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    result: list[str] = []
    if node["attributes"]["has_dbias"]:
        dbias = tensor_by_uid[output_uids["dbias"]]
        if (
            dbias["dimensions"][0] != node["attributes"]["batch"]
            or dbias["dimensions"][1] != node["attributes"]["heads"]
        ):
            result.append("_zero_contiguous_kernel")
    result.append("_sdpa_bwd_dq_dbias_kernel")
    q = tensor_by_uid[input_uids["q"]]
    k = tensor_by_uid[input_uids["k"]]
    v = tensor_by_uid[input_uids["v"]]
    if k["dimensions"][1] == v["dimensions"][1] and (
        q["dimensions"][3] == v["dimensions"][3]
    ):
        result.append("_sdpa_bwd_dkdv_kernel")
    else:
        result.extend(["_sdpa_bwd_dk_kernel", "_sdpa_bwd_dv_kernel"])
    return result


def compile_case(
    provider: Any,
    value: dict[str, Any],
    root: Path,
    name: str,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    request_path = root / f"{name}.json"
    request_bytes = write_request(request_path, value)
    output = root / f"{name}-artifact"
    result = provider.compile_request(request_path, output, "libtriton_jit")
    if result.get("torch_loaded") is not False:
        fail("mthreads compiler imported Torch during artifact planning")
    operation = value["graph"]["nodes"][0]["type"]
    nd_packed_wgrad = uses_nd_packed_wgrad(value)
    standard_wgrad = uses_standard_wgrad(value)
    standard_wgrad_stage_count = 2
    if standard_wgrad:
        node = value["graph"]["nodes"][0]
        output_uid = {port["name"]: port["uid"] for port in node["outputs"]}[
            "dw"
        ]
        weight = next(
            tensor
            for tensor in value["graph"]["tensors"]
            if tensor["uid"] == output_uid
        )
        standard_wgrad_stage_count = (
            2 if weight["dimensions"][2:] == [1, 1] else 3
        )
    expected_stage_count = (
        len(attention_expected_functions(value))
        if operation.startswith("sdpa")
        else (
            3
            if nd_packed_wgrad
            else (
                3
                if uses_dense_stride2_dgrad(value)
                else (
                    standard_wgrad_stage_count
                    if standard_wgrad
                    else (
                        2
                        if (uses_p5_wgrad(value) or uses_stem_wgrad(value))
                        else 2 if uses_im2col_fprop(value) else 1
                    )
                )
            )
        )
    )
    if (
        result.get("status") != "success"
        or result.get("node_count") != value["graph"]["node_count"]
        or result.get("stage_count") != expected_stage_count
    ):
        fail("mthreads compiler result counts differ from the Graph plan")
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(
        manifest,
        request_bytes=request_bytes,
        output=output,
        autotune=bool(value["build_options"]["autotune"]),
        request=value,
    )
    return manifest_bytes, manifest, result


def validate_add_square_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    nodes = graph["nodes"]
    tensors = graph["tensors"]
    square_node, add_node = nodes
    tensor_by_uid = {tensor["uid"]: tensor for tensor in tensors}
    right_uid = square_node["inputs"][0]["uid"]
    left_uid = add_node["inputs"][0]["uid"]
    output_uid = add_node["outputs"][0]["uid"]
    output_tensor = tensor_by_uid[output_uid]
    n_elements = math.prod(output_tensor["dimensions"])
    expected_bindings = [
        tensor["uid"] for tensor in tensors if not tensor["virtual"]
    ]
    if set(manifest) != ROOT_KEYS:
        fail("AddSquare manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != 4096
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("AddSquare manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/composite.py"
    ):
        fail("AddSquare manifest files table is invalid")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("AddSquare materialized source integrity differs")
    try:
        source_module = ast.parse(source.read_bytes(), filename=str(source))
    except (SyntaxError, ValueError) as error:
        fail(f"AddSquare materialized source is invalid: {error}")
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if functions != {"add_square_tensor_kernel"}:
        fail("AddSquare artifact exposes unexpected kernel functions")

    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 1
        or len(program["stages"]) != 1
    ):
        fail("AddSquare execution program schema differs")
    stage = program["stages"][0]
    if (
        set(stage) != STAGE_KEYS
        or stage["id"] != 0
        or stage["node_id"] != add_node["id"]
        or stage["operation"] != "add_square"
        or stage["dependencies"] != []
        or stage["source"] != "kernels/composite.py"
        or stage["function"] != "add_square_tensor_kernel"
    ):
        fail("AddSquare fused stage semantics differ")
    autotune_value = stage["autotune"]
    if (
        set(autotune_value)
        != {"enabled", "warmup", "repetitions", "selection_cache"}
        or autotune_value["enabled"] is not autotune
        or autotune_value["warmup"] != 3
        or autotune_value["repetitions"] != 10
        or autotune_value["selection_cache"] != "tuning/stage-0.json"
    ):
        fail("AddSquare autotune metadata differs")

    expected_candidates = (
        {
            (block_size, num_warps, 1)
            for block_size in (128, 256, 512)
            for num_warps in (2, 4, 8)
        }
        if autotune
        else {(256, 4, 1)}
    )
    actual_candidates: set[tuple[int, int, int]] = set()
    expected_arguments = [
        ("tensor", "left", left_uid, None),
        ("tensor", "right", right_uid, None),
        ("tensor", "output", output_uid, None),
        (
            "scalar_i32",
            "n_elements",
            None,
            struct.pack("<i", n_elements).hex(),
        ),
    ]
    for variant in stage["variants"]:
        if set(variant) != VARIANT_KEYS:
            fail("AddSquare variant schema is not closed")
        parts = variant["variant_id"].split("-")
        if (
            len(parts) != 6
            or parts[0] != "block"
            or parts[2] != "warps"
            or parts[4] != "stages"
        ):
            fail("AddSquare variant id is malformed")
        try:
            block_size = int(parts[1])
            num_warps = int(parts[3])
            num_stages = int(parts[5])
        except ValueError:
            fail("AddSquare variant id contains a noninteger")
        candidate = (block_size, num_warps, num_stages)
        actual_candidates.add(candidate)
        arguments = variant["arguments"]
        actual_arguments = [
            (
                argument["kind"],
                argument["semantic_name"],
                argument["uid"],
                argument["scalar_bits"],
            )
            for argument in arguments
            if set(argument) == ARGUMENT_KEYS
        ]
        signature_tokens = variant["full_signature"].split(",")
        if (
            len(arguments) != 4
            or len(actual_arguments) != 4
            or actual_arguments != expected_arguments
            or variant["source"] != stage["source"]
            or variant["source_sha256"] != manifest["source_sha256"]
            or variant["function"] != stage["function"]
            or variant["grid"]
            != [(n_elements + block_size - 1) // block_size, 1, 1]
            or variant["num_warps"] != num_warps
            or variant["num_stages"] != num_stages
            or len(signature_tokens) != 7
            or signature_tokens[-4:] != ["i32", "1", str(block_size), "1"]
        ):
            fail("AddSquare variant launch ABI differs")
    if actual_candidates != expected_candidates:
        fail("AddSquare tuning candidate set differs")


def validate_conv_bias_relu_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    convolution_node, _, relu_node = graph["nodes"]
    tensors = graph["tensors"]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in tensors}
    image_uid = convolution_node["inputs"][0]["uid"]
    filter_uid = convolution_node["inputs"][1]["uid"]
    convolution_uid = convolution_node["outputs"][0]["uid"]
    bias_uid = graph["nodes"][1]["inputs"][1]["uid"]
    output_uid = relu_node["outputs"][0]["uid"]
    image = tensor_by_uid[image_uid]
    filter_tensor = tensor_by_uid[filter_uid]
    convolution = tensor_by_uid[convolution_uid]
    output_tensor = tensor_by_uid[output_uid]
    attributes = convolution_node["attributes"]
    expected_bindings = [
        tensor["uid"] for tensor in tensors if not tensor["virtual"]
    ]
    if set(manifest) != ROOT_KEYS:
        fail("ConvBiasRelu manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != 4096
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("ConvBiasRelu manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/conv_bias_relu.py"
    ):
        fail("ConvBiasRelu manifest files table is invalid")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("ConvBiasRelu source integrity differs")
    try:
        source_module = ast.parse(source.read_bytes(), filename=str(source))
    except (SyntaxError, ValueError) as error:
        fail(f"ConvBiasRelu materialized source is invalid: {error}")
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if functions != {"conv_bias_relu_2d_kernel"}:
        fail("ConvBiasRelu artifact exposes unexpected kernel functions")

    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 1
        or len(program["stages"]) != 1
    ):
        fail("ConvBiasRelu execution program schema differs")
    stage = program["stages"][0]
    if (
        set(stage) != STAGE_KEYS
        or stage["id"] != 0
        or stage["node_id"] != relu_node["id"]
        or stage["operation"] != "conv_bias_relu"
        or stage["dependencies"] != []
        or stage["source"] != "kernels/conv_bias_relu.py"
        or stage["function"] != "conv_bias_relu_2d_kernel"
    ):
        fail("ConvBiasRelu fused stage semantics differ")
    autotune_value = stage["autotune"]
    if (
        set(autotune_value)
        != {"enabled", "warmup", "repetitions", "selection_cache"}
        or autotune_value["enabled"] is not autotune
        or autotune_value["warmup"] != 3
        or autotune_value["repetitions"] != 10
        or autotune_value["selection_cache"] != "tuning/stage-0.json"
    ):
        fail("ConvBiasRelu autotune metadata differs")

    expected_candidates = (
        {
            (block_size, num_warps, num_stages)
            for block_size in (16, 32)
            for num_warps in (4, 8)
            for num_stages in (1, 2)
        }
        if autotune
        else {(32, 4, 2)}
    )
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[image["data_type"]]
    expected_arguments = [
        ("tensor", "input", image_uid, None),
        ("tensor", "filter", filter_uid, None),
        ("tensor", "bias", bias_uid, None),
        ("tensor", "output", output_uid, None),
    ]
    expected_signature_tail_prefix = [
        str(image["dimensions"][2]),
        str(image["dimensions"][3]),
        str(output_tensor["dimensions"][2]),
        str(output_tensor["dimensions"][3]),
        str(image["dimensions"][1]),
        str(filter_tensor["dimensions"][0]),
        str(image["dimensions"][1] // attributes["groups"]),
        str(filter_tensor["dimensions"][0] // attributes["groups"]),
        str(attributes["groups"]),
        str(attributes["stride"][0]),
        str(attributes["stride"][1]),
        str(attributes["pre_padding"][0]),
        str(attributes["pre_padding"][1]),
        str(attributes["dilation"][0]),
        str(attributes["dilation"][1]),
        str(filter_tensor["dimensions"][2]),
        str(filter_tensor["dimensions"][3]),
        "1",
    ]
    output_spatial = (
        output_tensor["dimensions"][2] * output_tensor["dimensions"][3]
    )
    actual_candidates: set[tuple[int, int, int]] = set()
    for variant in stage["variants"]:
        if set(variant) != VARIANT_KEYS:
            fail("ConvBiasRelu variant schema is not closed")
        parts = variant["variant_id"].split("-")
        if (
            len(parts) != 6
            or parts[0] != "block"
            or parts[2] != "warps"
            or parts[4] != "stages"
        ):
            fail("ConvBiasRelu variant id is malformed")
        try:
            block_size = int(parts[1])
            num_warps = int(parts[3])
            num_stages = int(parts[5])
        except ValueError:
            fail("ConvBiasRelu variant id contains a noninteger")
        actual_candidates.add((block_size, num_warps, num_stages))
        arguments = variant["arguments"]
        actual_arguments = [
            (
                argument["kind"],
                argument["semantic_name"],
                argument["uid"],
                argument["scalar_bits"],
            )
            for argument in arguments
            if set(argument) == ARGUMENT_KEYS
        ]
        signature_tokens = variant["full_signature"].split(",")
        expected_signature_tail = [
            *expected_signature_tail_prefix,
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
            str(dtype_id),
            "0",
            *(str(value) for value in image["strides"]),
            *(str(value) for value in filter_tensor["strides"]),
            *(str(value) for value in output_tensor["strides"]),
        ]
        expected_grid = [
            ((output_spatial + block_size - 1) // block_size)
            * (
                (
                    filter_tensor["dimensions"][0] // attributes["groups"]
                    + block_size
                    - 1
                )
                // block_size
            ),
            image["dimensions"][0] * attributes["groups"],
            1,
        ]
        if (
            len(arguments) != 4
            or len(actual_arguments) != 4
            or actual_arguments != expected_arguments
            or variant["source"] != stage["source"]
            or variant["source_sha256"] != manifest["source_sha256"]
            or variant["function"] != stage["function"]
            or variant["grid"] != expected_grid
            or variant["num_warps"] != num_warps
            or variant["num_stages"] != num_stages
            or len(signature_tokens) != 40
            or signature_tokens[4:] != expected_signature_tail
        ):
            fail("ConvBiasRelu variant launch ABI differs")
    if actual_candidates != expected_candidates:
        fail("ConvBiasRelu tuning candidate set differs")


def validate_normalization_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    operation = node["type"]
    tensors = graph["tensors"]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in tensors}
    inputs = {port["name"]: port["uid"] for port in node["inputs"]}
    outputs = {port["name"]: port["uid"] for port in node["outputs"]}
    expected_bindings = [tensor["uid"] for tensor in tensors]
    if set(manifest) != ROOT_KEYS or (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != 4096
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail(f"{operation} manifest root contract differs")

    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/normalization.py"
    ):
        fail(f"{operation} source descriptor differs")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail(f"{operation} source integrity differs")
    try:
        module = ast.parse(source.read_bytes(), filename=str(source))
    except (SyntaxError, ValueError) as error:
        fail(f"{operation} materialized source is invalid: {error}")
    source_functions = {
        item.name
        for item in module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_functions = {
        "layernorm": {"layer_norm_kernel"},
        "rmsnorm": {"rms_norm_kernel"},
        "batchnorm": {"batch_norm_nchw_kernel", "batch_norm_kernel"},
        "batchnorm_inference": {
            "batch_norm_inference_nchw_kernel",
            "batch_norm_inference_kernel",
        },
    }[operation]
    if not required_functions.issubset(source_functions):
        fail(f"{operation} source is missing a registry entry point")

    x = tensor_by_uid[inputs["x"]]
    y = tensor_by_uid[outputs["y"]]
    row_major = x["strides"] == dense_strides(x["dimensions"]) and y[
        "strides"
    ] == dense_strides(y["dimensions"])
    batch_block = 1 << (x["dimensions"][0] - 1).bit_length()
    specialized = row_major and (
        operation != "batchnorm" or batch_block <= 256
    )
    function = {
        "layernorm": "layer_norm_kernel",
        "rmsnorm": "rms_norm_kernel",
        "batchnorm": (
            "batch_norm_nchw_kernel" if specialized else "batch_norm_kernel"
        ),
        "batchnorm_inference": (
            "batch_norm_inference_nchw_kernel"
            if specialized and math.prod(x["dimensions"][2:]) > 1
            else "batch_norm_inference_kernel"
        ),
    }[operation]
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 1
        or len(program["stages"]) != 1
    ):
        fail(f"{operation} program schema differs")
    stage = program["stages"][0]
    if (
        set(stage) != STAGE_KEYS
        or stage["id"] != 0
        or stage["node_id"] != node["id"]
        or stage["operation"] != operation
        or stage["dependencies"] != []
        or stage["source"] != "kernels/normalization.py"
        or stage["function"] != function
    ):
        fail(f"{operation} stage contract differs")
    tuning = stage["autotune"]
    if (
        set(tuning) != {"enabled", "warmup", "repetitions", "selection_cache"}
        or tuning["enabled"] is not autotune
        or tuning["warmup"] != 3
        or tuning["repetitions"] != 10
        or tuning["selection_cache"] != "tuning/stage-0.json"
    ):
        fail(f"{operation} autotune contract differs")

    if operation in {"layernorm", "rmsnorm"}:
        normalized_elements = node["attributes"]["normalized_elements"]
        steady_block = (
            min(4096, 1 << (normalized_elements - 1).bit_length())
            if normalized_elements > 513
            else 256
        )
        steady_candidate = (
            steady_block,
            4 if steady_block <= 1024 else 8,
            1,
        )
        expected_candidates = (
            {
                steady_candidate,
                *(
                    (block, warps, 1)
                    for block in (256, 512)
                    for warps in (4, 8)
                ),
            }
            if autotune
            else {steady_candidate}
        )
    else:
        candidate_blocks = (128, 256, 512)
        shape_candidates: set[tuple[int, int, int]] = set()
        if operation == "batchnorm" and specialized:
            items_per_channel = x["dimensions"][0] * math.prod(
                x["dimensions"][2:]
            )
            if items_per_channel > 512:
                shape_block = min(
                    16384,
                    1 << (items_per_channel - 1).bit_length(),
                )
                shape_candidates.add(
                    (
                        shape_block,
                        4 if shape_block <= 1024 else 8,
                        1,
                    )
                )
        expected_candidates = (
            shape_candidates
            | {
                (block, warps, 1)
                for block in candidate_blocks
                if not (
                    operation == "batchnorm"
                    and specialized
                    and block < batch_block
                )
                for warps in (4, 8)
            }
            if autotune
            else {(256, 4, 1)}
        )

    pointer_types = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }

    def pointer_token(tensor: dict[str, Any]) -> str:
        token = pointer_types[tensor["data_type"]]
        return token + (":16" if tensor["alignment"] >= 16 else "")

    def tensor_argument(name: str, uid: int) -> tuple[str, str, int, None]:
        return ("tensor", name, uid, None)

    def scalar_argument(name: str, value: int) -> tuple[str, str, None, str]:
        return ("scalar_i32", name, None, struct.pack("<i", value).hex())

    attributes = node["attributes"]
    dimensions = x["dimensions"]
    batch = dimensions[0]
    channels = dimensions[1] if len(dimensions) >= 2 else 0
    spatial = math.prod(dimensions[2:]) if len(dimensions) >= 2 else 0
    n_elements = math.prod(dimensions)
    metadata = (
        [1] * (8 - len(dimensions))
        + dimensions
        + [0] * (8 - len(dimensions))
        + x["strides"]
        + [0] * (8 - len(dimensions))
        + y["strides"]
    )
    if operation == "layernorm":
        tensor_order = [
            ("x", inputs["x"]),
            ("y", outputs["y"]),
            ("mean", outputs["mean"]),
            ("inv_variance", outputs["inv_variance"]),
            ("scale", inputs["scale"]),
            ("bias", inputs["bias"]),
        ]
        expected_arguments = [
            tensor_argument(name, uid) for name, uid in tensor_order
        ] + [scalar_argument("rows", attributes["rows"])]
    elif operation == "rmsnorm":
        tensor_order = [
            ("x", inputs["x"]),
            ("y", outputs["y"]),
            ("scale", inputs["scale"]),
            ("bias", inputs["bias"]),
            ("inv_variance", outputs["inv_variance"]),
        ]
        expected_arguments = [
            tensor_argument(name, uid) for name, uid in tensor_order
        ] + [scalar_argument("rows", attributes["rows"])]
    elif operation == "batchnorm":
        tensor_order = [
            ("x", inputs["x"]),
            ("y", outputs["y"]),
            ("previous_running_mean", inputs["previous_running_mean"]),
            (
                "previous_running_variance",
                inputs["previous_running_variance"],
            ),
            ("scale", inputs["scale"]),
            ("bias", inputs["bias"]),
            ("mean", outputs["mean"]),
            ("inv_variance", outputs["inv_variance"]),
            ("next_running_mean", outputs["next_running_mean"]),
            ("next_running_variance", outputs["next_running_variance"]),
        ]
        expected_arguments = [
            tensor_argument(name, uid) for name, uid in tensor_order
        ]
        if not specialized:
            expected_arguments += [
                scalar_argument("batch", batch),
                scalar_argument("channels", channels),
                scalar_argument("spatial", spatial),
            ]
    else:
        tensor_order = [
            ("x", inputs["x"]),
            ("mean", inputs["mean"]),
            ("inv_variance", inputs["inv_variance"]),
            ("scale", inputs["scale"]),
            ("bias", inputs["bias"]),
            ("y", outputs["y"]),
        ]
        expected_arguments = [
            tensor_argument(name, uid) for name, uid in tensor_order
        ]
        if not specialized:
            expected_arguments += [
                scalar_argument("n_elements", n_elements),
                scalar_argument("channels", channels),
                scalar_argument("spatial", spatial),
            ]

    actual_candidates: set[tuple[int, int, int]] = set()
    for variant in stage["variants"]:
        if set(variant) != VARIANT_KEYS:
            fail(f"{operation} variant schema is not closed")
        parts = variant["variant_id"].split("-")
        if (
            len(parts) != 6
            or parts[0] != "block"
            or parts[2] != "warps"
            or parts[4] != "stages"
        ):
            fail(f"{operation} variant id is malformed")
        try:
            block = int(parts[1])
            warps = int(parts[3])
            stages = int(parts[5])
        except ValueError:
            fail(f"{operation} variant id contains a noninteger")
        actual_candidates.add((block, warps, stages))
        actual_arguments = [
            (
                argument["kind"],
                argument["semantic_name"],
                argument["uid"],
                argument["scalar_bits"],
            )
            for argument in variant["arguments"]
            if set(argument) == ARGUMENT_KEYS
        ]
        signature = [
            pointer_token(tensor_by_uid[uid]) for _, uid in tensor_order
        ]
        if operation == "layernorm":
            signature += [
                "i32",
                repr(float(attributes["epsilon"])),
                str(attributes["normalized_elements"]),
                str(block),
                "1",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
            ]
            expected_grid = [attributes["rows"], 1, 1]
        elif operation == "rmsnorm":
            signature += [
                "i32",
                str(attributes["normalized_elements"]),
                repr(float(attributes["epsilon"])),
                str(block),
                "1",
                "1",
                "1",
                "1",
                "0",
            ]
            expected_grid = [attributes["rows"], 1, 1]
        elif operation == "batchnorm":
            if specialized:
                signature += [
                    str(batch),
                    str(channels),
                    str(spatial),
                    repr(float(attributes["epsilon"])),
                    repr(float(attributes["momentum"])),
                    str(block),
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                ]
            else:
                signature += ["i32", "i32", "i32"]
                signature += [
                    repr(float(attributes["epsilon"])),
                    repr(float(attributes["momentum"])),
                    str(block),
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    *[str(value) for value in metadata],
                ]
            expected_grid = [channels, 1, 1]
        elif specialized:
            signature += [
                str(channels),
                str(spatial),
                "0.0",
                str(block),
                "1",
                "1",
                "1",
            ]
            expected_grid = [
                batch * channels,
                (spatial + block - 1) // block,
                1,
            ]
        else:
            signature += ["i32", "i32", "i32", "0.0", str(block)]
            signature += [
                "1",
                "1",
                "1",
                "1",
                *[str(value) for value in metadata],
            ]
            expected_grid = [(n_elements + block - 1) // block, 1, 1]
        if (
            len(actual_arguments) != len(expected_arguments)
            or actual_arguments != expected_arguments
            or variant["source"] != stage["source"]
            or variant["source_sha256"] != manifest["source_sha256"]
            or variant["function"] != function
            or variant["full_signature"].split(",") != signature
            or len(signature)
            != len(
                next(
                    node.args.args
                    for node in module.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == function
                )
            )
            or variant["grid"] != expected_grid
            or variant["num_warps"] != warps
            or variant["num_stages"] != stages
        ):
            fail(f"{operation} variant launch ABI differs")
    if actual_candidates != expected_candidates:
        fail(f"{operation} tuning candidates differ")


def validate_attention_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    expected_functions = attention_expected_functions(request)
    operation = request["graph"]["nodes"][0]["type"]
    attributes = request["graph"]["nodes"][0]["attributes"]
    expected_workspace = 4096
    if operation == "sdpa_backward":
        raw = (
            4
            * attributes["batch"]
            * attributes["heads"]
            * attributes["sequence_q"]
        )
        expected_workspace = max(4096, ((raw + 255) // 256) * 256)
    expected_bindings = [
        tensor["uid"]
        for tensor in request["graph"]["tensors"]
        if tensor["virtual"] is False
    ]
    if set(manifest) != ROOT_KEYS:
        fail("Attention manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("Attention manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/attention.py"
    ):
        fail("Attention manifest source descriptor is invalid")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("Attention materialized source integrity differs")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != len(expected_functions)
        or len(program["stages"]) != len(expected_functions)
    ):
        fail("Attention execution program schema differs")
    arities = {
        "_sdpa_fwd_kernel": (48, 37),
        "_zero_contiguous_kernel": (3, 2),
        "_sdpa_bwd_dq_dbias_kernel": (71, 15),
        "_sdpa_bwd_dkdv_kernel": (59, 14),
        "_sdpa_bwd_dk_kernel": (56, 13),
        "_sdpa_bwd_dv_kernel": (46, 11),
        "_zero_sdpa_fp8_fwd_amax_kernel": (2, 2),
        "_sdpa_fp8_fwd_kernel": (55, 45),
        "_zero_sdpa_fp8_bwd_amax_kernel": (4, 4),
        "_sdpa_fp8_bwd_dq_kernel": (58, 22),
        "_sdpa_fp8_bwd_dkdv_kernel": (67, 27),
    }
    for stage_index, (stage, expected_function) in enumerate(
        zip(program["stages"], expected_functions, strict=True)
    ):
        fixed = expected_function.startswith("_zero_")
        expected_dependencies = [] if stage_index == 0 else [stage_index - 1]
        expected_candidates = (
            [(1, 4, 1)]
            if fixed
            else (
                [
                    (block, warps, stages)
                    for block in (16, 32)
                    for warps in (2, 4)
                    for stages in (1, 2)
                ]
                if autotune
                else [(32, 4, 2)]
            )
        )
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != request["graph"]["nodes"][0]["id"]
            or stage["operation"] != operation
            or stage["dependencies"] != expected_dependencies
            or stage["source"] != "kernels/attention.py"
            or stage["function"] != expected_function
            or stage["autotune"]
            != {
                "enabled": autotune and not fixed,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != len(expected_candidates)
        ):
            fail(f"Attention stage {stage_index} metadata differs")
        for variant, candidate in zip(
            stage["variants"], expected_candidates, strict=True
        ):
            block, warps, stages = candidate
            expected_variant_id = (
                "fixed-warps-4-stages-1"
                if fixed
                else f"block-{block}-warps-{warps}-stages-{stages}"
            )
            signature = variant["full_signature"].split(",")
            runtime_tokens = [
                token
                for token in signature
                if token.startswith("*") or token in {"i32", "fp32"}
            ]
            arguments = variant["arguments"]
            if (
                set(variant) != VARIANT_KEYS
                or variant["variant_id"] != expected_variant_id
                or variant["source"] != stage["source"]
                or variant["source_sha256"] != manifest["source_sha256"]
                or variant["function"] != expected_function
                or len(signature) != arities[expected_function][0]
                or len(runtime_tokens) != arities[expected_function][1]
                or len(arguments) != len(runtime_tokens)
                or variant["num_warps"] != warps
                or variant["num_stages"] != stages
                or not isinstance(variant["grid"], list)
                or len(variant["grid"]) != 3
                or any(value <= 0 for value in variant["grid"])
            ):
                fail(f"Attention {expected_function} variant differs")
            for argument, token in zip(arguments, runtime_tokens, strict=True):
                if set(argument) != ARGUMENT_KEYS:
                    fail("Attention argument schema is not closed")
                if token.startswith("*"):
                    if argument["kind"] not in {"tensor", "workspace"}:
                        fail("Attention pointer argument kind differs")
                elif argument["kind"] != (
                    "scalar_i32" if token == "i32" else "scalar_f32"
                ):
                    fail("Attention scalar argument kind differs")


def validate_im2col_fprop_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[input_uids["input"]]
    weight = tensor_by_uid[input_uids["filter"]]
    result = tensor_by_uid[output_uids["output"]]
    batch, cin, input_height, input_width = image["dimensions"]
    cout = result["dimensions"][1]
    output_height, output_width = result["dimensions"][2:]
    filter_height, filter_width = weight["dimensions"][2:]
    output_area = output_height * output_width
    reduction_extent = cin * filter_height * filter_width
    column_strides = (
        reduction_extent * output_area,
        output_area,
        1,
    )
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[image["data_type"]]
    raw_workspace = batch * reduction_extent * output_area * element_size
    expected_workspace = max(4096, (raw_workspace + 255) // 256 * 256)
    expected_bindings = [tensor["uid"] for tensor in graph["tensors"]]
    if set(manifest) != ROOT_KEYS:
        fail("im2col Fprop manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("im2col Fprop manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/convolution.py"
    ):
        fail("im2col Fprop source descriptor differs")
    source = output / files[0]["path"]
    source_bytes = source.read_bytes() if source.is_file() else b""
    if (
        not source.is_file()
        or len(source_bytes) != files[0]["size"]
        or hashlib.sha256(source_bytes).hexdigest() != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("im2col Fprop source integrity differs")
    source_module = ast.parse(source_bytes, filename=str(source))
    source_functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if {
        "_conv_fprop2d_im2col_kernel",
        "_conv_fprop2d_im2col_mm_kernel",
    }.difference(source_functions):
        fail("im2col Fprop artifact is missing stage entry points")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 2
        or len(program["stages"]) != 2
    ):
        fail("im2col Fprop execution program schema differs")
    pointer_type = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }
    pointer = lambda tensor: (  # noqa: E731
        pointer_type[tensor["data_type"]] + ":16"
        if tensor["alignment"] >= 16
        else pointer_type[tensor["data_type"]]
    )
    workspace_pointer = pointer_type[image["data_type"]] + ":16"
    stage_specs = (
        (
            "_conv_fprop2d_im2col_kernel",
            [],
            [
                ("tensor", "input", image["uid"]),
                ("workspace", "fprop_columns", None),
            ],
            (64, 4, 1),
        ),
        (
            "_conv_fprop2d_im2col_mm_kernel",
            [0],
            [
                ("tensor", "filter", weight["uid"]),
                ("workspace", "fprop_columns", None),
                ("tensor", "output", result["uid"]),
            ],
            (64, 8, 1),
        ),
    )
    large_reduction = batch == 1 and reduction_extent >= 1024
    mm_block_m = 32 if large_reduction else 64
    mm_block_k = (
        64 if large_reduction or image["data_type"] == "float32" else 32
    )
    for stage_index, (stage, specification) in enumerate(
        zip(program["stages"], stage_specs, strict=True)
    ):
        function, dependencies, expected_arguments, candidate = specification
        block, warps, stages = candidate
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != node["id"]
            or stage["operation"] != node["type"]
            or stage["dependencies"] != dependencies
            or stage["source"] != "kernels/convolution.py"
            or stage["function"] != function
            or stage["autotune"]
            != {
                "enabled": False,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != 1
        ):
            fail(f"im2col Fprop stage {stage_index} metadata differs")
        variant = stage["variants"][0]
        if stage_index == 0:
            expected_signature = ",".join(
                [
                    pointer(image),
                    workspace_pointer,
                    str(output_area),
                    str(input_height),
                    str(input_width),
                    str(output_height),
                    str(output_width),
                    str(cin),
                    str(filter_height),
                    str(filter_width),
                    *(str(value) for value in node["attributes"]["stride"]),
                    *(
                        str(value)
                        for value in node["attributes"]["pre_padding"]
                    ),
                    *(str(value) for value in node["attributes"]["dilation"]),
                    *(str(value) for value in image["strides"]),
                    *(str(value) for value in column_strides),
                    str(block),
                    "32",
                ]
            )
            expected_grid = [
                math.ceil(output_area / block)
                * math.ceil(reduction_extent / 32),
                batch,
                1,
            ]
        else:
            expected_signature = ",".join(
                [
                    pointer(weight),
                    workspace_pointer,
                    pointer(result),
                    str(output_area),
                    str(cout),
                    str(cin),
                    str(filter_height),
                    str(filter_width),
                    *(str(value) for value in weight["strides"]),
                    *(str(value) for value in result["strides"]),
                    str(output_width),
                    *(str(value) for value in column_strides),
                    "1" if image["data_type"] == "float32" else "0",
                    str(block),
                    str(mm_block_m),
                    str(mm_block_k),
                    "8",
                ]
            )
            expected_grid = [
                math.ceil(cout / block) * math.ceil(output_area / mm_block_m),
                batch,
                1,
            ]
        arguments = variant["arguments"]
        if (
            not isinstance(arguments, list)
            or any(set(item) != ARGUMENT_KEYS for item in arguments)
            or any(item["scalar_bits"] is not None for item in arguments)
        ):
            fail(f"im2col Fprop stage {stage_index} arguments differ")
        actual_arguments = [
            (item["kind"], item["semantic_name"], item["uid"])
            for item in arguments
        ]
        if (
            set(variant) != VARIANT_KEYS
            or variant["variant_id"]
            != f"block-{block}-warps-{warps}-stages-{stages}"
            or variant["source"] != stage["source"]
            or variant["source_sha256"] != manifest["source_sha256"]
            or variant["function"] != function
            or variant["full_signature"] != expected_signature
            or variant["grid"] != expected_grid
            or variant["num_warps"] != warps
            or variant["num_stages"] != stages
            or actual_arguments != expected_arguments
        ):
            fail(f"im2col Fprop stage {stage_index} variant differs")


def validate_dense_dgrad_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[output_uids["dx"]]
    loss = tensor_by_uid[input_uids["dy"]]
    weight = tensor_by_uid[input_uids["w"]]
    cin = image["dimensions"][1]
    cout = loss["dimensions"][1]
    oh, ow = loss["dimensions"][2:]
    xh, xw = image["dimensions"][2:]
    loss_rows = image["dimensions"][0] * oh * ow
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[image["data_type"]]
    packed_filter_bytes = 16 * cin * cout * element_size
    aligned_filter_bytes = (packed_filter_bytes + 255) // 256 * 256
    loss_offset = aligned_filter_bytes // element_size
    raw_workspace = aligned_filter_bytes + 4 * cout * loss_rows * element_size
    expected_workspace = max(4096, (raw_workspace + 255) // 256 * 256)
    expected_bindings = [tensor["uid"] for tensor in graph["tensors"]]
    if set(manifest) != ROOT_KEYS:
        fail("dense Dgrad manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("dense Dgrad manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/convolution.py"
    ):
        fail("dense Dgrad source descriptor differs")
    source = output / files[0]["path"]
    source_bytes = source.read_bytes() if source.is_file() else b""
    if (
        not source.is_file()
        or len(source_bytes) != files[0]["size"]
        or hashlib.sha256(source_bytes).hexdigest() != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("dense Dgrad source integrity differs")
    try:
        source_module = ast.parse(source_bytes, filename=str(source))
    except (SyntaxError, ValueError) as error:
        fail(f"dense Dgrad source is invalid: {error}")
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_functions = {
        "conv_dgrad_nd_kernel",
        "_conv_dgrad_dot",
        "_conv_dgrad2d_dense_pack_filter_kernel",
        "_conv_dgrad2d_dense_pack_loss_kernel",
        "_conv_dgrad2d_dense_mm_kernel",
    }
    if (
        not required_functions.issubset(functions)
        or "conv_wgrad_nd_kernel" in functions
    ):
        fail("dense Dgrad artifact exposes unexpected kernel functions")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 3
        or len(program["stages"]) != 3
    ):
        fail("dense Dgrad execution program schema differs")
    pointer_type = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }
    pointer = lambda tensor: (  # noqa: E731
        pointer_type[tensor["data_type"]] + ":16"
        if tensor["alignment"] >= 16
        else pointer_type[tensor["data_type"]]
    )
    workspace_pointer = pointer_type[image["data_type"]] + ":16"
    stage_specs = (
        (
            "_conv_dgrad2d_dense_pack_filter_kernel",
            [],
            [
                ("tensor", "w", weight["uid"]),
                ("workspace", "dgrad_dense_filter", None),
            ],
            [(32, 8, 1)],
        ),
        (
            "_conv_dgrad2d_dense_pack_loss_kernel",
            [],
            [
                ("tensor", "dy", loss["uid"]),
                ("workspace", "dgrad_dense_loss", None),
            ],
            [(32, 8, 1)],
        ),
        (
            "_conv_dgrad2d_dense_mm_kernel",
            [0, 1],
            [
                ("workspace", "dgrad_dense_filter", None),
                ("workspace", "dgrad_dense_loss", None),
                ("tensor", "dx", image["uid"]),
            ],
            (
                [(64, warps, stages) for warps in (4, 8) for stages in (1, 2)]
                if autotune
                else [(64, 8, 1)]
            ),
        ),
    )
    for stage_index, (stage, specification) in enumerate(
        zip(program["stages"], stage_specs, strict=True)
    ):
        function, dependencies, expected_arguments, candidates = specification
        stage_autotune = autotune if stage_index == 2 else False
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != node["id"]
            or stage["operation"] != "convolution_dgrad"
            or stage["dependencies"] != dependencies
            or stage["source"] != "kernels/convolution.py"
            or stage["function"] != function
            or stage["autotune"]
            != {
                "enabled": stage_autotune,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != len(candidates)
        ):
            fail(f"dense Dgrad stage {stage_index} metadata differs")
        for variant, (block, warps, stages) in zip(
            stage["variants"], candidates, strict=True
        ):
            if stage_index == 0:
                expected_signature = ",".join(
                    [
                        pointer(weight),
                        workspace_pointer,
                        str(cin),
                        str(cout),
                        *(str(value) for value in weight["strides"]),
                        str(block),
                        str(block),
                    ]
                )
                expected_grid = [
                    math.ceil(4 * cin / block),
                    math.ceil(4 * cout / block),
                    1,
                ]
            elif stage_index == 1:
                expected_signature = ",".join(
                    [
                        pointer(loss),
                        workspace_pointer,
                        str(loss_offset),
                        str(loss_rows),
                        str(oh),
                        str(ow),
                        str(cout),
                        *(str(value) for value in loss["strides"]),
                        str(block),
                        str(block),
                    ]
                )
                expected_grid = [
                    math.ceil(4 * cout / block),
                    math.ceil(loss_rows / block),
                    1,
                ]
            else:
                expected_signature = ",".join(
                    [
                        workspace_pointer,
                        workspace_pointer,
                        pointer(image),
                        str(loss_offset),
                        str(loss_rows),
                        str(oh),
                        str(ow),
                        str(xh),
                        str(xw),
                        str(cin),
                        str(cout),
                        *(str(value) for value in image["strides"]),
                        "1" if image["data_type"] == "float32" else "0",
                        str(block),
                        str(block),
                        str(block),
                        "8",
                    ]
                )
                expected_grid = [
                    math.ceil(4 * cin / block) * math.ceil(loss_rows / block),
                    1,
                    1,
                ]
            arguments = variant["arguments"]
            if (
                not isinstance(arguments, list)
                or any(set(item) != ARGUMENT_KEYS for item in arguments)
                or any(item["scalar_bits"] is not None for item in arguments)
            ):
                fail(f"dense Dgrad stage {stage_index} arguments differ")
            actual_arguments = [
                (item["kind"], item["semantic_name"], item["uid"])
                for item in arguments
            ]
            if (
                set(variant) != VARIANT_KEYS
                or variant["variant_id"]
                != f"block-{block}-warps-{warps}-stages-{stages}"
                or variant["source"] != stage["source"]
                or variant["source_sha256"] != manifest["source_sha256"]
                or variant["function"] != function
                or variant["full_signature"] != expected_signature
                or variant["grid"] != expected_grid
                or variant["num_warps"] != warps
                or variant["num_stages"] != stages
                or actual_arguments != expected_arguments
            ):
                fail(f"dense Dgrad stage {stage_index} variant differs")


def validate_p5_wgrad_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[input_uids["x"]]
    loss = tensor_by_uid[input_uids["dy"]]
    weight = tensor_by_uid[output_uids["dw"]]
    cin = image["dimensions"][1]
    cout = loss["dimensions"][1]
    cik = cin * 9
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[image["data_type"]]
    raw_workspace = 400 * cik * element_size
    expected_workspace = max(4096, (raw_workspace + 255) // 256 * 256)
    expected_bindings = [tensor["uid"] for tensor in graph["tensors"]]
    if set(manifest) != ROOT_KEYS:
        fail("P5 Wgrad manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("P5 Wgrad manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/convolution.py"
    ):
        fail("P5 Wgrad source descriptor differs")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("P5 Wgrad source integrity differs")
    source_module = ast.parse(source.read_bytes(), filename=str(source))
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if functions != {
        "conv_wgrad_nd_kernel",
        "_conv_wgrad_nd_im2row_kernel",
        "_conv_wgrad_nd_rowmajor_kernel",
        "_conv_wgrad_nd_reduce_kernel",
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
        "_conv_wgrad2d_direct_split_kernel",
        "_conv_wgrad2d_im2row_kernel",
        "_conv_wgrad2d_rowmajor_kernel",
        "_conv_wgrad2d_1x1_split_kernel",
        "_conv_wgrad2d_stem_split_kernel",
        "_conv_wgrad2d_stem_reduce_kernel",
    }:
        fail("P5 Wgrad artifact exposes unexpected kernel functions")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 2
        or len(program["stages"]) != 2
    ):
        fail("P5 Wgrad execution program schema differs")
    base_candidates = (
        [
            (block, warps, stages)
            for block in (16, 32)
            for warps in (4, 8)
            for stages in (1, 2)
        ]
        if autotune
        else [(32, 4, 2)]
    )
    stage_candidates = (
        base_candidates,
        (
            [
                *base_candidates,
                *(
                    (block, warps, stages)
                    for block in (64,)
                    for warps in (4, 8)
                    for stages in (1, 2)
                ),
            ]
            if autotune and image["data_type"] != "float32"
            else base_candidates
        ),
    )
    pointer_type = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }
    pointer = lambda tensor: (  # noqa: E731
        pointer_type[tensor["data_type"]] + ":16"
        if tensor["alignment"] >= 16
        else pointer_type[tensor["data_type"]]
    )
    workspace_pointer = pointer_type[image["data_type"]] + ":16"
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[image["data_type"]]
    stage_specs = (
        (
            "_conv_wgrad2d_p5_pack_image_kernel",
            [],
            [
                ("tensor", "x", image["uid"]),
                ("workspace", "packed_image", None),
            ],
        ),
        (
            "_conv_wgrad2d_p5_mm_kernel",
            [0],
            [
                ("tensor", "dy", loss["uid"]),
                ("workspace", "packed_image", None),
                ("tensor", "dw", weight["uid"]),
            ],
        ),
    )
    for stage_index, (stage, specification, candidates) in enumerate(
        zip(program["stages"], stage_specs, stage_candidates, strict=True)
    ):
        function, dependencies, expected_arguments = specification
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != node["id"]
            or stage["operation"] != "convolution_wgrad"
            or stage["dependencies"] != dependencies
            or stage["source"] != "kernels/convolution.py"
            or stage["function"] != function
            or stage["autotune"]
            != {
                "enabled": autotune,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != len(candidates)
        ):
            fail(f"P5 Wgrad stage {stage_index} metadata differs")
        for variant, (block, warps, stages) in zip(
            stage["variants"], candidates, strict=True
        ):
            if stage_index == 0:
                expected_signature = ",".join(
                    [
                        pointer(image),
                        workspace_pointer,
                        str(cin),
                        str(image["strides"][1]),
                        str(image["strides"][2]),
                        str(image["strides"][3]),
                        "400",
                        str(cik),
                        str(block),
                        str(block),
                        str(block),
                        "8",
                    ]
                )
                expected_grid = [
                    math.ceil(400 / block),
                    math.ceil(cik / block),
                    1,
                ]
            else:
                expected_signature = ",".join(
                    [
                        pointer(loss),
                        workspace_pointer,
                        pointer(weight),
                        str(cout),
                        str(cik),
                        "400",
                        str(dtype_id),
                        str(block),
                        str(block),
                        str(block),
                        "8",
                    ]
                )
                expected_grid = [
                    math.ceil(cout / block) * math.ceil(cik / block),
                    1,
                    1,
                ]
            arguments = variant["arguments"]
            actual_arguments = [
                (item["kind"], item["semantic_name"], item["uid"])
                for item in arguments
                if set(item) == ARGUMENT_KEYS and item["scalar_bits"] is None
            ]
            if (
                set(variant) != VARIANT_KEYS
                or variant["variant_id"]
                != f"block-{block}-warps-{warps}-stages-{stages}"
                or variant["source"] != stage["source"]
                or variant["source_sha256"] != manifest["source_sha256"]
                or variant["function"] != function
                or variant["full_signature"] != expected_signature
                or variant["grid"] != expected_grid
                or variant["num_warps"] != warps
                or variant["num_stages"] != stages
                or actual_arguments != expected_arguments
            ):
                fail(f"P5 Wgrad stage {stage_index} variant differs")


def validate_stem_wgrad_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[input_uids["x"]]
    loss = tensor_by_uid[input_uids["dy"]]
    weight = tensor_by_uid[output_uids["dw"]]
    cout = loss["dimensions"][1]
    cin = image["dimensions"][1]
    cik = cin * 9
    partial_stride_split = cout * cik
    raw_workspace = 64 * partial_stride_split * 4
    expected_workspace = max(4096, (raw_workspace + 255) // 256 * 256)
    expected_bindings = [tensor["uid"] for tensor in graph["tensors"]]
    if set(manifest) != ROOT_KEYS:
        fail("stem Wgrad manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("stem Wgrad manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/convolution.py"
    ):
        fail("stem Wgrad source descriptor differs")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("stem Wgrad source integrity differs")
    source_module = ast.parse(source.read_bytes(), filename=str(source))
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if functions != {
        "conv_wgrad_nd_kernel",
        "_conv_wgrad_nd_im2row_kernel",
        "_conv_wgrad_nd_rowmajor_kernel",
        "_conv_wgrad_nd_reduce_kernel",
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
        "_conv_wgrad2d_direct_split_kernel",
        "_conv_wgrad2d_im2row_kernel",
        "_conv_wgrad2d_rowmajor_kernel",
        "_conv_wgrad2d_1x1_split_kernel",
        "_conv_wgrad2d_stem_split_kernel",
        "_conv_wgrad2d_stem_reduce_kernel",
    }:
        fail("stem Wgrad artifact exposes unexpected kernel functions")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 2
        or len(program["stages"]) != 2
    ):
        fail("stem Wgrad execution program schema differs")
    split_candidates = (
        [
            (block, warps, stages)
            for block in (32, 64)
            for warps in (4, 8)
            for stages in (1, 2)
        ]
        if autotune
        else [(64, 4, 2)]
    )
    reduce_candidates = (
        [(block, warps, 1) for block in (128, 256) for warps in (4, 8)]
        if autotune
        else [(256, 4, 1)]
    )
    pointer_type = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }
    pointer = lambda tensor: (  # noqa: E731
        pointer_type[tensor["data_type"]] + ":16"
        if tensor["alignment"] >= 16
        else pointer_type[tensor["data_type"]]
    )
    partial_pointer = "*fp32:16"
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[image["data_type"]]
    stage_specs = (
        (
            "_conv_wgrad2d_stem_split_kernel",
            [],
            [
                ("tensor", "dy", loss["uid"]),
                ("tensor", "x", image["uid"]),
                ("workspace", "wgrad_partial", None),
            ],
            split_candidates,
        ),
        (
            "_conv_wgrad2d_stem_reduce_kernel",
            [0],
            [
                ("workspace", "wgrad_partial", None),
                ("tensor", "dw", weight["uid"]),
            ],
            reduce_candidates,
        ),
    )
    for stage_index, (stage, specification) in enumerate(
        zip(program["stages"], stage_specs, strict=True)
    ):
        function, dependencies, expected_arguments, candidates = specification
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != node["id"]
            or stage["operation"] != "convolution_wgrad"
            or stage["dependencies"] != dependencies
            or stage["source"] != "kernels/convolution.py"
            or stage["function"] != function
            or stage["autotune"]
            != {
                "enabled": autotune,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != len(candidates)
        ):
            fail(f"stem Wgrad stage {stage_index} metadata differs")
        for variant, (block, warps, stages) in zip(
            stage["variants"], candidates, strict=True
        ):
            if stage_index == 0:
                expected_signature = ",".join(
                    [
                        pointer(loss),
                        pointer(image),
                        partial_pointer,
                        "5",
                        "320",
                        "320",
                        "640",
                        "640",
                        str(cout),
                        str(cin),
                        "3",
                        "3",
                        "2",
                        "2",
                        "1",
                        "1",
                        "1",
                        "1",
                        *(str(value) for value in loss["strides"][1:]),
                        *(str(value) for value in image["strides"][1:]),
                        str(partial_stride_split),
                        str(cik),
                        "1",
                        str(dtype_id),
                        str(block),
                        "32",
                        "64",
                    ]
                )
                expected_grid = [math.ceil(cout / block), 64, 1]
            else:
                expected_signature = ",".join(
                    [
                        partial_pointer,
                        pointer(weight),
                        str(cout * cik),
                        str(cik),
                        str(cin),
                        "3",
                        "3",
                        "64",
                        str(partial_stride_split),
                        str(cik),
                        "1",
                        *(str(value) for value in weight["strides"]),
                        str(block),
                    ]
                )
                expected_grid = [math.ceil(cout * cik / block), 1, 1]
            arguments = variant["arguments"]
            actual_arguments = [
                (item["kind"], item["semantic_name"], item["uid"])
                for item in arguments
                if set(item) == ARGUMENT_KEYS and item["scalar_bits"] is None
            ]
            if (
                set(variant) != VARIANT_KEYS
                or variant["variant_id"]
                != f"block-{block}-warps-{warps}-stages-{stages}"
                or variant["source"] != stage["source"]
                or variant["source_sha256"] != manifest["source_sha256"]
                or variant["function"] != function
                or variant["full_signature"] != expected_signature
                or variant["grid"] != expected_grid
                or variant["num_warps"] != warps
                or variant["num_stages"] != stages
                or actual_arguments != expected_arguments
            ):
                fail(f"stem Wgrad stage {stage_index} variant differs")


def validate_nd_packed_wgrad_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[input_uids["x"]]
    loss = tensor_by_uid[input_uids["dy"]]
    weight = tensor_by_uid[output_uids["dw"]]
    spatial_rank = node["attributes"]["spatial_rank"]
    input_spatial = [1] * (3 - spatial_rank) + image["dimensions"][2:]
    output_spatial = [1] * (3 - spatial_rank) + loss["dimensions"][2:]
    kernel_spatial = [1] * (3 - spatial_rank) + weight["dimensions"][2:]
    stride = [1] * (3 - spatial_rank) + node["attributes"]["stride"]
    padding = [0] * (3 - spatial_rank) + node["attributes"]["pre_padding"]
    dilation = [1] * (3 - spatial_rank) + node["attributes"]["dilation"]
    input_strides = (
        image["strides"][:2] + [0] * (3 - spatial_rank) + image["strides"][2:]
    )
    batch = image["dimensions"][0]
    cin = image["dimensions"][1]
    cout = loss["dimensions"][1]
    output_area = math.prod(loss["dimensions"][2:])
    reduction_extent = cin * math.prod(weight["dimensions"][2:])
    total_rows = batch * output_area
    total_weights = cout * reduction_extent
    num_splits = 16 if total_rows >= 4096 else 8
    rows_per_split = (total_rows + num_splits - 1) // num_splits
    partial_raw = num_splits * total_weights * 4
    partial_aligned = (partial_raw + 255) // 256 * 256
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[image["data_type"]]
    column_offset = partial_aligned // element_size
    raw_workspace = (
        partial_aligned + total_rows * reduction_extent * element_size
    )
    expected_workspace = max(4096, (raw_workspace + 255) // 256 * 256)
    expected_bindings = [tensor["uid"] for tensor in graph["tensors"]]
    if set(manifest) != ROOT_KEYS:
        fail("ND packed Wgrad manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("ND packed Wgrad manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/convolution.py"
    ):
        fail("ND packed Wgrad source descriptor differs")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("ND packed Wgrad source integrity differs")
    source_module = ast.parse(source.read_bytes(), filename=str(source))
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if functions != {
        "conv_wgrad_nd_kernel",
        "_conv_wgrad_nd_im2row_kernel",
        "_conv_wgrad_nd_rowmajor_kernel",
        "_conv_wgrad_nd_reduce_kernel",
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
        "_conv_wgrad2d_direct_split_kernel",
        "_conv_wgrad2d_im2row_kernel",
        "_conv_wgrad2d_rowmajor_kernel",
        "_conv_wgrad2d_1x1_split_kernel",
        "_conv_wgrad2d_stem_split_kernel",
        "_conv_wgrad2d_stem_reduce_kernel",
    }:
        fail("ND packed Wgrad artifact exposes unexpected kernel functions")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 3
        or len(program["stages"]) != 3
    ):
        fail("ND packed Wgrad execution program schema differs")

    pointer_type = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }
    pointer = lambda tensor: (  # noqa: E731
        pointer_type[tensor["data_type"]] + ":16"
        if tensor["alignment"] >= 16
        else pointer_type[tensor["data_type"]]
    )
    packed_pointer = pointer_type[image["data_type"]] + ":16"
    partial_pointer = "*fp32:16"
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[image["data_type"]]
    pack_candidates = (
        [(block, warps, 1) for block in (32, 64) for warps in (4, 8)]
        if autotune
        else [(64, 4, 1)]
    )
    gemm_candidates = (
        [
            (block, warps, stages)
            for block in (32, 64)
            for warps in (4, 8)
            for stages in (1, 2)
        ]
        if autotune
        else [(64, 8, 1)]
    )
    reduce_candidates = (
        [(block, warps, 1) for block in (128, 256) for warps in (4, 8)]
        if autotune
        else [(256, 4, 1)]
    )
    stage_specs = (
        (
            "_conv_wgrad_nd_im2row_kernel",
            [],
            [
                ("tensor", "x", image["uid"]),
                ("workspace", "wgrad_nd_columns", None),
            ],
            pack_candidates,
        ),
        (
            "_conv_wgrad_nd_rowmajor_kernel",
            [0],
            [
                ("tensor", "dy", loss["uid"]),
                ("workspace", "wgrad_nd_columns", None),
                ("workspace", "wgrad_nd_partial", None),
            ],
            gemm_candidates,
        ),
        (
            "_conv_wgrad_nd_reduce_kernel",
            [1],
            [
                ("workspace", "wgrad_nd_partial", None),
                ("tensor", "dw", weight["uid"]),
            ],
            reduce_candidates,
        ),
    )
    for stage_index, (stage, specification) in enumerate(
        zip(program["stages"], stage_specs, strict=True)
    ):
        function, dependencies, expected_arguments, candidates = specification
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != node["id"]
            or stage["operation"] != "convolution_wgrad"
            or stage["dependencies"] != dependencies
            or stage["source"] != "kernels/convolution.py"
            or stage["function"] != function
            or stage["autotune"]
            != {
                "enabled": autotune,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != len(candidates)
        ):
            fail(f"ND packed Wgrad stage {stage_index} metadata differs")
        for variant, (block, warps, stages) in zip(
            stage["variants"], candidates, strict=True
        ):
            if function == "_conv_wgrad_nd_im2row_kernel":
                signature_tokens = [
                    pointer(image),
                    packed_pointer,
                    str(output_area),
                    *(str(value) for value in input_spatial),
                    str(output_spatial[1]),
                    str(output_spatial[2]),
                    str(cin),
                    *(str(value) for value in kernel_spatial),
                    *(str(value) for value in stride),
                    *(str(value) for value in padding),
                    *(str(value) for value in dilation),
                    *(str(value) for value in input_strides),
                    str(reduction_extent),
                    str(column_offset),
                    "1",
                    str(block),
                    str(block),
                ]
                expected_grid = [
                    math.ceil(output_area / block)
                    * math.ceil(reduction_extent / block),
                    batch,
                    1,
                ]
            elif function == "_conv_wgrad_nd_rowmajor_kernel":
                signature_tokens = [
                    pointer(loss),
                    packed_pointer,
                    partial_pointer,
                    str(total_rows),
                    str(rows_per_split),
                    str(output_area),
                    str(cout),
                    str(reduction_extent),
                    str(loss["strides"][0]),
                    str(loss["strides"][1]),
                    str(loss["strides"][-1]),
                    str(reduction_extent),
                    str(column_offset),
                    "1",
                    str(total_weights),
                    str(reduction_extent),
                    "1",
                    str(dtype_id),
                    str(block),
                    str(block),
                    str(block),
                ]
                expected_grid = [
                    math.ceil(cout / block)
                    * math.ceil(reduction_extent / block),
                    num_splits,
                    1,
                ]
            else:
                if function != "_conv_wgrad_nd_reduce_kernel":
                    fail("ND packed Wgrad selected an unknown stage function")
                signature_tokens = [
                    partial_pointer,
                    pointer(weight),
                    str(total_weights),
                    str(num_splits),
                    str(total_weights),
                    str(block),
                ]
                expected_grid = [math.ceil(total_weights / block), 1, 1]
            expected_signature = ",".join(signature_tokens)
            arguments = variant["arguments"]
            actual_arguments = [
                (item["kind"], item["semantic_name"], item["uid"])
                for item in arguments
                if set(item) == ARGUMENT_KEYS and item["scalar_bits"] is None
            ]
            if (
                set(variant) != VARIANT_KEYS
                or variant["variant_id"]
                != f"block-{block}-warps-{warps}-stages-{stages}"
                or variant["source"] != stage["source"]
                or variant["source_sha256"] != manifest["source_sha256"]
                or variant["function"] != function
                or variant["full_signature"] != expected_signature
                or variant["grid"] != expected_grid
                or variant["num_warps"] != warps
                or variant["num_stages"] != stages
                or actual_arguments != expected_arguments
            ):
                fail(f"ND packed Wgrad stage {stage_index} variant differs")


def validate_standard_wgrad_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    graph = request["graph"]
    node = graph["nodes"][0]
    tensor_by_uid = {tensor["uid"]: tensor for tensor in graph["tensors"]}
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    image = tensor_by_uid[input_uids["x"]]
    loss = tensor_by_uid[input_uids["dy"]]
    weight = tensor_by_uid[output_uids["dw"]]
    _, _, xh, xw = image["dimensions"]
    _, _, oh, ow = loss["dimensions"]
    cout, cin, kh, kw = weight["dimensions"]
    cik = cin * kh * kw
    total = cout * cik
    is_1x1 = [kh, kw] == [1, 1]
    num_splits = 8 if is_1x1 else image["dimensions"][0]
    total_rows = image["dimensions"][0] * oh * ow
    rows_per_split = (total_rows + num_splits - 1) // num_splits
    partial_stride_split = total
    partial_raw = num_splits * total * 4
    partial_aligned = (partial_raw + 255) // 256 * 256
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[image["data_type"]]
    column_offset = partial_aligned // element_size
    if is_1x1:
        raw_workspace = partial_aligned
    else:
        raw_workspace = partial_aligned + total_rows * cik * element_size
    expected_workspace = max(4096, (raw_workspace + 255) // 256 * 256)
    expected_bindings = [tensor["uid"] for tensor in graph["tensors"]]
    if set(manifest) != ROOT_KEYS:
        fail("standard Wgrad manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["flagdnn_version"] != request["flagdnn_version"]
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["compiler_identity"] != request["compiler_identity"]
        or manifest["workspace_size"] != expected_workspace
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("standard Wgrad manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != "kernels/convolution.py"
    ):
        fail("standard Wgrad source descriptor differs")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("standard Wgrad source integrity differs")
    source_module = ast.parse(source.read_bytes(), filename=str(source))
    functions = {
        item.name
        for item in source_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if functions != {
        "conv_wgrad_nd_kernel",
        "_conv_wgrad_nd_im2row_kernel",
        "_conv_wgrad_nd_rowmajor_kernel",
        "_conv_wgrad_nd_reduce_kernel",
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
        "_conv_wgrad2d_direct_split_kernel",
        "_conv_wgrad2d_im2row_kernel",
        "_conv_wgrad2d_rowmajor_kernel",
        "_conv_wgrad2d_1x1_split_kernel",
        "_conv_wgrad2d_stem_split_kernel",
        "_conv_wgrad2d_stem_reduce_kernel",
    }:
        fail("standard Wgrad artifact exposes unexpected kernel functions")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != (2 if is_1x1 else 3)
        or len(program["stages"]) != (2 if is_1x1 else 3)
    ):
        fail("standard Wgrad execution program schema differs")
    pointer_type = {
        "float32": "*fp32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
    }
    pointer = lambda tensor: (  # noqa: E731
        pointer_type[tensor["data_type"]] + ":16"
        if tensor["alignment"] >= 16
        else pointer_type[tensor["data_type"]]
    )
    partial_pointer = "*fp32:16"
    packed_pointer = pointer_type[image["data_type"]] + ":16"
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[image["data_type"]]
    if is_1x1:
        split_candidates = (
            [
                (block, warps, stages)
                for block in (16, 32)
                for warps in (4, 8)
                for stages in (1, 2)
            ]
            if autotune
            else [(16, 4, 2)]
        )
        reduce_candidates = (
            [(block, warps, 1) for block in (128, 256) for warps in (4, 8)]
            if autotune
            else [(256, 4, 1)]
        )
        stage_specs = (
            (
                "_conv_wgrad2d_1x1_split_kernel",
                [],
                [
                    ("tensor", "dy", loss["uid"]),
                    ("tensor", "x", image["uid"]),
                    ("workspace", "wgrad_partial", None),
                ],
                split_candidates,
            ),
            (
                "_conv_wgrad2d_stem_reduce_kernel",
                [0],
                [
                    ("workspace", "wgrad_partial", None),
                    ("tensor", "dw", weight["uid"]),
                ],
                reduce_candidates,
            ),
        )
    else:
        pack_candidates = (
            [(block, warps, 1) for block in (32, 64) for warps in (4, 8)]
            if autotune
            else [(64, 4, 1)]
        )
        gemm_candidates = (
            [
                (block, warps, stages)
                for block in (32, 64)
                for warps in (4, 8)
                for stages in (1, 2)
            ]
            if autotune
            else [(64, 8, 1)]
        )
        reduce_candidates = (
            [(block, warps, 1) for block in (128, 256) for warps in (4, 8)]
            if autotune
            else [(256, 4, 1)]
        )
        stage_specs = (
            (
                "_conv_wgrad2d_im2row_kernel",
                [],
                [
                    ("tensor", "x", image["uid"]),
                    ("workspace", "wgrad_columns", None),
                ],
                pack_candidates,
            ),
            (
                "_conv_wgrad2d_rowmajor_kernel",
                [0],
                [
                    ("tensor", "dy", loss["uid"]),
                    ("workspace", "wgrad_columns", None),
                    ("workspace", "wgrad_partial", None),
                ],
                gemm_candidates,
            ),
            (
                "_conv_wgrad2d_stem_reduce_kernel",
                [1],
                [
                    ("workspace", "wgrad_partial", None),
                    ("tensor", "dw", weight["uid"]),
                ],
                reduce_candidates,
            ),
        )
    for stage_index, (stage, specification) in enumerate(
        zip(program["stages"], stage_specs, strict=True)
    ):
        function, dependencies, expected_arguments, candidates = specification
        if (
            set(stage) != STAGE_KEYS
            or stage["id"] != stage_index
            or stage["node_id"] != node["id"]
            or stage["operation"] != "convolution_wgrad"
            or stage["dependencies"] != dependencies
            or stage["source"] != "kernels/convolution.py"
            or stage["function"] != function
            or stage["autotune"]
            != {
                "enabled": autotune,
                "warmup": 3,
                "repetitions": 10,
                "selection_cache": f"tuning/stage-{stage_index}.json",
            }
            or len(stage["variants"]) != len(candidates)
        ):
            fail(f"standard Wgrad stage {stage_index} metadata differs")
        for variant, (block, warps, stages) in zip(
            stage["variants"], candidates, strict=True
        ):
            if function == "_conv_wgrad2d_im2row_kernel":
                signature_tokens = [
                    pointer(image),
                    packed_pointer,
                    str(oh * ow),
                    str(xh),
                    str(xw),
                    str(ow),
                    str(cin),
                    str(kh),
                    str(kw),
                    *(str(value) for value in node["attributes"]["stride"]),
                    *(
                        str(value)
                        for value in node["attributes"]["pre_padding"]
                    ),
                    *(str(value) for value in node["attributes"]["dilation"]),
                    *(str(value) for value in image["strides"]),
                    str(cik),
                    str(column_offset),
                    "1",
                    str(block),
                    str(block),
                ]
                expected_grid = [
                    math.ceil((oh * ow) / block) * math.ceil(cik / block),
                    image["dimensions"][0],
                    1,
                ]
            elif function == "_conv_wgrad2d_rowmajor_kernel":
                signature_tokens = [
                    pointer(loss),
                    packed_pointer,
                    partial_pointer,
                    str(oh * ow),
                    str(cout),
                    str(cin),
                    str(kh),
                    str(kw),
                    *(str(value) for value in loss["strides"]),
                    str(column_offset),
                    str(cik),
                    "1",
                    str(total),
                    str(cik),
                    "1",
                    str(dtype_id),
                    str(block),
                    str(block),
                    str(block),
                ]
                expected_grid = [
                    math.ceil(cout / block) * math.ceil(cik / block),
                    image["dimensions"][0],
                    1,
                ]
            elif function == "_conv_wgrad2d_1x1_split_kernel":
                signature_tokens = [
                    pointer(loss),
                    pointer(image),
                    partial_pointer,
                    str(total_rows),
                    str(rows_per_split),
                    str(oh * ow),
                    str(image["dimensions"][1]),
                    str(loss["dimensions"][1]),
                    str(cin),
                    str(cout),
                    "1",
                    str(partial_stride_split),
                    str(cik),
                    "1",
                    str(dtype_id),
                    str(block),
                    str(block),
                    "64",
                ]
                expected_grid = [
                    math.ceil(cout / block) * math.ceil(cin / block),
                    num_splits,
                    1,
                ]
            else:
                if function != "_conv_wgrad2d_stem_reduce_kernel":
                    fail("standard Wgrad selected an unknown stage function")
                signature_tokens = [
                    partial_pointer,
                    pointer(weight),
                    str(total),
                    str(cik),
                    str(cin),
                    str(kh),
                    str(kw),
                    str(num_splits),
                    str(partial_stride_split),
                    str(cik),
                    "1",
                    *(str(value) for value in weight["strides"]),
                    str(block),
                ]
                expected_grid = [math.ceil(total / block), 1, 1]
            expected_signature = ",".join(signature_tokens)
            arguments = variant["arguments"]
            actual_arguments = [
                (item["kind"], item["semantic_name"], item["uid"])
                for item in arguments
                if set(item) == ARGUMENT_KEYS and item["scalar_bits"] is None
            ]
            if (
                set(variant) != VARIANT_KEYS
                or variant["variant_id"]
                != f"block-{block}-warps-{warps}-stages-{stages}"
                or variant["source"] != stage["source"]
                or variant["source_sha256"] != manifest["source_sha256"]
                or variant["function"] != function
                or variant["full_signature"] != expected_signature
                or variant["grid"] != expected_grid
                or variant["num_warps"] != warps
                or variant["num_stages"] != stages
                or actual_arguments != expected_arguments
            ):
                fail(f"standard Wgrad stage {stage_index} variant differs")


def validate_manifest(
    manifest: dict[str, Any],
    *,
    request_bytes: bytes,
    output: Path,
    autotune: bool,
    request: dict[str, Any],
) -> None:
    operation = request["graph"]["nodes"][0]["type"]
    if uses_im2col_fprop(request):
        validate_im2col_fprop_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if uses_dense_stride2_dgrad(request):
        validate_dense_dgrad_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if uses_nd_packed_wgrad(request):
        validate_nd_packed_wgrad_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if uses_standard_wgrad(request):
        validate_standard_wgrad_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if uses_stem_wgrad(request):
        validate_stem_wgrad_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if uses_p5_wgrad(request):
        validate_p5_wgrad_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if operation.startswith("sdpa"):
        validate_attention_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if operation in {
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "batchnorm_inference",
    }:
        validate_normalization_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if request["graph"]["node_count"] == 3:
        validate_conv_bias_relu_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    if request["graph"]["node_count"] == 2:
        validate_add_square_manifest(
            manifest,
            request_bytes=request_bytes,
            output=output,
            autotune=autotune,
            request=request,
        )
        return
    node = request["graph"]["nodes"][0]
    operation = node["type"]
    pointwise_mode = node["attributes"].get("mode")
    unary = operation in UNARY_OPERATIONS
    ternary = operation == "binary_select"
    layout = operation in {"reshape", "transpose", "slice"}
    reduction = operation in {
        "reduction_sum",
        "reduction_avg",
        "reduction_mul",
    }
    matmul = operation == "matmul"
    matmul_tle = matmul_tle_config(request) if matmul else None
    matmul_descriptor = matmul and uses_matmul_descriptor(request)
    convolution = operation in {
        "conv2d_fprop",
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    }
    expected_bindings = [
        tensor["uid"] for tensor in request["graph"]["tensors"]
    ]
    expected_source = (
        "kernels/unary.py"
        if unary
        else (
            "kernels/ternary.py"
            if ternary
            else (
                "kernels/layout.py"
                if layout
                else (
                    "kernels/reduction.py"
                    if reduction
                    else (
                        "kernels/matmul.py"
                        if matmul
                        else (
                            "kernels/convolution.py"
                            if convolution
                            else "kernels/binary.py"
                        )
                    )
                )
            )
        )
    )
    expected_functions = (
        {
            "unary_pointwise_contiguous_kernel",
            "unary_pointwise_strided_kernel",
        }
        if unary
        else (
            {
                "binary_select_tensor_kernel",
                "binary_select_strided_kernel",
            }
            if ternary
            else (
                {"reshape_contiguous_kernel", "layout_copy_kernel"}
                if operation == "reshape"
                else (
                    {"slice_copy_kernel", "layout_copy_kernel"}
                    if operation == "slice"
                    else (
                        {
                            "transpose_physical_copy_kernel",
                            "layout_copy_kernel",
                        }
                        if operation == "transpose"
                        else (
                            {"layout_copy_kernel"}
                            if layout
                            else (
                                {
                                    "reduction_2d_kernel",
                                    "reduction_3d_kernel",
                                    "reduction_strided_kernel",
                                }
                                if reduction
                                else (
                                    (
                                        {"matmul_tle_kernel"}
                                        if matmul_tle is not None
                                        else (
                                            {"matmul_descriptor_kernel"}
                                            if matmul_descriptor
                                            else {"matmul_strided_kernel"}
                                        )
                                    )
                                    if matmul
                                    else (
                                        {"conv_dgrad_nd_kernel"}
                                        if operation == "convolution_dgrad"
                                        else (
                                            {"conv_wgrad_nd_kernel"}
                                            if operation == "convolution_wgrad"
                                            else (
                                                {
                                                    "conv1d_gemm_kernel",
                                                    "conv2d_spatial_nchw_kernel",
                                                    "conv3d_spatial_ncdhw_m_kernel",
                                                    "conv_dgrad_nd_kernel",
                                                    "conv_wgrad_nd_kernel",
                                                }
                                                if convolution
                                                else {
                                                    "binary_contiguous_kernel",
                                                    "binary_strided_kernel",
                                                }
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    if set(manifest) != ROOT_KEYS:
        fail("mthreads manifest root schema is not closed")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "flagdnn_execution_program"
        or manifest["backend"] != "mthreads"
        or manifest["target"] != TARGET
        or manifest["engine"] != "libtriton_jit"
        or manifest["request_sha256"]
        != hashlib.sha256(request_bytes).hexdigest()
        or manifest["workspace_size"] != 4096
        or manifest["workspace_alignment"] != 256
        or manifest["external_binding_uids"] != expected_bindings
    ):
        fail("mthreads manifest root semantics differ")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != 1
        or set(files[0]) != {"path", "size", "sha256"}
        or files[0]["path"] != expected_source
    ):
        fail("mthreads manifest files table is invalid")
    source = output / files[0]["path"]
    if (
        not source.is_file()
        or source.stat().st_size != files[0]["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest()
        != files[0]["sha256"]
        or files[0]["sha256"] != manifest["source_sha256"]
    ):
        fail("materialized pointwise source integrity differs")
    if convolution:
        source_bytes = source.read_bytes()
        try:
            source_module = ast.parse(source_bytes, filename=str(source))
        except (SyntaxError, ValueError) as error:
            fail(f"materialized convolution source is invalid: {error}")
        materialized_functions = {
            item.name
            for item in source_module.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not expected_functions.issubset(materialized_functions):
            fail("materialized convolution source is missing an entry point")
        if "conv1d_depthwise_kernel" in materialized_functions or len(
            source_bytes
        ) >= (128 << 10):
            fail("materialized convolution source was not candidate-sliced")
    program = manifest["program"]
    if (
        set(program) != {"schema_version", "stage_count", "stages"}
        or program["schema_version"] != 1
        or program["stage_count"] != 1
        or len(program["stages"]) != 1
    ):
        fail("mthreads execution program schema differs")
    stage = program["stages"][0]
    if (
        set(stage) != STAGE_KEYS
        or stage["id"] != 0
        or stage["node_id"] != 0
        or stage["operation"] != operation
        or stage["dependencies"] != []
        or stage["source"] != files[0]["path"]
        or stage["function"] not in expected_functions
    ):
        fail("mthreads pointwise stage schema or semantics differ")
    autotune_value = stage["autotune"]
    expected_autotune = autotune and not matmul_descriptor
    if (
        set(autotune_value)
        != {"enabled", "warmup", "repetitions", "selection_cache"}
        or autotune_value["enabled"] is not expected_autotune
        or autotune_value["warmup"] != 3
        or autotune_value["repetitions"] != 10
        or autotune_value["selection_cache"] != "tuning/stage-0.json"
    ):
        fail("mthreads pointwise autotune metadata differs")
    variants = stage["variants"]
    if matmul_descriptor:
        expected_variants = 1
    elif autotune and (matmul or convolution):
        expected_variants = 8
    elif autotune and reduction:
        block_n = 1 << (node["attributes"]["reduction"] - 1).bit_length()
        expected_variants = sum(
            2 for block_m in (1, 4, 16) if block_m * block_n <= 65536
        )
    else:
        expected_variants = (
            9
            if autotune
            and stage["function"]
            in {
                "binary_contiguous_kernel",
                "unary_pointwise_contiguous_kernel",
                "binary_select_tensor_kernel",
                "layout_copy_kernel",
                "reshape_contiguous_kernel",
                "slice_copy_kernel",
                "transpose_physical_copy_kernel",
            }
            else 4 if autotune else 1
        )
    if len(variants) != expected_variants:
        fail("mthreads pointwise variant count differs")
    identifiers: set[str] = set()
    input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
    output_uids = {port["name"]: port["uid"] for port in node["outputs"]}
    for variant in variants:
        if (
            set(variant) != VARIANT_KEYS
            or variant["variant_id"] in identifiers
            or variant["source"] != stage["source"]
            or variant["source_sha256"] != manifest["source_sha256"]
            or variant["function"] != stage["function"]
            or len(variant["grid"]) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                for item in variant["grid"]
            )
        ):
            fail("mthreads pointwise variant schema differs")
        identifiers.add(variant["variant_id"])
        if matmul_tle is not None:
            block_m, block_n, _, pipeline_stages, _ = matmul_tle
            a_tensor, b_tensor, output_tensor = request["graph"]["tensors"]
            batch = math.prod(output_tensor["dimensions"][:-2])
            m = a_tensor["dimensions"][-2]
            n = b_tensor["dimensions"][-1]
            expected_variant_id = (
                f"block-{block_m}-warps-16-stages-{pipeline_stages}"
            )
            expected_grid = [
                batch * (m // block_m) * (n // block_n),
                1,
                1,
            ]
            if (
                variant["variant_id"] != expected_variant_id
                or variant["grid"] != expected_grid
            ):
                fail("mthreads TLE Matmul launch configuration differs")
        if operation == "convolution_wgrad":
            block_size = int(variant["variant_id"].split("-")[1])
            filter_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == output_uids["dw"]
            )
            groups = node["attributes"]["groups"]
            out_per_group = filter_tensor["dimensions"][0] // groups
            in_per_group = filter_tensor["dimensions"][1]
            expected_grid = [
                math.ceil(out_per_group / block_size)
                * math.ceil(in_per_group / block_size),
                math.prod(filter_tensor["dimensions"][2:]),
                groups,
            ]
            if variant["grid"] != expected_grid:
                fail("mthreads Wgrad grid must tile output channels")
        if uses_stride2_packed2_1d_dgrad(request):
            block_size = int(variant["variant_id"].split("-")[1])
            image_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == output_uids["dx"]
            )
            groups = node["attributes"]["groups"]
            in_per_group = image_tensor["dimensions"][1] // groups
            base_rows = image_tensor["dimensions"][0] * (
                (image_tensor["dimensions"][2] + 1) // 2
            )
            expected_grid = [
                math.ceil(base_rows / block_size)
                * math.ceil(in_per_group / block_size),
                groups,
                1,
            ]
            if variant["grid"] != expected_grid:
                fail(
                    "mthreads packed 1D stride-2 Dgrad grid must tile "
                    "paired image rows"
                )
        if uses_stride2_tile4_dgrad(request):
            block_size = int(variant["variant_id"].split("-")[1])
            image_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == output_uids["dx"]
            )
            loss_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == input_uids["dy"]
            )
            groups = node["attributes"]["groups"]
            in_per_group = image_tensor["dimensions"][1] // groups
            loss_rows = image_tensor["dimensions"][0] * math.prod(
                loss_tensor["dimensions"][2:]
            )
            block_m = (
                block_size * 4
                if uses_stride2_packed4_dgrad(request)
                else block_size
            )
            channel_block = (
                block_size // 4
                if uses_stride2_packed4_dgrad(request)
                else block_size
            )
            expected_grid = [
                math.ceil(loss_rows / block_m)
                * math.ceil(in_per_group / channel_block),
                groups,
                1,
            ]
            if variant["grid"] != expected_grid:
                fail("mthreads stride-2 Dgrad grid must tile loss rows")
        arguments = variant["arguments"]
        if convolution:
            if operation in {"conv2d_fprop", "convolution_fprop"}:
                expected_argument_kinds = [
                    "tensor",
                    "tensor",
                    "tensor",
                    "tensor",
                ]
                expected_argument_names = [
                    "input",
                    "filter",
                    "bias_placeholder",
                    "output",
                ]
                expected_argument_uids = [
                    input_uids["input"],
                    input_uids["filter"],
                    input_uids["input"],
                    output_uids["output"],
                ]
            elif operation == "convolution_dgrad":
                expected_argument_kinds = ["tensor", "tensor", "tensor"]
                expected_argument_names = ["dy", "w", "dx"]
                expected_argument_uids = [
                    input_uids["dy"],
                    input_uids["w"],
                    output_uids["dx"],
                ]
            else:
                expected_argument_kinds = ["tensor", "tensor", "tensor"]
                expected_argument_names = ["dy", "x", "dw"]
                expected_argument_uids = [
                    input_uids["dy"],
                    input_uids["x"],
                    output_uids["dw"],
                ]
            expected_token_count = {
                "conv1d_gemm_kernel": 30,
                "conv2d_spatial_nchw_kernel": 40,
                "conv3d_spatial_ncdhw_m_kernel": 48,
                "conv_dgrad_nd_kernel": 45,
                "conv_wgrad_nd_kernel": 44,
            }[stage["function"]]
            mode_token_index = None
        elif matmul:
            expected_argument_kinds = ["tensor", "tensor", "tensor"]
            expected_argument_names = ["a", "b", "output"]
            expected_argument_uids = [
                node["inputs"][0]["uid"],
                node["inputs"][1]["uid"],
                node["outputs"][0]["uid"],
            ]
            expected_token_count = (
                15
                if matmul_tle is not None
                else (
                    14
                    if stage["function"] == "matmul_descriptor_kernel"
                    else 42
                )
            )
            mode_token_index = None
        elif reduction:
            expected_argument_kinds = [
                "tensor",
                "tensor",
                "scalar_i32",
            ]
            scalar_name = (
                "outer"
                if stage["function"] == "reduction_2d_kernel"
                else "output_elements"
            )
            expected_argument_names = ["input", "output", scalar_name]
            expected_argument_uids = [
                node["inputs"][0]["uid"],
                node["outputs"][0]["uid"],
                None,
            ]
            expected_token_count = {
                "reduction_2d_kernel": 9,
                "reduction_3d_kernel": 11,
                "reduction_strided_kernel": 32,
            }[stage["function"]]
            mode_token_index = -3
        elif layout:
            expected_argument_kinds = [
                "tensor",
                "tensor",
                "scalar_i32",
            ]
            expected_argument_names = ["input", "output", "n_elements"]
            expected_argument_uids = [
                node["inputs"][0]["uid"],
                node["outputs"][0]["uid"],
                None,
            ]
            expected_token_count = (
                4
                if stage["function"] == "reshape_contiguous_kernel"
                else (
                    30
                    if stage["function"] == "slice_copy_kernel"
                    else (
                        4
                        if stage["function"]
                        == "transpose_physical_copy_kernel"
                        else 37
                    )
                )
            )
            mode_token_index = None
        elif ternary:
            expected_argument_kinds = [
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "scalar_i32",
            ]
            expected_argument_names = [
                "a",
                "b",
                "t",
                "output",
                "n_elements",
            ]
            expected_argument_uids = [
                node["inputs"][0]["uid"],
                node["inputs"][1]["uid"],
                node["inputs"][2]["uid"],
                node["outputs"][0]["uid"],
                None,
            ]
            expected_token_count = (
                6 if stage["function"] == "binary_select_tensor_kernel" else 46
            )
            mode_token_index = None
        elif unary:
            expected_argument_kinds = ["tensor", "tensor", "scalar_i32"]
            expected_argument_names = ["input", "output", "n_elements"]
            expected_argument_uids = [
                node["inputs"][0]["uid"],
                node["outputs"][0]["uid"],
                None,
            ]
            expected_token_count = (
                13
                if stage["function"] == "unary_pointwise_contiguous_kernel"
                else 38
            )
            mode_token_index = -10
        else:
            expected_argument_kinds = [
                "tensor",
                "tensor",
                "tensor",
                "scalar_i32",
            ]
            expected_argument_names = [
                "left",
                "right",
                "output",
                "n_elements",
            ]
            expected_argument_uids = [
                node["inputs"][0]["uid"],
                node["inputs"][1]["uid"],
                node["outputs"][0]["uid"],
                None,
            ]
            expected_token_count = (
                7 if stage["function"] == "binary_contiguous_kernel" else 39
            )
            mode_token_index = -3
        expected_scalar_bits = None
        if not matmul and not convolution:
            scalar_value = (
                node["attributes"]["outer"]
                if reduction and stage["function"] == "reduction_2d_kernel"
                else (
                    node["attributes"]["output_elements"]
                    if reduction
                    else node["attributes"]["n_elements"]
                )
            )
            expected_scalar_bits = struct.pack("<i", scalar_value).hex()
        expected_argument_scalar_bits = [None] * len(expected_argument_kinds)
        if expected_scalar_bits is not None:
            expected_argument_scalar_bits[-1] = expected_scalar_bits
        if (
            len(arguments) != len(expected_argument_kinds)
            or any(set(argument) != ARGUMENT_KEYS for argument in arguments)
            or [argument["kind"] for argument in arguments]
            != expected_argument_kinds
            or [argument["semantic_name"] for argument in arguments]
            != expected_argument_names
            or [argument["uid"] for argument in arguments]
            != expected_argument_uids
            or [argument["scalar_bits"] for argument in arguments]
            != expected_argument_scalar_bits
        ):
            fail("mthreads pointwise runtime argument ABI differs")
        signature_tokens = variant["full_signature"].split(",")
        if len(signature_tokens) != expected_token_count:
            fail("mthreads pointwise full signature token count differs")
        if matmul and stage["function"] == "matmul_strided_kernel":
            input_is_float32 = (
                request["graph"]["tensors"][0]["data_type"] == "float32"
            )
            a_dimensions = request["graph"]["tensors"][0]["dimensions"]
            b_dimensions = request["graph"]["tensors"][1]["dimensions"]
            uses_tf32 = (
                input_is_float32
                and min(
                    a_dimensions[-2],
                    b_dimensions[-1],
                    a_dimensions[-1],
                )
                >= 512
            )
            if signature_tokens[36] != ("1" if input_is_float32 else "0"):
                fail("mthreads Matmul input type flag differs")
            if signature_tokens[37] != ("1" if uses_tf32 else "0"):
                fail("mthreads Matmul float32 input precision policy differs")
        if operation in {"conv2d_fprop", "convolution_fprop"}:
            image_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == input_uids["input"]
            )
            filter_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == input_uids["filter"]
            )
            groups = request["graph"]["nodes"][0]["attributes"]["groups"]
            reduction_extent = (
                image_tensor["dimensions"][1]
                // groups
                * math.prod(filter_tensor["dimensions"][2:])
            )
            expected_input_precision = (
                "1"
                if image_tensor["data_type"] == "float32"
                and reduction_extent >= 64
                else "0"
            )
            precision_token_index = {
                "conv1d_gemm_kernel": 29,
                "conv2d_spatial_nchw_kernel": 27,
                "conv3d_spatial_ncdhw_m_kernel": 47,
            }[stage["function"]]
            if (
                signature_tokens[precision_token_index]
                != expected_input_precision
            ):
                fail("mthreads Fprop float32 input precision policy differs")
        if operation == "convolution_wgrad":
            image_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == input_uids["x"]
            )
            expected_input_precision = (
                "1" if image_tensor["data_type"] == "float32" else "0"
            )
            if signature_tokens[39] != expected_input_precision:
                fail("mthreads Wgrad float32 input precision policy differs")
        if uses_stride2_tile4_dgrad(request) or uses_stride2_packed2_1d_dgrad(
            request
        ):
            image_tensor = next(
                tensor
                for tensor in request["graph"]["tensors"]
                if tensor["uid"] == output_uids["dx"]
            )
            expected_input_precision = (
                "1" if image_tensor["data_type"] == "float32" else "0"
            )
            if signature_tokens[39] != expected_input_precision:
                fail(
                    "mthreads packed stride-2 Dgrad float32 precision policy "
                    "differs"
                )
        if mode_token_index is not None and signature_tokens[
            mode_token_index
        ] != str(pointwise_mode + 1 if reduction else pointwise_mode):
            fail("mthreads pointwise mode is missing from full signature")


def expect_rejected(
    parser: Callable[..., object],
    value: dict[str, Any] | bytes,
    *,
    identity: str,
    label: str,
) -> None:
    payload = (
        value
        if isinstance(value, bytes)
        else json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
    )
    try:
        parser(
            payload,
            expected_target=TARGET,
            expected_identity=identity,
        )
    except (ValueError, RuntimeError):
        return
    fail(f"invalid compiler request was accepted: {label}")


def run_rejection_matrix(
    provider: Any,
    fixture: dict[str, Any],
    identity: str,
) -> int:
    parser = provider.parse_add_request
    base = add_case(fixture, identity)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = []

    def mutation(
        name: str,
    ) -> Callable[[Callable[[dict[str, Any]], None]], None]:
        def register(function: Callable[[dict[str, Any]], None]) -> None:
            mutations.append((name, function))

        return register

    @mutation("duplicate tensor UID")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][1]["uid"] = 100

    @mutation("duplicate node ID")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"].append(
            copy.deepcopy(value["graph"]["nodes"][0])
        )
        value["graph"]["node_count"] = 2

    @mutation("missing virtual producer")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["virtual"] = True

    @mutation("zero dimension")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["dimensions"][0] = 0

    @mutation("negative stride")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["strides"][0] = -1

    @mutation("rank above eight")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["dimensions"] = [1] * 9
        value["graph"]["tensors"][0]["strides"] = [1] * 9

    @mutation("storage overflow")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["dimensions"] = [
            (1 << 31) - 1,
            (1 << 31) - 1,
        ]
        value["graph"]["tensors"][0]["strides"] = [(1 << 31) - 1, 1]

    @mutation("linear index tail overflow")
    def _(value: dict[str, Any]) -> None:
        extent = MAX_LINEAR_ELEMENTS + 1
        for tensor in value["graph"]["tensors"]:
            tensor["dimensions"] = [extent]
            tensor["strides"] = [1]
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = extent

    @mutation("wrong Add arity")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["inputs"].pop()

    @mutation("wrong Add role")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["inputs"][0]["name"] = "x"

    @mutation("wrong Add mode")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["mode"] = 17

    @mutation("wrong Add dtype")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][1]["data_type"] = "float16"

    @mutation("wrong Add compute dtype")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["compute_data_type"] = "float16"

    @mutation("wrong Add output shape")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][2]["dimensions"] = [2, 3, 5]
        value["graph"]["tensors"][2]["strides"] = [15, 5, 1]
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = 30

    @mutation("nonfinite Add alpha")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["alpha"] = float("nan")

    @mutation("unknown request key")
    def _(value: dict[str, Any]) -> None:
        value["source"] = "../kernels/binary.py"

    @mutation("unknown tensor key")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["path"] = "../../escape"

    @mutation("unknown Add attribute")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["BLOCK_SIZE"] = 4096

    @mutation("boolean integer")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = True

    for label, apply in mutations:
        value = copy.deepcopy(base)
        apply(value)
        expect_rejected(parser, value, identity=identity, label=label)

    forward = copy.deepcopy(base)
    virtual = copy.deepcopy(forward["graph"]["tensors"][2])
    virtual.update(uid=103, virtual=True)
    forward["graph"]["tensors"].append(virtual)
    forward["graph"]["tensor_count"] = 4
    consumer = forward["graph"]["nodes"][0]
    consumer["inputs"][0]["uid"] = 103
    producer = copy.deepcopy(consumer)
    producer["id"] = 1
    producer["inputs"][0]["uid"] = 100
    producer["outputs"][0]["uid"] = 103
    forward["graph"]["nodes"].append(producer)
    forward["graph"]["node_count"] = 2
    expect_rejected(
        parser, forward, identity=identity, label="forward dependency"
    )

    canonical = json.dumps(base, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    duplicate = canonical.replace(b"{", b'{"schema_version":3,', 1)
    expect_rejected(
        parser, duplicate, identity=identity, label="duplicate JSON key"
    )
    return len(mutations) + 2


def run_ternary_rejection_matrix(
    provider: Any,
    fixture: dict[str, Any],
    identity: str,
) -> int:
    parser = provider.parse_pointwise_request
    base = ternary_case(fixture, identity)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = []

    def mutation(
        name: str,
    ) -> Callable[[Callable[[dict[str, Any]], None]], None]:
        def register(function: Callable[[dict[str, Any]], None]) -> None:
            mutations.append((name, function))

        return register

    @mutation("ternary missing predicate input")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["inputs"].pop()

    @mutation("ternary wrong input role")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["inputs"][2]["name"] = "condition"

    @mutation("ternary duplicate input role")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["inputs"][1]["name"] = "a"

    @mutation("ternary wrong output role")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["outputs"][0]["name"] = "result"

    @mutation("ternary duplicate tensor binding")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["inputs"][2]["uid"] = 100

    @mutation("ternary floating predicate")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][2]["data_type"] = "float32"

    @mutation("ternary nonfloating value")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][0]["data_type"] = "boolean"

    @mutation("ternary mismatched right dtype")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][1]["data_type"] = "float16"

    @mutation("ternary mismatched output dtype")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][3]["data_type"] = "float16"

    @mutation("ternary wrong compute dtype")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["compute_data_type"] = "float16"

    @mutation("ternary wrong output shape")
    def _(value: dict[str, Any]) -> None:
        output = value["graph"]["tensors"][3]
        output["dimensions"] = [2, 3, 5]
        output["strides"] = [15, 5, 1]
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = 30

    @mutation("ternary incompatible broadcast")
    def _(value: dict[str, Any]) -> None:
        right = value["graph"]["tensors"][1]
        right["dimensions"] = [2, 5, 4]
        right["strides"] = [20, 4, 1]

    @mutation("ternary wrong mode")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["mode"] = 1

    @mutation("ternary wrong operation")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["type"] = "add"

    @mutation("ternary unknown attribute")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["alpha"] = 1.0

    @mutation("ternary element count mismatch")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = 23

    @mutation("ternary boolean element count")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = True

    @mutation("ternary virtual predicate")
    def _(value: dict[str, Any]) -> None:
        value["graph"]["tensors"][2]["virtual"] = True

    @mutation("ternary unexpected fifth tensor")
    def _(value: dict[str, Any]) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][3])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 5

    for label, apply in mutations:
        value = copy.deepcopy(base)
        apply(value)
        expect_rejected(parser, value, identity=identity, label=label)
    return len(mutations)


def run_layout_rejection_matrix(
    provider: Any,
    fixture: dict[str, Any],
    identity: str,
) -> int:
    parser = provider.parse_compiler_request
    reshape = layout_case(
        fixture,
        identity,
        operation="reshape",
        input_dimensions=[2, 3, 4],
        output_dimensions=[6, 4],
    )
    transpose = layout_case(
        fixture,
        identity,
        operation="transpose",
        input_dimensions=[2, 3, 4],
        output_dimensions=[4, 2, 3],
        permutation=[2, 0, 1],
    )
    sliced = layout_case(
        fixture,
        identity,
        operation="slice",
        input_dimensions=[2, 4, 5],
        output_dimensions=[2, 2, 5],
        starts=[0, 1, 0],
        limits=[2, 4, 5],
        slice_strides=[1, 2, 1],
    )
    mutations: list[
        tuple[str, dict[str, Any], Callable[[dict[str, Any]], None]]
    ] = [
        (
            "layout wrong input role",
            reshape,
            lambda value: value["graph"]["nodes"][0]["inputs"][0].update(
                name="x"
            ),
        ),
        (
            "layout mismatched dtype",
            reshape,
            lambda value: value["graph"]["tensors"][1].update(
                data_type="float16"
            ),
        ),
        (
            "layout wrong compute dtype",
            reshape,
            lambda value: value["graph"]["nodes"][0].update(
                compute_data_type="float16"
            ),
        ),
        (
            "reshape view-only mode",
            reshape,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                reshape_mode=1
            ),
        ),
        (
            "reshape input rank mismatch",
            reshape,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                input_rank=2
            ),
        ),
        (
            "layout element count mismatch",
            reshape,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                n_elements=23
            ),
        ),
        (
            "layout duplicated input dimensions mismatch",
            reshape,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                input_dimensions=[2, 2, 6]
            ),
        ),
        (
            "transpose duplicate permutation axis",
            transpose,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                permutation=[2, 0, 0]
            ),
        ),
        (
            "transpose output shape mismatch",
            transpose,
            lambda value: value["graph"]["tensors"][1].update(
                dimensions=[4, 3, 2], strides=[1, 4, 12]
            ),
        ),
        (
            "transpose missing permutation",
            transpose,
            lambda value: value["graph"]["nodes"][0]["attributes"].pop(
                "permutation"
            ),
        ),
        (
            "slice negative start",
            sliced,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                starts=[0, -1, 0]
            ),
        ),
        (
            "slice limit exceeds input",
            sliced,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                limits=[2, 5, 5]
            ),
        ),
        (
            "slice zero stride",
            sliced,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                slice_strides=[1, 0, 1]
            ),
        ),
        (
            "layout virtual input",
            reshape,
            lambda value: value["graph"]["tensors"][0].update(virtual=True),
        ),
        (
            "layout unknown attribute",
            reshape,
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                block_size=256
            ),
        ),
    ]
    for label, base, apply in mutations:
        value = copy.deepcopy(base)
        apply(value)
        expect_rejected(parser, value, identity=identity, label=label)
    return len(mutations)


def run_reduction_rejection_matrix(
    provider: Any,
    fixture: dict[str, Any],
    identity: str,
) -> int:
    parser = provider.parse_reduction_request
    base = reduction_case(
        fixture,
        identity,
        operation="reduction_sum",
        mode=0,
        input_dimensions=[2, 4, 8],
        axis=1,
        keep_dimensions=True,
    )

    def unexpected_tensor(value: dict[str, Any]) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][1])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 3

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "reduction wrong input role",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].update(
                name="x"
            ),
        ),
        (
            "reduction mismatched dtype",
            lambda value: value["graph"]["tensors"][1].update(
                data_type="float16"
            ),
        ),
        (
            "reduction boolean storage",
            lambda value: [
                tensor.update(data_type="boolean")
                for tensor in value["graph"]["tensors"]
            ],
        ),
        (
            "reduction wrong compute dtype",
            lambda value: value["graph"]["nodes"][0].update(
                compute_data_type="float16"
            ),
        ),
        (
            "reduction mode and operation mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                mode=1
            ),
        ),
        (
            "reduction negative axis",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                axis=-1
            ),
        ),
        (
            "reduction boolean keep dimensions",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                keep_dimensions=True
            ),
        ),
        (
            "reduction wrong output shape",
            lambda value: value["graph"]["tensors"][1].update(
                dimensions=[2, 8], strides=[8, 1]
            ),
        ),
        (
            "reduction outer mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                outer=3
            ),
        ),
        (
            "reduction extent mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                reduction=5
            ),
        ),
        (
            "reduction inner mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                inner=7
            ),
        ),
        (
            "reduction output elements mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                output_elements=15
            ),
        ),
        (
            "reduction unknown attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                block_n=8
            ),
        ),
        (
            "reduction virtual input",
            lambda value: value["graph"]["tensors"][0].update(virtual=True),
        ),
        ("reduction unexpected tensor", unexpected_tensor),
    ]
    for label, apply in mutations:
        value = copy.deepcopy(base)
        apply(value)
        expect_rejected(parser, value, identity=identity, label=label)
    return len(mutations)


def run_matmul_rejection_matrix(
    provider: Any,
    fixture: dict[str, Any],
    identity: str,
) -> int:
    parser = provider.parse_matmul_request
    base = matmul_case(
        fixture,
        identity,
        a_dimensions=[2, 1, 17, 30],
        b_dimensions=[3, 30, 23],
    )

    def unexpected_tensor(value: dict[str, Any]) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][2])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 4

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "Matmul wrong input role",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].update(
                name="left"
            ),
        ),
        (
            "Matmul mismatched dtype",
            lambda value: value["graph"]["tensors"][1].update(
                data_type="float16"
            ),
        ),
        (
            "Matmul boolean storage",
            lambda value: [
                tensor.update(data_type="boolean")
                for tensor in value["graph"]["tensors"]
            ],
        ),
        (
            "Matmul wrong compute dtype",
            lambda value: value["graph"]["nodes"][0].update(
                compute_data_type="float16"
            ),
        ),
        (
            "Matmul contraction mismatch",
            lambda value: value["graph"]["tensors"][1].update(
                dimensions=[3, 31, 23]
            ),
        ),
        (
            "Matmul nonbroadcast batch",
            lambda value: value["graph"]["tensors"][0].update(
                dimensions=[2, 2, 17, 30]
            ),
        ),
        (
            "Matmul output shape mismatch",
            lambda value: value["graph"]["tensors"][2].update(
                dimensions=[2, 3, 17, 24]
            ),
        ),
        *[
            (
                f"Matmul {attribute} attribute mismatch",
                lambda value, name=attribute: value["graph"]["nodes"][0][
                    "attributes"
                ].update(
                    {name: value["graph"]["nodes"][0]["attributes"][name] + 1}
                ),
            )
            for attribute in ("batch", "m", "n", "k")
        ],
        (
            "Matmul virtual input",
            lambda value: value["graph"]["tensors"][0].update(virtual=True),
        ),
        (
            "Matmul unknown attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                block_m=64
            ),
        ),
        (
            "Matmul optional port",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].update(
                optional=True
            ),
        ),
        ("Matmul unexpected tensor", unexpected_tensor),
    ]
    for label, apply in mutations:
        value = copy.deepcopy(base)
        apply(value)
        expect_rejected(parser, value, identity=identity, label=label)
    return len(mutations)


def run_convolution_rejection_matrix(
    provider: Any,
    fixture: dict[str, Any],
    identity: str,
) -> int:
    parser = provider.parse_convolution_request
    base = convolution_case(
        fixture,
        identity,
        operation="convolution_dgrad",
        image_dimensions=[2, 4, 7, 8],
        filter_dimensions=[6, 2, 3, 3],
        pre_padding=[1, 0],
        post_padding=[0, 1],
        stride=[1, 1],
        dilation=[1, 1],
        groups=2,
        convolution_mode=1,
    )

    def unexpected_tensor(value: dict[str, Any]) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][2])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 4

    def wrong_loss_shape(value: dict[str, Any]) -> None:
        dimensions = list(value["graph"]["tensors"][2]["dimensions"])
        dimensions[-1] += 1
        value["graph"]["tensors"][2].update(
            dimensions=dimensions,
            strides=dense_strides(dimensions),
        )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "convolution wrong input role",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].update(
                name="loss"
            ),
        ),
        (
            "convolution duplicate tensor UID",
            lambda value: value["graph"]["tensors"][1].update(uid=100),
        ),
        (
            "convolution mismatched dtype",
            lambda value: value["graph"]["tensors"][1].update(
                data_type="float16"
            ),
        ),
        (
            "convolution boolean storage",
            lambda value: [
                tensor.update(data_type="boolean")
                for tensor in value["graph"]["tensors"]
            ],
        ),
        (
            "convolution wrong compute dtype",
            lambda value: value["graph"]["nodes"][0].update(
                compute_data_type="float16"
            ),
        ),
        (
            "convolution spatial rank mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                spatial_rank=3
            ),
        ),
        (
            "convolution padding length mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"][
                "pre_padding"
            ].pop(),
        ),
        (
            "convolution negative padding",
            lambda value: value["graph"]["nodes"][0]["attributes"][
                "post_padding"
            ].__setitem__(0, -1),
        ),
        (
            "convolution zero stride",
            lambda value: value["graph"]["nodes"][0]["attributes"][
                "stride"
            ].__setitem__(0, 0),
        ),
        (
            "convolution zero dilation",
            lambda value: value["graph"]["nodes"][0]["attributes"][
                "dilation"
            ].__setitem__(0, 0),
        ),
        (
            "convolution indivisible groups",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                groups=3
            ),
        ),
        (
            "convolution filter channel mismatch",
            lambda value: value["graph"]["tensors"][1].update(
                dimensions=[6, 1, 3, 3]
            ),
        ),
        ("convolution loss shape mismatch", wrong_loss_shape),
        (
            "convolution output count mismatch",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                n_outputs=value["graph"]["nodes"][0]["attributes"]["n_outputs"]
                + 1
            ),
        ),
        (
            "convolution invalid mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                convolution_mode=2
            ),
        ),
        (
            "convolution unknown attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].update(
                block_m=32
            ),
        ),
        (
            "convolution optional port",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].update(
                optional=True
            ),
        ),
        (
            "convolution virtual tensor",
            lambda value: value["graph"]["tensors"][0].update(virtual=True),
        ),
        ("convolution unexpected tensor", unexpected_tensor),
    ]
    for label, apply in mutations:
        value = copy.deepcopy(base)
        apply(value)
        expect_rejected(parser, value, identity=identity, label=label)
    return len(mutations)


def installed_layout_contract(
    *,
    source_root: Path,
    compiler: Path,
    provider_path: Path,
    environment_report: Path,
    python: Path,
    environment: dict[str, str],
    identity: str,
    request: dict[str, Any],
    root: Path,
) -> None:
    prefix = root / "installed"
    resource = prefix / "share/flagdnn"
    installed_compiler_package = resource / "compiler/flagdnn_codegen"
    installed_backend = resource / "backends/mthreads"
    installed_common = resource / "kernels"
    shutil.copytree(compiler.parent, installed_compiler_package)
    shutil.copytree(
        source_root / "kernels/common", installed_common / "common"
    )
    shutil.copy2(
        source_root / "kernels/registry.json",
        installed_common / "registry.json",
    )
    installed_backend.mkdir(parents=True)
    for name in (
        "compiler.py",
        "environment_identity.py",
    ):
        shutil.copy2(provider_path.parent / name, installed_backend / name)
    for directory in ("dispatch", "codegen"):
        shutil.copytree(
            provider_path.parent / directory, installed_backend / directory
        )
    shutil.copytree(
        provider_path.parent / "kernels", installed_backend / "kernels"
    )
    shutil.copytree(
        provider_path.parent / "tuning", installed_backend / "tuning"
    )
    shutil.copy2(
        environment_report,
        installed_backend / "flagdnn_mthreads_environment.json",
    )

    installed_environment = dict(environment)
    installed_environment.pop("FLAGDNN_MTHREADS_ENVIRONMENT_REPORT", None)
    installed_environment["FLAGDNN_BACKEND_ROOT"] = str(resource / "backends")
    identity_output = root / "installed-identity.txt"
    installed_identity = identify(
        compiler=installed_compiler_package / "main.py",
        python=python,
        target=TARGET,
        engine="libtriton_jit",
        output=identity_output,
        environment=installed_environment,
        expect_success=True,
    )
    if installed_identity != identity:
        fail("installed compiler identity differs from source layout")
    metadata = json.loads(
        identity_output.read_text(encoding="utf-8").splitlines()[1]
    )
    forbidden = str(source_root)
    if any(path.startswith(forbidden) for path in metadata.get("files", [])):
        fail("installed identity reached the source tree")
    installed_request = root / "installed-request.json"
    write_request(installed_request, request)
    installed_output = root / "installed-artifact"
    run(
        [
            str(python),
            str(installed_compiler_package / "main.py"),
            "--request",
            str(installed_request),
            "--output-dir",
            str(installed_output),
            "--execution-engine",
            "libtriton_jit",
            "--quiet",
        ],
        environment=installed_environment,
        expect_success=True,
    )
    manifest = json.loads(
        (installed_output / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["compiler_identity"] != identity:
        fail("installed artifact did not use the installed provider")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--capture-executable", type=Path, required=True)
    parser.add_argument("--capture-compiler", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def check_fp8_target_plans(provider, fixture, identity):
    """FP8 capability checks must use the request target, not a fixed device."""
    extended = importlib.import_module(
        provider.__package__ + ".dispatch.extended"
    )
    request = copy.deepcopy(fixture)
    request["compiler_identity"] = identity
    tensors = request["graph"]["tensors"]
    for tensor, dtype, shape in zip(
        tensors,
        ("fp8_e4m3", "fp8_e5m2", "float32"),
        ([1, 16, 32], [1, 32, 24], [1, 16, 24]),
    ):
        tensor.update(
            data_type=dtype,
            dimensions=shape,
            strides=[shape[1] * shape[2], shape[2], 1],
            alignment=16,
            virtual=False,
        )
    node = request["graph"]["nodes"][0]
    node["type"] = "matmul_fp8"
    node["inputs"] = [
        {"name": "a", "uid": tensors[0]["uid"]},
        {"name": "b", "uid": tensors[1]["uid"]},
    ]
    node["outputs"] = [{"name": "output", "uid": tensors[2]["uid"]}]
    node["attributes"] = {
        "batch": 1,
        "m": 16,
        "n": 24,
        "k": 32,
        "scale_mode": 0,
    }
    for architecture in (30, 31, 32):
        request["target"] = f"musa-mtgpu-cc{architecture}-w32"
        parsed = provider.parse_compiler_request(
            json.dumps(request).encode(),
            expected_target=request["target"],
            expected_identity=identity,
        )
        try:
            result = extended.plan_extended(parsed, "0" * 64)
        except ValueError as error:
            if architecture >= 31 or "requires MUSA cc31" not in str(error):
                raise
        else:
            if architecture < 31 or len(result.stages) != 1:
                fail("FP8 target capability check differs from the request")


def check_activation_backward_plans(provider, fixture, identity):
    """Exercise public input roles and typed ABI without loading Torch."""
    extended = importlib.import_module(
        provider.__package__ + ".dispatch.extended"
    )
    metadata = importlib.import_module(
        provider.__package__ + ".dispatch.metadata"
    )
    operations = (
        "relu_backward",
        "tanh_backward",
        "elu_backward",
        "gelu_backward",
        "softplus_backward",
        "swish_backward",
        "gelu_approx_tanh_backward",
    )
    for operation in operations:
        for dtype, pointer in (
            ("float32", "*fp32:16"),
            ("float16", "*fp16:16"),
            ("bfloat16", "*bf16:16"),
        ):
            request = copy.deepcopy(fixture)
            request["compiler_identity"] = identity
            graph = request["graph"]
            tensors = graph["tensors"]
            for tensor in tensors:
                tensor.update(
                    data_type=dtype,
                    dimensions=[1, 1, 17],
                    strides=[17, 17, 1],
                    alignment=16,
                    virtual=False,
                )
            node = graph["nodes"][0]
            node["type"] = operation
            node["inputs"] = [
                {"name": "left", "uid": tensors[0]["uid"]},
                {"name": "right", "uid": tensors[1]["uid"]},
            ]
            node["outputs"] = [{"name": "output", "uid": tensors[2]["uid"]}]
            node["attributes"] = {
                "n_elements": 17,
                "alpha": 1.0,
                "pointwise_mode": metadata.BINARY_POINTWISE_MODES[operation],
                "has_upper_clip": 0,
            }

            def plan(value):
                parsed = provider.parse_compiler_request(
                    json.dumps(value).encode(),
                    expected_target=TARGET,
                    expected_identity=identity,
                )
                return extended.plan_extended(parsed, "0" * 64)

            result = plan(request)
            variant = result.stages[0].variants[0]
            if variant.full_signature.split(",")[:3] != [pointer] * 3:
                fail("activation backward pointer ABI differs")
            if [arg.uid for arg in variant.arguments[:3]] != [
                tensor["uid"] for tensor in tensors
            ]:
                fail("activation backward input roles differ")
            for mutation in ("role", "type", "elements"):
                invalid = copy.deepcopy(request)
                if mutation == "role":
                    invalid["graph"]["nodes"][0]["inputs"][0]["name"] = "x"
                elif mutation == "type":
                    invalid["graph"]["tensors"][0]["data_type"] = "int32"
                else:
                    invalid["graph"]["nodes"][0]["attributes"][
                        "n_elements"
                    ] = 18
                try:
                    plan(invalid)
                except ValueError:
                    pass
                else:
                    fail("invalid activation backward accepted: " + mutation)


def main() -> int:
    arguments = parse_arguments()
    provider_path = arguments.provider.resolve(strict=True)
    compiler = arguments.compiler.resolve(strict=True)
    environment_report = arguments.environment_report.resolve(strict=True)
    capture_executable = arguments.capture_executable.resolve(strict=True)
    capture_compiler = arguments.capture_compiler.resolve(strict=True)
    python = arguments.python.expanduser().absolute()
    if not python.is_file():
        fail("configured compiler Python does not exist")
    if arguments.target != TARGET:
        fail("compiler contract target differs from the validated target")
    source_root = compiler.parents[2]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "FLAGDNN_BACKEND_ROOT": str(provider_path.parent.parent),
            "FLAGDNN_MTHREADS_ENVIRONMENT_REPORT": str(environment_report),
            "TRITON_JIT_BACKEND": "MTGPU",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        }
    )

    sys.path.insert(0, str(compiler.parent.parent))
    try:
        loader = importlib.import_module("flagdnn_codegen.provider_loader")
        provider = loader.get_provider("mthreads")
    finally:
        sys.path.pop(0)

    if Path(provider.__file__).resolve() != provider_path:
        fail("provider loader selected a different mthreads compiler")

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-mthreads-compiler-contract-"
    ) as temporary:
        root = Path(temporary)
        fixture = captured_request(
            executable=capture_executable,
            capture_compiler=capture_compiler,
            python=python,
            environment=environment,
            root=root,
        )
        add_square_fixture = captured_request(
            executable=capture_executable,
            capture_compiler=capture_compiler,
            python=python,
            environment=environment,
            root=root,
            graph_kind="add_square",
        )
        conv_bias_relu_fixture = captured_request(
            executable=capture_executable,
            capture_compiler=capture_compiler,
            python=python,
            environment=environment,
            root=root,
            graph_kind="conv_bias_relu",
        )
        normalization_fixtures = {
            graph_kind: captured_request(
                executable=capture_executable,
                capture_compiler=capture_compiler,
                python=python,
                environment=environment,
                root=root,
                graph_kind=graph_kind,
            )
            for graph_kind in (
                "layernorm",
                "rmsnorm",
                "batchnorm",
                "batchnorm_inference",
            )
        }
        attention_fixtures = {
            graph_kind: captured_request(
                executable=capture_executable,
                capture_compiler=capture_compiler,
                python=python,
                environment=environment,
                root=root,
                graph_kind=graph_kind,
            )
            for graph_kind in (
                "sdpa",
                "sdpa_backward",
                "sdpa_fp8",
                "sdpa_fp8_backward",
            )
        }
        identity = identify(
            compiler=compiler,
            python=python,
            target=TARGET,
            engine="libtriton_jit",
            output=root / "identity.txt",
            environment=environment,
            expect_success=True,
        )
        assert identity is not None
        alternate_identity = identify(
            compiler=compiler,
            python=python,
            target="musa-mtgpu-cc32-w32",
            engine="libtriton_jit",
            output=root / "altered-target.txt",
            environment=environment,
            expect_success=True,
        )
        if alternate_identity == identity:
            fail("mthreads target does not contribute to compiler identity")
        identify(
            compiler=compiler,
            python=python,
            target="musa-mtgpu-cc31-w16",
            engine="libtriton_jit",
            output=root / "unsupported-target.txt",
            environment=environment,
            expect_success=False,
        )
        identify(
            compiler=compiler,
            python=python,
            target=TARGET,
            engine="external_artifact",
            output=root / "external-engine.txt",
            environment=environment,
            expect_success=False,
        )
        check_activation_backward_plans(provider, fixture, identity)
        check_fp8_target_plans(provider, fixture, identity)
        negative_cases = (
            run_rejection_matrix(provider, fixture, identity)
            + run_ternary_rejection_matrix(provider, fixture, identity)
            + run_layout_rejection_matrix(provider, fixture, identity)
            + run_reduction_rejection_matrix(provider, fixture, identity)
        )
        negative_cases += run_matmul_rejection_matrix(
            provider, fixture, identity
        )
        negative_cases += run_convolution_rejection_matrix(
            provider, fixture, identity
        )

        add_square_request = copy.deepcopy(add_square_fixture)
        add_square_request["compiler_identity"] = identity
        add_square_request["build_options"]["autotune"] = True
        conv_bias_relu_request = copy.deepcopy(conv_bias_relu_fixture)
        conv_bias_relu_request["compiler_identity"] = identity
        conv_bias_relu_request["build_options"]["autotune"] = True
        normalization_requests = {
            operation: copy.deepcopy(request)
            for operation, request in normalization_fixtures.items()
        }
        for request in normalization_requests.values():
            request["compiler_identity"] = identity
            request["build_options"]["autotune"] = True
        long_row_layernorm = copy.deepcopy(normalization_requests["layernorm"])
        long_row_graph = long_row_layernorm["graph"]
        long_row_graph["name"] = "layernorm-bf16-long-row-autotune"
        long_row_node = long_row_graph["nodes"][0]
        long_row_tensors = {
            tensor["uid"]: tensor for tensor in long_row_graph["tensors"]
        }
        long_row_inputs = {
            port["name"]: port["uid"] for port in long_row_node["inputs"]
        }
        long_row_outputs = {
            port["name"]: port["uid"] for port in long_row_node["outputs"]
        }
        for name in ("x", "y"):
            uid = (
                long_row_inputs[name]
                if name in long_row_inputs
                else long_row_outputs[name]
            )
            tensor = long_row_tensors[uid]
            tensor["data_type"] = "bfloat16"
            tensor["dimensions"] = [2, 3, 4096]
            tensor["strides"] = [12288, 4096, 1]
        for name in ("scale", "bias"):
            tensor = long_row_tensors[long_row_inputs[name]]
            tensor["data_type"] = "bfloat16"
            tensor["dimensions"] = [1, 1, 4096]
            tensor["strides"] = [4096, 4096, 1]
        for name in ("mean", "inv_variance"):
            tensor = long_row_tensors[long_row_outputs[name]]
            tensor["dimensions"] = [2, 3, 1]
            tensor["strides"] = [3, 1, 1]
        long_row_node["attributes"]["normalized_elements"] = 4096
        long_row_node["attributes"]["rows"] = 6
        attention_requests = {
            operation: copy.deepcopy(request)
            for operation, request in attention_fixtures.items()
        }
        for request in attention_requests.values():
            request["compiler_identity"] = identity
            request["build_options"]["autotune"] = True

        def row_major_normalization_request(
            request: dict[str, Any],
        ) -> dict[str, Any]:
            result = copy.deepcopy(request)
            graph = result["graph"]
            node = graph["nodes"][0]
            tensor_by_uid = {
                tensor["uid"]: tensor for tensor in graph["tensors"]
            }
            input_uids = {port["name"]: port["uid"] for port in node["inputs"]}
            output_uids = {
                port["name"]: port["uid"] for port in node["outputs"]
            }
            x = tensor_by_uid[input_uids["x"]]
            y = tensor_by_uid[output_uids["y"]]
            x["strides"] = dense_strides(x["dimensions"])
            y["strides"] = dense_strides(y["dimensions"])
            node["attributes"]["x_strides"] = x["strides"]
            node["attributes"]["y_strides"] = y["strides"]
            return result

        batchnorm_row_major = row_major_normalization_request(
            normalization_requests["batchnorm"]
        )
        large_batchnorm_row_major = copy.deepcopy(batchnorm_row_major)
        large_batchnorm_graph = large_batchnorm_row_major["graph"]
        large_batchnorm_graph["name"] = "batchnorm-large-nchw-autotune"
        large_batchnorm_node = large_batchnorm_graph["nodes"][0]
        large_batchnorm_tensors = {
            tensor["uid"]: tensor
            for tensor in large_batchnorm_graph["tensors"]
        }
        large_batchnorm_inputs = {
            port["name"]: port["uid"]
            for port in large_batchnorm_node["inputs"]
        }
        large_batchnorm_outputs = {
            port["name"]: port["uid"]
            for port in large_batchnorm_node["outputs"]
        }
        large_dimensions = [8, 64, 56, 56]
        large_strides = dense_strides(large_dimensions)
        for uid in (
            large_batchnorm_inputs["x"],
            large_batchnorm_outputs["y"],
        ):
            tensor = large_batchnorm_tensors[uid]
            tensor["dimensions"] = large_dimensions
            tensor["strides"] = large_strides
        parameter_dimensions = [1, 64, 1, 1]
        parameter_strides = dense_strides(parameter_dimensions)
        for uid in (
            *(
                large_batchnorm_inputs[name]
                for name in (
                    "scale",
                    "bias",
                    "previous_running_mean",
                    "previous_running_variance",
                )
            ),
            *(
                large_batchnorm_outputs[name]
                for name in (
                    "mean",
                    "inv_variance",
                    "next_running_mean",
                    "next_running_variance",
                )
            ),
        ):
            tensor = large_batchnorm_tensors[uid]
            tensor["dimensions"] = parameter_dimensions
            tensor["strides"] = parameter_strides
        large_batchnorm_node["attributes"].update(
            {
                "batch": 8,
                "channels": 64,
                "dimensions": large_dimensions,
                "n_elements": math.prod(large_dimensions),
                "rank": 4,
                "spatial": 56 * 56,
                "x_strides": large_strides,
                "y_strides": large_strides,
            }
        )
        batchnorm_inference_row_major = row_major_normalization_request(
            normalization_requests["batchnorm_inference"]
        )
        successful_cases = {
            "fp32-negative": add_case(fixture, identity, alpha=-0.75),
            "fp16-positive": add_case(
                fixture, identity, data_type="float16", alpha=1.5
            ),
            "bf16-negative": add_case(
                fixture, identity, data_type="bfloat16", alpha=-2.0
            ),
            "padded-strided": add_case(
                fixture,
                identity,
                dimensions=[2, 3, 4],
                strides=[32, 8, 1],
            ),
            "broadcast": add_case(
                fixture,
                identity,
                right_dimensions=[1, 3, 1],
                right_strides=[3, 1, 1],
            ),
            "autotune-dense": add_case(fixture, identity, autotune=True),
            "autotune-strided": add_case(
                fixture,
                identity,
                dimensions=[2, 3, 4],
                strides=[32, 8, 1],
                autotune=True,
            ),
            "sub-alpha-neg2": binary_case(
                fixture,
                identity,
                operation="sub",
                mode=17,
                data_type="float16",
                alpha=-2.0,
            ),
            "mod-fp32": binary_case(
                fixture,
                identity,
                operation="mod",
                mode=22,
            ),
            "cmp-eq-fp16": binary_case(
                fixture,
                identity,
                operation="cmp_eq",
                mode=25,
                data_type="float16",
                output_data_type="boolean",
                compute_data_type="boolean",
            ),
            "logical-and": binary_case(
                fixture,
                identity,
                operation="logical_and",
                mode=31,
                data_type="boolean",
                output_data_type="boolean",
                compute_data_type="boolean",
            ),
            "sigmoid-backward-bf16": binary_case(
                fixture,
                identity,
                operation="sigmoid_backward",
                mode=40,
                data_type="bfloat16",
            ),
            "unary-relu": unary_case(
                fixture,
                identity,
                operation="relu",
                mode=2,
            ),
            "unary-relu-strided-autotune": unary_case(
                fixture,
                identity,
                operation="relu",
                mode=2,
                strides=[32, 8, 1],
                autotune=True,
                negative_slope=0.25,
                lower_clip=-0.5,
                upper_clip=6.0,
                has_upper_clip=True,
            ),
            "unary-logical-not": unary_case(
                fixture,
                identity,
                operation="logical_not",
                mode=24,
                data_type="boolean",
                compute_data_type="boolean",
            ),
            "unary-swish-fp16": unary_case(
                fixture,
                identity,
                operation="swish",
                mode=38,
                data_type="float16",
                swish_beta=1.5,
            ),
            "unary-elu-bf16": unary_case(
                fixture,
                identity,
                operation="elu",
                mode=35,
                data_type="bfloat16",
                elu_alpha=0.25,
            ),
            "unary-softplus": unary_case(
                fixture,
                identity,
                operation="softplus",
                mode=37,
                softplus_beta=2.0,
            ),
            "ternary-binary-select": ternary_case(
                fixture,
                identity,
            ),
            "ternary-broadcast-autotune": ternary_case(
                fixture,
                identity,
                data_type="float16",
                a_dimensions=[2, 3, 4],
                b_dimensions=[1, 3, 1],
                predicate_dimensions=[2, 1, 4],
                output_dimensions=[2, 3, 4],
                autotune=True,
            ),
            "layout-reshape-autotune": layout_case(
                fixture,
                identity,
                operation="reshape",
                input_dimensions=[2, 3, 4],
                output_dimensions=[6, 4],
                autotune=True,
            ),
            "layout-transpose": layout_case(
                fixture,
                identity,
                operation="transpose",
                input_dimensions=[2, 3, 4],
                output_dimensions=[4, 2, 3],
                permutation=[2, 0, 1],
                data_type="bfloat16",
            ),
            "layout-slice": layout_case(
                fixture,
                identity,
                operation="slice",
                input_dimensions=[2, 4, 5],
                output_dimensions=[2, 2, 5],
                starts=[0, 1, 0],
                limits=[2, 4, 5],
                slice_strides=[1, 2, 1],
                data_type="float16",
            ),
            "reduction-sum-2d": reduction_case(
                fixture,
                identity,
                operation="reduction_sum",
                mode=0,
                input_dimensions=[7, 256],
                axis=1,
                keep_dimensions=False,
            ),
            "reduction-avg-3d-autotune": reduction_case(
                fixture,
                identity,
                operation="reduction_avg",
                mode=1,
                input_dimensions=[2, 4, 8],
                axis=1,
                keep_dimensions=True,
                data_type="float16",
                autotune=True,
            ),
            "reduction-mul-strided": reduction_case(
                fixture,
                identity,
                operation="reduction_mul",
                mode=2,
                input_dimensions=[2, 3, 5, 5],
                axis=1,
                keep_dimensions=True,
                data_type="bfloat16",
                input_strides=[75, 1, 15, 3],
            ),
            "reduction-scalar-output": reduction_case(
                fixture,
                identity,
                operation="reduction_sum",
                mode=0,
                input_dimensions=[8],
                axis=0,
                keep_dimensions=False,
            ),
            "matmul-fp32-2d": matmul_case(
                fixture,
                identity,
                a_dimensions=[17, 30],
                b_dimensions=[30, 23],
            ),
            "matmul-fp16-batched": matmul_case(
                fixture,
                identity,
                a_dimensions=[2, 17, 30],
                b_dimensions=[2, 30, 23],
                data_type="float16",
            ),
            "matmul-fp16-tle-autotune": matmul_case(
                fixture,
                identity,
                a_dimensions=[32, 512, 512],
                b_dimensions=[32, 512, 512],
                data_type="float16",
                autotune=True,
            ),
            "matmul-bf16-tle-autotune": matmul_case(
                fixture,
                identity,
                a_dimensions=[32, 512, 512],
                b_dimensions=[32, 512, 512],
                data_type="bfloat16",
                autotune=True,
            ),
            "matmul-bf16-broadcast-autotune": matmul_case(
                fixture,
                identity,
                a_dimensions=[2, 1, 17, 30],
                b_dimensions=[3, 30, 23],
                data_type="bfloat16",
                autotune=True,
            ),
            "convolution-dgrad-fp32-stride2-tile4-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_dgrad",
                image_dimensions=[1, 3, 640, 640],
                filter_dimensions=[16, 3, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="float32",
                autotune=True,
            ),
            "convolution-dgrad-fp32-stride2-packed1d-autotune": (
                convolution_case(
                    fixture,
                    identity,
                    operation="convolution_dgrad",
                    image_dimensions=[8, 64, 255],
                    filter_dimensions=[96, 64, 5],
                    pre_padding=[2],
                    post_padding=[1],
                    stride=[2],
                    dilation=[1],
                    groups=1,
                    convolution_mode=0,
                    data_type="float32",
                    autotune=True,
                )
            ),
            "convolution-dgrad-fp16-dense-stride2-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_dgrad",
                image_dimensions=[1, 128, 40, 40],
                filter_dimensions=[256, 128, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="float16",
                autotune=True,
            ),
            "matmul-bf16-descriptor-autotune": matmul_case(
                fixture,
                identity,
                a_dimensions=[2, 128, 64],
                b_dimensions=[2, 64, 128],
                data_type="bfloat16",
                autotune=True,
            ),
            "matmul-fp32-strided": matmul_case(
                fixture,
                identity,
                a_dimensions=[2, 17, 30],
                b_dimensions=[2, 30, 23],
                a_strides=[600, 31, 1],
                b_strides=[800, 1, 32],
                output_strides=[500, 25, 1],
            ),
            "convolution-fprop-1d-fp32": convolution_case(
                fixture,
                identity,
                operation="convolution_fprop",
                image_dimensions=[2, 4, 11],
                filter_dimensions=[6, 2, 3],
                pre_padding=[1],
                post_padding=[2],
                stride=[2],
                dilation=[1],
                groups=2,
            ),
            "convolution-fprop-fp16-im2col-p5-large-reduction": (
                convolution_case(
                    fixture,
                    identity,
                    operation="convolution_fprop",
                    image_dimensions=[1, 128, 40, 40],
                    filter_dimensions=[256, 128, 3, 3],
                    pre_padding=[1, 1],
                    post_padding=[1, 1],
                    stride=[2, 2],
                    dilation=[1, 1],
                    data_type="float16",
                    autotune=True,
                )
            ),
            "conv2d-fprop-bf16-im2col-asymmetric": (
                convolution_case(
                    fixture,
                    identity,
                    operation="conv2d_fprop",
                    image_dimensions=[4, 32, 35, 37],
                    filter_dimensions=[48, 32, 3, 5],
                    pre_padding=[1, 0],
                    post_padding=[1, 2],
                    stride=[1, 2],
                    dilation=[1, 1],
                    data_type="bfloat16",
                )
            ),
            "convolution-fprop-fp32-im2col-stem": convolution_case(
                fixture,
                identity,
                operation="convolution_fprop",
                image_dimensions=[1, 3, 640, 640],
                filter_dimensions=[64, 3, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                data_type="float32",
                autotune=True,
            ),
            "conv2d-fprop-fp16-strided": convolution_case(
                fixture,
                identity,
                operation="conv2d_fprop",
                image_dimensions=[2, 4, 9, 8],
                filter_dimensions=[6, 2, 3, 3],
                pre_padding=[1, 2],
                post_padding=[0, 1],
                stride=[2, 1],
                dilation=[1, 2],
                groups=2,
                data_type="float16",
                image_strides=[288, 1, 32, 4],
                filter_strides=[18, 1, 6, 2],
                result_strides=[168, 1, 42, 6],
            ),
            "convolution-fprop-3d-bf16-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_fprop",
                image_dimensions=[1, 4, 5, 6, 7],
                filter_dimensions=[8, 2, 3, 2, 3],
                pre_padding=[1, 0, 1],
                post_padding=[1, 1, 0],
                stride=[1, 2, 1],
                dilation=[1, 1, 2],
                groups=2,
                data_type="bfloat16",
                autotune=True,
            ),
            "convolution-dgrad-fp32-flipped": convolution_case(
                fixture,
                identity,
                operation="convolution_dgrad",
                image_dimensions=[2, 4, 7, 8],
                filter_dimensions=[6, 2, 3, 3],
                pre_padding=[1, 0],
                post_padding=[0, 1],
                stride=[1, 1],
                dilation=[1, 1],
                groups=2,
                convolution_mode=1,
            ),
            "convolution-wgrad-bf16-grouped": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 6, 8, 9],
                filter_dimensions=[9, 2, 2, 3],
                pre_padding=[0, 1],
                post_padding=[1, 0],
                stride=[2, 1],
                dilation=[1, 2],
                groups=3,
                data_type="bfloat16",
                convolution_mode=0,
            ),
            "composite-add-square-autotune": add_square_request,
            "convolution-wgrad-fp32-tf32": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 32, 40, 40],
                filter_dimensions=[64, 32, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                data_type="float32",
                convolution_mode=0,
                autotune=True,
            ),
            "convolution-wgrad-fp16-p5": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 32, 40, 40],
                filter_dimensions=[64, 32, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                data_type="float16",
                convolution_mode=0,
            ),
            "convolution-wgrad-bf16-p5-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 32, 40, 40],
                filter_dimensions=[64, 32, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                data_type="bfloat16",
                convolution_mode=0,
                autotune=True,
            ),
            "convolution-wgrad-fp32-stem": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 3, 640, 640],
                filter_dimensions=[16, 3, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                data_type="float32",
                convolution_mode=0,
            ),
            "convolution-wgrad-bf16-stem-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 3, 640, 640],
                filter_dimensions=[96, 3, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                data_type="bfloat16",
                convolution_mode=0,
                autotune=True,
            ),
            "convolution-wgrad-fp16-standard-stride2-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[8, 64, 56, 56],
                filter_dimensions=[128, 64, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                data_type="float16",
                convolution_mode=0,
                autotune=True,
            ),
            "convolution-wgrad-bf16-standard-1x1": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[8, 64, 28, 28],
                filter_dimensions=[128, 64, 1, 1],
                pre_padding=[0, 0],
                post_padding=[0, 0],
                stride=[1, 1],
                dilation=[1, 1],
                groups=1,
                data_type="bfloat16",
                convolution_mode=0,
            ),
            "convolution-wgrad-fp16-nd-1d-autotune": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[16, 32, 256],
                filter_dimensions=[64, 32, 3],
                pre_padding=[1],
                post_padding=[1],
                stride=[1],
                dilation=[1],
                groups=1,
                data_type="float16",
                convolution_mode=0,
                autotune=True,
            ),
            "convolution-wgrad-bf16-nd-3d-asymmetric": convolution_case(
                fixture,
                identity,
                operation="convolution_wgrad",
                image_dimensions=[1, 8, 10, 12, 14],
                filter_dimensions=[12, 8, 2, 3, 3],
                pre_padding=[1, 0, 1],
                post_padding=[0, 1, 2],
                stride=[1, 1, 1],
                dilation=[1, 1, 1],
                groups=1,
                data_type="bfloat16",
                convolution_mode=0,
            ),
            "composite-conv-bias-relu-autotune": conv_bias_relu_request,
            "layernorm-autotune": normalization_requests["layernorm"],
            "layernorm-bf16-long-row-autotune": long_row_layernorm,
            "rmsnorm-autotune": normalization_requests["rmsnorm"],
            "batchnorm-channels-last-autotune": normalization_requests[
                "batchnorm"
            ],
            "batchnorm-row-major-autotune": batchnorm_row_major,
            "batchnorm-large-nchw-autotune": large_batchnorm_row_major,
            "batchnorm-inference-channels-last-autotune": (
                normalization_requests["batchnorm_inference"]
            ),
            "batchnorm-inference-row-major-autotune": (
                batchnorm_inference_row_major
            ),
            "sdpa-autotune": attention_requests["sdpa"],
            "sdpa-backward-autotune": attention_requests["sdpa_backward"],
            "sdpa-fp8-autotune": attention_requests["sdpa_fp8"],
            "sdpa-fp8-backward-autotune": attention_requests[
                "sdpa_fp8_backward"
            ],
        }
        manifests: dict[str, bytes] = {}
        for name, request in successful_cases.items():
            manifest_bytes, manifest, _ = compile_case(
                provider, request, root, name
            )
            manifests[name] = manifest_bytes
            function = manifest["program"]["stages"][0]["function"]
            operation = request["graph"]["nodes"][0]["type"]
            if operation in {
                "sdpa",
                "sdpa_backward",
                "sdpa_fp8",
                "sdpa_fp8_backward",
            }:
                expected_attention_functions = {
                    "sdpa": ["_sdpa_fwd_kernel"],
                    "sdpa_backward": [
                        "_sdpa_bwd_dq_dbias_kernel",
                        "_sdpa_bwd_dk_kernel",
                        "_sdpa_bwd_dv_kernel",
                    ],
                    "sdpa_fp8": [
                        "_zero_sdpa_fp8_fwd_amax_kernel",
                        "_sdpa_fp8_fwd_kernel",
                    ],
                    "sdpa_fp8_backward": [
                        "_zero_sdpa_fp8_bwd_amax_kernel",
                        "_sdpa_fp8_bwd_dq_kernel",
                        "_sdpa_fp8_bwd_dkdv_kernel",
                    ],
                }[operation]
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != expected_attention_functions:
                    fail(f"{name} selected unexpected Attention stages")
                continue
            if uses_im2col_fprop(request):
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != [
                    "_conv_fprop2d_im2col_kernel",
                    "_conv_fprop2d_im2col_mm_kernel",
                ]:
                    fail(f"{name} selected unexpected im2col Fprop stages")
                continue
            if uses_dense_stride2_dgrad(request):
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != [
                    "_conv_dgrad2d_dense_pack_filter_kernel",
                    "_conv_dgrad2d_dense_pack_loss_kernel",
                    "_conv_dgrad2d_dense_mm_kernel",
                ]:
                    fail(f"{name} selected unexpected dense Dgrad stages")
                continue
            if uses_nd_packed_wgrad(request):
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != [
                    "_conv_wgrad_nd_im2row_kernel",
                    "_conv_wgrad_nd_rowmajor_kernel",
                    "_conv_wgrad_nd_reduce_kernel",
                ]:
                    fail(f"{name} selected unexpected ND packed Wgrad stages")
                continue
            if uses_standard_wgrad(request):
                node = request["graph"]["nodes"][0]
                tensor_by_uid = {
                    tensor["uid"]: tensor
                    for tensor in request["graph"]["tensors"]
                }
                output_uid = {
                    port["name"]: port["uid"] for port in node["outputs"]
                }["dw"]
                weight = tensor_by_uid[output_uid]
                expected_wgrad_functions = (
                    [
                        "_conv_wgrad2d_1x1_split_kernel",
                        "_conv_wgrad2d_stem_reduce_kernel",
                    ]
                    if weight["dimensions"][2:] == [1, 1]
                    else [
                        "_conv_wgrad2d_im2row_kernel",
                        "_conv_wgrad2d_rowmajor_kernel",
                        "_conv_wgrad2d_stem_reduce_kernel",
                    ]
                )
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != expected_wgrad_functions:
                    fail(f"{name} selected unexpected standard Wgrad stages")
                continue
            if uses_stem_wgrad(request):
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != [
                    "_conv_wgrad2d_stem_split_kernel",
                    "_conv_wgrad2d_stem_reduce_kernel",
                ]:
                    fail(f"{name} selected unexpected stem Wgrad stages")
                continue
            if uses_p5_wgrad(request):
                actual_functions = [
                    stage["function"]
                    for stage in manifest["program"]["stages"]
                ]
                if actual_functions != [
                    "_conv_wgrad2d_p5_pack_image_kernel",
                    "_conv_wgrad2d_p5_mm_kernel",
                ]:
                    fail(f"{name} selected unexpected P5 Wgrad stages")
                continue

            if operation in {
                "layernorm",
                "rmsnorm",
                "batchnorm",
                "batchnorm_inference",
            }:
                node = request["graph"]["nodes"][0]
                tensor_by_uid = {
                    tensor["uid"]: tensor
                    for tensor in request["graph"]["tensors"]
                }
                input_uids = {
                    port["name"]: port["uid"] for port in node["inputs"]
                }
                output_uids = {
                    port["name"]: port["uid"] for port in node["outputs"]
                }
                x = tensor_by_uid[input_uids["x"]]
                y = tensor_by_uid[output_uids["y"]]
                row_major = x["strides"] == dense_strides(
                    x["dimensions"]
                ) and y["strides"] == dense_strides(y["dimensions"])
                batch_block = 1 << (x["dimensions"][0] - 1).bit_length()
                expected_normalization_function = {
                    "layernorm": "layer_norm_kernel",
                    "rmsnorm": "rms_norm_kernel",
                    "batchnorm": (
                        "batch_norm_nchw_kernel"
                        if row_major and batch_block <= 256
                        else "batch_norm_kernel"
                    ),
                    "batchnorm_inference": (
                        "batch_norm_inference_nchw_kernel"
                        if row_major and math.prod(x["dimensions"][2:]) > 1
                        else "batch_norm_inference_kernel"
                    ),
                }[operation]
                if function != expected_normalization_function:
                    fail(f"{name} selected an unexpected normalization kernel")
                continue
            if request["graph"]["node_count"] == 2:
                if function != "add_square_tensor_kernel":
                    fail("AddSquare selected an unexpected fused kernel")
                continue
            if request["graph"]["node_count"] == 3:
                if function != "conv_bias_relu_2d_kernel":
                    fail("ConvBiasRelu selected an unexpected fused kernel")
                continue
            strided = name in {
                "padded-strided",
                "broadcast",
                "autotune-strided",
                "unary-relu-strided-autotune",
                "ternary-broadcast-autotune",
            }
            unary = request["graph"]["nodes"][0]["type"] in UNARY_OPERATIONS
            ternary = request["graph"]["nodes"][0]["type"] == "binary_select"
            layout = request["graph"]["nodes"][0]["type"] in {
                "reshape",
                "transpose",
                "slice",
            }
            reduction = request["graph"]["nodes"][0]["type"].startswith(
                "reduction_"
            )
            matmul = request["graph"]["nodes"][0]["type"] == "matmul"
            convolution = request["graph"]["nodes"][0]["type"] in {
                "conv2d_fprop",
                "convolution_fprop",
                "convolution_dgrad",
                "convolution_wgrad",
            }
            tensors = request["graph"]["tensors"]

            def row_major(tensor: dict[str, Any]) -> bool:
                return tensor["strides"] == dense_strides(tensor["dimensions"])

            reduction_function = ""
            if reduction:
                attributes = request["graph"]["nodes"][0]["attributes"]
                if not all(row_major(tensor) for tensor in tensors):
                    reduction_function = "reduction_strided_kernel"
                elif attributes["inner"] == 1:
                    reduction_function = "reduction_2d_kernel"
                else:
                    reduction_function = "reduction_3d_kernel"
            expected_function = (
                (
                    "conv_dgrad_nd_kernel"
                    if request["graph"]["nodes"][0]["type"]
                    == "convolution_dgrad"
                    else (
                        "conv_wgrad_nd_kernel"
                        if request["graph"]["nodes"][0]["type"]
                        == "convolution_wgrad"
                        else {
                            1: "conv1d_gemm_kernel",
                            2: "conv2d_spatial_nchw_kernel",
                            3: "conv3d_spatial_ncdhw_m_kernel",
                        }[
                            request["graph"]["nodes"][0]["attributes"][
                                "spatial_rank"
                            ]
                        ]
                    )
                )
                if convolution
                else (
                    (
                        "matmul_tle_kernel"
                        if matmul_tle_config(request) is not None
                        else (
                            "matmul_descriptor_kernel"
                            if uses_matmul_descriptor(request)
                            else "matmul_strided_kernel"
                        )
                    )
                    if matmul
                    else (
                        reduction_function
                        if reduction
                        else (
                            "reshape_contiguous_kernel"
                            if operation == "reshape"
                            and all(row_major(tensor) for tensor in tensors)
                            else (
                                "slice_copy_kernel"
                                if operation == "slice"
                                else (
                                    "transpose_physical_copy_kernel"
                                    if operation == "transpose"
                                    else (
                                        "layout_copy_kernel"
                                        if layout
                                        else (
                                            "unary_pointwise_strided_kernel"
                                            if unary and strided
                                            else (
                                                "unary_pointwise_contiguous_kernel"
                                                if unary
                                                else (
                                                    "binary_select_strided_kernel"
                                                    if ternary and strided
                                                    else (
                                                        "binary_select_tensor_kernel"
                                                        if ternary
                                                        else (
                                                            "binary_strided_kernel"
                                                            if strided
                                                            else "binary_contiguous_kernel"
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if function != expected_function:
                fail(f"{name} selected an unexpected kernel")

        duplicate_request = successful_cases["fp32-negative"]
        second_manifest, _, _ = compile_case(
            provider, duplicate_request, root, "deterministic-copy"
        )
        if second_manifest != manifests["fp32-negative"]:
            fail("mthreads manifest bytes are not deterministic")

        core_handoff = root / "core-handoff"
        core_handoff.mkdir()
        core_request_path = core_handoff / "request.json"
        core_request_bytes = write_request(
            core_request_path, duplicate_request
        )
        provider.compile_request(
            core_request_path, core_handoff, "libtriton_jit"
        )
        if (
            core_request_path.read_bytes() != core_request_bytes
            or not (core_handoff / "manifest.json").is_file()
        ):
            fail("core request.json handoff was not published atomically")

        bad_identity = add_case(fixture, "0" * 64)
        bad_identity_path = root / "bad-identity.json"
        write_request(bad_identity_path, bad_identity)
        try:
            provider.compile_request(
                bad_identity_path,
                root / "bad-identity-output",
                "libtriton_jit",
            )
        except ValueError:
            pass
        else:
            fail("mismatched compiler identity was accepted")

        nonempty = root / "nonempty"
        nonempty.mkdir()
        (nonempty / "foreign").write_text("foreign\n", encoding="utf-8")
        request_path = root / "nonempty-request.json"
        write_request(request_path, successful_cases["fp32-negative"])
        try:
            provider.compile_request(request_path, nonempty, "libtriton_jit")
        except ValueError:
            pass
        else:
            fail("preexisting nonempty artifact destination was accepted")

        installed_layout_contract(
            source_root=source_root,
            compiler=compiler,
            provider_path=provider_path,
            environment_report=environment_report,
            python=python,
            environment=environment,
            identity=identity,
            request=successful_cases["fp32-negative"],
            root=root,
        )

    print(
        json.dumps(
            {
                "backend": "mthreads",
                "identity": identity,
                "negative_cases": negative_cases,
                "successful_cases": len(successful_cases),
                "installed_layout": True,
                "torch_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
