#!/usr/bin/env python3
"""Mutation contract for the mthreads C++ artifact trust boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable


TARGET = "musa-mtgpu-cc31-w32"
MAX_LINEAR_ELEMENTS = (1 << 31) - 1 - ((1 << 16) - 1)
JsonObject = dict[str, Any]
Mutation = Callable[[JsonObject], None]
FilesystemMutation = Callable[[Path], None]


def fail(message: str) -> None:
    raise RuntimeError(message)


def encode_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: object) -> bytes:
    payload = encode_json(value)
    path.write_bytes(payload)
    return payload


def run_parser(
    executable: Path,
    request: Path,
    artifact: Path,
    *,
    target: str = TARGET,
    valid: bool,
) -> None:
    result = subprocess.run(
        [
            str(executable),
            str(request),
            str(artifact),
            target,
            "valid" if valid else "invalid",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )
    if result.returncode != 0:
        fail(
            f"artifact parser contract failed for {artifact.name}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def request_fixture(
    identity: str,
    version: str,
    *,
    autotune: bool,
    strided: bool,
    alpha: float,
    operation: str = "add",
    mode: int = 1,
    data_type: str = "float32",
    output_data_type: str | None = None,
    compute_data_type: str = "float32",
) -> JsonObject:
    strides = [32, 8, 1] if strided else [12, 4, 1]
    tensors = [
        {
            "uid": uid,
            "data_type": data_type,
            "dimensions": [2, 3, 4],
            "strides": list(strides),
            "alignment": 16,
            "virtual": False,
        }
        for uid in (100, 101, 102)
    ]
    tensors[2]["data_type"] = output_data_type or data_type
    return {
        "schema_version": 3,
        "flagdnn_version": version,
        "backend": "mthreads",
        "target": TARGET,
        "build_options": {
            "heuristic_modes": ["A"],
            "autotune": autotune,
        },
        "graph": {
            "name": "mthreads C++ artifact mutation contract",
            "tensor_count": 3,
            "tensors": tensors,
            "node_count": 1,
            "nodes": [
                {
                    "id": 7,
                    "type": operation,
                    "name": operation,
                    "compute_data_type": compute_data_type,
                    "inputs": [
                        {"name": "left", "uid": 100},
                        {"name": "right", "uid": 101},
                    ],
                    "outputs": [{"name": "output", "uid": 102}],
                    "attributes": {
                        "mode": mode,
                        "alpha": alpha,
                        "n_elements": 24,
                        "pointwise_mode": mode,
                    },
                }
            ],
        },
        "compiler_identity": identity,
    }


def add_square_request_fixture(
    identity: str,
    version: str,
    *,
    data_type: str = "float32",
    autotune: bool = False,
) -> JsonObject:
    def tensor(uid: int, *, virtual: bool) -> JsonObject:
        return {
            "uid": uid,
            "data_type": data_type,
            "dimensions": [2, 3, 4],
            "strides": [12, 4, 1],
            "alignment": 16,
            "virtual": virtual,
        }

    return {
        "schema_version": 3,
        "flagdnn_version": version,
        "backend": "mthreads",
        "target": TARGET,
        "build_options": {
            "heuristic_modes": ["A"],
            "autotune": autotune,
        },
        "graph": {
            "name": "mthreads C++ AddSquare artifact mutation contract",
            "tensor_count": 4,
            "tensors": [
                tensor(101, virtual=False),
                tensor(103, virtual=True),
                tensor(100, virtual=False),
                tensor(102, virtual=False),
            ],
            "node_count": 2,
            "nodes": [
                {
                    "id": 0,
                    "type": "mul",
                    "name": "square",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 101},
                        {"name": "right", "uid": 101},
                    ],
                    "outputs": [{"name": "output", "uid": 103}],
                    "attributes": {
                        "mode": 18,
                        "alpha": 1.0,
                        "n_elements": 24,
                        "pointwise_mode": 18,
                    },
                },
                {
                    "id": 1,
                    "type": "add",
                    "name": "add_square",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 100},
                        {"name": "right", "uid": 103},
                    ],
                    "outputs": [{"name": "output", "uid": 102}],
                    "attributes": {
                        "mode": 1,
                        "alpha": 1.0,
                        "n_elements": 24,
                        "pointwise_mode": 1,
                    },
                },
            ],
        },
        "compiler_identity": identity,
    }


def conv_bias_relu_request_fixture(
    identity: str,
    version: str,
    *,
    data_type: str = "float32",
    autotune: bool = False,
) -> JsonObject:
    def tensor(
        uid: int,
        dimensions: list[int],
        strides: list[int],
        *,
        virtual: bool,
    ) -> JsonObject:
        return {
            "uid": uid,
            "data_type": data_type,
            "dimensions": dimensions,
            "strides": strides,
            "alignment": 16,
            "virtual": virtual,
        }

    return {
        "schema_version": 3,
        "flagdnn_version": version,
        "backend": "mthreads",
        "target": TARGET,
        "build_options": {
            "heuristic_modes": ["A"],
            "autotune": autotune,
        },
        "graph": {
            "name": "mthreads C++ ConvBiasRelu artifact mutation contract",
            "tensor_count": 6,
            "tensors": [
                tensor(200, [1, 4, 5, 6], [120, 1, 24, 4], virtual=False),
                tensor(201, [6, 4, 3, 3], [36, 1, 12, 4], virtual=False),
                tensor(204, [1, 6, 5, 6], [180, 1, 36, 6], virtual=True),
                tensor(202, [1, 6, 1, 1], [6, 1, 6, 6], virtual=False),
                tensor(205, [1, 6, 5, 6], [180, 1, 36, 6], virtual=True),
                tensor(203, [1, 6, 5, 6], [180, 1, 36, 6], virtual=False),
            ],
            "node_count": 3,
            "nodes": [
                {
                    "id": 0,
                    "type": "convolution_fprop",
                    "name": "convolution",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "input", "uid": 200},
                        {"name": "filter", "uid": 201},
                    ],
                    "outputs": [{"name": "output", "uid": 204}],
                    "attributes": {
                        "dilation": [1, 1],
                        "groups": 1,
                        "n_outputs": 180,
                        "post_padding": [1, 1],
                        "pre_padding": [1, 1],
                        "spatial_rank": 2,
                        "stride": [1, 1],
                    },
                },
                {
                    "id": 1,
                    "type": "add",
                    "name": "bias_add",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 204},
                        {"name": "right", "uid": 202},
                    ],
                    "outputs": [{"name": "output", "uid": 205}],
                    "attributes": {
                        "alpha": 1.0,
                        "mode": 1,
                        "n_elements": 180,
                        "pointwise_mode": 1,
                    },
                },
                {
                    "id": 2,
                    "type": "relu",
                    "name": "relu",
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 205}],
                    "outputs": [{"name": "output", "uid": 203}],
                    "attributes": {
                        "elu_alpha": 1.0,
                        "has_upper_clip": 0,
                        "lower_clip": 0.0,
                        "mode": 2,
                        "n_elements": 180,
                        "negative_slope": 0.0,
                        "relu_lower_clip": 0.0,
                        "relu_lower_clip_slope": 0.0,
                        "relu_upper_clip": 0.0,
                        "relu_upper_clip_set": False,
                        "softplus_beta": 1.0,
                        "swish_beta": 1.0,
                        "upper_clip": 0.0,
                    },
                },
            ],
        },
        "compiler_identity": identity,
    }


def unary_request_fixture(
    identity: str,
    version: str,
    *,
    operation: str,
    mode: int,
    data_type: str = "float32",
    compute_data_type: str = "float32",
    autotune: bool = False,
    strided: bool = False,
    negative_slope: float = 0.0,
    lower_clip: float = 0.0,
    upper_clip: float = 0.0,
    has_upper_clip: bool = False,
    swish_beta: float = 1.0,
    elu_alpha: float = 1.0,
    softplus_beta: float = 1.0,
) -> JsonObject:
    request = request_fixture(
        identity,
        version,
        autotune=autotune,
        strided=strided,
        alpha=1.0,
        data_type=data_type,
        compute_data_type=compute_data_type,
    )
    input_tensor = request["graph"]["tensors"][0]
    output_tensor = request["graph"]["tensors"][2]
    request["graph"]["tensors"] = [input_tensor, output_tensor]
    request["graph"]["tensor_count"] = 2
    node = request["graph"]["nodes"][0]
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
            "n_elements": 24,
            "has_upper_clip": int(has_upper_clip),
            "negative_slope": negative_slope,
            "lower_clip": lower_clip,
            "upper_clip": upper_clip,
        },
    )
    return request


def ternary_request_fixture(
    identity: str,
    version: str,
    *,
    data_type: str = "float32",
    autotune: bool = False,
    broadcast: bool = False,
) -> JsonObject:
    request = request_fixture(
        identity,
        version,
        autotune=autotune,
        strided=False,
        alpha=1.0,
        data_type=data_type,
    )
    original = request["graph"]["tensors"]
    a = copy.deepcopy(original[0])
    b = copy.deepcopy(original[1])
    predicate = copy.deepcopy(original[2])
    output = copy.deepcopy(original[2])
    output["uid"] = 103
    dimensions = (
        ([2, 3, 4], [1, 3, 1], [2, 1, 4], [2, 3, 4])
        if broadcast
        else ([2, 3, 4],) * 4
    )

    def contiguous_strides(shape: list[int]) -> list[int]:
        result = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            result[index] = running
            running *= shape[index]
        return result

    for tensor, shape, tensor_type in (
        (a, dimensions[0], data_type),
        (b, dimensions[1], data_type),
        (predicate, dimensions[2], "boolean"),
        (output, dimensions[3], data_type),
    ):
        tensor.update(
            data_type=tensor_type,
            dimensions=list(shape),
            strides=contiguous_strides(list(shape)),
        )
    request["graph"]["tensors"] = [a, b, predicate, output]
    request["graph"]["tensor_count"] = 4
    node = request["graph"]["nodes"][0]
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
        attributes={"mode": 41, "n_elements": 24},
    )
    return request


def layout_request_fixture(
    identity: str,
    version: str,
    *,
    operation: str,
    data_type: str = "float32",
    autotune: bool = False,
) -> JsonObject:
    if operation == "reshape":
        input_dimensions = [2, 3, 4]
        output_dimensions = [6, 4]
    elif operation == "transpose":
        input_dimensions = [2, 3, 4]
        output_dimensions = [4, 2, 3]
    elif operation == "slice":
        input_dimensions = [2, 4, 5]
        output_dimensions = [2, 2, 5]
    else:
        fail(f"unknown layout fixture operation: {operation}")
    request = request_fixture(
        identity,
        version,
        autotune=autotune,
        strided=False,
        alpha=1.0,
        data_type=data_type,
    )

    def contiguous_strides(shape: list[int]) -> list[int]:
        result = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            result[index] = running
            running *= shape[index]
        return result

    original = request["graph"]["tensors"]
    input_tensor = copy.deepcopy(original[0])
    output = copy.deepcopy(original[2])
    input_strides = contiguous_strides(input_dimensions)
    if operation == "transpose":
        permutation = [2, 0, 1]
        output_strides = [input_strides[axis] for axis in permutation]
    elif operation == "slice":
        slice_strides = [1, 2, 1]
        output_strides = [
            stride * step
            for stride, step in zip(input_strides, slice_strides, strict=True)
        ]
    else:
        output_strides = contiguous_strides(output_dimensions)
    input_tensor.update(
        data_type=data_type,
        dimensions=input_dimensions,
        strides=input_strides,
    )
    output.update(
        data_type=data_type,
        dimensions=output_dimensions,
        strides=output_strides,
    )
    request["graph"]["tensors"] = [input_tensor, output]
    request["graph"]["tensor_count"] = 2
    attributes: JsonObject = {
        "n_elements": 24 if operation != "slice" else 20,
        "input_dimensions": input_dimensions,
        "input_strides": input_strides,
        "output_dimensions": output_dimensions,
        "output_strides": output_strides,
    }
    if operation == "reshape":
        attributes.update(
            input_rank=3,
            output_rank=2,
            reshape_mode=2,
        )
    elif operation == "transpose":
        attributes.update(rank=3, permutation=permutation)
    else:
        attributes.update(
            rank=3,
            starts=[0, 1, 0],
            limits=[2, 4, 5],
            slice_strides=slice_strides,
        )
    request["graph"]["nodes"][0].update(
        type=operation,
        name=operation,
        compute_data_type="float32",
        inputs=[{"name": "input", "uid": input_tensor["uid"]}],
        outputs=[{"name": "output", "uid": output["uid"]}],
        attributes=attributes,
    )
    return request


def reduction_request_fixture(
    identity: str,
    version: str,
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
) -> JsonObject:
    operations = {
        0: "reduction_sum",
        1: "reduction_avg",
        2: "reduction_mul",
    }
    if operations.get(mode) != operation:
        fail("reduction fixture operation and mode differ")
    if not input_dimensions or not 0 <= axis < len(input_dimensions):
        fail("reduction fixture rank or axis is invalid")

    def contiguous_strides(shape: list[int]) -> list[int]:
        result = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            result[index] = running
            running *= shape[index]
        return result

    request = request_fixture(
        identity,
        version,
        autotune=autotune,
        strided=False,
        alpha=1.0,
        data_type=data_type,
    )
    original = request["graph"]["tensors"]
    input_tensor = copy.deepcopy(original[0])
    output = copy.deepcopy(original[2])
    selected_input_strides = (
        list(input_strides)
        if input_strides is not None
        else contiguous_strides(input_dimensions)
    )
    output_dimensions = list(input_dimensions)
    if keep_dimensions:
        output_dimensions[axis] = 1
    else:
        del output_dimensions[axis]
    selected_output_strides = (
        list(output_strides)
        if output_strides is not None
        else contiguous_strides(output_dimensions)
    )
    input_tensor.update(
        data_type=data_type,
        dimensions=list(input_dimensions),
        strides=selected_input_strides,
    )
    output.update(
        data_type=data_type,
        dimensions=output_dimensions,
        strides=selected_output_strides,
    )
    request["graph"]["tensors"] = [input_tensor, output]
    request["graph"]["tensor_count"] = 2
    outer = math.prod(input_dimensions[:axis])
    extent = input_dimensions[axis]
    inner = math.prod(input_dimensions[axis + 1 :])
    request["graph"]["nodes"][0].update(
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
    return request


def matmul_request_fixture(
    identity: str,
    version: str,
    *,
    a_dimensions: list[int],
    b_dimensions: list[int],
    data_type: str = "float32",
    a_strides: list[int] | None = None,
    b_strides: list[int] | None = None,
    output_strides: list[int] | None = None,
    autotune: bool = False,
) -> JsonObject:
    if len(a_dimensions) < 2 or len(b_dimensions) < 2:
        fail("Matmul fixture tensors must have rank at least two")
    if a_dimensions[-1] != b_dimensions[-2]:
        fail("Matmul fixture contraction dimensions differ")

    def contiguous_strides(shape: list[int]) -> list[int]:
        result = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            result[index] = running
            running *= shape[index]
        return result

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
            fail("Matmul fixture batch dimensions do not broadcast")
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    output_dimensions = [
        *batch_dimensions,
        a_dimensions[-2],
        b_dimensions[-1],
    ]
    request = request_fixture(
        identity,
        version,
        autotune=autotune,
        strided=False,
        alpha=1.0,
        data_type=data_type,
    )
    for tensor, dimensions, strides in (
        (request["graph"]["tensors"][0], a_dimensions, a_strides),
        (request["graph"]["tensors"][1], b_dimensions, b_strides),
        (
            request["graph"]["tensors"][2],
            output_dimensions,
            output_strides,
        ),
    ):
        tensor.update(
            data_type=data_type,
            dimensions=list(dimensions),
            strides=(
                list(strides)
                if strides is not None
                else contiguous_strides(dimensions)
            ),
        )
    request["graph"]["nodes"][0].update(
        type="matmul",
        name="matmul",
        compute_data_type="float32",
        inputs=[
            {"name": "a", "uid": 100},
            {"name": "b", "uid": 101},
        ],
        outputs=[{"name": "output", "uid": 102}],
        attributes={
            "batch": math.prod(batch_dimensions),
            "m": a_dimensions[-2],
            "n": b_dimensions[-1],
            "k": a_dimensions[-1],
        },
    )
    return request


def convolution_request_fixture(
    identity: str,
    version: str,
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
) -> JsonObject:
    if operation not in {
        "conv2d_fprop",
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    }:
        fail("unknown convolution fixture operation")
    spatial_rank = len(image_dimensions) - 2
    if (
        spatial_rank not in {1, 2, 3}
        or len(filter_dimensions) != spatial_rank + 2
        or any(
            len(values) != spatial_rank
            for values in (pre_padding, post_padding, stride, dilation)
        )
    ):
        fail("convolution fixture spatial metadata is invalid")

    def contiguous_strides(shape: list[int]) -> list[int]:
        result = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            result[index] = running
            running *= shape[index]
        return result

    result_dimensions = [image_dimensions[0], filter_dimensions[0]]
    for axis in range(spatial_rank):
        effective = (filter_dimensions[axis + 2] - 1) * dilation[axis] + 1
        padded = (
            image_dimensions[axis + 2] + pre_padding[axis] + post_padding[axis]
        )
        if padded < effective:
            fail("convolution fixture filter exceeds padded input")
        result_dimensions.append((padded - effective) // stride[axis] + 1)

    request = request_fixture(
        identity,
        version,
        autotune=autotune,
        strided=False,
        alpha=1.0,
        data_type=data_type,
    )
    image, filter_tensor, result = request["graph"]["tensors"]
    for tensor, dimensions, selected_strides in (
        (image, image_dimensions, image_strides),
        (filter_tensor, filter_dimensions, filter_strides),
        (result, result_dimensions, result_strides),
    ):
        tensor.update(
            data_type=data_type,
            dimensions=list(dimensions),
            strides=(
                list(selected_strides)
                if selected_strides is not None
                else contiguous_strides(dimensions)
            ),
        )
    attributes: JsonObject = {
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
    request["graph"]["nodes"][0].update(
        type=operation,
        name=operation,
        compute_data_type="float32",
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
    )
    return request


def normalization_request_fixture(
    identity: str,
    version: str,
    *,
    operation: str,
    data_type: str = "float32",
    autotune: bool = False,
    row_major: bool = False,
) -> JsonObject:
    if operation not in {
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "batchnorm_inference",
    }:
        fail(f"unknown normalization fixture operation: {operation}")

    def tensor(
        uid: int,
        tensor_type: str,
        dimensions: list[int],
        strides: list[int],
    ) -> JsonObject:
        return {
            "uid": uid,
            "data_type": tensor_type,
            "dimensions": list(dimensions),
            "strides": list(strides),
            "alignment": 16,
            "virtual": False,
        }

    if operation in {"layernorm", "rmsnorm"}:
        base_uid = 300 if operation == "layernorm" else 310
        tensors = [
            tensor(base_uid, data_type, [2, 3, 4], [12, 4, 1]),
            tensor(base_uid + 1, data_type, [1, 1, 4], [4, 4, 1]),
            tensor(base_uid + 2, data_type, [1, 1, 4], [4, 4, 1]),
            tensor(base_uid + 3, data_type, [2, 3, 4], [12, 4, 1]),
        ]
        inputs = [
            {"name": "x", "uid": base_uid},
            {"name": "scale", "uid": base_uid + 1},
            {"name": "bias", "uid": base_uid + 2},
        ]
        outputs = [{"name": "y", "uid": base_uid + 3}]
        if operation == "layernorm":
            tensors.extend(
                [
                    tensor(base_uid + 4, "float32", [2, 3, 1], [3, 1, 1]),
                    tensor(base_uid + 5, "float32", [2, 3, 1], [3, 1, 1]),
                ]
            )
            outputs.extend(
                [
                    {"name": "mean", "uid": base_uid + 4},
                    {"name": "inv_variance", "uid": base_uid + 5},
                ]
            )
        else:
            tensors.append(
                tensor(base_uid + 4, "float32", [2, 3, 1], [3, 1, 1])
            )
            outputs.append({"name": "inv_variance", "uid": base_uid + 4})
        attributes: JsonObject = {
            "epsilon": 0.0010000000474974513,
            "forward_phase": 2,
            "normalized_elements": 4,
            "rows": 6,
        }
    else:
        inference = operation == "batchnorm_inference"
        base_uid = 340 if inference else 320
        dimensions = [2, 4, 3, 5]
        x_strides = [60, 15, 5, 1] if row_major else [60, 1, 20, 4]
        parameter_dimensions = [1, 4, 1, 1]
        parameter_strides = [4, 1, 1, 1]
        tensors = [
            tensor(base_uid, data_type, dimensions, x_strides),
        ]
        if inference:
            tensors.extend(
                tensor(
                    base_uid + offset,
                    "float32",
                    parameter_dimensions,
                    parameter_strides,
                )
                for offset in range(1, 5)
            )
            tensors.append(
                tensor(base_uid + 5, data_type, dimensions, x_strides)
            )
            inputs = [
                {"name": "x", "uid": base_uid},
                {"name": "mean", "uid": base_uid + 1},
                {"name": "inv_variance", "uid": base_uid + 2},
                {"name": "scale", "uid": base_uid + 3},
                {"name": "bias", "uid": base_uid + 4},
            ]
            outputs = [{"name": "y", "uid": base_uid + 5}]
        else:
            tensors.extend(
                tensor(
                    base_uid + offset,
                    data_type if offset in {1, 2} else "float32",
                    parameter_dimensions,
                    parameter_strides,
                )
                for offset in range(1, 5)
            )
            tensors.append(
                tensor(base_uid + 5, data_type, dimensions, x_strides)
            )
            tensors.extend(
                tensor(
                    base_uid + offset,
                    "float32",
                    parameter_dimensions,
                    parameter_strides,
                )
                for offset in range(6, 10)
            )
            inputs = [
                {"name": "x", "uid": base_uid},
                {"name": "scale", "uid": base_uid + 1},
                {"name": "bias", "uid": base_uid + 2},
                {"name": "previous_running_mean", "uid": base_uid + 3},
                {
                    "name": "previous_running_variance",
                    "uid": base_uid + 4,
                },
            ]
            outputs = [
                {"name": "y", "uid": base_uid + 5},
                {"name": "mean", "uid": base_uid + 6},
                {"name": "inv_variance", "uid": base_uid + 7},
                {"name": "next_running_mean", "uid": base_uid + 8},
                {
                    "name": "next_running_variance",
                    "uid": base_uid + 9,
                },
            ]
        attributes = {
            "channels": 4,
            "dimensions": dimensions,
            "n_elements": 120,
            "rank": 4,
            "spatial": 15,
            "x_strides": x_strides,
            "y_strides": x_strides,
        }
        if not inference:
            attributes.update(
                batch=2,
                epsilon=0.0010000000474974513,
                momentum=0.10000000149011612,
            )

    return {
        "schema_version": 3,
        "flagdnn_version": version,
        "backend": "mthreads",
        "target": TARGET,
        "build_options": {
            "heuristic_modes": ["A"],
            "autotune": autotune,
        },
        "graph": {
            "name": f"mthreads C++ {operation} artifact mutation contract",
            "tensor_count": len(tensors),
            "tensors": tensors,
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "name": operation,
                    "compute_data_type": "float32",
                    "inputs": inputs,
                    "outputs": outputs,
                    "attributes": attributes,
                }
            ],
        },
        "compiler_identity": identity,
    }


def set_path(
    root: JsonObject, path: tuple[str | int, ...], value: Any
) -> None:
    current: Any = root
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value


def delete_path(root: JsonObject, path: tuple[str | int, ...]) -> None:
    current: Any = root
    for component in path[:-1]:
        current = current[component]
    del current[path[-1]]


def add_key(
    root: JsonObject,
    path: tuple[str | int, ...],
    key: str,
    value: Any,
) -> None:
    current: Any = root
    for component in path:
        current = current[component]
    current[key] = value


def append_path(
    root: JsonObject, path: tuple[str | int, ...], value: Any
) -> None:
    current: Any = root
    for component in path:
        current = current[component]
    current.append(value)


class Matrix:
    def __init__(
        self,
        root: Path,
        executable: Path,
        baseline_request: Path,
        baseline_artifact: Path,
    ) -> None:
        self.root = root
        self.executable = executable
        self.baseline_request = baseline_request
        self.baseline_artifact = baseline_artifact
        self.count = 0

    def reject(
        self,
        label: str,
        *,
        manifest_mutation: Mutation | None = None,
        request_mutation: Mutation | None = None,
        filesystem_mutation: FilesystemMutation | None = None,
        raw_manifest: Callable[[bytes], bytes] | None = None,
        raw_request: Callable[[bytes], bytes] | None = None,
        target: str = TARGET,
        artifact_override: Callable[[Path, Path], Path] | None = None,
    ) -> None:
        self.count += 1
        slug = f"{self.count:03d}-{label.replace(' ', '-')}"
        artifact = self.root / f"{slug}-artifact"
        request_path = self.root / f"{slug}-request.json"
        shutil.copytree(self.baseline_artifact, artifact)
        request_bytes = self.baseline_request.read_bytes()
        request_value = json.loads(request_bytes)
        if request_mutation is not None:
            request_mutation(request_value)
            request_bytes = encode_json(request_value)
        if raw_request is not None:
            request_bytes = raw_request(request_bytes)
        request_path.write_bytes(request_bytes)

        manifest_path = artifact / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if request_mutation is not None or raw_request is not None:
            manifest["request_sha256"] = hashlib.sha256(
                request_bytes
            ).hexdigest()
        if manifest_mutation is not None:
            manifest_mutation(manifest)
        manifest_bytes = encode_json(manifest)
        if raw_manifest is not None:
            manifest_bytes = raw_manifest(manifest_bytes)
        manifest_path.write_bytes(manifest_bytes)
        if filesystem_mutation is not None:
            filesystem_mutation(artifact)
        selected_artifact = (
            artifact_override(self.root, artifact)
            if artifact_override is not None
            else artifact
        )
        run_parser(
            self.executable,
            request_path,
            selected_artifact,
            target=target,
            valid=False,
        )


def compile_fixture(
    provider: Any,
    root: Path,
    name: str,
    request: JsonObject,
) -> tuple[Path, Path]:
    request_path = root / f"{name}-request.json"
    write_json(request_path, request)
    artifact = root / f"{name}-artifact"
    result = provider.compile_request(request_path, artifact, "libtriton_jit")
    if (
        result.get("status") != "success"
        or result.get("torch_loaded") is not False
    ):
        fail("mthreads provider did not produce a clean fixture artifact")
    return request_path, artifact


def capture_attention_request_fixture(
    *,
    executable: Path,
    capture_compiler: Path,
    python: Path,
    root: Path,
    graph_kind: str,
    identity: str,
) -> JsonObject:
    if graph_kind not in {
        "sdpa",
        "sdpa_backward",
        "sdpa_fp8",
        "sdpa_fp8_backward",
    }:
        fail(f"unknown Attention capture kind: {graph_kind}")
    capture = root / f"captured-{graph_kind}-request.json"
    environment = dict(os.environ)
    environment["FLAGDNN_MTHREADS_CAPTURE_REQUEST"] = str(capture)
    environment["FLAGDNN_MTHREADS_CAPTURE_GRAPH"] = graph_kind
    result = subprocess.run(
        [
            str(executable),
            str(python),
            str(capture_compiler),
            str(root / f"capture-{graph_kind}-cache"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if result.returncode != 0 or not capture.is_file():
        fail(
            f"public Graph capture failed for {graph_kind}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    request = json.loads(capture.read_bytes())
    graph = request.get("graph")
    if (
        not isinstance(graph, dict)
        or graph.get("node_count") != 1
        or graph.get("nodes", [{}])[0].get("type") != graph_kind
        or request.get("backend") != "mthreads"
        or request.get("target") != TARGET
    ):
        fail(f"captured public {graph_kind} Graph request differs")
    request["compiler_identity"] = identity
    request["build_options"]["autotune"] = True
    return request


def manifest_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "manifest unknown root key",
            lambda value: add_key(value, (), "unknown", 1),
        ),
        (
            "manifest missing backend",
            lambda value: delete_path(value, ("backend",)),
        ),
        (
            "manifest schema",
            lambda value: set_path(value, ("schema_version",), 2),
        ),
        (
            "manifest artifact kind",
            lambda value: set_path(value, ("artifact_kind",), "source"),
        ),
        (
            "manifest version",
            lambda value: set_path(value, ("flagdnn_version",), "9.9.9"),
        ),
        (
            "manifest backend",
            lambda value: set_path(value, ("backend",), "nvidia"),
        ),
        (
            "manifest target",
            lambda value: set_path(value, ("target",), "musa-mtgpu-cc32-w32"),
        ),
        (
            "manifest engine",
            lambda value: set_path(value, ("engine",), "external_artifact"),
        ),
        (
            "manifest request hash",
            lambda value: set_path(value, ("request_sha256",), "0" * 64),
        ),
        (
            "manifest compiler identity",
            lambda value: set_path(value, ("compiler_identity",), "0" * 64),
        ),
        (
            "manifest source hash",
            lambda value: set_path(value, ("source_sha256",), "0" * 64),
        ),
        (
            "workspace size",
            lambda value: set_path(value, ("workspace_size",), 8192),
        ),
        (
            "workspace alignment",
            lambda value: set_path(value, ("workspace_alignment",), 512),
        ),
        (
            "workspace non power alignment",
            lambda value: set_path(value, ("workspace_alignment",), 384),
        ),
        (
            "binding order",
            lambda value: set_path(
                value, ("external_binding_uids",), [101, 100, 102]
            ),
        ),
        (
            "binding unknown uid",
            lambda value: set_path(value, ("external_binding_uids", 2), 999),
        ),
        (
            "file count",
            lambda value: append_path(
                value, ("files",), copy.deepcopy(value["files"][0])
            ),
        ),
        (
            "file unknown key",
            lambda value: add_key(value, ("files", 0), "mode", "source"),
        ),
        (
            "file path",
            lambda value: set_path(
                value, ("files", 0, "path"), "../binary.py"
            ),
        ),
        (
            "file size",
            lambda value: set_path(value, ("files", 0, "size"), 1),
        ),
        (
            "file hash",
            lambda value: set_path(value, ("files", 0, "sha256"), "0" * 64),
        ),
        (
            "program unknown key",
            lambda value: add_key(value, ("program",), "unknown", 1),
        ),
        (
            "program schema",
            lambda value: set_path(value, ("program", "schema_version"), 2),
        ),
        (
            "program stage count",
            lambda value: set_path(value, ("program", "stage_count"), 2),
        ),
        (
            "stage unknown key",
            lambda value: add_key(
                value, ("program", "stages", 0), "unknown", 1
            ),
        ),
        (
            "stage id",
            lambda value: set_path(value, ("program", "stages", 0, "id"), 1),
        ),
        (
            "stage node id",
            lambda value: set_path(
                value, ("program", "stages", 0, "node_id"), 9
            ),
        ),
        (
            "stage operation",
            lambda value: set_path(
                value, ("program", "stages", 0, "operation"), "mul"
            ),
        ),
        (
            "stage dependency",
            lambda value: set_path(
                value, ("program", "stages", 0, "dependencies"), [0]
            ),
        ),
        (
            "stage source",
            lambda value: set_path(
                value, ("program", "stages", 0, "source"), "../binary.py"
            ),
        ),
        (
            "stage function",
            lambda value: set_path(
                value, ("program", "stages", 0, "function"), "evil"
            ),
        ),
        (
            "autotune unknown key",
            lambda value: add_key(
                value,
                ("program", "stages", 0, "autotune"),
                "unknown",
                1,
            ),
        ),
        (
            "autotune enabled",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "autotune", "enabled"),
                True,
            ),
        ),
        (
            "autotune warmup",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "autotune", "warmup"),
                4,
            ),
        ),
        (
            "autotune repetitions",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "autotune", "repetitions"),
                11,
            ),
        ),
        (
            "autotune cache path",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "autotune",
                    "selection_cache",
                ),
                "../cache.json",
            ),
        ),
        (
            "variant count",
            lambda value: append_path(
                value,
                ("program", "stages", 0, "variants"),
                copy.deepcopy(value["program"]["stages"][0]["variants"][0]),
            ),
        ),
        (
            "variant unknown key",
            lambda value: add_key(
                value,
                ("program", "stages", 0, "variants", 0),
                "unknown",
                1,
            ),
        ),
        (
            "variant id",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "variant_id"),
                "block-1-warps-1-stages-1",
            ),
        ),
        (
            "variant source",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "source"),
                "../binary.py",
            ),
        ),
        (
            "variant source hash",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "source_sha256",
                ),
                "0" * 64,
            ),
        ),
        (
            "variant function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "function"),
                "evil",
            ),
        ),
        (
            "variant signature",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "full_signature",
                ),
                "*fp32",
            ),
        ),
        (
            "variant grid rank",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "grid"),
                [1, 1],
            ),
        ),
        (
            "variant grid value",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "grid", 0),
                2,
            ),
        ),
        (
            "variant warps",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "num_warps"),
                8,
            ),
        ),
        (
            "variant stages",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "num_stages"),
                2,
            ),
        ),
        (
            "argument count",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                ),
                [],
            ),
        ),
        (
            "argument unknown key",
            lambda value: add_key(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    0,
                ),
                "unknown",
                1,
            ),
        ),
        (
            "argument kind",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    0,
                    "kind",
                ),
                "scalar_i32",
            ),
        ),
        (
            "argument semantic",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    0,
                    "semantic_name",
                ),
                "right",
            ),
        ),
        (
            "argument uid",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    0,
                    "uid",
                ),
                101,
            ),
        ),
        (
            "tensor argument scalar bits",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    0,
                    "scalar_bits",
                ),
                "00000000",
            ),
        ),
        (
            "scalar argument uid",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    3,
                    "uid",
                ),
                24,
            ),
        ),
        (
            "scalar argument bits",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    3,
                    "scalar_bits",
                ),
                "00000018",
            ),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def request_matrix(matrix: Matrix) -> None:
    def linear_index_tail_overflow(value: JsonObject) -> None:
        extent = MAX_LINEAR_ELEMENTS + 1
        for tensor in value["graph"]["tensors"]:
            tensor["dimensions"] = [extent]
            tensor["strides"] = [1]
        value["graph"]["nodes"][0]["attributes"]["n_elements"] = extent

    mutations: list[tuple[str, Mutation]] = [
        (
            "request unknown root key",
            lambda value: add_key(value, (), "unknown", 1),
        ),
        (
            "request schema",
            lambda value: set_path(value, ("schema_version",), 4),
        ),
        (
            "request backend",
            lambda value: set_path(value, ("backend",), "nvidia"),
        ),
        (
            "request target",
            lambda value: set_path(value, ("target",), "musa-mtgpu-cc32-w32"),
        ),
        (
            "request compiler identity",
            lambda value: set_path(value, ("compiler_identity",), "x" * 64),
        ),
        (
            "request options unknown key",
            lambda value: add_key(value, ("build_options",), "unknown", 1),
        ),
        (
            "request empty heuristic",
            lambda value: set_path(
                value, ("build_options", "heuristic_modes"), []
            ),
        ),
        (
            "request duplicate heuristic",
            lambda value: set_path(
                value, ("build_options", "heuristic_modes"), ["A", "A"]
            ),
        ),
        (
            "request non boolean autotune",
            lambda value: set_path(value, ("build_options", "autotune"), 1),
        ),
        (
            "request graph unknown key",
            lambda value: add_key(value, ("graph",), "unknown", 1),
        ),
        (
            "request tensor count",
            lambda value: set_path(value, ("graph", "tensor_count"), 2),
        ),
        (
            "request duplicate tensor uid",
            lambda value: set_path(value, ("graph", "tensors", 1, "uid"), 100),
        ),
        (
            "request tensor unknown key",
            lambda value: add_key(
                value, ("graph", "tensors", 0), "unknown", 1
            ),
        ),
        (
            "request unsupported dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "data_type"), "int32"
            ),
        ),
        (
            "request zero rank",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "dimensions"), []
            ),
        ),
        (
            "request overlapping strides",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "strides"), [1, 1, 1]
            ),
        ),
        (
            "request bad alignment",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "alignment"), 3
            ),
        ),
        (
            "request virtual tensor",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "virtual"), True
            ),
        ),
        (
            "request node count",
            lambda value: set_path(value, ("graph", "node_count"), 2),
        ),
        (
            "request node unknown key",
            lambda value: add_key(value, ("graph", "nodes", 0), "unknown", 1),
        ),
        (
            "request negative node id",
            lambda value: set_path(value, ("graph", "nodes", 0, "id"), -1),
        ),
        (
            "request node operation",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "type"), "mul"
            ),
        ),
        (
            "request compute dtype",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "compute_data_type"),
                "float16",
            ),
        ),
        (
            "request optional port",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "inputs", 0),
                "optional",
                True,
            ),
        ),
        (
            "request wrong port role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 0, "name"),
                "x",
            ),
        ),
        (
            "request duplicate tensor role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 1, "uid"),
                100,
            ),
        ),
        (
            "request attributes unknown key",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "BLOCK_SIZE",
                4096,
            ),
        ),
        (
            "request Add mode",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "mode"),
                2,
            ),
        ),
        (
            "request element count",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_elements"),
                23,
            ),
        ),
        (
            "request linear index tail overflow",
            linear_index_tail_overflow,
        ),
        (
            "request alpha float overflow",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "alpha"),
                1.0e100,
            ),
        ),
        (
            "request broadcast output",
            lambda value: set_path(
                value,
                ("graph", "tensors", 2, "dimensions"),
                [2, 3, 5],
            ),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def ternary_request_matrix(matrix: Matrix) -> None:
    def wrong_output_shape(value: JsonObject) -> None:
        set_path(
            value,
            ("graph", "tensors", 3, "dimensions"),
            [2, 3, 5],
        )
        set_path(
            value,
            ("graph", "tensors", 3, "strides"),
            [15, 5, 1],
        )
        set_path(
            value,
            ("graph", "nodes", 0, "attributes", "n_elements"),
            30,
        )

    def incompatible_broadcast(value: JsonObject) -> None:
        set_path(
            value,
            ("graph", "tensors", 1, "dimensions"),
            [2, 5, 4],
        )
        set_path(
            value,
            ("graph", "tensors", 1, "strides"),
            [20, 4, 1],
        )

    def unexpected_fifth_tensor(value: JsonObject) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][3])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 5

    mutations: list[tuple[str, Mutation]] = [
        (
            "ternary missing predicate input",
            lambda value: delete_path(
                value, ("graph", "nodes", 0, "inputs", 2)
            ),
        ),
        (
            "ternary wrong input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 2, "name"),
                "condition",
            ),
        ),
        (
            "ternary duplicate input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 1, "name"),
                "a",
            ),
        ),
        (
            "ternary wrong output role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "outputs", 0, "name"),
                "result",
            ),
        ),
        (
            "ternary duplicate tensor binding",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 2, "uid"),
                100,
            ),
        ),
        (
            "ternary floating predicate",
            lambda value: set_path(
                value,
                ("graph", "tensors", 2, "data_type"),
                "float32",
            ),
        ),
        (
            "ternary nonfloating value",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "data_type"),
                "boolean",
            ),
        ),
        (
            "ternary mismatched right dtype",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "data_type"),
                "float16",
            ),
        ),
        (
            "ternary mismatched output dtype",
            lambda value: set_path(
                value,
                ("graph", "tensors", 3, "data_type"),
                "float16",
            ),
        ),
        (
            "ternary wrong compute dtype",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "compute_data_type"),
                "float16",
            ),
        ),
        ("ternary wrong output shape", wrong_output_shape),
        ("ternary incompatible broadcast", incompatible_broadcast),
        (
            "ternary wrong mode",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "mode"),
                1,
            ),
        ),
        (
            "ternary wrong operation",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "type"), "add"
            ),
        ),
        (
            "ternary unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "alpha",
                1.0,
            ),
        ),
        (
            "ternary element count mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_elements"),
                23,
            ),
        ),
        (
            "ternary boolean element count",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_elements"),
                True,
            ),
        ),
        (
            "ternary virtual predicate",
            lambda value: set_path(
                value, ("graph", "tensors", 2, "virtual"), True
            ),
        ),
        ("ternary unexpected fifth tensor", unexpected_fifth_tensor),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def layout_request_matrix(matrix: Matrix) -> None:
    reshape = json.loads(matrix.baseline_request.read_text(encoding="utf-8"))

    def replace_request(value: JsonObject, replacement: JsonObject) -> None:
        value.clear()
        value.update(copy.deepcopy(replacement))

    identity = reshape["compiler_identity"]
    version = reshape["flagdnn_version"]
    transpose = layout_request_fixture(
        identity,
        version,
        operation="transpose",
    )
    sliced = layout_request_fixture(
        identity,
        version,
        operation="slice",
    )

    mutations: list[tuple[str, Mutation]] = [
        (
            "layout wrong input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 0, "name"),
                "x",
            ),
        ),
        (
            "layout mismatched dtype",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "data_type"),
                "float16",
            ),
        ),
        (
            "layout wrong compute dtype",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "compute_data_type"),
                "float16",
            ),
        ),
        (
            "reshape view-only mode",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "reshape_mode"),
                1,
            ),
        ),
        (
            "reshape input rank mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "input_rank"),
                2,
            ),
        ),
        (
            "layout element count mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_elements"),
                23,
            ),
        ),
        (
            "layout duplicated input dimensions mismatch",
            lambda value: set_path(
                value,
                (
                    "graph",
                    "nodes",
                    0,
                    "attributes",
                    "input_dimensions",
                ),
                [2, 2, 6],
            ),
        ),
        (
            "transpose duplicate permutation axis",
            lambda value: (
                replace_request(value, transpose),
                set_path(
                    value,
                    (
                        "graph",
                        "nodes",
                        0,
                        "attributes",
                        "permutation",
                    ),
                    [2, 0, 0],
                ),
            ),
        ),
        (
            "transpose output shape mismatch",
            lambda value: (
                replace_request(value, transpose),
                set_path(
                    value,
                    ("graph", "tensors", 1, "dimensions"),
                    [4, 3, 2],
                ),
                set_path(
                    value,
                    ("graph", "tensors", 1, "strides"),
                    [1, 4, 12],
                ),
            ),
        ),
        (
            "transpose missing permutation",
            lambda value: (
                replace_request(value, transpose),
                delete_path(
                    value,
                    (
                        "graph",
                        "nodes",
                        0,
                        "attributes",
                        "permutation",
                    ),
                ),
            ),
        ),
        (
            "slice negative start",
            lambda value: (
                replace_request(value, sliced),
                set_path(
                    value,
                    ("graph", "nodes", 0, "attributes", "starts"),
                    [0, -1, 0],
                ),
            ),
        ),
        (
            "slice limit exceeds input",
            lambda value: (
                replace_request(value, sliced),
                set_path(
                    value,
                    ("graph", "nodes", 0, "attributes", "limits"),
                    [2, 5, 5],
                ),
            ),
        ),
        (
            "slice zero stride",
            lambda value: (
                replace_request(value, sliced),
                set_path(
                    value,
                    (
                        "graph",
                        "nodes",
                        0,
                        "attributes",
                        "slice_strides",
                    ),
                    [1, 0, 1],
                ),
            ),
        ),
        (
            "layout virtual input",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "virtual"),
                True,
            ),
        ),
        (
            "layout unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "block_size",
                256,
            ),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def reduction_request_matrix(matrix: Matrix) -> None:
    def boolean_storage(value: JsonObject) -> None:
        for tensor in value["graph"]["tensors"]:
            tensor["data_type"] = "boolean"

    def unexpected_tensor(value: JsonObject) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][1])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 3

    mutations: list[tuple[str, Mutation]] = [
        (
            "reduction wrong input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 0, "name"),
                "x",
            ),
        ),
        (
            "reduction mismatched dtype",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "data_type"),
                "bfloat16",
            ),
        ),
        ("reduction boolean storage", boolean_storage),
        (
            "reduction wrong compute dtype",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "compute_data_type"),
                "float16",
            ),
        ),
        (
            "reduction mode operation mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "mode"),
                0,
            ),
        ),
        (
            "reduction negative axis",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "axis"),
                -1,
            ),
        ),
        (
            "reduction boolean keep dimensions",
            lambda value: set_path(
                value,
                (
                    "graph",
                    "nodes",
                    0,
                    "attributes",
                    "keep_dimensions",
                ),
                True,
            ),
        ),
        (
            "reduction wrong output shape",
            lambda value: (
                set_path(
                    value,
                    ("graph", "tensors", 1, "dimensions"),
                    [2, 8],
                ),
                set_path(
                    value,
                    ("graph", "tensors", 1, "strides"),
                    [8, 1],
                ),
            ),
        ),
        (
            "reduction outer mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "outer"),
                3,
            ),
        ),
        (
            "reduction extent mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "reduction"),
                5,
            ),
        ),
        (
            "reduction inner mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "inner"),
                7,
            ),
        ),
        (
            "reduction output elements mismatch",
            lambda value: set_path(
                value,
                (
                    "graph",
                    "nodes",
                    0,
                    "attributes",
                    "output_elements",
                ),
                15,
            ),
        ),
        (
            "reduction unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "block_n",
                8,
            ),
        ),
        (
            "reduction virtual input",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "virtual"),
                True,
            ),
        ),
        ("reduction unexpected tensor", unexpected_tensor),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def reduction_manifest_matrix(matrix: Matrix) -> None:
    variant = ("program", "stages", 0, "variants", 0)
    mutations: list[tuple[str, Mutation]] = [
        (
            "reduction stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "reduction_2d_kernel",
            ),
        ),
        (
            "reduction variant function",
            lambda value: set_path(
                value,
                variant + ("function",),
                "reduction_strided_kernel",
            ),
        ),
        (
            "reduction variant signature",
            lambda value: set_path(
                value,
                variant + ("full_signature",),
                "*fp16:16,*fp16:16,i32,4,8,32,8,1,2,1,4,extra",
            ),
        ),
        (
            "reduction variant grid",
            lambda value: set_path(
                value,
                variant + ("grid", 0),
                17,
            ),
        ),
        (
            "reduction candidate id",
            lambda value: set_path(
                value,
                variant + ("variant_id",),
                "block-2-warps-2-stages-1",
            ),
        ),
        (
            "reduction scalar semantic",
            lambda value: set_path(
                value,
                variant + ("arguments", 2, "semantic_name"),
                "outer",
            ),
        ),
        (
            "reduction scalar bits",
            lambda value: set_path(
                value,
                variant + ("arguments", 2, "scalar_bits"),
                "0f000000",
            ),
        ),
        (
            "reduction autotune candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def matmul_request_matrix(matrix: Matrix) -> None:
    def boolean_storage(value: JsonObject) -> None:
        for tensor in value["graph"]["tensors"]:
            tensor["data_type"] = "boolean"

    def unexpected_tensor(value: JsonObject) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][2])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 4

    mutations: list[tuple[str, Mutation]] = [
        (
            "Matmul wrong input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 0, "name"),
                "left",
            ),
        ),
        (
            "Matmul mismatched dtype",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "data_type"),
                "float32",
            ),
        ),
        ("Matmul boolean storage", boolean_storage),
        (
            "Matmul wrong compute dtype",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "compute_data_type"),
                "float16",
            ),
        ),
        (
            "Matmul contraction mismatch",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "dimensions"),
                [3, 31, 23],
            ),
        ),
        (
            "Matmul nonbroadcast batch",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "dimensions"),
                [2, 2, 17, 30],
            ),
        ),
        (
            "Matmul output shape mismatch",
            lambda value: set_path(
                value,
                ("graph", "tensors", 2, "dimensions"),
                [2, 3, 17, 24],
            ),
        ),
        *[
            (
                f"Matmul {attribute} attribute mismatch",
                lambda value, name=attribute: set_path(
                    value,
                    ("graph", "nodes", 0, "attributes", name),
                    value["graph"]["nodes"][0]["attributes"][name] + 1,
                ),
            )
            for attribute in ("batch", "m", "n", "k")
        ],
        (
            "Matmul unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "block_m",
                64,
            ),
        ),
        (
            "Matmul virtual input",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "virtual"),
                True,
            ),
        ),
        (
            "Matmul optional port",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "inputs", 0),
                "optional",
                True,
            ),
        ),
        ("Matmul unexpected tensor", unexpected_tensor),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def matmul_manifest_matrix(matrix: Matrix) -> None:
    variant = ("program", "stages", 0, "variants", 0)
    mutations: list[tuple[str, Mutation]] = [
        (
            "Matmul stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "binary_contiguous_kernel",
            ),
        ),
        (
            "Matmul variant function",
            lambda value: set_path(
                value,
                variant + ("function",),
                "binary_strided_kernel",
            ),
        ),
        (
            "Matmul variant signature",
            lambda value: set_path(
                value,
                variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "Matmul x grid",
            lambda value: set_path(
                value,
                variant + ("grid", 0),
                value["program"]["stages"][0]["variants"][0]["grid"][0] + 1,
            ),
        ),
        (
            "Matmul batch grid",
            lambda value: set_path(
                value,
                variant + ("grid", 1),
                value["program"]["stages"][0]["variants"][0]["grid"][1] + 1,
            ),
        ),
        (
            "Matmul candidate id",
            lambda value: set_path(
                value,
                variant + ("variant_id",),
                "block-16-warps-4-stages-1",
            ),
        ),
        (
            "Matmul candidate warps",
            lambda value: set_path(value, variant + ("num_warps",), 2),
        ),
        (
            "Matmul argument semantic",
            lambda value: set_path(
                value,
                variant + ("arguments", 0, "semantic_name"),
                "left",
            ),
        ),
        (
            "Matmul argument uid",
            lambda value: set_path(
                value,
                variant + ("arguments", 1, "uid"),
                999,
            ),
        ),
        (
            "Matmul argument scalar bits",
            lambda value: set_path(
                value,
                variant + ("arguments", 2, "scalar_bits"),
                "00000000",
            ),
        ),
        (
            "Matmul autotune candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def convolution_request_matrix(matrix: Matrix) -> None:
    def boolean_storage(value: JsonObject) -> None:
        for tensor in value["graph"]["tensors"]:
            tensor["data_type"] = "boolean"

    def unexpected_tensor(value: JsonObject) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][2])
        extra["uid"] = 104
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] = 4

    mutations: list[tuple[str, Mutation]] = [
        (
            "convolution wrong input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 0, "name"),
                "loss",
            ),
        ),
        (
            "convolution mismatched dtype",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "data_type"),
                "float32",
            ),
        ),
        ("convolution boolean storage", boolean_storage),
        (
            "convolution wrong compute dtype",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "compute_data_type"),
                "float16",
            ),
        ),
        (
            "convolution spatial rank mismatch",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "spatial_rank"),
                3,
            ),
        ),
        (
            "convolution short padding",
            lambda value: value["graph"]["nodes"][0]["attributes"][
                "pre_padding"
            ].pop(),
        ),
        (
            "convolution negative padding",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "post_padding", 0),
                -1,
            ),
        ),
        (
            "convolution zero stride",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "stride", 0),
                0,
            ),
        ),
        (
            "convolution zero dilation",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "dilation", 0),
                0,
            ),
        ),
        (
            "convolution indivisible groups",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "groups"),
                3,
            ),
        ),
        (
            "convolution filter channels",
            lambda value: set_path(
                value,
                ("graph", "tensors", 1, "dimensions", 1),
                1,
            ),
        ),
        (
            "convolution loss shape",
            lambda value: set_path(
                value,
                ("graph", "tensors", 2, "dimensions", 3),
                value["graph"]["tensors"][2]["dimensions"][3] + 1,
            ),
        ),
        (
            "convolution output count",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_outputs"),
                value["graph"]["nodes"][0]["attributes"]["n_outputs"] + 1,
            ),
        ),
        (
            "convolution invalid mode",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "convolution_mode"),
                2,
            ),
        ),
        (
            "convolution mode signature",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "convolution_mode"),
                0,
            ),
        ),
        (
            "convolution stride signature",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "strides"),
                [224, 1, 32, 4],
            ),
        ),
        (
            "convolution unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "block_m",
                32,
            ),
        ),
        (
            "convolution virtual tensor",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "virtual"), True
            ),
        ),
        (
            "convolution optional port",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "inputs", 0),
                "optional",
                True,
            ),
        ),
        ("convolution unexpected tensor", unexpected_tensor),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def convolution_manifest_matrix(matrix: Matrix) -> None:
    variant = ("program", "stages", 0, "variants", 0)
    mutations: list[tuple[str, Mutation]] = [
        (
            "convolution stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "matmul_strided_kernel",
            ),
        ),
        (
            "convolution variant function",
            lambda value: set_path(
                value, variant + ("function",), "conv_wgrad_nd_kernel"
            ),
        ),
        (
            "convolution variant signature",
            lambda value: set_path(
                value,
                variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        *[
            (
                f"convolution grid axis {axis}",
                lambda value, selected=axis: set_path(
                    value,
                    variant + ("grid", selected),
                    value["program"]["stages"][0]["variants"][0]["grid"][
                        selected
                    ]
                    + 1,
                ),
            )
            for axis in range(3)
        ],
        (
            "convolution candidate id",
            lambda value: set_path(
                value,
                variant + ("variant_id",),
                "block-64-warps-4-stages-2",
            ),
        ),
        (
            "convolution candidate warps",
            lambda value: set_path(value, variant + ("num_warps",), 2),
        ),
        (
            "convolution candidate stages",
            lambda value: set_path(value, variant + ("num_stages",), 3),
        ),
        (
            "convolution argument semantic",
            lambda value: set_path(
                value,
                variant + ("arguments", 0, "semantic_name"),
                "loss",
            ),
        ),
        (
            "convolution argument uid",
            lambda value: set_path(
                value, variant + ("arguments", 1, "uid"), 999
            ),
        ),
        (
            "convolution argument scalar bits",
            lambda value: set_path(
                value,
                variant + ("arguments", 2, "scalar_bits"),
                "00000000",
            ),
        ),
        (
            "convolution autotune candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def dense_dgrad_manifest_matrix(matrix: Matrix) -> None:
    filter_variant = ("program", "stages", 0, "variants", 0)
    loss_variant = ("program", "stages", 1, "variants", 0)
    mm_variant = ("program", "stages", 2, "variants", 0)

    def increment(value: JsonObject, path: tuple[str | int, ...]) -> None:
        current: Any = value
        for component in path:
            current = current[component]
        set_path(value, path, current + 1)

    mutations: list[tuple[str, Mutation]] = [
        (
            "dense Dgrad workspace size",
            lambda value: set_path(value, ("workspace_size",), 4096),
        ),
        (
            "dense Dgrad stage count",
            lambda value: set_path(value, ("program", "stage_count"), 2),
        ),
        (
            "dense Dgrad filter pack dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "dependencies"),
                [0],
            ),
        ),
        (
            "dense Dgrad loss pack dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "dependencies"),
                [0],
            ),
        ),
        (
            "dense Dgrad matmul dependency order",
            lambda value: set_path(
                value,
                ("program", "stages", 2, "dependencies"),
                [1, 0],
            ),
        ),
        (
            "dense Dgrad filter stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "conv_dgrad_nd_kernel",
            ),
        ),
        (
            "dense Dgrad loss stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "function"),
                "conv_dgrad_nd_kernel",
            ),
        ),
        (
            "dense Dgrad matmul stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 2, "function"),
                "conv_dgrad_nd_kernel",
            ),
        ),
        (
            "dense Dgrad filter variant function",
            lambda value: set_path(
                value,
                filter_variant + ("function",),
                "conv_dgrad_nd_kernel",
            ),
        ),
        (
            "dense Dgrad filter signature",
            lambda value: set_path(
                value,
                filter_variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "dense Dgrad loss signature",
            lambda value: set_path(
                value,
                loss_variant + ("full_signature",),
                value["program"]["stages"][1]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "dense Dgrad matmul signature",
            lambda value: set_path(
                value,
                mm_variant + ("full_signature",),
                value["program"]["stages"][2]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "dense Dgrad filter grid",
            lambda value: increment(value, filter_variant + ("grid", 0)),
        ),
        (
            "dense Dgrad loss grid",
            lambda value: increment(value, loss_variant + ("grid", 1)),
        ),
        (
            "dense Dgrad matmul grid",
            lambda value: increment(value, mm_variant + ("grid", 0)),
        ),
        (
            "dense Dgrad filter workspace semantic",
            lambda value: set_path(
                value,
                filter_variant + ("arguments", 1, "semantic_name"),
                "temporary",
            ),
        ),
        (
            "dense Dgrad loss workspace kind",
            lambda value: set_path(
                value,
                loss_variant + ("arguments", 1, "kind"),
                "tensor",
            ),
        ),
        (
            "dense Dgrad matmul workspace semantic",
            lambda value: set_path(
                value,
                mm_variant + ("arguments", 0, "semantic_name"),
                "dgrad_dense_loss",
            ),
        ),
        (
            "dense Dgrad output uid",
            lambda value: set_path(
                value,
                mm_variant + ("arguments", 2, "uid"),
                999,
            ),
        ),
        (
            "dense Dgrad pack autotune enabled",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "autotune", "enabled"),
                True,
            ),
        ),
        (
            "dense Dgrad matmul autotune cache",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    2,
                    "autotune",
                    "selection_cache",
                ),
                "tuning/stage-0.json",
            ),
        ),
        (
            "dense Dgrad matmul candidate count",
            lambda value: value["program"]["stages"][2]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def p5_wgrad_manifest_matrix(matrix: Matrix) -> None:
    pack_variant = ("program", "stages", 0, "variants", 0)
    mm_variant = ("program", "stages", 1, "variants", 0)

    def increment(value: JsonObject, path: tuple[str | int, ...]) -> None:
        current: Any = value
        for component in path:
            current = current[component]
        set_path(value, path, current + 1)

    mutations: list[tuple[str, Mutation]] = [
        (
            "P5 Wgrad workspace size",
            lambda value: set_path(value, ("workspace_size",), 4096),
        ),
        (
            "P5 Wgrad stage count",
            lambda value: set_path(value, ("program", "stage_count"), 1),
        ),
        (
            "P5 Wgrad pack dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "dependencies"),
                [0],
            ),
        ),
        (
            "P5 Wgrad matmul dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "dependencies"),
                [],
            ),
        ),
        (
            "P5 Wgrad pack stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "P5 Wgrad matmul stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "function"),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "P5 Wgrad pack variant function",
            lambda value: set_path(
                value,
                pack_variant + ("function",),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "P5 Wgrad pack signature",
            lambda value: set_path(
                value,
                pack_variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "P5 Wgrad matmul signature",
            lambda value: set_path(
                value,
                mm_variant + ("full_signature",),
                value["program"]["stages"][1]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "P5 Wgrad pack grid",
            lambda value: increment(value, pack_variant + ("grid", 0)),
        ),
        (
            "P5 Wgrad matmul grid",
            lambda value: increment(value, mm_variant + ("grid", 0)),
        ),
        (
            "P5 Wgrad pack workspace semantic",
            lambda value: set_path(
                value,
                pack_variant + ("arguments", 1, "semantic_name"),
                "temporary",
            ),
        ),
        (
            "P5 Wgrad matmul workspace kind",
            lambda value: set_path(
                value,
                mm_variant + ("arguments", 1, "kind"),
                "tensor",
            ),
        ),
        (
            "P5 Wgrad output uid",
            lambda value: set_path(
                value,
                mm_variant + ("arguments", 2, "uid"),
                999,
            ),
        ),
        (
            "P5 Wgrad matmul autotune cache",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    1,
                    "autotune",
                    "selection_cache",
                ),
                "tuning/stage-0.json",
            ),
        ),
        (
            "P5 Wgrad matmul candidate count",
            lambda value: value["program"]["stages"][1]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def stem_wgrad_manifest_matrix(matrix: Matrix) -> None:
    split_variant = ("program", "stages", 0, "variants", 0)
    reduce_variant = ("program", "stages", 1, "variants", 0)

    def increment(value: JsonObject, path: tuple[str | int, ...]) -> None:
        current: Any = value
        for component in path:
            current = current[component]
        set_path(value, path, current + 1)

    mutations: list[tuple[str, Mutation]] = [
        (
            "stem Wgrad workspace size",
            lambda value: set_path(value, ("workspace_size",), 4096),
        ),
        (
            "stem Wgrad stage count",
            lambda value: set_path(value, ("program", "stage_count"), 1),
        ),
        (
            "stem Wgrad split dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "dependencies"),
                [0],
            ),
        ),
        (
            "stem Wgrad reduce dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "dependencies"),
                [],
            ),
        ),
        (
            "stem Wgrad split stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "stem Wgrad reduce stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "function"),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "stem Wgrad split variant function",
            lambda value: set_path(
                value,
                split_variant + ("function",),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "stem Wgrad split signature",
            lambda value: set_path(
                value,
                split_variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "stem Wgrad reduce signature",
            lambda value: set_path(
                value,
                reduce_variant + ("full_signature",),
                value["program"]["stages"][1]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "stem Wgrad split grid",
            lambda value: increment(value, split_variant + ("grid", 0)),
        ),
        (
            "stem Wgrad reduce grid",
            lambda value: increment(value, reduce_variant + ("grid", 0)),
        ),
        (
            "stem Wgrad split workspace semantic",
            lambda value: set_path(
                value,
                split_variant + ("arguments", 2, "semantic_name"),
                "temporary",
            ),
        ),
        (
            "stem Wgrad reduce workspace kind",
            lambda value: set_path(
                value,
                reduce_variant + ("arguments", 0, "kind"),
                "tensor",
            ),
        ),
        (
            "stem Wgrad output uid",
            lambda value: set_path(
                value,
                reduce_variant + ("arguments", 1, "uid"),
                999,
            ),
        ),
        (
            "stem Wgrad reduce autotune cache",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    1,
                    "autotune",
                    "selection_cache",
                ),
                "tuning/stage-0.json",
            ),
        ),
        (
            "stem Wgrad split candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
        (
            "stem Wgrad reduce candidate count",
            lambda value: value["program"]["stages"][1]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def standard_wgrad_manifest_matrix(
    matrix: Matrix, *, label_prefix: str = "standard Wgrad"
) -> None:
    split_variant = ("program", "stages", 0, "variants", 0)
    reduce_variant = ("program", "stages", 2, "variants", 0)

    def increment(value: JsonObject, path: tuple[str | int, ...]) -> None:
        current: Any = value
        for component in path:
            current = current[component]
        set_path(value, path, current + 1)

    mutations: list[tuple[str, Mutation]] = [
        (
            "standard Wgrad workspace size",
            lambda value: set_path(value, ("workspace_size",), 4096),
        ),
        (
            "standard Wgrad stage count",
            lambda value: set_path(value, ("program", "stage_count"), 1),
        ),
        (
            "standard Wgrad split dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "dependencies"),
                [0],
            ),
        ),
        (
            "standard Wgrad reduce dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 2, "dependencies"),
                [],
            ),
        ),
        (
            "standard Wgrad split stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "standard Wgrad reduce stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 2, "function"),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "standard Wgrad split variant function",
            lambda value: set_path(
                value,
                split_variant + ("function",),
                "conv_wgrad_nd_kernel",
            ),
        ),
        (
            "standard Wgrad split signature",
            lambda value: set_path(
                value,
                split_variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "standard Wgrad reduce signature",
            lambda value: set_path(
                value,
                reduce_variant + ("full_signature",),
                value["program"]["stages"][2]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            "standard Wgrad split grid",
            lambda value: increment(value, split_variant + ("grid", 0)),
        ),
        (
            "standard Wgrad reduce grid",
            lambda value: increment(value, reduce_variant + ("grid", 0)),
        ),
        (
            "standard Wgrad split workspace semantic",
            lambda value: set_path(
                value,
                split_variant + ("arguments", 1, "semantic_name"),
                "temporary",
            ),
        ),
        (
            "standard Wgrad reduce workspace kind",
            lambda value: set_path(
                value,
                reduce_variant + ("arguments", 0, "kind"),
                "tensor",
            ),
        ),
        (
            "standard Wgrad output uid",
            lambda value: set_path(
                value,
                reduce_variant + ("arguments", 1, "uid"),
                999,
            ),
        ),
        (
            "standard Wgrad reduce autotune cache",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    2,
                    "autotune",
                    "selection_cache",
                ),
                "tuning/stage-0.json",
            ),
        ),
        (
            "standard Wgrad split candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
        (
            "standard Wgrad reduce candidate count",
            lambda value: value["program"]["stages"][2]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        unique_label = label.replace("standard Wgrad", label_prefix, 1)
        matrix.reject(unique_label, manifest_mutation=mutation)


def filesystem_matrix(matrix: Matrix) -> None:
    matrix.reject(
        "mutated source bytes",
        filesystem_mutation=lambda artifact: (
            artifact / "kernels/binary.py"
        ).write_bytes(b"mutated"),
    )
    matrix.reject(
        "extra file",
        filesystem_mutation=lambda artifact: (
            artifact / "evil.bin"
        ).write_bytes(b"x"),
    )
    matrix.reject(
        "extra directory",
        filesystem_mutation=lambda artifact: (artifact / "extra").mkdir(),
    )
    matrix.reject(
        "mismatched request file",
        filesystem_mutation=lambda artifact: (
            artifact / "request.json"
        ).write_bytes(b"{}"),
    )

    def source_symlink(artifact: Path) -> None:
        source = artifact / "kernels/binary.py"
        source.unlink()
        source.symlink_to(matrix.baseline_artifact / "kernels/binary.py")

    matrix.reject("source symlink", filesystem_mutation=source_symlink)
    matrix.reject(
        "dangling extra symlink",
        filesystem_mutation=lambda artifact: (
            artifact / "dangling"
        ).symlink_to(artifact / "missing"),
    )
    matrix.reject(
        "empty tuning directory",
        filesystem_mutation=lambda artifact: (artifact / "tuning").mkdir(),
    )

    def cache_symlink(artifact: Path) -> None:
        tuning = artifact / "tuning"
        tuning.mkdir()
        (tuning / "stage-0.json").symlink_to(
            matrix.baseline_artifact / "manifest.json"
        )

    matrix.reject("autotune cache symlink", filesystem_mutation=cache_symlink)

    def manifest_symlink(artifact: Path) -> None:
        manifest = artifact / "manifest.json"
        manifest.unlink()
        manifest.symlink_to(matrix.baseline_artifact / "manifest.json")

    matrix.reject("manifest symlink", filesystem_mutation=manifest_symlink)

    def root_symlink(root: Path, artifact: Path) -> Path:
        link = root / f"{artifact.name}-link"
        link.symlink_to(artifact, target_is_directory=True)
        return link

    matrix.reject("artifact root symlink", artifact_override=root_symlink)
    matrix.reject(
        "duplicate manifest key",
        raw_manifest=lambda payload: payload.replace(
            b"{", b'{"schema_version":1,', 1
        ),
    )
    matrix.reject(
        "oversized manifest",
        raw_manifest=lambda payload: b" " * ((16 << 20) + 1),
    )
    matrix.reject(
        "duplicate request key",
        raw_request=lambda payload: payload.replace(
            b"{", b'{"schema_version":3,', 1
        ),
    )


def add_square_request_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "AddSquare node order",
            lambda value: value["graph"]["nodes"].reverse(),
        ),
        (
            "AddSquare duplicate node id",
            lambda value: set_path(value, ("graph", "nodes", 1, "id"), 0),
        ),
        (
            "AddSquare square operands differ",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 1, "uid"),
                100,
            ),
        ),
        (
            "AddSquare disconnected virtual",
            lambda value: set_path(
                value,
                ("graph", "nodes", 1, "inputs", 1, "uid"),
                101,
            ),
        ),
        (
            "AddSquare square external",
            lambda value: set_path(
                value, ("graph", "tensors", 1, "virtual"), False
            ),
        ),
        (
            "AddSquare left virtual",
            lambda value: set_path(
                value, ("graph", "tensors", 2, "virtual"), True
            ),
        ),
        (
            "AddSquare virtual dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 1, "data_type"), "float16"
            ),
        ),
        (
            "AddSquare virtual layout",
            lambda value: set_path(
                value, ("graph", "tensors", 1, "strides"), [24, 8, 2]
            ),
        ),
        (
            "AddSquare Mul mode",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "mode"),
                17,
            ),
        ),
        (
            "AddSquare Add alpha",
            lambda value: set_path(
                value,
                ("graph", "nodes", 1, "attributes", "alpha"),
                0.5,
            ),
        ),
        (
            "AddSquare optional port",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "inputs", 0),
                "optional",
                False,
            ),
        ),
        (
            "AddSquare extra tensor",
            lambda value: set_path(value, ("graph", "tensor_count"), 5),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def add_square_manifest_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "AddSquare binding includes virtual",
            lambda value: set_path(
                value, ("external_binding_uids",), [101, 103, 100, 102]
            ),
        ),
        (
            "AddSquare stage node",
            lambda value: set_path(
                value, ("program", "stages", 0, "node_id"), 0
            ),
        ),
        (
            "AddSquare stage operation",
            lambda value: set_path(
                value, ("program", "stages", 0, "operation"), "add"
            ),
        ),
        (
            "AddSquare stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "binary_contiguous_kernel",
            ),
        ),
        (
            "AddSquare variant function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "function"),
                "binary_contiguous_kernel",
            ),
        ),
        (
            "AddSquare variant signature",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "full_signature"),
                "*fp32:16,*fp32:16,*fp32:16,i32,18,1.0,256",
            ),
        ),
        (
            "AddSquare right argument uid",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    1,
                    "uid",
                ),
                103,
            ),
        ),
        (
            "AddSquare candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def conv_bias_relu_request_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "ConvBiasRelu node order",
            lambda value: value["graph"]["nodes"].reverse(),
        ),
        (
            "ConvBiasRelu duplicate node id",
            lambda value: set_path(value, ("graph", "nodes", 2, "id"), 1),
        ),
        (
            "ConvBiasRelu convolution type",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "type"), "conv2d_fprop"
            ),
        ),
        (
            "ConvBiasRelu disconnected convolution",
            lambda value: set_path(
                value, ("graph", "nodes", 1, "inputs", 0, "uid"), 200
            ),
        ),
        (
            "ConvBiasRelu disconnected ReLU",
            lambda value: set_path(
                value, ("graph", "nodes", 2, "inputs", 0, "uid"), 204
            ),
        ),
        (
            "ConvBiasRelu convolution external",
            lambda value: set_path(
                value, ("graph", "tensors", 2, "virtual"), False
            ),
        ),
        (
            "ConvBiasRelu biased external",
            lambda value: set_path(
                value, ("graph", "tensors", 4, "virtual"), False
            ),
        ),
        (
            "ConvBiasRelu output virtual",
            lambda value: set_path(
                value, ("graph", "tensors", 5, "virtual"), True
            ),
        ),
        (
            "ConvBiasRelu virtual dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 4, "data_type"), "float16"
            ),
        ),
        (
            "ConvBiasRelu virtual layout",
            lambda value: set_path(
                value, ("graph", "tensors", 4, "strides"), [360, 2, 72, 12]
            ),
        ),
        (
            "ConvBiasRelu bias shape",
            lambda value: set_path(
                value, ("graph", "tensors", 3, "dimensions"), [1, 6, 1, 2]
            ),
        ),
        (
            "ConvBiasRelu spatial rank",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "spatial_rank"),
                3,
            ),
        ),
        (
            "ConvBiasRelu convolution output count",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_outputs"),
                179,
            ),
        ),
        (
            "ConvBiasRelu bias mode",
            lambda value: set_path(
                value, ("graph", "nodes", 1, "attributes", "mode"), 18
            ),
        ),
        (
            "ConvBiasRelu bias alpha",
            lambda value: set_path(
                value, ("graph", "nodes", 1, "attributes", "alpha"), 0.5
            ),
        ),
        (
            "ConvBiasRelu ReLU mode",
            lambda value: set_path(
                value, ("graph", "nodes", 2, "attributes", "mode"), 33
            ),
        ),
        (
            "ConvBiasRelu ReLU slope",
            lambda value: set_path(
                value,
                ("graph", "nodes", 2, "attributes", "negative_slope"),
                0.25,
            ),
        ),
        (
            "ConvBiasRelu optional port",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "inputs", 0),
                "optional",
                True,
            ),
        ),
        (
            "ConvBiasRelu tensor count",
            lambda value: set_path(value, ("graph", "tensor_count"), 7),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def conv_bias_relu_manifest_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "ConvBiasRelu binding includes virtual",
            lambda value: set_path(
                value,
                ("external_binding_uids",),
                [200, 201, 204, 202, 203],
            ),
        ),
        (
            "ConvBiasRelu stage node",
            lambda value: set_path(
                value, ("program", "stages", 0, "node_id"), 1
            ),
        ),
        (
            "ConvBiasRelu stage operation",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "operation"),
                "convolution_fprop",
            ),
        ),
        (
            "ConvBiasRelu stage source",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "source"),
                "kernels/convolution.py",
            ),
        ),
        (
            "ConvBiasRelu stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "function"),
                "conv2d_spatial_nchw_kernel",
            ),
        ),
        (
            "ConvBiasRelu variant signature",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "full_signature"),
                "*fp32:16,*fp32:16,*fp32:16,*fp32:16",
            ),
        ),
        (
            "ConvBiasRelu bias argument uid",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    0,
                    "variants",
                    0,
                    "arguments",
                    2,
                    "uid",
                ),
                204,
            ),
        ),
        (
            "ConvBiasRelu variant grid",
            lambda value: set_path(
                value,
                ("program", "stages", 0, "variants", 0, "grid"),
                [3, 1, 1],
            ),
        ),
        (
            "ConvBiasRelu candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def normalization_request_matrix(matrix: Matrix) -> None:
    def unexpected_tensor(value: JsonObject) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][-1])
        extra["uid"] = 399
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] += 1

    mutations: list[tuple[str, Mutation]] = [
        (
            "LayerNorm wrong input role",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "inputs", 0, "name"), "input"
            ),
        ),
        (
            "LayerNorm missing mean output",
            lambda value: value["graph"]["nodes"][0]["outputs"].pop(1),
        ),
        (
            "LayerNorm duplicate tensor binding",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "outputs", 0, "uid"), 300
            ),
        ),
        (
            "LayerNorm virtual tensor",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "virtual"), True
            ),
        ),
        (
            "LayerNorm compute dtype",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "compute_data_type"), "float16"
            ),
        ),
        (
            "LayerNorm mismatched output dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 3, "data_type"), "bfloat16"
            ),
        ),
        (
            "LayerNorm noncontiguous input",
            lambda value: set_path(
                value, ("graph", "tensors", 0, "strides"), [12, 1, 3]
            ),
        ),
        (
            "LayerNorm scale suffix",
            lambda value: set_path(
                value, ("graph", "tensors", 1, "dimensions"), [1, 1, 3]
            ),
        ),
        (
            "LayerNorm statistic dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 4, "data_type"), "float16"
            ),
        ),
        (
            "LayerNorm statistic shape",
            lambda value: set_path(
                value, ("graph", "tensors", 4, "dimensions"), [2, 3]
            ),
        ),
        (
            "LayerNorm nonpositive epsilon",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "epsilon"), 0.0
            ),
        ),
        (
            "LayerNorm forward phase",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "forward_phase"),
                1,
            ),
        ),
        (
            "LayerNorm row count",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "rows"), 5
            ),
        ),
        (
            "LayerNorm normalized extent",
            lambda value: set_path(
                value,
                (
                    "graph",
                    "nodes",
                    0,
                    "attributes",
                    "normalized_elements",
                ),
                3,
            ),
        ),
        (
            "LayerNorm optional port",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "inputs", 0),
                "optional",
                False,
            ),
        ),
        (
            "LayerNorm unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "axis",
                -1,
            ),
        ),
        ("LayerNorm unexpected tensor", unexpected_tensor),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def batchnorm_request_matrix(matrix: Matrix) -> None:
    def unexpected_tensor(value: JsonObject) -> None:
        extra = copy.deepcopy(value["graph"]["tensors"][-1])
        extra["uid"] = 399
        value["graph"]["tensors"].append(extra)
        value["graph"]["tensor_count"] += 1

    mutations: list[tuple[str, Mutation]] = [
        (
            "BatchNorm wrong input role",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 3, "name"),
                "running_mean",
            ),
        ),
        (
            "BatchNorm missing output",
            lambda value: value["graph"]["nodes"][0]["outputs"].pop(),
        ),
        (
            "BatchNorm duplicate binding",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "outputs", 0, "uid"), 320
            ),
        ),
        (
            "BatchNorm scale dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 1, "data_type"), "bfloat16"
            ),
        ),
        (
            "BatchNorm statistic dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 6, "data_type"), "float16"
            ),
        ),
        (
            "BatchNorm parameter elements",
            lambda value: set_path(
                value, ("graph", "tensors", 3, "dimensions"), [1, 5, 1, 1]
            ),
        ),
        (
            "BatchNorm output dimensions",
            lambda value: set_path(
                value, ("graph", "tensors", 5, "dimensions"), [2, 4, 3, 4]
            ),
        ),
        (
            "BatchNorm duplicated dimensions",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "dimensions"),
                [2, 4, 3, 4],
            ),
        ),
        (
            "BatchNorm duplicated strides",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "x_strides"),
                [60, 15, 5, 1],
            ),
        ),
        (
            "BatchNorm rank",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "rank"), 3
            ),
        ),
        (
            "BatchNorm batch",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "batch"), 3
            ),
        ),
        (
            "BatchNorm channels",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "channels"), 5
            ),
        ),
        (
            "BatchNorm spatial",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "spatial"), 14
            ),
        ),
        (
            "BatchNorm element count",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "n_elements"),
                119,
            ),
        ),
        (
            "BatchNorm epsilon",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "epsilon"), -1.0
            ),
        ),
        (
            "BatchNorm momentum",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "attributes", "momentum"), 1.1
            ),
        ),
        (
            "BatchNorm virtual tensor",
            lambda value: set_path(
                value, ("graph", "tensors", 9, "virtual"), True
            ),
        ),
        (
            "BatchNorm unknown attribute",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "training",
                True,
            ),
        ),
        ("BatchNorm unexpected tensor", unexpected_tensor),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def batchnorm_inference_request_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "BatchNorm inference operation",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "type"), "batchnorm"
            ),
        ),
        (
            "BatchNorm inference mean role",
            lambda value: set_path(
                value, ("graph", "nodes", 0, "inputs", 1, "name"), "mean_in"
            ),
        ),
        (
            "BatchNorm inference parameter dtype",
            lambda value: set_path(
                value, ("graph", "tensors", 3, "data_type"), "float16"
            ),
        ),
        (
            "BatchNorm inference output strides",
            lambda value: set_path(
                value, ("graph", "tensors", 5, "strides"), [60, 15, 5, 1]
            ),
        ),
        (
            "BatchNorm inference duplicated output strides",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "y_strides"),
                [60, 15, 5, 1],
            ),
        ),
        (
            "BatchNorm inference forbidden epsilon",
            lambda value: add_key(
                value,
                ("graph", "nodes", 0, "attributes"),
                "epsilon",
                0.001,
            ),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def normalization_manifest_matrix(matrix: Matrix, label_prefix: str) -> None:
    stage = ("program", "stages", 0)
    variant = stage + ("variants", 0)
    mutations: list[tuple[str, Mutation]] = [
        (
            f"{label_prefix} stage operation",
            lambda value: set_path(value, stage + ("operation",), "add"),
        ),
        (
            f"{label_prefix} stage source",
            lambda value: set_path(
                value, stage + ("source",), "kernels/binary.py"
            ),
        ),
        (
            f"{label_prefix} stage function",
            lambda value: set_path(
                value, stage + ("function",), "binary_contiguous_kernel"
            ),
        ),
        (
            f"{label_prefix} variant source",
            lambda value: set_path(
                value, variant + ("source",), "kernels/binary.py"
            ),
        ),
        (
            f"{label_prefix} variant function",
            lambda value: set_path(
                value,
                variant + ("function",),
                "binary_contiguous_kernel",
            ),
        ),
        (
            f"{label_prefix} full signature",
            lambda value: set_path(
                value,
                variant + ("full_signature",),
                value["program"]["stages"][0]["variants"][0]["full_signature"]
                + ",0",
            ),
        ),
        (
            f"{label_prefix} grid",
            lambda value: set_path(
                value,
                variant + ("grid", 0),
                value["program"]["stages"][0]["variants"][0]["grid"][0] + 1,
            ),
        ),
        (
            f"{label_prefix} argument semantic",
            lambda value: set_path(
                value,
                variant + ("arguments", 0, "semantic_name"),
                "invalid",
            ),
        ),
        (
            f"{label_prefix} argument uid",
            lambda value: set_path(
                value, variant + ("arguments", 0, "uid"), 999
            ),
        ),
        (
            f"{label_prefix} tensor scalar bits",
            lambda value: set_path(
                value,
                variant + ("arguments", 0, "scalar_bits"),
                "00000000",
            ),
        ),
        (
            f"{label_prefix} candidate warps",
            lambda value: set_path(value, variant + ("num_warps",), 2),
        ),
        (
            f"{label_prefix} candidate count",
            lambda value: value["program"]["stages"][0]["variants"].pop(),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def attention_request_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "Attention operation",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "type"),
                "sdpa_fp8",
            ),
        ),
        (
            "Attention missing input",
            lambda value: delete_path(
                value, ("graph", "nodes", 0, "inputs", 17)
            ),
        ),
        (
            "Attention input name",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "inputs", 0, "name"),
                "query",
            ),
        ),
        (
            "Attention head count",
            lambda value: set_path(
                value,
                ("graph", "nodes", 0, "attributes", "heads"),
                3,
            ),
        ),
        (
            "Attention tensor stride",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "strides", 2),
                129,
            ),
        ),
        (
            "Attention fp8 type",
            lambda value: set_path(
                value,
                ("graph", "tensors", 0, "data_type"),
                "fp8_e5m2",
            ),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, request_mutation=mutation)


def attention_manifest_matrix(matrix: Matrix) -> None:
    mutations: list[tuple[str, Mutation]] = [
        (
            "Attention workspace size",
            lambda value: set_path(value, ("workspace_size",), 8192),
        ),
        (
            "Attention stage count",
            lambda value: set_path(value, ("program", "stage_count"), 2),
        ),
        (
            "Attention stage dependency",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "dependencies"),
                [],
            ),
        ),
        (
            "Attention stage function",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "function"),
                "_sdpa_fp8_fwd_kernel",
            ),
        ),
        (
            "Attention variant function",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "variants", 0, "function"),
                "_sdpa_fp8_fwd_kernel",
            ),
        ),
        (
            "Attention full signature",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "variants", 0, "full_signature"),
                value["program"]["stages"][1]["variants"][0]["full_signature"]
                + ",1",
            ),
        ),
        (
            "Attention launch grid",
            lambda value: set_path(
                value,
                ("program", "stages", 1, "variants", 0, "grid"),
                [1, 1, 1],
            ),
        ),
        (
            "Attention argument kind",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    1,
                    "variants",
                    0,
                    "arguments",
                    0,
                    "kind",
                ),
                "workspace",
            ),
        ),
        (
            "Attention autotune cache path",
            lambda value: set_path(
                value,
                (
                    "program",
                    "stages",
                    2,
                    "autotune",
                    "selection_cache",
                ),
                "tuning/stage-1.json",
            ),
        ),
    ]
    for label, mutation in mutations:
        matrix.reject(label, manifest_mutation=mutation)


def extended_artifact_matrix(provider, root, executable, identity, version):
    """Validate the new generic program ABI independently in the C++ reader."""
    valid_count = 0
    for dtype in ("float32", "float16", "bfloat16"):
        for strided in (False, True):
            fixture = request_fixture(
                identity,
                version,
                autotune=False,
                strided=strided,
                alpha=1.0,
                operation="relu_backward",
                mode=42,
                data_type=dtype,
            )
            fixture["graph"]["nodes"][0]["attributes"] = {
                "alpha": 1.0,
                "n_elements": 24,
                "pointwise_mode": 42,
                "has_upper_clip": 0,
            }
            request, artifact = compile_fixture(
                provider,
                root,
                f"extended-{dtype}-{strided}",
                fixture,
            )
            run_parser(executable, request, artifact, valid=True)
            valid_count += 1
    matrix = Matrix(root, executable, request, artifact)

    def reject(label, path, key, value):
        def mutate(manifest):
            selected = manifest
            for component in path:
                selected = selected[component]
            selected[key] = value

        matrix.reject("extended " + label, manifest_mutation=mutate)

    stage = ("program", "stages", 0)
    variant = (*stage, "variants", 0)
    argument = (*variant, "arguments")
    reject("workspace", (), "workspace_size", 0)
    reject("workspace alignment", (), "workspace_alignment", 128)
    reject("function", stage, "function", "unregistered_kernel")
    reject("dependency cycle", stage, "dependencies", [0])
    reject("autotune", (*stage, "autotune"), "enabled", True)
    reject("zero grid", variant, "grid", [0, 1, 1])
    reject("oversized grid", variant, "grid", [1, 65536, 1])
    reject("warps", variant, "num_warps", 8)
    reject("stages", variant, "num_stages", 2)
    reject("source path", variant, "source", "../activation_backward.py")
    reject("pointer dtype", variant, "full_signature", "*fp32:16,i32,1")
    reject("tensor uid", (*argument, 0), "uid", 999)
    reject("argument name", (*argument, 0), "semantic_name", "other_ptr")
    reject("scalar value", (*argument, 3), "scalar_bits", "00000000")
    matrix.reject(
        "extended missing argument",
        manifest_mutation=lambda m: m["program"]["stages"][0]["variants"][0][
            "arguments"
        ].pop(),
    )
    matrix.reject(
        "extended argument order",
        manifest_mutation=lambda m: m["program"]["stages"][0]["variants"][0][
            "arguments"
        ].reverse(),
    )
    return valid_count, matrix.count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--capture-executable", type=Path, required=True)
    parser.add_argument("--capture-compiler", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--flagdnn-version", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    provider_path = arguments.provider.resolve(strict=True)
    compiler = arguments.compiler.resolve(strict=True)
    executable = arguments.parser.resolve(strict=True)
    capture_executable = arguments.capture_executable.resolve(strict=True)
    capture_compiler = arguments.capture_compiler.resolve(strict=True)
    python = arguments.python.resolve(strict=True)
    if arguments.target != TARGET:
        fail("artifact contract target differs from the configured target")

    sys.path.insert(0, str(compiler.parent.parent))
    try:
        loader = importlib.import_module("flagdnn_codegen.provider_loader")
        provider = loader.get_provider("mthreads")
    finally:
        sys.path.pop(0)
    if Path(provider.__file__).resolve() != provider_path:
        fail("provider loader selected a different mthreads compiler")
    identity = provider.compiler_identity(TARGET, "libtriton_jit")[
        "identity_sha256"
    ]

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-mthreads-artifact-contract-"
    ) as temporary:
        root = Path(temporary)
        extended_valid, extended_rejected = extended_artifact_matrix(
            provider, root, executable, identity, arguments.flagdnn_version
        )
        dense_request, dense_artifact = compile_fixture(
            provider,
            root,
            "dense",
            request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=False,
                strided=False,
                alpha=-0.75,
            ),
        )
        strided_request, strided_artifact = compile_fixture(
            provider,
            root,
            "strided-autotune",
            request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=True,
                strided=True,
                alpha=0.1,
            ),
        )
        run_parser(executable, dense_request, dense_artifact, valid=True)
        run_parser(executable, strided_request, strided_artifact, valid=True)
        valid_binary_requests = {
            "sub-alpha": request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=False,
                strided=False,
                alpha=-2.0,
                operation="sub",
                mode=17,
                data_type="float16",
            ),
            "mod": request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=False,
                strided=False,
                alpha=1.0,
                operation="mod",
                mode=22,
            ),
            "cmp-eq": request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=False,
                strided=False,
                alpha=1.0,
                operation="cmp_eq",
                mode=25,
                data_type="float16",
                output_data_type="boolean",
                compute_data_type="boolean",
            ),
            "logical-or": request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=False,
                strided=False,
                alpha=1.0,
                operation="logical_or",
                mode=32,
                data_type="boolean",
                output_data_type="boolean",
                compute_data_type="boolean",
            ),
            "sigmoid-backward": request_fixture(
                identity,
                arguments.flagdnn_version,
                autotune=False,
                strided=False,
                alpha=1.0,
                operation="sigmoid_backward",
                mode=40,
                data_type="bfloat16",
            ),
        }
        normalized_sigmoid = copy.deepcopy(
            valid_binary_requests["sigmoid-backward"]
        )
        normalized_sigmoid["graph"]["nodes"][0]["attributes"][
            "has_upper_clip"
        ] = 0
        valid_binary_requests["sigmoid-backward-normalized"] = (
            normalized_sigmoid
        )
        sigmoid_rejected = 0
        for name, request in valid_binary_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            if name == "sigmoid-backward-normalized":
                sigmoid_matrix = Matrix(
                    root, executable, request_path, artifact
                )
                for bad_clip in (1, False, "0"):

                    def mutate(value, replacement=bad_clip):
                        set_path(
                            value,
                            (
                                "graph",
                                "nodes",
                                0,
                                "attributes",
                                "has_upper_clip",
                            ),
                            replacement,
                        )

                    invalid = copy.deepcopy(request)
                    mutate(invalid)
                    try:
                        provider.parse_compiler_request(
                            json.dumps(invalid).encode(),
                            expected_target=TARGET,
                            expected_identity=identity,
                        )
                    except ValueError:
                        pass
                    else:
                        fail(
                            "compiler accepted an invalid Sigmoid backward clip flag"
                        )
                    sigmoid_matrix.reject(
                        "sigmoid backward clip " + repr(bad_clip),
                        request_mutation=mutate,
                    )
                sigmoid_rejected = sigmoid_matrix.count
        valid_unary_requests = {
            "unary-relu": unary_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="relu",
                mode=2,
            ),
            "unary-relu-strided-autotune": unary_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="relu",
                mode=2,
                autotune=True,
                strided=True,
                negative_slope=0.25,
                lower_clip=-0.5,
                upper_clip=6.0,
                has_upper_clip=True,
            ),
            "unary-logical-not": unary_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="logical_not",
                mode=24,
                data_type="boolean",
                compute_data_type="boolean",
            ),
            "unary-swish": unary_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="swish",
                mode=38,
                data_type="float16",
                swish_beta=1.5,
            ),
            "unary-elu": unary_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="elu",
                mode=35,
                data_type="bfloat16",
                elu_alpha=0.25,
            ),
            "unary-softplus": unary_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="softplus",
                mode=37,
                softplus_beta=2.0,
            ),
        }
        for name, request in valid_unary_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
        valid_ternary_requests = {
            "ternary-binary-select": ternary_request_fixture(
                identity,
                arguments.flagdnn_version,
            ),
            "ternary-broadcast-autotune": ternary_request_fixture(
                identity,
                arguments.flagdnn_version,
                data_type="float16",
                autotune=True,
                broadcast=True,
            ),
        }
        valid_ternary_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_ternary_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_ternary_artifacts[name] = (request_path, artifact)
        valid_layout_requests = {
            "layout-reshape-autotune": layout_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="reshape",
                autotune=True,
            ),
            "layout-transpose-bfloat16": layout_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="transpose",
                data_type="bfloat16",
            ),
            "layout-slice-float16": layout_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="slice",
                data_type="float16",
            ),
        }
        sparse_slice = copy.deepcopy(
            valid_layout_requests["layout-slice-float16"]
        )
        sparse_slice["graph"]["tensors"][0].update(
            dimensions=[3, 5, 7], strides=[35, 7, 1]
        )
        sparse_slice["graph"]["tensors"][1].update(
            dimensions=[2, 3, 3], strides=[35, 14, 2]
        )
        sparse_slice["graph"]["nodes"][0]["attributes"].update(
            n_elements=18,
            input_dimensions=[3, 5, 7],
            input_strides=[35, 7, 1],
            output_dimensions=[2, 3, 3],
            output_strides=[35, 14, 2],
            starts=[1, 0, 1],
            limits=[3, 5, 7],
            slice_strides=[1, 2, 2],
        )
        valid_layout_requests["layout-slice-gapped-strides"] = sparse_slice
        valid_layout_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_layout_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_layout_artifacts[name] = (request_path, artifact)
        valid_reduction_requests = {
            "reduction-sum-2d": reduction_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="reduction_sum",
                mode=0,
                input_dimensions=[7, 256],
                axis=1,
                keep_dimensions=False,
            ),
            "reduction-avg-3d-autotune": reduction_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="reduction_avg",
                mode=1,
                input_dimensions=[2, 4, 8],
                axis=1,
                keep_dimensions=True,
                data_type="float16",
                autotune=True,
            ),
            "reduction-mul-strided": reduction_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="reduction_mul",
                mode=2,
                input_dimensions=[2, 3, 5, 5],
                axis=1,
                keep_dimensions=True,
                data_type="bfloat16",
                input_strides=[75, 1, 15, 3],
            ),
            "reduction-scalar-output": reduction_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="reduction_sum",
                mode=0,
                input_dimensions=[8],
                axis=0,
                keep_dimensions=False,
            ),
        }
        valid_reduction_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_reduction_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_reduction_artifacts[name] = (request_path, artifact)
        valid_matmul_requests = {
            "matmul-fp32-2d": matmul_request_fixture(
                identity,
                arguments.flagdnn_version,
                a_dimensions=[17, 30],
                b_dimensions=[30, 23],
            ),
            "matmul-fp16-batched": matmul_request_fixture(
                identity,
                arguments.flagdnn_version,
                a_dimensions=[2, 17, 30],
                b_dimensions=[2, 30, 23],
                data_type="float16",
            ),
            "matmul-fp16-tle-autotune": matmul_request_fixture(
                identity,
                arguments.flagdnn_version,
                a_dimensions=[32, 512, 512],
                b_dimensions=[32, 512, 512],
                data_type="float16",
                autotune=True,
            ),
            "matmul-bf16-tle-autotune": matmul_request_fixture(
                identity,
                arguments.flagdnn_version,
                a_dimensions=[32, 512, 512],
                b_dimensions=[32, 512, 512],
                data_type="bfloat16",
                autotune=True,
            ),
            "matmul-bf16-broadcast-autotune": matmul_request_fixture(
                identity,
                arguments.flagdnn_version,
                a_dimensions=[2, 1, 17, 30],
                b_dimensions=[3, 30, 23],
                data_type="bfloat16",
                autotune=True,
            ),
            "matmul-fp32-strided": matmul_request_fixture(
                identity,
                arguments.flagdnn_version,
                a_dimensions=[2, 17, 30],
                b_dimensions=[2, 30, 23],
                a_strides=[600, 31, 1],
                b_strides=[800, 1, 32],
                output_strides=[500, 25, 1],
            ),
        }
        valid_matmul_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_matmul_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_matmul_artifacts[name] = (request_path, artifact)
        valid_convolution_requests = {
            "convolution-fprop-1d": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_fprop",
                image_dimensions=[2, 4, 11],
                filter_dimensions=[6, 2, 3],
                pre_padding=[1],
                post_padding=[2],
                stride=[2],
                dilation=[1],
                groups=2,
            ),
            "convolution-fprop-im2col-stride2": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_fprop",
                image_dimensions=[8, 64, 56, 56],
                filter_dimensions=[128, 64, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                data_type="float16",
                autotune=True,
            ),
            "convolution-fprop-im2col-p5-large-reduction": (
                convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation="convolution_fprop",
                    image_dimensions=[1, 128, 40, 40],
                    filter_dimensions=[256, 128, 3, 3],
                    pre_padding=[1, 1],
                    post_padding=[1, 1],
                    stride=[2, 2],
                    dilation=[1, 1],
                    data_type="float16",
                )
            ),
            "conv2d-fprop-im2col-asymmetric": (
                convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
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
            "convolution-fprop-im2col-fp32-stem": (
                convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation="convolution_fprop",
                    image_dimensions=[1, 3, 640, 640],
                    filter_dimensions=[64, 3, 3, 3],
                    pre_padding=[1, 1],
                    post_padding=[1, 1],
                    stride=[2, 2],
                    dilation=[1, 1],
                    data_type="float32",
                    autotune=True,
                )
            ),
            "conv2d-fprop-strided": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
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
            "convolution-fprop-3d": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_fprop",
                image_dimensions=[1, 4, 5, 6, 7],
                filter_dimensions=[8, 2, 3, 2, 3],
                pre_padding=[1, 0, 1],
                post_padding=[1, 1, 0],
                stride=[1, 2, 1],
                dilation=[1, 1, 2],
                groups=2,
                data_type="bfloat16",
            ),
            "convolution-dgrad-autotune": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_dgrad",
                image_dimensions=[2, 4, 7, 8],
                filter_dimensions=[6, 2, 3, 3],
                pre_padding=[1, 0],
                post_padding=[0, 1],
                stride=[1, 1],
                dilation=[1, 1],
                groups=2,
                convolution_mode=1,
                data_type="bfloat16",
                autotune=True,
            ),
            "convolution-dgrad-packed1d-stride2-fp32": (
                convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
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
            "convolution-dgrad-dense-stride2-fp16": (
                convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
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
                )
            ),
            "convolution-wgrad": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 6, 8, 9],
                filter_dimensions=[9, 2, 2, 3],
                pre_padding=[0, 1],
                post_padding=[1, 0],
                stride=[2, 1],
                dilation=[1, 2],
                groups=3,
                data_type="bfloat16",
            ),
            "convolution-wgrad-p5": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 32, 40, 40],
                filter_dimensions=[64, 32, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="float32",
                autotune=True,
            ),
            "convolution-wgrad-p5-fp16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 32, 40, 40],
                filter_dimensions=[64, 32, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="float16",
            ),
            "convolution-wgrad-p5-bf16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 32, 40, 40],
                filter_dimensions=[64, 32, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="bfloat16",
                autotune=True,
            ),
            "convolution-wgrad-stem-fp16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 3, 640, 640],
                filter_dimensions=[16, 3, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="float16",
            ),
            "convolution-wgrad-stem-bf16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 3, 640, 640],
                filter_dimensions=[96, 3, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="bfloat16",
                autotune=True,
            ),
            "convolution-wgrad-standard-stride2-fp16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[8, 64, 56, 56],
                filter_dimensions=[128, 64, 3, 3],
                pre_padding=[1, 1],
                post_padding=[1, 1],
                stride=[2, 2],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="float16",
                autotune=True,
            ),
            "convolution-wgrad-standard-1x1-bf16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[8, 64, 28, 28],
                filter_dimensions=[128, 64, 1, 1],
                pre_padding=[0, 0],
                post_padding=[0, 0],
                stride=[1, 1],
                dilation=[1, 1],
                groups=1,
                convolution_mode=0,
                data_type="bfloat16",
            ),
            "convolution-wgrad-nd-1d-fp16-autotune": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[16, 32, 256],
                filter_dimensions=[64, 32, 3],
                pre_padding=[1],
                post_padding=[1],
                stride=[1],
                dilation=[1],
                groups=1,
                convolution_mode=0,
                data_type="float16",
                autotune=True,
            ),
            "convolution-wgrad-nd-3d-asymmetric-bf16": convolution_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="convolution_wgrad",
                image_dimensions=[1, 8, 10, 12, 14],
                filter_dimensions=[12, 8, 2, 3, 3],
                pre_padding=[1, 0, 1],
                post_padding=[0, 1, 2],
                stride=[1, 1, 1],
                dilation=[1, 1, 1],
                groups=1,
                convolution_mode=0,
                data_type="bfloat16",
            ),
        }
        for operation in (
            "convolution_fprop",
            "convolution_dgrad",
            "convolution_wgrad",
        ):
            for precision in (1, 2):
                request = convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation=operation,
                    image_dimensions=[1, 64, 40, 40],
                    filter_dimensions=[128, 64, 3, 3],
                    pre_padding=[1, 1],
                    post_padding=[1, 1],
                    stride=[2, 2],
                    dilation=[1, 1],
                    data_type="float32",
                )
                request["graph"]["nodes"][0]["attributes"][
                    "input_precision"
                ] = precision
                valid_convolution_requests[
                    f"{operation}-precision-{precision}"
                ] = request
        for operation in ("convolution_dgrad", "convolution_wgrad"):
            for precision in (1, 2):
                request = convolution_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation=operation,
                    image_dimensions=[2, 8, 16, 16],
                    filter_dimensions=[16, 8, 3, 3],
                    pre_padding=[1, 1],
                    post_padding=[1, 1],
                    stride=[1, 1],
                    dilation=[1, 1],
                    data_type="float32",
                )
                request["graph"]["nodes"][0]["attributes"][
                    "input_precision"
                ] = precision
                valid_convolution_requests[
                    f"{operation}-generic-precision-{precision}"
                ] = request
        valid_convolution_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_convolution_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_convolution_artifacts[name] = (request_path, artifact)
            precision = request["graph"]["nodes"][0]["attributes"].get(
                "input_precision", 0
            )
            manifest = json.loads((artifact / "manifest.json").read_text())
            if precision:
                for stage in manifest["program"]["stages"]:
                    if stage["function"] in {
                        "conv_dgrad_nd_kernel",
                        "conv_wgrad_nd_kernel",
                    }:
                        for variant in stage["variants"]:
                            if variant["full_signature"].split(",")[39] != str(
                                int(precision == 2)
                            ):
                                fail(
                                    "explicit backward convolution precision was overwritten"
                                )
        valid_add_square_requests = {
            "add-square-fp32": add_square_request_fixture(
                identity, arguments.flagdnn_version
            ),
            "add-square-bf16-autotune": add_square_request_fixture(
                identity,
                arguments.flagdnn_version,
                data_type="bfloat16",
                autotune=True,
            ),
        }
        valid_add_square_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_add_square_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_add_square_artifacts[name] = (request_path, artifact)
        valid_conv_bias_relu_requests = {
            "conv-bias-relu-fp32": conv_bias_relu_request_fixture(
                identity, arguments.flagdnn_version
            ),
            "conv-bias-relu-bf16-autotune": (
                conv_bias_relu_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    data_type="bfloat16",
                    autotune=True,
                )
            ),
        }
        valid_conv_bias_relu_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_conv_bias_relu_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_conv_bias_relu_artifacts[name] = (request_path, artifact)
        valid_normalization_requests = {
            "layernorm-fp16-autotune": normalization_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="layernorm",
                data_type="float16",
                autotune=True,
            ),
            "rmsnorm-bf16-autotune": normalization_request_fixture(
                identity,
                arguments.flagdnn_version,
                operation="rmsnorm",
                data_type="bfloat16",
                autotune=True,
            ),
            "batchnorm-fp16-channels-last-autotune": (
                normalization_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation="batchnorm",
                    data_type="float16",
                    autotune=True,
                )
            ),
            "batchnorm-fp32-row-major-autotune": (
                normalization_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation="batchnorm",
                    autotune=True,
                    row_major=True,
                )
            ),
            "batchnorm-inference-bf16-channels-last-autotune": (
                normalization_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation="batchnorm_inference",
                    data_type="bfloat16",
                    autotune=True,
                )
            ),
            "batchnorm-inference-fp32-row-major-autotune": (
                normalization_request_fixture(
                    identity,
                    arguments.flagdnn_version,
                    operation="batchnorm_inference",
                    autotune=True,
                    row_major=True,
                )
            ),
        }
        valid_normalization_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_normalization_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_normalization_artifacts[name] = (request_path, artifact)
        valid_attention_requests = {
            graph_kind: capture_attention_request_fixture(
                executable=capture_executable,
                capture_compiler=capture_compiler,
                python=python,
                root=root,
                graph_kind=graph_kind,
                identity=identity,
            )
            for graph_kind in (
                "sdpa",
                "sdpa_backward",
                "sdpa_fp8",
                "sdpa_fp8_backward",
            )
        }
        valid_attention_artifacts: dict[str, tuple[Path, Path]] = {}
        for name, request in valid_attention_requests.items():
            request_path, artifact = compile_fixture(
                provider, root, name, request
            )
            run_parser(executable, request_path, artifact, valid=True)
            valid_attention_artifacts[name] = (request_path, artifact)

        core_artifact = root / "core-handoff-artifact"
        shutil.copytree(dense_artifact, core_artifact)
        (core_artifact / "request.json").write_bytes(
            dense_request.read_bytes()
        )
        run_parser(executable, dense_request, core_artifact, valid=True)

        cache_artifact = root / "cache-layout-artifact"
        shutil.copytree(dense_artifact, cache_artifact)
        (cache_artifact / "tuning").mkdir()
        (cache_artifact / "tuning/stage-0.json").write_text(
            "{}\n", encoding="utf-8"
        )
        run_parser(executable, dense_request, cache_artifact, valid=True)

        matrix = Matrix(
            root,
            executable,
            dense_request,
            dense_artifact,
        )
        manifest_matrix(matrix)
        request_matrix(matrix)
        filesystem_matrix(matrix)
        matrix.reject(
            "context target mismatch",
            target="musa-mtgpu-cc32-w32",
        )
        matrix.reject("malformed context target", target="invalid")

        ternary_request_path, ternary_artifact = valid_ternary_artifacts[
            "ternary-binary-select"
        ]
        ternary_matrix = Matrix(
            root,
            executable,
            ternary_request_path,
            ternary_artifact,
        )
        ternary_request_matrix(ternary_matrix)

        layout_request_path, layout_artifact = valid_layout_artifacts[
            "layout-reshape-autotune"
        ]
        layout_matrix = Matrix(
            root,
            executable,
            layout_request_path,
            layout_artifact,
        )
        layout_request_matrix(layout_matrix)

        reduction_request_path, reduction_artifact = valid_reduction_artifacts[
            "reduction-avg-3d-autotune"
        ]
        reduction_matrix = Matrix(
            root,
            executable,
            reduction_request_path,
            reduction_artifact,
        )
        reduction_request_matrix(reduction_matrix)
        reduction_manifest_matrix(reduction_matrix)

        matmul_request_path, matmul_artifact = valid_matmul_artifacts[
            "matmul-bf16-broadcast-autotune"
        ]
        matmul_matrix = Matrix(
            root,
            executable,
            matmul_request_path,
            matmul_artifact,
        )
        matmul_request_matrix(matmul_matrix)
        matmul_manifest_matrix(matmul_matrix)

        convolution_request_path, convolution_artifact = (
            valid_convolution_artifacts["convolution-dgrad-autotune"]
        )
        convolution_matrix = Matrix(
            root,
            executable,
            convolution_request_path,
            convolution_artifact,
        )
        convolution_request_matrix(convolution_matrix)
        convolution_manifest_matrix(convolution_matrix)

        dense_dgrad_request_path, dense_dgrad_artifact = (
            valid_convolution_artifacts["convolution-dgrad-dense-stride2-fp16"]
        )
        dense_dgrad_matrix = Matrix(
            root,
            executable,
            dense_dgrad_request_path,
            dense_dgrad_artifact,
        )
        dense_dgrad_manifest_matrix(dense_dgrad_matrix)

        p5_request_path, p5_artifact = valid_convolution_artifacts[
            "convolution-wgrad-p5"
        ]
        p5_matrix = Matrix(
            root,
            executable,
            p5_request_path,
            p5_artifact,
        )
        p5_wgrad_manifest_matrix(p5_matrix)

        stem_request_path, stem_artifact = valid_convolution_artifacts[
            "convolution-wgrad-stem-bf16"
        ]
        stem_matrix = Matrix(
            root,
            executable,
            stem_request_path,
            stem_artifact,
        )
        stem_wgrad_manifest_matrix(stem_matrix)

        standard_request_path, standard_artifact = valid_convolution_artifacts[
            "convolution-wgrad-standard-stride2-fp16"
        ]
        standard_matrix = Matrix(
            root,
            executable,
            standard_request_path,
            standard_artifact,
        )
        standard_wgrad_manifest_matrix(standard_matrix)

        nd_wgrad_request_path, nd_wgrad_artifact = valid_convolution_artifacts[
            "convolution-wgrad-nd-1d-fp16-autotune"
        ]
        nd_wgrad_matrix = Matrix(
            root,
            executable,
            nd_wgrad_request_path,
            nd_wgrad_artifact,
        )
        standard_wgrad_manifest_matrix(
            nd_wgrad_matrix, label_prefix="ND packed Wgrad"
        )

        add_square_request_path, add_square_artifact = (
            valid_add_square_artifacts["add-square-bf16-autotune"]
        )
        add_square_matrix = Matrix(
            root,
            executable,
            add_square_request_path,
            add_square_artifact,
        )
        add_square_request_matrix(add_square_matrix)
        add_square_manifest_matrix(add_square_matrix)

        conv_bias_relu_request_path, conv_bias_relu_artifact = (
            valid_conv_bias_relu_artifacts["conv-bias-relu-bf16-autotune"]
        )
        conv_bias_relu_matrix = Matrix(
            root,
            executable,
            conv_bias_relu_request_path,
            conv_bias_relu_artifact,
        )
        conv_bias_relu_request_matrix(conv_bias_relu_matrix)
        conv_bias_relu_manifest_matrix(conv_bias_relu_matrix)

        layernorm_request_path, layernorm_artifact = (
            valid_normalization_artifacts["layernorm-fp16-autotune"]
        )
        layernorm_matrix = Matrix(
            root,
            executable,
            layernorm_request_path,
            layernorm_artifact,
        )
        normalization_request_matrix(layernorm_matrix)
        normalization_manifest_matrix(layernorm_matrix, "LayerNorm")

        rmsnorm_request_path, rmsnorm_artifact = valid_normalization_artifacts[
            "rmsnorm-bf16-autotune"
        ]
        rmsnorm_matrix = Matrix(
            root,
            executable,
            rmsnorm_request_path,
            rmsnorm_artifact,
        )
        normalization_manifest_matrix(rmsnorm_matrix, "RMSNorm")

        batchnorm_request_path, batchnorm_artifact = (
            valid_normalization_artifacts[
                "batchnorm-fp16-channels-last-autotune"
            ]
        )
        batchnorm_matrix = Matrix(
            root,
            executable,
            batchnorm_request_path,
            batchnorm_artifact,
        )
        batchnorm_request_matrix(batchnorm_matrix)
        normalization_manifest_matrix(batchnorm_matrix, "BatchNorm")

        batchnorm_inference_request_path, batchnorm_inference_artifact = (
            valid_normalization_artifacts[
                "batchnorm-inference-bf16-channels-last-autotune"
            ]
        )
        batchnorm_inference_matrix = Matrix(
            root,
            executable,
            batchnorm_inference_request_path,
            batchnorm_inference_artifact,
        )
        batchnorm_inference_request_matrix(batchnorm_inference_matrix)
        normalization_manifest_matrix(
            batchnorm_inference_matrix, "BatchNorm inference"
        )

        attention_request_path, attention_artifact = valid_attention_artifacts[
            "sdpa_fp8_backward"
        ]
        attention_matrix = Matrix(
            root,
            executable,
            attention_request_path,
            attention_artifact,
        )
        attention_request_matrix(attention_matrix)
        attention_manifest_matrix(attention_matrix)

        non_directory = root / "artifact-is-file"
        non_directory.write_text("x", encoding="utf-8")
        run_parser(
            executable,
            dense_request,
            non_directory,
            valid=False,
        )
        print(
            json.dumps(
                {
                    "valid_cases": (
                        4
                        + extended_valid
                        + len(valid_binary_requests)
                        + len(valid_unary_requests)
                        + len(valid_ternary_requests)
                        + len(valid_layout_requests)
                        + len(valid_reduction_requests)
                        + len(valid_matmul_requests)
                        + len(valid_convolution_requests)
                        + len(valid_add_square_requests)
                        + len(valid_conv_bias_relu_requests)
                        + len(valid_normalization_requests)
                        + len(valid_attention_requests)
                    ),
                    "rejected_mutations": (
                        sigmoid_rejected
                        + extended_rejected
                        + matrix.count
                        + ternary_matrix.count
                        + layout_matrix.count
                        + reduction_matrix.count
                        + matmul_matrix.count
                        + convolution_matrix.count
                        + dense_dgrad_matrix.count
                        + p5_matrix.count
                        + add_square_matrix.count
                        + conv_bias_relu_matrix.count
                        + layernorm_matrix.count
                        + rmsnorm_matrix.count
                        + batchnorm_matrix.count
                        + batchnorm_inference_matrix.count
                        + attention_matrix.count
                        + 1
                    ),
                    "parser_processes": (
                        sigmoid_rejected
                        + extended_valid
                        + extended_rejected
                        + matrix.count
                        + ternary_matrix.count
                        + layout_matrix.count
                        + reduction_matrix.count
                        + matmul_matrix.count
                        + convolution_matrix.count
                        + dense_dgrad_matrix.count
                        + p5_matrix.count
                        + add_square_matrix.count
                        + conv_bias_relu_matrix.count
                        + layernorm_matrix.count
                        + rmsnorm_matrix.count
                        + batchnorm_matrix.count
                        + batchnorm_inference_matrix.count
                        + 5
                        + len(valid_binary_requests)
                        + len(valid_unary_requests)
                        + len(valid_ternary_requests)
                        + len(valid_layout_requests)
                        + len(valid_reduction_requests)
                        + len(valid_matmul_requests)
                        + len(valid_convolution_requests)
                        + len(valid_add_square_requests)
                        + len(valid_conv_bias_relu_requests)
                        + len(valid_normalization_requests)
                        + len(valid_attention_requests)
                        + attention_matrix.count
                    ),
                    "torch_loaded": "torch" in sys.modules,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
