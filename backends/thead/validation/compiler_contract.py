#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the THead compiler provider and artifact generation."""

from __future__ import annotations

import copy
import ast
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable


sys.dont_write_bytecode = True

SOURCE_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = SOURCE_ROOT / "compiler"
BACKEND_ROOT = SOURCE_ROOT / "backends"
TARGET = "ppu_contract_cc80"


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def python_cache_entries() -> set[Path]:
    backend = SOURCE_ROOT / "backends/thead"
    return {
        path.resolve()
        for path in backend.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    }


def write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def request(
    identity: str,
    operation: str = "add",
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    if operation in {
        "binary_select",
        "logical_not",
        "logical_and",
        "logical_or",
    }:
        document = request(
            identity,
            "abs" if operation == "logical_not" else "add",
            autotune=autotune,
        )
        graph = document["graph"]
        node = graph["nodes"][0]
        node["type"] = operation
        node["name"] = operation
        mode = {
            "binary_select": 41,
            "logical_not": 24,
            "logical_and": 31,
            "logical_or": 32,
        }[operation]
        node["attributes"]["mode"] = mode
        if operation == "binary_select":
            graph["tensor_count"] = 4
            output = copy.deepcopy(graph["tensors"][0])
            output["uid"] = 4
            graph["tensors"].append(output)
            graph["tensors"][2]["data_type"] = "boolean"
            node["inputs"] = [
                {"name": name, "uid": uid}
                for name, uid in (("a", 1), ("b", 2), ("t", 3))
            ]
            node["outputs"] = [{"name": "output", "uid": 4}]
            node["attributes"] = {"mode": mode, "n_elements": 16}
        else:
            for tensor in graph["tensors"]:
                tensor["data_type"] = "boolean"
            node["compute_data_type"] = "boolean"
            if operation != "logical_not":
                node["attributes"]["pointwise_mode"] = mode
        return document
    if operation == "add_square":

        def binary_node(
            node_id: int,
            operation_type: str,
            left_uid: int,
            right_uid: int,
            output_uid: int,
            mode: int,
        ) -> dict[str, Any]:
            return {
                "id": node_id,
                "type": operation_type,
                "name": operation_type,
                "compute_data_type": "float32",
                "inputs": [
                    {"name": "left", "uid": left_uid},
                    {"name": "right", "uid": right_uid},
                ],
                "outputs": [{"name": "output", "uid": output_uid}],
                "attributes": {
                    "alpha": 1.0,
                    "mode": mode,
                    "n_elements": 16,
                    "pointwise_mode": mode,
                },
            }

        return {
            "schema_version": 3,
            "flagdnn_version": "0.2.0",
            "backend": "thead",
            "target": TARGET,
            "compiler_identity": identity,
            "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
            "graph": {
                "name": "thead compiler add_square contract",
                "tensor_count": 4,
                "tensors": [
                    {
                        "uid": uid,
                        "data_type": "float32",
                        "dimensions": [16],
                        "strides": [1],
                        "alignment": 16,
                        "virtual": uid == 3,
                    }
                    for uid in (1, 2, 3, 4)
                ],
                "node_count": 2,
                "nodes": [
                    binary_node(0, "mul", 2, 2, 3, 18),
                    binary_node(1, "add", 1, 3, 4, 1),
                ],
            },
        }
    lowered_operation = {
        "scale": "mul",
        "leaky_relu": "relu",
    }.get(operation, operation)
    if operation in (
        "relu",
        "leaky_relu",
        "sigmoid",
        "tanh",
        "elu",
        "identity",
        "gelu",
        "sqrt",
        "neg",
        "abs",
        "ceil",
        "floor",
        "exp",
        "log",
        "cos",
        "rsqrt",
        "sin",
        "tan",
        "softplus",
        "swish",
        "gelu_approx_tanh",
        "reciprocal",
        "erf",
    ):
        pointwise_mode = {
            "relu": 2,
            "leaky_relu": 2,
            "sigmoid": 33,
            "tanh": 34,
            "elu": 35,
            "identity": 5,
            "gelu": 36,
            "sqrt": 3,
            "neg": 8,
            "abs": 9,
            "ceil": 10,
            "floor": 12,
            "exp": 6,
            "log": 7,
            "cos": 11,
            "rsqrt": 13,
            "sin": 14,
            "tan": 15,
            "softplus": 37,
            "swish": 38,
            "gelu_approx_tanh": 39,
            "reciprocal": 16,
            "erf": 4,
        }[operation]
        return {
            "schema_version": 3,
            "flagdnn_version": "0.2.0",
            "backend": "thead",
            "target": TARGET,
            "compiler_identity": identity,
            "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
            "graph": {
                "name": f"thead compiler {operation} contract",
                "tensor_count": 2,
                "tensors": [
                    {
                        "uid": uid,
                        "data_type": "float32",
                        "dimensions": [16],
                        "strides": [1],
                        "alignment": 16,
                        "virtual": False,
                    }
                    for uid in (1, 2)
                ],
                "node_count": 1,
                "nodes": [
                    {
                        "id": 0,
                        "type": lowered_operation,
                        "name": f"{operation}_0",
                        "compute_data_type": "float32",
                        "inputs": [{"name": "input", "uid": 1}],
                        "outputs": [{"name": "output", "uid": 2}],
                        "attributes": {
                            "elu_alpha": 1.0,
                            "has_upper_clip": 0,
                            "lower_clip": 0.0,
                            "mode": pointwise_mode,
                            "n_elements": 16,
                            "negative_slope": (
                                0.20000000298023224
                                if operation == "leaky_relu"
                                else 0.0
                            ),
                            "relu_lower_clip": 0.0,
                            "relu_lower_clip_slope": (
                                0.20000000298023224
                                if operation == "leaky_relu"
                                else 0.0
                            ),
                            "relu_upper_clip": 0.0,
                            "relu_upper_clip_set": False,
                            "softplus_beta": 1.0,
                            "swish_beta": (
                                1.25 if operation == "swish" else 1.0
                            ),
                            "upper_clip": 0.0,
                        },
                    }
                ],
            },
        }
    pointwise_mode = {
        "add": 1,
        "sub": 17,
        "mul": 18,
        "scale": 18,
        "min": 20,
        "max": 21,
        "div": 19,
        "mod": 22,
        "pow": 23,
        "sigmoid_backward": 40,
        "cmp_eq": 25,
        "cmp_neq": 26,
        "cmp_gt": 27,
        "cmp_ge": 28,
        "cmp_lt": 29,
        "cmp_le": 30,
    }.get(operation, 1)
    comparison = operation in {
        "cmp_eq",
        "cmp_neq",
        "cmp_gt",
        "cmp_ge",
        "cmp_lt",
        "cmp_le",
    }
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"thead compiler {operation} contract",
            "tensor_count": 3,
            "tensors": [
                {
                    "uid": uid,
                    "data_type": (
                        "boolean" if comparison and uid == 3 else "float32"
                    ),
                    "dimensions": [16],
                    "strides": [1],
                    "alignment": 16,
                    "virtual": False,
                }
                for uid in (1, 2, 3)
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": lowered_operation,
                    "name": f"{operation}_0",
                    "compute_data_type": (
                        "boolean" if comparison else "float32"
                    ),
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "alpha": 1.0,
                        "mode": pointwise_mode,
                        "n_elements": 16,
                        "pointwise_mode": pointwise_mode,
                    },
                }
            ],
        },
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout_request(
    identity: str,
    operation: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    definitions: dict[str, dict[str, Any]] = {
        "reshape": {
            "input_dimensions": [2, 3, 4],
            "input_strides": [12, 4, 1],
            "output_dimensions": [6, 4],
            "output_strides": [4, 1],
            "attributes": {
                "input_rank": 3,
                "output_rank": 2,
                "reshape_mode": 2,
            },
        },
        "transpose": {
            "input_dimensions": [2, 3, 4],
            "input_strides": [12, 4, 1],
            "output_dimensions": [4, 2, 3],
            "output_strides": [1, 12, 4],
            "attributes": {"rank": 3, "permutation": [2, 0, 1]},
        },
        "slice": {
            "input_dimensions": [2, 4, 5],
            "input_strides": [20, 5, 1],
            "output_dimensions": [2, 2, 5],
            "output_strides": [20, 10, 1],
            "attributes": {
                "rank": 3,
                "starts": [0, 1, 0],
                "limits": [2, 4, 5],
                "slice_strides": [1, 2, 1],
            },
        },
    }
    if operation not in definitions:
        raise ValueError(f"unknown layout operation: {operation}")
    definition = definitions[operation]
    attributes = {
        "n_elements": math.prod(definition["output_dimensions"]),
        "input_dimensions": definition["input_dimensions"],
        "input_strides": definition["input_strides"],
        "output_dimensions": definition["output_dimensions"],
        "output_strides": definition["output_strides"],
        **definition["attributes"],
    }
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"thead compiler {operation} contract",
            "tensor_count": 2,
            "tensors": [
                {
                    "uid": 1,
                    "data_type": "float32",
                    "dimensions": definition["input_dimensions"],
                    "strides": definition["input_strides"],
                    "alignment": 16,
                    "virtual": False,
                },
                {
                    "uid": 2,
                    "data_type": "float32",
                    "dimensions": definition["output_dimensions"],
                    "strides": definition["output_strides"],
                    "alignment": 16,
                    "virtual": False,
                },
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "name": f"{operation}_0",
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 1}],
                    "outputs": [{"name": "output", "uid": 2}],
                    "attributes": attributes,
                }
            ],
        },
    }


def reduction_request(
    identity: str,
    operation: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    if operation not in {"reduction_sum", "reduction_avg", "reduction_mul"}:
        raise ValueError(f"unknown reduction operation: {operation}")
    reduction_mode = {
        "reduction_sum": 0,
        "reduction_avg": 1,
        "reduction_mul": 2,
    }[operation]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"thead compiler {operation} contract",
            "tensor_count": 2,
            "tensors": [
                {
                    "uid": 1,
                    "data_type": "float32",
                    "dimensions": [2, 4, 8, 8],
                    "strides": [256, 64, 8, 1],
                    "alignment": 16,
                    "virtual": False,
                },
                {
                    "uid": 2,
                    "data_type": "float32",
                    "dimensions": [2, 1, 8, 8],
                    "strides": [64, 64, 8, 1],
                    "alignment": 16,
                    "virtual": False,
                },
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "name": "reduction",
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 1}],
                    "outputs": [{"name": "output", "uid": 2}],
                    "attributes": {
                        "mode": reduction_mode,
                        "outer": 2,
                        "reduction": 4,
                        "inner": 64,
                        "output_elements": 128,
                        "axis": 1,
                        "keep_dimensions": 1,
                    },
                }
            ],
        },
    }


def batchnorm_request(
    identity: str,
    operation: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    if operation not in {"batchnorm", "batchnorm_inference"}:
        raise ValueError(f"unknown batchnorm operation: {operation}")
    inference = operation == "batchnorm_inference"
    data_dimensions = [2, 8, 16, 16] if inference else [2, 8, 8, 8]
    data_strides = [2048, 256, 16, 1] if inference else [512, 64, 8, 1]
    parameter_dimensions = [1, 8, 1, 1]
    parameter_strides = [8, 1, 1, 1]
    tensor_count = 6 if inference else 10
    tensors = []
    for uid in range(1, tensor_count + 1):
        data_tensor = uid in ({1, 6} if inference else {1, 6})
        tensors.append(
            {
                "uid": uid,
                "data_type": "float32",
                "dimensions": (
                    data_dimensions if data_tensor else parameter_dimensions
                ),
                "strides": data_strides if data_tensor else parameter_strides,
                "alignment": 16,
                "virtual": False,
            }
        )
    if inference:
        inputs = [
            {"name": name, "uid": uid}
            for name, uid in (
                ("x", 1),
                ("mean", 2),
                ("inv_variance", 3),
                ("scale", 4),
                ("bias", 5),
            )
        ]
        outputs = [{"name": "y", "uid": 6}]
        attributes: dict[str, Any] = {
            "n_elements": 4096,
            "channels": 8,
            "spatial": 256,
            "rank": 4,
            "dimensions": data_dimensions,
            "x_strides": data_strides,
            "y_strides": data_strides,
        }
    else:
        inputs = [
            {"name": name, "uid": uid}
            for name, uid in (
                ("x", 1),
                ("scale", 2),
                ("bias", 3),
                ("previous_running_mean", 4),
                ("previous_running_variance", 5),
            )
        ]
        outputs = [
            {"name": name, "uid": uid}
            for name, uid in (
                ("y", 6),
                ("mean", 7),
                ("inv_variance", 8),
                ("next_running_mean", 9),
                ("next_running_variance", 10),
            )
        ]
        attributes = {
            "n_elements": 1024,
            "batch": 2,
            "channels": 8,
            "spatial": 64,
            "rank": 4,
            "epsilon": 0.0010000000474974513,
            "momentum": 0.10000000149011612,
            "dimensions": data_dimensions,
            "x_strides": data_strides,
            "y_strides": data_strides,
        }
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {
            "heuristic_modes": ["A", "FALLBACK"],
            "autotune": autotune,
        },
        "graph": {
            "name": f"thead compiler {operation} contract",
            "tensor_count": tensor_count,
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
    }


def normalization_request(
    identity: str,
    operation: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    if operation not in {"layernorm", "rmsnorm"}:
        raise ValueError(f"unknown normalization operation: {operation}")
    layernorm = operation == "layernorm"
    tensors = [
        {
            "uid": 1,
            "data_type": "float32",
            "dimensions": [2, 5, 17],
            "strides": [85, 17, 1],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 2,
            "data_type": "float32",
            "dimensions": [1, 1, 17],
            "strides": [17, 17, 1],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 3,
            "data_type": "float32",
            "dimensions": [1, 1, 17],
            "strides": [17, 17, 1],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 4,
            "data_type": "float32",
            "dimensions": [2, 5, 17],
            "strides": [85, 17, 1],
            "alignment": 16,
            "virtual": False,
        },
    ]
    if layernorm:
        tensors.append(
            {
                "uid": 5,
                "data_type": "float32",
                "dimensions": [2, 5, 1],
                "strides": [5, 1, 1],
                "alignment": 16,
                "virtual": False,
            }
        )
    tensors.append(
        {
            "uid": 6 if layernorm else 5,
            "data_type": "float32",
            "dimensions": [2, 5, 1],
            "strides": [5, 1, 1],
            "alignment": 16,
            "virtual": False,
        }
    )
    outputs = [{"name": "y", "uid": 4}]
    if layernorm:
        outputs.append({"name": "mean", "uid": 5})
    outputs.append({"name": "inv_variance", "uid": 6 if layernorm else 5})
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {
            "heuristic_modes": ["A", "FALLBACK"],
            "autotune": autotune,
        },
        "graph": {
            "name": f"thead compiler {operation} contract",
            "tensor_count": len(tensors),
            "tensors": tensors,
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "name": operation,
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "x", "uid": 1},
                        {"name": "scale", "uid": 2},
                        {"name": "bias", "uid": 3},
                    ],
                    "outputs": outputs,
                    "attributes": {
                        "rows": 10,
                        "normalized_elements": 17,
                        "epsilon": 0.0010000000474974513,
                        "forward_phase": 2,
                    },
                }
            ],
        },
    }


def matmul_request(
    identity: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {
            "heuristic_modes": ["A", "FALLBACK"],
            "autotune": autotune,
        },
        "graph": {
            "name": "thead compiler matmul contract",
            "tensor_count": 3,
            "tensors": [
                {
                    "uid": 1,
                    "data_type": "float32",
                    "dimensions": [4, 16, 32],
                    "strides": [512, 32, 1],
                    "alignment": 16,
                    "virtual": False,
                },
                {
                    "uid": 2,
                    "data_type": "float32",
                    "dimensions": [4, 32, 24],
                    "strides": [768, 24, 1],
                    "alignment": 16,
                    "virtual": False,
                },
                {
                    "uid": 3,
                    "data_type": "float32",
                    "dimensions": [4, 16, 24],
                    "strides": [384, 24, 1],
                    "alignment": 16,
                    "virtual": False,
                },
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "matmul",
                    "name": "matmul",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "a", "uid": 1},
                        {"name": "b", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {"batch": 4, "m": 16, "n": 24, "k": 32},
                }
            ],
        },
    }


def convolution_request(
    identity: str,
    operation: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    if operation not in {
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    }:
        raise ValueError(f"unknown convolution operation: {operation}")
    tensors = [
        {
            "uid": 1,
            "data_type": "float32",
            "dimensions": [1, 2, 5, 5],
            "strides": [50, 25, 5, 1],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 2,
            "data_type": "float32",
            "dimensions": [2, 2, 3, 3],
            "strides": [18, 9, 3, 1],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 3,
            "data_type": "float32",
            "dimensions": [1, 2, 5, 5],
            "strides": [50, 25, 5, 1],
            "alignment": 16,
            "virtual": False,
        },
    ]
    if operation == "convolution_fprop":
        inputs = [
            {"name": "input", "uid": 1},
            {"name": "filter", "uid": 2},
        ]
        outputs = [{"name": "output", "uid": 3}]
        n_outputs = 50
    elif operation == "convolution_dgrad":
        inputs = [
            {"name": "dy", "uid": 3},
            {"name": "w", "uid": 2},
        ]
        outputs = [{"name": "dx", "uid": 1}]
        n_outputs = 50
    else:
        inputs = [
            {"name": "dy", "uid": 3},
            {"name": "x", "uid": 1},
        ]
        outputs = [{"name": "dw", "uid": 2}]
        n_outputs = 36
    attributes: dict[str, Any] = {
        "spatial_rank": 2,
        "groups": 1,
        "n_outputs": n_outputs,
        "pre_padding": [1, 1],
        "post_padding": [1, 1],
        "stride": [1, 1],
        "dilation": [1, 1],
    }
    if operation != "convolution_fprop":
        attributes["convolution_mode"] = 0
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {
            "heuristic_modes": ["A", "FALLBACK"],
            "autotune": autotune,
        },
        "graph": {
            "name": f"thead compiler {operation} contract",
            "tensor_count": 3,
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
    }


def conv_bias_relu_request(
    identity: str,
    *,
    autotune: bool = False,
) -> dict[str, Any]:
    output_dimensions = [2, 16, 16, 16]
    output_strides = [4096, 1, 256, 16]
    tensors = [
        {
            "uid": 1,
            "data_type": "float32",
            "dimensions": [2, 8, 16, 16],
            "strides": [2048, 1, 128, 8],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 2,
            "data_type": "float32",
            "dimensions": [16, 8, 3, 3],
            "strides": [72, 1, 24, 8],
            "alignment": 16,
            "virtual": False,
        },
        {
            "uid": 3,
            "data_type": "float32",
            "dimensions": [1, 16, 1, 1],
            "strides": [16, 1, 16, 16],
            "alignment": 16,
            "virtual": False,
        },
        *[
            {
                "uid": uid,
                "data_type": "float32",
                "dimensions": output_dimensions,
                "strides": output_strides,
                "alignment": 16,
                "virtual": uid in (4, 5),
            }
            for uid in (4, 5, 6)
        ],
    ]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "thead",
        "target": TARGET,
        "compiler_identity": identity,
        "build_options": {
            "heuristic_modes": ["A", "FALLBACK"],
            "autotune": autotune,
        },
        "graph": {
            "name": "thead compiler conv_bias_relu contract",
            "tensor_count": 6,
            "tensors": tensors,
            "node_count": 3,
            "nodes": [
                {
                    "id": 0,
                    "type": "convolution_fprop",
                    "name": "convolution",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "input", "uid": 1},
                        {"name": "filter", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 4}],
                    "attributes": {
                        "spatial_rank": 2,
                        "groups": 1,
                        "n_outputs": 8192,
                        "pre_padding": [1, 1],
                        "post_padding": [1, 1],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                },
                {
                    "id": 1,
                    "type": "add",
                    "name": "bias_add",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 4},
                        {"name": "right", "uid": 3},
                    ],
                    "outputs": [{"name": "output", "uid": 5}],
                    "attributes": {
                        "alpha": 1.0,
                        "mode": 1,
                        "n_elements": 8192,
                        "pointwise_mode": 1,
                    },
                },
                {
                    "id": 2,
                    "type": "relu",
                    "name": "relu",
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 5}],
                    "outputs": [{"name": "output", "uid": 6}],
                    "attributes": {
                        "elu_alpha": 1.0,
                        "has_upper_clip": 0,
                        "lower_clip": 0.0,
                        "mode": 2,
                        "n_elements": 8192,
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
    }


def registry_sha256() -> str:
    """Hash the ordered common/platform registry byte snapshots."""

    digest = hashlib.sha256()
    for path in (
        SOURCE_ROOT / "kernels/registry.json",
        SOURCE_ROOT / "backends/thead/kernels/registry.json",
    ):
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def load_artifact(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    require(
        manifest_path.is_file(),
        "THead compiler did not write manifest.json",
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def assert_add_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "add.json"
    output = temporary / "add-artifact"
    document = request(identity)
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Add unexpectedly returned {result!r}",
    )
    require(
        result
        == {
            "artifact_directory": str(output),
            "backend": "thead",
            "execution_engine": "libtriton_jit",
            "node_count": 1,
            "provider": "thead_triton",
            "schema_version": 1,
            "stage_count": 1,
            "status": "success",
            "target": TARGET,
            "warp_size": 32,
            "workspace_alignment": 256,
            "workspace_size": 0,
        },
        "Add compiler success result is not exact",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("backend") == "thead"
        and manifest.get("target") == TARGET
        and manifest.get("engine") == "libtriton_jit"
        and manifest.get("warp_size") == 32,
        "Add manifest backend/target/engine identity is invalid",
    )
    require(
        manifest.get("compiler")
        == {
            "provider": "thead_triton",
            "provider_version": "1",
            "identity_sha256": identity,
        },
        "Add manifest compiler identity is invalid",
    )
    require(
        manifest.get("request_sha256") == sha256_file(request_path),
        "Add manifest request hash does not match exact Graph IR bytes",
    )
    require(
        manifest.get("external_uids") == [1, 2, 3]
        and manifest.get("workspace") == {"alignment": 256, "size": 0},
        "Add manifest binding/workspace contract is invalid",
    )
    require(
        manifest.get("tensor_count") == 3
        and [tensor.get("uid") for tensor in manifest.get("tensors", [])]
        == [1, 2, 3],
        "Add manifest tensor table is invalid",
    )

    program = manifest.get("program", {})
    stages = program.get("stages", [])
    require(
        program.get("schema_version") == 1
        and program.get("stage_count") == 1
        and len(stages) == 1,
        "Add manifest must contain exactly one execution stage",
    )
    stage = stages[0]
    require(
        stage.get("stage_id") == 0
        and stage.get("source_node_ids") == [0]
        and stage.get("dependencies") == []
        and stage.get("operation") == "add"
        and stage.get("tuning") is None,
        "Add stage topology or tuning state is invalid",
    )
    kernel = stage.get("kernel", {})
    require(
        kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "binary_contiguous_kernel",
        "Add did not select the common contiguous kernel",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/binary.py"),
        "Add materialized source identity is invalid",
    )
    require(
        kernel.get("registry_sha256") == registry_sha256(),
        "Add registry identity is missing or invalid",
    )

    variants = stage.get("variants", [])
    require(len(variants) == 1, "Add must emit one non-autotuned variant")
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,1,1.0,256"
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        "Add variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 4
        and [argument.get("uid") for argument in arguments[:3]] == [1, 2, 3]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "tensor", "scalar_i32"]
        and arguments[3]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        },
        "Add runtime ABI or tensor UID order is invalid",
    )
    launch = variant.get("launch", {})
    block_size = int(variant["full_signature"].rsplit(",", 1)[1])
    require(
        launch.get("grid") == [(16 + block_size - 1) // block_size, 1, 1]
        and launch.get("block") == [128, 1, 1]
        and launch.get("shared_memory") == 0,
        "Add launch geometry does not minimally cover n_elements",
    )

    alpha_document = request(identity)
    alpha_document["graph"]["nodes"][0]["attributes"]["alpha"] = 0.5
    alpha_path = temporary / "add-alpha-half.json"
    alpha_output = temporary / "add-alpha-half-artifact"
    write_json(alpha_path, alpha_document)
    alpha_result = provider.compile_request(
        alpha_path, alpha_output, "libtriton_jit"
    )
    alpha_variant = load_artifact(alpha_output)["program"]["stages"][0][
        "variants"
    ][0]
    require(
        alpha_result.get("status") == "success"
        and alpha_variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,1,0.5,256",
        "Add alpha specialization is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            "bad-broadcast",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "dimensions", [1]
            ),
            "broadcast",
        ),
        (
            "bad-dtype",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 17
            ),
            "mode",
        ),
        (
            "bad-lowered-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "pointwise_mode", 17
            ),
            "pointwise_mode",
        ),
        (
            "bad-uid",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].__setitem__(
                "uid", 99
            ),
            "unknown tensor",
        ),
        (
            "bad-output-shape",
            lambda value: value["graph"]["tensors"][2].__setitem__(
                "dimensions", [8]
            ),
            "output shape",
        ),
        (
            "storage-size-overflow",
            lambda value: (
                [
                    tensor.update({"dimensions": [2], "strides": [2**62]})
                    for tensor in value["graph"]["tensors"]
                ],
                value["graph"]["nodes"][0]["attributes"].__setitem__(
                    "n_elements", 2
                ),
            ),
            "storage size",
        ),
    ]
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Add mutation {name} wrote an artifact directory",
        )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_mul_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "mul.json"
    output = temporary / "mul-artifact"
    document = request(identity, "mul")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Mul unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path),
        "Mul manifest request hash does not match exact Graph IR bytes",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Mul must emit exactly one execution stage")
    stage = stages[0]
    require(
        stage.get("operation") == "mul" and stage.get("tuning") is None,
        "Mul stage operation or tuning state is invalid",
    )
    kernel = stage.get("kernel", {})
    require(
        kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "binary_contiguous_kernel",
        "Mul did not select the common contiguous kernel",
    )
    variants = stage.get("variants", [])
    require(len(variants) == 1, "Mul must emit one non-autotuned variant")
    variant = variants[0]
    require(
        variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,18,1.0,256",
        "Mul variant did not specialize FLAGDNN_POINTWISE_MUL (18)",
    )
    require(
        [argument.get("uid") for argument in variant.get("arguments", [])[:3]]
        == [1, 2, 3]
        and variant.get("arguments", [])[-1]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        },
        "Mul runtime ABI or tensor UID order is invalid",
    )

    for name, attribute, value, detail in (
        ("mul-bad-mode", "mode", 1, "mode"),
        ("mul-bad-lowered-mode", "pointwise_mode", 1, "pointwise_mode"),
    ):
        mutated = copy.deepcopy(document)
        mutated["graph"]["nodes"][0]["attributes"][attribute] = value
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Mul mutation {name} wrote an artifact directory",
        )

    autotune_request_path = temporary / "mul-autotune.json"
    autotune_output = temporary / "mul-autotune-artifact"
    write_json(autotune_request_path, request(identity, "mul", autotune=True))
    autotune_result = provider.compile_request(
        autotune_request_path, autotune_output, "libtriton_jit"
    )
    require(
        autotune_result.get("status") == "success",
        "autotuned Mul did not compile",
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[-3] == "18"
            for candidate in autotune_variants
        ),
        "autotuned Mul candidates are incomplete or use the wrong mode",
    )
    require(
        autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Mul candidate identity is missing",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_scale_alias_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "scale.json"
    output = temporary / "scale-artifact"
    document = request(identity, "scale")
    node = document["graph"]["nodes"][0]
    require(
        node["type"] == "mul"
        and node["name"] == "scale_0"
        and node["attributes"]["mode"] == 18
        and node["attributes"]["pointwise_mode"] == 18,
        "Scale alias contract did not preserve the canonical Mul Graph IR",
    )
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Scale-to-Mul alias unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path),
        "Scale alias manifest request hash does not match exact Graph IR bytes",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(
        len(stages) == 1, "Scale alias must emit exactly one execution stage"
    )
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == "mul"
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "binary_contiguous_kernel"
        and len(variants) == 1
        and variants[0].get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,18,1.0,256",
        "Scale alias did not reuse the exact qualified Mul stage",
    )

    wrong_mode = copy.deepcopy(document)
    wrong_mode["graph"]["nodes"][0]["attributes"]["mode"] = 1
    wrong_mode_path = temporary / "scale-bad-mode.json"
    wrong_mode_output = temporary / "scale-bad-mode-output"
    write_json(wrong_mode_path, wrong_mode)
    expect_value_error(
        lambda: provider.compile_request(
            wrong_mode_path, wrong_mode_output, "libtriton_jit"
        ),
        "mode",
    )
    require(
        not wrong_mode_output.exists(),
        "rejected Scale alias mode mutation wrote an artifact directory",
    )

    phantom_operation = copy.deepcopy(document)
    phantom_operation["graph"]["nodes"][0]["type"] = "scale"
    phantom_path = temporary / "scale-phantom-operation.json"
    phantom_output = temporary / "scale-phantom-operation-output"
    write_json(phantom_path, phantom_operation)
    unsupported = provider.compile_request(
        phantom_path, phantom_output, "libtriton_jit"
    )
    require(
        unsupported.get("status") == "unsupported"
        and unsupported.get("reason_code")
        == "operation_family_not_implemented"
        and unsupported.get("operation_types") == ["scale"]
        and not phantom_output.exists(),
        "THead compiler accepted a non-canonical Scale IR operation",
    )

    autotune_path = temporary / "scale-autotune.json"
    autotune_output = temporary / "scale-autotune-artifact"
    write_json(autotune_path, request(identity, "scale", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and autotune_stage.get("operation") == "mul"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[-3] == "18"
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Scale alias did not reuse the qualified Mul candidates",
    )

    if compile_kernel:
        compile_add_kernel(output, variants[0], kernel)


def assert_relu_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "relu.json"
    output = temporary / "relu-artifact"
    document = request(identity, "relu")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Relu unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2,
        "Relu manifest request, binding, or tensor identity is invalid",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Relu must emit exactly one execution stage")
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == "relu"
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "unary_pointwise_contiguous_kernel"
        and len(variants) == 1,
        "Relu did not select one non-autotuned common unary stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/unary.py")
        and kernel.get("registry_sha256") == registry_sha256(),
        "Relu common unary source or registry identity is invalid",
    )
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == ("*fp32:16,*fp32:16,i32,2,0.0,0.0,0.0,0," "1.0,1.0,1.0,1,256")
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        "Relu variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 3
        and [argument.get("uid") for argument in arguments[:2]] == [1, 2]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "scalar_i32"]
        and arguments[2]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        }
        and variant.get("launch")
        == {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        "Relu runtime ABI or launch geometry is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            "relu-bad-stride",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "strides", [0]
            ),
            "non-overlapping",
        ),
        (
            "relu-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "relu-bad-output-shape",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "dimensions", [8]
            ),
            "shape",
        ),
        (
            "relu-bad-slope",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "negative_slope", 0.25
            ),
            "negative slope is not qualified",
        ),
        (
            "relu-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 9
            ),
            "mode",
        ),
        (
            "relu-bad-lowered-slope-alias",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "relu_lower_clip_slope", 0.25
            ),
            "negative slope is not qualified",
        ),
        (
            "relu-bad-upper-clip",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "has_upper_clip", 1
            ),
            "default attributes",
        ),
        (
            "relu-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
        (
            "relu-bad-port",
            lambda value: value["graph"]["nodes"][0]["inputs"][0].__setitem__(
                "name", "x"
            ),
            "input",
        ),
    ]
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Relu mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / "relu-autotune.json"
    autotune_output = temporary / "relu-autotune-artifact"
    write_json(autotune_path, request(identity, "relu", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[3:12]
            == ["2", "0.0", "0.0", "0.0", "0", "1.0", "1.0", "1.0", "1"]
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Relu candidates are incomplete or have wrong constants",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_sigmoid_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "sigmoid.json"
    output = temporary / "sigmoid-artifact"
    document = request(identity, "sigmoid")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Sigmoid unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2,
        "Sigmoid manifest request, binding, or tensor identity is invalid",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Sigmoid must emit exactly one execution stage")
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == "sigmoid"
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "unary_pointwise_contiguous_kernel"
        and len(variants) == 1,
        "Sigmoid did not select one non-autotuned common unary stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/unary.py")
        and kernel.get("registry_sha256") == registry_sha256(),
        "Sigmoid common unary source or registry identity is invalid",
    )
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == ("*fp32:16,*fp32:16,i32,33,0.0,0.0,0.0,0," "1.0,1.0,1.0,1,256")
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        "Sigmoid variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 3
        and [argument.get("uid") for argument in arguments[:2]] == [1, 2]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "scalar_i32"]
        and arguments[2]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        }
        and variant.get("launch")
        == {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        "Sigmoid runtime ABI or launch geometry is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            "sigmoid-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 2
            ),
            "mode",
        ),
        (
            "sigmoid-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "sigmoid-bad-attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "swish_beta", 2.0
            ),
            "default attributes",
        ),
        (
            "sigmoid-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
    ]
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Sigmoid mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / "sigmoid-autotune.json"
    autotune_output = temporary / "sigmoid-autotune-artifact"
    write_json(autotune_path, request(identity, "sigmoid", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[3:12]
            == ["33", "0.0", "0.0", "0.0", "0", "1.0", "1.0", "1.0", "1"]
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Sigmoid candidates are incomplete or have wrong constants",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_tanh_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "tanh.json"
    output = temporary / "tanh-artifact"
    document = request(identity, "tanh")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Tanh unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2,
        "Tanh manifest request, binding, or tensor identity is invalid",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Tanh must emit exactly one execution stage")
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == "tanh"
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "unary_pointwise_contiguous_kernel"
        and len(variants) == 1,
        "Tanh did not select one non-autotuned common unary stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/unary.py")
        and kernel.get("registry_sha256") == registry_sha256(),
        "Tanh common unary source or registry identity is invalid",
    )
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == ("*fp32:16,*fp32:16,i32,34,0.0,0.0,0.0,0," "1.0,1.0,1.0,1,256")
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        "Tanh variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 3
        and [argument.get("uid") for argument in arguments[:2]] == [1, 2]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "scalar_i32"]
        and arguments[2]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        }
        and variant.get("launch")
        == {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        "Tanh runtime ABI or launch geometry is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            "tanh-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 33
            ),
            "mode",
        ),
        (
            "tanh-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "tanh-bad-attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "swish_beta", 2.0
            ),
            "default attributes",
        ),
        (
            "tanh-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
    ]
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Tanh mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / "tanh-autotune.json"
    autotune_output = temporary / "tanh-autotune-artifact"
    write_json(autotune_path, request(identity, "tanh", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[3:12]
            == ["34", "0.0", "0.0", "0.0", "0", "1.0", "1.0", "1.0", "1"]
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Tanh candidates are incomplete or have wrong constants",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_elu_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "elu.json"
    output = temporary / "elu-artifact"
    document = request(identity, "elu")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Elu unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2,
        "Elu manifest request, binding, or tensor identity is invalid",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Elu must emit exactly one execution stage")
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == "elu"
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "unary_pointwise_contiguous_kernel"
        and len(variants) == 1,
        "Elu did not select one non-autotuned common unary stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/unary.py")
        and kernel.get("registry_sha256") == registry_sha256(),
        "Elu common unary source or registry identity is invalid",
    )
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == ("*fp32:16,*fp32:16,i32,35,0.0,0.0,0.0,0," "1.0,1.0,1.0,1,256")
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        "Elu variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 3
        and [argument.get("uid") for argument in arguments[:2]] == [1, 2]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "scalar_i32"]
        and arguments[2]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        }
        and variant.get("launch")
        == {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        "Elu runtime ABI or launch geometry is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            "elu-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 34
            ),
            "mode",
        ),
        (
            "elu-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "elu-bad-alpha",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "elu_alpha", 2.0
            ),
            "default attributes",
        ),
        (
            "elu-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
    ]
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Elu mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / "elu-autotune.json"
    autotune_output = temporary / "elu-autotune-artifact"
    write_json(autotune_path, request(identity, "elu", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[3:12]
            == ["35", "0.0", "0.0", "0.0", "0", "1.0", "1.0", "1.0", "1"]
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Elu candidates are incomplete or have wrong constants",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_identity_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "identity.json"
    output = temporary / "identity-artifact"
    document = request(identity, "identity")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Identity unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2,
        "Identity manifest request, binding, or tensor identity is invalid",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Identity must emit exactly one execution stage")
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == "identity"
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "unary_pointwise_contiguous_kernel"
        and len(variants) == 1,
        "Identity did not select one non-autotuned common unary stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/unary.py")
        and kernel.get("registry_sha256") == registry_sha256(),
        "Identity common unary source or registry identity is invalid",
    )
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == ("*fp32:16,*fp32:16,i32,5,0.0,0.0,0.0,0," "1.0,1.0,1.0,1,256")
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        "Identity variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 3
        and [argument.get("uid") for argument in arguments[:2]] == [1, 2]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "scalar_i32"]
        and arguments[2]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        }
        and variant.get("launch")
        == {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        "Identity runtime ABI or launch geometry is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            "identity-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 35
            ),
            "mode",
        ),
        (
            "identity-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "identity-bad-attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "swish_beta", 2.0
            ),
            "default attributes",
        ),
        (
            "identity-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
    ]
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Identity mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / "identity-autotune.json"
    autotune_output = temporary / "identity-autotune-artifact"
    write_json(autotune_path, request(identity, "identity", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[3:12]
            == ["5", "0.0", "0.0", "0.0", "0", "1.0", "1.0", "1.0", "1"]
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Identity candidates are incomplete or have wrong constants",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_performance_artifacts(
    provider: object, temporary: Path, identity: str, *, compile_kernel: bool
) -> None:
    for operation in ("add", "abs", "pow"):
        for data_type in ("float32", "float16", "bfloat16"):
            document = request(identity, operation, autotune=True)
            for tensor in document["graph"]["tensors"]:
                tensor.update(
                    data_type=data_type, dimensions=[65536], strides=[1]
                )
            document["graph"]["nodes"][0]["attributes"]["n_elements"] = 65536
            name = f"optimized-{operation}-{data_type}"
            path, output = temporary / f"{name}.json", temporary / name
            write_json(path, document)
            provider.compile_request(path, output, "libtriton_jit")
            stage = load_artifact(output)["program"]["stages"][0]
            optimized = stage["variants"][-1]
            require(
                len(stage["variants"]) == 3 and stage["tuning"] is not None,
                f"{name} lost the measured autotune candidate",
            )
            require(
                optimized["full_signature"].split(",")[
                    len(document["graph"]["tensors"])
                ]
                == "i32:16",
                f"{name} lost proven scalar divisibility",
            )
            require(
                optimized["arguments"][-1]["value"] == 65536,
                f"{name} specialized away the runtime count",
            )
            if compile_kernel:
                compile_add_kernel(output, optimized, stage["kernel"])

    document = batchnorm_request(identity, "batchnorm", autotune=True)
    dimensions, strides = [8, 8, 56, 56], [25088, 3136, 56, 1]
    for tensor in document["graph"]["tensors"]:
        if tensor["uid"] in (1, 6):
            tensor.update(dimensions=dimensions, strides=strides)
    document["graph"]["nodes"][0]["attributes"].update(
        batch=8,
        spatial=3136,
        n_elements=200704,
        dimensions=dimensions,
        x_strides=strides,
        y_strides=strides,
    )
    path, output = temporary / "optimized-bn.json", temporary / "optimized-bn"
    write_json(path, document)
    provider.compile_request(path, output, "libtriton_jit")
    stage = load_artifact(output)["program"]["stages"][0]
    optimized = stage["variants"][-1]
    require(
        len(stage["variants"]) == 3
        and optimized["launch"]["shared_memory"] == 32,
        "large BatchNorm lost its eight-warp reduction candidate",
    )
    if compile_kernel:
        compile_add_kernel(output, optimized, stage["kernel"])

    for operation in (
        "convolution_fprop",
        "convolution_dgrad",
        "convolution_wgrad",
    ):
        for dtype in ("float32", "float16", "bfloat16"):
            document = convolution_request(identity, operation, autotune=True)
            dimensions = ([1, 3, 640, 640], [16, 3, 3, 3], [1, 16, 320, 320])
            for tensor, shape in zip(document["graph"]["tensors"], dimensions):
                tensor.update(
                    data_type=dtype,
                    dimensions=shape,
                    strides=[math.prod(shape[i + 1 :]) for i in range(4)],
                )
            output = document["graph"]["nodes"][0]["outputs"][0]["uid"]
            document["graph"]["nodes"][0]["attributes"].update(
                stride=[2, 2], n_outputs=math.prod(dimensions[output - 1])
            )
            name = f"optimized-{operation}-{dtype}"
            path, output = temporary / f"{name}.json", temporary / name
            write_json(path, document)
            provider.compile_request(path, output, "libtriton_jit")
            stage = load_artifact(output)["program"]["stages"][0]
            require(
                len(stage["variants"]) == 3, f"{name} lost optimized candidate"
            )
            if compile_kernel:
                compile_add_kernel(
                    output, stage["variants"][-1], stage["kernel"]
                )

    # The PPU 64x32x32 FProp lowering can omit output stores for this shape.
    # Keep both default and autotuned configurations on qualified tiles;
    # the unchanged native acDNN standard_3x3 case checks numerical output.
    for dtype in ("float16", "bfloat16"):
        for channels_last in (False, True):
            for autotune in (False, True):
                document = convolution_request(
                    identity, "convolution_fprop", autotune=autotune
                )
                shapes = ([8, 32, 32, 32], [64, 32, 3, 3], [8, 64, 32, 32])
                for tensor, shape in zip(document["graph"]["tensors"], shapes):
                    strides = [math.prod(shape[i + 1 :]) for i in range(4)]
                    if channels_last and tensor["uid"] != 2:
                        strides = [
                            math.prod(shape[1:]),
                            1,
                            shape[3] * shape[1],
                            shape[1],
                        ]
                    tensor.update(
                        data_type=dtype, dimensions=shape, strides=strides
                    )
                document["graph"]["nodes"][0]["attributes"]["n_outputs"] = (
                    math.prod(shapes[2])
                )
                name = f"fprop-ci32-{dtype}-nhwc{channels_last}-tune{autotune}"
                path, output = temporary / f"{name}.json", temporary / name
                write_json(path, document)
                provider.compile_request(path, output, "libtriton_jit")
                stage = load_artifact(output)["program"]["stages"][0]
                variants = stage["variants"]
                require(
                    len(variants) == (2 if autotune else 1),
                    f"{name} duplicated the existing registry candidate",
                )
                require(
                    variants[-1]["full_signature"].split(",")[-3:]
                    == ["32"] * 3,
                    f"{name} selected an unqualified FProp tile",
                )
                if compile_kernel:
                    for variant in variants:
                        compile_add_kernel(output, variant, stage["kernel"])

    for dtype in ("float32", "float16", "bfloat16"):
        for stride in (1, 2):
            for autotune in (False, True):
                document = convolution_request(
                    identity, "convolution_wgrad", autotune=autotune
                )
                extent, kernel, padding = (
                    (28, 1, 0) if stride == 1 else (56, 3, 1)
                )
                shapes = (
                    [8, 64, extent, extent],
                    [128, 64, kernel, kernel],
                    [8, 128, 28, 28],
                )
                for tensor, shape in zip(document["graph"]["tensors"], shapes):
                    tensor.update(
                        data_type=dtype,
                        dimensions=shape,
                        strides=[math.prod(shape[i + 1 :]) for i in range(4)],
                    )
                document["graph"]["nodes"][0]["attributes"].update(
                    stride=[stride, stride],
                    pre_padding=[padding, padding],
                    post_padding=[padding, padding],
                    n_outputs=math.prod(shapes[1]),
                )
                name = f"wgrad-reduction-{dtype}-stride{stride}-tune{autotune}"
                path, output = temporary / f"{name}.json", temporary / name
                write_json(path, document)
                provider.compile_request(path, output, "libtriton_jit")
                stage = load_artifact(output)["program"]["stages"][0]
                variants = stage["variants"]
                wide_reduction = dtype != "float32" and stride == 1
                expected_count = (
                    (3 if wide_reduction else 2) if autotune else 1
                )
                require(
                    len(variants) == expected_count,
                    f"{name} has incorrect candidate coverage",
                )
                # WGrad's final constexpr arguments are OC, CI, M.
                expected_tile = (
                    [16, 16, 128]
                    if wide_reduction
                    else ([32, 32, 32] if autotune else [16, 16, 16])
                )
                require(
                    variants[-1]["full_signature"].split(",")[-3:]
                    == list(map(str, expected_tile)),
                    f"{name} applied the unit-stride half reduction policy incorrectly",
                )
                if compile_kernel:
                    compile_add_kernel(output, variants[-1], stage["kernel"])

    for dtype in ("float16", "bfloat16"):
        for autotune in (False, True):
            document = convolution_request(
                identity, "convolution_fprop", autotune=autotune
            )
            shapes = ([16, 32, 256], [64, 32, 3], [16, 64, 256])
            for tensor, shape in zip(document["graph"]["tensors"], shapes):
                strides = [math.prod(shape[i + 1 :]) for i in range(3)]
                if tensor["uid"] == 3:
                    strides = [64 * 256, 1, 64]
                tensor.update(
                    data_type=dtype, dimensions=shape, strides=strides
                )
            document["graph"]["nodes"][0]["attributes"].update(
                spatial_rank=1,
                stride=[1],
                pre_padding=[1],
                post_padding=[1],
                dilation=[1],
                n_outputs=math.prod(shapes[2]),
            )
            name = f"fprop-width-nwc-{dtype}-tune{autotune}"
            path, output = temporary / f"{name}.json", temporary / name
            write_json(path, document)
            provider.compile_request(path, output, "libtriton_jit")
            stage = load_artifact(output)["program"]["stages"][0]
            variants = stage["variants"]
            require(
                len(variants) == (3 if autotune else 1),
                f"{name} lost the qualified NWC candidate",
            )
            require(
                variants[-1]["full_signature"].split(",")[-3:]
                == ["64", "32", "32"],
                f"{name} unnecessarily applied the NCHW store workaround",
            )
            if compile_kernel:
                compile_add_kernel(output, variants[-1], stage["kernel"])

    # Retain useful parallelism for the small FP32 P5 gradient and avoid
    # oversized scratch tiles for short 3-D reductions. Compile every
    # candidate so the resource contract also covers autotune execution.
    for geometry, dtype in (
        ("p5", "float32"),
        ("p5-batch2", "float32"),
        ("3d", "float16"),
        ("3d", "bfloat16"),
        ("3d", "float32"),
    ):
        for autotune in (False, True):
            document = convolution_request(
                identity, "convolution_dgrad", autotune=autotune
            )
            if geometry.startswith("p5"):
                batch = 2 if geometry == "p5-batch2" else 1
                shapes = (
                    [batch, 128, 40, 40],
                    [256, 128, 3, 3],
                    [batch, 256, 20, 20],
                )
                rank, stride = 2, [2, 2]
            else:
                shapes = (
                    [2, 8, 8, 16, 16],
                    [16, 8, 3, 3, 3],
                    [2, 16, 8, 16, 16],
                )
                rank, stride = 3, [1, 1, 1]
            for tensor, shape in zip(document["graph"]["tensors"], shapes):
                tensor.update(
                    data_type=dtype,
                    dimensions=shape,
                    strides=[
                        math.prod(shape[i + 1 :]) for i in range(len(shape))
                    ],
                )
            document["graph"]["nodes"][0]["attributes"].update(
                spatial_rank=rank,
                stride=stride,
                pre_padding=[1] * rank,
                post_padding=[1] * rank,
                dilation=[1] * rank,
                n_outputs=math.prod(shapes[0]),
            )
            name = f"dgrad-occupancy-{geometry}-{dtype}-tune{autotune}"
            path, output = temporary / f"{name}.json", temporary / name
            write_json(path, document)
            provider.compile_request(path, output, "libtriton_jit")
            stage = load_artifact(output)["program"]["stages"][0]
            variants = stage["variants"]
            half_3d = geometry == "3d" and dtype != "float32"
            optimized = half_3d or geometry == "p5-batch2"
            expected_count = (3 if optimized else 2) if autotune else 1
            require(
                len(variants) == expected_count,
                f"{name} has incorrect DGrad candidate coverage",
            )
            expected_tile = (
                [64, 16, 16]
                if half_3d
                else (
                    [128, 32, 32]
                    if geometry == "p5-batch2"
                    else [32, 32, 32] if autotune else [16, 16, 16]
                )
            )
            # DGrad's final constexpr arguments are M, CI, K, GROUP_M.
            require(
                variants[-1]["full_signature"].split(",")[-4:-1]
                == list(map(str, expected_tile)),
                f"{name} selected a DGrad tile with measured regression",
            )
            if compile_kernel:
                for variant in variants:
                    compile_add_kernel(output, variant, stage["kernel"])

    # Wide small-channel FProp tiles regress on short 3-D volumes for every
    # dtype. Retain original defaults and the two qualified autotune tiles.
    for dtype in ("float32", "float16", "bfloat16"):
        for autotune in (False, True):
            document = convolution_request(
                identity, "convolution_fprop", autotune=autotune
            )
            shapes = ([2, 8, 8, 16, 16], [16, 8, 3, 3, 3], [2, 16, 8, 16, 16])
            for tensor, shape in zip(document["graph"]["tensors"], shapes):
                tensor.update(
                    data_type=dtype,
                    dimensions=shape,
                    strides=[math.prod(shape[i + 1 :]) for i in range(5)],
                )
            document["graph"]["nodes"][0]["attributes"].update(
                spatial_rank=3,
                stride=[1, 1, 1],
                pre_padding=[1, 1, 1],
                post_padding=[1, 1, 1],
                dilation=[1, 1, 1],
                n_outputs=math.prod(shapes[2]),
            )
            name = f"fprop-short-volume-{dtype}-tune{autotune}"
            path, output = temporary / f"{name}.json", temporary / name
            write_json(path, document)
            provider.compile_request(path, output, "libtriton_jit")
            stage = load_artifact(output)["program"]["stages"][0]
            variants = stage["variants"]
            require(
                len(variants) == (2 if autotune else 1),
                f"{name} retained a regressing wide volume tile",
            )
            require(
                variants[-1]["full_signature"].split(",")[-3:]
                == (["32"] * 3 if autotune else ["16"] * 3),
                f"{name} lost the original qualified volume tile",
            )
            if compile_kernel:
                for variant in variants:
                    compile_add_kernel(output, variant, stage["kernel"])


def assert_generic_binary_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    operation: str,
    label: str,
    mode: int,
    wrong_mode: int,
    compile_kernel: bool,
    pointer_signature: str = "*fp32:16,*fp32:16,*fp32:16",
    bad_dtype_detail: str = "matching",
) -> None:
    request_path = temporary / f"{operation}.json"
    output = temporary / f"{operation}-artifact"
    document = request(identity, operation)
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"{label} unexpectedly returned {result!r}",
    )
    manifest = load_artifact(output)
    stages = manifest.get("program", {}).get("stages", [])
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2, 3]
        and manifest.get("tensor_count") == 3
        and len(stages) == 1,
        f"{label} manifest or stage identity is invalid",
    )
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("operation") == operation
        and stage.get("tuning") is None
        and kernel.get("provider")
        == ("thead_triton" if operation == "pow" else "common_triton")
        and kernel.get("ownership")
        == ("platform" if operation == "pow" else "common")
        and kernel.get("function")
        == (
            "activation_backward_contiguous_kernel"
            if operation == "sigmoid_backward"
            else "binary_contiguous_kernel"
        )
        and len(variants) == 1,
        f"{label} did not select one non-autotuned common binary stage",
    )
    variant = variants[0]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == (
            f"{pointer_signature},i32,{mode},1.0,256"
            + (
                ",0.0,0.0,0.0,False,1.0,1.0,1.0,True"
                if operation == "sigmoid_backward"
                else ""
            )
        )
        and variant.get("argument_count") == 4
        and [
            argument.get("uid")
            for argument in variant.get("arguments", [])[:3]
        ]
        == [1, 2, 3],
        f"{label} variant specialization or runtime ABI is invalid",
    )
    for name, mutate, detail in (
        (
            f"{operation}-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", wrong_mode
            ),
            "mode",
        ),
        (
            f"{operation}-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            bad_dtype_detail,
        ),
        (
            f"{operation}-bad-stride",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "strides", [0]
            ),
            "non-overlapping",
        ),
        (
            f"{operation}-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
    ):
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected {label} mutation {name} wrote an artifact directory",
        )
    autotune_path = temporary / f"{operation}-autotune.json"
    autotune_output = temporary / f"{operation}-autotune-artifact"
    write_json(autotune_path, request(identity, operation, autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[4] == str(mode)
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        f"autotuned {label} candidates are incomplete or have wrong mode",
    )
    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_add_square_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "add-square.json"
    output = temporary / "add-square-artifact"
    document = request(identity, "add_square")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"AddSquare unexpectedly returned {result!r}",
    )
    manifest = load_artifact(output)
    stages = manifest.get("program", {}).get("stages", [])
    tensors = manifest.get("tensors", [])
    require(
        manifest.get("external_uids") == [1, 2, 4]
        and manifest.get("tensor_count") == 4
        and manifest.get("workspace") == {"size": 0, "alignment": 256}
        and len(stages) == 1
        and tensors[2].get("uid") == 3
        and tensors[2].get("virtual") is True
        and "workspace_offset" not in tensors[2],
        "AddSquare manifest binding or workspace identity is invalid",
    )
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("source_node_ids") == [0, 1]
        and stage.get("dependencies") == []
        and stage.get("operation") == "add_square"
        and stage.get("tuning") is None
        and kernel.get("provider") == "thead_triton"
        and kernel.get("ownership") == "platform"
        and kernel.get("function") == "add_square_contiguous_kernel"
        and len(variants) == 1,
        "AddSquare did not select one fused THead stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(BACKEND_ROOT / "thead/kernels/add_square.py"),
        "AddSquare platform kernel source identity is invalid",
    )
    variant = variants[0]
    require(
        variant.get("full_signature") == "*fp32:16,*fp32:16,*fp32:16,i32,256"
        and [
            argument.get("uid")
            for argument in variant.get("arguments", [])[:3]
        ]
        == [2, 1, 4]
        and variant.get("arguments", [None])[-1]
        == {"kind": "scalar_i32", "name": "n_elements", "value": 16},
        "AddSquare fused kernel ABI is invalid",
    )

    for data_type, pointer in (("float16", "fp16"), ("bfloat16", "bf16")):
        typed_document = request(identity, "add_square")
        for tensor in typed_document["graph"]["tensors"]:
            tensor["data_type"] = data_type
        typed_path = temporary / f"add-square-{data_type}.json"
        typed_output = temporary / f"add-square-{data_type}-artifact"
        write_json(typed_path, typed_document)
        typed_result = provider.compile_request(
            typed_path, typed_output, "libtriton_jit"
        )
        typed_manifest = load_artifact(typed_output)
        typed_stage = typed_manifest["program"]["stages"][0]
        typed_variant = typed_stage["variants"][0]
        require(
            typed_result.get("status") == "success"
            and typed_variant.get("full_signature")
            == f"*{pointer}:16,*{pointer}:16,*{pointer}:16,i32,256"
            and typed_manifest["workspace"]["size"] == 0
            and "workspace_offset" not in typed_manifest["tensors"][2]
            and all(
                tensor["storage_size"] == 32
                for tensor in typed_manifest["tensors"]
            ),
            f"AddSquare {data_type} ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                typed_output, typed_variant, typed_stage["kernel"]
            )

    strided_document = request(identity, "add_square")
    for tensor in strided_document["graph"]["tensors"]:
        tensor["dimensions"] = [1, 4, 8, 16]
        # Keep a real gap between rows: dense NHWC now uses the linear kernel.
        tensor["strides"] = [640, 1, 80, 4]
    for node in strided_document["graph"]["nodes"]:
        node["attributes"]["n_elements"] = 512
    strided_path = temporary / "add-square-strided.json"
    strided_output = temporary / "add-square-strided-artifact"
    write_json(strided_path, strided_document)
    strided_result = provider.compile_request(
        strided_path, strided_output, "libtriton_jit"
    )
    strided_stage = load_artifact(strided_output)["program"]["stages"][0]
    strided_variant = strided_stage["variants"][0]
    require(
        strided_result.get("status") == "success"
        and strided_stage["kernel"]["function"] == "add_square_strided_kernel"
        and len(strided_variant["full_signature"].split(",")) == 37,
        "AddSquare explicit-strided kernel ABI is invalid",
    )
    if compile_kernel:
        compile_add_kernel(
            strided_output, strided_variant, strided_stage["kernel"]
        )
    for name, mutate, detail in (
        (
            "add-square-bad-mul-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", 1
            ),
            "Mul",
        ),
        (
            "add-square-bad-dataflow",
            lambda value: value["graph"]["nodes"][1]["inputs"][1].__setitem__(
                "uid", 2
            ),
            "dataflow",
        ),
        (
            "add-square-nonvirtual-intermediate",
            lambda value: value["graph"]["tensors"][2].__setitem__(
                "virtual", False
            ),
            "virtual",
        ),
        (
            "add-square-mismatched-dtype",
            lambda value: value["graph"]["tensors"][2].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "add-square-bad-alpha",
            lambda value: value["graph"]["nodes"][1]["attributes"].__setitem__(
                "alpha", 0.5
            ),
            "alpha",
        ),
    ):
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected AddSquare mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / "add-square-autotune.json"
    autotune_output = temporary / "add-square-autotune-artifact"
    write_json(autotune_path, request(identity, "add_square", autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    require(
        autotune_result.get("status") == "success"
        and [
            candidate.get("variant_id")
            for candidate in autotune_stage.get("variants", [])
        ]
        == ["block128_w4_s1", "block256_w4_s1"]
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned AddSquare candidates are incomplete",
    )
    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_reduction_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    for operation, reduction_mode in (
        ("reduction_sum", 1),
        ("reduction_avg", 2),
        ("reduction_mul", 3),
    ):
        request_path = temporary / f"{operation}.json"
        output = temporary / f"{operation}-artifact"
        document = reduction_request(identity, operation)
        write_json(request_path, document)
        result = provider.compile_request(
            request_path, output, "libtriton_jit"
        )
        require(
            result.get("status") == "success",
            f"{operation} unexpectedly returned {result!r}",
        )
        manifest = load_artifact(output)
        stages = manifest.get("program", {}).get("stages", [])
        require(
            manifest.get("external_uids") == [1, 2]
            and manifest.get("workspace") == {"size": 0, "alignment": 256}
            and len(stages) == 1,
            f"{operation} manifest identity is invalid",
        )
        stage = stages[0]
        kernel = stage.get("kernel", {})
        variants = stage.get("variants", [])
        require(
            stage.get("operation") == operation
            and stage.get("source_node_ids") == [0]
            and stage.get("tuning") is None
            and kernel.get("provider") == "common_triton"
            and kernel.get("ownership") == "common"
            and kernel.get("function") == "reduction_3d_kernel"
            and len(variants) == 1,
            f"{operation} did not select the common 3D reduction kernel",
        )
        source = kernel.get("materialized_source", {})
        source_path = output / source.get("path", "")
        require(
            source_path.is_file()
            and source.get("sha256") == sha256_file(source_path)
            and source.get("sha256")
            == sha256_file(SOURCE_ROOT / "kernels/common/reduction.py"),
            f"{operation} common source identity is invalid",
        )
        variant = variants[0]
        require(
            variant.get("full_signature")
            == (
                "*fp32:16,*fp32:16,i32,4,64,256,64,1," f"{reduction_mode},8,32"
            )
            and variant.get("argument_count") == 3
            and [
                argument.get("uid")
                for argument in variant.get("arguments", [])[:2]
            ]
            == [1, 2]
            and variant.get("arguments", [None])[-1]
            == {
                "kind": "scalar_i32",
                "name": "output_elements",
                "value": 128,
            }
            and variant.get("launch")
            == {
                "grid": [16, 1, 1],
                "block": [128, 1, 1],
                "shared_memory": 128,
            },
            f"{operation} reduction ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(output, variant, kernel)

    for data_type, pointer in (("float16", "fp16"), ("bfloat16", "bf16")):
        typed_document = reduction_request(identity, "reduction_sum")
        for tensor in typed_document["graph"]["tensors"]:
            tensor["data_type"] = data_type
        typed_path = temporary / f"reduction-sum-{data_type}.json"
        typed_output = temporary / f"reduction-sum-{data_type}-artifact"
        write_json(typed_path, typed_document)
        typed_result = provider.compile_request(
            typed_path, typed_output, "libtriton_jit"
        )
        typed_manifest = load_artifact(typed_output)
        typed_stage = typed_manifest["program"]["stages"][0]
        typed_variant = typed_stage["variants"][0]
        require(
            typed_result.get("status") == "success"
            and typed_variant.get("full_signature", "").startswith(
                f"*{pointer}:16,*{pointer}:16,i32,"
            )
            and [
                tensor["storage_size"] for tensor in typed_manifest["tensors"]
            ]
            == [1024, 256],
            f"Reduction {data_type} ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                typed_output, typed_variant, typed_stage["kernel"]
            )

    strided_cases = {
        "channels-last": {
            "input_dimensions": [2, 3, 5, 5],
            "input_strides": [75, 1, 15, 3],
            "input_alignment": 16,
            "output_dimensions": [2, 1, 5, 5],
            "output_strides": [25, 25, 5, 1],
            "attributes": {
                "outer": 2,
                "reduction": 3,
                "inner": 25,
                "output_elements": 50,
                "axis": 1,
                "keep_dimensions": 1,
            },
        },
        "unaligned": {
            "input_dimensions": [2, 4, 8, 8],
            "input_strides": [288, 72, 9, 1],
            "input_alignment": 4,
            "output_dimensions": [2, 1, 8, 8],
            "output_strides": [64, 64, 8, 1],
            "attributes": {
                "outer": 2,
                "reduction": 4,
                "inner": 64,
                "output_elements": 128,
                "axis": 1,
                "keep_dimensions": 1,
            },
        },
    }
    for name, values in strided_cases.items():
        strided_document = reduction_request(identity, "reduction_sum")
        input_tensor, output_tensor = strided_document["graph"]["tensors"]
        input_tensor["dimensions"] = values["input_dimensions"]
        input_tensor["strides"] = values["input_strides"]
        input_tensor["alignment"] = values["input_alignment"]
        output_tensor["dimensions"] = values["output_dimensions"]
        output_tensor["strides"] = values["output_strides"]
        strided_document["graph"]["nodes"][0]["attributes"].update(
            values["attributes"]
        )
        strided_path = temporary / f"reduction-{name}.json"
        strided_output = temporary / f"reduction-{name}-artifact"
        write_json(strided_path, strided_document)
        strided_result = provider.compile_request(
            strided_path, strided_output, "libtriton_jit"
        )
        strided_stage = load_artifact(strided_output)["program"]["stages"][0]
        strided_variant = strided_stage["variants"][0]
        require(
            strided_result.get("status") == "success"
            and strided_stage["kernel"]["function"]
            == "reduction_strided_kernel",
            f"Reduction {name} strided path is invalid",
        )
        if name == "unaligned":
            require(
                strided_variant.get("full_signature", "").startswith(
                    "*fp32,*fp32:16,i32,"
                ),
                "Reduction unaligned pointer was not lowered to the safe "
                "libtriton_jit alignment hint",
            )
        if compile_kernel:
            compile_add_kernel(
                strided_output, strided_variant, strided_stage["kernel"]
            )
        if name == "channels-last":
            for data_type, pointer in (
                ("float16", "fp16"),
                ("bfloat16", "bf16"),
            ):
                typed_document = copy.deepcopy(strided_document)
                for tensor in typed_document["graph"]["tensors"]:
                    tensor["data_type"] = data_type
                typed_path = temporary / f"reduction-{name}-{data_type}.json"
                typed_output = temporary / (
                    f"reduction-{name}-{data_type}-artifact"
                )
                write_json(typed_path, typed_document)
                typed_result = provider.compile_request(
                    typed_path, typed_output, "libtriton_jit"
                )
                typed_stage = load_artifact(typed_output)["program"]["stages"][
                    0
                ]
                typed_variant = typed_stage["variants"][0]
                require(
                    typed_result.get("status") == "success"
                    and typed_variant.get("full_signature", "").startswith(
                        f"*{pointer}:16,*{pointer}:16,i32,"
                    ),
                    f"Reduction {name} {data_type} ABI is invalid",
                )
                if compile_kernel:
                    compile_add_kernel(
                        typed_output,
                        typed_variant,
                        typed_stage["kernel"],
                    )

    scalar_document = reduction_request(identity, "reduction_sum")
    scalar_document["graph"]["tensors"][0].update(
        {"dimensions": [8], "strides": [1]}
    )
    scalar_document["graph"]["tensors"][1].update(
        {"dimensions": [], "strides": []}
    )
    scalar_document["graph"]["nodes"][0]["attributes"].update(
        {
            "outer": 1,
            "reduction": 8,
            "inner": 1,
            "output_elements": 1,
            "axis": 0,
            "keep_dimensions": 0,
        }
    )
    scalar_path = temporary / "reduction-scalar.json"
    scalar_output = temporary / "reduction-scalar-artifact"
    write_json(scalar_path, scalar_document)
    scalar_result = provider.compile_request(
        scalar_path, scalar_output, "libtriton_jit"
    )
    scalar_stage = load_artifact(scalar_output)["program"]["stages"][0]
    scalar_variant = scalar_stage["variants"][0]
    require(
        scalar_result.get("status") == "success"
        and scalar_stage["kernel"]["function"] == "reduction_2d_kernel",
        "Reduction scalar-output path is invalid",
    )
    if compile_kernel:
        compile_add_kernel(
            scalar_output, scalar_variant, scalar_stage["kernel"]
        )

    canonical = reduction_request(identity, "reduction_sum")
    for name, mutate, detail in (
        (
            "mismatched-dtype",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "bad-axis",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "axis", 2
            ),
            "output shape",
        ),
        (
            "bad-output-shape",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "dimensions", [2, 1, 8, 7]
            ),
            "output shape",
        ),
    ):
        mutated = copy.deepcopy(canonical)
        mutate(mutated)
        path = temporary / f"reduction-{name}.json"
        mutated_output = temporary / f"reduction-{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(not mutated_output.exists(), "rejected reduction wrote output")

    autotune_path = temporary / "reduction-autotune.json"
    autotune_output = temporary / "reduction-autotune-artifact"
    write_json(
        autotune_path,
        reduction_request(identity, "reduction_sum", autotune=True),
    )
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    require(
        autotune_result.get("status") == "success"
        and [
            candidate.get("variant_id")
            for candidate in autotune_stage.get("variants", [])
        ]
        == ["blockm8_blockn32_w4_s1", "blockm16_blockn64_w4_s1"]
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned reduction candidates are incomplete",
    )


def assert_layout_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    operation: str,
    compile_kernel: bool,
) -> None:
    request_path = temporary / f"{operation}.json"
    output = temporary / f"{operation}-artifact"
    document = layout_request(identity, operation)
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"{operation} unexpectedly returned {result!r}",
    )
    manifest = load_artifact(output)
    stages = manifest.get("program", {}).get("stages", [])
    require(
        manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2
        and manifest.get("workspace") == {"size": 0, "alignment": 256}
        and len(stages) == 1,
        f"{operation} manifest binding or workspace identity is invalid",
    )
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    require(
        stage.get("source_node_ids") == [0]
        and stage.get("dependencies") == []
        and stage.get("operation") == operation
        and stage.get("tuning") is None
        and kernel.get("provider")
        == ("thead_triton" if operation == "transpose" else "common_triton")
        and kernel.get("ownership")
        == ("platform" if operation == "transpose" else "common")
        and kernel.get("function") == "layout_copy_kernel"
        and len(variants) == 1,
        f"{operation} did not select the common Layout stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(
            SOURCE_ROOT
            / (
                "backends/thead/kernels/layout.py"
                if operation == "transpose"
                else "kernels/common/layout.py"
            )
        ),
        f"{operation} common Layout source identity is invalid",
    )
    variant = variants[0]
    signature = variant.get("full_signature", "").split(",")
    n_elements = math.prod(document["graph"]["tensors"][1]["dimensions"])
    require(
        len(signature) == 37
        and signature[:3] == ["*fp32:16", "*fp32:16", "i32"]
        and signature[-1] == "256"
        and variant.get("argument_count") == 3
        and [
            argument.get("uid")
            for argument in variant.get("arguments", [])[:2]
        ]
        == [1, 2]
        and variant.get("arguments", [None])[-1]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": n_elements,
        },
        f"{operation} Layout kernel ABI is invalid",
    )

    for data_type, pointer in (("float16", "fp16"), ("bfloat16", "bf16")):
        typed_document = layout_request(identity, operation)
        for tensor in typed_document["graph"]["tensors"]:
            tensor["data_type"] = data_type
        typed_path = temporary / f"{operation}-{data_type}.json"
        typed_output = temporary / f"{operation}-{data_type}-artifact"
        write_json(typed_path, typed_document)
        typed_result = provider.compile_request(
            typed_path, typed_output, "libtriton_jit"
        )
        typed_manifest = load_artifact(typed_output)
        typed_stage = typed_manifest["program"]["stages"][0]
        typed_variant = typed_stage["variants"][0]
        require(
            typed_result.get("status") == "success"
            and typed_variant.get("full_signature", "").startswith(
                f"*{pointer}:16,*{pointer}:16,i32,"
            ),
            f"{operation}/{data_type} Layout ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                typed_output, typed_variant, typed_stage["kernel"]
            )

    mutations: tuple[
        tuple[str, Callable[[dict[str, Any]], None], str], ...
    ] = (
        (
            "mismatched-dtype",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", n_elements - 1
            ),
            "n_elements",
        ),
        (
            "bad-input-stride",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "strides", [99, *value["graph"]["tensors"][0]["strides"][1:]]
            ),
            "contiguous",
        ),
    )
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{operation}-{name}.json"
        mutated_output = temporary / f"{operation}-{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected {operation} mutation {name} wrote an artifact",
        )

    autotune_path = temporary / f"{operation}-autotune.json"
    autotune_output = temporary / f"{operation}-autotune-artifact"
    write_json(
        autotune_path,
        layout_request(identity, operation, autotune=True),
    )
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    require(
        autotune_result.get("status") == "success"
        and [
            candidate.get("variant_id")
            for candidate in autotune_stage.get("variants", [])
        ]
        == ["block128_w4_s1", "block256_w4_s1"]
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        f"autotuned {operation} candidates are incomplete",
    )
    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_batchnorm_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    expectations = {
        "batchnorm_inference": {
            "function": "batch_norm_inference_nchw_kernel",
            "provider": "thead_triton",
            "ownership": "platform",
            "uids": [1, 2, 3, 4, 5, 6],
            "arguments": [1, 2, 3, 4, 5, 6],
            "signature": (
                "*fp32:16,*fp32:16,*fp32:16,*fp32:16,*fp32:16,"
                "*fp32:16,8,256,0.0,256,1,1,1"
            ),
            "grid": [16, 1, 1],
            "shared": 0,
        },
        "batchnorm": {
            "function": "batch_norm_nchw_kernel",
            "provider": "common_triton",
            "ownership": "common",
            "uids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "arguments": [1, 6, 4, 5, 2, 3, 7, 8, 9, 10],
            "signature": (
                "*fp32:16,*fp32:16,*fp32:16,*fp32:16,*fp32:16,"
                "*fp32:16,*fp32:16,*fp32:16,*fp32:16,*fp32:16,"
                "2,8,64,0.0010000000474974513,0.10000000149011612,"
                "256,1,1,1,1,1"
            ),
            "grid": [8, 1, 1],
            "shared": 16,
        },
    }
    for operation, expected in expectations.items():
        request_path = temporary / f"{operation}.json"
        output = temporary / f"{operation}-artifact"
        write_json(request_path, batchnorm_request(identity, operation))
        result = provider.compile_request(
            request_path, output, "libtriton_jit"
        )
        manifest = load_artifact(output)
        stage = manifest["program"]["stages"][0]
        variant = stage["variants"][0]
        kernel = stage["kernel"]
        require(
            result.get("status") == "success"
            and manifest.get("external_uids") == expected["uids"]
            and manifest.get("workspace") == {"size": 0, "alignment": 256}
            and stage.get("operation") == operation
            and kernel.get("provider") == expected["provider"]
            and kernel.get("ownership") == expected["ownership"]
            and kernel.get("function") == expected["function"]
            and variant.get("full_signature") == expected["signature"]
            and variant.get("argument_count") == len(expected["arguments"])
            and [
                argument.get("uid")
                for argument in variant.get("arguments", [])
            ]
            == expected["arguments"]
            and variant.get("launch")
            == {
                "grid": expected["grid"],
                "block": [128, 1, 1],
                "shared_memory": expected["shared"],
            },
            f"{operation} artifact ABI is invalid",
        )
        source = output / kernel["materialized_source"]["path"]
        expected_source = (
            SOURCE_ROOT / "backends/thead/kernels/normalization.py"
            if operation == "batchnorm_inference"
            else SOURCE_ROOT / "kernels/common/normalization.py"
        )
        require(
            source.is_file()
            and kernel["materialized_source"]["sha256"]
            == sha256_file(source)
            == sha256_file(expected_source),
            f"{operation} source identity is invalid",
        )
        if compile_kernel:
            compile_add_kernel(output, variant, kernel)

        data_uids = (
            {1, 6} if operation == "batchnorm_inference" else {1, 2, 3, 6}
        )
        for data_type, pointer in (
            ("float16", "fp16"),
            ("bfloat16", "bf16"),
        ):
            typed_document = batchnorm_request(identity, operation)
            for tensor in typed_document["graph"]["tensors"]:
                if tensor["uid"] in data_uids:
                    tensor["data_type"] = data_type
            typed_path = temporary / f"{operation}-{data_type}.json"
            typed_output = temporary / f"{operation}-{data_type}-artifact"
            write_json(typed_path, typed_document)
            typed_result = provider.compile_request(
                typed_path, typed_output, "libtriton_jit"
            )
            typed_manifest = load_artifact(typed_output)
            typed_stage = typed_manifest["program"]["stages"][0]
            typed_variant = typed_stage["variants"][0]
            typed_tensors = {
                tensor["uid"]: tensor for tensor in typed_manifest["tensors"]
            }
            signature = typed_variant["full_signature"].split(",")
            require(
                typed_result.get("status") == "success"
                and typed_tensors[1]["data_type"] == data_type
                and typed_tensors[6]["data_type"] == data_type
                and typed_tensors[1]["storage_size"]
                == math.prod(typed_tensors[1]["dimensions"])
                * (2 if data_type != "float32" else 4)
                and signature[0] == f"*{pointer}:16"
                and signature[expected["arguments"].index(6)]
                == f"*{pointer}:16",
                f"{operation}/{data_type} mixed-data ABI is invalid",
            )
            if compile_kernel:
                compile_add_kernel(
                    typed_output, typed_variant, typed_stage["kernel"]
                )

        channels_last = batchnorm_request(identity, operation)
        data_strides = (
            [2048, 1, 128, 8]
            if operation == "batchnorm_inference"
            else [512, 1, 64, 8]
        )
        for tensor in channels_last["graph"]["tensors"]:
            if tensor["uid"] in {1, 6}:
                tensor["strides"] = data_strides
        channels_last["graph"]["nodes"][0]["attributes"].update(
            {"x_strides": data_strides, "y_strides": data_strides}
        )
        channels_last_path = temporary / f"{operation}-channels-last.json"
        channels_last_output = (
            temporary / f"{operation}-channels-last-artifact"
        )
        write_json(channels_last_path, channels_last)
        channels_last_result = provider.compile_request(
            channels_last_path, channels_last_output, "libtriton_jit"
        )
        channels_last_stage = load_artifact(channels_last_output)["program"][
            "stages"
        ][0]
        channels_last_variant = channels_last_stage["variants"][0]
        expected_strided_function = (
            "batch_norm_inference_kernel"
            if operation == "batchnorm_inference"
            else "batch_norm_kernel"
        )
        require(
            channels_last_result.get("status") == "success"
            and channels_last_stage["kernel"]["function"]
            == expected_strided_function,
            f"{operation} channels-last path is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                channels_last_output,
                channels_last_variant,
                channels_last_stage["kernel"],
            )

    large_batch = batchnorm_request(identity, "batchnorm")
    large_dimensions = [513, 8, 1, 1]
    large_strides = [8, 1, 1, 1]
    for tensor in large_batch["graph"]["tensors"]:
        if tensor["uid"] in {1, 6}:
            tensor["dimensions"] = large_dimensions
            tensor["strides"] = large_strides
    large_attributes = large_batch["graph"]["nodes"][0]["attributes"]
    large_attributes.update(
        {
            "n_elements": 513 * 8,
            "batch": 513,
            "spatial": 1,
            "dimensions": large_dimensions,
            "x_strides": large_strides,
            "y_strides": large_strides,
        }
    )
    large_path = temporary / "batchnorm-large-batch.json"
    large_output = temporary / "batchnorm-large-batch-artifact"
    write_json(large_path, large_batch)
    large_result = provider.compile_request(
        large_path, large_output, "libtriton_jit"
    )
    large_stage = load_artifact(large_output)["program"]["stages"][0]
    require(
        large_result.get("status") == "success"
        and large_stage["kernel"]["function"] == "batch_norm_kernel"
        and [
            argument.get("name")
            for argument in large_stage["variants"][0]["arguments"][-3:]
        ]
        == ["batch", "channels", "spatial"],
        "large-batch dense BatchNorm did not select the general kernel",
    )

    canonical = batchnorm_request(identity, "batchnorm_inference")
    for name, mutate, detail in (
        (
            "mismatched-data-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching data",
        ),
        (
            "bad-spatial",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "spatial", 255
            ),
            "attributes",
        ),
    ):
        mutated = copy.deepcopy(canonical)
        mutate(mutated)
        path = temporary / f"batchnorm-{name}.json"
        mutated_output = temporary / f"batchnorm-{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(not mutated_output.exists(), "rejected batchnorm wrote output")

    oversized = batchnorm_request(identity, "batchnorm")
    oversized_dimensions = [2**31, 8, 1, 1]
    oversized_strides = [8, 1, 1, 1]
    for tensor in oversized["graph"]["tensors"]:
        if tensor["uid"] in {1, 6}:
            tensor["dimensions"] = oversized_dimensions
            tensor["strides"] = oversized_strides
    oversized["graph"]["nodes"][0]["attributes"].update(
        {
            "n_elements": (2**31) * 8,
            "batch": 2**31,
            "spatial": 1,
            "dimensions": oversized_dimensions,
            "x_strides": oversized_strides,
            "y_strides": oversized_strides,
        }
    )
    oversized_path = temporary / "batchnorm-oversized.json"
    oversized_output = temporary / "batchnorm-oversized-artifact"
    write_json(oversized_path, oversized)
    expect_value_error(
        lambda: provider.compile_request(
            oversized_path, oversized_output, "libtriton_jit"
        ),
        "int32",
    )
    require(not oversized_output.exists(), "oversized BatchNorm wrote output")

    autotune_path = temporary / "batchnorm-inference-autotune.json"
    autotune_output = temporary / "batchnorm-inference-autotune-artifact"
    write_json(
        autotune_path,
        batchnorm_request(identity, "batchnorm_inference", autotune=True),
    )
    provider.compile_request(autotune_path, autotune_output, "libtriton_jit")
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    require(
        [variant["variant_id"] for variant in autotune_stage["variants"]]
        == ["block256_w4_s1", "block512_w4_s1"]
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned batchnorm candidates are incomplete",
    )


def assert_normalization_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    expectations = {
        "layernorm": {
            "function": "layer_norm_kernel",
            "uids": [1, 2, 3, 4, 5, 6],
            "arguments": [1, 4, 5, 6, 2, 3, None],
            "signature": (
                "*fp32:16,*fp32:16,*fp32:16,*fp32:16,*fp32:16,"
                "*fp32:16,i32,0.0010000000474974513,17,256,1,1,1,1,0,False,False"
            ),
        },
        "rmsnorm": {
            "function": "rms_norm_kernel",
            "uids": [1, 2, 3, 4, 5],
            "arguments": [1, 4, 2, 3, 5, None],
            "signature": (
                "*fp32:16,*fp32:16,*fp32:16,*fp32:16,*fp32:16,"
                "i32,17,0.0010000000474974513,256,1,1,1,1,0"
            ),
        },
    }
    for operation, expected in expectations.items():
        request_path = temporary / f"{operation}.json"
        output = temporary / f"{operation}-artifact"
        write_json(request_path, normalization_request(identity, operation))
        result = provider.compile_request(
            request_path, output, "libtriton_jit"
        )
        require(
            result.get("status") == "success",
            f"{operation} was not compiled",
        )
        manifest = load_artifact(output)
        stage = manifest["program"]["stages"][0]
        variant = stage["variants"][0]
        kernel = stage["kernel"]
        require(
            manifest.get("external_uids") == expected["uids"]
            and manifest.get("workspace") == {"size": 0, "alignment": 256}
            and stage.get("operation") == operation
            and kernel.get("provider") == "common_triton"
            and kernel.get("ownership") == "common"
            and kernel.get("function") == expected["function"]
            and variant.get("full_signature") == expected["signature"]
            and variant.get("argument_count") == len(expected["arguments"])
            and [
                argument.get("uid")
                for argument in variant.get("arguments", [])
            ]
            == expected["arguments"]
            and variant.get("launch")
            == {
                "grid": [10, 1, 1],
                "block": [128, 1, 1],
                "shared_memory": 1024,
            },
            f"{operation} artifact ABI is invalid",
        )
        source = output / kernel["materialized_source"]["path"]
        expected_source = SOURCE_ROOT / "kernels/common/normalization.py"
        require(
            source.is_file()
            and kernel["materialized_source"]["sha256"]
            == sha256_file(source)
            == sha256_file(expected_source),
            f"{operation} source identity is invalid",
        )
        if compile_kernel:
            compile_add_kernel(output, variant, kernel)

        statistic_uids = {5, 6} if operation == "layernorm" else {5}
        for data_type, pointer in (("float16", "fp16"), ("bfloat16", "bf16")):
            typed_document = normalization_request(identity, operation)
            for tensor in typed_document["graph"]["tensors"]:
                if tensor["uid"] not in statistic_uids:
                    tensor["data_type"] = data_type
            typed_path = temporary / f"{operation}-{data_type}.json"
            typed_output = temporary / f"{operation}-{data_type}-artifact"
            write_json(typed_path, typed_document)
            typed_result = provider.compile_request(
                typed_path, typed_output, "libtriton_jit"
            )
            typed_manifest = load_artifact(typed_output)
            typed_stage = typed_manifest["program"]["stages"][0]
            typed_variant = typed_stage["variants"][0]
            signature = typed_variant["full_signature"].split(",")
            expected_pointers = (
                [pointer, pointer, "fp32", "fp32", pointer, pointer]
                if operation == "layernorm"
                else [pointer, pointer, pointer, pointer, "fp32"]
            )
            require(
                typed_result.get("status") == "success"
                and signature[: len(expected_pointers)]
                == [f"*{value}:16" for value in expected_pointers]
                and signature[len(expected_pointers)] == "i32",
                f"{operation}/{data_type} mixed-statistic ABI is invalid",
            )
            if compile_kernel:
                compile_add_kernel(
                    typed_output, typed_variant, typed_stage["kernel"]
                )

        large_suffix = normalization_request(identity, operation)
        data_shape = [2, 4, 4096]
        data_strides = [16384, 4096, 1]
        parameter_shape = [1, 1, 4096]
        parameter_strides = [4096, 4096, 1]
        statistic_shape = [2, 4, 1]
        statistic_strides = [4, 1, 1]
        for tensor in large_suffix["graph"]["tensors"]:
            if tensor["uid"] in {1, 4}:
                tensor["dimensions"] = data_shape
                tensor["strides"] = data_strides
            elif tensor["uid"] in {2, 3}:
                tensor["dimensions"] = parameter_shape
                tensor["strides"] = parameter_strides
            else:
                tensor["dimensions"] = statistic_shape
                tensor["strides"] = statistic_strides
        large_suffix["graph"]["nodes"][0]["attributes"].update(
            {"rows": 8, "normalized_elements": 4096}
        )
        large_path = temporary / f"{operation}-suffix4096.json"
        large_output = temporary / f"{operation}-suffix4096-artifact"
        write_json(large_path, large_suffix)
        large_result = provider.compile_request(
            large_path, large_output, "libtriton_jit"
        )
        require(
            large_result.get("status") == "success",
            f"{operation} suffix4096 was not compiled",
        )

        odd_suffix = normalization_request(identity, operation)
        data_shape = [3, 257, 513]
        data_strides = [131841, 513, 1]
        parameter_shape = [1, 1, 513]
        parameter_strides = [513, 513, 1]
        statistic_shape = [3, 257, 1]
        statistic_strides = [257, 1, 1]
        statistic_uids = {5, 6} if operation == "layernorm" else {5}
        for tensor in odd_suffix["graph"]["tensors"]:
            if tensor["uid"] in {1, 4}:
                tensor["dimensions"] = data_shape
                tensor["strides"] = data_strides
            elif tensor["uid"] in {2, 3}:
                tensor["dimensions"] = parameter_shape
                tensor["strides"] = parameter_strides
            else:
                tensor["dimensions"] = statistic_shape
                tensor["strides"] = statistic_strides
            if tensor["uid"] not in statistic_uids:
                tensor["data_type"] = "bfloat16"
        odd_suffix["graph"]["nodes"][0]["attributes"].update(
            {"rows": 771, "normalized_elements": 513}
        )
        odd_path = temporary / f"{operation}-suffix513-bfloat16.json"
        odd_output = temporary / f"{operation}-suffix513-bfloat16-artifact"
        write_json(odd_path, odd_suffix)
        odd_result = provider.compile_request(
            odd_path, odd_output, "libtriton_jit"
        )
        odd_manifest = load_artifact(odd_output)
        odd_stage = odd_manifest["program"]["stages"][0]
        odd_variant = odd_stage["variants"][0]
        require(
            odd_result.get("status") == "success"
            and odd_variant["launch"]["shared_memory"] == 512,
            f"{operation} bfloat16 suffix513 launch ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(odd_output, odd_variant, odd_stage["kernel"])

    def set_oversized_suffix(value: dict[str, Any]) -> None:
        for index in (0, 3):
            value["graph"]["tensors"][index]["dimensions"] = [2, 5, 65537]
            value["graph"]["tensors"][index]["strides"] = [327685, 65537, 1]
        for index in (1, 2):
            value["graph"]["tensors"][index]["dimensions"] = [1, 1, 65537]
            value["graph"]["tensors"][index]["strides"] = [65537, 65537, 1]
        value["graph"]["nodes"][0]["attributes"]["normalized_elements"] = 65537

    canonical = normalization_request(identity, "layernorm")
    for name, mutate, detail in (
        (
            "mismatched-data-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching data",
        ),
        (
            "bad-shape",
            lambda value: (
                value["graph"]["tensors"][1].__setitem__(
                    "dimensions", [1, 1, 16]
                ),
                value["graph"]["tensors"][1].__setitem__(
                    "strides", [16, 16, 1]
                ),
            ),
            "scale",
        ),
        (
            "bad-phase",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "forward_phase", 1
            ),
            "TRAINING",
        ),
        (
            "oversized-suffix",
            set_oversized_suffix,
            "normalized suffix",
        ),
    ):
        mutated = copy.deepcopy(canonical)
        mutate(mutated)
        path = temporary / f"normalization-{name}.json"
        mutated_output = temporary / f"normalization-{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            "rejected normalization wrote output",
        )

    for operation in ("layernorm", "rmsnorm"):
        autotune_path = temporary / f"{operation}-autotune.json"
        autotune_output = temporary / f"{operation}-autotune-artifact"
        write_json(
            autotune_path,
            normalization_request(identity, operation, autotune=True),
        )
        provider.compile_request(
            autotune_path, autotune_output, "libtriton_jit"
        )
        autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
        require(
            [variant["variant_id"] for variant in autotune_stage["variants"]]
            == ["block128_rows1_w4_s1", "block256_rows1_w4_s1"]
            and [
                variant["launch"]["shared_memory"]
                for variant in autotune_stage["variants"]
            ]
            == [16, 1024]
            and autotune_stage.get("tuning", {}).get("candidate_identity"),
            f"autotuned {operation} candidates are incomplete",
        )
        if compile_kernel:
            for variant in autotune_stage["variants"]:
                compile_add_kernel(
                    autotune_output, variant, autotune_stage["kernel"]
                )


def assert_matmul_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "matmul.json"
    output = temporary / "matmul-artifact"
    write_json(request_path, matmul_request(identity))
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(result.get("status") == "success", "MatMul was not compiled")

    manifest = load_artifact(output)
    stage = manifest["program"]["stages"][0]
    kernel = stage["kernel"]
    variant = stage["variants"][0]
    constants = [
        16,
        24,
        32,
        1,
        1,
        1,
        1,
        1,
        4,
        0,
        0,
        0,
        0,
        0,
        512,
        0,
        0,
        0,
        0,
        0,
        768,
        0,
        0,
        0,
        0,
        0,
        384,
        32,
        1,
        24,
        1,
        24,
        1,
        1,
        0,
        16,
        16,
        16,
        1,
    ]
    expected_signature = "*fp32:16,*fp32:16,*fp32:16," + ",".join(
        map(str, constants)
    )
    require(
        manifest.get("external_uids") == [1, 2, 3]
        and manifest.get("workspace") == {"size": 0, "alignment": 256}
        and stage.get("operation") == "matmul"
        and stage.get("source_node_ids") == [0]
        and stage.get("dependencies") == []
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "matmul_strided_kernel"
        and variant.get("variant_id") == "default"
        and variant.get("full_signature") == expected_signature
        and variant.get("argument_count") == 3
        and [item.get("uid") for item in variant.get("arguments", [])]
        == [1, 2, 3]
        and variant.get("launch")
        == {"grid": [2, 4, 1], "block": [128, 1, 1], "shared_memory": 2048},
        "MatMul artifact ABI is invalid",
    )
    source = output / kernel["materialized_source"]["path"]
    expected_source = SOURCE_ROOT / "kernels/common/matmul.py"
    require(
        source.is_file()
        and kernel["materialized_source"]["sha256"]
        == sha256_file(source)
        == sha256_file(expected_source),
        "MatMul source identity is invalid",
    )
    if compile_kernel:
        compile_add_kernel(output, variant, kernel)

    for data_type, pointer, expected_sizes in (
        ("float16", "fp16", [4096, 6144, 3072]),
        ("bfloat16", "bf16", [4096, 6144, 3072]),
    ):
        typed_document = matmul_request(identity)
        for tensor in typed_document["graph"]["tensors"]:
            tensor["data_type"] = data_type
        typed_path = temporary / f"matmul-{data_type}.json"
        typed_output = temporary / f"matmul-{data_type}-artifact"
        write_json(typed_path, typed_document)
        typed_result = provider.compile_request(
            typed_path, typed_output, "libtriton_jit"
        )
        typed_manifest = load_artifact(typed_output)
        typed_stage = typed_manifest["program"]["stages"][0]
        typed_variant = typed_stage["variants"][0]
        require(
            typed_result.get("status") == "success"
            and typed_variant.get("full_signature", "").startswith(
                f"*{pointer}:16,*{pointer}:16,*{pointer}:16,"
            )
            and typed_variant.get("full_signature", "").split(",")[-6:-4]
            == ["0", "0"]
            and [
                tensor["storage_size"] for tensor in typed_manifest["tensors"]
            ]
            == expected_sizes,
            f"MatMul {data_type} ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                typed_output, typed_variant, typed_stage["kernel"]
            )

    for data_type, pointer in (
        ("float32", "fp32"),
        ("float16", "fp16"),
        ("bfloat16", "bf16"),
    ):
        large_document = matmul_request(identity)
        large_tensors = large_document["graph"]["tensors"]
        for tensor in large_tensors:
            tensor["data_type"] = data_type
        large_tensors[0]["dimensions"] = [32, 512, 512]
        large_tensors[0]["strides"] = [262144, 512, 1]
        large_tensors[1]["dimensions"] = [32, 512, 512]
        large_tensors[1]["strides"] = [262144, 512, 1]
        large_tensors[2]["dimensions"] = [32, 512, 512]
        large_tensors[2]["strides"] = [262144, 512, 1]
        large_document["graph"]["nodes"][0]["attributes"] = {
            "batch": 32,
            "m": 512,
            "n": 512,
            "k": 512,
        }
        large_path = temporary / f"matmul-large-{data_type}.json"
        large_output = temporary / f"matmul-large-{data_type}-artifact"
        write_json(large_path, large_document)
        large_result = provider.compile_request(
            large_path, large_output, "libtriton_jit"
        )
        large_manifest = load_artifact(large_output)
        large_stage = large_manifest["program"]["stages"][0]
        large_variant = large_stage["variants"][0]
        require(
            large_result.get("status") == "success"
            and large_variant.get("full_signature", "").startswith(
                f"*{pointer}:16,*{pointer}:16,*{pointer}:16,"
            )
            and large_variant.get("launch")
            == {
                "grid": [64 if data_type == "float32" else 16, 32, 1],
                "block": [128, 1, 1],
                "shared_memory": 32768 if data_type == "float32" else 65536,
            },
            f"large MatMul {data_type} launch metadata is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                large_output, large_variant, large_stage["kernel"]
            )

    canonical = matmul_request(identity)
    for name, mutate, detail in (
        (
            "mismatched-dtype",
            lambda value: value["graph"]["tensors"][1].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "bad-shape",
            lambda value: value["graph"]["tensors"][1][
                "dimensions"
            ].__setitem__(-2, 31),
            "contraction",
        ),
        (
            "bad-attributes",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "m", 15
            ),
            "attributes",
        ),
    ):
        mutated = copy.deepcopy(canonical)
        mutate(mutated)
        path = temporary / f"matmul-{name}.json"
        rejected = temporary / f"matmul-{name}-artifact"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, rejected=rejected: provider.compile_request(
                path, rejected, "libtriton_jit"
            ),
            detail,
        )
        require(not rejected.exists(), "rejected MatMul wrote output")

    autotune_path = temporary / "matmul-autotune.json"
    autotune_output = temporary / "matmul-autotune-artifact"
    write_json(autotune_path, matmul_request(identity, autotune=True))
    provider.compile_request(autotune_path, autotune_output, "libtriton_jit")
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    require(
        [variant["variant_id"] for variant in autotune_stage["variants"]]
        == [
            "blockm64_blockn64_blockk32_group8_w4_s1",
            "blockm128_blockn128_blockk32_group8_w8_s1",
        ]
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned MatMul candidates are incomplete",
    )


def assert_convolution_indexing() -> None:
    """Compile large strides without allocating or executing unsafe layouts.

    The regression is a 2**31 element offset being multiplied in int32 before
    conversion at tt.addptr. Inspect the multiply itself, not the pointer type.
    Numerical execution remains exclusively compared with acDNN by the runner.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    path = BACKEND_ROOT / "thead/kernels/convolution.py"
    name = "_flagdnn_thead_convolution_indexing"
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        "missing convolution source",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        for function_name in (
            "conv_fprop_nd_kernel",
            "conv_dgrad_nd_kernel",
            "conv_wgrad_nd_kernel",
            "conv2d_bias_relu_kernel",
        ):
            function = getattr(module, function_name)
            pointer_count = (
                4 if function_name == "conv2d_bias_relu_kernel" else 3
            )
            names = function.arg_names
            for stride_axis in ("N", "W"):
                values = {key: 1 for key in names[pointer_count:]}
                values.update(
                    {key: 0 for key in values if key.startswith("PAD_")}
                )
                values.update(
                    {key: 16 for key in values if key.startswith("BLOCK_")}
                )
                values.update(INPUT_PRECISION="ieee")
                if stride_axis == "W":
                    values.update(XW=3, OW=3)
                values[f"X_STRIDE_{stride_axis}"] = 2**30
                if "M" in values:
                    values["M"] = 3
                if "FLIP_FILTER" in values:
                    values["FLIP_FILTER"] = False
                source = ASTSource(
                    function,
                    signature={
                        key: "*fp32" if i < pointer_count else "constexpr"
                        for i, key in enumerate(names)
                    },
                    constexprs={
                        (i,): values[key]
                        for i, key in enumerate(names)
                        if i >= pointer_count
                    },
                    attrs={
                        (i,): [["tt.divisibility", 16]]
                        for i in range(pointer_count)
                    },
                )
                compiled = triton.compile(
                    source,
                    target=GPUTarget("cuda", 80, 32),
                    options={"num_warps": 4, "num_stages": 1},
                )
                ir = compiled.asm["ttir"]
                constant = re.search(
                    r"(%[\w]+) = arith.constant (?:dense<1073741824>|1073741824)"
                    r" : (?:tensor<[^>]*xi64>|i64)",
                    ir,
                )
                require(
                    constant is not None,
                    f"{function_name}/{stride_axis}: stride was narrowed to int32",
                )
                require(
                    any(
                        "arith.muli" in line
                        and constant[1] in line
                        and "i64" in line
                        for line in ir.splitlines()
                    ),
                    f"{function_name}/{stride_axis}: large stride multiplication must happen in int64",
                )
    finally:
        sys.modules.pop(name, None)


def assert_convolution_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    expectations = {
        "convolution_fprop": {
            "provider": "thead_triton",
            "ownership": "platform",
            "function": "conv_fprop_nd_kernel",
            "uids": [1, 2, 3],
            "grid": [2, 1, 1],
        },
        "convolution_dgrad": {
            "provider": "thead_triton",
            "ownership": "platform",
            "function": "conv_dgrad_nd_kernel",
            "uids": [3, 2, 1],
            "grid": [2, 1, 1],
        },
        "convolution_wgrad": {
            "provider": "thead_triton",
            "ownership": "platform",
            "function": "conv_wgrad_nd_kernel",
            "uids": [3, 1, 2],
            "grid": [1, 9, 1],
        },
    }
    for operation, expected in expectations.items():
        request_path = temporary / f"{operation}.json"
        output = temporary / f"{operation}-artifact"
        document = convolution_request(identity, operation)
        write_json(request_path, document)
        result = provider.compile_request(
            request_path, output, "libtriton_jit"
        )
        require(
            result.get("status") == "success",
            f"{operation} was not compiled: {result!r}",
        )
        manifest = load_artifact(output)
        stage = manifest["program"]["stages"][0]
        kernel = stage["kernel"]
        variants = stage["variants"]
        require(
            manifest.get("external_uids") == [1, 2, 3]
            and manifest.get("workspace") == {"size": 0, "alignment": 256}
            and stage.get("operation") == operation
            and stage.get("source_node_ids") == [0]
            and stage.get("dependencies") == []
            and stage.get("tuning") is None
            and kernel.get("provider") == expected["provider"]
            and kernel.get("ownership") == expected["ownership"]
            and kernel.get("function") == expected["function"]
            and len(variants) == 1,
            f"{operation} stage identity is invalid",
        )
        variant = variants[0]
        require(
            variant.get("variant_id") == "default"
            and variant.get("argument_count") == 3
            and [
                argument.get("uid")
                for argument in variant.get("arguments", [])
            ]
            == expected["uids"]
            and variant.get("launch", {}).get("grid") == expected["grid"]
            and variant.get("launch", {}).get("block") == [128, 1, 1]
            and variant.get("launch", {}).get("shared_memory") == 2048,
            f"{operation} runtime ABI or launch is invalid",
        )
        source_path = output / kernel["materialized_source"]["path"]
        require(
            source_path.is_file()
            and kernel["materialized_source"]["sha256"]
            == sha256_file(source_path),
            f"{operation} materialized source identity is invalid",
        )
        if compile_kernel:
            compile_add_kernel(output, variant, kernel)

        for data_type, pointer in (
            ("float16", "fp16"),
            ("bfloat16", "bf16"),
        ):
            typed_document = convolution_request(identity, operation)
            for tensor in typed_document["graph"]["tensors"]:
                tensor["data_type"] = data_type
            typed_path = temporary / f"{operation}-{data_type}.json"
            typed_output = temporary / f"{operation}-{data_type}-artifact"
            write_json(typed_path, typed_document)
            typed_result = provider.compile_request(
                typed_path, typed_output, "libtriton_jit"
            )
            typed_manifest = load_artifact(typed_output)
            typed_stage = typed_manifest["program"]["stages"][0]
            typed_variant = typed_stage["variants"][0]
            require(
                typed_result.get("status") == "success"
                and typed_variant["full_signature"].startswith(
                    f"*{pointer}:16,*{pointer}:16,*{pointer}:16,"
                )
                and all(
                    tensor["storage_size"] % 2 == 0
                    for tensor in typed_manifest["tensors"]
                ),
                f"{operation}/{data_type} convolution ABI is invalid",
            )
            if compile_kernel:
                compile_add_kernel(
                    typed_output, typed_variant, typed_stage["kernel"]
                )

        for spatial_rank, shapes, strides, n_outputs in (
            (
                1,
                ([2, 4, 16], [6, 4, 3], [2, 6, 16]),
                ([64, 16, 1], [12, 3, 1], [96, 16, 1]),
                {
                    "convolution_fprop": 192,
                    "convolution_dgrad": 128,
                    "convolution_wgrad": 72,
                }[operation],
            ),
            (
                3,
                ([1, 2, 5, 6, 7], [4, 2, 3, 3, 3], [1, 4, 5, 6, 7]),
                (
                    [420, 210, 42, 7, 1],
                    [54, 27, 9, 3, 1],
                    [840, 210, 42, 7, 1],
                ),
                {
                    "convolution_fprop": 840,
                    "convolution_dgrad": 420,
                    "convolution_wgrad": 216,
                }[operation],
            ),
        ):
            ranked_document = convolution_request(identity, operation)
            for tensor, dimensions, tensor_strides in zip(
                ranked_document["graph"]["tensors"],
                shapes,
                strides,
                strict=True,
            ):
                tensor["dimensions"] = dimensions
                tensor["strides"] = tensor_strides
            ranked_attributes = ranked_document["graph"]["nodes"][0][
                "attributes"
            ]
            ranked_attributes.update(
                {
                    "spatial_rank": spatial_rank,
                    "n_outputs": n_outputs,
                    "pre_padding": [1] * spatial_rank,
                    "post_padding": [1] * spatial_rank,
                    "stride": [1] * spatial_rank,
                    "dilation": [1] * spatial_rank,
                }
            )
            ranked_path = temporary / f"{operation}-{spatial_rank}d.json"
            ranked_output = temporary / f"{operation}-{spatial_rank}d-artifact"
            write_json(ranked_path, ranked_document)
            ranked_result = provider.compile_request(
                ranked_path, ranked_output, "libtriton_jit"
            )
            ranked_stage = load_artifact(ranked_output)["program"]["stages"][0]
            ranked_variant = ranked_stage["variants"][0]
            require(
                ranked_result.get("status") == "success"
                and ranked_stage["kernel"]["function"] == expected["function"],
                f"{operation}/{spatial_rank}D convolution was not compiled",
            )
            if compile_kernel:
                compile_add_kernel(
                    ranked_output, ranked_variant, ranked_stage["kernel"]
                )

        for name, mutate, detail in (
            (
                "mismatched-dtype",
                lambda value: value["graph"]["tensors"][0].__setitem__(
                    "data_type", "float16"
                ),
                "matching",
            ),
            (
                "bad-padding",
                lambda value: value["graph"]["nodes"][0][
                    "attributes"
                ].__setitem__("post_padding", [0, 1]),
                "output shape",
            ),
            (
                "bad-groups",
                lambda value: value["graph"]["nodes"][0][
                    "attributes"
                ].__setitem__("groups", 3),
                "groups",
            ),
            (
                "int64-coordinate-overflow",
                lambda value: value["graph"]["nodes"][0][
                    "attributes"
                ].__setitem__("dilation", [2**62, 2**62]),
                "spatial coordinates exceed int64",
            ),
            (
                "int32-index-overflow",
                lambda value, operation=operation: (
                    value["graph"]["tensors"][0]["dimensions"].__setitem__(
                        0, 2**30
                    ),
                    value["graph"]["tensors"][2]["dimensions"].__setitem__(
                        0, 2**30
                    ),
                    value["graph"]["nodes"][0]["attributes"].__setitem__(
                        "n_outputs",
                        (
                            2**30 * 2 * 5 * 5
                            if operation != "convolution_wgrad"
                            else 2 * 2 * 3 * 3
                        ),
                    ),
                ),
                "int32",
            ),
        ):
            mutated = copy.deepcopy(document)
            mutate(mutated)
            path = temporary / f"{operation}-{name}.json"
            rejected = temporary / f"{operation}-{name}-artifact"
            write_json(path, mutated)
            expect_value_error(
                lambda path=path, rejected=rejected: provider.compile_request(
                    path, rejected, "libtriton_jit"
                ),
                detail,
            )
            require(
                not rejected.exists(),
                f"rejected {operation} mutation wrote an artifact",
            )

        autotune_path = temporary / f"{operation}-autotune.json"
        autotune_output = temporary / f"{operation}-autotune-artifact"
        write_json(
            autotune_path,
            convolution_request(identity, operation, autotune=True),
        )
        provider.compile_request(
            autotune_path, autotune_output, "libtriton_jit"
        )
        autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
        require(
            [
                candidate["variant_id"]
                for candidate in autotune_stage["variants"]
            ]
            == ["tile16_w4_s1", "tile32_w4_s1"]
            and autotune_stage.get("tuning", {}).get("candidate_identity"),
            f"autotuned {operation} candidates are incomplete",
        )


def assert_conv_bias_relu_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "conv-bias-relu.json"
    output = temporary / "conv-bias-relu-artifact"
    document = conv_bias_relu_request(identity)
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"ConvBiasRelu was not compiled: {result!r}",
    )
    manifest = load_artifact(output)
    stage = manifest["program"]["stages"][0]
    kernel = stage["kernel"]
    variant = stage["variants"][0]
    require(
        manifest.get("external_uids") == [1, 2, 3, 6]
        and manifest.get("tensor_count") == 6
        and manifest.get("workspace") == {"size": 0, "alignment": 256}
        and all(
            "workspace_offset" not in tensor
            for tensor in manifest.get("tensors", [])
            if tensor.get("virtual") is True
        )
        and stage.get("operation") == "conv_bias_relu"
        and stage.get("source_node_ids") == [0, 1, 2]
        and stage.get("dependencies") == []
        and stage.get("tuning") is None
        and kernel.get("provider") == "thead_triton"
        and kernel.get("ownership") == "platform"
        and kernel.get("function") == "conv2d_bias_relu_kernel"
        and variant.get("variant_id") == "default"
        and variant.get("argument_count") == 4
        and [item.get("uid") for item in variant.get("arguments", [])]
        == [1, 2, 3, 6]
        and variant.get("launch")
        == {"grid": [16, 2, 1], "block": [128, 1, 1], "shared_memory": 2048},
        "ConvBiasRelu artifact ABI is invalid",
    )
    materialized = output / kernel["materialized_source"]["path"]
    require(
        materialized.is_file()
        and kernel["materialized_source"]["sha256"]
        == sha256_file(materialized)
        == sha256_file(BACKEND_ROOT / "thead/kernels/convolution.py"),
        "ConvBiasRelu platform kernel identity is invalid",
    )

    for data_type, pointer in (
        ("float16", "fp16"),
        ("bfloat16", "bf16"),
    ):
        typed_document = conv_bias_relu_request(identity)
        for tensor in typed_document["graph"]["tensors"]:
            tensor["data_type"] = data_type
        typed_path = temporary / f"conv-bias-relu-{data_type}.json"
        typed_output = temporary / f"conv-bias-relu-{data_type}-artifact"
        write_json(typed_path, typed_document)
        typed_result = provider.compile_request(
            typed_path, typed_output, "libtriton_jit"
        )
        typed_manifest = load_artifact(typed_output)
        typed_stage = typed_manifest["program"]["stages"][0]
        typed_variant = typed_stage["variants"][0]
        require(
            typed_result.get("status") == "success"
            and typed_variant["full_signature"].startswith(
                f"*{pointer}:16,*{pointer}:16,*{pointer}:16,*{pointer}:16,"
            )
            and typed_manifest["workspace"] == {"size": 0, "alignment": 256}
            and all(
                "workspace_offset" not in tensor
                for tensor in typed_manifest["tensors"]
                if tensor.get("virtual") is True
            ),
            f"ConvBiasRelu/{data_type} ABI is invalid",
        )
        if compile_kernel:
            compile_add_kernel(
                typed_output, typed_variant, typed_stage["kernel"]
            )

    for name, mutate, detail in (
        (
            "mismatched-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            "bad-dataflow",
            lambda value: value["graph"]["nodes"][1]["inputs"][0].__setitem__(
                "uid", 6
            ),
            "dataflow",
        ),
        (
            "bad-relu-mode",
            lambda value: value["graph"]["nodes"][2]["attributes"].__setitem__(
                "mode", 36
            ),
            "ReLU",
        ),
    ):
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"conv-bias-relu-{name}.json"
        rejected = temporary / f"conv-bias-relu-{name}-artifact"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, rejected=rejected: provider.compile_request(
                path, rejected, "libtriton_jit"
            ),
            detail,
        )
        require(not rejected.exists(), "rejected ConvBiasRelu wrote output")

    autotune_path = temporary / "conv-bias-relu-autotune.json"
    autotune_output = temporary / "conv-bias-relu-autotune-artifact"
    write_json(autotune_path, conv_bias_relu_request(identity, autotune=True))
    provider.compile_request(autotune_path, autotune_output, "libtriton_jit")
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    require(
        [item["variant_id"] for item in autotune_stage["variants"]]
        == ["tile16_w4_s1", "tile32_w4_s1"]
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned ConvBiasRelu candidates are incomplete",
    )
    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_generic_unary_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    operation: str,
    label: str,
    mode: int,
    wrong_mode: int,
    compile_kernel: bool,
    stage_operation: str | None = None,
    expected_constants: list[str] | None = None,
) -> None:
    request_path = temporary / f"{operation}.json"
    output = temporary / f"{operation}-artifact"
    document = request(identity, operation)
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"{label} unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path)
        and manifest.get("external_uids") == [1, 2]
        and manifest.get("tensor_count") == 2,
        f"{label} manifest request, binding, or tensor identity is invalid",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, f"{label} must emit exactly one execution stage")
    stage = stages[0]
    kernel = stage.get("kernel", {})
    variants = stage.get("variants", [])
    expected_stage_operation = stage_operation or operation
    require(
        stage.get("operation") == expected_stage_operation
        and stage.get("tuning") is None
        and kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "unary_pointwise_contiguous_kernel"
        and len(variants) == 1,
        f"{label} did not select one non-autotuned common unary stage",
    )
    source = kernel.get("materialized_source", {})
    source_path = output / source.get("path", "")
    require(
        source_path.is_file()
        and source.get("size") == source_path.stat().st_size
        and source.get("sha256") == sha256_file(source_path)
        and source.get("sha256")
        == sha256_file(SOURCE_ROOT / "kernels/common/unary.py")
        and kernel.get("registry_sha256") == registry_sha256(),
        f"{label} common unary source or registry identity is invalid",
    )
    variant = variants[0]
    if expected_constants is None:
        expected_constants = [
            str(mode),
            "0.0",
            "0.0",
            "0.0",
            "0",
            "1.0",
            "1.0",
            "1.0",
            "1",
        ]
    require(
        variant.get("variant_id") == "default"
        and variant.get("full_signature")
        == "*fp32:16,*fp32:16,i32," + ",".join(expected_constants) + ",256"
        and variant.get("compile_options")
        == {
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        f"{label} variant specialization is invalid",
    )
    arguments = variant.get("arguments", [])
    require(
        variant.get("argument_count") == 3
        and [argument.get("uid") for argument in arguments[:2]] == [1, 2]
        and [argument.get("kind") for argument in arguments]
        == ["tensor", "tensor", "scalar_i32"]
        and arguments[2]
        == {"kind": "scalar_i32", "name": "n_elements", "value": 16}
        and variant.get("launch")
        == {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        f"{label} runtime ABI or launch geometry is invalid",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
        (
            f"{operation}-bad-mode",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "mode", wrong_mode
            ),
            "mode",
        ),
        (
            f"{operation}-bad-dtype",
            lambda value: value["graph"]["tensors"][0].__setitem__(
                "data_type", "float16"
            ),
            "matching",
        ),
        (
            f"{operation}-bad-attribute",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "swish_beta", 2.0
            ),
            "default attributes",
        ),
        (
            f"{operation}-bad-elements",
            lambda value: value["graph"]["nodes"][0]["attributes"].__setitem__(
                "n_elements", 15
            ),
            "n_elements",
        ),
    ]
    if operation == "leaky_relu":
        mutations.extend(
            [
                (
                    "leaky-relu-unqualified-slope",
                    lambda value: (
                        value["graph"]["nodes"][0]["attributes"].__setitem__(
                            "negative_slope", 0.25
                        ),
                        value["graph"]["nodes"][0]["attributes"].__setitem__(
                            "relu_lower_clip_slope", 0.25
                        ),
                    ),
                    "slope",
                ),
                (
                    "leaky-relu-mismatched-lowered-slope",
                    lambda value: value["graph"]["nodes"][0][
                        "attributes"
                    ].__setitem__("relu_lower_clip_slope", 0.0),
                    "slope",
                ),
            ]
        )
    for name, mutate, detail in mutations:
        mutated = copy.deepcopy(document)
        mutate(mutated)
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected {label} mutation {name} wrote an artifact directory",
        )

    autotune_path = temporary / f"{operation}-autotune.json"
    autotune_output = temporary / f"{operation}-autotune-artifact"
    write_json(autotune_path, request(identity, operation, autotune=True))
    autotune_result = provider.compile_request(
        autotune_path, autotune_output, "libtriton_jit"
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        autotune_result.get("status") == "success"
        and [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[3:12]
            == expected_constants
            for candidate in autotune_variants
        )
        and autotune_stage.get("tuning", {}).get("candidate_identity"),
        f"autotuned {label} candidates are incomplete or have wrong constants",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def attention_requests(identity: str) -> list[dict[str, Any]]:
    # Public forward profiles, plus each backward profile's forward inputs.
    profiles = (
        (False, 1, 2, 2, 64, 64, 64, 64, "float16", False, False, True),
        (False, 1, 4, 2, 48, 48, 64, 64, "bfloat16", True, False, True),
        (False, 2, 2, 2, 32, 40, 64, 64, "float16", False, True, True),
        (False, 1, 2, 2, 16, 24, 64, 64, "float16", False, False, False),
        (True, 1, 2, 2, 32, 32, 32, 32, "float16", False, False, True),
        (True, 1, 4, 2, 32, 32, 64, 64, "bfloat16", True, False, True),
        (True, 1, 2, 2, 24, 32, 32, 64, "float16", False, False, True),
        (True, 2, 4, 4, 32, 40, 64, 64, "float16", False, True, True),
        (False, 1, 2, 2, 64, 64, 128, 128, "fp8_e4m3", False, False, True),
        (False, 1, 4, 2, 48, 48, 128, 128, "fp8_e5m2", True, False, True),
        (False, 1, 2, 2, 32, 40, 128, 128, "fp8_e4m3", False, False, True),
        (False, 1, 2, 2, 32, 40, 128, 128, "fp8_e4m3", False, False, False),
        (True, 1, 2, 2, 64, 64, 128, 128, "fp8_e4m3", False, False, True),
        (True, 1, 4, 2, 48, 48, 128, 128, "fp8_e5m2", True, False, True),
    )
    documents = []
    for index, profile in enumerate(profiles):
        (
            backward,
            batch,
            heads,
            kv_heads,
            sq,
            sk,
            d,
            dv,
            dtype,
            causal,
            bias,
            stats,
        ) = profile
        for is_backward in (True, False) if backward else (False,):
            fp8 = dtype.startswith("fp8_")
            operation = (
                ("sdpa_fp8_backward" if is_backward else "sdpa_fp8")
                if fp8
                else ("sdpa_backward" if is_backward else "sdpa")
            )
            inputs = ["q", "k", "v"] + (
                ["o", "do", "stats"] if is_backward else []
            )
            if bias:
                inputs.append("bias")
            outputs = ["dq", "dk", "dv"] if is_backward else ["o", "stats"]
            if bias and is_backward:
                outputs.append("dbias")
            shapes = {
                "q": [batch, heads, sq, d],
                "k": [batch, kv_heads, sk, d],
                "v": [batch, kv_heads, sk, dv],
                "o": [batch, heads, sq, dv],
                "stats": [batch, heads, sq, 1],
                "bias": [1, heads, sq, sk],
            }
            shapes.update(
                dq=shapes["q"],
                dk=shapes["k"],
                dv=shapes["v"],
                do=shapes["o"],
                dbias=shapes["bias"],
            )
            float_names = {"stats"}
            if fp8:
                scales = (
                    [
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
                    ]
                    if is_backward
                    else [
                        "descale_q",
                        "descale_k",
                        "descale_v",
                        "descale_s",
                        "scale_s",
                        "scale_o",
                    ]
                )
                maxima = (
                    ["amax_dq", "amax_dk", "amax_dv", "amax_dp"]
                    if is_backward
                    else ["amax_s", "amax_o"]
                )
                inputs += scales
                outputs += maxima
                float_names.update(scales + maxima)
                shapes.update({name: [1, 1, 1, 1] for name in scales + maxima})
            names = inputs + outputs
            uids = {name: uid for uid, name in enumerate(names, 1)}
            tensors = []
            for name in names:
                shape = shapes[name]
                tensors.append(
                    {
                        "uid": uids[name],
                        "data_type": (
                            "float32" if name in float_names else dtype
                        ),
                        "dimensions": shape,
                        "strides": [
                            math.prod(shape[i + 1 :]) for i in range(4)
                        ],
                        "alignment": 16,
                        "virtual": name == "stats" and not stats,
                    }
                )
            attributes = dict(
                attn_scale=0.2 if d == 32 and dv == 64 else 1 / math.sqrt(d),
                attn_scale_set=True,
                banded=int(causal),
                batch=batch,
                causal_top_left=int(causal and sq == sk),
                diagonal_alignment=0,
                diagonal_band_left_bound=0,
                diagonal_band_right_bound=0,
                generate_stats=int(stats),
                has_bias=int(bias),
                has_dbias=int(bias and is_backward),
                head_dimension=d,
                heads=heads,
                key_heads=kv_heads,
                left_bound_set=False,
                max_diag=0 if causal else 1 << 30,
                min_diag=-(1 << 30),
                q_per_k=heads // kv_heads,
                q_per_v=heads // kv_heads,
                reverse_causal=int(causal),
                right_bound_set=causal,
                sequence_kv=sk,
                sequence_q=sq,
                value_dimension=dv,
                value_heads=kv_heads,
            )
            documents.append(
                {
                    "schema_version": 3,
                    "flagdnn_version": "0.2.0",
                    "backend": "thead",
                    "target": TARGET,
                    "compiler_identity": identity,
                    "build_options": {
                        "heuristic_modes": ["A"],
                        "autotune": True,
                    },
                    "graph": {
                        "name": f"attention-{index}-{operation}",
                        "tensor_count": len(tensors),
                        "tensors": tensors,
                        "node_count": 1,
                        "nodes": [
                            {
                                "id": 0,
                                "type": operation,
                                "name": operation,
                                "compute_data_type": "float32",
                                "inputs": [
                                    {"name": name, "uid": uids[name]}
                                    for name in inputs
                                ],
                                "outputs": [
                                    {"name": name, "uid": uids[name]}
                                    for name in outputs
                                ],
                                "attributes": attributes,
                            }
                        ],
                    },
                }
            )
    return documents


def assert_attention_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
    fp8_only: bool = False,
) -> None:
    for document in attention_requests(identity):
        if fp8_only and not document["graph"]["nodes"][0]["type"].startswith(
            "sdpa_fp8"
        ):
            continue
        graph = document["graph"]
        name = graph["name"]
        source = temporary / f"{name}.json"
        output = temporary / f"{name}-artifact"
        write_json(source, document)
        result = provider.compile_request(source, output, "libtriton_jit")
        require(result["status"] == "success", f"{name} compile failed")
        manifest = load_artifact(output)
        external = [t["uid"] for t in graph["tensors"] if not t["virtual"]]
        require(
            manifest["external_uids"] == external,
            f"{name} external UID order changed",
        )
        require(
            manifest["workspace"] == {"size": 0, "alignment": 256},
            f"{name} introduced unbound attention scratch memory",
        )
        stages = manifest["program"]["stages"]
        require(
            len(stages) == 1 and stages[0]["source_node_ids"] == [0],
            f"{name} no longer captures one kernel per Graph node",
        )
        stage = stages[0]
        require(
            stage["tuning"] and len(stage["variants"]) == 2,
            f"{name} lost its autotune candidates",
        )
        for variant in stage["variants"]:
            arguments = variant["arguments"]
            require(
                variant["argument_count"] == len(external)
                and all(arg["kind"] == "tensor" for arg in arguments)
                and sorted(arg["uid"] for arg in arguments)
                == sorted(external),
                f"{name} runtime ABI includes optional pointers or constants",
            )
            node = graph["nodes"][0]
            attrs = node["attributes"]
            fp8 = node["type"].startswith("sdpa_fp8")
            fp8_uids = {
                t["uid"]
                for t in graph["tensors"]
                if t["data_type"].startswith("fp8_")
            }
            for argument in arguments:
                require(
                    (argument.get("storage_view") == "fp8_bytes")
                    == (argument["uid"] in fp8_uids),
                    f"{name} lost explicit FP8 byte storage views",
                )
            if fp8:
                require(
                    variant["launch"]["shared_memory"]
                    == (24576 if node["type"].endswith("backward") else 8192),
                    f"{name} FP8 shared memory metadata differs from native compilation",
                )
            absent = int(not attrs["has_bias"]) + (
                int(not attrs["has_dbias"])
                if node["type"] == "sdpa_backward"
                else int(not attrs["generate_stats"])
            )
            if fp8:
                absent = int(not attrs["generate_stats"])
            require(
                variant["full_signature"].split(",").count("nullopt")
                == absent,
                f"{name} lost specialized None pointers",
            )
            if compile_kernel:
                compile_add_kernel(output, variant, stage["kernel"])
        fields = [
            "dtype",
            "compute",
            "geometry",
            "stride",
            "port",
            "scale",
            "mask",
            "group",
            "storage",
        ]
        if graph["nodes"][0]["type"].startswith("sdpa_fp8"):
            fields += ["scalar_dtype", "scalar_shape", "amax_dtype", "bias"]
        for field in fields:
            invalid = copy.deepcopy(document)
            node = invalid["graph"]["nodes"][0]
            tensor = invalid["graph"]["tensors"][0]
            if field == "dtype":
                tensor["data_type"] = "float32"
            elif field == "compute":
                node["compute_data_type"] = "float16"
            elif field == "geometry":
                tensor["dimensions"][2] += 1
            elif field == "stride":
                tensor["strides"][0] += 16
            elif field == "port":
                node["inputs"][0]["name"] = "invalid"
            elif field == "scale":
                node["attributes"]["attn_scale"] = "0.125"
            elif field == "mask":
                node["attributes"]["left_bound_set"] = True
            elif field == "group":
                node["attributes"]["q_per_k"] += 1
            elif field == "scalar_dtype":
                uid = next(
                    port["uid"]
                    for port in node["inputs"]
                    if port["name"] == "scale_s"
                )
                next(
                    t for t in invalid["graph"]["tensors"] if t["uid"] == uid
                )["data_type"] = "float16"
            elif field == "scalar_shape":
                uid = next(
                    port["uid"]
                    for port in node["inputs"]
                    if port["name"] == "scale_s"
                )
                next(
                    t for t in invalid["graph"]["tensors"] if t["uid"] == uid
                )["dimensions"][-1] = 2
            elif field == "amax_dtype":
                uid = next(
                    port["uid"]
                    for port in node["outputs"]
                    if port["name"].startswith("amax_")
                )
                next(
                    t for t in invalid["graph"]["tensors"] if t["uid"] == uid
                )["data_type"] = "float16"
            elif field == "bias":
                node["attributes"]["has_bias"] = 1
            else:
                tensor["virtual"] = True
            bad = temporary / f"{name}-{field}-invalid.json"
            bad_output = temporary / f"{name}-{field}-invalid-artifact"
            write_json(bad, invalid)
            expect_value_error(
                lambda: provider.compile_request(
                    bad, bad_output, "libtriton_jit"
                ),
                "",
            )
            require(
                not bad_output.exists(),
                f"{name}/{field} rejection wrote an artifact",
            )


def assert_boolean_pointwise_artifacts(
    provider: object, temporary: Path, identity: str, *, compile_kernel: bool
) -> None:
    for operation in (
        "logical_not",
        "logical_and",
        "logical_or",
        "binary_select",
    ):
        dtypes = (
            ("float32", "float16", "bfloat16")
            if operation == "binary_select"
            else ("boolean",)
        )
        for dtype in dtypes:
            for strided in (
                (False, True) if operation == "binary_select" else (False,)
            ):
                document = request(identity, operation, autotune=True)
                graph = document["graph"]
                if operation == "binary_select":
                    for tensor in graph["tensors"]:
                        if tensor["uid"] != 3:
                            tensor["data_type"] = dtype
                    if strided:
                        for tensor, strides in zip(
                            graph["tensors"],
                            ([31, 9, 1], [37, 11, 1], [12, 4, 1], [43, 14, 1]),
                        ):
                            tensor["dimensions"] = [2, 3, 4]
                            tensor["strides"] = strides
                        graph["nodes"][0]["attributes"]["n_elements"] = 24
                name = (
                    f"{operation}-{dtype}-{'strided' if strided else 'dense'}"
                )
                source = temporary / f"{name}.json"
                output = temporary / f"{name}-artifact"
                write_json(source, document)
                result = provider.compile_request(
                    source, output, "libtriton_jit"
                )
                require(
                    result["status"] == "success",
                    f"{name} compile failed: {result}",
                )
                stage = load_artifact(output)["program"]["stages"][0]
                require(
                    stage["tuning"] is not None
                    and len(stage["variants"]) == 2,
                    f"{name} lost its autotune candidates",
                )
                for variant in stage["variants"]:
                    require(
                        variant["argument_count"] == len(graph["tensors"]) + 1,
                        f"{name} runtime arity differs from graph",
                    )
                    require(
                        [arg["uid"] for arg in variant["arguments"][:-1]]
                        == [t["uid"] for t in graph["tensors"]],
                        f"{name} tensor ABI order changed",
                    )
                    if operation == "binary_select":
                        expected = (
                            "binary_select_strided_kernel"
                            if strided
                            else "binary_select_tensor_kernel"
                        )
                        require(
                            stage["kernel"]["function"] == expected,
                            f"{name} kernel mismatch",
                        )
                        require(
                            variant["full_signature"].split(",")[2]
                            == "*i8:16",
                            f"{name} mask pointer is not byte boolean",
                        )
                    else:
                        require(
                            all(
                                token == "*i8:16"
                                for token in variant["full_signature"].split(
                                    ","
                                )[: len(graph["tensors"])]
                            ),
                            f"{name} logical tensor ABI is not byte boolean",
                        )
                    if compile_kernel:
                        compile_add_kernel(output, variant, stage["kernel"])
                for field in ("dtype", "compute", "mode", "geometry", "port"):
                    invalid = copy.deepcopy(document)
                    node = invalid["graph"]["nodes"][0]
                    if field == "dtype":
                        invalid["graph"]["tensors"][
                            2 if operation == "binary_select" else 0
                        ]["data_type"] = "float32"
                    elif field == "compute":
                        node["compute_data_type"] = (
                            "boolean"
                            if operation == "binary_select"
                            else "float32"
                        )
                    elif field == "mode":
                        node["attributes"]["mode"] = 1
                    elif field == "geometry":
                        invalid["graph"]["tensors"][0]["dimensions"][0] += 1
                    else:
                        node["inputs"][0]["name"] = "invalid"
                    bad = temporary / f"{name}-{field}-invalid.json"
                    write_json(bad, invalid)
                    expect_value_error(
                        lambda: provider.compile_request(
                            bad, output, "libtriton_jit"
                        ),
                        "",
                    )

    for dtype in ("float32", "float16", "bfloat16"):
        document = request(identity, "binary_select", autotune=False)
        for tensor in document["graph"]["tensors"]:
            tensor["dimensions"] = [524288]
            tensor["strides"] = [1]
            if tensor["uid"] != 3:
                tensor["data_type"] = dtype
        document["graph"]["nodes"][0]["attributes"]["n_elements"] = 524288
        source = temporary / f"binary-select-large-{dtype}.json"
        output = temporary / f"binary-select-large-{dtype}-artifact"
        write_json(source, document)
        provider.compile_request(source, output, "libtriton_jit")
        stage = load_artifact(output)["program"]["stages"][0]
        variant = stage["variants"][0]
        require(
            variant["full_signature"].endswith(",1024")
            and variant["launch"]["shared_memory"]
            == (4096 if dtype == "float32" else 0),
            "large BinarySelect tile has incorrect PPU shared memory",
        )
        if compile_kernel:
            compile_add_kernel(output, variant, stage["kernel"])


def assert_typed_pointwise_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    for operation, tensor_count, function, mode in (
        ("add", 3, "binary_contiguous_kernel", 1),
        ("abs", 2, "unary_pointwise_contiguous_kernel", 9),
    ):
        for data_type, pointer in (
            ("float16", "fp16"),
            ("bfloat16", "bf16"),
        ):
            document = request(identity, operation)
            for tensor in document["graph"]["tensors"]:
                tensor["data_type"] = data_type
            request_path = temporary / f"{operation}-{data_type}.json"
            output = temporary / f"{operation}-{data_type}-artifact"
            write_json(request_path, document)
            result = provider.compile_request(
                request_path, output, "libtriton_jit"
            )
            require(
                result.get("status") == "success",
                f"{operation}/{data_type} unexpectedly returned {result!r}",
            )
            manifest = load_artifact(output)
            require(
                manifest.get("tensor_count") == tensor_count
                and all(
                    tensor.get("data_type") == data_type
                    and tensor.get("storage_size") == 32
                    for tensor in manifest.get("tensors", [])
                ),
                f"{operation}/{data_type} tensor storage is invalid",
            )
            stage = manifest["program"]["stages"][0]
            variant = stage["variants"][0]
            expected_pointer_prefix = ",".join(
                f"*{pointer}:16" for _ in range(tensor_count)
            )
            require(
                stage["kernel"]["function"] == function
                and variant["full_signature"].startswith(
                    expected_pointer_prefix + ",i32,"
                )
                and f",{mode}," in variant["full_signature"],
                f"{operation}/{data_type} typed signature is invalid",
            )
            if compile_kernel:
                compile_add_kernel(output, variant, stage["kernel"])


def assert_strided_pointwise_artifacts(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    definitions = (
        (
            "add",
            "binary_strided_kernel",
            ([31, 9, 1], [37, 11, 1], [43, 13, 1]),
            4,
        ),
        (
            "abs",
            "unary_pointwise_strided_kernel",
            ([31, 9, 1], [37, 11, 1]),
            3,
        ),
    )
    for operation, function, strides, argument_count in definitions:
        document = request(identity, operation)
        for tensor, tensor_strides in zip(
            document["graph"]["tensors"], strides, strict=True
        ):
            tensor["dimensions"] = [2, 3, 4]
            tensor["strides"] = tensor_strides
        document["graph"]["nodes"][0]["attributes"]["n_elements"] = 24
        request_path = temporary / f"{operation}-strided.json"
        output = temporary / f"{operation}-strided-artifact"
        write_json(request_path, document)
        result = provider.compile_request(
            request_path, output, "libtriton_jit"
        )
        require(
            result.get("status") == "success",
            f"{operation}/strided unexpectedly returned {result!r}",
        )
        stage = load_artifact(output)["program"]["stages"][0]
        variant = stage["variants"][0]
        signature = variant.get("full_signature", "").split(",")
        require(
            stage["kernel"]["function"] == function
            and variant.get("argument_count") == argument_count
            and signature[: len(strides)] == ["*fp32:16"] * len(strides)
            and signature[len(strides)] == "i32"
            and signature[len(strides) + 1 : len(strides) + 9]
            == ["1", "1", "1", "1", "1", "2", "3", "4"],
            f"{operation}/strided signature or kernel is invalid",
        )
        require(
            [
                tensor.get("storage_size")
                for tensor in load_artifact(output)["tensors"]
            ]
            == [
                4
                * (
                    1
                    + sum(
                        (dimension - 1) * stride
                        for dimension, stride in zip(
                            [2, 3, 4], tensor_strides, strict=True
                        )
                    )
                )
                for tensor_strides in strides
            ],
            f"{operation}/strided storage spans are invalid",
        )
        if compile_kernel:
            compile_add_kernel(output, variant, stage["kernel"])

        overlapping = copy.deepcopy(document)
        overlapping["graph"]["tensors"][0]["strides"] = [4, 4, 1]
        overlapping_path = temporary / f"{operation}-overlapping.json"
        overlapping_output = temporary / f"{operation}-overlapping-artifact"
        write_json(overlapping_path, overlapping)
        expect_value_error(
            lambda path=overlapping_path, rejected=overlapping_output: provider.compile_request(
                path, rejected, "libtriton_jit"
            ),
            "non-overlapping",
        )
        require(
            not overlapping_output.exists(),
            f"rejected {operation} overlapping strides wrote artifacts",
        )

    # Dense permutations must preserve element pairing across operands; padding
    # or mismatched physical orders must never use the linear fast path.
    for operation, contiguous_function in (
        ("add", "binary_contiguous_kernel"),
        ("abs", "unary_pointwise_contiguous_kernel"),
        ("add_square", "add_square_contiguous_kernel"),
    ):
        for label, dimensions, strides, mixed, expected_contiguous in (
            ("nhwc", [2, 3, 4, 5], [60, 1, 15, 3], False, True),
            ("padded_nhwc", [2, 3, 4, 5], [64, 1, 15, 3], False, False),
            ("mixed_layout", [2, 3, 4, 5], [60, 1, 15, 3], True, False),
            ("singleton", [2, 1, 4, 5], [20, 999, 5, 1], False, True),
            (
                "large_nhwc",
                [8, 16, 64, 128],
                [131072, 1, 2048, 16],
                False,
                True,
            ),
        ):
            document = request(identity, operation)
            for tensor in document["graph"]["tensors"]:
                tensor["dimensions"] = dimensions
                tensor["strides"] = strides
            if mixed:
                document["graph"]["tensors"][-1]["strides"] = [60, 20, 5, 1]
            if label == "singleton":
                document["graph"]["tensors"][-1]["strides"] = [20, 1, 5, 1]
            for node in document["graph"]["nodes"]:
                node["attributes"]["n_elements"] = math.prod(dimensions)
            request_path = temporary / f"{operation}-{label}.json"
            output = temporary / f"{operation}-{label}-artifact"
            write_json(request_path, document)
            provider.compile_request(request_path, output, "libtriton_jit")
            stage = load_artifact(output)["program"]["stages"][0]
            require(
                (stage["kernel"]["function"] == contiguous_function)
                == expected_contiguous,
                f"{operation}/{label} selected an incorrect physical traversal",
            )
            if compile_kernel:
                compile_add_kernel(
                    output, stage["variants"][0], stage["kernel"]
                )


def assert_sub_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "sub.json"
    output = temporary / "sub-artifact"
    document = request(identity, "sub")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Sub unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path),
        "Sub manifest request hash does not match exact Graph IR bytes",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Sub must emit exactly one execution stage")
    stage = stages[0]
    require(
        stage.get("operation") == "sub" and stage.get("tuning") is None,
        "Sub stage operation or tuning state is invalid",
    )
    kernel = stage.get("kernel", {})
    require(
        kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "binary_contiguous_kernel",
        "Sub did not select the common contiguous kernel",
    )
    variants = stage.get("variants", [])
    require(len(variants) == 1, "Sub must emit one non-autotuned variant")
    variant = variants[0]
    require(
        variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,17,1.0,256",
        "Sub variant did not specialize FLAGDNN_POINTWISE_SUB (17)",
    )
    require(
        [argument.get("uid") for argument in variant.get("arguments", [])[:3]]
        == [1, 2, 3]
        and variant.get("arguments", [])[-1]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        },
        "Sub runtime ABI or tensor UID order is invalid",
    )

    alpha_document = request(identity, "sub")
    alpha_document["graph"]["nodes"][0]["attributes"]["alpha"] = -2.0
    alpha_path = temporary / "sub-alpha-negative.json"
    alpha_output = temporary / "sub-alpha-negative-artifact"
    write_json(alpha_path, alpha_document)
    alpha_result = provider.compile_request(
        alpha_path, alpha_output, "libtriton_jit"
    )
    alpha_variant = load_artifact(alpha_output)["program"]["stages"][0][
        "variants"
    ][0]
    require(
        alpha_result.get("status") == "success"
        and alpha_variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,17,-2.0,256",
        "Sub alpha specialization is invalid",
    )

    for name, attribute, value, detail in (
        ("sub-bad-mode", "mode", 1, "mode"),
        ("sub-bad-lowered-mode", "pointwise_mode", 1, "pointwise_mode"),
    ):
        mutated = copy.deepcopy(document)
        mutated["graph"]["nodes"][0]["attributes"][attribute] = value
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Sub mutation {name} wrote an artifact directory",
        )

    autotune_request_path = temporary / "sub-autotune.json"
    autotune_output = temporary / "sub-autotune-artifact"
    write_json(autotune_request_path, request(identity, "sub", autotune=True))
    autotune_result = provider.compile_request(
        autotune_request_path, autotune_output, "libtriton_jit"
    )
    require(
        autotune_result.get("status") == "success",
        "autotuned Sub did not compile",
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[-3] == "17"
            for candidate in autotune_variants
        ),
        "autotuned Sub candidates are incomplete or use the wrong mode",
    )
    require(
        autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Sub candidate identity is missing",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_min_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "min.json"
    output = temporary / "min-artifact"
    document = request(identity, "min")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Min unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path),
        "Min manifest request hash does not match exact Graph IR bytes",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Min must emit exactly one execution stage")
    stage = stages[0]
    require(
        stage.get("operation") == "min" and stage.get("tuning") is None,
        "Min stage operation or tuning state is invalid",
    )
    kernel = stage.get("kernel", {})
    require(
        kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "binary_contiguous_kernel",
        "Min did not select the common contiguous kernel",
    )
    variants = stage.get("variants", [])
    require(len(variants) == 1, "Min must emit one non-autotuned variant")
    variant = variants[0]
    require(
        variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,20,1.0,256",
        "Min variant did not specialize FLAGDNN_POINTWISE_MIN (20)",
    )
    require(
        [argument.get("uid") for argument in variant.get("arguments", [])[:3]]
        == [1, 2, 3]
        and variant.get("arguments", [])[-1]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        },
        "Min runtime ABI or tensor UID order is invalid",
    )

    for name, attribute, value, detail in (
        ("min-bad-mode", "mode", 1, "mode"),
        ("min-bad-lowered-mode", "pointwise_mode", 1, "pointwise_mode"),
    ):
        mutated = copy.deepcopy(document)
        mutated["graph"]["nodes"][0]["attributes"][attribute] = value
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Min mutation {name} wrote an artifact directory",
        )

    autotune_request_path = temporary / "min-autotune.json"
    autotune_output = temporary / "min-autotune-artifact"
    write_json(autotune_request_path, request(identity, "min", autotune=True))
    autotune_result = provider.compile_request(
        autotune_request_path, autotune_output, "libtriton_jit"
    )
    require(
        autotune_result.get("status") == "success",
        "autotuned Min did not compile",
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[-3] == "20"
            for candidate in autotune_variants
        ),
        "autotuned Min candidates are incomplete or use the wrong mode",
    )
    require(
        autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Min candidate identity is missing",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def assert_max_artifact(
    provider: object,
    temporary: Path,
    identity: str,
    *,
    compile_kernel: bool,
) -> None:
    request_path = temporary / "max.json"
    output = temporary / "max-artifact"
    document = request(identity, "max")
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(
        result.get("status") == "success",
        f"Max unexpectedly returned {result!r}",
    )

    manifest = load_artifact(output)
    require(
        manifest.get("request_sha256") == sha256_file(request_path),
        "Max manifest request hash does not match exact Graph IR bytes",
    )
    stages = manifest.get("program", {}).get("stages", [])
    require(len(stages) == 1, "Max must emit exactly one execution stage")
    stage = stages[0]
    require(
        stage.get("operation") == "max" and stage.get("tuning") is None,
        "Max stage operation or tuning state is invalid",
    )
    kernel = stage.get("kernel", {})
    require(
        kernel.get("provider") == "common_triton"
        and kernel.get("ownership") == "common"
        and kernel.get("function") == "binary_contiguous_kernel",
        "Max did not select the common contiguous kernel",
    )
    variants = stage.get("variants", [])
    require(len(variants) == 1, "Max must emit one non-autotuned variant")
    variant = variants[0]
    require(
        variant.get("full_signature")
        == "*fp32:16,*fp32:16,*fp32:16,i32,21,1.0,256",
        "Max variant did not specialize FLAGDNN_POINTWISE_MAX (21)",
    )
    require(
        [argument.get("uid") for argument in variant.get("arguments", [])[:3]]
        == [1, 2, 3]
        and variant.get("arguments", [])[-1]
        == {
            "kind": "scalar_i32",
            "name": "n_elements",
            "value": 16,
        },
        "Max runtime ABI or tensor UID order is invalid",
    )

    for name, attribute, value, detail in (
        ("max-bad-mode", "mode", 1, "mode"),
        ("max-bad-lowered-mode", "pointwise_mode", 1, "pointwise_mode"),
    ):
        mutated = copy.deepcopy(document)
        mutated["graph"]["nodes"][0]["attributes"][attribute] = value
        path = temporary / f"{name}.json"
        mutated_output = temporary / f"{name}-output"
        write_json(path, mutated)
        expect_value_error(
            lambda path=path, mutated_output=mutated_output: provider.compile_request(
                path, mutated_output, "libtriton_jit"
            ),
            detail,
        )
        require(
            not mutated_output.exists(),
            f"rejected Max mutation {name} wrote an artifact directory",
        )

    autotune_request_path = temporary / "max-autotune.json"
    autotune_output = temporary / "max-autotune-artifact"
    write_json(autotune_request_path, request(identity, "max", autotune=True))
    autotune_result = provider.compile_request(
        autotune_request_path, autotune_output, "libtriton_jit"
    )
    require(
        autotune_result.get("status") == "success",
        "autotuned Max did not compile",
    )
    autotune_stage = load_artifact(autotune_output)["program"]["stages"][0]
    autotune_variants = autotune_stage.get("variants", [])
    require(
        [candidate.get("variant_id") for candidate in autotune_variants]
        == ["block128_w4_s1", "block256_w4_s1"]
        and all(
            candidate.get("full_signature", "").split(",")[-3] == "21"
            for candidate in autotune_variants
        ),
        "autotuned Max candidates are incomplete or use the wrong mode",
    )
    require(
        autotune_stage.get("tuning", {}).get("candidate_identity"),
        "autotuned Max candidate identity is missing",
    )

    if compile_kernel:
        compile_add_kernel(output, variant, kernel)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def assert_add_autotune_artifact(
    provider: object,
    temporary: Path,
    identity: str,
) -> None:
    request_path = temporary / "add-autotune.json"
    output = temporary / "add-autotune-artifact"
    document = request(identity, autotune=True)
    write_json(request_path, document)
    result = provider.compile_request(request_path, output, "libtriton_jit")
    require(result.get("status") == "success", "autotuned Add did not compile")

    manifest = load_artifact(output)
    stage = manifest["program"]["stages"][0]
    variants = stage.get("variants", [])
    expected_configurations = [
        {
            "META": {"BLOCK_SIZE": 128},
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
        {
            "META": {"BLOCK_SIZE": 256},
            "maxnreg": None,
            "num_stages": 1,
            "num_warps": 4,
            "ppu_compiler_options": {},
        },
    ]
    require(
        [variant.get("variant_id") for variant in variants]
        == ["block128_w4_s1", "block256_w4_s1"],
        "autotuned Add must emit the two ordered legal PPU candidates",
    )
    for variant, configuration in zip(variants, expected_configurations):
        block_size = configuration["META"]["BLOCK_SIZE"]
        require(
            variant.get("full_signature")
            == f"*fp32:16,*fp32:16,*fp32:16,i32,1,1.0,{block_size}"
            and variant.get("compile_options")
            == {
                "maxnreg": configuration["maxnreg"],
                "num_stages": configuration["num_stages"],
                "num_warps": configuration["num_warps"],
                "ppu_compiler_options": configuration["ppu_compiler_options"],
            }
            and variant.get("launch")
            == {
                "grid": [(16 + block_size - 1) // block_size, 1, 1],
                "block": [128, 1, 1],
                "shared_memory": 0,
            },
            "autotuned Add candidate specialization is incomplete",
        )

    source_sha256 = stage["kernel"]["materialized_source"]["sha256"]
    tuning_path = SOURCE_ROOT / "backends/thead/tuning/common.yaml"
    identity_payload = {
        "schema_version": 1,
        "backend": "thead",
        "target": TARGET,
        "engine": "libtriton_jit",
        "compiler_identity": identity,
        "kernel": {
            "provider": "common_triton",
            "ownership": "common",
            "operation": "add",
            "function": "binary_contiguous_kernel",
            "registry_sha256": registry_sha256(),
            "source_sha256": source_sha256,
            "specialization": {"alpha": 1.0},
        },
        "tuning": {
            "source_sha256": sha256_file(tuning_path),
            "table": "binary",
            "key": "n_elements",
            "key_value": 16,
            "strategy": "align32",
            "warmup": 5,
            "repetitions": 10,
            "configurations": expected_configurations,
        },
    }
    expected_identity = canonical_sha256(identity_payload)
    require(
        stage.get("tuning")
        == {
            "warmup": 5,
            "repetitions": 10,
            "candidate_identity": expected_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        },
        "autotuned Add winner-cache identity is incomplete",
    )


def selected_jit_paths() -> dict[str, Path]:
    # Use the backend's coherent build/install layout, including lib64 and
    # the bundled JIT shipped with an installed FlagDNN SDK.
    from flagdnn_codegen import provider_loader

    provider = provider_loader.get_provider("thead")
    identity = importlib.import_module(
        provider.__package__ + ".compiler_identity"
    )
    return identity._jit_paths()


def selected_codegen_backend() -> str:
    import triton
    from flagdnn_codegen import provider_loader

    provider = provider_loader.get_provider("thead")
    compat = importlib.import_module(provider.__package__ + ".triton_compat")
    return compat.ppu_codegen_backend(
        Path(triton.__file__).resolve().parent.parent
    )


def compile_add_kernel(
    artifact_directory: Path,
    variant: dict[str, Any],
    kernel: dict[str, Any],
) -> None:
    jit = selected_jit_paths()
    sdk_root = Path(
        os.environ.get("FLAGDNN_THEAD_PPU_SDK_ROOT", "/usr/local/PPU_SDK")
    ).resolve()
    source = artifact_directory / kernel["materialized_source"]["path"]
    options = variant["compile_options"]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRITON_JIT_BACKEND": "CUDA",
            "TRITON_OVERRIDE_ARCH": "sm80",
            "CUDA_PATH": str(sdk_root / "CUDA_SDK"),
            "TRITON_PTXAS_PATH": str(sdk_root / "CUDA_SDK/bin/ptxas"),
            "TRITON_IR_FORMATTER_PATH": (
                str(sdk_root / "bin/llvm-irformatter")
            ),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(jit["standalone_compile"]),
            str(source),
            "--kernel-name",
            kernel["function"],
            "--device-id",
            "0",
            "--num-warps",
            str(options["num_warps"]),
            "--num-stages",
            str(options["num_stages"]),
            "--signature",
            variant["full_signature"],
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    require(
        completed.returncode == 0,
        "Triton kernel compilation failed\nstdout:\n"
        + completed.stdout
        + "\nstderr:\n"
        + completed.stderr,
    )
    lines = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    require(lines, "standalone compiler returned no cache directory")
    cache = Path(lines[-1]).resolve()
    require(cache.is_dir(), "standalone compiler cache directory is missing")
    metadata = tuple(cache.glob("*.json"))
    extension = "hgbin" if selected_codegen_backend() == "ppu" else "cubin"
    cubins = tuple(cache.glob(f"*.{extension}"))
    require(metadata, "Triton cache lacks kernel metadata")
    require(len(cubins) == 1, "Triton cache does not contain one cubin")
    require(
        cubins[0].is_file()
        and not cubins[0].is_symlink()
        and cubins[0].stat().st_size > 0,
        "Triton CUDA backend did not emit a concrete cubin",
    )
    kernel_metadata_path = cache / f"{kernel['function']}.json"
    require(
        kernel_metadata_path.is_file(),
        "Triton cache lacks exact kernel metadata",
    )
    kernel_metadata = json.loads(
        kernel_metadata_path.read_text(encoding="utf-8")
    )
    require(
        kernel_metadata.get("shared")
        == variant.get("launch", {}).get("shared_memory"),
        "Triton shared-memory metadata differs from the artifact: "
        f"kernel={kernel['function']} "
        f"actual={kernel_metadata.get('shared')} "
        f"expected={variant.get('launch', {}).get('shared_memory')}",
    )


def expect_value_error(function: Callable[[], object], detail: str) -> None:
    try:
        function()
    except ValueError as error:
        require(
            detail in str(error),
            f"error {error!s} did not contain expected detail {detail!r}",
        )
        return
    fail(f"expected ValueError containing {detail!r}")


def required_dependency_paths() -> set[Path]:
    import torch
    import triton
    import importlib

    backend_name = selected_codegen_backend()
    triton_package = Path(triton.__file__).resolve().parent
    jit = selected_jit_paths()
    sdk_root = Path(
        os.environ.get("FLAGDNN_THEAD_PPU_SDK_ROOT", "/usr/local/PPU_SDK")
    ).resolve()
    result = {
        (SOURCE_ROOT / "backends/thead/compiler.py").resolve(),
        (SOURCE_ROOT / "backends/thead/compiler_identity.py").resolve(),
        (
            SOURCE_ROOT / "backends/thead/python_environment_identity.py"
        ).resolve(),
        (
            SOURCE_ROOT / "compiler/flagdnn_codegen/kernel_registry.py"
        ).resolve(),
        (SOURCE_ROOT / "kernels/registry.json").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/registry.json").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/add_square.py").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/attention.py").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/fp8_attention.py").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/convolution.py").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/normalization.py").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/layout.py").resolve(),
        (SOURCE_ROOT / "backends/thead/kernels/pow.py").resolve(),
        (SOURCE_ROOT / "backends/thead/tuning/common.yaml").resolve(),
        Path(sys.executable).resolve(),
        Path(torch.__file__).resolve(),
        Path(torch._C.__file__).resolve(),
        (triton_package / "__init__.py").resolve(),
        (triton_package / f"backends/{backend_name}/compiler.py").resolve(),
        (triton_package / f"backends/{backend_name}/driver.py").resolve(),
        Path(
            importlib.import_module("triton.compiler.compiler").__file__
        ).resolve(),
        (triton_package / "_C/libtriton.so").resolve(),
        (
            triton_package
            / "backends"
            / backend_name
            / "lib"
            / (
                "libdevice.ppu.bc"
                if backend_name == "ppu"
                else "libdevice.10.bc"
            )
        ).resolve(),
        (sdk_root / "release.yaml").resolve(),
        (sdk_root / "bin/llvm-irformatter").resolve(),
        (sdk_root / "CUDA_SDK/bin/ptxas").resolve(),
        (sdk_root / "CUDA_SDK/bin/nvcc").resolve(),
        (sdk_root / "CUDA_SDK/include/cuda.h").resolve(),
        (sdk_root / "CUDA_SDK/lib64/libcuda.so.1").resolve(),
        (sdk_root / "CUDA_SDK/nvvm/libdevice/libdevice.10.bc").resolve(),
    }
    if backend_name == "ppu":
        result.add((sdk_root / "bin/ppu-llc").resolve())
    result.add((SOURCE_ROOT / "backends/thead/triton_compat.py").resolve())
    result.update(
        path.resolve()
        for path in (SOURCE_ROOT / "kernels/common").glob("*.py")
    )
    result.update(path.resolve() for path in jit.values())
    return result


def run_generic_identify(root: Path) -> tuple[str, dict[str, Any]]:
    identity_output = root / "identity.txt"
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FLAGDNN_BACKEND_ROOT"] = str(BACKEND_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "compiler/flagdnn_codegen/main.py"),
            "--identify",
            "--backend",
            "thead",
            "--target",
            TARGET,
            "--execution-engine",
            "libtriton_jit",
            "--identity-output",
            str(identity_output),
            "--quiet",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if completed.returncode != 0:
        fail(
            "generic identify failed\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    lines = identity_output.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 2, "generic identity output must have two lines")
    return lines[0], json.loads(lines[1])


def load_isolated_provider(path: Path, package_name: str) -> ModuleType:
    module_name = f"{package_name}.compiler"
    package = ModuleType(package_name)
    package.__file__ = str(path.parent)
    package.__package__ = package_name
    package.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    specification = importlib.util.spec_from_file_location(module_name, path)
    require(
        specification is not None and specification.loader is not None,
        "cannot load isolated installed THead provider",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        raise
    return module


def assert_source_tree_jit_discovery(temporary: Path) -> None:
    source_container = temporary / "relocated-source-tree"
    provider_root = source_container / "FlagDNN/backends/thead"
    provider_root.mkdir(parents=True)
    for name in (
        "compiler.py",
        "compiler_identity.py",
        "python_environment_identity.py",
        "triton_compat.py",
    ):
        shutil.copy2(
            SOURCE_ROOT / "backends/thead" / name, provider_root / name
        )
    expected_jit_root = source_container / "libtriton_jit"
    expected_jit_root.mkdir()

    previous_jit_root = os.environ.pop("FLAGDNN_THEAD_TRITON_JIT_ROOT", None)
    package_name = "_flagdnn_thead_relocated_source_contract"
    try:
        load_isolated_provider(provider_root / "compiler.py", package_name)
        identity_module = sys.modules[f"{package_name}.compiler_identity"]
        selected = identity_module._selected_jit_root()
        require(
            selected == expected_jit_root.resolve(),
            "source-tree THead provider did not discover the sibling "
            "libtriton_jit root",
        )
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        if previous_jit_root is not None:
            os.environ["FLAGDNN_THEAD_TRITON_JIT_ROOT"] = previous_jit_root


def assert_environment_identity_filter(provider: ModuleType) -> None:
    module = importlib.import_module(
        provider.__package__ + ".python_environment_identity"
    )
    names = (
        "FLAGDNN_CACHE_PATH",
        "FLAGDNN_ADD_CASE",
        "FLAGDNN_THEAD_ISOLATED_BACKWARD_FUNCTIONAL_CHILD",
        "FLAGDNN_THEAD_QUALIFY_PROBES",
        "FLAGDNN_THEAD_RESOURCE_ROOT",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["FLAGDNN_CACHE_PATH"] = "/tmp/identity-neutral-cache"
        os.environ["FLAGDNN_ADD_CASE"] = "identity_neutral_case"
        os.environ["FLAGDNN_THEAD_ISOLATED_BACKWARD_FUNCTIONAL_CHILD"] = "1"
        os.environ["FLAGDNN_THEAD_QUALIFY_PROBES"] = "1"
        os.environ["FLAGDNN_THEAD_RESOURCE_ROOT"] = str(SOURCE_ROOT)
        captured = module._cache_invalidating_environment()
        require(
            "FLAGDNN_CACHE_PATH" not in captured
            and "FLAGDNN_ADD_CASE" not in captured
            and "FLAGDNN_THEAD_ISOLATED_BACKWARD_FUNCTIONAL_CHILD"
            not in captured
            and "FLAGDNN_THEAD_QUALIFY_PROBES" not in captured,
            "THead compiler identity includes test/cache-only FlagDNN "
            "environment",
        )
        require(
            captured.get("FLAGDNN_THEAD_RESOURCE_ROOT") == str(SOURCE_ROOT),
            "THead compiler identity omitted a codegen-affecting backend "
            "environment value",
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def assert_installed_private_jit_discovery(temporary: Path) -> None:
    repository_container = temporary / "installed-parent-repository"
    repository_container.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(repository_container)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    unrelated_marker = repository_container / "unrelated.txt"
    unrelated_marker.write_text("initial\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository_container), "add", "unrelated.txt"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_container),
            "-c",
            "user.name=FlagDNN Contract",
            "-c",
            "user.email=contract@flagdnn.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial unrelated state",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    unrelated_git_metadata = (repository_container / ".git").resolve()
    sdk = repository_container / "installed-sdk"
    resource_root = sdk / "share/flagdnn"
    provider_root = resource_root / "backends/thead"
    provider_root.mkdir(parents=True)
    for name in (
        "compiler.py",
        "compiler_identity.py",
        "python_environment_identity.py",
        "triton_compat.py",
    ):
        shutil.copy2(
            SOURCE_ROOT / "backends/thead" / name, provider_root / name
        )
    shutil.copytree(
        SOURCE_ROOT / "backends/thead/kernels", provider_root / "kernels"
    )
    shutil.copytree(
        SOURCE_ROOT / "backends/thead/tuning", provider_root / "tuning"
    )
    (resource_root / "kernels").mkdir()
    shutil.copy2(
        SOURCE_ROOT / "kernels/registry.json",
        resource_root / "kernels/registry.json",
    )
    shutil.copytree(
        SOURCE_ROOT / "kernels/common", resource_root / "kernels/common"
    )

    configured_jit = selected_jit_paths()
    private_library = sdk / "lib/flagdnn/thead/libtriton_jit.so"
    private_scripts = sdk / "lib/flagdnn/share/triton_jit/scripts"
    private_library.parent.mkdir(parents=True)
    private_scripts.mkdir(parents=True)
    shutil.copy2(configured_jit["library"], private_library)
    for name in ("standalone_compile.py", "gen_ssig.py"):
        shutil.copy2(configured_jit[Path(name).stem], private_scripts / name)

    environment_path = (
        provider_root / "flagdnn_thead_compiler_environment.json"
    )
    write_json(
        environment_path,
        {
            "schema_version": 1,
            "backend": "CUDA",
            "libtriton_jit_soname": "libtriton_jit.so",
            "libtriton_jit_install_relative_path": (
                "lib/flagdnn/thead/libtriton_jit.so"
            ),
            "jit_script_install_relative_path": (
                "lib/flagdnn/share/triton_jit/scripts"
            ),
            "libtriton_jit_sha256": sha256_file(private_library),
            "triton_jit_provenance_sha256": "1" * 64,
            "standalone_compile_sha256": sha256_file(
                private_scripts / "standalone_compile.py"
            ),
            "gen_ssig_sha256": sha256_file(private_scripts / "gen_ssig.py"),
        },
    )

    previous_jit_root = os.environ.pop("FLAGDNN_THEAD_TRITON_JIT_ROOT", None)
    previous_resource_root = os.environ.get("FLAGDNN_THEAD_RESOURCE_ROOT")
    os.environ["FLAGDNN_THEAD_RESOURCE_ROOT"] = str(resource_root)
    package_name = "_flagdnn_thead_installed_contract"
    try:
        installed_provider = load_isolated_provider(
            provider_root / "compiler.py", package_name
        )
        dependencies = {
            Path(path).resolve()
            for path in installed_provider.compiler_identity_dependencies(
                TARGET, "libtriton_jit"
            )
        }
        required = {
            private_library.resolve(),
            (private_scripts / "standalone_compile.py").resolve(),
            (private_scripts / "gen_ssig.py").resolve(),
            environment_path.resolve(),
        }
        require(
            dependencies.issuperset(required),
            "installed THead provider omitted private JIT dependencies",
        )
        require(
            not dependencies.intersection(configured_jit.values()),
            "installed THead provider leaked the build-tree libtriton_jit root",
        )
        require(
            not any(
                path.is_relative_to(unrelated_git_metadata)
                for path in dependencies
            ),
            "installed THead provider inherited unrelated parent Git metadata",
        )
        identity = installed_provider.compiler_identity(
            TARGET, "libtriton_jit"
        )
        require(
            isinstance(identity.get("identity_sha256"), str),
            "installed THead provider did not produce an identity",
        )
        unrelated_marker.write_text("changed\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repository_container), "add", "unrelated.txt"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repository_container),
                "-c",
                "user.name=FlagDNN Contract",
                "-c",
                "user.email=contract@flagdnn.invalid",
                "commit",
                "--quiet",
                "-m",
                "changed unrelated state",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(
            installed_provider.compiler_identity(TARGET, "libtriton_jit")
            == identity,
            "installed THead identity inherited unrelated parent Git state",
        )
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        if previous_jit_root is not None:
            os.environ["FLAGDNN_THEAD_TRITON_JIT_ROOT"] = previous_jit_root
        if previous_resource_root is None:
            os.environ.pop("FLAGDNN_THEAD_RESOURCE_ROOT", None)
        else:
            os.environ["FLAGDNN_THEAD_RESOURCE_ROOT"] = previous_resource_root


def assert_normalization_planning(provider: Any) -> None:
    from flagdnn_codegen import kernel_registry

    for operation in ("layernorm", "rmsnorm"):
        candidate = kernel_registry.select_kernel_candidate("thead", operation)
        provider._validate_normalization_candidate(candidate, operation)
        source = kernel_registry.resolve_kernel_source(
            Path(provider.__file__), candidate
        )
        functions = {
            node.name: node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        for dtype in ("float32", "float16", "bfloat16"):
            graph = normalization_request("0" * 64, operation)["graph"]
            for tensor in graph["tensors"]:
                if tensor["uid"] <= 4:
                    tensor["data_type"] = dtype
            plan = provider._validate_normalization_graph(graph, operation)
            variant = provider._normalization_variant(
                plan,
                {
                    "META": {"BLOCK_SIZE": 256, "ROWS_PER_PROGRAM": 1},
                    "num_warps": 4,
                    "num_stages": 1,
                    "maxnreg": None,
                    "ppu_compiler_options": {},
                },
                "default",
            )
            function = functions[plan["function"]]
            signature = variant["full_signature"].split(",")
            require(
                len(signature) == len(function.args.args),
                "Normalization JIT signature does not cover the kernel arguments",
            )
            defaults = [
                str(ast.literal_eval(node)) for node in function.args.defaults
            ]
            require(
                signature[-len(defaults) :] == defaults,
                "Normalization JIT defaults differ from the common kernel",
            )
            require(
                variant["argument_count"]
                == (7 if operation == "layernorm" else 6),
                "Normalization runtime ABI changed",
            )


def assert_sigmoid_backward_planning(provider: Any) -> None:
    from flagdnn_codegen import kernel_registry

    candidate = kernel_registry.select_kernel_candidate(
        "thead", "sigmoid_backward"
    )
    provider._validate_binary_pointwise_candidate(
        candidate, "sigmoid_backward"
    )
    source = kernel_registry.resolve_kernel_source(
        Path(provider.__file__), candidate
    )
    functions = {
        node.name: node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    configuration = {
        "META": {"BLOCK_SIZE": 256},
        "num_warps": 4,
        "num_stages": 1,
        "maxnreg": None,
        "ppu_compiler_options": {},
    }
    for dtype in ("float32", "float16", "bfloat16"):
        for strided in (False, True):
            graph = request("0" * 64, "sigmoid_backward")["graph"]
            for tensor in graph["tensors"]:
                tensor["data_type"] = dtype
                if strided:
                    tensor["strides"] = [
                        stride * 2 for stride in tensor["strides"]
                    ]
            plan = provider._validate_binary_pointwise_graph(
                graph, "sigmoid_backward"
            )
            require(
                plan["function"] in candidate.functions,
                "SigmoidBackward selected an unregistered kernel",
            )
            variant = provider._binary_pointwise_variant(
                graph["tensors"],
                plan["n_elements"],
                40,
                1.0,
                plan["stride_constants"],
                configuration,
                "default",
            )
            signature = variant["full_signature"].split(",")
            function = functions[plan["function"]]
            require(
                len(signature) == len(function.args.args),
                "SigmoidBackward JIT signature does not cover the kernel arguments",
            )
            tail = ["0.0", "0.0", "0.0", "False", "1.0", "1.0", "1.0"]
            if not strided:
                tail.append("True")
            require(
                signature[-len(tail) :] == tail
                and variant["argument_count"] == 4,
                "SigmoidBackward constexpr defaults or runtime ABI changed",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "host_planning",
            "add",
            "sub",
            "mul",
            "min",
            "max",
            "scale",
            "relu",
            "sigmoid",
            "tanh",
            "elu",
            "identity",
            "gelu",
            "leaky_relu",
            "sqrt",
            "neg",
            "abs",
            "ceil",
            "floor",
            "exp",
            "log",
            "cos",
            "rsqrt",
            "sin",
            "tan",
            "softplus",
            "swish",
            "gelu_approx_tanh",
            "div",
            "pow",
            "mod",
            "sigmoid_backward",
            "reciprocal",
            "add_square",
            "cmp_eq",
            "cmp_neq",
            "cmp_gt",
            "cmp_ge",
            "cmp_lt",
            "cmp_le",
            "reshape",
            "transpose",
            "slice",
            "reduction",
            "batchnorm",
            "normalization",
            "matmul",
            "convolution",
            "conv_bias_relu",
            "typed_pointwise",
            "boolean_pointwise",
            "attention",
            "fp8_attention",
            "erf",
            "strided_pointwise",
            "performance",
        ),
        default="all",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the generated pointwise source through PPU-aware Triton",
    )
    arguments = parser.parse_args()
    caches_before = python_cache_entries()
    sys.path.insert(0, str(COMPILER_ROOT))
    os.environ["FLAGDNN_BACKEND_ROOT"] = str(BACKEND_ROOT)
    sdk = os.environ.get("FLAGDNN_THEAD_PPU_SDK_ROOT", "/usr/local/PPU_SDK")
    os.environ.setdefault("PPU_SDK", sdk)
    os.environ.setdefault("PPU_HOME", sdk)

    from flagdnn_codegen import provider_loader

    provider = provider_loader.get_provider("thead")
    assert_sigmoid_backward_planning(provider)
    assert_normalization_planning(provider)
    if arguments.case == "host_planning":
        from unittest.mock import patch

        # SDK discovery is unavailable on host-only CI. Exercise real graph
        # validation, registries, tuning and artifact I/O with a fixed identity.
        identity = {
            "provider": provider.PROVIDER_NAME,
            "provider_version": provider.PROVIDER_VERSION,
            "identity_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory(
            prefix="flagdnn-thead-host-"
        ) as temporary:
            with patch.object(
                provider, "compiler_identity", return_value=identity
            ):
                assert_generic_binary_artifact(
                    provider,
                    Path(temporary),
                    "0" * 64,
                    operation="sigmoid_backward",
                    label="SigmoidBackward",
                    mode=40,
                    wrong_mode=23,
                    compile_kernel=False,
                )
                assert_normalization_artifacts(
                    provider, Path(temporary), "0" * 64, compile_kernel=False
                )
        print(
            "THead host plans/artifacts: PASS (6 SigmoidBackward and 6 normalization combinations)"
        )
        return 0
    assert_environment_identity_filter(provider)
    for function_name in (
        "compiler_identity_dependencies",
        "compiler_identity",
        "compile_request",
    ):
        require(
            callable(getattr(provider, function_name, None)),
            f"THead provider does not implement {function_name}()",
        )

    expect_value_error(
        lambda: provider.compiler_identity("sm_80", "libtriton_jit"),
        "target",
    )
    expect_value_error(
        lambda: provider.compiler_identity(TARGET, "external_artifact"),
        "libtriton_jit",
    )

    sdk_root = Path(
        os.environ.get("FLAGDNN_THEAD_PPU_SDK_ROOT", "/usr/local/PPU_SDK")
    ).resolve()
    effective_jit_environment = {
        "PPU_SDK": str(sdk_root),
        "PPU_HOME": str(sdk_root),
        "CUDA_PATH": str(sdk_root / "CUDA_SDK"),
        "TRITON_PTXAS_PATH": str(sdk_root / "CUDA_SDK/bin/ptxas"),
        "TRITON_IR_FORMATTER_PATH": str(sdk_root / "bin/llvm-irformatter"),
        "TRITON_JIT_BACKEND": "CUDA",
        "TRITON_OVERRIDE_ARCH": "sm80",
    }
    if selected_codegen_backend() == "ppu":
        effective_jit_environment["TRITON_PPU_LLC_PATH"] = str(
            sdk_root / "bin/ppu-llc"
        )
        previous_llc = os.environ.get("TRITON_PPU_LLC_PATH")
        try:
            os.environ["TRITON_PPU_LLC_PATH"] = "/usr/bin/true"
            expect_value_error(
                lambda: provider.compiler_identity(TARGET, "libtriton_jit"),
                "TRITON_PPU_LLC_PATH",
            )
        finally:
            if previous_llc is None:
                os.environ.pop("TRITON_PPU_LLC_PATH", None)
            else:
                os.environ["TRITON_PPU_LLC_PATH"] = previous_llc
    saved_jit_environment = {
        name: os.environ.pop(name, None) for name in effective_jit_environment
    }
    try:
        first_identity = provider.compiler_identity(TARGET, "libtriton_jit")
        os.environ.update(effective_jit_environment)
        second_identity = provider.compiler_identity(TARGET, "libtriton_jit")
    finally:
        for name, value in saved_jit_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    require(
        first_identity == second_identity,
        "compiler identity is not deterministic",
    )
    require(
        first_identity.get("provider") == "thead_triton"
        and first_identity.get("provider_version") == "1",
        "compiler identity provider metadata is invalid",
    )
    digest = first_identity.get("identity_sha256")
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "compiler identity is not a lowercase SHA-256",
    )

    dependencies = tuple(
        Path(path).resolve()
        for path in provider.compiler_identity_dependencies(
            TARGET, "libtriton_jit"
        )
    )
    require(
        dependencies == tuple(sorted(set(dependencies))),
        "compiler dependencies are not unique and sorted",
    )
    require(
        all(path.is_file() for path in dependencies),
        "compiler dependency closure contains a non-file",
    )
    missing = required_dependency_paths().difference(dependencies)
    require(
        not missing,
        "compiler dependency closure is missing: "
        + ", ".join(str(path) for path in sorted(missing)),
    )
    require(
        not any(
            token in path.name.lower()
            for path in dependencies
            for token in ("cublas", "acblas", "openblas", "mkl")
        ),
        "THead compiler identity directly depends on a BLAS library",
    )

    with tempfile.TemporaryDirectory(prefix="flagdnn-thead-compiler-") as tmp:
        temporary = Path(tmp)
        if arguments.case in ("all", "add"):
            assert_add_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
            assert_add_autotune_artifact(provider, temporary, digest)
        if arguments.case in ("all", "mul"):
            assert_mul_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "sub"):
            assert_sub_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "min"):
            assert_min_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "max"):
            assert_max_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "scale"):
            assert_scale_alias_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "relu"):
            assert_relu_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "sigmoid"):
            assert_sigmoid_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "tanh"):
            assert_tanh_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "elu"):
            assert_elu_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "identity"):
            assert_identity_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "gelu"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="gelu",
                label="Gelu",
                mode=36,
                wrong_mode=35,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "leaky_relu"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="leaky_relu",
                label="LeakyRelu",
                mode=2,
                wrong_mode=36,
                stage_operation="relu",
                expected_constants=[
                    "2",
                    "0.20000000298023224",
                    "0.0",
                    "0.0",
                    "0",
                    "1.0",
                    "1.0",
                    "1.0",
                    "1",
                ],
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "sqrt"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="sqrt",
                label="Sqrt",
                mode=3,
                wrong_mode=8,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "neg"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="neg",
                label="Neg",
                mode=8,
                wrong_mode=3,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "abs"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="abs",
                label="Abs",
                mode=9,
                wrong_mode=8,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "ceil"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="ceil",
                label="Ceil",
                mode=10,
                wrong_mode=9,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "floor"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="floor",
                label="Floor",
                mode=12,
                wrong_mode=10,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "exp"):
            assert_generic_unary_artifact(
                provider,
                temporary,
                digest,
                operation="exp",
                label="Exp",
                mode=6,
                wrong_mode=12,
                compile_kernel=arguments.compile,
            )
        unary_descriptor_cases = (
            ("erf", "Erf", 4, 5, None),
            ("log", "Log", 7, 6, None),
            ("cos", "Cos", 11, 7, None),
            ("rsqrt", "Rsqrt", 13, 11, None),
            ("sin", "Sin", 14, 13, None),
            ("tan", "Tan", 15, 14, None),
            ("softplus", "Softplus", 37, 15, None),
            (
                "swish",
                "Swish",
                38,
                37,
                ["38", "0.0", "0.0", "0.0", "0", "1.25", "1.0", "1.0", "1"],
            ),
            ("gelu_approx_tanh", "GeluApproxTanh", 39, 38, None),
            ("reciprocal", "Reciprocal", 16, 15, None),
        )
        for (
            operation,
            label,
            mode,
            wrong_mode,
            expected_constants,
        ) in unary_descriptor_cases:
            if arguments.case in ("all", operation):
                assert_generic_unary_artifact(
                    provider,
                    temporary,
                    digest,
                    operation=operation,
                    label=label,
                    mode=mode,
                    wrong_mode=wrong_mode,
                    expected_constants=expected_constants,
                    compile_kernel=arguments.compile,
                )
        binary_descriptor_cases = (
            ("div", "Div", 19, 18),
            ("pow", "Pow", 23, 19),
            ("mod", "Mod", 22, 23),
            ("sigmoid_backward", "SigmoidBackward", 40, 23),
        )
        for operation, label, mode, wrong_mode in binary_descriptor_cases:
            if arguments.case in ("all", operation):
                assert_generic_binary_artifact(
                    provider,
                    temporary,
                    digest,
                    operation=operation,
                    label=label,
                    mode=mode,
                    wrong_mode=wrong_mode,
                    compile_kernel=arguments.compile,
                )
        comparison_descriptor_cases = (
            ("cmp_eq", "CmpEq", 25, 26),
            ("cmp_neq", "CmpNeq", 26, 25),
            ("cmp_gt", "CmpGt", 27, 28),
            ("cmp_ge", "CmpGe", 28, 27),
            ("cmp_lt", "CmpLt", 29, 30),
            ("cmp_le", "CmpLe", 30, 29),
        )
        for operation, label, mode, wrong_mode in comparison_descriptor_cases:
            if arguments.case in ("all", operation):
                assert_generic_binary_artifact(
                    provider,
                    temporary,
                    digest,
                    operation=operation,
                    label=label,
                    mode=mode,
                    wrong_mode=wrong_mode,
                    pointer_signature="*fp32:16,*fp32:16,*i8:16",
                    bad_dtype_detail="comparison inputs",
                    compile_kernel=arguments.compile,
                )
        if arguments.case in ("all", "attention", "fp8_attention"):
            assert_attention_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
                fp8_only=arguments.case == "fp8_attention",
            )
        if arguments.case in ("all", "boolean_pointwise"):
            assert_boolean_pointwise_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "typed_pointwise"):
            assert_typed_pointwise_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "strided_pointwise"):
            assert_strided_pointwise_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "add_square"):
            assert_add_square_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        for operation in ("reshape", "transpose", "slice"):
            if arguments.case in ("all", operation):
                assert_layout_artifact(
                    provider,
                    temporary,
                    digest,
                    operation=operation,
                    compile_kernel=arguments.compile,
                )
        if arguments.case in ("all", "reduction"):
            assert_reduction_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "batchnorm"):
            assert_batchnorm_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "normalization"):
            assert_normalization_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "performance"):
            assert_performance_artifacts(
                provider, temporary, digest, compile_kernel=arguments.compile
            )
        if arguments.case in ("all", "matmul"):
            assert_matmul_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "convolution"):
            if arguments.compile:
                assert_convolution_indexing()
            assert_convolution_artifacts(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        if arguments.case in ("all", "conv_bias_relu"):
            assert_conv_bias_relu_artifact(
                provider,
                temporary,
                digest,
                compile_kernel=arguments.compile,
            )
        valid_path = temporary / "valid.json"
        output = temporary / "output"
        valid = request(digest, "unsupported")
        write_json(valid_path, valid)
        unsupported = provider.compile_request(
            valid_path, output, "libtriton_jit"
        )
        require(
            unsupported
            == {
                "backend": "thead",
                "execution_engine": "libtriton_jit",
                "operation_types": ["unsupported"],
                "reason_code": "operation_family_not_implemented",
                "schema_version": 1,
                "status": "unsupported",
                "target": TARGET,
            },
            "compiler skeleton did not return the typed unsupported result",
        )
        require(
            not output.exists(),
            "unsupported compiler skeleton unexpectedly wrote artifacts",
        )

        mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
            (
                "backend",
                lambda value: value.__setitem__("backend", "nvidia"),
                "another backend",
            ),
            (
                "target",
                lambda value: value.__setitem__("target", "sm_80"),
                "target",
            ),
            (
                "schema",
                lambda value: value.__setitem__("schema_version", 2),
                "schema_version",
            ),
            (
                "identity",
                lambda value: value.__setitem__("compiler_identity", "0" * 64),
                "identity",
            ),
            (
                "tensor-count",
                lambda value: value["graph"].__setitem__("tensor_count", 2),
                "tensor_count",
            ),
            (
                "tensor-dimension",
                lambda value: value["graph"]["tensors"][0].__setitem__(
                    "dimensions", [0]
                ),
                "dimensions",
            ),
            (
                "tensor-stride",
                lambda value: value["graph"]["tensors"][0].__setitem__(
                    "strides", []
                ),
                "strides",
            ),
            (
                "node-count",
                lambda value: value["graph"].__setitem__("node_count", 2),
                "node_count",
            ),
            (
                "node-type",
                lambda value: value["graph"]["nodes"][0].__setitem__(
                    "type", 1
                ),
                "node type",
            ),
            (
                "unknown-port",
                lambda value: value["graph"]["nodes"][0]["inputs"][
                    0
                ].__setitem__("uid", 99),
                "unknown tensor",
            ),
        ]
        for name, mutate, detail in mutations:
            document = copy.deepcopy(valid)
            mutate(document)
            path = temporary / f"{name}.json"
            write_json(path, document)
            expect_value_error(
                lambda path=path: provider.compile_request(
                    path, temporary / f"{name}-output", "libtriton_jit"
                ),
                detail,
            )
        expect_value_error(
            lambda: provider.compile_request(
                valid_path, temporary / "wrong-engine", "external_artifact"
            ),
            "libtriton_jit",
        )

        duplicate = temporary / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":3,"schema_version":3}', encoding="utf-8"
        )
        expect_value_error(
            lambda: provider.compile_request(
                duplicate, temporary / "duplicate-output", "libtriton_jit"
            ),
            "duplicate JSON key",
        )

        resource_root = temporary / "resource"
        (resource_root / "kernels").mkdir(parents=True)
        shutil.copy2(
            SOURCE_ROOT / "kernels/registry.json",
            resource_root / "kernels/registry.json",
        )
        shutil.copytree(
            SOURCE_ROOT / "kernels/common", resource_root / "kernels/common"
        )
        shutil.copytree(
            SOURCE_ROOT / "backends/thead/kernels",
            resource_root / "backends/thead/kernels",
        )
        shutil.copytree(
            SOURCE_ROOT / "backends/thead/tuning",
            resource_root / "backends/thead/tuning",
        )
        previous_resource = os.environ.get("FLAGDNN_THEAD_RESOURCE_ROOT")
        os.environ["FLAGDNN_THEAD_RESOURCE_ROOT"] = str(resource_root)
        try:
            copied_first = provider.compiler_identity(TARGET, "libtriton_jit")
            copied_kernel = resource_root / "kernels/common/binary.py"
            copied_kernel.write_bytes(
                copied_kernel.read_bytes() + b"\n# mutation\n"
            )
            copied_second = provider.compiler_identity(TARGET, "libtriton_jit")
            stale_request = temporary / "stale-identity.json"
            write_json(
                stale_request,
                request(str(copied_first["identity_sha256"])),
            )
            expect_value_error(
                lambda: provider.compile_request(
                    stale_request,
                    temporary / "stale-identity-output",
                    "libtriton_jit",
                ),
                "identity",
            )
            copied_tuning = resource_root / "backends/thead/tuning/common.yaml"
            copied_tuning.write_bytes(
                copied_tuning.read_bytes() + b"\n# mutation\n"
            )
            copied_third = provider.compiler_identity(TARGET, "libtriton_jit")
        finally:
            if previous_resource is None:
                os.environ.pop("FLAGDNN_THEAD_RESOURCE_ROOT", None)
            else:
                os.environ["FLAGDNN_THEAD_RESOURCE_ROOT"] = previous_resource
        require(
            copied_first["identity_sha256"]
            != copied_second["identity_sha256"],
            "compiler identity did not change with copied kernel bytes",
        )
        require(
            copied_second["identity_sha256"]
            != copied_third["identity_sha256"],
            "compiler identity did not change with copied tuning bytes",
        )

        assert_installed_private_jit_discovery(temporary)
        assert_source_tree_jit_discovery(temporary)

        generic_digest, metadata = run_generic_identify(temporary)
        require(
            generic_digest == digest,
            "generic identify and provider identity disagree",
        )
        require(
            metadata.get("dependencies_complete") is True,
            "generic identify marked dependency closure incomplete",
        )
        require(
            set(
                Path(path).resolve() for path in metadata.get("files", ())
            ).issuperset(dependencies),
            "generic identify omitted provider dependencies",
        )

    caches_after = python_cache_entries()
    require(
        caches_after == caches_before,
        "compiler contract created Python bytecode cache entries: "
        + ", ".join(
            str(path) for path in sorted(caches_after - caches_before)
        ),
    )
    print("THead compiler provider contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
