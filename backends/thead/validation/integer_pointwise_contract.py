#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for exact INT32 THead pointwise plans and artifacts."""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from compiler_contract import request

sys.dont_write_bytecode = True
SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT / "compiler"))
os.environ["FLAGDNN_BACKEND_ROOT"] = str(SOURCE_ROOT / "backends")



MODES = {
    "add": 1, "sub": 17, "mul": 18, "div": 19, "pow": 23,
    "min": 20, "max": 21, "mod": 22, "cmp_eq": 25,
}
CONFIGURATION = {
    "META": {"BLOCK_SIZE": 1024}, "num_warps": 4, "num_stages": 1,
    "maxnreg": None, "ppu_compiler_options": {},
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_rejected(operation, message):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError(message)


def document_for(operation, shapes=None, strides=None):
    document = request("0" * 64, operation)
    graph = document["graph"]
    for tensor in graph["tensors"]:
        if tensor["data_type"] != "boolean":
            tensor["data_type"] = "int32"
    if shapes is not None:
        for tensor, shape, stride in zip(
            graph["tensors"], shapes, strides, strict=True
        ):
            tensor["dimensions"] = shape
            tensor["strides"] = stride
        for node in graph["nodes"]:
            node["attributes"]["n_elements"] = math.prod(shapes[-1])
    return document


def verify_binary_plans(dispatch):
    # Each expected stride is specified independently of the planner.
    layouts = (
        ("dense", [[2, 3], [2, 3], [2, 3]], [[3, 1]] * 3, None),
        ("dense_permutation", [[2, 3]] * 3, [[1, 2]] * 3, None),
        ("scalar_right", [[2, 3], [1], [2, 3]],
         [[3, 1], [1], [3, 1]], [[3, 1], [0, 0], [3, 1]]),
        ("scalar_left", [[1], [2, 3], [2, 3]],
         [[1], [3, 1], [3, 1]], [[0, 0], [3, 1], [3, 1]]),
        ("right_aligned", [[2, 3], [3], [2, 3]],
         [[3, 1], [1], [3, 1]], [[3, 1], [0, 1], [3, 1]]),
        ("two_sided", [[2, 1], [1, 3], [2, 3]],
         [[1, 1], [3, 1], [3, 1]], [[1, 0], [0, 1], [3, 1]]),
        ("padded", [[2, 3]] * 3, [[9, 2], [8, 2], [10, 3]],
         [[9, 2], [8, 2], [10, 3]]),
    )
    for operation, mode in MODES.items():
        for label, shapes, strides, expected_strides in layouts:
            graph = document_for(operation, shapes, strides)["graph"]
            if operation in {"add", "sub"}:
                graph["nodes"][0]["attributes"]["alpha"] = 16777217
            plan = dispatch._validate_binary_pointwise_graph(graph, operation)
            require(
                plan["n_elements"] == 6,
                f"{operation}/{label} wrong output count",
            )
            require(
                type(plan["alpha"]) is int,
                f"{operation}/{label} alpha lost integer type",
            )
            expected_alpha = 16777217 if operation in {"add", "sub"} else 1
            require(plan["alpha"] == expected_alpha, "integer alpha changed")
            if expected_strides is None:
                require(
                    plan["stride_constants"] is None,
                    "dense layout became strided",
                )
                require(
                    plan["function"] == "binary_contiguous_kernel",
                    "wrong dense kernel",
                )
            else:
                expected = [1] * 6 + [2, 3]
                for stride in expected_strides:
                    expected += [0] * 6 + stride
                require(
                    plan["stride_constants"] == expected,
                    f"{operation}/{label} wrong broadcast addresses",
                )
                require(
                    plan["function"] == "binary_strided_kernel",
                    "wrong broadcast kernel",
                )
            variant = dispatch._binary_pointwise_variant(
                graph["tensors"], plan["n_elements"], mode, plan["alpha"],
                plan["stride_constants"], CONFIGURATION, "contract",
            )
            parts = variant["full_signature"].split(",")
            require(
                parts[:2] == ["*i32:16"] * 2,
                "integer pointers became floating",
            )
            require(
                parts[2] == ("*i8:16" if operation == "cmp_eq" else "*i32:16"),
                "wrong output pointer type",
            )
            require(
                parts[-3:] == [str(mode), str(expected_alpha), "1024"],
                "integer constexpr signature changed",
            )
            require(
                variant["launch"]["shared_memory"] == 0,
                "integer operation uses floating reduction resources",
            )

    graph = document_for(
        "add", [[2, 1, 17], [3, 1], [2, 3, 17]],
        [[17, 17, 1], [1, 1], [51, 17, 1]],
    )["graph"]
    plan = dispatch._validate_binary_pointwise_graph(graph, "add")
    require(plan["n_elements"] == 102, "multi-axis output count changed")
    require(
        plan["stride_constants"] == (
            [1] * 5 + [2, 3, 17]
            + [0] * 5 + [17, 0, 1]
            + [0] * 5 + [0, 1, 0]
            + [0] * 5 + [51, 17, 1]
        ),
        "multi-axis rank-mismatched broadcast addresses changed",
    )

    floating = request("0" * 64, "pow")["graph"]
    variant = dispatch._binary_pointwise_variant(
        floating["tensors"], 16, 23, 1.0, None, CONFIGURATION, "float",
    )
    require(
        variant["launch"]["shared_memory"] == 16,
        "floating POW resources changed",
    )


def verify_rejections(dispatch):
    base = document_for("add")["graph"]
    for alpha in (0.5, 2**31, -(2**31) - 1, math.nan, math.inf, True):
        graph = copy.deepcopy(base)
        graph["nodes"][0]["attributes"]["alpha"] = alpha
        expect_rejected(
            lambda: dispatch._validate_binary_pointwise_graph(graph, "add"),
            "invalid integer alpha accepted",
        )
    for alpha in (-(2**31), -16777217, 16777217, 2**31 - 1):
        graph = copy.deepcopy(base)
        graph["nodes"][0]["attributes"]["alpha"] = alpha
        plan = dispatch._validate_binary_pointwise_graph(graph, "add")
        require(
            plan["alpha"] == alpha and type(plan["alpha"]) is int,
            "alpha boundary changed",
        )
        variant = dispatch._binary_pointwise_variant(
            graph["tensors"], plan["n_elements"], 1, plan["alpha"],
            plan["stride_constants"], CONFIGURATION, "alpha_boundary",
        )
        require(
            variant["full_signature"].split(",")[-2] == str(alpha),
            "alpha boundary signature became floating",
        )
    for operation in ("cmp_gt", "logical_and", "sigmoid_backward"):
        graph = document_for(operation)["graph"]
        # Logical fixtures begin as Boolean, so explicitly test rejected INT32.
        for tensor in graph["tensors"][:2]:
            tensor["data_type"] = "int32"
        expect_rejected(
            lambda: dispatch._validate_binary_pointwise_graph(
                graph, operation
            ),
            "unrelated operation acquired INT32 support",
        )
    for mutate in (
        lambda g: g["tensors"][1].update(data_type="float32"),
        lambda g: g["tensors"][2].update(data_type="boolean"),
        lambda g: g["tensors"][0].update(strides=[0, 0, 0]),
        lambda g: g["nodes"][0].update(compute_data_type="boolean"),
    ):
        graph = copy.deepcopy(base)
        mutate(graph)
        expect_rejected(
            lambda: dispatch._validate_binary_pointwise_graph(graph, "add"),
            "invalid integer graph accepted",
        )

    graph = document_for(
        "add",
        [[2, 1], [3], [2, 3]],
        [[1, 1], [1], [3, 1]],
    )["graph"]
    graph["nodes"][0]["attributes"]["n_elements"] = 2
    expect_rejected(
        lambda: dispatch._validate_binary_pointwise_graph(graph, "add"),
        "input count accepted as output count",
    )
    graph["nodes"][0]["attributes"]["n_elements"] = 6
    graph["tensors"][2]["dimensions"] = [3, 2]
    expect_rejected(
        lambda: dispatch._validate_binary_pointwise_graph(graph, "add"),
        "invalid broadcast output accepted",
    )
    graph = document_for(
        "add",
        [[2, 2], [3], [2, 3]],
        [[2, 1], [1], [3, 1]],
    )["graph"]
    expect_rejected(
        lambda: dispatch._validate_binary_pointwise_graph(graph, "add"),
        "incompatible input broadcast accepted",
    )
    graph = document_for(
        "add",
        [[2, 1], [3], [2, 3]],
        [[1, 1], [1], [3, 1]],
    )["graph"]
    for tensor in graph["tensors"]:
        tensor["data_type"] = "float32"
    expect_rejected(
        lambda: dispatch._validate_binary_pointwise_graph(graph, "add"),
        "float broadcast scope changed",
    )


def verify_add_square(dispatch):
    for strides in ([1], [2]):
        graph = document_for("add_square")["graph"]
        for tensor in graph["tensors"]:
            tensor["strides"] = strides
        plan = dispatch._validate_add_square_graph(graph)
        require(plan["n_elements"] == 16, "add_square output count changed")
        argument_uids = [t["uid"] for t in plan["argument_tensors"]]
        require(
            argument_uids == [2, 1, 4],
            "add_square argument roles changed",
        )
        require(
            plan["virtual_tensor"]["data_type"] == "int32",
            "add_square virtual type changed",
        )
        require(
            (plan["stride_constants"] is None) == (strides == [1]),
            "add_square layout specialization changed",
        )
        variant = dispatch._add_square_variant(
            plan["argument_tensors"], 16, plan["stride_constants"],
            CONFIGURATION, "square",
        )
        require(
            variant["full_signature"].split(",")[:3] == ["*i32:16"] * 3,
            "add_square pointers became float",
        )


def verify_artifacts(provider):
    identity = {
        "provider": provider.PROVIDER_NAME,
        "provider_version": provider.PROVIDER_VERSION,
        "identity_sha256": "0" * 64,
    }
    artifact_codegen = importlib.import_module(
        provider.__package__ + ".codegen.artifacts",
    )
    # Stub SDK discovery and device launch-resource inspection only.
    # Validation, registry selection and manifest generation use real code.
    with tempfile.TemporaryDirectory(
        prefix="thead-int32-contract-"
    ) as temporary:
        root = Path(temporary)
        with (
            patch.object(provider, "compiler_identity", return_value=identity),
            patch.object(artifact_codegen, "populate_launch_resources"),
        ):
            for operation in (*MODES, "add_square"):
                if operation == "add_square":
                    document = document_for(operation)
                else:
                    document = document_for(
                        operation,
                        [[2, 1], [3], [2, 3]],
                        [[1, 1], [1], [3, 1]],
                    )
                    if operation in {"add", "sub"}:
                        node = document["graph"]["nodes"][0]
                        node["attributes"]["alpha"] = 16777217
                request_path = root / f"{operation}.json"
                request_path.write_text(json.dumps(document), encoding="utf-8")
                output = root / operation
                provider.compile_request(request_path, output, "libtriton_jit")
                manifest = json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8"),
                )
                stage = manifest["program"]["stages"][0]
                signature = stage["variants"][0]["full_signature"].split(",")
                require(
                    signature[:2] == ["*i32:16"] * 2,
                    "artifact lost integer pointers",
                )
                if operation in {"add", "sub"}:
                    require(
                        signature[-2] == "16777217",
                        "codegen rounded integer alpha",
                    )
                require(
                    stage["variants"][0]["launch"]["shared_memory"] == 0,
                    "integer artifact uses floating shared memory",
                )


def main() -> int:
    from flagdnn_codegen import provider_loader

    provider = provider_loader.get_provider("thead")
    dispatch = importlib.import_module(
        provider.__package__ + ".dispatch.pointwise",
    )
    verify_binary_plans(dispatch)
    verify_rejections(dispatch)
    verify_add_square(dispatch)
    verify_artifacts(provider)
    print(
        "PASS THead INT32 pointwise: 63 layouts, validation, "
        "alpha and 10 artifacts",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
