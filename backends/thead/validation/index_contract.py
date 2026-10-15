#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for zero-input index and one-stage raw concatenation."""

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "compiler"))
from flagdnn_codegen import provider_loader  # noqa: E402


def tensor(uid, dtype, shape, strides):
    return {
        "uid": uid,
        "data_type": dtype,
        "dimensions": shape,
        "strides": strides,
        "alignment": 16,
        "virtual": False,
    }


def graph(operation, tensors, axis):
    inputs = tensors[:-1]
    parameters = {"axis": axis, "n_elements": 1}
    for extent in tensors[-1]["dimensions"]:
        parameters["n_elements"] *= extent
    if operation == "concatenate":
        parameters["input_count"] = len(inputs)
    return {
        "name": operation,
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
                    {"name": f"input_{i}", "uid": item["uid"]}
                    for i, item in enumerate(inputs)
                ],
                "outputs": [{"name": "output", "uid": tensors[-1]["uid"]}],
                "attributes": parameters,
            }
        ],
    }


def require_rejected(action):
    try:
        action()
    except ValueError:
        return
    raise RuntimeError("malformed index graph was accepted")


def verify_repeated_inputs(parser, extended, artifacts, emit):
    document = graph(
        "concatenate",
        [
            tensor(1, "float32", [2], [1]),
            tensor(2, "float32", [3], [2]),
            tensor(3, "float32", [7], [2]),
        ],
        0,
    )
    node = document["nodes"][0]
    node["inputs"] = [
        {"name": f"input_{index}", "uid": uid}
        for index, uid in enumerate((1, 2, 1))
    ]
    node["attributes"]["input_count"] = 3
    assert parser(document) == ["concatenate"]
    with tempfile.TemporaryDirectory(prefix="flagdnn-concat-repeat-") as temp:
        output = Path(temp)
        with patch.object(artifacts, "populate_launch_resources"):
            emit._compile_extended(
                request={"graph": document, "flagdnn_version": "0.2.0"},
                request_bytes=b"repeated inputs",
                identity={},
                target="ppu_contract_cc80",
                output_directory=output,
                enable_autotune=False,
                operation="concatenate",
            )
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["tensor_count"] == len(manifest["tensors"]) == 3
        assert manifest["external_uids"] == [1, 2, 3]
        assert manifest["program"]["stage_count"] == 1
        stage = manifest["program"]["stages"][0]
        assert stage["source_node_ids"] == [0]
        assert stage["kernel"]["ownership"] == "platform"
        argument_uids = [a["uid"] for a in stage["variants"][0]["arguments"]]
        assert argument_uids == [1, 2, 3]
        source = ast.parse((output / "generated_stage_0.py").read_text())
        entry = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "concatenate_kernel"
        )
        assert [p.arg for p in entry.args.args] == [
            "input_0",
            "input_1",
            "output_ptr",
            "BLOCK_SIZE",
        ]
        calls = [
            n
            for n in ast.walk(entry)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_concatenate_region"
        ]
        assert [n.args[0].id for n in calls] == [
            "input_0",
            "input_1",
            "input_0",
        ]
        assert [(n.args[8].value, n.args[9].value) for n in calls] == [
            (0, 2),
            (2, 5),
            (5, 7),
        ]
    invalid = copy.deepcopy(document)
    invalid["nodes"][0]["outputs"][0]["uid"] = 1
    require_rejected(lambda: extended.kernel_configuration(invalid))


class _Address:
    def __init__(self, memory, offset=0):
        assert -(2**63) <= offset < 2**63, "kernel address exceeds int64"
        self.memory, self.offset = memory, offset

    def to(self, storage):
        return self

    def __add__(self, offset):
        return _Address(self.memory, self.offset + offset)


class _ScalarLanguage:
    """Execute one real kernel lane on the host, with strict memory checks."""

    constexpr = int
    int64 = "int64"

    @staticmethod
    def zeros(shape, dtype):
        assert shape == (1,) and dtype == "int64"
        return 0

    @staticmethod
    def full(shape, value, dtype):
        assert shape == (1,) and dtype == "int64"
        assert -(2**63) <= value < 2**63, "kernel constant exceeds int64"
        return value

    static_range = staticmethod(range)
    pointer_type = staticmethod(lambda storage: storage)
    where = staticmethod(
        lambda predicate, left, right: left if predicate else right
    )

    @staticmethod
    def load(address, mask, other):
        return address.memory[address.offset] if mask else other

    @staticmethod
    def store(address, value, mask):
        if mask:
            assert address.offset in address.memory, "output padding written"
            assert address.memory[address.offset] is None, "duplicate write"
            address.memory[address.offset] = value


def verify_kernel_addresses(parser, extended):
    from itertools import product
    import math

    path = ROOT / "backends/thead/kernels/index.py"
    tree = ast.parse(path.read_text())
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_concatenate_region"
    )
    helper.decorator_list = []
    namespace = {"tl": _ScalarLanguage}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    execute = namespace["_concatenate_region"]

    def strides(shape, reverse):
        result, span = [0] * len(shape), 2
        axes = range(len(shape)) if reverse else reversed(range(len(shape)))
        for axis in axes:
            result[axis] = span
            span = (shape[axis] - 1) * span + span + 1
        return result

    geometries = [([[3], [1]], [[1], [2**62]], [4], [1], 0)]
    for rank in range(1, 9):
        for axis in range(rank):
            shape = [2 if i % 2 else 1 for i in range(rank)]
            shapes = [list(shape) for _ in range(3)]
            for shape, extent in zip(shapes, (2, 3, 1)):
                shape[axis] = extent
            output = list(shapes[0])
            output[axis] = 6
            geometries.append(
                (
                    shapes,
                    [strides(s, i % 2) for i, s in enumerate(shapes)],
                    output,
                    strides(output, True),
                    axis,
                )
            )
    for geometry in geometries:
        shapes, source_strides, output_shape, output_strides, axis = geometry
        tensors = [
            tensor(i + 1, "int32", shape, stride)
            for i, (shape, stride) in enumerate(zip(shapes, source_strides))
        ]
        tensors.append(
            tensor(len(tensors) + 1, "int32", output_shape, output_strides)
        )
        document = graph("concatenate", tensors, axis)
        assert parser(document) == ["concatenate"]
        extended.kernel_configuration(document)
        # Construct expected destinations by enumerating input coordinates,
        # independently of the kernel's output-coordinate decomposition.
        for replay in range(2):
            sources, expected, begin = [], {}, 0
            for index, (shape, stride) in enumerate(
                zip(shapes, source_strides)
            ):
                memory = {}
                coordinates = product(*(range(d) for d in shape))
                for logical, coordinate in enumerate(coordinates):
                    value = 0x80000000 + index * 65536 + logical
                    value ^= replay * 0x12345678
                    offset = sum(a * b for a, b in zip(coordinate, stride))
                    memory[offset] = value
                    destination = list(coordinate)
                    destination[axis] += begin
                    offset = sum(
                        a * b for a, b in zip(destination, output_strides)
                    )
                    expected[offset] = value
                sources.append(memory)
                begin += shape[axis]
            actual = dict.fromkeys(expected)
            original_sources = copy.deepcopy(sources)
            begin = 0
            for shape, stride, source in zip(shapes, source_strides, sources):
                end = begin + shape[axis]
                for logical in range(math.prod(output_shape) + 7):
                    execute(
                        _Address(source),
                        _Address(actual),
                        logical,
                        math.prod(output_shape),
                        tuple(output_shape),
                        tuple(stride),
                        tuple(output_strides),
                        axis,
                        begin,
                        end,
                        "uint32",
                        1,
                    )
                begin = end
            assert actual == expected
            assert sources == original_sources


def main():
    provider = provider_loader.get_provider("thead")
    modules = sys.modules
    parser = modules[provider.__package__ + ".dispatch.graph"]._parse_graph
    extended = modules[provider.__package__ + ".dispatch.extended"]
    artifacts = modules[provider.__package__ + ".codegen.artifacts"]
    emit = modules[provider.__package__ + ".codegen.extended"]
    index = graph("gen_index", [tensor(1, "int32", [2, 3], [8, 2])], 1)
    assert parser(index) == ["gen_index"]
    _, plan = extended.kernel_configuration(index)
    assert plan[0] == "gen_index_kernel"
    assert plan[2]["AXIS_EXTENT"] == 3 and plan[2]["INNER"] == 1
    compact = graph("gen_index", [tensor(1, "int32", [2, 2], [2, 3])], 1)
    assert parser(compact) == ["gen_index"]
    _, compact_plan = extended.kernel_configuration(compact)
    assert compact_plan[2]["OUTPUT_STRIDE_6"] == 2
    assert compact_plan[2]["OUTPUT_STRIDE_7"] == 3
    for overlapping in ([1, 1], [2, 2]):
        invalid = copy.deepcopy(compact)
        invalid["tensors"][0]["strides"] = overlapping
        require_rejected(lambda: extended.kernel_configuration(invalid))
    for operation in ("concatenate", "add", "rng"):
        invalid = copy.deepcopy(index)
        invalid["nodes"][0]["type"] = operation
        require_rejected(lambda: parser(invalid))
    invalid = copy.deepcopy(index)
    invalid["nodes"][0]["outputs"] = []
    require_rejected(lambda: parser(invalid))
    for field, value in (("axis", 2), ("n_elements", 5)):
        invalid = copy.deepcopy(index)
        invalid["nodes"][0]["attributes"][field] = value
        require_rejected(lambda: extended.kernel_configuration(invalid))

    dtype_names = (
        "float32",
        "int32",
        "float16",
        "bfloat16",
        "boolean",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e8m0",
    )
    with tempfile.TemporaryDirectory(prefix="flagdnn-thead-index-") as temp:
        for dtype in dtype_names:
            document = graph(
                "concatenate",
                [
                    tensor(1, dtype, [2, 3], [11, 2]),
                    tensor(2, dtype, [2, 5], [17, 3]),
                    tensor(3, dtype, [2, 8], [25, 2]),
                ],
                1,
            )
            assert parser(document) == ["concatenate"]
            output = Path(temp) / dtype
            with patch.object(artifacts, "populate_launch_resources"):
                result = emit._compile_extended(
                    request={"graph": document, "flagdnn_version": "0.2.0"},
                    request_bytes=b"contract",
                    identity={},
                    target="ppu_contract_cc80",
                    output_directory=output,
                    enable_autotune=False,
                    operation="concatenate",
                )
            assert result["stage_count"] == 1
            manifest = json.loads((output / "manifest.json").read_text())
            stage = manifest["program"]["stages"][0]
            assert stage["source_node_ids"] == [0]
            assert stage["kernel"]["ownership"] == "platform"
            variant = stage["variants"][0]
            assert [a["uid"] for a in variant["arguments"]] == [1, 2, 3]
            if dtype.startswith("fp8_"):
                assert variant["full_signature"] == "*i8:16,*i8:16,*i8:16,256"
                assert all(
                    a.get("storage_view") == "fp8_bytes"
                    for a in variant["arguments"]
                )
            else:
                assert all(
                    "storage_view" not in a for a in variant["arguments"]
                )
            source = ast.parse((output / "generated_stage_0.py").read_text())
            entry = next(
                n
                for n in source.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "concatenate_kernel"
            )
            calls = [
                n
                for n in ast.walk(entry)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_concatenate_region"
            ]
            assert len(calls) == 2
            # The two input regions meet exactly at coordinate three.
            assert [(n.args[8].value, n.args[9].value) for n in calls] == [
                (0, 3),
                (3, 8),
            ]
            invalid = copy.deepcopy(document)
            invalid["nodes"][0]["attributes"]["input_count"] = 1
            require_rejected(lambda: extended.kernel_configuration(invalid))
            invalid = copy.deepcopy(document)
            invalid["tensors"][1]["dimensions"][0] = 3
            require_rejected(lambda: extended.kernel_configuration(invalid))
            invalid = copy.deepcopy(document)
            invalid["tensors"][0]["strides"] = [1, 1]
            require_rejected(lambda: extended.kernel_configuration(invalid))
    verify_repeated_inputs(parser, extended, artifacts, emit)
    verify_kernel_addresses(parser, extended)
    print("THead index planning contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
