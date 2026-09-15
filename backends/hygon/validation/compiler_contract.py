#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python contracts for Hygon NN planning and autotuning."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


sys.dont_write_bytecode = True


_JIT_DISCOVERY_ENVIRONMENT = (
    "FLAGDNN_BACKEND_PATH",
    "FLAGDNN_HYGON_COMPILER_ENVIRONMENT",
    "FLAGDNN_HYGON_TRITON_JIT_DIR",
    "FLAGDNN_HYGON_TRITON_JIT_LIBRARY",
    "FLAGDNN_HYGON_TRITON_JIT_ROOT",
    "FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR",
)


@contextmanager
def _isolated_jit_discovery_environment() -> Iterator[None]:
    previous = {
        name: os.environ.get(name) for name in _JIT_DISCOVERY_ENVIRONMENT
    }
    for name in _JIT_DISCOVERY_ENVIRONMENT:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_modules(
    source_root: Path,
) -> tuple[ModuleType, ModuleType, ModuleType, Any]:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "compiler"))

    from backends.hygon import compiler, compiler_nn, compiler_tensor
    from flagdnn_codegen.kernel_registry import select_kernel_candidate

    return compiler, compiler_nn, compiler_tensor, select_kernel_candidate


def _check_pointer_range_specialization(compiler: ModuleType) -> None:
    assert not compiler._triton_supports_pointer_range("3.1.0")
    assert not compiler._triton_supports_pointer_range("3.2.0")
    assert compiler._triton_supports_pointer_range("3.3.0")
    assert compiler._triton_supports_pointer_range("3.6.0+git")

    function = type(
        "FakeKernel",
        (),
        {
            "arg_names": (
                "small_ptr",
                "large_ptr",
                "unaligned_ptr",
                "count",
            )
        },
    )()
    runtime_signature = {
        "small_ptr": "*fp16",
        "large_ptr": "*fp16",
        "unaligned_ptr": "*fp16",
        "count": "i32",
    }
    argument_abi = [
        {"kind": "tensor", "alignment": 16, "size": 4096},
        {
            "kind": "workspace_tensor",
            "alignment": 16,
            "size": compiler.MAX_I32 + 1,
        },
        {"kind": "tensor", "alignment": 4, "size": 4096},
        {"kind": "scalar_i32"},
        {"kind": "global_scratch_pointer"},
        {"kind": "profile_scratch_pointer"},
    ]

    previous_support = compiler.TRITON_SUPPORTS_POINTER_RANGE
    try:
        compiler.TRITON_SUPPORTS_POINTER_RANGE = True
        specialized = compiler._jit_runtime_signature(
            function, runtime_signature, argument_abi
        )
        assert specialized == {
            "small_ptr": "*fp16:16S",
            "large_ptr": "*fp16:16",
            "unaligned_ptr": "*fp16:S",
            "count": "i32",
        }

        compiler.TRITON_SUPPORTS_POINTER_RANGE = False
        legacy = compiler._jit_runtime_signature(
            function, runtime_signature, argument_abi
        )
        assert legacy == {
            "small_ptr": "*fp16:16",
            "large_ptr": "*fp16:16",
            "unaligned_ptr": "*fp16",
            "count": "i32",
        }
    finally:
        compiler.TRITON_SUPPORTS_POINTER_RANGE = previous_support

    from backends.hygon.prepare_standalone_compile import _HYGON

    specialization_block = textwrap.dedent(_HYGON)
    for minor in (1, 2):
        namespace = {
            "signature": ["*fp16:16S"],
            "constants": {},
            "signature_without_spec": {0: "*fp16"},
            "triton_version": type(
                "TritonVersion", (), {"major": 3, "minor": minor}
            )(),
        }
        try:
            exec(specialization_block, namespace)
        except RuntimeError as error:
            assert "requires Triton 3.3 or newer" in str(error)
        else:
            raise AssertionError(
                "Triton 3.1/3.2 accepted HCU pointer-range specialization"
            )

    namespace = {
        "signature": ["*fp16:16S"],
        "constants": {},
        "signature_without_spec": {0: "*fp16"},
        "triton_version": type(
            "TritonVersion", (), {"major": 3, "minor": 3}
        )(),
    }
    exec(specialization_block, namespace)
    assert namespace["attrs"] == {
        (0,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
    }


def _check_hcu_llvm_compatibility() -> None:
    from backends.hygon.prepare_standalone_compile import (
        _HCU_PATCH,
        _HCU_PATCH_ANCHOR,
        _UPSTREAM,
        _flagdnn_hygon_compatible_llir,
        _prepare_source,
    )

    source = (
        "define void @kernel() #5 {\n"
        "  call void @helper() #5\n"
        "  call void @other() #50\n"
        "}\n"
        "attributes #5 = { memory(none) }\n"
        "attributes #6 = { nounwind memory(argmem: read) }\n"
        "attributes #50 = { nounwind }\n"
    )
    assert _flagdnn_hygon_compatible_llir(source, 17) is source
    assert _flagdnn_hygon_compatible_llir(source, 18) is source

    compatible = _flagdnn_hygon_compatible_llir(source, 15)
    assert "memory(" not in compatible
    assert "\nattributes #5 =" not in compatible
    assert "call void @helper() #5" not in compatible
    assert "attributes #6 = { nounwind }" in compatible
    assert "call void @other() #50" in compatible
    assert "attributes #50 = { nounwind }" in compatible

    for value in (None, b"llvm"):
        try:
            _flagdnn_hygon_compatible_llir(value, 18)
        except TypeError:
            pass
        else:
            raise AssertionError("non-text LLVM IR was accepted")
    for value in (True, 18.0, "18"):
        try:
            _flagdnn_hygon_compatible_llir(source, value)
        except TypeError:
            pass
        else:
            raise AssertionError("invalid clang major version was accepted")
    try:
        _flagdnn_hygon_compatible_llir(source, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative clang major version was accepted")

    upstream_fixture = (
        "prefix\n" + _HCU_PATCH_ANCHOR + "middle\n" + _UPSTREAM + "suffix\n"
    )
    prepared = _prepare_source(upstream_fixture)
    assert _UPSTREAM not in prepared
    assert prepared.count(_HCU_PATCH) == 1
    assert 'if get_backend() == "HCU":' in prepared
    for invalid in (
        upstream_fixture.replace(_HCU_PATCH_ANCHOR, ""),
        upstream_fixture.replace(_HCU_PATCH_ANCHOR, _HCU_PATCH_ANCHOR * 2),
        upstream_fixture.replace(_UPSTREAM, ""),
        upstream_fixture.replace(_UPSTREAM, _UPSTREAM * 2),
    ):
        try:
            _prepare_source(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "standalone helper source drift did not fail closed"
            )


def _check_fp8_layout(compiler_tensor: ModuleType) -> None:
    pointer_types = {
        "fp8_e4m3": "*fp8e4nv",
        "fp8_e5m2": "*fp8e5",
    }
    assert not set(pointer_types).intersection(compiler_tensor.FLOAT_TYPES)

    operation_cases = {
        "reshape": (
            [2, 3],
            [3, 2],
            {
                "input_rank": 2,
                "output_rank": 2,
                "reshape_mode": 2,
            },
        ),
        "transpose": (
            [2, 3],
            [3, 2],
            {"rank": 2, "permutation": [1, 0]},
        ),
        "slice": (
            [4, 6],
            [2, 3],
            {
                "rank": 2,
                "starts": [0, 0],
                "limits": [4, 6],
                "slice_strides": [2, 2],
            },
        ),
    }
    for data_type, pointer_type in pointer_types.items():
        for operation, (
            input_shape,
            output_shape,
            attributes,
        ) in operation_cases.items():
            input_tensor = _tensor(1, data_type, input_shape)
            output_tensor = _tensor(2, data_type, output_shape)
            parameters = {
                **attributes,
                "n_elements": 6,
                "input_dimensions": input_tensor["dimensions"],
                "input_strides": input_tensor["strides"],
                "output_dimensions": output_tensor["dimensions"],
                "output_strides": output_tensor["strides"],
            }
            node = {
                "id": 0,
                "type": operation,
                "compute_data_type": "float32",
                "attributes": parameters,
                "inputs": [{"name": "input", "uid": 1, "optional": False}],
                "outputs": [{"name": "output", "uid": 2, "optional": False}],
            }
            parsed = compiler_tensor.parse_node(
                node, 0, 1, {1: input_tensor, 2: output_tensor}
            )
            configuration = compiler_tensor.kernel_configuration(parsed)
            assert configuration.function_name == "layout_copy_kernel"
            assert configuration.runtime_signature["input_ptr"] == pointer_type
            assert (
                configuration.runtime_signature["output_ptr"] == pointer_type
            )
            assert compiler_tensor.tensor_storage_size(input_tensor) == (
                6 if operation != "slice" else 24
            )


def _contiguous_strides(dimensions: list[int]) -> list[int]:
    result = [1] * len(dimensions)
    stride = 1
    for axis in range(len(dimensions) - 1, -1, -1):
        result[axis] = stride
        stride *= dimensions[axis]
    return result


def _tensor(uid: int, data_type: str, dimensions: list[int]) -> dict[str, Any]:
    return {
        "uid": uid,
        "data_type": data_type,
        "dimensions": dimensions,
        "strides": _contiguous_strides(dimensions),
        "alignment": 16,
        "virtual": False,
    }


def _parse_and_plan(
    compiler_nn: ModuleType,
    operation: str,
    specifications: dict[str, tuple[str, list[int]]],
    attributes: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    schema = compiler_nn.OPERATION_SCHEMAS[operation]
    port_names = (*schema.input_ports, *schema.output_ports)
    assert set(specifications) == set(port_names)

    registry: dict[int, dict[str, Any]] = {}
    port_uids: dict[str, int] = {}
    for uid, name in enumerate(port_names, start=1):
        data_type, dimensions = specifications[name]
        registry[uid] = _tensor(uid, data_type, dimensions)
        port_uids[name] = uid

    def ports(names: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {"name": name, "uid": port_uids[name], "optional": False}
            for name in names
        ]

    node = {
        "id": 0,
        "type": operation,
        "compute_data_type": "float32",
        "attributes": attributes,
        "inputs": ports(schema.input_ports),
        "outputs": ports(schema.output_ports),
    }
    parsed = compiler_nn.parse_node(node, 0, 1, registry)
    return parsed, compiler_nn.plan_kernel_stages(parsed)


def _tuning_configurations(
    compiler: ModuleType,
    select_kernel_candidate: Any,
    operation: str,
    stage: Any,
) -> list[dict[str, Any]]:
    candidate = select_kernel_candidate("hygon", operation)
    configurations, _ = compiler._load_nn_tuning(
        compiler._compiler_entry_path(), candidate, stage
    )
    return configurations


def _assert_mixed_stage_emission(
    compiler: ModuleType, parsed: dict[str, Any]
) -> None:
    maximum_uid = max(tensor["uid"] for tensor in parsed["tensors"])
    plans, workspaces, _ = compiler._nn_workspace_layout(
        [parsed], 0, maximum_uid
    )
    plan = plans[parsed["id"]]
    emitted: list[dict[str, Any]] = []
    local_stage_ids: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as temporary:
        for configuration in plan.stages:
            stage_id = len(emitted)
            dependencies = sorted(
                local_stage_ids[name] for name in configuration.dependencies
            )
            emitted.append(
                compiler._compile_nn_stage(
                    stage_id=stage_id,
                    node=parsed,
                    configuration=configuration,
                    dependencies=dependencies,
                    workspace={},
                    local_workspace=workspaces[parsed["id"]],
                    compiler_path=compiler._compiler_entry_path(),
                    output_directory=Path(temporary),
                    enable_autotune=True,
                )
            )
            local_stage_ids[configuration.stage_name] = stage_id
    assert any("variants" not in stage for stage in emitted)
    assert any("variants" in stage for stage in emitted)
    for stage in emitted:
        assert ("variants" in stage) == ("tuning" in stage)


def _convolution_case(
    compiler_nn: ModuleType,
    operation: str,
    image: list[int],
    weight: list[int],
    stride: list[int],
    padding: list[int],
    *,
    post_padding: list[int] | None = None,
    dilation: list[int] | None = None,
    data_type: str = "float16",
    groups: int = 1,
    convolution_mode: int = 0,
) -> tuple[dict[str, Any], Any]:
    rank = len(image) - 2
    assert rank in (1, 2, 3)
    assert len(weight) == len(image)
    assert len(stride) == rank
    assert len(padding) == rank
    post = padding if post_padding is None else post_padding
    dilations = [1] * rank if dilation is None else dilation
    assert len(post) == rank
    assert len(dilations) == rank
    output = [image[0], weight[0]]
    for axis in range(rank):
        effective_filter = (weight[axis + 2] - 1) * dilations[axis] + 1
        output.append(
            (image[axis + 2] + padding[axis] + post[axis] - effective_filter)
            // stride[axis]
            + 1
        )
    target = (
        image
        if operation == "convolution_dgrad"
        else weight if operation == "convolution_wgrad" else output
    )
    n_outputs = 1
    for dimension in target:
        n_outputs *= dimension
    attributes = {
        "spatial_rank": rank,
        "pre_padding": padding,
        "post_padding": post,
        "stride": stride,
        "dilation": dilations,
        "groups": groups,
        "convolution_mode": convolution_mode,
        "n_outputs": n_outputs,
    }
    if operation in ("conv2d_fprop", "convolution_fprop"):
        specifications = {
            "input": (data_type, image),
            "filter": (data_type, weight),
            "output": (data_type, output),
        }
    elif operation == "convolution_dgrad":
        specifications = {
            "dy": (data_type, output),
            "w": (data_type, weight),
            "dx": (data_type, image),
        }
    else:
        assert operation == "convolution_wgrad"
        specifications = {
            "dy": (data_type, output),
            "x": (data_type, image),
            "dw": (data_type, weight),
        }
    return _parse_and_plan(compiler_nn, operation, specifications, attributes)


def _assert_workspace_layout(plan: Any) -> None:
    assert 0 <= plan.workspace_size <= 512 * 1024 * 1024
    ordered = sorted(plan.workspace_tensors, key=lambda tensor: tensor.offset)
    previous_end = 0
    names: set[str] = set()
    for tensor in ordered:
        assert tensor.name not in names
        names.add(tensor.name)
        assert tensor.offset % tensor.alignment == 0
        assert tensor.offset >= previous_end
        assert tensor.offset + tensor.size <= plan.workspace_size
        previous_end = tensor.offset + tensor.size


def _check_private_convolution_autotune(
    compiler: ModuleType,
    compiler_nn: ModuleType,
    select_kernel_candidate: Any,
) -> None:
    _, fprop_1x1 = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 64, 28, 28],
        [128, 64, 1, 1],
        [1, 1],
        [0, 0],
    )
    _, fprop_low_ci = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 3, 640, 640],
        [96, 3, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, fprop_low_ci_c4 = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 4, 16, 16],
        [8, 4, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, fprop_packed = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 64, 40, 40],
        [128, 64, 3, 3],
        [2, 2],
        [1, 1],
    )
    fprop_standard_node, fprop_standard = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
    )
    _, fprop_standard_alias = _convolution_case(
        compiler_nn,
        "convolution_fprop",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
    )
    _, fprop_standard_fp16 = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float16",
    )
    _, fprop_standard_bf16 = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="bfloat16",
    )
    _, fprop_standard_batch = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [4, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
    )
    _, fprop_standard_grouped = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 32, 32, 32],
        [64, 16, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
        groups=2,
    )
    _, fprop_standard_post_padding = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
        post_padding=[0, 1],
    )
    _, fprop_standard_dilation = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
        dilation=[2, 2],
    )
    fprop_standard_shape_fallbacks: list[Any] = []
    for image, weight, stride, padding in (
        ([8, 16, 32, 32], [64, 16, 3, 3], [1, 1], [1, 1]),
        ([8, 32, 32, 32], [32, 32, 3, 3], [1, 1], [1, 1]),
        ([8, 32, 28, 28], [64, 32, 3, 3], [1, 1], [1, 1]),
        ([8, 32, 32, 32], [64, 32, 5, 3], [1, 1], [2, 1]),
        ([8, 32, 32, 32], [64, 32, 3, 3], [2, 2], [1, 1]),
        ([8, 32, 32, 32], [64, 32, 3, 3], [1, 1], [0, 0]),
    ):
        _, fallback = _convolution_case(
            compiler_nn,
            "conv2d_fprop",
            image,
            weight,
            stride,
            padding,
            data_type="float32",
        )
        fprop_standard_shape_fallbacks.append(fallback)
    fprop_standard_noncontiguous: list[Any] = []
    for tensor_name in ("image", "weight", "result"):
        noncontiguous_node = copy.deepcopy(fprop_standard_node)
        noncontiguous_node["derived"][tensor_name]["strides"][-2] += 1
        fprop_standard_noncontiguous.append(
            compiler_nn.plan_kernel_stages(noncontiguous_node)
        )
    fprop_yolo_x_node, fprop_yolo_x = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 768, 40, 40],
        [768, 768, 3, 3],
        [2, 2],
        [1, 1],
        data_type="float32",
    )
    _, fprop_yolo_x_fp16 = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 768, 40, 40],
        [768, 768, 3, 3],
        [2, 2],
        [1, 1],
        data_type="float16",
    )
    _, fprop_yolo_ml_fp32 = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 512, 40, 40],
        [512, 512, 3, 3],
        [2, 2],
        [1, 1],
        data_type="float32",
    )
    _, fprop_yolo_x_post_padding = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 768, 40, 40],
        [768, 768, 3, 3],
        [2, 2],
        [1, 1],
        data_type="float32",
        post_padding=[0, 1],
    )
    _, fprop_yolo_x_dilation = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 768, 40, 40],
        [768, 768, 3, 3],
        [2, 2],
        [1, 1],
        data_type="float32",
        dilation=[2, 2],
    )
    _, fprop_yolo_x_grouped = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 768, 40, 40],
        [768, 384, 3, 3],
        [2, 2],
        [1, 1],
        data_type="float32",
        groups=2,
    )
    _, dgrad_1x1 = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 64, 28, 28],
        [128, 64, 1, 1],
        [1, 1],
        [0, 0],
    )
    _, dgrad_stride1 = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
    )
    dgrad_exact_node, dgrad_exact = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float32",
    )
    dgrad_exact_bf16_node, dgrad_exact_bf16 = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="bfloat16",
    )
    _, dgrad_exact_fp16_fallback = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
        data_type="float16",
    )
    _, dgrad_stride2 = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [1, 3, 640, 640],
        [96, 3, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, dgrad_stride2_regular = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 64, 56, 56],
        [128, 64, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, dgrad_stride2_grouped = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [8, 64, 56, 56],
        [128, 32, 3, 3],
        [2, 2],
        [1, 1],
        groups=2,
    )
    _, dgrad_stride2_c4 = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [1, 4, 16, 16],
        [8, 4, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, dgrad_stride2_c5 = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [1, 5, 41, 43],
        [33, 5, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, dgrad_stride2_flip = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [1, 5, 16, 16],
        [8, 5, 3, 3],
        [2, 2],
        [1, 1],
        convolution_mode=1,
    )
    _, wgrad_1x1 = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [8, 64, 28, 28],
        [128, 64, 1, 1],
        [1, 1],
        [0, 0],
    )
    _, wgrad_split = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [1, 3, 640, 640],
        [96, 3, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, wgrad_packed_split = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [8, 32, 32, 32],
        [64, 32, 3, 3],
        [1, 1],
        [1, 1],
    )
    _, wgrad_multirow = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [8, 64, 56, 56],
        [128, 64, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, wgrad_multirow_tail = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [2, 5, 35, 37],
        [33, 5, 3, 3],
        [2, 2],
        [1, 1],
    )
    wgrad_p5_plans: dict[str, Any] = {}
    for name, c_in, c_out in (
        ("n", 128, 256),
        ("s", 256, 512),
        ("ml", 512, 512),
        ("x", 768, 768),
    ):
        _, plan = _convolution_case(
            compiler_nn,
            "convolution_wgrad",
            [1, c_in, 40, 40],
            [c_out, c_in, 3, 3],
            [2, 2],
            [1, 1],
        )
        wgrad_p5_plans[name] = plan
    _, wgrad_p5_tail = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [1, 5, 40, 40],
        [33, 5, 3, 3],
        [2, 2],
        [1, 1],
    )
    _, wgrad_multirow_flip = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [8, 64, 56, 56],
        [128, 64, 3, 3],
        [2, 2],
        [1, 1],
        convolution_mode=1,
    )
    _, wgrad_p5_flip = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [1, 512, 40, 40],
        [512, 512, 3, 3],
        [2, 2],
        [1, 1],
        convolution_mode=1,
    )
    _, wgrad_grouped = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [8, 64, 56, 56],
        [128, 32, 3, 3],
        [2, 2],
        [1, 1],
        groups=2,
    )
    wgrad_noncontiguous_node, _ = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [8, 64, 56, 56],
        [128, 64, 3, 3],
        [2, 2],
        [1, 1],
    )
    wgrad_noncontiguous_node["derived"]["image"]["strides"] = [
        64 * 56 * 64,
        56 * 64,
        64,
        1,
    ]
    wgrad_noncontiguous = compiler_nn.plan_kernel_stages(
        wgrad_noncontiguous_node
    )
    wgrad_fp32_resource_plans: list[Any] = []
    for image, weight, stride, padding, options in (
        ([2, 8, 16, 16], [16, 8, 3, 3], [1, 1], [1, 1], {}),
        ([8, 32, 32, 32], [64, 32, 3, 3], [1, 1], [1, 1], {}),
        ([1, 3, 640, 640], [96, 3, 3, 3], [2, 2], [1, 1], {}),
        ([8, 64, 56, 56], [128, 64, 3, 3], [2, 2], [1, 1], {}),
        ([1, 128, 40, 40], [256, 128, 3, 3], [2, 2], [1, 1], {}),
        ([1, 768, 40, 40], [768, 768, 3, 3], [2, 2], [1, 1], {}),
        ([8, 64, 28, 28], [128, 64, 1, 1], [1, 1], [0, 0], {}),
        (
            [8, 64, 56, 56],
            [128, 64, 3, 3],
            [2, 2],
            [1, 1],
            {"convolution_mode": 1},
        ),
        (
            [8, 64, 56, 56],
            [128, 32, 3, 3],
            [2, 2],
            [1, 1],
            {"groups": 2},
        ),
    ):
        _, plan = _convolution_case(
            compiler_nn,
            "convolution_wgrad",
            image,
            weight,
            stride,
            padding,
            data_type="float32",
            **options,
        )
        wgrad_fp32_resource_plans.append(plan)
    assert [stage.stage_name for stage in fprop_packed.stages] == [
        "fprop_im2col",
        "fprop_gemm",
    ]
    assert [stage.stage_name for stage in fprop_standard.stages] == [
        "fprop_standard_3x3"
    ]
    assert [stage.function_name for stage in fprop_standard.stages] == [
        "hygon_conv2d_fprop_standard_3x3_nchw_kernel"
    ]
    assert [stage.function_name for stage in fprop_standard_alias.stages] == [
        "hygon_conv2d_fprop_standard_3x3_nchw_kernel"
    ]
    assert fprop_standard.stages[0].runtime_signature == {
        "x_ptr": "*fp32",
        "w_ptr": "*fp32",
        "y_ptr": "*fp32",
    }
    assert fprop_standard.stages[0].constants == {
        "BLOCK_OC_S": 64,
        "BLOCK_K_S": 32,
    }
    assert (
        fprop_standard.stages[0].tuning
        == compiler_nn.CONV_FPROP_STANDARD_3X3_TUNING
    )
    assert fprop_standard.stages[0].grid_spec.kind == (
        "conv_private_fprop_standard_3x3"
    )
    assert fprop_standard.workspace_size == 0
    assert not fprop_standard.workspace_tensors
    standard_configurations = _tuning_configurations(
        compiler,
        select_kernel_candidate,
        fprop_standard.operation,
        fprop_standard.stages[0],
    )
    assert standard_configurations == [
        {
            "META": {
                "BLOCK_OC_S": 64,
                "BLOCK_K_S": 32,
            },
            "num_warps": 4,
            "num_stages": 1,
        },
        {
            "META": {
                "BLOCK_OC_S": 64,
                "BLOCK_K_S": 32,
            },
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "META": {
                "BLOCK_OC_S": 64,
                "BLOCK_K_S": 64,
            },
            "num_warps": 4,
            "num_stages": 1,
        },
        {
            "META": {
                "BLOCK_OC_S": 32,
                "BLOCK_K_S": 64,
            },
            "num_warps": 4,
            "num_stages": 1,
        },
    ]
    standard_grids = {
        fprop_standard.stages[0].variant(
            configuration["META"],
            num_warps=configuration["num_warps"],
            num_stages=configuration["num_stages"],
        )[1]
        for configuration in standard_configurations
    }
    assert standard_grids == {(32, 8, 1), (64, 8, 1)}
    for generic_plan in (
        fprop_standard_fp16,
        fprop_standard_bf16,
        fprop_standard_batch,
        fprop_standard_post_padding,
        fprop_standard_dilation,
        *fprop_standard_shape_fallbacks,
    ):
        assert [stage.stage_name for stage in generic_plan.stages] == [
            "fprop_im2col",
            "fprop_gemm",
        ]
        assert generic_plan.stages[1].function_name == (
            "hygon_conv2d_fprop_im2col_kernel"
        )
    assert [
        stage.function_name for stage in fprop_standard_grouped.stages
    ] == ["conv2d_spatial_nchw_kernel"]
    for noncontiguous in fprop_standard_noncontiguous:
        assert [stage.function_name for stage in noncontiguous.stages] == [
            "conv2d_spatial_nchw_kernel"
        ]
    assert [stage.stage_name for stage in fprop_yolo_x.stages] == [
        "fprop_im2col",
        "fprop_gemm",
    ]
    assert [stage.function_name for stage in fprop_yolo_x.stages] == [
        "hygon_conv2d_im2col_nchw_kernel",
        "hygon_conv2d_fprop_yolo_x_p5_gemm_kernel",
    ]
    assert [stage.dependencies for stage in fprop_yolo_x.stages] == [
        (),
        ("fprop_im2col",),
    ]
    assert fprop_yolo_x.stages[0].tuning.table == "conv_fprop"
    assert fprop_yolo_x.stages[1].tuning.table == "conv_fprop"
    assert (
        fprop_yolo_x.stages[1].tuning
        == compiler_nn.CONV_FPROP_YOLO_X_P5_GEMM_TUNING
    )
    assert fprop_yolo_x.workspace_size == 11_059_200
    assert len(fprop_yolo_x.workspace_tensors) == 1
    assert fprop_yolo_x.workspace_tensors[0].data_type == "float32"
    assert fprop_yolo_x.workspace_tensors[0].dimensions == (1, 6912, 400)
    yolo_x_gemm_configurations = _tuning_configurations(
        compiler,
        select_kernel_candidate,
        fprop_yolo_x.operation,
        fprop_yolo_x.stages[1],
    )
    assert yolo_x_gemm_configurations == [
        {
            "META": {
                "BLOCK_M_X": 32,
                "BLOCK_OC_X": 32,
                "BLOCK_K_X": 32,
                "GROUP_M_X": 8,
                "YOLO_X_P5": 1,
            },
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "META": {
                "BLOCK_M_X": 32,
                "BLOCK_OC_X": 64,
                "BLOCK_K_X": 64,
                "GROUP_M_X": 12,
                "YOLO_X_P5": 1,
            },
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "META": {
                "BLOCK_M_X": 32,
                "BLOCK_OC_X": 64,
                "BLOCK_K_X": 64,
                "GROUP_M_X": 16,
                "YOLO_X_P5": 1,
            },
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "META": {
                "BLOCK_M_X": 32,
                "BLOCK_OC_X": 64,
                "BLOCK_K_X": 64,
                "GROUP_M_X": 24,
                "YOLO_X_P5": 1,
            },
            "num_warps": 4,
            "num_stages": 2,
        },
    ]
    yolo_x_gemm_grids = {
        fprop_yolo_x.stages[1].variant(
            configuration["META"],
            num_warps=configuration["num_warps"],
            num_stages=configuration["num_stages"],
        )[1]
        for configuration in yolo_x_gemm_configurations
    }
    assert yolo_x_gemm_grids == {(312, 1, 1), (156, 1, 1)}
    for generic_plan in (
        fprop_packed,
        fprop_yolo_x_fp16,
        fprop_yolo_ml_fp32,
        fprop_yolo_x_post_padding,
        fprop_yolo_x_dilation,
    ):
        assert generic_plan.stages[1].tuning.table == "conv_fprop"
        assert generic_plan.stages[1].function_name == (
            "hygon_conv2d_fprop_im2col_kernel"
        )
    assert [stage.function_name for stage in fprop_yolo_x_grouped.stages] == [
        "conv2d_spatial_nchw_kernel"
    ]
    for tensor_name in ("image", "weight", "result"):
        noncontiguous_node = copy.deepcopy(fprop_yolo_x_node)
        noncontiguous_node["derived"][tensor_name]["strides"][-2] += 1
        noncontiguous = compiler_nn.plan_kernel_stages(noncontiguous_node)
        assert [stage.function_name for stage in noncontiguous.stages] == [
            "conv2d_spatial_nchw_kernel"
        ]
    assert [stage.stage_name for stage in fprop_low_ci.stages] == [
        "fprop_stride2_low_ci"
    ]
    assert [stage.function_name for stage in fprop_low_ci.stages] == [
        "hygon_conv2d_fprop_stride2_low_ci_nchw_kernel"
    ]
    assert fprop_low_ci.stages[0].grid_spec.kind == ("conv_private_fprop_rows")
    assert fprop_low_ci.workspace_size == 0
    assert [stage.stage_name for stage in fprop_low_ci_c4.stages] == [
        "fprop_im2col",
        "fprop_gemm",
    ]
    assert [stage.stage_name for stage in dgrad_exact.stages] == [
        "dgrad_s1_exact_gemm",
        "dgrad_s1_exact_col2im",
    ]
    assert [stage.function_name for stage in dgrad_exact.stages] == [
        "hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
        "hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel",
    ]
    assert [stage.stage_name for stage in dgrad_exact_bf16.stages] == [
        "dgrad_s1_exact_gemm",
        "dgrad_s1_exact_col2im",
    ]
    assert [stage.function_name for stage in dgrad_exact_bf16.stages] == [
        "hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
        "hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel",
    ]
    assert [stage.runtime_signature for stage in dgrad_exact_bf16.stages] == [
        {"dy_ptr": "*bf16", "w_ptr": "*bf16", "col_ptr": "*fp32"},
        {"col_ptr": "*fp32", "dx_ptr": "*bf16"},
    ]
    assert [
        stage.function_name for stage in dgrad_exact_fp16_fallback.stages
    ] == ["hygon_conv_dgrad2d_stride1_kernel"]
    assert [stage.dependencies for stage in dgrad_exact.stages] == [
        (),
        ("dgrad_s1_exact_gemm",),
    ]
    assert [stage.grid_spec.kind for stage in dgrad_exact.stages] == [
        "conv_private_dgrad_columns",
        "conv_private_dgrad_fixed_2d",
    ]
    assert [stage.grid_spec.extents for stage in dgrad_exact.stages] == [
        (32 * 32, 32 * 3 * 3, 8),
        (32, 8 * 32),
    ]
    assert [stage.default_grid for stage in dgrad_exact.stages] == [
        (144, 8, 1),
        (8, 256, 1),
    ]
    assert [stage.default_num_warps for stage in dgrad_exact.stages] == [
        4,
        2,
    ]
    assert [stage.default_num_stages for stage in dgrad_exact.stages] == [
        1,
        1,
    ]
    assert len(dgrad_exact.workspace_tensors) == 1
    exact_columns = dgrad_exact.workspace_tensors[0]
    assert exact_columns.name == "dgrad_s1_exact_columns"
    assert exact_columns.data_type == "float32"
    assert exact_columns.dimensions == (8, 32 * 3 * 3, 32 * 32)
    assert exact_columns.strides == (32 * 3 * 3 * 32 * 32, 32 * 32, 1)
    assert exact_columns.offset == 0
    assert exact_columns.alignment == compiler_nn.WORKSPACE_ALIGNMENT
    assert exact_columns.size == 9_437_184
    assert exact_columns.size % compiler_nn.WORKSPACE_ALIGNMENT == 0
    assert exact_columns.size == compiler_nn.DGRAD_EXACT_3X3_S1_WORKSPACE_CAP
    assert dgrad_exact.workspace_size == exact_columns.size
    assert dgrad_exact_bf16.workspace_tensors == dgrad_exact.workspace_tensors
    assert dgrad_exact_bf16.workspace_size == dgrad_exact.workspace_size

    exact_gemm, exact_col2im = dgrad_exact.stages
    assert exact_gemm.constants["EXACT_3X3_S1"] == 1
    assert exact_gemm.constants["BLOCK_CI"] == 32
    assert exact_gemm.constants["BLOCK_CO"] == 32
    assert exact_gemm.constants["BLOCK_M"] == 64
    assert exact_gemm.constants["GROUP_M"] == 16
    assert exact_col2im.constants["EXACT_3X3_S1"] == 1
    assert exact_col2im.constants["BLOCK_H"] == 4
    exact_gemm_configurations = _tuning_configurations(
        compiler,
        select_kernel_candidate,
        "convolution_dgrad",
        exact_gemm,
    )
    exact_col2im_configurations = _tuning_configurations(
        compiler,
        select_kernel_candidate,
        "convolution_dgrad",
        exact_col2im,
    )
    assert exact_gemm_configurations == [
        {
            "META": {
                "BLOCK_M": 64,
                "GROUP_M": 16,
                "EXACT_3X3_S1": 1,
            },
            "num_warps": 2,
            "num_stages": 1,
        },
        {
            "META": {
                "BLOCK_M": 64,
                "GROUP_M": 16,
                "EXACT_3X3_S1": 1,
            },
            "num_warps": 4,
            "num_stages": 1,
        },
        {
            "META": {
                "BLOCK_M": 128,
                "GROUP_M": 16,
                "EXACT_3X3_S1": 1,
            },
            "num_warps": 4,
            "num_stages": 1,
        },
    ]
    assert exact_col2im_configurations == [
        {
            "META": {"BLOCK_H": 4, "EXACT_3X3_S1": 1},
            "num_warps": 2,
            "num_stages": 1,
        },
        {
            "META": {"BLOCK_H": 4, "EXACT_3X3_S1": 1},
            "num_warps": 4,
            "num_stages": 1,
        },
        {
            "META": {"BLOCK_H": 8, "EXACT_3X3_S1": 1},
            "num_warps": 4,
            "num_stages": 1,
        },
    ]
    exact_gemm_grids: set[tuple[int, int, int]] = set()
    for configuration in exact_gemm_configurations:
        constants, grid = exact_gemm.variant(
            configuration["META"],
            num_warps=configuration["num_warps"],
            num_stages=configuration["num_stages"],
        )
        assert constants["BLOCK_CI"] == 32
        assert constants["BLOCK_CO"] == 32
        assert constants["M"] % constants["BLOCK_M"] == 0
        assert (
            constants["C_IN"] * constants["KH"] * constants["KW"]
        ) % constants["BLOCK_CI"] == 0
        assert constants["C_OUT"] % constants["BLOCK_CO"] == 0
        exact_gemm_grids.add(grid)
    assert exact_gemm_grids == {(144, 8, 1), (72, 8, 1)}
    exact_col2im_grids = {
        exact_col2im.variant(
            configuration["META"],
            num_warps=configuration["num_warps"],
            num_stages=configuration["num_stages"],
        )[1]
        for configuration in exact_col2im_configurations
    }
    assert exact_col2im_grids == {(8, 256, 1), (4, 256, 1)}

    exact_functions = {
        "hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
        "hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel",
    }
    stride1_fallback_cases = (
        (
            [7, 32, 32, 32],
            [64, 32, 3, 3],
            [1, 1],
            [1, 1],
            {},
        ),
        (
            [8, 16, 32, 32],
            [64, 16, 3, 3],
            [1, 1],
            [1, 1],
            {},
        ),
        (
            [8, 32, 32, 32],
            [32, 32, 3, 3],
            [1, 1],
            [1, 1],
            {},
        ),
        (
            [8, 32, 31, 32],
            [64, 32, 3, 3],
            [1, 1],
            [1, 1],
            {},
        ),
        (
            [8, 32, 32, 31],
            [64, 32, 3, 3],
            [1, 1],
            [1, 1],
            {},
        ),
        (
            [8, 32, 32, 32],
            [64, 16, 3, 3],
            [1, 1],
            [1, 1],
            {"groups": 2},
        ),
        (
            [8, 32, 32, 32],
            [64, 32, 3, 3],
            [1, 1],
            [1, 1],
            {"convolution_mode": 1},
        ),
        (
            [8, 32, 32, 32],
            [64, 32, 3, 3],
            [1, 1],
            [1, 1],
            {"post_padding": [0, 1]},
        ),
        (
            [8, 32, 32, 32],
            [64, 32, 3, 3],
            [1, 1],
            [2, 2],
            {"dilation": [2, 2]},
        ),
    )
    for image, weight, stride, padding, options in stride1_fallback_cases:
        _, fallback = _convolution_case(
            compiler_nn,
            "convolution_dgrad",
            image,
            weight,
            stride,
            padding,
            data_type="float32",
            **options,
        )
        assert [stage.function_name for stage in fallback.stages] == [
            "hygon_conv_dgrad2d_stride1_kernel"
        ]
        assert exact_functions.isdisjoint(
            stage.function_name for stage in fallback.stages
        )
    assert [stage.function_name for stage in dgrad_stride1.stages] == [
        "hygon_conv_dgrad2d_stride1_kernel"
    ]
    for exact_node in (dgrad_exact_node, dgrad_exact_bf16_node):
        for tensor_name in ("image", "weight", "result"):
            noncontiguous_node = copy.deepcopy(exact_node)
            noncontiguous_node["derived"][tensor_name]["strides"][-2] += 1
            noncontiguous = compiler_nn.plan_kernel_stages(noncontiguous_node)
            assert [stage.function_name for stage in noncontiguous.stages] == [
                "conv_dgrad_nd_kernel"
            ]

    assert [stage.stage_name for stage in dgrad_stride2.stages] == [
        "dgrad_s2_contribution_gemm",
        "dgrad_s2_contribution_col2im",
    ]
    assert [stage.function_name for stage in dgrad_stride2.stages] == [
        "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel",
        "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
    ]
    assert [stage.dependencies for stage in dgrad_stride2.stages] == [
        (),
        ("dgrad_s2_contribution_gemm",),
    ]
    assert [stage.grid_spec.kind for stage in dgrad_stride2.stages] == [
        "conv_private_dgrad_columns",
        "linear",
    ]
    assert len(dgrad_stride2.workspace_tensors) == 1
    dgrad_columns = dgrad_stride2.workspace_tensors[0]
    assert dgrad_columns.name == "dgrad_s2_contribution_columns"
    assert dgrad_columns.data_type == "float32"
    assert dgrad_columns.dimensions == (1, 320 * 320, 3 * 3 * 3)
    assert dgrad_columns.offset == 0
    assert dgrad_columns.size == 320 * 320 * 3 * 3 * 3 * 4
    assert dgrad_stride2.workspace_size >= dgrad_columns.size
    assert dgrad_stride2.workspace_size <= 512 * 1024 * 1024
    assert [stage.function_name for stage in dgrad_stride2_c4.stages] == [
        "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel",
        "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
    ]
    assert [stage.function_name for stage in dgrad_stride2_c5.stages] == [
        "hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel",
        "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
    ]
    tail_columns = dgrad_stride2_c5.workspace_tensors[0]
    assert tail_columns.dimensions == (1, 5 * 3 * 3, 21 * 22)
    assert tail_columns.strides == (5 * 3 * 3 * 21 * 22, 21 * 22, 1)
    assert [stage.dependencies for stage in dgrad_stride2_c5.stages] == [
        (),
        ("dgrad_s2_contribution_gemm",),
    ]
    assert [stage.grid_spec.kind for stage in dgrad_stride2_c5.stages] == [
        "conv_private_dgrad_columns",
        "linear",
    ]
    assert [stage.grid_spec.extents for stage in dgrad_stride2_c5.stages] == [
        (21 * 22, 5 * 3 * 3, 1),
        (1 * 5 * 41 * 43,),
    ]
    assert [stage.default_grid for stage in dgrad_stride2_c5.stages] == [
        (30, 1, 1),
        (35, 1, 1),
    ]
    assert [stage.function_name for stage in dgrad_stride2_flip.stages] == [
        "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel",
        "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
    ]
    assert dgrad_stride2_flip.stages[0].constants["FLIP_FILTER"] is True
    assert [stage.function_name for stage in dgrad_stride2_regular.stages] == [
        "hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel",
        "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
    ]
    regular_columns = dgrad_stride2_regular.workspace_tensors[0]
    assert regular_columns.dimensions == (8, 64 * 3 * 3, 28 * 28)
    assert regular_columns.strides == (
        64 * 3 * 3 * 28 * 28,
        28 * 28,
        1,
    )
    assert [stage.dependencies for stage in dgrad_stride2_regular.stages] == [
        (),
        ("dgrad_s2_contribution_gemm",),
    ]
    assert [
        stage.grid_spec.kind for stage in dgrad_stride2_regular.stages
    ] == ["conv_private_dgrad_columns", "linear"]
    assert [
        stage.grid_spec.extents for stage in dgrad_stride2_regular.stages
    ] == [
        (28 * 28, 64 * 3 * 3, 8),
        (8 * 64 * 56 * 56,),
    ]
    assert [stage.default_grid for stage in dgrad_stride2_regular.stages] == [
        (450, 8, 1),
        (6272, 1, 1),
    ]
    assert [stage.stage_name for stage in dgrad_stride2_grouped.stages] == [
        "dgrad_s2_p00",
        "dgrad_s2_p01",
        "dgrad_s2_p10",
        "dgrad_s2_p11",
    ]
    assert [stage.function_name for stage in dgrad_stride2_grouped.stages] == [
        "hygon_conv_dgrad2d_stride2_parity_kernel"
    ] * 4
    assert [stage.stage_name for stage in wgrad_1x1.stages] == [
        "wgrad_1x1_split",
        "wgrad_reduce",
    ]
    wgrad_1x1_split = wgrad_1x1.stages[0]
    assert wgrad_1x1_split.constants["TOTAL_ROWS"] == 6272
    assert wgrad_1x1_split.constants["ROWS_PER_SPLIT"] == 784
    assert any(
        784 % int(configuration["META"]["BLOCK_M"]) != 0
        for configuration in _tuning_configurations(
            compiler,
            select_kernel_candidate,
            "convolution_wgrad",
            wgrad_1x1_split,
        )
    )
    assert [stage.stage_name for stage in wgrad_split.stages] == [
        "wgrad_stem_split",
        "wgrad_reduce",
    ]
    assert [stage.dependencies for stage in wgrad_split.stages] == [
        (),
        ("wgrad_stem_split",),
    ]
    assert [stage.function_name for stage in wgrad_split.stages] == [
        "hygon_conv_wgrad2d_stem_split_kernel",
        "hygon_conv_wgrad2d_reduce_kernel",
    ]
    assert len(wgrad_split.workspace_tensors) == 1
    direct_partial = wgrad_split.workspace_tensors[0]
    assert direct_partial.name == "wgrad_partial"
    assert direct_partial.data_type == "float32"
    assert direct_partial.dimensions == (64, 96, 3 * 3 * 3)
    assert direct_partial.offset == 0
    assert wgrad_split.workspace_size >= direct_partial.size
    assert [stage.stage_name for stage in wgrad_multirow.stages] == [
        "wgrad_multirow_split",
        "wgrad_reduce",
    ]
    assert [stage.function_name for stage in wgrad_multirow.stages] == [
        "hygon_conv_wgrad2d_multirow_split_kernel",
        "hygon_conv_wgrad2d_reduce_kernel",
    ]
    assert [stage.dependencies for stage in wgrad_multirow.stages] == [
        (),
        ("wgrad_multirow_split",),
    ]
    assert len(wgrad_multirow.workspace_tensors) == 1
    multirow_partial = wgrad_multirow.workspace_tensors[0]
    assert multirow_partial.name == "wgrad_partial"
    assert multirow_partial.data_type == "float32"
    assert multirow_partial.dimensions == (8, 128, 64 * 3 * 3)
    assert multirow_partial.strides == (128 * 64 * 3 * 3, 64 * 3 * 3, 1)
    multirow_stage = wgrad_multirow.stages[0]
    assert multirow_stage.tuning.table == "conv_wgrad"
    assert multirow_stage.constants["ROW_PITCH"] == 32
    assert multirow_stage.default_grid == (9, 8, 1)
    multirow_configurations = _tuning_configurations(
        compiler,
        select_kernel_candidate,
        "convolution_wgrad",
        multirow_stage,
    )
    assert len(multirow_configurations) >= 2
    multirow_block_ms: set[int] = set()
    for configuration in multirow_configurations:
        block_m = int(configuration["META"]["BLOCK_M"])
        row_pitch = int(multirow_stage.constants["ROW_PITCH"])
        assert block_m >= row_pitch
        assert block_m % row_pitch == 0
        multirow_block_ms.add(block_m)
    assert {64, 128} <= multirow_block_ms

    assert [stage.stage_name for stage in wgrad_multirow_tail.stages] == [
        "wgrad_multirow_split",
        "wgrad_reduce",
    ]
    tail_partial = wgrad_multirow_tail.workspace_tensors[0]
    assert tail_partial.dimensions == (2, 33, 5 * 3 * 3)
    assert tail_partial.strides == (33 * 5 * 3 * 3, 5 * 3 * 3, 1)
    assert wgrad_multirow_tail.stages[0].default_grid == (1, 2, 1)

    expected_p5_reduction = {
        "n": 128 * 3 * 3,
        "s": 256 * 3 * 3,
        "ml": 512 * 3 * 3,
        "x": 768 * 3 * 3,
    }
    expected_p5_gemm = {
        "n": "hygon_conv_wgrad2d_rowmajor_kernel",
        "s": "hygon_conv_wgrad2d_rowmajor_kernel",
        "ml": "hygon_conv_wgrad2d_p5_block_ptr_kernel",
        "x": "hygon_conv_wgrad2d_p5_block_ptr_kernel",
    }
    p5_pack_keys: set[int] = set()
    for name, plan in wgrad_p5_plans.items():
        assert [stage.stage_name for stage in plan.stages] == [
            "wgrad_im2row",
            "wgrad_rowmajor",
        ]
        assert [stage.function_name for stage in plan.stages] == [
            "hygon_conv_wgrad2d_im2row_kernel",
            expected_p5_gemm[name],
        ]
        assert [stage.dependencies for stage in plan.stages] == [
            (),
            ("wgrad_im2row",),
        ]
        assert len(plan.workspace_tensors) == 1
        columns = plan.workspace_tensors[0]
        reduction_extent = expected_p5_reduction[name]
        assert columns.name == "wgrad_rowmajor_columns"
        assert columns.data_type == "float16"
        assert columns.dimensions == (20 * 20, reduction_extent)
        assert columns.strides == (reduction_extent, 1)
        pack = plan.stages[0]
        assert pack.tuning.table == "conv_wgrad"
        assert pack.tuning_key_value == 20 * 20 * reduction_extent
        p5_pack_keys.add(pack.tuning_key_value)
        gemm = plan.stages[1]
        if name in ("ml", "x"):
            assert gemm.constants["REDUCTION_EXTENT"] == reduction_extent
            assert "DY_STRIDE_H" not in gemm.constants
            assert "DY_STRIDE_W" not in gemm.constants
            assert "OW" not in gemm.constants
        else:
            assert gemm.constants["CIN_PER_GROUP"] in (128, 256)
            assert gemm.constants["OW"] == 20
    assert p5_pack_keys == {
        460800,
        921600,
        1843200,
        2764800,
    }

    assert [stage.stage_name for stage in wgrad_p5_tail.stages] == [
        "wgrad_im2row",
        "wgrad_rowmajor",
    ]
    assert wgrad_p5_tail.stages[1].function_name == (
        "hygon_conv_wgrad2d_rowmajor_kernel"
    )
    p5_tail_columns = wgrad_p5_tail.workspace_tensors[0]
    assert p5_tail_columns.dimensions == (20 * 20, 5 * 3 * 3)
    assert p5_tail_columns.strides == (5 * 3 * 3, 1)

    assert [stage.function_name for stage in wgrad_multirow_flip.stages] == [
        "hygon_conv_wgrad2d_direct_split_kernel",
        "hygon_conv_wgrad2d_reduce_kernel",
    ]
    assert wgrad_multirow_flip.stages[1].constants["FLIP_FILTER"] is True
    assert [stage.function_name for stage in wgrad_p5_flip.stages] == [
        "hygon_conv2d_im2col_nchw_kernel",
        "hygon_conv_wgrad2d_im2col_kernel",
    ]
    assert wgrad_p5_flip.stages[1].constants["FLIP_FILTER"] is True
    for fallback in (wgrad_grouped, wgrad_noncontiguous):
        assert [stage.function_name for stage in fallback.stages] == [
            "conv_wgrad_nd_kernel"
        ]
        assert fallback.workspace_size == 0

    wgrad_dot_right_meta = {
        "conv_wgrad_nd_kernel": "BLOCK_CI",
        "hygon_conv_wgrad2d_1x1_split_kernel": "BLOCK_CI",
        "hygon_conv_wgrad2d_direct_split_kernel": "BLOCK_CI_K",
        "hygon_conv_wgrad2d_im2col_kernel": "BLOCK_CI_K",
        "hygon_conv_wgrad2d_im2col_split_kernel": "BLOCK_CI_K",
        "hygon_conv_wgrad2d_multirow_split_kernel": "BLOCK_CI_K",
        "hygon_conv_wgrad2d_p5_block_ptr_kernel": "BLOCK_CI_K",
        "hygon_conv_wgrad2d_rowmajor_kernel": "BLOCK_CI_K",
        "hygon_conv_wgrad2d_stem_split_kernel": "BLOCK_CI_K",
    }
    assert compiler._WGRAD_DOT_RIGHT_META == wgrad_dot_right_meta
    assert compiler.HYGON_MAX_SHARED_MEMORY_BYTES == 64 * 1024

    def collect_wgrad_dot_stages(plans: tuple[Any, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for plan in plans:
            for stage in plan.stages:
                if stage.function_name not in wgrad_dot_right_meta:
                    continue
                assert stage.function_name not in result
                result[stage.function_name] = stage
        assert set(result) == set(wgrad_dot_right_meta)
        return result

    fp32_dot_stages = collect_wgrad_dot_stages(
        tuple(wgrad_fp32_resource_plans)
    )
    fp16_dot_stages = collect_wgrad_dot_stages(
        (
            wgrad_grouped,
            wgrad_1x1,
            wgrad_multirow_flip,
            wgrad_p5_flip,
            wgrad_packed_split,
            wgrad_split,
            wgrad_multirow,
            wgrad_p5_plans["n"],
            wgrad_p5_plans["x"],
        )
    )
    for function_name, stage in fp32_dot_stages.items():
        assert stage.runtime_signature["dy_ptr"] == "*fp32"
        right_meta = wgrad_dot_right_meta[function_name]
        configurations = _tuning_configurations(
            compiler,
            select_kernel_candidate,
            "convolution_wgrad",
            stage,
        )
        assert len(configurations) >= 2
        for configuration in configurations:
            meta = configuration["META"]
            required_bytes = (
                int(meta["BLOCK_M"])
                * (int(meta["BLOCK_OC"]) + int(meta[right_meta]))
                * 4
            )
            assert required_bytes <= 64 * 1024
            assert compiler._wgrad_dot_tuning_fits_shared_memory(stage, meta)

    p5_large_configurations = _tuning_configurations(
        compiler,
        select_kernel_candidate,
        "convolution_wgrad",
        wgrad_p5_plans["x"].stages[1],
    )
    p5_large_tiles = {
        (
            int(configuration["META"]["BLOCK_OC"]),
            int(configuration["META"]["BLOCK_CI_K"]),
            int(configuration["META"]["BLOCK_M"]),
            int(configuration["num_warps"]),
            int(configuration["num_stages"]),
        )
        for configuration in p5_large_configurations
    }
    assert (128, 64, 32, 8, 2) in p5_large_tiles
    assert (128, 128, 32, 8, 2) in p5_large_tiles

    for function_name, stage in fp16_dot_stages.items():
        assert stage.runtime_signature["dy_ptr"] == "*fp16"
        right_meta = wgrad_dot_right_meta[function_name]
        configurations = _tuning_configurations(
            compiler,
            select_kernel_candidate,
            "convolution_wgrad",
            stage,
        )
        assert len(configurations) >= 2
        expected_right = 32 if right_meta == "BLOCK_CI" else 64
        assert any(
            configuration["META"]["BLOCK_M"] == 128
            and configuration["META"]["BLOCK_OC"] == 128
            and configuration["META"][right_meta] == expected_right
            for configuration in configurations
        )

    sample_stage = fp32_dot_stages["hygon_conv_wgrad2d_im2col_kernel"]
    sample_meta = dict(
        _tuning_configurations(
            compiler,
            select_kernel_candidate,
            "convolution_wgrad",
            sample_stage,
        )[0]["META"]
    )
    missing_meta = dict(sample_meta)
    missing_meta.pop("BLOCK_M")
    try:
        compiler._wgrad_dot_tuning_fits_shared_memory(
            sample_stage, missing_meta
        )
    except ValueError as error:
        assert "META is missing" in str(error)
    else:
        raise AssertionError("WGrad LDS contract accepted missing META")
    invalid_token_stage = type(
        "InvalidWGradDotStage",
        (),
        {
            "function_name": sample_stage.function_name,
            "operation": sample_stage.operation,
            "runtime_signature": {"dy_ptr": "*invalid"},
        },
    )()
    try:
        compiler._wgrad_dot_tuning_fits_shared_memory(
            invalid_token_stage, sample_meta
        )
    except ValueError as error:
        assert "dy_ptr ABI token" in str(error)
    else:
        raise AssertionError("WGrad LDS contract accepted an invalid token")
    assert [stage.stage_name for stage in wgrad_packed_split.stages] == [
        "wgrad_im2col",
        "wgrad_split",
        "wgrad_reduce",
    ]
    assert [stage.dependencies for stage in wgrad_packed_split.stages] == [
        (),
        ("wgrad_im2col",),
        ("wgrad_split",),
    ]

    specialized_plans = (
        fprop_1x1,
        fprop_low_ci,
        fprop_packed,
        fprop_standard,
        fprop_standard_alias,
        fprop_yolo_x,
        dgrad_1x1,
        dgrad_stride1,
        dgrad_exact,
        dgrad_exact_bf16,
        dgrad_stride2,
        dgrad_stride2_c5,
        dgrad_stride2_flip,
        dgrad_stride2_regular,
        dgrad_stride2_grouped,
        wgrad_1x1,
        wgrad_split,
        wgrad_multirow,
        wgrad_multirow_tail,
        wgrad_packed_split,
        *tuple(wgrad_p5_plans.values()),
        wgrad_p5_tail,
        wgrad_multirow_flip,
        wgrad_p5_flip,
    )
    seen: set[str] = set()
    for plan in specialized_plans:
        plan.validate_dependencies()
        _assert_workspace_layout(plan)
        candidate = select_kernel_candidate("hygon", plan.operation)
        assert candidate.ownership == "platform"
        assert candidate.provider == "hygon_triton"
        assert candidate.source == "convolution.py"
        assert (
            candidate.functions
            == compiler_nn.REGISTRY_FUNCTIONS[plan.operation]
        )
        for stage in plan.stages:
            if not stage.function_name.startswith("hygon_"):
                continue
            seen.add(stage.function_name)
            assert stage.tuning.source == "convolution.yaml"
            assert stage.grid_spec.kind != "fixed"
            configurations = _tuning_configurations(
                compiler,
                select_kernel_candidate,
                plan.operation,
                stage,
            )
            assert len(configurations) >= 2
            grids: set[tuple[int, int, int]] = set()
            for configuration in configurations:
                constants, grid = stage.variant(
                    configuration["META"],
                    num_warps=configuration["num_warps"],
                    num_stages=configuration["num_stages"],
                )
                assert grid == stage.grid_spec.evaluate(constants)
                grids.add(grid)
            assert len(grids) >= 2, stage.stage_name
    expected = {
        function
        for operation in compiler_nn.CONVOLUTION_OPERATIONS
        for function in compiler_nn.REGISTRY_FUNCTIONS[operation]
        if function.startswith("hygon_")
    }
    assert seen == expected

    # The parity kernel has four logical domains. A dimension smaller than two
    # would create an empty domain, so it must remain on the generic fallback.
    _, tiny_dgrad = _convolution_case(
        compiler_nn,
        "convolution_dgrad",
        [1, 3, 1, 2],
        [4, 3, 3, 3],
        [2, 2],
        [1, 1],
    )
    assert [stage.function_name for stage in tiny_dgrad.stages] == [
        "conv_dgrad_nd_kernel"
    ]

    # Columns alone fit below 512 MiB, but columns plus 64 FP32 partials do
    # not. The total cap must select the generic zero-workspace fallback.
    _, capped_wgrad = _convolution_case(
        compiler_nn,
        "convolution_wgrad",
        [1, 64, 672, 672],
        [128, 64, 3, 3],
        [1, 1],
        [1, 1],
    )
    assert [stage.function_name for stage in capped_wgrad.stages] == [
        "conv_wgrad_nd_kernel"
    ]
    assert capped_wgrad.workspace_size == 0

    # Generic paths are still platform-owned and use the per-direction Hygon
    # table rather than silently falling back to common.yaml.
    _, generic_fprop = _convolution_case(
        compiler_nn,
        "conv2d_fprop",
        [1, 2, 5, 6, 7],
        [4, 2, 3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )
    for plan in (
        generic_fprop,
        tiny_dgrad,
        capped_wgrad,
        wgrad_grouped,
        wgrad_noncontiguous,
    ):
        stage = plan.stages[0]
        assert stage.tuning.source == "convolution.yaml"
        configurations = _tuning_configurations(
            compiler,
            select_kernel_candidate,
            plan.operation,
            stage,
        )
        assert len(configurations) >= 2


_ATTENTION_TILE_CONSTANTS = (
    "BLOCK_D",
    "BLOCK_DV",
    "BLOCK_D_FULL",
    "BLOCK_D_OUT",
    "BLOCK_DV_OUT",
)


def _assert_attention_tile_floor(plan: Any) -> None:
    seen: set[str] = set()
    for stage in plan.stages:
        for name in _ATTENTION_TILE_CONSTANTS:
            if name not in stage.constants:
                continue
            seen.add(name)
            assert stage.constants[name] >= 16, (
                stage.stage_name,
                name,
                stage.constants[name],
            )
    assert seen


def _attention_attributes(
    *,
    heads: int,
    key_heads: int,
    value_heads: int,
    sequence_q: int,
    sequence_kv: int,
    head_dimension: int,
    value_dimension: int,
) -> dict[str, Any]:
    return {
        "batch": 1,
        "heads": heads,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "sequence_q": sequence_q,
        "sequence_kv": sequence_kv,
        "head_dimension": head_dimension,
        "value_dimension": value_dimension,
        "q_per_k": heads // key_heads,
        "q_per_v": heads // value_heads,
        "has_bias": 0,
        "has_dbias": 0,
        "min_diag": -(1 << 30),
        "max_diag": 1 << 30,
        "banded": 0,
        "causal_top_left": 0,
        "reverse_causal": 0,
        "generate_stats": 1,
        "attn_scale": 0.25,
    }


def _attention_specifications(
    operation: str,
    *,
    heads: int,
    key_heads: int,
    value_heads: int,
    sequence_q: int,
    sequence_kv: int,
    head_dimension: int,
    value_dimension: int,
) -> dict[str, tuple[str, list[int]]]:
    q_shape = [1, heads, sequence_q, head_dimension]
    k_shape = [1, key_heads, sequence_kv, head_dimension]
    v_shape = [1, value_heads, sequence_kv, value_dimension]
    o_shape = [1, heads, sequence_q, value_dimension]
    specifications = {
        "q": ("float16", q_shape),
        "k": ("float16", k_shape),
        "v": ("float16", v_shape),
        "o": ("float16", o_shape),
        "stats": ("float32", [1, heads, sequence_q, 1]),
    }
    if operation == "sdpa_backward":
        specifications.update(
            {
                "do": ("float16", o_shape),
                "dq": ("float16", q_shape),
                "dk": ("float16", k_shape),
                "dv": ("float16", v_shape),
            }
        )
    else:
        assert operation == "sdpa"
    return specifications


def _expect_attention_error(
    compiler_nn: ModuleType,
    operation: str,
    specifications: dict[str, tuple[str, list[int]]],
    attributes: dict[str, Any],
    message: str,
) -> None:
    try:
        _parse_and_plan(compiler_nn, operation, specifications, attributes)
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(
            f"{operation} should reject invalid attention metadata"
        )


def _check_attention_planner_contracts(
    compiler: ModuleType,
    compiler_nn: ModuleType,
    select_kernel_candidate: Any,
) -> None:
    tail_arguments = {
        "heads": 2,
        "key_heads": 2,
        "value_heads": 2,
        "sequence_q": 33,
        "sequence_kv": 35,
        "head_dimension": 8,
        "value_dimension": 8,
    }
    tail_attributes = _attention_attributes(**tail_arguments)
    _, tail_plan = _parse_and_plan(
        compiler_nn,
        "sdpa_backward",
        _attention_specifications("sdpa_backward", **tail_arguments),
        tail_attributes,
    )
    _assert_attention_tile_floor(tail_plan)
    full_attention_stages = [
        stage
        for stage in tail_plan.stages
        if "FULL_ATTENTION" in stage.constants
    ]
    assert [stage.stage_name for stage in full_attention_stages] == [
        "dq_delta_dbias",
        "dk_dv",
    ]
    for stage in full_attention_stages:
        assert stage.constants["FULL_ATTENTION"] is False
        for configuration in _tuning_configurations(
            compiler, select_kernel_candidate, "sdpa_backward", stage
        ):
            constants, _ = stage.variant(
                configuration["META"],
                num_warps=configuration["num_warps"],
                num_stages=configuration["num_stages"],
            )
            assert constants["FULL_ATTENTION"] is False
            for name in _ATTENTION_TILE_CONSTANTS:
                if name in constants:
                    assert constants[name] >= 16

    _, small_forward_plan = _parse_and_plan(
        compiler_nn,
        "sdpa",
        _attention_specifications("sdpa", **tail_arguments),
        tail_attributes,
    )
    _assert_attention_tile_floor(small_forward_plan)

    split_arguments = {**tail_arguments, "value_dimension": 4}
    _, split_backward_plan = _parse_and_plan(
        compiler_nn,
        "sdpa_backward",
        _attention_specifications("sdpa_backward", **split_arguments),
        _attention_attributes(**split_arguments),
    )
    assert [stage.stage_name for stage in split_backward_plan.stages] == [
        "dq_delta_dbias",
        "dk",
        "dv",
    ]
    _assert_attention_tile_floor(split_backward_plan)

    for dimensions in (
        {"head_dimension": 257, "value_dimension": 8},
        {"head_dimension": 8, "value_dimension": 257},
    ):
        oversized_arguments = {**tail_arguments, **dimensions}
        _expect_attention_error(
            compiler_nn,
            "sdpa",
            _attention_specifications("sdpa", **oversized_arguments),
            _attention_attributes(**oversized_arguments),
            "head dimensions greater than 256",
        )

    grouped_query_arguments = {
        **tail_arguments,
        "heads": 8,
        "key_heads": 2,
        "value_heads": 4,
    }
    grouped_query_parsed, grouped_query_forward = _parse_and_plan(
        compiler_nn,
        "sdpa",
        _attention_specifications("sdpa", **grouped_query_arguments),
        _attention_attributes(**grouped_query_arguments),
    )
    assert grouped_query_parsed["derived"]["q_per_k"] == 4
    assert grouped_query_parsed["derived"]["q_per_v"] == 2
    _assert_attention_tile_floor(grouped_query_forward)
    _expect_attention_error(
        compiler_nn,
        "sdpa_backward",
        _attention_specifications("sdpa_backward", **grouped_query_arguments),
        _attention_attributes(**grouped_query_arguments),
        "matching K/V head counts",
    )


def _check_fp8_backward(
    compiler: ModuleType,
    compiler_nn: ModuleType,
    select_kernel_candidate: Any,
) -> None:
    batch = 1
    heads = 1
    sequence_q = 5
    sequence_kv = 7
    head_dimension = 8
    q_shape = [batch, heads, sequence_q, head_dimension]
    kv_shape = [batch, heads, sequence_kv, head_dimension]
    stats_shape = [batch, heads, sequence_q, 1]

    fp8 = "fp8_e4m3"
    fp32 = "float32"
    scalar_names = (
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
    specifications = {
        "q": (fp8, q_shape),
        "k": (fp8, kv_shape),
        "v": (fp8, kv_shape),
        "o": (fp8, q_shape),
        "do": (fp8, q_shape),
        "stats": (fp32, stats_shape),
        "dq": (fp8, q_shape),
        "dk": (fp8, kv_shape),
        "dv": (fp8, kv_shape),
        **{name: (fp32, [1]) for name in scalar_names},
    }
    attributes = {
        "batch": batch,
        "heads": heads,
        "key_heads": heads,
        "value_heads": heads,
        "sequence_q": sequence_q,
        "sequence_kv": sequence_kv,
        "head_dimension": head_dimension,
        "value_dimension": head_dimension,
        "q_per_k": 1,
        "q_per_v": 1,
        "has_bias": 0,
        "has_dbias": 0,
        "min_diag": -(1 << 30),
        "max_diag": 1 << 30,
        "banded": 0,
        "causal_top_left": 0,
        "reverse_causal": 0,
        "generate_stats": 1,
        "attn_scale": 0.25,
    }
    parsed, plan = _parse_and_plan(
        compiler_nn,
        "sdpa_fp8_backward",
        specifications,
        attributes,
    )
    assert parsed["derived"]["sequence_q"] == sequence_q
    assert parsed["derived"]["sequence_kv"] == sequence_kv
    assert parsed["derived"]["head_dimension"] == head_dimension
    _assert_attention_tile_floor(plan)
    _assert_mixed_stage_emission(compiler, parsed)

    relevant = [
        stage for stage in plan.stages if "FULL_BLOCKS" in stage.constants
    ]
    assert [stage.stage_name for stage in relevant] == ["dq", "dk_dv"]
    for stage in relevant:
        assert stage.constants["FULL_BLOCKS"] is False
        configurations = _tuning_configurations(
            compiler,
            select_kernel_candidate,
            "sdpa_fp8_backward",
            stage,
        )
        assert len(configurations) == 4
        for configuration in configurations:
            constants, _ = stage.variant(
                configuration["META"],
                num_warps=configuration["num_warps"],
                num_stages=configuration["num_stages"],
            )
            assert constants["FULL_BLOCKS"] is False


def _normalization_specifications(
    operation: str,
) -> dict[str, tuple[str, list[int]]]:
    data_type = "float16"
    result = {
        "x": (data_type, [2, 8192]),
        "scale": (data_type, [8192]),
        "bias": (data_type, [8192]),
        "y": (data_type, [2, 8192]),
        "inv_variance": ("float32", [2, 1]),
    }
    if operation == "layernorm":
        result["mean"] = ("float32", [2, 1])
    return result


def _check_large_normalization(
    compiler: ModuleType,
    compiler_nn: ModuleType,
    select_kernel_candidate: Any,
) -> None:
    layernorm_candidate = select_kernel_candidate("hygon", "layernorm")
    assert layernorm_candidate.ownership == "platform"
    assert layernorm_candidate.provider == "hygon_triton"
    assert layernorm_candidate.source_layout == "platform"
    assert layernorm_candidate.source == "normalization.py"
    assert layernorm_candidate.functions == ("layer_norm_kernel",)
    normalization_source = compiler.resolve_kernel_source(
        compiler._compiler_entry_path(), layernorm_candidate
    )
    assert normalization_source.parent.name == "kernels"
    assert normalization_source.parent.parent.name == "hygon"
    source_text = normalization_source.read_text(encoding="utf-8")
    assert "def layer_norm_kernel(" in source_text
    assert "running_m2" in source_text
    assert "batch_m2" in source_text
    assert "sum_squares" not in source_text

    attributes = {
        "rows": 2,
        "normalized_elements": 8192,
        "epsilon": 1.0e-5,
        "forward_phase": 2,
    }
    for operation in ("layernorm", "rmsnorm"):
        _, plan = _parse_and_plan(
            compiler_nn,
            operation,
            _normalization_specifications(operation),
            attributes,
        )
        assert len(plan.stages) == 1
        stage = plan.stages[0]
        candidate = select_kernel_candidate("hygon", operation)
        source = compiler.resolve_kernel_source(
            compiler._compiler_entry_path(), candidate
        )
        function = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == stage.function_name
        )
        signature = type(
            "KernelSignature",
            (),
            {"arg_names": [argument.arg for argument in function.args.args]},
        )()
        compiler._jit_full_signature(
            signature, stage.runtime_signature, stage.constants
        )
        configurations = _tuning_configurations(
            compiler, select_kernel_candidate, operation, stage
        )
        assert len(configurations) == 10
        for configuration in configurations:
            stage.variant(
                configuration["META"],
                num_warps=configuration["num_warps"],
                num_stages=configuration["num_stages"],
            )


def _batchnorm_case(
    compiler_nn: ModuleType, batch: int
) -> tuple[dict[str, Any], Any]:
    channels = 4
    spatial = 2
    dimensions = [batch, channels, spatial]
    parameter_shape = [channels]
    specifications = {
        "x": ("float32", dimensions),
        "scale": ("float32", parameter_shape),
        "bias": ("float32", parameter_shape),
        "previous_running_mean": ("float32", parameter_shape),
        "previous_running_variance": ("float32", parameter_shape),
        "y": ("float32", dimensions),
        "mean": ("float32", parameter_shape),
        "inv_variance": ("float32", parameter_shape),
        "next_running_mean": ("float32", parameter_shape),
        "next_running_variance": ("float32", parameter_shape),
    }
    strides = _contiguous_strides(dimensions)
    attributes = {
        "batch": batch,
        "channels": channels,
        "spatial": spatial,
        "n_elements": batch * channels * spatial,
        "rank": len(dimensions),
        "dimensions": dimensions,
        "x_strides": strides,
        "y_strides": strides,
        "epsilon": 1.0e-5,
        "momentum": 0.1,
    }
    return _parse_and_plan(
        compiler_nn, "batchnorm", specifications, attributes
    )


def _check_batchnorm_dispatch(
    compiler: ModuleType,
    compiler_nn: ModuleType,
    select_kernel_candidate: Any,
) -> None:
    candidate = select_kernel_candidate("hygon", "batchnorm")
    assert candidate.ownership == "platform"
    assert candidate.provider == "hygon_triton"
    assert candidate.source_layout == "platform"
    assert candidate.source == "normalization.py"
    assert candidate.functions == (
        "batch_norm_nchw_kernel",
        "batch_norm_kernel",
    )

    _, nchw_plan = _batchnorm_case(compiler_nn, 512)
    assert len(nchw_plan.stages) == 1
    nchw_stage = nchw_plan.stages[0]
    assert nchw_stage.function_name == "batch_norm_nchw_kernel"
    nchw_configurations = _tuning_configurations(
        compiler, select_kernel_candidate, "batchnorm", nchw_stage
    )
    assert len(nchw_configurations) == 5
    assert [
        configuration["META"]["BLOCK_SIZE"]
        for configuration in nchw_configurations
    ] == [512, 1024, 1024, 2048, 4096]

    _, generic_plan = _batchnorm_case(compiler_nn, 1024)
    assert len(generic_plan.stages) == 1
    generic_stage = generic_plan.stages[0]
    assert generic_stage.function_name == "batch_norm_kernel"
    generic_configurations = _tuning_configurations(
        compiler, select_kernel_candidate, "batchnorm", generic_stage
    )
    assert len(generic_configurations) == 6
    assert [
        configuration["META"]["BLOCK_SIZE"]
        for configuration in generic_configurations
    ] == [256, 512, 1024, 1024, 2048, 4096]


def _check_sigmoid_backward_common_source(compiler: ModuleType) -> None:
    # Sigmoid backward shares the activation-gradient module. Exercise actual
    # source materialization and the complete JIT ABI without a device SDK.
    with tempfile.TemporaryDirectory(prefix="flagdnn-hygon-sigmoid-") as path:
        for dtype in ("float32", "float16", "bfloat16"):
            for strided in (False, True):
                registry = {
                    uid: _tensor(uid, dtype, [2, 3]) for uid in (1, 2, 3)
                }
                if strided:
                    for tensor in registry.values():
                        tensor["strides"] = [8, 2]
                node = {
                    "id": 0,
                    "type": "sigmoid_backward",
                    "compute_data_type": "float32",
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": 40,
                        "alpha": 1.0,
                    },
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                }
                parsed = compiler._parse_pointwise_node(node, 0, 1, registry)
                stage = compiler._compile_pointwise_stage(
                    stage_id=0,
                    node=parsed,
                    dependencies=[],
                    workspace={},
                    compiler_path=compiler._compiler_entry_path(),
                    output_directory=Path(path),
                    enable_autotune=False,
                )
                assert stage["kernel"]["source"] == "activation_backward.py"
                expected = (
                    "activation_backward_strided_kernel"
                    if strided
                    else "activation_backward_contiguous_kernel"
                )
                assert stage["kernel"]["function"] == expected


def _check_pointwise_kernel_ownership(
    compiler: ModuleType, select_kernel_candidate: Any
) -> None:
    add = select_kernel_candidate("hygon", "add")
    assert add.ownership == "common"
    assert add.provider == "common_triton"
    assert add.source_layout == "kernels"
    assert add.source == "binary.py"
    assert (
        compiler.resolve_kernel_source(
            compiler._compiler_entry_path(), add
        ).name
        == "binary.py"
    )

    for operation in ("min", "max"):
        candidate = select_kernel_candidate("hygon", operation)
        assert candidate.ownership == "platform"
        assert candidate.provider == "hygon_triton"
        assert candidate.source_layout == "platform"
        assert candidate.source == "binary_minmax.py"
        source = compiler.resolve_kernel_source(
            compiler._compiler_entry_path(), candidate
        )
        assert source.name == "binary_minmax.py"
        assert source.parent.name == "kernels"
        assert source.parent.parent.name == "hygon"

    identity = select_kernel_candidate("hygon", "identity")
    assert identity.ownership == "platform"
    assert identity.provider == "hygon_triton"
    assert identity.source_layout == "platform"
    assert identity.source == "unary.py"
    assert "identity_packed_contiguous_kernel" in identity.functions

    registry = {uid: _tensor(uid, "float32", [2, 3]) for uid in (1, 2, 3)}
    expected = {
        "add": (1, "common", "common_triton", "binary.py"),
        "min": (20, "platform", "hygon_triton", "binary_minmax.py"),
        "max": (21, "platform", "hygon_triton", "binary_minmax.py"),
    }
    with tempfile.TemporaryDirectory(
        prefix="flagdnn-hygon-pointwise-contract-"
    ) as temporary:
        for stage_id, (operation, contract) in enumerate(expected.items()):
            mode, ownership, provider, source_name = contract
            node = {
                "id": 0,
                "type": operation,
                "compute_data_type": "float32",
                "attributes": {
                    "n_elements": 6,
                    "pointwise_mode": mode,
                    "alpha": 1.0,
                },
                "inputs": [
                    {"name": "left", "uid": 1, "optional": False},
                    {"name": "right", "uid": 2, "optional": False},
                ],
                "outputs": [{"name": "output", "uid": 3, "optional": False}],
            }
            parsed = compiler._parse_pointwise_node(node, 0, 1, registry)
            stage = compiler._compile_pointwise_stage(
                stage_id=stage_id,
                node=parsed,
                dependencies=[],
                workspace={},
                compiler_path=compiler._compiler_entry_path(),
                output_directory=Path(temporary),
                enable_autotune=False,
            )
            assert stage["kernel"]["ownership"] == ownership
            assert stage["kernel"]["provider"] == provider
            assert stage["kernel"]["source"] == source_name
            assert (
                stage["source_sha256"]
                == stage["kernel"]["materialized_source"]["sha256"]
            )

        identity_registry = {
            uid: _tensor(uid, "float32", [8, 1024, 2048]) for uid in (1, 2)
        }
        identity_node = {
            "id": 0,
            "type": "identity",
            "compute_data_type": "float32",
            "attributes": {
                "n_elements": 8 * 1024 * 2048,
                "has_upper_clip": 0,
            },
            "inputs": [{"name": "input", "uid": 1, "optional": False}],
            "outputs": [{"name": "output", "uid": 2, "optional": False}],
        }
        parsed_identity = compiler._parse_pointwise_node(
            identity_node, 0, 1, identity_registry
        )
        identity_stage = compiler._compile_pointwise_stage(
            stage_id=4,
            node=parsed_identity,
            dependencies=[],
            workspace={},
            compiler_path=compiler._compiler_entry_path(),
            output_directory=Path(temporary),
            enable_autotune=True,
        )
        assert identity_stage["kernel"]["function"] == (
            "identity_packed_contiguous_kernel"
        )
        variants = identity_stage["variants"]
        assert len(variants) == 18
        assert all(
            set(variant["config"]["META"])
            == {"BLOCK_SIZE", "TILES_PER_PROGRAM"}
            for variant in variants
        )
        for variant in variants:
            meta = variant["config"]["META"]
            elements_per_program = (
                meta["BLOCK_SIZE"] * meta["TILES_PER_PROGRAM"] * 2
            )
            assert (
                variant["launch"]["grid"][0]
                == (8 * 1024 * 2048 + elements_per_program - 1)
                // elements_per_program
            )

        strided_registry = {
            uid: _tensor(uid, "float32", [8, 1024, 2048]) for uid in (1, 2)
        }
        strided_registry[1]["strides"] = [1024 * 2049, 2049, 1]
        parsed_strided = compiler._parse_pointwise_node(
            identity_node, 0, 1, strided_registry
        )
        strided_stage = compiler._compile_pointwise_stage(
            stage_id=5,
            node=parsed_strided,
            dependencies=[],
            workspace={},
            compiler_path=compiler._compiler_entry_path(),
            output_directory=Path(temporary),
            enable_autotune=True,
        )
        assert strided_stage["kernel"]["function"] == (
            "unary_pointwise_strided_kernel"
        )
        assert len(strided_stage["variants"]) == 18
        for variant in strided_stage["variants"]:
            meta = variant["config"]["META"]
            elements_per_program = (
                meta["BLOCK_SIZE"] * meta["TILES_PER_PROGRAM"]
            )
            assert (
                variant["launch"]["grid"][0]
                == (8 * 1024 * 2048 + elements_per_program - 1)
                // elements_per_program
            )


def _check_platform_kernel_identity_inputs(compiler: ModuleType) -> None:
    from backends.hygon import compiler_identity

    inputs = compiler_identity._identity_inputs(
        Path(compiler.__file__).resolve(), compiler._compiler_entry_path()
    )
    common_normalization = (
        "kernel:common:kernels:common_triton:normalization.py"
    )
    platform_normalization = (
        "kernel:platform:platform:hygon_triton:normalization.py"
    )
    common_unary = "kernel:common:kernels:common_triton:unary.py"
    platform_unary = "kernel:platform:platform:hygon_triton:unary.py"
    for label in (
        common_normalization,
        platform_normalization,
        common_unary,
        platform_unary,
        "kernel:platform:platform:hygon_triton:reduction.py",
        "kernel:platform:platform:hygon_triton:convolution.py",
    ):
        assert label in inputs
        assert inputs[label].is_file()
    assert inputs[common_normalization] != inputs[platform_normalization]
    assert inputs[common_unary] != inputs[platform_unary]


def _write_compiler_environment_fixture(
    compiler_identity: ModuleType,
    root: Path,
    *,
    legacy_layout: bool,
    installed_layout: bool = False,
) -> tuple[Path, Path, Path, dict[str, Any], str]:
    provider_directory = (
        root / "share/flagdnn/backends/hygon"
        if installed_layout
        else root / "backends/hygon"
    )
    provider_directory.mkdir(parents=True)
    provider = provider_directory / "compiler_identity.py"
    provider.write_text(
        "# compiler identity layout contract\n", encoding="utf-8"
    )
    environment_source = provider.with_name(
        "flagdnn_hygon_compiler_environment.json"
    )
    soname = "libtriton_jit.so.987654321"

    if legacy_layout:
        library = provider_directory / soname
        script_directory = (
            provider_directory.parent / "share/triton_jit/scripts"
        )
    else:
        if installed_layout:
            library = root / "lib/flagdnn/hygon" / soname
            script_directory = root / "lib/flagdnn/share/triton_jit/scripts"
        else:
            library = provider_directory / "flagdnn/hygon" / soname
            script_directory = (
                provider_directory / "flagdnn/share/triton_jit/scripts"
            )

    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_bytes(b"contract-private-libtriton-jit\n")
    script_directory.mkdir(parents=True, exist_ok=True)
    script_contents = {
        "standalone_compile.py": b"# standalone compile contract\n",
        "gen_ssig.py": b"# signature generator contract\n",
        "flagdnn_python_environment_identity.py": (
            b"# Python environment identity contract\n"
        ),
    }
    for name, contents in script_contents.items():
        (script_directory / name).write_bytes(contents)

    standalone_sha256 = compiler_identity._sha256_file(
        script_directory / "standalone_compile.py"
    )
    gen_ssig_sha256 = compiler_identity._sha256_file(
        script_directory / "gen_ssig.py"
    )
    helper_sha256 = compiler_identity._sha256_file(
        script_directory / "flagdnn_python_environment_identity.py"
    )
    scripts_sha256 = compiler_identity._script_bundle_sha256(
        standalone_sha256,
        gen_ssig_sha256,
        helper_sha256,
    )
    runtime_environment_sha256 = hashlib.sha256(
        b"compiler-contract-runtime-environment"
    ).hexdigest()
    library_sha256 = compiler_identity._sha256_file(library)
    provenance_sha256 = hashlib.sha256(
        b"compiler-contract-triton-jit-provenance"
    ).hexdigest()
    environment = {
        "schema_version": 3,
        "libtriton_jit_soname": soname,
        "libtriton_jit_install_relative_path": (f"lib/flagdnn/hygon/{soname}"),
        "jit_script_install_relative_path": (
            "lib/flagdnn/share/triton_jit/scripts"
        ),
        "libtriton_jit_sha256": library_sha256,
        "triton_jit_provenance_sha256": provenance_sha256,
        "python_environment_sha256": runtime_environment_sha256,
        "python_environment_helper_sha256": helper_sha256,
        "standalone_compile_sha256": standalone_sha256,
        "gen_ssig_sha256": gen_ssig_sha256,
        "jit_scripts_sha256": scripts_sha256,
        "build_identity": (
            f"{provenance_sha256[:16]}-{runtime_environment_sha256[:16]}-"
            f"{scripts_sha256[:16]}"
        ),
    }
    environment_source.write_text(
        json.dumps(environment, sort_keys=True), encoding="utf-8"
    )
    return (
        provider,
        library,
        script_directory,
        environment,
        runtime_environment_sha256,
    )


def _check_private_build_mirror_layout(compiler: ModuleType) -> None:
    from backends.hygon import compiler_identity

    with _isolated_jit_discovery_environment(), tempfile.TemporaryDirectory(
        prefix="flagdnn-hygon-compiler-layout-contract-"
    ) as temporary:
        temporary_root = Path(temporary)
        (
            provider,
            library,
            script_directory,
            environment,
            runtime_environment_sha256,
        ) = _write_compiler_environment_fixture(
            compiler_identity,
            temporary_root / "private",
            legacy_layout=False,
        )
        environment_source = provider.with_name(
            "flagdnn_hygon_compiler_environment.json"
        )
        validated = compiler_identity._validate_compiler_environment(
            environment,
            environment_source,
            runtime_environment_sha256,
        )

        assert validated["origin"] == "cmake"
        assert validated["triton_jit_provenance_sha256"] == (
            environment["triton_jit_provenance_sha256"]
        )
        assert (
            validated["verified_library_sha256"]
            == environment["libtriton_jit_sha256"]
        )
        assert validated["verified_private_jit_script_directory"] == str(
            script_directory.resolve()
        )
        assert validated["libtriton_jit_install_relative_path"] == (
            environment["libtriton_jit_install_relative_path"]
        )
        assert validated["jit_script_install_relative_path"] == (
            environment["jit_script_install_relative_path"]
        )

        missing_provenance = dict(environment)
        del missing_provenance["triton_jit_provenance_sha256"]
        try:
            compiler_identity._validate_compiler_environment(
                missing_provenance,
                environment_source,
                runtime_environment_sha256,
            )
        except ValueError as error:
            assert "triton_jit_provenance_sha256" in str(error)
        else:
            raise AssertionError(
                "configured compiler identity must require JIT provenance"
            )

        unconsumed_provenance = dict(environment)
        unconsumed_provenance["triton_jit_provenance_sha256"] = "0" * 64
        try:
            compiler_identity._validate_compiler_environment(
                unconsumed_provenance,
                environment_source,
                runtime_environment_sha256,
            )
        except ValueError as error:
            assert "build_identity" in str(error)
        else:
            raise AssertionError(
                "configured build identity must consume JIT provenance"
            )

        library_candidates = set(
            compiler_identity._libtriton_jit_candidates(provider)
        )
        script_candidates = {
            path.resolve()
            for path in compiler_identity._jit_script_directory_candidates(
                provider
            )
        }
        assert library.resolve() in library_candidates
        assert script_directory.resolve() in script_candidates
        assert (provider.parent / library.name).resolve() not in (
            library_candidates
        )

        dependencies = set(
            compiler_identity.compiler_identity_dependency_paths(
                provider_path=provider,
                compiler_entry=compiler._compiler_entry_path(),
            )
        )
        expected_dependencies = {
            environment_source.resolve(),
            library.resolve(),
            script_directory.resolve(),
            *(path.resolve() for path in script_directory.iterdir()),
        }
        assert expected_dependencies <= dependencies

        (
            installed_provider,
            installed_library,
            installed_script_directory,
            installed_environment,
            installed_runtime_environment_sha256,
        ) = _write_compiler_environment_fixture(
            compiler_identity,
            temporary_root / "installed-sdk",
            legacy_layout=False,
            installed_layout=True,
        )
        installed_environment_source = installed_provider.with_name(
            "flagdnn_hygon_compiler_environment.json"
        )
        assert not (
            temporary_root / "installed-sdk/share/triton_jit/scripts"
        ).exists()
        installed_validated = compiler_identity._validate_compiler_environment(
            installed_environment,
            installed_environment_source,
            installed_runtime_environment_sha256,
        )
        assert installed_validated["verified_library_sha256"] == (
            installed_environment["libtriton_jit_sha256"]
        )
        assert installed_validated[
            "verified_private_jit_script_directory"
        ] == str(installed_script_directory.resolve())
        assert installed_library.is_file()

        (
            legacy_provider,
            legacy_library,
            legacy_script_directory,
            legacy_environment,
            legacy_runtime_environment_sha256,
        ) = _write_compiler_environment_fixture(
            compiler_identity,
            temporary_root / "legacy",
            legacy_layout=True,
        )
        legacy_environment_source = legacy_provider.with_name(
            "flagdnn_hygon_compiler_environment.json"
        )
        expected_private_library = (
            legacy_provider.parent / "flagdnn/hygon" / legacy_library.name
        )
        expected_private_scripts = (
            legacy_provider.parent / "flagdnn/share/triton_jit/scripts"
        )
        assert legacy_library.is_file()
        assert legacy_script_directory.is_dir()
        assert not expected_private_library.exists()
        assert not expected_private_scripts.exists()

        legacy_library_candidates = set(
            compiler_identity._libtriton_jit_candidates(legacy_provider)
        )
        assert legacy_library.resolve() not in legacy_library_candidates
        assert expected_private_library.resolve() in legacy_library_candidates
        try:
            compiler_identity._validate_compiler_environment(
                legacy_environment,
                legacy_environment_source,
                legacy_runtime_environment_sha256,
            )
        except RuntimeError as error:
            assert (
                "cannot locate the CMake-selected libtriton_jit image"
                in str(error)
            )
        else:
            raise AssertionError(
                "legacy adjacent libtriton_jit and ../share scripts must not "
                "satisfy the private build-mirror identity"
            )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} SOURCE_ROOT")
    source_root = Path(argv[1]).resolve()
    if not (source_root / "backends/hygon/compiler.py").is_file():
        raise SystemExit(f"invalid FlagDNN source root: {source_root}")

    compiler, compiler_nn, compiler_tensor, select_kernel_candidate = (
        _load_modules(source_root)
    )
    for unsupported_target in ("gfx90a", "gfx942", "gfx1100"):
        try:
            compiler.compiler_identity(unsupported_target)
        except ValueError as error:
            assert "gfx936" in str(error)
        else:
            raise AssertionError(
                f"unsupported Hygon target {unsupported_target} was accepted"
            )
    _check_pointer_range_specialization(compiler)
    _check_hcu_llvm_compatibility()
    _check_fp8_layout(compiler_tensor)
    _check_attention_planner_contracts(
        compiler, compiler_nn, select_kernel_candidate
    )
    _check_private_convolution_autotune(
        compiler, compiler_nn, select_kernel_candidate
    )
    _check_fp8_backward(compiler, compiler_nn, select_kernel_candidate)
    _check_large_normalization(compiler, compiler_nn, select_kernel_candidate)
    _check_batchnorm_dispatch(compiler, compiler_nn, select_kernel_candidate)
    _check_pointwise_kernel_ownership(compiler, select_kernel_candidate)
    _check_sigmoid_backward_common_source(compiler)
    _check_platform_kernel_identity_inputs(compiler)
    _check_private_build_mirror_layout(compiler)
    print("Hygon compiler contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
