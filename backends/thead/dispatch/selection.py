# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate platform kernel registry candidates."""

from __future__ import annotations

from flagdnn_codegen import kernel_registry
from typing import Any
import hashlib


def _registry_sha256() -> str:
    sources = kernel_registry.iter_kernel_registry_sources("thead")
    if len(sources) != 2:
        raise ValueError(
            "THead requires common and platform kernel registries"
        )
    digest = hashlib.sha256()
    for source in sources:
        contents = source.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _validate_binary_pointwise_candidate(
    candidate: Any, operation: str
) -> None:
    operation_label = operation.capitalize()
    select = operation == "binary_select"
    power = operation == "pow"
    backward = operation.endswith("_backward")
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != ("thead_triton" if power else "common_triton")
        or candidate.ownership != ("platform" if power else "common")
        or candidate.source_layout != ("platform" if power else "kernels")
        or candidate.source_format != "module"
        or candidate.source
        != (
            "ternary.py"
            if select
            else (
                "pow.py"
                if power
                else "activation_backward.py" if backward else "binary.py"
            )
        )
        or candidate.functions
        != (
            ("binary_select_tensor_kernel", "binary_select_strided_kernel")
            if select
            else (
                (
                    "activation_backward_contiguous_kernel",
                    "activation_backward_strided_kernel",
                )
                if backward
                else ("binary_contiguous_kernel", "binary_strided_kernel")
            )
        )
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "binary"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation_label} common kernel registry contract is"
            " invalid"
        )


def _validate_unary_pointwise_candidate(
    candidate: Any, operation: str
) -> None:
    operation_label = operation.capitalize()
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "unary.py"
        or candidate.functions
        != (
            "unary_pointwise_contiguous_kernel",
            "unary_pointwise_strided_kernel",
        )
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "relu"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation_label} common kernel registry contract is"
            " invalid"
        )


def _validate_add_square_candidate(candidate: Any) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != "add_square"
        or candidate.provider != "thead_triton"
        or candidate.ownership != "platform"
        or candidate.source_layout != "platform"
        or candidate.source_format != "module"
        or candidate.source != "add_square.py"
        or candidate.functions
        != ("add_square_contiguous_kernel", "add_square_strided_kernel")
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "binary"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError("THead AddSquare kernel registry contract is invalid")


def _validate_layout_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider
        != ("thead_triton" if operation == "transpose" else "common_triton")
        or candidate.ownership
        != ("platform" if operation == "transpose" else "common")
        or candidate.source_layout
        != ("platform" if operation == "transpose" else "kernels")
        or candidate.source_format != "module"
        or candidate.source != "layout.py"
        or candidate.functions
        != (
            ("layout_copy_kernel",)
            if operation == "transpose"
            else ("layout_copy_kernel", "matrix_transpose_kernel")
        )
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "binary"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
    ):
        raise ValueError(
            f"THead {operation} common kernel contract is invalid"
        )


def _validate_reduction_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "reduction.py"
        or "reduction_2d_kernel" not in candidate.functions
        or "reduction_3d_kernel" not in candidate.functions
        or "reduction_strided_kernel" not in candidate.functions
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "reduction"
        or tuning.key != "output_elements"
        or tuning.strategy != "reduction"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} common reduction contract is invalid"
        )


def _validate_batchnorm_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    expected_functions = (
        {"batch_norm_inference_nchw_kernel", "batch_norm_inference_kernel"}
        if operation == "batchnorm_inference"
        else {"batch_norm_nchw_kernel", "batch_norm_kernel"}
    )
    expected_key = (
        "n_elements" if operation == "batchnorm_inference" else "channels"
    )
    expected_strategy = (
        "align32" if operation == "batchnorm_inference" else "fixed_grid"
    )
    expected_provider = (
        "thead_triton"
        if operation == "batchnorm_inference"
        else "common_triton"
    )
    expected_ownership = (
        "platform" if operation == "batchnorm_inference" else "common"
    )
    expected_layout = (
        "platform" if operation == "batchnorm_inference" else "kernels"
    )
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != expected_provider
        or candidate.ownership != expected_ownership
        or candidate.source_layout != expected_layout
        or candidate.source_format != "module"
        or candidate.source != "normalization.py"
        or not expected_functions.issubset(set(candidate.functions))
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "batch_norm"
        or tuning.key != expected_key
        or tuning.strategy != expected_strategy
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} common normalization contract is invalid"
        )


def _validate_normalization_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    expected_function = (
        "layer_norm_kernel" if operation == "layernorm" else "rms_norm_kernel"
    )
    expected_table = "layer_norm" if operation == "layernorm" else "rms_norm"
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "normalization.py"
        or candidate.functions != (expected_function,)
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != expected_table
        or tuning.key != "normalized_elements"
        or tuning.strategy != "fixed_grid"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} common normalization contract is invalid"
        )


def _validate_matmul_candidate(candidate: Any) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != "matmul"
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "matmul.py"
        or candidate.functions != ("matmul_strided_kernel",)
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "matmul"
        or tuning.key != "m"
        or tuning.strategy != "matmul"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError("THead MatMul common kernel contract is invalid")


def _validate_convolution_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    fprop = operation == "convolution_fprop"
    fused = operation == "conv_bias_relu"
    expected_function = (
        "conv2d_bias_relu_kernel"
        if fused
        else (
            "conv_fprop_nd_kernel"
            if fprop
            else (
                "conv_dgrad_nd_kernel"
                if operation == "convolution_dgrad"
                else "conv_wgrad_nd_kernel"
            )
        )
    )
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "thead_triton"
        or candidate.ownership != "platform"
        or candidate.source_layout != "platform"
        or candidate.source_format != "module"
        or candidate.source != "convolution.py"
        or expected_function not in candidate.functions
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "conv2d_spatial"
        or tuning.key != "n_outputs"
        or tuning.strategy != "convolution"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} kernel registry contract is invalid"
        )
