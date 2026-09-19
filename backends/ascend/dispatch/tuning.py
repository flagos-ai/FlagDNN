"""Ascend dispatch tuning implementation."""

from __future__ import annotations

import struct
from ..tuning_decoder import (
    TuningConfiguration,
    canonical_sha256,
)
from .common import (
    PointwiseStagePlan,
)
from typing import (
    Any,
)


def _autotune_candidate_id(operation: str, configuration: TuningConfiguration) -> str:
    return "config_" + canonical_sha256(
        {
            "operation": operation,
            "configuration": configuration.as_dict(),
        },
        "Ascend binary tuning configuration",
    )


def _autotune_candidate_descriptor(
    candidate_id: str, configuration: TuningConfiguration
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "meta": dict(configuration.meta),
        "num_warps": configuration.num_warps,
        "num_stages": configuration.num_stages,
    }


def _f64_identity(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _stage_semantic_attributes(
    plan: PointwiseStagePlan,
) -> dict[str, Any]:
    if plan.kernel_family == "binary":
        return {"alpha": _f64_identity(plan.alpha)}
    if plan.kernel_family == "unary":
        return {
            "elu_alpha": _f64_identity(plan.elu_alpha),
            "has_upper_clip": plan.has_upper_clip,
            "lower_clip": _f64_identity(plan.lower_clip),
            "negative_slope": _f64_identity(plan.negative_slope),
            "softplus_beta": _f64_identity(plan.softplus_beta),
            "swish_beta": _f64_identity(plan.swish_beta),
            "upper_clip": _f64_identity(plan.upper_clip),
        }
    if plan.kernel_family == "reduction":
        return {
            name.lower(): int(plan.meta[name])
            for name in (
                "AXIS",
                "KEEP_DIMENSIONS",
                "OUTER",
                "REDUCTION_SIZE",
                "INNER",
                "OUTPUT_ELEMENTS",
                "REDUCTION_MODE",
            )
        }
    if plan.kernel_family == "matmul":
        roles = ("a", "b", "output")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "batch": int(plan.meta["BATCH"]),
            "m": int(plan.meta["M"]),
            "n": int(plan.meta["N"]),
            "k": int(plan.meta["K"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "convolution_fprop":
        roles = ("input", "filter", "output")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "meta": {
                name.lower(): int(value)
                for name, value in plan.meta.items()
                if name != "BLOCK_SIZE"
            },
            "n_elements": plan.n_elements,
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "batchnorm_inference":
        roles = ("x", "mean", "inv_variance", "scale", "bias", "y")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "channels": int(plan.meta["CHANNELS"]),
            "n_elements": plan.n_elements,
            "rank": int(plan.meta["RANK"]),
            "spatial": int(plan.meta["SPATIAL"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "batchnorm":
        roles = (
            "x",
            "scale",
            "bias",
            "previous_running_mean",
            "previous_running_variance",
            "y",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        )
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "batch": int(plan.meta["BATCH"]),
            "channels": int(plan.meta["CHANNELS"]),
            "epsilon": _f64_identity(float(plan.meta["EPSILON"])),
            "momentum": _f64_identity(float(plan.meta["MOMENTUM"])),
            "n_elements": plan.n_elements,
            "rank": int(plan.meta["RANK"]),
            "reduction_elements": int(plan.meta["REDUCTION_ELEMENTS"]),
            "spatial": int(plan.meta["SPATIAL"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "rmsnorm":
        roles = ("x", "scale", "bias", "y", "inv_variance")
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "epsilon": _f64_identity(float(plan.meta["EPSILON"])),
            "normalized_elements": int(plan.meta["NORMALIZED_ELEMENTS"]),
            "rows": int(plan.meta["ROWS"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    if plan.kernel_family == "layernorm":
        roles = (
            "x",
            "scale",
            "bias",
            "y",
            "mean",
            "inv_variance",
        )
        return {
            "arguments": [dict(argument) for argument in plan.argument_sources],
            "epsilon": _f64_identity(float(plan.meta["EPSILON"])),
            "normalized_elements": int(plan.meta["NORMALIZED_ELEMENTS"]),
            "rows": int(plan.meta["ROWS"]),
            "tensors": [
                {
                    "alignment": tensor.alignment,
                    "data_type": tensor.data_type,
                    "dimensions": list(tensor.dimensions),
                    "role": role,
                    "storage_size": tensor.storage_size,
                    "strides": list(tensor.strides),
                    "virtual": tensor.virtual,
                }
                for role, tensor in zip(roles, plan.tensors, strict=True)
            ],
        }
    return {}


def _order_stage_configurations(
    plan: PointwiseStagePlan,
    configurations: tuple[TuningConfiguration, ...],
) -> tuple[TuningConfiguration, ...]:
    if (
        plan.kernel_family == "unary"
        and plan.operation in {"log", "rsqrt"}
        and plan.function_name == "unary_pointwise_contiguous_kernel"
        and plan.n_elements < 1024
    ):
        preferred = next(
            (
                configuration
                for configuration in configurations
                if int(configuration.meta["BLOCK_SIZE"]) == 256
            ),
            configurations[0],
        )
        return (preferred,) + tuple(
            configuration
            for configuration in configurations
            if configuration is not preferred
        )
    if plan.kernel_family not in {"rmsnorm", "layernorm"}:
        return configurations
    normalized_elements = int(plan.meta["NORMALIZED_ELEMENTS"])
    covering = [
        configuration
        for configuration in configurations
        if int(configuration.meta["BLOCK_SIZE"]) >= normalized_elements
    ]
    preferred = (
        min(covering, key=lambda value: int(value.meta["BLOCK_SIZE"]))
        if covering
        else max(configurations, key=lambda value: int(value.meta["BLOCK_SIZE"]))
    )
    return (preferred,) + tuple(
        configuration
        for configuration in configurations
        if configuration is not preferred
    )
