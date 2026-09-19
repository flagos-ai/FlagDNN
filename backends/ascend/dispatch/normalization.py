"""Ascend dispatch normalization implementation."""

from __future__ import annotations

from .common import (
    MAX_RANK,
    TensorPlan,
)


def _batchnorm_inference_meta(
    x: TensorPlan,
    y: TensorPlan,
    *,
    channels: int,
    spatial: int,
) -> dict[str, int]:
    leading = MAX_RANK - len(x.dimensions)
    dimensions = [1] * leading + list(x.dimensions)
    x_strides = [0] * leading + list(x.strides)
    y_strides = [0] * leading + list(y.strides)
    result = {
        "RANK": len(x.dimensions),
        "CHANNELS": channels,
        "SPATIAL": spatial,
    }
    for axis in range(MAX_RANK):
        result[f"DIM_{axis}"] = dimensions[axis]
        result[f"X_STRIDE_{axis}"] = x_strides[axis]
        result[f"Y_STRIDE_{axis}"] = y_strides[axis]
    return result


def _batchnorm_training_meta(
    x: TensorPlan,
    y: TensorPlan,
    *,
    batch: int,
    channels: int,
    spatial: int,
    epsilon: float,
    momentum: float,
) -> dict[str, int | float]:
    result: dict[str, int | float] = _batchnorm_inference_meta(
        x, y, channels=channels, spatial=spatial
    )
    result.update(
        {
            "BATCH": batch,
            "REDUCTION_ELEMENTS": batch * spatial,
            "EPSILON": epsilon,
            "MOMENTUM": momentum,
        }
    )
    return result
