"""Ascend dispatch for random."""

import math
from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _require_integer,
    _has_non_overlapping_strides,
)


def _rng_kernel_configuration(parameters, tensors):
    if len(tensors) != 1:
        raise ValueError("RNG requires one output and no inputs")
    output = tensors[0]
    shape, strides = output["dimensions"], output["strides"]
    if (
        not 1 <= len(shape) <= 8
        or output["data_type"] not in FLOAT_DATA_TYPES
        or not _has_non_overlapping_strides(shape, strides)
    ):
        raise ValueError("RNG output metadata is invalid")
    elements = _require_integer(parameters, "n_elements")
    offset = _require_integer(
        parameters, "offset", minimum=0, maximum=2**63 - 1 - elements
    )
    seed = _require_integer(parameters, "seed", minimum=-(2**63), maximum=2**63 - 1)
    distribution = _require_integer(parameters, "distribution", minimum=1, maximum=3)
    probability = parameters.get("probability")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("RNG Bernoulli probability must be in [0,1]")
    if elements != math.prod(shape):
        raise ValueError("RNG element count is inconsistent")
    constants = {
        "N_ELEMENTS": elements,
        "SEED": seed,
        "OFFSET": offset,
        "DISTRIBUTION": distribution,
        "PROBABILITY": float(probability),
        "UNIFORM_BITS": {"float32": 24, "float16": 11, "bfloat16": 8}[
            output["data_type"]
        ],
        "BLOCK_SIZE": 256,
    }
    for axis, dim in enumerate([1] * (8 - len(shape)) + shape):
        constants[f"DIM_{axis}"] = dim
    for axis, stride in enumerate([0] * (8 - len(shape)) + strides):
        constants[f"OUTPUT_STRIDE_{axis}"] = stride
    grid = (elements + 255) // 256
    if grid > 2**31 - 1:
        raise ValueError("RNG launch grid exceeds Ascend limits")
    return (
        "rng_kernel",
        {"output_ptr": TRITON_POINTER_TYPES[output["data_type"]]},
        constants,
        (grid, 1, 1),
        [("tensor", None)],
    )
