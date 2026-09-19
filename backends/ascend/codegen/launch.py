"""Ascend codegen launch implementation."""

from __future__ import annotations

from ..dispatch.common import (
    LAUNCH_PAYLOAD_VERSION,
    PointwiseStagePlan,
)
from ..tuning_decoder import (
    TuningConfiguration,
    checked_grid,
    validate_add_configuration,
)
from .signature import (
    _ascend_full_signature,
    _runtime_argument_abi,
)
from typing import (
    Any,
)


def _batchnorm_inference_grid(
    stage: PointwiseStagePlan,
    block_size: int,
    capabilities: dict[str, Any],
) -> tuple[int, int, int]:
    if (
        stage.function_name == "batchnorm_inference_strided_persistent_kernel"
        and int(stage.meta["RANK"]) == 5
    ):
        channels = int(stage.meta["CHANNELS"])
        spatial = int(stage.meta["SPATIAL"])
        batch_elements = channels * spatial
        if batch_elements <= 0 or stage.n_elements % batch_elements != 0:
            raise ValueError("BatchNorm inference dimensions are inconsistent")
        batches = stage.n_elements // batch_elements
        block_spatial = block_size // 128
        if block_spatial <= 0:
            raise ValueError("BatchNorm inference block size is invalid")
        work_items = (
            batches
            * ((channels + 1) // 2)
            * ((spatial + block_spatial - 1) // block_spatial)
        )
        return checked_grid(work_items, 1, capabilities)
    return checked_grid(stage.n_elements, block_size, capabilities)


def _candidate_payload(
    *,
    stage: PointwiseStagePlan,
    configuration: TuningConfiguration,
    capabilities: dict[str, Any],
    source_path: str,
    source_sha256: str,
    worker_count: int,
) -> dict[str, Any]:
    validate_add_configuration(
        configuration, capabilities, kernel_family=stage.kernel_family
    )
    meta = dict(stage.meta)
    meta.update(configuration.meta)
    meta["WORKER_COUNT"] = worker_count
    if stage.kernel_family in {
        "reduction",
        "batchnorm",
        "rmsnorm",
        "layernorm",
    }:
        work_items = (
            int(meta["ROWS"])
            if stage.kernel_family in {"rmsnorm", "layernorm"}
            else (
                int(meta["CHANNELS"])
                if stage.kernel_family == "batchnorm"
                else stage.n_elements
            )
        )
        grid = checked_grid(work_items, 1, capabilities)
    elif stage.kernel_family == "batchnorm_inference":
        grid = _batchnorm_inference_grid(stage, int(meta["BLOCK_SIZE"]), capabilities)
    elif stage.kernel_family == "matmul":
        block_size = int(meta["BLOCK_SIZE"])
        tiles = ((int(meta["M"]) + block_size - 1) // block_size) * (
            (int(meta["N"]) + block_size - 1) // block_size
        )
        batch = int(meta["BATCH"])
        grid = checked_grid(tiles * batch, 1, capabilities)
    elif stage.kernel_family == "layout" and int(meta["LAYOUT_MODE"]) == 2:
        inner_elements = int(meta["OUTPUT_DIM_7"])
        if inner_elements <= 0 or stage.n_elements % inner_elements != 0:
            raise ValueError("shared-row layout dimensions are inconsistent")
        grid = checked_grid(stage.n_elements // inner_elements, 1, capabilities)
    else:
        grid = checked_grid(stage.n_elements, int(meta["BLOCK_SIZE"]), capabilities)
    grid = (min(grid[0], worker_count), grid[1], grid[2])
    return {
        "schema_version": LAUNCH_PAYLOAD_VERSION,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "entry_point": stage.function_name,
        "full_signature": _ascend_full_signature(stage, meta),
        "grid": list(grid),
        "compile_options": {
            "num_warps": configuration.num_warps,
            "num_stages": configuration.num_stages,
        },
        "meta": meta,
        "argument_abi": _runtime_argument_abi(stage),
    }
