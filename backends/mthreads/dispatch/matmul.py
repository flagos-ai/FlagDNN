"""MThreads dispatch matmul implementation."""

from __future__ import annotations

from .common import (
    _MATMUL_SOURCE_RELATIVE_PATH,
    _MAX_I32,
    _SELECTION_CACHE,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _pointer_token,
)
from .graph import ParsedMatmulRequest
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import is_row_major_contiguous, matmul_constants
from .tuning import _load_tuning

_MATMUL_TLE_CONFIGS = {
    (32, 512, 512, 512): (128, 128, 32, 3, 2),
    (16, 1024, 1024, 1024): (256, 256, 32, 3, 2),
    (16, 2048, 2048, 512): (256, 256, 32, 3, 2),
    (8, 2048, 2048, 2048): (256, 256, 64, 3, 2),
    (32, 1024, 1024, 4096): (256, 256, 64, 3, 4),
    (4, 4096, 4096, 4096): (256, 256, 64, 3, 2),
}


def _matmul_runtime_arguments(
    request: ParsedMatmulRequest,
) -> tuple[RuntimeArgument, ...]:
    return (
        RuntimeArgument("tensor", "a", request.a.uid, None),
        RuntimeArgument("tensor", "b", request.b.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
    )


def _uses_matmul_descriptor(request: ParsedMatmulRequest) -> bool:
    batch_dimensions = request.output.dimensions[:-2]
    return (
        request.a.data_type in {"float16", "bfloat16"}
        and request.a.dimensions[:-2] == batch_dimensions
        and request.b.dimensions[:-2] == batch_dimensions
        and all(
            is_row_major_contiguous(tensor)
            for tensor in (request.a, request.b, request.output)
        )
        and min(
            request.a.alignment,
            request.b.alignment,
            request.output.alignment,
        )
        >= 16
        and request.m >= 128
        and request.n >= 128
        and request.k >= 64
        and request.m % 128 == 0
        and request.n % 128 == 0
        and request.k % 64 == 0
        and request.batch * request.m <= _MAX_I32
        and request.batch * request.k <= _MAX_I32
    )


def _matmul_tle_config(
    request: ParsedMatmulRequest,
) -> tuple[int, int, int, int, int] | None:
    if not _uses_matmul_descriptor(request):
        return None
    return _MATMUL_TLE_CONFIGS.get(
        (request.batch, request.m, request.n, request.k)
    )


def _uses_matmul_tle(request: ParsedMatmulRequest) -> bool:
    return _matmul_tle_config(request) is not None


def _matmul_tle_full_signature(
    request: ParsedMatmulRequest,
) -> str:
    config = _matmul_tle_config(request)
    if config is None:
        raise ValueError("mthreads TLE Matmul configuration is unavailable")
    block_m, block_n, block_k, pipeline_stages, panel_width = config
    dtype = "fp16" if request.a.data_type == "float16" else "bf16"
    return ",".join(
        (
            f"tensordesc<{dtype}[{block_m},{block_k}]>",
            f"tensordesc<{dtype}[{block_k},{block_n}]>",
            _pointer_token(request.output.data_type, request.output.alignment),
            str(request.m),
            str(request.n),
            str(request.k),
            str(request.batch),
            str(block_m),
            str(block_n),
            str(block_k),
            str(pipeline_stages),
            "1" if request.a.data_type == "bfloat16" else "0",
            str(panel_width),
        )
    )


def _matmul_descriptor_full_signature(
    request: ParsedMatmulRequest,
    *,
    block_m: int,
) -> str:
    dtype = "fp16" if request.a.data_type == "float16" else "bf16"
    block_n = 128
    return ",".join(
        (
            f"tensordesc<{dtype}[{block_m},64]>",
            f"tensordesc<{dtype}[64,{block_n}]>",
            f"tensordesc<{dtype}[{block_m},{block_n}]>",
            str(request.m),
            str(request.n),
            str(request.k),
            str(request.batch),
            str(block_m),
            str(block_n),
            "64",
            "8",
        )
    )


def _matmul_full_signature(
    request: ParsedMatmulRequest,
    *,
    block_size: int,
) -> str:
    block_n = (
        block_size * 2
        if request.a.data_type != "float32"
        and block_size <= 64
        and request.n >= block_size * 2
        else block_size
    )
    tokens = [
        _pointer_token(request.a.data_type, request.a.alignment),
        _pointer_token(request.b.data_type, request.b.alignment),
        _pointer_token(request.output.data_type, request.output.alignment),
    ]
    tokens.extend(
        str(value)
        for value in (
            *matmul_constants(
                request.a, request.b, request.output, request.input_precision
            ),
            block_size,
            block_n,
            32 if request.a.data_type == "float32" else 64,
            8,
        )
    )
    return ",".join(tokens)


def _matmul_plan(
    request: ParsedMatmulRequest,
    source_sha256: str,
) -> ExecutionPlan:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="matmul",
        warp_size=target_warp,
    )
    tle_config = _matmul_tle_config(request)
    descriptor = _uses_matmul_descriptor(request)
    launch = (
        (tle_config[0], 16, tle_config[3])
        if tle_config is not None
        else (
            (128, 4, 1)
            if descriptor
            else (
                (64, 8, 2)
                if request.a.data_type != "float32"
                and request.m >= 64
                and request.n >= 128
                and request.k >= 64
                else default
            )
        )
    )
    selected = (
        (launch,)
        if descriptor
        else candidates if request.autotune else (launch,)
    )
    function = (
        "matmul_tle_kernel"
        if tle_config is not None
        else (
            "matmul_descriptor_kernel"
            if descriptor
            else "matmul_strided_kernel"
        )
    )
    arguments = _matmul_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_MATMUL_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=(
                _matmul_tle_full_signature(request)
                if tle_config is not None
                else (
                    _matmul_descriptor_full_signature(
                        request, block_m=block_size
                    )
                    if descriptor
                    else _matmul_full_signature(request, block_size=block_size)
                )
            ),
            grid=(
                (
                    (
                        request.batch
                        * (request.m // tle_config[0])
                        * (request.n // tle_config[1])
                    )
                    if tle_config is not None
                    else ((request.m + block_size - 1) // block_size)
                    * (
                        (
                            request.n
                            + (
                                block_size * 2
                                if request.a.data_type != "float32"
                                and block_size <= 64
                                and request.n >= block_size * 2
                                else block_size
                            )
                            - 1
                        )
                        // (
                            block_size * 2
                            if request.a.data_type != "float32"
                            and block_size <= 64
                            and request.n >= block_size * 2
                            else block_size
                        )
                    )
                ),
                1 if tle_config is not None else request.batch,
                1,
            ),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    stage = KernelStage(
        id=0,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=(),
        source=_MATMUL_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune and not descriptor,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=_SELECTION_CACHE,
        ),
    )
    return ExecutionPlan(
        stages=(stage,),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_WORKSPACE_SIZE,
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )
