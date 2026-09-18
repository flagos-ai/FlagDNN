"""MThreads dispatch reduction implementation."""

from __future__ import annotations

from .common import (
    _REDUCTION_SOURCE_RELATIVE_PATH,
    _SELECTION_CACHE,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _pointer_token,
    _scalar_i32_bits,
)
from .graph import ParsedReductionRequest
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import is_row_major_contiguous, reduction_strided_constants
from .tuning import _load_tuning


def _reduction_function(request: ParsedReductionRequest) -> str:
    contiguous = is_row_major_contiguous(
        request.input
    ) and is_row_major_contiguous(request.output)
    if contiguous and request.inner == 1:
        return "reduction_2d_kernel"
    if contiguous:
        return "reduction_3d_kernel"
    return "reduction_strided_kernel"


def _reduction_runtime_arguments(
    request: ParsedReductionRequest,
    function: str,
) -> tuple[RuntimeArgument, ...]:
    rows = (
        request.outer
        if function == "reduction_2d_kernel"
        else request.output_elements
    )
    scalar_name = (
        "outer" if function == "reduction_2d_kernel" else "output_elements"
    )
    return (
        RuntimeArgument("tensor", "input", request.input.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
        RuntimeArgument(
            "scalar_i32", scalar_name, None, _scalar_i32_bits(rows)
        ),
    )


def _reduction_full_signature(
    request: ParsedReductionRequest,
    *,
    function: str,
    block_m: int,
    block_n: int,
) -> str:
    tokens = [
        _pointer_token(request.input.data_type, request.input.alignment),
        _pointer_token(request.output.data_type, request.output.alignment),
        "i32",
    ]
    operation = request.reduction_mode + 1
    if function == "reduction_2d_kernel":
        constants = (
            request.extent,
            request.extent,
            1,
            operation,
            block_m,
            block_n,
        )
    elif function == "reduction_3d_kernel":
        constants = (
            request.extent,
            request.inner,
            request.extent * request.inner,
            request.inner,
            1,
            operation,
            block_m,
            block_n,
        )
    else:
        constants = (
            request.extent,
            *reduction_strided_constants(
                request.input,
                request.output,
                request.axis,
                request.keep_dimensions,
            ),
            operation,
            block_m,
            block_n,
        )
    tokens.extend(str(value) for value in constants)
    return ",".join(tokens)


def _reduction_plan(
    request: ParsedReductionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    function = _reduction_function(request)
    block_n = 1 << (request.extent - 1).bit_length()
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="reduction",
        warp_size=target_warp,
    )
    valid_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate[0] * block_n <= 65536
    )
    if default not in valid_candidates:
        raise ValueError(
            "mthreads reduction default exceeds the Triton tile limit"
        )
    launch = (16, 2, 1) if block_n * 16 <= 65536 else default
    selected = valid_candidates if request.autotune else (launch,)
    rows = (
        request.outer
        if function == "reduction_2d_kernel"
        else request.output_elements
    )
    arguments = _reduction_runtime_arguments(request, function)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_m}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_REDUCTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_reduction_full_signature(
                request,
                function=function,
                block_m=block_m,
                block_n=block_n,
            ),
            grid=((rows + block_m - 1) // block_m, 1, 1),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_m, num_warps, num_stages in selected
    )
    stage = KernelStage(
        id=0,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=(),
        source=_REDUCTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
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
