"""MThreads dispatch layout implementation."""

from __future__ import annotations

from .common import (
    _LAYOUT_SOURCE_RELATIVE_PATH,
    _MAX_I32,
    _SELECTION_CACHE,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _pointer_token,
    _scalar_i32_bits,
)
from .graph import ParsedLayoutRequest
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import is_row_major_contiguous, layout_constants
from .tuning import _load_tuning


def _layout_runtime_arguments(
    request: ParsedLayoutRequest,
) -> tuple[RuntimeArgument, ...]:
    return (
        RuntimeArgument("tensor", "input", request.input.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
        RuntimeArgument(
            "scalar_i32",
            "n_elements",
            None,
            _scalar_i32_bits(request.n_elements),
        ),
    )


def _layout_full_signature(
    request: ParsedLayoutRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    tokens = [
        _pointer_token(request.input.data_type, request.input.alignment),
        _pointer_token(request.output.data_type, request.output.alignment),
        "i32",
    ]
    if function == "slice_copy_kernel":
        rank = len(request.logical_input_dimensions)
        leading = 8 - rank
        tokens.extend(
            str(value)
            for value in (
                request.input_base,
                rank,
                *((1,) * leading + request.logical_input_dimensions),
                *((0,) * leading + request.logical_input_strides),
                *((0,) * leading + request.output.strides),
            )
        )
    elif function not in {
        "reshape_contiguous_kernel",
        "transpose_physical_copy_kernel",
    }:
        tokens.extend(
            str(value)
            for value in layout_constants(
                request.logical_input_dimensions,
                request.logical_input_strides,
                request.input_base,
                request.output,
            )
        )
    tokens.append(str(block_size))
    return ",".join(tokens)


def _layout_plan(
    request: ParsedLayoutRequest,
    source_sha256: str,
) -> ExecutionPlan:
    contiguous_reshape = (
        request.operation == "reshape"
        and request.input_base == 0
        and is_row_major_contiguous(request.input)
        and is_row_major_contiguous(request.output)
    )
    input_last = request.input_base + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(
            request.logical_input_dimensions,
            request.logical_input_strides,
            strict=True,
        )
    )
    output_last = sum(
        (dimension - 1) * stride
        for dimension, stride in zip(
            request.output.dimensions,
            request.output.strides,
            strict=True,
        )
    )
    specialized_slice = (
        request.operation == "slice"
        and input_last <= _MAX_I32
        and output_last <= _MAX_I32
    )
    physical_transpose = (
        request.operation == "transpose"
        and request.input_base == 0
        and request.logical_input_dimensions == request.output.dimensions
        and request.logical_input_strides == request.output.strides
        and request.input.storage_size == request.output.storage_size
    )
    function = (
        "reshape_contiguous_kernel"
        if contiguous_reshape
        else (
            "transpose_physical_copy_kernel"
            if physical_transpose
            else (
                "slice_copy_kernel"
                if specialized_slice
                else "layout_copy_kernel"
            )
        )
    )
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="layout",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _layout_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_LAYOUT_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_layout_full_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=(
                (request.n_elements + block_size - 1) // block_size,
                1,
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
        source=_LAYOUT_SOURCE_RELATIVE_PATH,
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
