"""MThreads dispatch composite implementation."""

from __future__ import annotations

from .common import (
    _CONV_BIAS_RELU_SOURCE_RELATIVE_PATH,
    _SELECTION_CACHE,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _pointer_token,
)
from .graph import ParsedConvBiasReluRequest
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tuning import _load_tuning


def _conv_bias_relu_runtime_arguments(
    request: ParsedConvBiasReluRequest,
) -> tuple[RuntimeArgument, ...]:
    return (
        RuntimeArgument("tensor", "input", request.image.uid, None),
        RuntimeArgument("tensor", "filter", request.filter.uid, None),
        RuntimeArgument("tensor", "bias", request.bias.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
    )


def _conv_bias_relu_full_signature(
    request: ParsedConvBiasReluRequest,
    *,
    block_size: int,
) -> str:
    _, _, input_height, input_width = request.image.dimensions
    _, _, filter_height, filter_width = request.filter.dimensions
    _, _, output_height, output_width = request.output.dimensions
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    pointer = lambda tensor: _pointer_token(  # noqa: E731
        tensor.data_type, tensor.alignment
    )
    tokens = [
        pointer(request.image),
        pointer(request.filter),
        pointer(request.bias),
        pointer(request.output),
    ]
    tokens.extend(
        str(value)
        for value in (
            input_height,
            input_width,
            output_height,
            output_width,
            request.in_channels,
            request.out_channels,
            request.in_per_group,
            request.out_per_group,
            request.groups,
            request.stride[0],
            request.stride[1],
            request.pre_padding[0],
            request.pre_padding[1],
            request.dilation[0],
            request.dilation[1],
            filter_height,
            filter_width,
            1,
            block_size,
            block_size,
            block_size,
            8,
            dtype_id,
            0,
            *request.image.strides,
            *request.filter.strides,
            *request.output.strides,
        )
    )
    return ",".join(tokens)


def _conv_bias_relu_plan(
    request: ParsedConvBiasReluRequest,
    source_sha256: str,
) -> ExecutionPlan:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _conv_bias_relu_runtime_arguments(request)
    output_spatial = (
        request.output.dimensions[2] * request.output.dimensions[3]
    )
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONV_BIAS_RELU_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function="conv_bias_relu_2d_kernel",
            full_signature=_conv_bias_relu_full_signature(
                request, block_size=block_size
            ),
            grid=(
                ((output_spatial + block_size - 1) // block_size)
                * ((request.out_per_group + block_size - 1) // block_size),
                request.batch * request.groups,
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
        source=_CONV_BIAS_RELU_SOURCE_RELATIVE_PATH,
        function="conv_bias_relu_2d_kernel",
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
