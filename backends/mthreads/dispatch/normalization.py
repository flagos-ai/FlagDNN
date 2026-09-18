"""MThreads dispatch normalization implementation."""

from __future__ import annotations

from .common import (
    _NORMALIZATION_SOURCE_RELATIVE_PATH,
    _SELECTION_CACHE,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _float32_token,
    _pointer_token,
    _scalar_i32_bits,
)
from .graph import (
    ParsedBatchnormInferenceRequest,
    ParsedBatchnormRequest,
    ParsedNormalizationRequest,
)
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import is_row_major_contiguous
from .tuning import _load_tuning


def _normalization_runtime_arguments(
    request: ParsedNormalizationRequest,
) -> tuple[RuntimeArgument, ...]:
    if request.operation == "layernorm":
        if request.mean is None:
            raise ValueError("LayerNorm request is missing mean output")
        tensors = (
            ("x", request.x),
            ("y", request.y),
            ("mean", request.mean),
            ("inv_variance", request.inv_variance),
            ("scale", request.scale),
            ("bias", request.bias),
        )
    else:
        tensors = (
            ("x", request.x),
            ("y", request.y),
            ("scale", request.scale),
            ("bias", request.bias),
            ("inv_variance", request.inv_variance),
        )
    return tuple(
        RuntimeArgument("tensor", name, tensor.uid, None)
        for name, tensor in tensors
    ) + (
        RuntimeArgument(
            "scalar_i32",
            "rows",
            None,
            _scalar_i32_bits(request.rows),
        ),
    )


def _normalization_full_signature(
    request: ParsedNormalizationRequest,
    *,
    block_size: int,
) -> str:
    if request.operation == "layernorm":
        if request.mean is None:
            raise ValueError("LayerNorm request is missing mean output")
        tensors = (
            request.x,
            request.y,
            request.mean,
            request.inv_variance,
            request.scale,
            request.bias,
        )
        constants = (
            _float32_token(request.epsilon),
            request.normalized_elements,
            block_size,
            1,
            1,
            1,
            1,
            0,  # STATIC_ROWS
            0,  # EVICT_INPUT_FIRST
            0,  # PAIRED_REDUCTION
        )
    else:
        tensors = (
            request.x,
            request.y,
            request.scale,
            request.bias,
            request.inv_variance,
        )
        constants = (
            request.normalized_elements,
            _float32_token(request.epsilon),
            block_size,
            1,
            1,
            1,
            1,
            0,  # STATIC_ROWS
        )
    tokens = [
        _pointer_token(tensor.data_type, tensor.alignment)
        for tensor in tensors
    ]
    tokens.append("i32")
    tokens.extend(str(value) for value in constants)
    return ",".join(tokens)


def _normalization_plan(
    request: ParsedNormalizationRequest,
    source_sha256: str,
) -> ExecutionPlan:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="normalization",
        warp_size=target_warp,
    )
    # A 256-element tile forces the long-row kernels to reload each row in
    # many small chunks for both the statistics and output passes.  MTGPU can
    # lower a 1024-element reduction directly, and eight warps provide enough
    # lanes to cover that tile without changing the public kernel ABI.
    steady_state_block = min(
        4096,
        1 << (request.normalized_elements - 1).bit_length(),
    )
    steady_state_default = (
        (
            steady_state_block,
            4 if steady_state_block <= 1024 else 8,
            1,
        )
        if request.normalized_elements > 513
        else default
    )
    # Autotune must never exclude the steady-state configuration used by the
    # default path.  The static normalization table intentionally contains
    # only small generic tiles, while long rows use a shape-specific full-row
    # tile above.  Omitting that tile from an autotuned graph made FP16/BF16
    # LayerNorm benchmark builds choose at most BLOCK_SIZE=512 even for
    # 1024/2048/4096-element rows, so enabling autotune could make execution
    # materially slower than the non-autotuned default.
    selected = (
        tuple(dict.fromkeys((steady_state_default, *candidates)))
        if request.autotune
        else (steady_state_default,)
    )
    function = (
        "layer_norm_kernel"
        if request.operation == "layernorm"
        else "rms_norm_kernel"
    )
    arguments = _normalization_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_NORMALIZATION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_normalization_full_signature(
                request, block_size=block_size
            ),
            grid=(request.rows, 1, 1),
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
        source=_NORMALIZATION_SOURCE_RELATIVE_PATH,
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


def _batchnorm_runtime_arguments(
    request: ParsedBatchnormRequest,
    *,
    specialized: bool,
) -> tuple[RuntimeArgument, ...]:
    tensors = (
        ("x", request.x),
        ("y", request.y),
        ("previous_running_mean", request.previous_running_mean),
        ("previous_running_variance", request.previous_running_variance),
        ("scale", request.scale),
        ("bias", request.bias),
        ("mean", request.mean),
        ("inv_variance", request.inv_variance),
        ("next_running_mean", request.next_running_mean),
        ("next_running_variance", request.next_running_variance),
    )
    result = tuple(
        RuntimeArgument("tensor", name, tensor.uid, None)
        for name, tensor in tensors
    )
    if specialized:
        return result
    return result + tuple(
        RuntimeArgument("scalar_i32", name, None, _scalar_i32_bits(value))
        for name, value in (
            ("batch", request.batch),
            ("channels", request.channels),
            ("spatial", request.spatial),
        )
    )


def _padded_normalization_metadata(
    request: ParsedBatchnormRequest | ParsedBatchnormInferenceRequest,
) -> tuple[int, ...]:
    leading = 8 - len(request.x.dimensions)
    return tuple(
        [1] * leading
        + list(request.x.dimensions)
        + [0] * leading
        + list(request.x.strides)
        + [0] * leading
        + list(request.y.strides)
    )


def _batchnorm_full_signature(
    request: ParsedBatchnormRequest,
    *,
    specialized: bool,
    block_size: int,
) -> str:
    tensors = (
        request.x,
        request.y,
        request.previous_running_mean,
        request.previous_running_variance,
        request.scale,
        request.bias,
        request.mean,
        request.inv_variance,
        request.next_running_mean,
        request.next_running_variance,
    )
    tokens = [
        _pointer_token(tensor.data_type, tensor.alignment)
        for tensor in tensors
    ]
    if specialized:
        tokens.extend(
            str(value)
            for value in (
                request.batch,
                request.channels,
                request.spatial,
                _float32_token(request.epsilon),
                _float32_token(request.momentum),
                block_size,
                1,
                1,
                1,
                1,
                1,
            )
        )
    else:
        tokens.extend(("i32", "i32", "i32"))
        tokens.extend(
            str(value)
            for value in (
                _float32_token(request.epsilon),
                _float32_token(request.momentum),
                block_size,
                1,
                1,
                1,
                1,
                1,
                1,
                *_padded_normalization_metadata(request),
            )
        )
    return ",".join(tokens)


def _batchnorm_plan(
    request: ParsedBatchnormRequest,
    source_sha256: str,
) -> ExecutionPlan:
    batch_block = 1 << (request.batch - 1).bit_length()
    specialized = (
        is_row_major_contiguous(request.x)
        and is_row_major_contiguous(request.y)
        and batch_block <= 256
    )
    function = "batch_norm_nchw_kernel" if specialized else "batch_norm_kernel"
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="batchnorm",
        warp_size=target_warp,
    )
    valid_candidates = tuple(
        candidate
        for candidate in candidates
        if not specialized or candidate[0] >= batch_block
    )
    if default not in valid_candidates:
        raise ValueError("BatchNorm default tile is invalid")
    # The specialized NCHW kernel visits every N*S item twice (statistics and
    # normalization).  Restricting autotune to the generic 512-element tile
    # leaves dozens of loop iterations for common training shapes such as
    # 8x64x56x56.  Add one shape-specific large tile while retaining all
    # generic candidates for small tensors and devices where they win.
    items_per_channel = request.batch * request.spatial
    if specialized and items_per_channel > 512:
        shape_block = min(
            16384,
            1 << (items_per_channel - 1).bit_length(),
        )
        shape_candidate = (
            shape_block,
            4 if shape_block <= 1024 else 8,
            1,
        )
        autotune_candidates = tuple(
            dict.fromkeys((shape_candidate, *valid_candidates))
        )
    else:
        autotune_candidates = valid_candidates
    selected = autotune_candidates if request.autotune else (default,)
    arguments = _batchnorm_runtime_arguments(request, specialized=specialized)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_NORMALIZATION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_batchnorm_full_signature(
                request,
                specialized=specialized,
                block_size=block_size,
            ),
            grid=(request.channels, 1, 1),
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
        source=_NORMALIZATION_SOURCE_RELATIVE_PATH,
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


def _batchnorm_inference_runtime_arguments(
    request: ParsedBatchnormInferenceRequest,
    *,
    specialized: bool,
) -> tuple[RuntimeArgument, ...]:
    tensors = (
        ("x", request.x),
        ("mean", request.mean),
        ("inv_variance", request.inv_variance),
        ("scale", request.scale),
        ("bias", request.bias),
        ("y", request.y),
    )
    result = tuple(
        RuntimeArgument("tensor", name, tensor.uid, None)
        for name, tensor in tensors
    )
    if specialized:
        return result
    return result + tuple(
        RuntimeArgument("scalar_i32", name, None, _scalar_i32_bits(value))
        for name, value in (
            ("n_elements", request.n_elements),
            ("channels", request.channels),
            ("spatial", request.spatial),
        )
    )


def _batchnorm_inference_full_signature(
    request: ParsedBatchnormInferenceRequest,
    *,
    specialized: bool,
    block_size: int,
) -> str:
    tensors = (
        request.x,
        request.mean,
        request.inv_variance,
        request.scale,
        request.bias,
        request.y,
    )
    tokens = [
        _pointer_token(tensor.data_type, tensor.alignment)
        for tensor in tensors
    ]
    if specialized:
        tokens.extend(
            str(value)
            for value in (
                request.channels,
                request.spatial,
                _float32_token(0.0),
                block_size,
                1,
                1,
                1,
            )
        )
    else:
        tokens.extend(("i32", "i32", "i32"))
        tokens.extend(
            str(value)
            for value in (
                _float32_token(0.0),
                block_size,
                1,
                1,
                1,
                1,
                *_padded_normalization_metadata(request),
            )
        )
    return ",".join(tokens)


def _batchnorm_inference_grid(
    request: ParsedBatchnormInferenceRequest,
    *,
    specialized: bool,
    block_size: int,
) -> tuple[int, int, int]:
    if not specialized:
        return (
            (request.n_elements + block_size - 1) // block_size,
            1,
            1,
        )
    return (
        request.batch * request.channels,
        (request.spatial + block_size - 1) // block_size,
        1,
    )


def _batchnorm_inference_plan(
    request: ParsedBatchnormInferenceRequest,
    source_sha256: str,
) -> ExecutionPlan:
    # The two-dimensional NCHW kernel is profitable when each channel owns
    # useful spatial work.  With S == 1 it launches one tiny program for every
    # N/C pair (for example 32768 programs for 32x1024x1x1).  The existing
    # contiguous path in the generic kernel batches those elements into full
    # blocks and preserves exactly the same channel mapping.
    specialized = (
        request.spatial > 1
        and is_row_major_contiguous(request.x)
        and is_row_major_contiguous(request.y)
    )
    function = (
        "batch_norm_inference_nchw_kernel"
        if specialized
        else "batch_norm_inference_kernel"
    )
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="batchnorm_inference",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _batchnorm_inference_runtime_arguments(
        request, specialized=specialized
    )
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_NORMALIZATION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_batchnorm_inference_full_signature(
                request,
                specialized=specialized,
                block_size=block_size,
            ),
            grid=_batchnorm_inference_grid(
                request,
                specialized=specialized,
                block_size=block_size,
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
        source=_NORMALIZATION_SOURCE_RELATIVE_PATH,
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
