"""MThreads dispatch pointwise implementation."""

from __future__ import annotations

from .common import (
    _BINARY_SOURCE_RELATIVE_PATH,
    _COMPOSITE_SOURCE_RELATIVE_PATH,
    _IDENTITY_SOURCE_RELATIVE_PATH,
    _SELECTION_CACHE,
    _TERNARY_SOURCE_RELATIVE_PATH,
    _UNARY_SOURCE_RELATIVE_PATH,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _float32_token,
    _pointer_token,
    _scalar_i32_bits,
)
from .graph import (
    ParsedAddSquareRequest,
    ParsedBinaryRequest,
    ParsedTernaryRequest,
    ParsedUnaryRequest,
)
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import (
    can_use_dense_binary,
    can_use_dense_ternary,
    can_use_dense_unary,
    pointwise_constants,
    ternary_pointwise_constants,
    unary_pointwise_constants,
)
from .tuning import _load_tuning


def _binary_runtime_arguments(
    request: ParsedBinaryRequest,
) -> tuple[RuntimeArgument, ...]:
    return (
        RuntimeArgument("tensor", "left", request.left.uid, None),
        RuntimeArgument("tensor", "right", request.right.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
        RuntimeArgument(
            "scalar_i32",
            "n_elements",
            None,
            _scalar_i32_bits(request.n_elements),
        ),
    )


def _binary_full_signature(
    request: ParsedBinaryRequest,
    *,
    dense: bool,
    block_size: int,
) -> str:
    tokens = [
        _pointer_token(request.left.data_type, request.left.alignment),
        _pointer_token(request.right.data_type, request.right.alignment),
        _pointer_token(request.output.data_type, request.output.alignment),
        "i32",
    ]
    if not dense:
        tokens.extend(
            str(value)
            for value in pointwise_constants(
                request.left, request.right, request.output
            )
        )
    tokens.extend(
        (
            str(request.pointwise_mode),
            _float32_token(request.alpha),
            str(block_size),
        )
    )
    return ",".join(tokens)


def _binary_plan(
    request: ParsedBinaryRequest,
    source_sha256: str,
) -> ExecutionPlan:
    dense = can_use_dense_binary(request.left, request.right, request.output)
    function = "binary_contiguous_kernel" if dense else "binary_strided_kernel"
    table = "binary_contiguous" if dense else "binary_strided"
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name=table,
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _binary_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_BINARY_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_binary_full_signature(
                request, dense=dense, block_size=block_size
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
        source=_BINARY_SOURCE_RELATIVE_PATH,
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


def _add_square_runtime_arguments(
    request: ParsedAddSquareRequest,
) -> tuple[RuntimeArgument, ...]:
    return (
        RuntimeArgument("tensor", "left", request.left.uid, None),
        RuntimeArgument("tensor", "right", request.right.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
        RuntimeArgument(
            "scalar_i32",
            "n_elements",
            None,
            _scalar_i32_bits(request.n_elements),
        ),
    )


def _add_square_full_signature(
    request: ParsedAddSquareRequest,
    *,
    block_size: int,
) -> str:
    return ",".join(
        (
            _pointer_token(request.left.data_type, request.left.alignment),
            _pointer_token(request.right.data_type, request.right.alignment),
            _pointer_token(request.output.data_type, request.output.alignment),
            "i32",
            "1",
            str(block_size),
            "1",
        )
    )


def _add_square_plan(
    request: ParsedAddSquareRequest,
    source_sha256: str,
) -> ExecutionPlan:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="binary_contiguous",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _add_square_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_COMPOSITE_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function="add_square_tensor_kernel",
            full_signature=_add_square_full_signature(
                request, block_size=block_size
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
        source=_COMPOSITE_SOURCE_RELATIVE_PATH,
        function="add_square_tensor_kernel",
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


def _unary_runtime_arguments(
    request: ParsedUnaryRequest,
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


def _identity_plan(
    request: ParsedUnaryRequest,
    source_sha256: str,
) -> ExecutionPlan:
    dense = can_use_dense_unary(request.input, request.output)
    packed = (
        dense
        and request.input.alignment >= 8
        and request.output.alignment >= 8
    )
    if packed:
        function = "identity_contiguous_packed_kernel"
        pack_size = {
            "int32": 2,
            "boolean": 8,
            "fp8_e4m3": 8,
            "fp8_e5m2": 8,
            "fp8_e8m0": 8,
            "float32": 2,
            "float16": 4,
            "bfloat16": 4,
        }[request.input.data_type]
        tiles_per_program = 4
        table = "unary_contiguous"
    elif dense:
        function = "identity_contiguous_kernel"
        pack_size = 1
        tiles_per_program = 8
        table = "unary_contiguous"
    else:
        function = "identity_strided_kernel"
        pack_size = 1
        tiles_per_program = 1
        table = "unary_strided"

    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name=table,
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _unary_runtime_arguments(request)

    def signature(block_size: int) -> str:
        tokens = [
            _pointer_token(request.input.data_type, request.input.alignment),
            _pointer_token(request.output.data_type, request.output.alignment),
            "i32",
        ]
        if packed:
            tokens.extend((str(pack_size), str(tiles_per_program)))
        elif dense:
            tokens.append(str(tiles_per_program))
        else:
            tokens.extend(
                str(value)
                for value in unary_pointwise_constants(
                    request.input, request.output
                )
            )
        tokens.append(str(block_size))
        return ",".join(tokens)

    def grid(block_size: int) -> tuple[int, int, int]:
        if packed:
            work_items = request.n_elements // pack_size
        else:
            work_items = request.n_elements
        work_per_program = block_size * tiles_per_program
        return (
            max(1, (work_items + work_per_program - 1) // work_per_program),
            1,
            1,
        )

    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_IDENTITY_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=signature(block_size),
            grid=grid(block_size),
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
        source=_IDENTITY_SOURCE_RELATIVE_PATH,
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


def _unary_full_signature(
    request: ParsedUnaryRequest,
    *,
    dense: bool,
    block_size: int,
    tiles_per_program: int,
) -> str:
    tokens = [
        _pointer_token(request.input.data_type, request.input.alignment),
        _pointer_token(request.output.data_type, request.output.alignment),
        "i32",
    ]
    if not dense:
        tokens.extend(
            str(value)
            for value in unary_pointwise_constants(
                request.input, request.output
            )
        )
        tokens.append("1")
    tokens.extend(
        (
            str(request.pointwise_mode),
            _float32_token(request.negative_slope),
            _float32_token(request.lower_clip),
            _float32_token(request.upper_clip),
            str(request.has_upper_clip),
            _float32_token(request.swish_beta),
            _float32_token(request.elu_alpha),
            _float32_token(request.softplus_beta),
            str(tiles_per_program),
            str(block_size),
        )
    )
    return ",".join(tokens)


def _unary_tiles_per_program(
    request: ParsedUnaryRequest, *, dense: bool
) -> int:
    return 1


def _unary_plan(
    request: ParsedUnaryRequest,
    source_sha256: str,
) -> ExecutionPlan:
    dense = can_use_dense_unary(request.input, request.output)
    function = (
        "unary_pointwise_contiguous_kernel"
        if dense
        else "unary_pointwise_strided_kernel"
    )
    table = "unary_contiguous" if dense else "unary_strided"
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name=table,
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _unary_runtime_arguments(request)
    tiles_per_program = _unary_tiles_per_program(request, dense=dense)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_UNARY_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_unary_full_signature(
                request,
                dense=dense,
                block_size=block_size,
                tiles_per_program=tiles_per_program,
            ),
            grid=(
                (request.n_elements + block_size * tiles_per_program - 1)
                // (block_size * tiles_per_program),
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
        source=_UNARY_SOURCE_RELATIVE_PATH,
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


def _ternary_runtime_arguments(
    request: ParsedTernaryRequest,
) -> tuple[RuntimeArgument, ...]:
    return (
        RuntimeArgument("tensor", "a", request.a.uid, None),
        RuntimeArgument("tensor", "b", request.b.uid, None),
        RuntimeArgument("tensor", "t", request.predicate.uid, None),
        RuntimeArgument("tensor", "output", request.output.uid, None),
        RuntimeArgument(
            "scalar_i32",
            "n_elements",
            None,
            _scalar_i32_bits(request.n_elements),
        ),
    )


def _ternary_full_signature(
    request: ParsedTernaryRequest,
    *,
    dense: bool,
    block_size: int,
) -> str:
    tokens = [
        _pointer_token(request.a.data_type, request.a.alignment),
        _pointer_token(request.b.data_type, request.b.alignment),
        _pointer_token(
            request.predicate.data_type, request.predicate.alignment
        ),
        _pointer_token(request.output.data_type, request.output.alignment),
        "i32",
    ]
    if not dense:
        tokens.extend(
            str(value)
            for value in ternary_pointwise_constants(
                request.a,
                request.b,
                request.predicate,
                request.output,
            )
        )
    tokens.append(str(block_size))
    return ",".join(tokens)


def _ternary_plan(
    request: ParsedTernaryRequest,
    source_sha256: str,
) -> ExecutionPlan:
    dense = can_use_dense_ternary(
        request.a, request.b, request.predicate, request.output
    )
    function = (
        "binary_select_tensor_kernel"
        if dense
        else "binary_select_strided_kernel"
    )
    table = "ternary_contiguous" if dense else "ternary_strided"
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name=table,
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _ternary_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_TERNARY_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_ternary_full_signature(
                request, dense=dense, block_size=block_size
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
        source=_TERNARY_SOURCE_RELATIVE_PATH,
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
