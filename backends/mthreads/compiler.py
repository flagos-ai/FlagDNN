"""MUSA/libtriton_jit compiler provider for the mthreads backend."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
from typing import Any

import yaml  # type: ignore[import-untyped]

from flagdnn_codegen.kernel_registry import (
    materialize_kernel_source,
    resolve_kernel_source,
    select_kernel_candidate,
)

from .compiler_graph import (
    ParsedAddRequest,
    ParsedAddSquareRequest,
    ParsedAttentionRequest,
    ParsedBatchnormInferenceRequest,
    ParsedBatchnormRequest,
    ParsedBinaryRequest,
    ParsedCompilerRequest,
    ParsedConvBiasReluRequest,
    ParsedConvolutionRequest,
    ParsedLayoutRequest,
    ParsedMatmulRequest,
    ParsedNormalizationRequest,
    ParsedPointwiseRequest,
    ParsedReductionRequest,
    ParsedTernaryRequest,
    ParsedUnaryRequest,
    parse_add_request,
    parse_add_square_request,
    parse_attention_request,
    parse_batchnorm_inference_request,
    parse_batchnorm_request,
    parse_binary_request,
    parse_compiler_request,
    parse_conv_bias_relu_request,
    parse_convolution_request,
    parse_layout_request,
    parse_matmul_request,
    parse_normalization_request,
    parse_pointwise_request,
    parse_reduction_request,
    parse_ternary_request,
    parse_unary_request,
)
from .compiler_identity import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROGRAM_VERSION,
    PROVIDER_NAME,
    SUPPORTED_ENGINE,
    build_compiler_identity,
    compiler_identity_dependencies as identity_dependencies,
)
from .compiler_tensor import (
    POINTER_TYPES,
    TensorSpec,
    can_use_dense_binary,
    can_use_dense_ternary,
    can_use_dense_unary,
    layout_constants,
    pointwise_constants,
    is_row_major_contiguous,
    matmul_constants,
    reduction_strided_constants,
    ternary_pointwise_constants,
    unary_pointwise_constants,
)
from .execution_plan import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)


_WORKSPACE_SIZE = 4096
_WORKSPACE_ALIGNMENT = 256
_BINARY_SOURCE_RELATIVE_PATH = "kernels/binary.py"
_UNARY_SOURCE_RELATIVE_PATH = "kernels/unary.py"
_IDENTITY_SOURCE_RELATIVE_PATH = "kernels/identity.py"
_TERNARY_SOURCE_RELATIVE_PATH = "kernels/ternary.py"
_LAYOUT_SOURCE_RELATIVE_PATH = "kernels/layout.py"
_REDUCTION_SOURCE_RELATIVE_PATH = "kernels/reduction.py"
_MATMUL_SOURCE_RELATIVE_PATH = "kernels/matmul.py"
_CONVOLUTION_SOURCE_RELATIVE_PATH = "kernels/convolution.py"
_COMPOSITE_SOURCE_RELATIVE_PATH = "kernels/composite.py"
_CONV_BIAS_RELU_SOURCE_RELATIVE_PATH = "kernels/conv_bias_relu.py"
_NORMALIZATION_SOURCE_RELATIVE_PATH = "kernels/normalization.py"
_ATTENTION_SOURCE_RELATIVE_PATH = "kernels/attention.py"
_SELECTION_CACHE = "tuning/stage-0.json"
_TUNING_ROOT_KEYS = {"schema_version", "backend", "defaults", "tables"}
_TUNING_DEFAULT_KEYS = {"block_size", "num_warps", "num_stages"}
_TUNING_TABLE_KEYS = {
    "strategy",
    "warmup",
    "repetitions",
    "dimensions",
}
_TUNING_DIMENSION_KEYS = {"block_size", "num_warps", "num_stages"}
_MAX_I32 = (1 << 31) - 1
_MATMUL_TLE_CONFIGS = {
    (32, 512, 512, 512): (128, 128, 32, 3, 2),
    (16, 1024, 1024, 1024): (256, 256, 32, 3, 2),
    (16, 2048, 2048, 512): (256, 256, 32, 3, 2),
    (8, 2048, 2048, 2048): (256, 256, 64, 3, 2),
    (32, 1024, 1024, 4096): (256, 256, 64, 3, 4),
    (4, 4096, 4096, 4096): (256, 256, 64, 3, 2),
}


def _bound_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_bound_names(element))
        return names
    return set()


def _definition_names(statement: ast.stmt) -> set[str]:
    if isinstance(
        statement,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
    ):
        return {statement.name}
    if isinstance(statement, ast.Assign):
        names: set[str] = set()
        for target in statement.targets:
            names.update(_bound_names(target))
        return names
    if isinstance(statement, ast.AnnAssign):
        return _bound_names(statement.target)
    return set()


def _statement_start(statement: ast.stmt) -> int:
    decorators = getattr(statement, "decorator_list", ())
    return min(
        [statement.lineno]
        + [decorator.lineno for decorator in decorators]
    )


def _materialize_mthreads_kernel_source(
    source_path: Path, candidate: Any
) -> bytes:
    """Materialize a compact MThreads artifact for oversized common modules.

    The common registry intentionally exposes only production entry points,
    while convolution.py also contains many auxiliary policy experiments.  A
    standalone MThreads artifact keeps the registry entries and the transitive
    closure of their top-level Python dependencies.  Other source families use
    the canonical module unchanged.
    """

    source_bytes = materialize_kernel_source(source_path, candidate)
    if candidate.source != "convolution.py":
        return source_bytes

    try:
        source_text = source_bytes.decode("utf-8")
        module = ast.parse(source_text, filename=str(source_path))
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValueError(
            "common convolution source cannot be candidate-sliced"
        ) from error

    definitions: dict[str, ast.stmt] = {}
    for statement in module.body:
        for name in _definition_names(statement):
            if name in definitions:
                raise ValueError(
                    f"common convolution source redefines {name!r}"
                )
            definitions[name] = statement

    selected = set(candidate.functions)
    missing = selected.difference(definitions)
    if missing:
        raise ValueError(
            "common convolution source is missing registry entry points: "
            + ", ".join(sorted(missing))
        )

    pending = list(selected)
    while pending:
        name = pending.pop()
        for item in ast.walk(definitions[name]):
            if (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id in definitions
                and item.id not in selected
            ):
                selected.add(item.id)
                pending.append(item.id)

    selected_statements = {
        id(statement)
        for name, statement in definitions.items()
        if name in selected
    }
    retained: list[ast.stmt] = []
    for index, statement in enumerate(module.body):
        module_docstring = (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        if (
            module_docstring
            or isinstance(statement, (ast.Import, ast.ImportFrom))
            or id(statement) in selected_statements
        ):
            retained.append(statement)
    if not retained:
        raise ValueError("common convolution source slicing retained nothing")

    lines = source_text.splitlines(keepends=True)
    first_statement = min(_statement_start(item) for item in module.body)
    segments = ["".join(lines[: first_statement - 1]).rstrip()]
    for statement in retained:
        if statement.end_lineno is None:
            raise ValueError(
                "common convolution source lacks AST end positions"
            )
        start = _statement_start(statement)
        segments.append(
            "".join(lines[start - 1 : statement.end_lineno]).rstrip()
        )
    materialized = "\n\n".join(
        segment for segment in segments if segment
    ) + "\n"
    materialized_bytes = materialized.encode("utf-8")

    materialized_module = ast.parse(
        materialized_bytes, filename=str(source_path)
    )
    materialized_functions = {
        item.name
        for item in materialized_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not set(candidate.functions).issubset(materialized_functions):
        raise ValueError(
            "candidate-sliced convolution source lost an entry point"
        )
    return materialized_bytes


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def compiler_identity(
    target_name: str,
    execution_engine: str = SUPPORTED_ENGINE,
) -> dict[str, Any]:
    return build_compiler_identity(target_name, execution_engine)


def compiler_identity_dependencies(
    target_name: str,
    execution_engine: str = SUPPORTED_ENGINE,
) -> tuple[Path, ...]:
    return identity_dependencies(target_name, execution_engine)


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys are invalid")


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _integer_dimension(
    value: object,
    context: str,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty array")
    result = tuple(
        _positive_integer(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicate candidates")
    return result


def _load_tuning(
    *,
    table_name: str,
    warp_size: int,
) -> tuple[
    tuple[int, int, int],
    tuple[tuple[int, int, int], ...],
    int,
    int,
]:
    source = Path(__file__).resolve().parent / "tuning/mthreads.yaml"
    if not source.is_file() or source.stat().st_size > (1 << 20):
        raise ValueError("mthreads tuning policy is missing or oversized")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mthreads tuning policy must be an object")
    _require_exact_keys(value, _TUNING_ROOT_KEYS, "tuning")
    if value["schema_version"] != 1 or value["backend"] != "mthreads":
        raise ValueError("mthreads tuning policy metadata is invalid")
    defaults = value["defaults"]
    tables = value["tables"]
    if not isinstance(defaults, dict) or not isinstance(tables, dict):
        raise ValueError("mthreads tuning policy sections are invalid")
    _require_exact_keys(defaults, _TUNING_DEFAULT_KEYS, "tuning.defaults")
    if set(tables) != {
        "binary_contiguous",
        "binary_strided",
        "unary_contiguous",
        "unary_strided",
        "ternary_contiguous",
        "ternary_strided",
        "layout",
        "reduction",
        "matmul",
        "convolution",
        "normalization",
        "batchnorm",
        "batchnorm_inference",
        "attention",
    }:
        raise ValueError("mthreads tuning tables are invalid")
    default_block_size = _positive_integer(
        defaults["block_size"], "default block_size"
    )
    default = (
        1
        if table_name == "reduction"
        else (
            32
            if table_name == "attention"
            else 64
            if table_name == "matmul"
            else (32 if table_name == "convolution" else default_block_size)
        ),
        _positive_integer(defaults["num_warps"], "default num_warps"),
        2
        if table_name in {"matmul", "convolution", "attention"}
        else _positive_integer(defaults["num_stages"], "default num_stages"),
    )

    table = tables[table_name]
    if not isinstance(table, dict):
        raise ValueError(f"tuning table {table_name} must be an object")
    _require_exact_keys(table, _TUNING_TABLE_KEYS, f"tuning.{table_name}")
    if table["strategy"] != "cartesian":
        raise ValueError(
            "mthreads pointwise tuning strategy must be cartesian"
        )
    warmup = _positive_integer(table["warmup"], "tuning warmup")
    repetitions = _positive_integer(
        table["repetitions"], "tuning repetitions"
    )
    dimensions = table["dimensions"]
    if not isinstance(dimensions, dict):
        raise ValueError("tuning dimensions must be an object")
    _require_exact_keys(
        dimensions,
        _TUNING_DIMENSION_KEYS,
        f"tuning.{table_name}.dimensions",
    )
    blocks = _integer_dimension(
        dimensions["block_size"], "tuning block_size"
    )
    warps = _integer_dimension(
        dimensions["num_warps"], "tuning num_warps"
    )
    stages = _integer_dimension(
        dimensions["num_stages"], "tuning num_stages"
    )
    candidates = tuple(itertools.product(blocks, warps, stages))
    if len(set(candidates)) != len(candidates) or len(candidates) > 32:
        raise ValueError(
            "mthreads pointwise tuning candidates are invalid"
        )
    for block_size, num_warps, num_stages in (*candidates, default):
        if (
            block_size > 1024
            or num_warps * warp_size > 1024
            or num_stages > 8
        ):
            raise ValueError(
                "mthreads pointwise tuning candidate exceeds device limits"
            )
    if default not in candidates:
        raise ValueError(
            "mthreads pointwise default is not a tuning candidate"
        )
    return default, candidates, warmup, repetitions


def _scalar_i32_bits(value: int) -> str:
    try:
        return struct.pack("<i", value).hex()
    except struct.error as error:
        raise ValueError("runtime int32 scalar is out of range") from error


def _scalar_f32_bits(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("runtime float32 scalar must be finite")
    try:
        return struct.pack("<f", value).hex()
    except (OverflowError, struct.error) as error:
        raise ValueError("runtime float32 scalar is out of range") from error


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


def _pointer_token(data_type: str, alignment: int) -> str:
    token = POINTER_TYPES[data_type]
    return f"{token}:16" if alignment >= 16 else token


def _float32_token(value: float) -> str:
    token = repr(value)
    if token in {"inf", "-inf", "nan"}:
        raise ValueError("float32 constexpr must be finite")
    return token


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
    dense = can_use_dense_binary(
        request.left, request.right, request.output
    )
    function = (
        "binary_contiguous_kernel"
        if dense
        else "binary_strided_kernel"
    )
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
            _pointer_token(
                request.left.data_type, request.left.alignment
            ),
            _pointer_token(
                request.right.data_type, request.right.alignment
            ),
            _pointer_token(
                request.output.data_type, request.output.alignment
            ),
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
                (
                    request.n_elements
                    + block_size * tiles_per_program
                    - 1
                )
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
        else "transpose_physical_copy_kernel"
        if physical_transpose
        else "slice_copy_kernel"
        if specialized_slice
        else "layout_copy_kernel"
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
        "outer"
        if function == "reduction_2d_kernel"
        else "output_elements"
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
                f"block-{block_m}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
            _pointer_token(
                request.output.data_type, request.output.alignment
            ),
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
            *matmul_constants(request.a, request.b, request.output),
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
        else (128, 4, 1)
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
    selected = (
        (launch,)
        if descriptor
        else candidates
        if request.autotune
        else (launch,)
    )
    function = (
        "matmul_tle_kernel"
        if tle_config is not None
        else "matmul_descriptor_kernel"
        if descriptor
        else "matmul_strided_kernel"
    )
    arguments = _matmul_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_MATMUL_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=(
                _matmul_tle_full_signature(request)
                if tle_config is not None
                else _matmul_descriptor_full_signature(
                    request, block_m=block_size
                )
                if descriptor
                else _matmul_full_signature(
                    request, block_size=block_size
                )
            ),
            grid=(
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


def _padded_convolution_tensor(
    tensor: Any, spatial_rank: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    leading = 3 - spatial_rank
    return (
        (*tensor.dimensions[:2], *([1] * leading), *tensor.dimensions[2:]),
        (*tensor.strides[:2], *([0] * leading), *tensor.strides[2:]),
    )


def _padded_convolution_spatial(
    values: tuple[int, ...], fill: int
) -> tuple[int, int, int]:
    return tuple((*([fill] * (3 - len(values))), *values))  # type: ignore[return-value]


def _convolution_function(request: ParsedConvolutionRequest) -> str:
    if request.operation in {"conv2d_fprop", "convolution_fprop"}:
        return {
            1: "conv1d_gemm_kernel",
            2: "conv2d_spatial_nchw_kernel",
            3: "conv3d_spatial_ncdhw_m_kernel",
        }[request.spatial_rank]
    if request.operation == "convolution_dgrad":
        return "conv_dgrad_nd_kernel"
    return "conv_wgrad_nd_kernel"


def _uses_stride2_tile4_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        request.operation == "convolution_dgrad"
        and request.spatial_rank == 2
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.filter.dimensions[2:] == (3, 3)
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
        and request.convolution_mode == 0
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
    )


def _uses_stride2_packed2_1d_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        request.operation == "convolution_dgrad"
        and request.spatial_rank == 1
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.filter.dimensions[2:] == (5,)
        and request.stride == (2,)
        and request.pre_padding == (2,)
        and request.post_padding == (1,)
        and request.dilation == (1,)
        and request.convolution_mode == 0
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
    )


def _uses_stride2_packed4_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        _uses_stride2_tile4_dgrad(request) and request.in_per_group <= 4
    )


def _stride2_packed4_dgrad_block_m(
    request: ParsedConvolutionRequest,
    block_size: int,
) -> int:
    return (
        block_size * 4
        if _uses_stride2_packed4_dgrad(request)
        else block_size
    )


def _convolution_runtime_arguments(
    request: ParsedConvolutionRequest,
) -> tuple[RuntimeArgument, ...]:
    if request.operation in {"conv2d_fprop", "convolution_fprop"}:
        return (
            RuntimeArgument("tensor", "input", request.image.uid, None),
            RuntimeArgument("tensor", "filter", request.filter.uid, None),
            RuntimeArgument(
                "tensor", "bias_placeholder", request.image.uid, None
            ),
            RuntimeArgument("tensor", "output", request.result.uid, None),
        )
    if request.operation == "convolution_dgrad":
        return (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "w", request.filter.uid, None),
            RuntimeArgument("tensor", "dx", request.image.uid, None),
        )
    return (
        RuntimeArgument("tensor", "dy", request.result.uid, None),
        RuntimeArgument("tensor", "x", request.image.uid, None),
        RuntimeArgument("tensor", "dw", request.filter.uid, None),
    )


def _convolution_full_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    image_dims, image_strides = _padded_convolution_tensor(
        request.image, request.spatial_rank
    )
    filter_dims, filter_strides = _padded_convolution_tensor(
        request.filter, request.spatial_rank
    )
    result_dims, result_strides = _padded_convolution_tensor(
        request.result, request.spatial_rank
    )
    _, _, xd, xh, xw = image_dims
    _, _, kd, kh, kw = filter_dims
    _, _, od, oh, ow = result_dims
    stride_d, stride_h, stride_w = _padded_convolution_spatial(
        request.stride, 1
    )
    pad_front, pad_top, pad_left = _padded_convolution_spatial(
        request.pre_padding, 0
    )
    dil_d, dil_h, dil_w = _padded_convolution_spatial(
        request.dilation, 1
    )
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    fprop_operation = request.operation in {
        "conv2d_fprop",
        "convolution_fprop",
    }
    fprop_reduction_extent = request.in_per_group * math.prod(
        request.filter.dimensions[2:]
    )
    input_precision = (
        1
        if request.image.data_type == "float32"
        and (not fprop_operation or fprop_reduction_extent >= 64)
        else 0
    )
    pointer = lambda tensor: _pointer_token(  # noqa: E731
        tensor.data_type, tensor.alignment
    )

    if function.startswith("conv") and request.operation in {
        "conv2d_fprop",
        "convolution_fprop",
    }:
        tokens = [
            pointer(request.image),
            pointer(request.filter),
            pointer(request.image),
            pointer(request.result),
        ]
    elif request.operation == "convolution_dgrad":
        tokens = [
            pointer(request.result),
            pointer(request.filter),
            pointer(request.image),
        ]
    else:
        tokens = [
            pointer(request.result),
            pointer(request.image),
            pointer(request.filter),
        ]

    if function == "conv1d_gemm_kernel":
        constants = (
            request.batch * ow,
            xw,
            ow,
            dtype_id,
            *request.image.strides,
            *request.filter.strides,
            1,
            *request.result.strides,
            request.in_per_group,
            request.out_per_group,
            kw,
            stride_w,
            pad_left,
            dil_w,
            0,
            block_size,
            block_size,
            block_size,
            8,
            input_precision,
        )
    elif function == "conv2d_spatial_nchw_kernel":
        constants = (
            xh,
            xw,
            oh,
            ow,
            request.in_channels,
            request.out_channels,
            request.in_per_group,
            request.out_per_group,
            request.groups,
            stride_h,
            stride_w,
            pad_top,
            pad_left,
            dil_h,
            dil_w,
            kh,
            kw,
            0,
            block_size,
            block_size,
            block_size,
            8,
            dtype_id,
            input_precision,
            image_strides[0],
            image_strides[1],
            image_strides[3],
            image_strides[4],
            filter_strides[0],
            filter_strides[1],
            filter_strides[3],
            filter_strides[4],
            result_strides[0],
            result_strides[1],
            result_strides[3],
            result_strides[4],
        )
    elif function == "conv3d_spatial_ncdhw_m_kernel":
        constants = (
            request.batch * od * oh * ow,
            xd,
            xh,
            xw,
            od,
            oh,
            ow,
            request.in_channels,
            request.out_channels,
            request.in_per_group,
            request.out_per_group,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dil_d,
            dil_h,
            dil_w,
            kd,
            kh,
            kw,
            0,
            block_size,
            block_size,
            block_size,
            8,
            *image_strides,
            *filter_strides,
            *result_strides,
            input_precision,
        )
    else:
        common = (
            xd,
            xh,
            xw,
            od,
            oh,
            ow,
            kd,
            kh,
            kw,
            request.in_per_group,
            request.out_per_group,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dil_d,
            dil_h,
            dil_w,
            request.convolution_mode,
            *result_strides,
            *image_strides,
            *filter_strides,
            1
            if request.image.data_type == "float32"
            and (
                request.operation == "convolution_wgrad"
                or _uses_stride2_tile4_dgrad(request)
                or _uses_stride2_packed2_1d_dgrad(request)
            )
            else 0,
        )
        if function == "conv_dgrad_nd_kernel":
            constants = (
                *common,
                request.batch * xd * xh * xw,
                _stride2_packed4_dgrad_block_m(request, block_size),
                block_size,
                block_size,
                8,
            )
        else:
            constants = (
                *common,
                request.batch * od * oh * ow,
                block_size,
                block_size,
                block_size,
            )
    tokens.extend(str(value) for value in constants)
    return ",".join(tokens)


def _convolution_grid(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> tuple[int, int, int]:
    image_dims, _ = _padded_convolution_tensor(
        request.image, request.spatial_rank
    )
    filter_dims, _ = _padded_convolution_tensor(
        request.filter, request.spatial_rank
    )
    result_dims, _ = _padded_convolution_tensor(
        request.result, request.spatial_rank
    )
    _, _, xd, xh, xw = image_dims
    _, _, kd, kh, kw = filter_dims
    _, _, od, oh, ow = result_dims
    ceil = lambda value: (value + block_size - 1) // block_size  # noqa: E731
    if function == "conv1d_gemm_kernel":
        return (
            ceil(request.batch * ow) * ceil(request.out_per_group),
            request.groups,
            1,
        )
    if function == "conv2d_spatial_nchw_kernel":
        return (
            ceil(oh * ow) * ceil(request.out_per_group),
            request.batch * request.groups,
            1,
        )
    if function == "conv3d_spatial_ncdhw_m_kernel":
        return (
            ceil(request.batch * od * oh * ow)
            * ceil(request.out_per_group),
            request.groups,
            1,
        )
    if function == "conv_dgrad_nd_kernel":
        rows = (
            request.batch * ((xw + 1) // 2)
            if _uses_stride2_packed2_1d_dgrad(request)
            else request.batch * od * oh * ow
            if _uses_stride2_tile4_dgrad(request)
            else request.batch * xd * xh * xw
        )
        block_m = _stride2_packed4_dgrad_block_m(request, block_size)
        channel_block = (
            block_size // 4
            if _uses_stride2_packed4_dgrad(request)
            else block_size
        )
        return (
            ((rows + block_m - 1) // block_m)
            * ((request.in_per_group + channel_block - 1) // channel_block),
            request.groups,
            1,
        )
    return (
        ceil(request.out_per_group) * ceil(request.in_per_group),
        kd * kh * kw,
        request.groups,
    )


def _uses_im2col_fprop(request: ParsedConvolutionRequest) -> bool:
    filter_area = math.prod(request.filter.dimensions[2:])
    standard_stride2_3x3 = (
        request.filter.dimensions[2:] == (3, 3)
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
    )
    stride2_3x3 = (
        standard_stride2_3x3 and request.in_per_group >= 64
    )
    fp32_stem = (
        standard_stride2_3x3
        and request.image.data_type == "float32"
        and request.image.dimensions == (1, 3, 640, 640)
        and request.filter.dimensions[1:] == (3, 3, 3)
        and request.out_per_group in {16, 32, 64, 96}
    )
    medium_batched = (
        request.batch >= 4
        and request.in_per_group >= 32
        and 2 <= filter_area <= 15
    )
    if not (
        request.operation in {"conv2d_fprop", "convolution_fprop"}
        and request.spatial_rank == 2
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.filter.data_type == request.image.data_type
        and request.result.data_type == request.image.data_type
        and request.groups == 1
        and request.convolution_mode == 0
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.filter)
        and is_row_major_contiguous(request.result)
        and (stride2_3x3 or fp32_stem or medium_batched)
    ):
        return False
    return _im2col_fprop_workspace_size(request) <= 512 * 1024 * 1024


def _im2col_fprop_geometry(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, tuple[int, int, int]]:
    output_area = math.prod(request.result.dimensions[2:])
    reduction_extent = (
        request.in_per_group * math.prod(request.filter.dimensions[2:])
    )
    column_strides = (
        reduction_extent * output_area,
        output_area,
        1,
    )
    return output_area, reduction_extent, column_strides


def _im2col_fprop_workspace_size(
    request: ParsedConvolutionRequest,
) -> int:
    output_area, reduction_extent, _ = _im2col_fprop_geometry(request)
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    raw_size = (
        request.batch * reduction_extent * output_area * element_size
    )
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _im2col_fprop_mm_blocks(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, int]:
    _, reduction_extent, _ = _im2col_fprop_geometry(request)
    if request.batch == 1 and reduction_extent >= 1024:
        return 64, 32, 64
    return (
        64,
        64,
        64 if request.image.data_type == "float32" else 32,
    )


def _im2col_fprop_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    output_area, reduction_extent, column_strides = (
        _im2col_fprop_geometry(request)
    )
    _, _, input_height, input_width = request.image.dimensions
    _, _, output_height, output_width = request.result.dimensions
    _, _, filter_height, filter_width = request.filter.dimensions
    workspace = _pointer_token(
        request.image.data_type, _WORKSPACE_ALIGNMENT
    )
    if function == "_conv_fprop2d_im2col_kernel":
        tokens = [
            _pointer_token(
                request.image.data_type, request.image.alignment
            ),
            workspace,
            str(output_area),
            str(input_height),
            str(input_width),
            str(output_height),
            str(output_width),
            str(request.in_per_group),
            str(filter_height),
            str(filter_width),
            *(str(value) for value in request.stride),
            *(str(value) for value in request.pre_padding),
            *(str(value) for value in request.dilation),
            *(str(value) for value in request.image.strides),
            *(str(value) for value in column_strides),
            str(block_size),
            "32",
        ]
    elif function == "_conv_fprop2d_im2col_mm_kernel":
        block_oc, block_m, block_k = _im2col_fprop_mm_blocks(request)
        if block_oc != block_size:
            raise ValueError("im2col Fprop candidate block differs")
        tokens = [
            _pointer_token(
                request.filter.data_type, request.filter.alignment
            ),
            workspace,
            _pointer_token(
                request.result.data_type, request.result.alignment
            ),
            str(output_area),
            str(request.out_per_group),
            str(request.in_per_group),
            str(filter_height),
            str(filter_width),
            *(str(value) for value in request.filter.strides),
            *(str(value) for value in request.result.strides),
            str(output_width),
            *(str(value) for value in column_strides),
            "1" if request.image.data_type == "float32" else "0",
            str(block_oc),
            str(block_m),
            str(block_k),
            "8",
        ]
    else:
        raise ValueError("unknown im2col Fprop stage function")
    if reduction_extent <= 0:
        raise ValueError("im2col Fprop reduction extent is invalid")
    return ",".join(tokens)


def _im2col_fprop_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    output_area, reduction_extent, _ = _im2col_fprop_geometry(request)
    if function == "_conv_fprop2d_im2col_kernel":
        block_size, num_warps, num_stages = (64, 4, 1)
        arguments = (
            RuntimeArgument("tensor", "input", request.image.uid, None),
            RuntimeArgument("workspace", "fprop_columns", None, None),
        )
        grid = (
            ((output_area + block_size - 1) // block_size)
            * ((reduction_extent + 31) // 32),
            request.batch,
            1,
        )
    elif function == "_conv_fprop2d_im2col_mm_kernel":
        block_size, num_warps, num_stages = (64, 8, 1)
        block_oc, block_m, _ = _im2col_fprop_mm_blocks(request)
        arguments = (
            RuntimeArgument("tensor", "filter", request.filter.uid, None),
            RuntimeArgument("workspace", "fprop_columns", None, None),
            RuntimeArgument("tensor", "output", request.result.uid, None),
        )
        grid = (
            ((request.out_per_group + block_oc - 1) // block_oc)
            * ((output_area + block_m - 1) // block_m),
            request.batch,
            1,
        )
    else:
        raise ValueError("unknown im2col Fprop stage function")
    variant = KernelVariant(
        variant_id=(
            f"block-{block_size}-warps-{num_warps}-"
            f"stages-{num_stages}"
        ),
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        source_sha256=source_sha256,
        function=function,
        full_signature=_im2col_fprop_signature(
            request,
            function=function,
            block_size=block_size,
        ),
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        arguments=arguments,
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=(variant,),
        autotune=AutotuneSpec(
            enabled=False,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _im2col_fprop_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    pack = _im2col_fprop_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_conv_fprop2d_im2col_kernel",
        dependencies=(),
    )
    matmul = _im2col_fprop_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_conv_fprop2d_im2col_mm_kernel",
        dependencies=(0,),
    )
    return ExecutionPlan(
        stages=(pack, matmul),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_im2col_fprop_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_dense_stride2_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        _uses_stride2_tile4_dgrad(request)
        and request.groups == 1
        and request.in_per_group > 4
    )


def _dense_dgrad_element_size(request: ParsedConvolutionRequest) -> int:
    return {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]


def _dense_dgrad_geometry(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, int, int]:
    loss_rows = request.batch * math.prod(request.result.dimensions[2:])
    packed_filter_elements = (
        16 * request.in_per_group * request.out_per_group
    )
    packed_loss_elements = 4 * request.out_per_group * loss_rows
    element_size = _dense_dgrad_element_size(request)
    packed_filter_bytes = packed_filter_elements * element_size
    aligned_filter_bytes = (
        (packed_filter_bytes + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    loss_offset = aligned_filter_bytes // element_size
    return (
        loss_rows,
        packed_filter_elements,
        packed_loss_elements,
        loss_offset,
    )


def _dense_dgrad_workspace_size(
    request: ParsedConvolutionRequest,
) -> int:
    _, _, packed_loss_elements, loss_offset = _dense_dgrad_geometry(request)
    element_size = _dense_dgrad_element_size(request)
    raw_size = (loss_offset + packed_loss_elements) * element_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _dense_dgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    loss_rows, _, _, loss_offset = _dense_dgrad_geometry(request)
    workspace = _pointer_token(
        request.image.data_type, _WORKSPACE_ALIGNMENT
    )
    if function == "_conv_dgrad2d_dense_pack_filter_kernel":
        tokens = [
            _pointer_token(
                request.filter.data_type, request.filter.alignment
            ),
            workspace,
            str(request.in_per_group),
            str(request.out_per_group),
            *(str(value) for value in request.filter.strides),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_dgrad2d_dense_pack_loss_kernel":
        tokens = [
            _pointer_token(
                request.result.data_type, request.result.alignment
            ),
            workspace,
            str(loss_offset),
            str(loss_rows),
            str(request.result.dimensions[2]),
            str(request.result.dimensions[3]),
            str(request.out_per_group),
            *(str(value) for value in request.result.strides),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_dgrad2d_dense_mm_kernel":
        tokens = [
            workspace,
            workspace,
            _pointer_token(request.image.data_type, request.image.alignment),
            str(loss_offset),
            str(loss_rows),
            str(request.result.dimensions[2]),
            str(request.result.dimensions[3]),
            str(request.image.dimensions[2]),
            str(request.image.dimensions[3]),
            str(request.in_per_group),
            str(request.out_per_group),
            *(str(value) for value in request.image.strides),
            "1" if request.image.data_type == "float32" else "0",
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
        ]
    else:
        raise ValueError("unknown dense Dgrad stage function")
    return ",".join(tokens)


def _dense_dgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    loss_rows, _, _, _ = _dense_dgrad_geometry(request)
    if function == "_conv_dgrad2d_dense_pack_filter_kernel":
        candidates = ((32, 8, 1),)
        arguments = (
            RuntimeArgument("tensor", "w", request.filter.uid, None),
            RuntimeArgument(
                "workspace", "dgrad_dense_filter", None, None
            ),
        )
        grid = lambda block: (  # noqa: E731
            (4 * request.in_per_group + block - 1) // block,
            (4 * request.out_per_group + block - 1) // block,
            1,
        )
        autotune = False
    elif function == "_conv_dgrad2d_dense_pack_loss_kernel":
        candidates = ((32, 8, 1),)
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "dgrad_dense_loss", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (4 * request.out_per_group + block - 1) // block,
            (loss_rows + block - 1) // block,
            1,
        )
        autotune = False
    elif function == "_conv_dgrad2d_dense_mm_kernel":
        default = (64, 8, 1)
        candidates = (
            tuple(
                (64, warps, stages)
                for warps in (4, 8)
                for stages in (1, 2)
            )
            if request.autotune
            else (default,)
        )
        arguments = (
            RuntimeArgument(
                "workspace", "dgrad_dense_filter", None, None
            ),
            RuntimeArgument("workspace", "dgrad_dense_loss", None, None),
            RuntimeArgument("tensor", "dx", request.image.uid, None),
        )
        grid = lambda block: (  # noqa: E731
            ((4 * request.in_per_group + block - 1) // block)
            * ((loss_rows + block - 1) // block),
            1,
            1,
        )
        autotune = request.autotune
    else:
        raise ValueError("unknown dense Dgrad stage function")
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_dense_dgrad_signature(
                request, function=function, block_size=block_size
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in candidates
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _dense_dgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    stage_specs = (
        ("_conv_dgrad2d_dense_pack_filter_kernel", ()),
        ("_conv_dgrad2d_dense_pack_loss_kernel", ()),
        ("_conv_dgrad2d_dense_mm_kernel", (0, 1)),
    )
    stages = tuple(
        _dense_dgrad_stage(
            request,
            source_sha256,
            stage_id=stage_id,
            function=function,
            dependencies=dependencies,
        )
        for stage_id, (function, dependencies) in enumerate(stage_specs)
    )
    return ExecutionPlan(
        stages=stages,
        external_binding_uids=request.external_binding_uids,
        workspace_size=_dense_dgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_nd_packed_wgrad(request: ParsedConvolutionRequest) -> bool:
    shape_key = (
        request.image.dimensions,
        request.result.dimensions,
        request.filter.dimensions,
        request.stride,
        request.pre_padding,
        request.post_padding,
        request.dilation,
    )
    supported_shapes = {
        (
            (16, 32, 256), (16, 64, 256), (64, 32, 3),
            (1,), (1,), (1,), (1,),
        ),
        (
            (2, 8, 8, 16, 16), (2, 16, 8, 16, 16),
            (16, 8, 3, 3, 3),
            (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
        ),
        (
            (1, 8, 10, 12, 14), (1, 12, 10, 11, 15),
            (12, 8, 2, 3, 3),
            (1, 1, 1), (1, 0, 1), (0, 1, 2), (1, 1, 1),
        ),
    }
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and shape_key in supported_shapes
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank in {1, 3}
        and request.groups == 1
        and request.convolution_mode == 0
    )


def _nd_packed_wgrad_spatial3(
    values: tuple[int, ...], fill: int,
) -> tuple[int, int, int]:
    return (fill,) * (3 - len(values)) + values


def _nd_packed_wgrad_num_splits(
    request: ParsedConvolutionRequest,
) -> int:
    total_rows = request.batch * math.prod(request.result.dimensions[2:])
    return 16 if total_rows >= 4096 else 8


def _nd_packed_wgrad_geometry(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, int, int, int]:
    output_area = math.prod(request.result.dimensions[2:])
    reduction_extent = (
        request.in_per_group * math.prod(request.filter.dimensions[2:])
    )
    total_rows = request.batch * output_area
    total_weights = request.out_per_group * reduction_extent
    num_splits = _nd_packed_wgrad_num_splits(request)
    return (
        output_area,
        reduction_extent,
        total_rows,
        total_weights,
        num_splits,
    )


def _nd_packed_wgrad_partial_size(
    request: ParsedConvolutionRequest,
) -> int:
    _, _, _, total_weights, num_splits = _nd_packed_wgrad_geometry(request)
    raw_size = num_splits * total_weights * 4
    return (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )


def _nd_packed_wgrad_column_offset(
    request: ParsedConvolutionRequest,
) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    return _nd_packed_wgrad_partial_size(request) // element_size


def _nd_packed_wgrad_workspace_size(
    request: ParsedConvolutionRequest,
) -> int:
    _, reduction_extent, total_rows, _, _ = _nd_packed_wgrad_geometry(request)
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    raw_size = (
        _nd_packed_wgrad_partial_size(request)
        + total_rows * reduction_extent * element_size
    )
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _nd_packed_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    output_area, reduction_extent, total_rows, total_weights, num_splits = (
        _nd_packed_wgrad_geometry(request)
    )
    rows_per_split = (total_rows + num_splits - 1) // num_splits
    input_spatial = _nd_packed_wgrad_spatial3(
        request.image.dimensions[2:], 1
    )
    output_spatial = _nd_packed_wgrad_spatial3(
        request.result.dimensions[2:], 1
    )
    kernel_spatial = _nd_packed_wgrad_spatial3(
        request.filter.dimensions[2:], 1
    )
    stride = _nd_packed_wgrad_spatial3(request.stride, 1)
    padding = _nd_packed_wgrad_spatial3(request.pre_padding, 0)
    dilation = _nd_packed_wgrad_spatial3(request.dilation, 1)
    input_strides = (
        request.image.strides[:2]
        + (0,) * (3 - request.spatial_rank)
        + request.image.strides[2:]
    )
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    packed = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    partial = _pointer_token("float32", _WORKSPACE_ALIGNMENT)
    if function == "_conv_wgrad_nd_im2row_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            packed,
            str(output_area),
            *(str(value) for value in input_spatial),
            str(output_spatial[1]),
            str(output_spatial[2]),
            str(request.in_per_group),
            *(str(value) for value in kernel_spatial),
            *(str(value) for value in stride),
            *(str(value) for value in padding),
            *(str(value) for value in dilation),
            *(str(value) for value in input_strides),
            str(reduction_extent),
            str(_nd_packed_wgrad_column_offset(request)),
            "1",
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad_nd_rowmajor_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            packed,
            partial,
            str(total_rows),
            str(rows_per_split),
            str(output_area),
            str(request.out_per_group),
            str(reduction_extent),
            str(request.result.strides[0]),
            str(request.result.strides[1]),
            str(request.result.strides[-1]),
            str(reduction_extent),
            str(_nd_packed_wgrad_column_offset(request)),
            "1",
            str(total_weights),
            str(reduction_extent),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad_nd_reduce_kernel":
        tokens = [
            partial,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(total_weights),
            str(num_splits),
            str(total_weights),
            str(block_size),
        ]
    else:
        raise ValueError("unknown ND packed Wgrad stage function")
    return ",".join(tokens)


def _nd_packed_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    output_area, reduction_extent, _, total_weights, num_splits = (
        _nd_packed_wgrad_geometry(request)
    )
    if function == "_conv_wgrad_nd_im2row_kernel":
        default = (64, 4, 1)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1,)))
        arguments = (
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_nd_columns", None, None),
        )
        grid = lambda block: (  # noqa: E731
            ((output_area + block - 1) // block)
            * ((reduction_extent + block - 1) // block),
            request.batch,
            1,
        )
    elif function == "_conv_wgrad_nd_rowmajor_kernel":
        default = (64, 8, 1)
        candidates = tuple(
            itertools.product((32, 64), (4, 8), (1, 2))
        )
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "wgrad_nd_columns", None, None),
            RuntimeArgument("workspace", "wgrad_nd_partial", None, None),
        )
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((reduction_extent + block - 1) // block),
            num_splits,
            1,
        )
    elif function == "_conv_wgrad_nd_reduce_kernel":
        default = (256, 4, 1)
        candidates = tuple(
            itertools.product((128, 256), (4, 8), (1,))
        )
        arguments = (
            RuntimeArgument("workspace", "wgrad_nd_partial", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        grid = lambda block: (  # noqa: E731
            (total_weights + block - 1) // block,
            1,
            1,
        )
    else:
        raise ValueError("unknown ND packed Wgrad stage function")
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_nd_packed_wgrad_signature(
                request, function=function, block_size=block_size
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _nd_packed_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    stage_specs = (
        ("_conv_wgrad_nd_im2row_kernel", ()),
        ("_conv_wgrad_nd_rowmajor_kernel", (0,)),
        ("_conv_wgrad_nd_reduce_kernel", (1,)),
    )
    stages = tuple(
        _nd_packed_wgrad_stage(
            request,
            source_sha256,
            stage_id=stage_id,
            function=function,
            dependencies=dependencies,
        )
        for stage_id, (function, dependencies) in enumerate(stage_specs)
    )
    return ExecutionPlan(
        stages=stages,
        external_binding_uids=request.external_binding_uids,
        workspace_size=_nd_packed_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_standard_wgrad(request: ParsedConvolutionRequest) -> bool:
    shape_key = (
        request.image.dimensions,
        request.result.dimensions,
        request.filter.dimensions,
        request.stride,
        request.pre_padding,
        request.post_padding,
    )
    supported_shapes = {
        (
            (8, 64, 56, 56), (8, 128, 28, 28), (128, 64, 3, 3),
            (2, 2), (1, 1), (1, 1),
        ),
        (
            (8, 32, 32, 32), (8, 64, 32, 32), (64, 32, 3, 3),
            (1, 1), (1, 1), (1, 1),
        ),
        (
            (8, 64, 28, 28), (8, 128, 28, 28), (128, 64, 1, 1),
            (1, 1), (0, 0), (0, 0),
        ),
    }
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and shape_key in supported_shapes
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank == 2
        and request.groups == 1
        and request.convolution_mode == 0
        and request.dilation == (1, 1)
    )


def _standard_wgrad_num_splits(request: ParsedConvolutionRequest) -> int:
    return 8 if request.filter.dimensions[2:] == (1, 1) else request.batch


def _standard_wgrad_partial_size(request: ParsedConvolutionRequest) -> int:
    kh, kw = request.filter.dimensions[2:]
    raw_size = (
        _standard_wgrad_num_splits(request)
        * request.out_per_group
        * request.in_per_group
        * kh
        * kw
        * 4
    )
    return (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )


def _standard_wgrad_column_offset(
    request: ParsedConvolutionRequest,
) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    return _standard_wgrad_partial_size(request) // element_size


def _standard_wgrad_workspace_size(request: ParsedConvolutionRequest) -> int:
    kh, kw = request.filter.dimensions[2:]
    partial_size = _standard_wgrad_partial_size(request)
    if (kh, kw) == (1, 1):
        raw_size = partial_size
    else:
        element_size = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
        }[request.image.data_type]
        _, _, oh, ow = request.result.dimensions
        column_size = (
            request.batch
            * oh
            * ow
            * request.in_per_group
            * kh
            * kw
            * element_size
        )
        raw_size = partial_size + column_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _standard_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    _, _, xh, xw = request.image.dimensions
    _, _, oh, ow = request.result.dimensions
    _, _, kh, kw = request.filter.dimensions
    total_rows = request.batch * oh * ow
    num_splits = _standard_wgrad_num_splits(request)
    rows_per_split = (total_rows + num_splits - 1) // num_splits
    partial_stride_oc = request.in_per_group * kh * kw
    partial_stride_split = request.out_per_group * partial_stride_oc
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    partial = _pointer_token("float32", _WORKSPACE_ALIGNMENT)
    packed = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    pointers = [
        _pointer_token(request.result.data_type, request.result.alignment),
        _pointer_token(request.image.data_type, request.image.alignment),
        partial,
    ]
    if function == "_conv_wgrad2d_im2row_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            packed,
            str(oh * ow),
            str(xh),
            str(xw),
            str(ow),
            str(request.in_per_group),
            str(kh),
            str(kw),
            *(str(value) for value in request.stride),
            *(str(value) for value in request.pre_padding),
            *(str(value) for value in request.dilation),
            *(str(value) for value in request.image.strides),
            str(partial_stride_oc),
            str(_standard_wgrad_column_offset(request)),
            "1",
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad2d_rowmajor_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            packed,
            partial,
            str(oh * ow),
            str(request.out_per_group),
            str(request.in_per_group),
            str(kh),
            str(kw),
            *(str(value) for value in request.result.strides),
            str(_standard_wgrad_column_offset(request)),
            str(partial_stride_oc),
            "1",
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad2d_direct_split_kernel":
        tokens = [
            *pointers,
            str(total_rows),
            str(rows_per_split),
            str(oh * ow),
            str(xh),
            str(xw),
            str(ow),
            str(request.out_per_group),
            str(request.in_per_group),
            str(kh),
            str(kw),
            *(str(value) for value in request.stride),
            *(str(value) for value in request.pre_padding),
            *(str(value) for value in request.dilation),
            *(str(value) for value in request.result.strides),
            *(str(value) for value in request.image.strides),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            "64",
        ]
    elif function == "_conv_wgrad2d_1x1_split_kernel":
        tokens = [
            *pointers,
            str(total_rows),
            str(rows_per_split),
            str(oh * ow),
            str(request.image.dimensions[1]),
            str(request.result.dimensions[1]),
            str(request.in_per_group),
            str(request.out_per_group),
            str(request.groups),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            "64",
        ]
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        tokens = [
            partial,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(request.out_per_group * partial_stride_oc),
            str(partial_stride_oc),
            str(request.in_per_group),
            str(kh),
            str(kw),
            str(num_splits),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            *(str(value) for value in request.filter.strides),
            str(block_size),
        ]
    else:
        raise ValueError("unknown standard Wgrad stage function")
    return ",".join(tokens)


def _standard_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    num_splits = _standard_wgrad_num_splits(request)
    kh, kw = request.filter.dimensions[2:]
    if function == "_conv_wgrad2d_im2row_kernel":
        default = (64, 4, 1)
        candidates = tuple(
            itertools.product((32, 64), (4, 8), (1,))
        )
        arguments = (
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_columns", None, None),
        )
        cik = request.in_per_group * kh * kw
        _, _, oh, ow = request.result.dimensions
        grid = lambda block: (  # noqa: E731
            ((oh * ow + block - 1) // block)
            * ((cik + block - 1) // block),
            request.batch,
            1,
        )
    elif function == "_conv_wgrad2d_rowmajor_kernel":
        default = (64, 8, 1)
        candidates = tuple(
            itertools.product((32, 64), (4, 8), (1, 2))
        )
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "wgrad_columns", None, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        cik = request.in_per_group * kh * kw
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((cik + block - 1) // block),
            request.batch,
            1,
        )
    elif function == "_conv_wgrad2d_direct_split_kernel":
        default = (64, 8, 1)
        candidates = tuple(
            itertools.product((32, 64), (4, 8), (1, 2))
        )
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        cik = request.in_per_group * kh * kw
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((cik + block - 1) // block),
            num_splits,
            1,
        )
    elif function == "_conv_wgrad2d_1x1_split_kernel":
        default = (16, 4, 2)
        candidates = tuple(
            itertools.product((16, 32), (4, 8), (1, 2))
        )
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((request.in_per_group + block - 1) // block),
            num_splits * request.groups,
            1,
        )
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        default = (256, 4, 1)
        candidates = tuple(
            itertools.product((128, 256), (4, 8), (1,))
        )
        arguments = (
            RuntimeArgument("workspace", "wgrad_partial", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        total = request.out_per_group * request.in_per_group * kh * kw
        grid = lambda block: (  # noqa: E731
            (total + block - 1) // block,
            1,
            1,
        )
    else:
        raise ValueError("unknown standard Wgrad stage function")
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_standard_wgrad_signature(
                request, function=function, block_size=block_size
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _standard_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    if request.filter.dimensions[2:] == (1, 1):
        stage_specs = (
            ("_conv_wgrad2d_1x1_split_kernel", ()),
            ("_conv_wgrad2d_stem_reduce_kernel", (0,)),
        )
    else:
        stage_specs = (
            ("_conv_wgrad2d_im2row_kernel", ()),
            ("_conv_wgrad2d_rowmajor_kernel", (0,)),
            ("_conv_wgrad2d_stem_reduce_kernel", (1,)),
        )
    stages = tuple(
        _standard_wgrad_stage(
            request,
            source_sha256,
            stage_id=stage_id,
            function=function,
            dependencies=dependencies,
        )
        for stage_id, (function, dependencies) in enumerate(stage_specs)
    )
    return ExecutionPlan(
        stages=stages,
        external_binding_uids=request.external_binding_uids,
        workspace_size=_standard_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_stem_wgrad(request: ParsedConvolutionRequest) -> bool:
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.image.dimensions == (1, 3, 640, 640)
        and request.result.dimensions[0] == 1
        and request.result.dimensions[1] in {16, 32, 64, 96}
        and request.result.dimensions[2:] == (320, 320)
        and request.filter.dimensions
        == (request.result.dimensions[1], 3, 3, 3)
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank == 2
        and request.groups == 1
        and request.convolution_mode == 0
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
    )


def _stem_wgrad_workspace_size(request: ParsedConvolutionRequest) -> int:
    raw_size = 64 * request.out_per_group * request.in_per_group * 9 * 4
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _stem_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    partial_stride_split = request.out_per_group * request.in_per_group * 9
    partial_stride_oc = request.in_per_group * 9
    partial = _pointer_token("float32", _WORKSPACE_ALIGNMENT)
    if function == "_conv_wgrad2d_stem_split_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            _pointer_token(request.image.data_type, request.image.alignment),
            partial,
            "5",
            "320",
            "320",
            "640",
            "640",
            str(request.out_per_group),
            str(request.in_per_group),
            "3",
            "3",
            "2",
            "2",
            "1",
            "1",
            "1",
            "1",
            str(request.result.strides[1]),
            str(request.result.strides[2]),
            str(request.result.strides[3]),
            str(request.image.strides[1]),
            str(request.image.strides[2]),
            str(request.image.strides[3]),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            "32",
            "64",
        ]
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        tokens = [
            partial,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(request.out_per_group * request.in_per_group * 9),
            str(request.in_per_group * 9),
            str(request.in_per_group),
            "3",
            "3",
            "64",
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(request.filter.strides[0]),
            str(request.filter.strides[1]),
            str(request.filter.strides[2]),
            str(request.filter.strides[3]),
            str(block_size),
        ]
    else:
        raise ValueError("unknown stem Wgrad stage function")
    return ",".join(tokens)


def _stem_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    if function == "_conv_wgrad2d_stem_split_kernel":
        default = (64, 4, 2)
        candidates = tuple(
            itertools.product((32, 64), (4, 8), (1, 2))
        )
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (request.out_per_group + block - 1) // block,
            64,
            1,
        )
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        default = (256, 4, 1)
        candidates = tuple(
            itertools.product((128, 256), (4, 8), (1,))
        )
        arguments = (
            RuntimeArgument("workspace", "wgrad_partial", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        total = request.out_per_group * request.in_per_group * 9
        grid = lambda block: (  # noqa: E731
            (total + block - 1) // block,
            1,
            1,
        )
    else:
        raise ValueError("unknown stem Wgrad stage function")
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_stem_wgrad_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _stem_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    split = _stem_wgrad_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_conv_wgrad2d_stem_split_kernel",
        dependencies=(),
    )
    reduce = _stem_wgrad_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_conv_wgrad2d_stem_reduce_kernel",
        dependencies=(0,),
    )
    return ExecutionPlan(
        stages=(split, reduce),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_stem_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_p5_wgrad(request: ParsedConvolutionRequest) -> bool:
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.image.dimensions[0] == 1
        and request.image.dimensions[2:] == (40, 40)
        and request.result.dimensions[0] == 1
        and request.result.dimensions[2:] == (20, 20)
        and request.filter.dimensions[2:] == (3, 3)
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank == 2
        and request.groups == 1
        and request.convolution_mode == 0
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
    )


def _p5_wgrad_workspace_size(request: ParsedConvolutionRequest) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    raw_size = 400 * request.in_per_group * 9 * element_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _p5_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    workspace = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    if function == "_conv_wgrad2d_p5_pack_image_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            workspace,
            str(request.in_per_group),
            str(request.image.strides[1]),
            str(request.image.strides[2]),
            str(request.image.strides[3]),
            "400",
            str(request.in_per_group * 9),
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
        ]
    elif function == "_conv_wgrad2d_p5_mm_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            workspace,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(request.out_per_group),
            str(request.in_per_group * 9),
            "400",
            str(dtype_id),
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
        ]
    else:
        raise ValueError("unknown P5 Wgrad stage function")
    return ",".join(tokens)


def _p5_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    stage_candidates = candidates
    if (
        function == "_conv_wgrad2d_p5_mm_kernel"
        and request.image.data_type != "float32"
    ):
        stage_candidates = (
            *candidates,
            *((64, warps, stages) for warps in (4, 8) for stages in (1, 2)),
        )
    selected = stage_candidates if request.autotune else (default,)
    if function == "_conv_wgrad2d_p5_pack_image_kernel":
        arguments = (
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "packed_image", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (400 + block - 1) // block,
            (request.in_per_group * 9 + block - 1) // block,
            1,
        )
    elif function == "_conv_wgrad2d_p5_mm_kernel":
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "packed_image", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((request.in_per_group * 9 + block - 1) // block),
            1,
            1,
        )
    else:
        raise ValueError("unknown P5 Wgrad stage function")
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_p5_wgrad_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _p5_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    pack = _p5_wgrad_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_conv_wgrad2d_p5_pack_image_kernel",
        dependencies=(),
    )
    matmul = _p5_wgrad_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_conv_wgrad2d_p5_mm_kernel",
        dependencies=(0,),
    )
    return ExecutionPlan(
        stages=(pack, matmul),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_p5_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )



def _convolution_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    if _uses_dense_stride2_dgrad(request):
        return _dense_dgrad_plan(request, source_sha256)
    if _uses_im2col_fprop(request):
        return _im2col_fprop_plan(request, source_sha256)
    function = _convolution_function(request)
    if _uses_nd_packed_wgrad(request):
        return _nd_packed_wgrad_plan(request, source_sha256)
    if _uses_standard_wgrad(request):
        return _standard_wgrad_plan(request, source_sha256)
    if _uses_stem_wgrad(request):
        return _stem_wgrad_plan(request, source_sha256)
    if _uses_p5_wgrad(request):
        return _p5_wgrad_plan(request, source_sha256)
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _convolution_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_convolution_full_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=_convolution_grid(
                request,
                function=function,
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
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
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
    output_spatial = request.output.dimensions[2] * request.output.dimensions[3]
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
            ),
            source=_CONV_BIAS_RELU_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function="conv_bias_relu_2d_kernel",
            full_signature=_conv_bias_relu_full_signature(
                request, block_size=block_size
            ),
            grid=(
                ((output_spatial + block_size - 1) // block_size)
                * (
                    (request.out_per_group + block_size - 1)
                    // block_size
                ),
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
        RuntimeArgument(
            "scalar_i32", name, None, _scalar_i32_bits(value)
        )
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
    function = (
        "batch_norm_nchw_kernel" if specialized else "batch_norm_kernel"
    )
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
    arguments = _batchnorm_runtime_arguments(
        request, specialized=specialized
    )
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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
        RuntimeArgument(
            "scalar_i32", name, None, _scalar_i32_bits(value)
        )
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
                f"block-{block_size}-warps-{num_warps}-"
                f"stages-{num_stages}"
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


_LOG2_E = 1.4426950408889634
_ATTENTION_ELEMENT_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}
_ATTENTION_FORWARD_POINTERS = (
    "q_ptr",
    "k_ptr",
    "v_ptr",
    "bias_ptr",
    "o_ptr",
    "stats_ptr",
)
_ATTENTION_FORWARD_I32 = (
    "HQ",
    "SQ",
    "SKV",
    "q_per_k",
    "q_per_v",
    "min_diag",
    "max_diag",
    "stride_qb",
    "stride_qh",
    "stride_qm",
    "stride_qd",
    "stride_kb",
    "stride_kh",
    "stride_kn",
    "stride_kd",
    "stride_vb",
    "stride_vh",
    "stride_vn",
    "stride_vd",
    "stride_bias_b",
    "stride_bias_h",
    "stride_bias_m",
    "stride_bias_n",
    "stride_ob",
    "stride_oh",
    "stride_om",
    "stride_od",
    "stride_sb",
    "stride_sh",
    "stride_sm",
)
_ATTENTION_BACKWARD_BASE_STRIDES = (
    "stride_qb",
    "stride_qh",
    "stride_qm",
    "stride_qd",
    "stride_kb",
    "stride_kh",
    "stride_kn",
    "stride_kd",
    "stride_vb",
    "stride_vh",
    "stride_vn",
    "stride_vd",
    "stride_bias_b",
    "stride_bias_h",
    "stride_bias_m",
    "stride_bias_n",
    "stride_dob",
    "stride_doh",
    "stride_dom",
    "stride_dod",
    "stride_sb",
    "stride_sh",
    "stride_sm",
)
_ATTENTION_PARAMETERS = {
    "_sdpa_fwd_kernel": (
        *_ATTENTION_FORWARD_POINTERS,
        "qk_scale",
        *_ATTENTION_FORWARD_I32,
        "HEAD_DIM",
        "V_DIM",
        "ELEM_SIZE",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D",
        "BLOCK_DV",
        "HAS_BIAS",
        "BANDED",
        "GENERATE_STATS",
        "REVERSE_CAUSAL",
    ),
    "_zero_contiguous_kernel": (
        "ptr",
        "n_elements",
        "BLOCK",
    ),
    "_sdpa_bwd_dq_dbias_kernel": (
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "bias_ptr",
        "o_ptr",
        "do_ptr",
        "stats_ptr",
        "delta_ptr",
        "dq_ptr",
        "dbias_ptr",
        "attn_scale",
        "HQ",
        "SQ",
        "SKV",
        "q_per_k",
        "q_per_v",
        "min_diag",
        "max_diag",
        *_ATTENTION_BACKWARD_BASE_STRIDES[:16],
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_od",
        *_ATTENTION_BACKWARD_BASE_STRIDES[16:],
        "stride_delta_b",
        "stride_delta_h",
        "stride_delta_m",
        "stride_dqb",
        "stride_dqh",
        "stride_dqm",
        "stride_dqd",
        "stride_dbias_b",
        "stride_dbias_h",
        "stride_dbias_m",
        "stride_dbias_n",
        "HEAD_DIM",
        "V_DIM",
        "DBIAS_BATCHES",
        "DBIAS_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D_FULL",
        "BLOCK_D_OUT",
        "BLOCK_DV",
        "FULL_ATTENTION",
        "HAS_BIAS",
        "HAS_DBIAS",
        "DBIAS_REDUCE",
        "BANDED",
        "CAUSAL_TOP_LEFT",
    ),
    "_sdpa_bwd_dk_kernel": (
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "bias_ptr",
        "do_ptr",
        "stats_ptr",
        "delta_ptr",
        "dk_ptr",
        "attn_scale",
        "HKV",
        "SQ",
        "SKV",
        "min_diag",
        "max_diag",
        *_ATTENTION_BACKWARD_BASE_STRIDES,
        "stride_delta_b",
        "stride_delta_h",
        "stride_delta_m",
        "stride_dkb",
        "stride_dkh",
        "stride_dkn",
        "stride_dkd",
        "HEAD_DIM",
        "V_DIM",
        "Q_PER",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D_FULL",
        "BLOCK_D_OUT",
        "BLOCK_DV",
        "FULL_ATTENTION",
        "HAS_BIAS",
        "BANDED",
        "CAUSAL_TOP_LEFT",
    ),
    "_sdpa_bwd_dkdv_kernel": (
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "bias_ptr",
        "do_ptr",
        "stats_ptr",
        "delta_ptr",
        "dk_ptr",
        "dv_ptr",
        "attn_scale",
        "HKV",
        "SQ",
        "SKV",
        "min_diag",
        "max_diag",
        *_ATTENTION_BACKWARD_BASE_STRIDES,
        "stride_delta_b",
        "stride_delta_h",
        "stride_delta_m",
        "stride_dkb",
        "stride_dkh",
        "stride_dkn",
        "stride_dkd",
        "stride_dvb",
        "stride_dvh",
        "stride_dvn",
        "stride_dvd",
        "HEAD_DIM",
        "Q_PER",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D_FULL",
        "BLOCK_D_OUT",
        "FULL_ATTENTION",
        "HAS_BIAS",
        "BANDED",
        "CAUSAL_TOP_LEFT",
    ),
    "_sdpa_bwd_dv_kernel": (
        "q_ptr",
        "k_ptr",
        "bias_ptr",
        "do_ptr",
        "stats_ptr",
        "dv_ptr",
        "attn_scale",
        "HKV",
        "SQ",
        "SKV",
        "min_diag",
        "max_diag",
        *_ATTENTION_BACKWARD_BASE_STRIDES[:8],
        *_ATTENTION_BACKWARD_BASE_STRIDES[12:],
        "stride_dvb",
        "stride_dvh",
        "stride_dvn",
        "stride_dvd",
        "HEAD_DIM",
        "V_DIM",
        "Q_PER",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D_FULL",
        "BLOCK_DV_OUT",
        "FULL_ATTENTION",
        "HAS_BIAS",
        "BANDED",
        "CAUSAL_TOP_LEFT",
    ),
    "_zero_sdpa_fp8_fwd_amax_kernel": (
        "amax_s_ptr",
        "amax_o_ptr",
    ),
    "_sdpa_fp8_fwd_kernel": (
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "bias_ptr",
        "o_ptr",
        "stats_ptr",
        "amax_s_ptr",
        "amax_o_ptr",
        "descale_q_ptr",
        "descale_k_ptr",
        "descale_v_ptr",
        "descale_s_ptr",
        "scale_s_ptr",
        "scale_o_ptr",
        "attn_scale",
        *_ATTENTION_FORWARD_I32,
        "HEAD_DIM",
        "V_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D",
        "BLOCK_DV",
        "HAS_BIAS",
        "BANDED",
        "GENERATE_STATS",
        "REVERSE_CAUSAL",
    ),
    "_zero_sdpa_fp8_bwd_amax_kernel": (
        "amax_dq_ptr",
        "amax_dk_ptr",
        "amax_dv_ptr",
        "amax_dp_ptr",
    ),
    "_sdpa_fp8_bwd_dq_kernel": (
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "do_ptr",
        "stats_ptr",
        "dq_ptr",
        "amax_dq_ptr",
        "descale_q_ptr",
        "descale_k_ptr",
        "descale_v_ptr",
        "descale_o_ptr",
        "descale_do_ptr",
        "descale_dp_ptr",
        "scale_dq_ptr",
        "scale_dp_ptr",
        "attn_scale",
        "HQ",
        "SQ",
        "SKV",
        "q_per_k",
        "q_per_v",
        "min_diag",
        "max_diag",
        *_ATTENTION_BACKWARD_BASE_STRIDES[:12],
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_od",
        *_ATTENTION_BACKWARD_BASE_STRIDES[16:],
        "stride_dqb",
        "stride_dqh",
        "stride_dqm",
        "stride_dqd",
        "HEAD_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D",
        "BANDED",
        "FULL_BLOCKS",
        "CAUSAL_TOP_LEFT",
    ),
    "_sdpa_fp8_bwd_dkdv_kernel": (
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "do_ptr",
        "stats_ptr",
        "dk_ptr",
        "dv_ptr",
        "amax_dk_ptr",
        "amax_dv_ptr",
        "amax_dp_ptr",
        "descale_q_ptr",
        "descale_k_ptr",
        "descale_v_ptr",
        "descale_o_ptr",
        "descale_do_ptr",
        "descale_s_ptr",
        "descale_dp_ptr",
        "scale_s_ptr",
        "scale_dk_ptr",
        "scale_dv_ptr",
        "scale_dp_ptr",
        "attn_scale",
        "HKV",
        "SQ",
        "SKV",
        "min_diag",
        "max_diag",
        *_ATTENTION_BACKWARD_BASE_STRIDES[:12],
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_od",
        *_ATTENTION_BACKWARD_BASE_STRIDES[16:],
        "stride_dkb",
        "stride_dkh",
        "stride_dkn",
        "stride_dkd",
        "stride_dvb",
        "stride_dvh",
        "stride_dvn",
        "stride_dvd",
        "HEAD_DIM",
        "Q_PER",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D",
        "BANDED",
        "FULL_BLOCKS",
        "CAUSAL_TOP_LEFT",
    ),
}


def _attention_constexpr_token(value: int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return _float32_token(value)


def _attention_full_signature(
    function: str,
    runtime_tokens: dict[str, str],
    constants: dict[str, int | float | bool],
) -> str:
    parameters = _ATTENTION_PARAMETERS[function]
    if (
        set(runtime_tokens).intersection(constants)
        or set(runtime_tokens).union(constants) != set(parameters)
        or len(set(parameters)) != len(parameters)
    ):
        raise ValueError(
            f"Attention signature coverage is invalid for {function}"
        )
    return ",".join(
        runtime_tokens[name]
        if name in runtime_tokens
        else _attention_constexpr_token(constants[name])
        for name in parameters
    )


def _attention_tensor_pointer(
    request: ParsedAttentionRequest,
    parameter: str,
    port: str,
    *,
    semantic_name: str | None = None,
) -> tuple[str, str, RuntimeArgument]:
    tensor = request.tensor(port)
    if tensor.virtual:
        raise ValueError(f"Attention {port} cannot be a tensor argument")
    return (
        parameter,
        _pointer_token(tensor.data_type, tensor.alignment),
        RuntimeArgument(
            "tensor", semantic_name or port, tensor.uid, None
        ),
    )


def _attention_workspace_pointer(
    parameter: str,
    semantic_name: str,
) -> tuple[str, str, RuntimeArgument]:
    return (
        parameter,
        _pointer_token("float32", _WORKSPACE_ALIGNMENT),
        RuntimeArgument("workspace", semantic_name, None, None),
    )


def _attention_scalar_argument(
    name: str,
    kind: str,
    value: int | float,
) -> tuple[str, str, RuntimeArgument]:
    if kind == "i32":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Attention scalar {name} is not int32")
        return (
            name,
            "i32",
            RuntimeArgument(
                "scalar_i32", name, None, _scalar_i32_bits(value)
            ),
        )
    if kind == "fp32":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Attention scalar {name} is not float32")
        return (
            name,
            "fp32",
            RuntimeArgument(
                "scalar_f32",
                name,
                None,
                _scalar_f32_bits(float(value)),
            ),
        )
    raise ValueError(f"Attention scalar {name} has an invalid ABI kind")


def _attention_variant(
    *,
    source_sha256: str,
    function: str,
    block_size: int,
    num_warps: int,
    num_stages: int,
    grid: tuple[int, int, int],
    pointers: tuple[tuple[str, str, RuntimeArgument], ...],
    scalars: tuple[tuple[str, str, RuntimeArgument], ...],
    constants: dict[str, int | float | bool],
    fixed: bool = False,
) -> KernelVariant:
    runtime = (*pointers, *scalars)
    runtime_tokens = {name: token for name, token, _ in runtime}
    runtime_arguments = {name: argument for name, _, argument in runtime}
    if len(runtime_tokens) != len(runtime):
        raise ValueError("Attention runtime argument name is duplicated")
    arguments = tuple(
        runtime_arguments[name]
        for name in _ATTENTION_PARAMETERS[function]
        if name in runtime_arguments
    )
    variant_id = (
        "fixed-warps-4-stages-1"
        if fixed
        else (
            f"block-{block_size}-warps-{num_warps}-"
            f"stages-{num_stages}"
        )
    )
    return KernelVariant(
        variant_id=variant_id,
        source=_ATTENTION_SOURCE_RELATIVE_PATH,
        source_sha256=source_sha256,
        function=function,
        full_signature=_attention_full_signature(
            function, runtime_tokens, constants
        ),
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        arguments=arguments,
    )


def _attention_tuned_stage(
    request: ParsedAttentionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
    pointers: tuple[tuple[str, str, RuntimeArgument], ...],
    scalars: tuple[tuple[str, str, RuntimeArgument], ...],
    constants_factory: Any,
    grid_factory: Any,
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="attention",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        _attention_variant(
            source_sha256=source_sha256,
            function=function,
            block_size=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid_factory(block_size),
            pointers=pointers,
            scalars=scalars,
            constants=constants_factory(block_size),
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_ATTENTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _attention_fixed_stage(
    request: ParsedAttentionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
    grid: tuple[int, int, int],
    pointers: tuple[tuple[str, str, RuntimeArgument], ...],
    scalars: tuple[tuple[str, str, RuntimeArgument], ...] = (),
    constants: dict[str, int | float | bool] | None = None,
) -> KernelStage:
    variant = _attention_variant(
        source_sha256=source_sha256,
        function=function,
        block_size=1,
        num_warps=4,
        num_stages=1,
        grid=grid,
        pointers=pointers,
        scalars=scalars,
        constants={} if constants is None else constants,
        fixed=True,
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_ATTENTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=(variant,),
        autotune=AutotuneSpec(
            enabled=False,
            warmup=3,
            repetitions=10,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _attention_full_dimension_block(value: int) -> int:
    return max(16, 1 << (value - 1).bit_length())


def _attention_output_dimension_block(value: int, block_size: int) -> int:
    return min(block_size, _attention_full_dimension_block(value))


def _attention_bias_strides(
    request: ParsedAttentionRequest,
) -> tuple[int, int, int, int]:
    bias = request.optional_tensor("bias")
    if bias is None:
        return (0, 0, 0, 0)
    return (
        0 if bias.dimensions[0] == 1 else bias.strides[0],
        0 if bias.dimensions[1] == 1 else bias.strides[1],
        bias.strides[2],
        bias.strides[3],
    )


def _attention_tensor_strides(
    tensor: TensorSpec,
) -> dict[str, int]:
    if len(tensor.strides) != 4:
        raise ValueError("Attention tensor must have four strides")
    return {
        "b": tensor.strides[0],
        "h": tensor.strides[1],
        "m": tensor.strides[2],
        "d": tensor.strides[3],
    }


def _attention_forward_scalars(
    request: ParsedAttentionRequest,
    *,
    fp8: bool,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    q = request.tensor("q")
    k = request.tensor("k")
    v = request.tensor("v")
    o = request.tensor("o")
    stats = request.tensor("stats")
    bias_strides = _attention_bias_strides(request)
    values = {
        "HQ": request.heads,
        "SQ": request.sequence_q,
        "SKV": request.sequence_kv,
        "q_per_k": request.q_per_k,
        "q_per_v": request.q_per_v,
        "min_diag": request.min_diag,
        "max_diag": request.max_diag,
        "stride_qb": q.strides[0],
        "stride_qh": q.strides[1],
        "stride_qm": q.strides[2],
        "stride_qd": q.strides[3],
        "stride_kb": k.strides[0],
        "stride_kh": k.strides[1],
        "stride_kn": k.strides[2],
        "stride_kd": k.strides[3],
        "stride_vb": v.strides[0],
        "stride_vh": v.strides[1],
        "stride_vn": v.strides[2],
        "stride_vd": v.strides[3],
        "stride_bias_b": bias_strides[0],
        "stride_bias_h": bias_strides[1],
        "stride_bias_m": bias_strides[2],
        "stride_bias_n": bias_strides[3],
        "stride_ob": o.strides[0],
        "stride_oh": o.strides[1],
        "stride_om": o.strides[2],
        "stride_od": o.strides[3],
        "stride_sb": stats.strides[0],
        "stride_sh": stats.strides[1],
        "stride_sm": stats.strides[2],
    }
    scale_name = "attn_scale" if fp8 else "qk_scale"
    scale = request.attn_scale if fp8 else request.attn_scale * _LOG2_E
    return (
        _attention_scalar_argument(scale_name, "fp32", scale),
        *(
            _attention_scalar_argument(name, "i32", values[name])
            for name in _ATTENTION_FORWARD_I32
        ),
    )


def _attention_forward_constants(
    request: ParsedAttentionRequest,
    *,
    fp8: bool,
    block_size: int,
) -> dict[str, int | float | bool]:
    result: dict[str, int | float | bool] = {
        "HEAD_DIM": request.head_dimension,
        "V_DIM": request.value_dimension,
        "BLOCK_M": block_size,
        "BLOCK_N": block_size,
        "BLOCK_D": _attention_full_dimension_block(
            request.head_dimension
        ),
        "BLOCK_DV": _attention_full_dimension_block(
            request.value_dimension
        ),
        "HAS_BIAS": request.has_bias,
        "BANDED": request.banded,
        "GENERATE_STATS": request.generate_stats,
        "REVERSE_CAUSAL": request.reverse_causal,
    }
    if not fp8:
        result["ELEM_SIZE"] = _ATTENTION_ELEMENT_SIZES[
            request.tensor("q").data_type
        ]
    return result


def _attention_bias_pointer(
    request: ParsedAttentionRequest,
) -> tuple[str, str, RuntimeArgument]:
    if request.has_bias:
        return _attention_tensor_pointer(request, "bias_ptr", "bias")
    return _attention_tensor_pointer(
        request,
        "bias_ptr",
        "q",
        semantic_name="bias_placeholder_q",
    )


def _attention_stats_pointer(
    request: ParsedAttentionRequest,
) -> tuple[str, str, RuntimeArgument]:
    if request.tensor("stats").virtual:
        return _attention_workspace_pointer(
            "stats_ptr", "stats_placeholder"
        )
    return _attention_tensor_pointer(request, "stats_ptr", "stats")


def _attention_forward_plan(
    request: ParsedAttentionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    pointers = (
        _attention_tensor_pointer(request, "q_ptr", "q"),
        _attention_tensor_pointer(request, "k_ptr", "k"),
        _attention_tensor_pointer(request, "v_ptr", "v"),
        _attention_bias_pointer(request),
        _attention_tensor_pointer(request, "o_ptr", "o"),
        _attention_stats_pointer(request),
    )
    scalars = _attention_forward_scalars(request, fp8=False)
    stage = _attention_tuned_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_sdpa_fwd_kernel",
        dependencies=(),
        pointers=pointers,
        scalars=scalars,
        constants_factory=lambda block_size: (
            _attention_forward_constants(
                request, fp8=False, block_size=block_size
            )
        ),
        grid_factory=lambda block_size: (
            (request.sequence_q + block_size - 1) // block_size,
            request.batch * request.heads,
            1,
        ),
    )
    return ExecutionPlan(
        stages=(stage,),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_WORKSPACE_SIZE,
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _attention_fp8_forward_plan(
    request: ParsedAttentionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    zero = _attention_fixed_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_zero_sdpa_fp8_fwd_amax_kernel",
        dependencies=(),
        grid=(1, 1, 1),
        pointers=(
            _attention_tensor_pointer(
                request, "amax_s_ptr", "amax_s"
            ),
            _attention_tensor_pointer(
                request, "amax_o_ptr", "amax_o"
            ),
        ),
    )
    pointers = (
        _attention_tensor_pointer(request, "q_ptr", "q"),
        _attention_tensor_pointer(request, "k_ptr", "k"),
        _attention_tensor_pointer(request, "v_ptr", "v"),
        _attention_bias_pointer(request),
        _attention_tensor_pointer(request, "o_ptr", "o"),
        _attention_stats_pointer(request),
        _attention_tensor_pointer(request, "amax_s_ptr", "amax_s"),
        _attention_tensor_pointer(request, "amax_o_ptr", "amax_o"),
        _attention_tensor_pointer(
            request, "descale_q_ptr", "descale_q"
        ),
        _attention_tensor_pointer(
            request, "descale_k_ptr", "descale_k"
        ),
        _attention_tensor_pointer(
            request, "descale_v_ptr", "descale_v"
        ),
        _attention_tensor_pointer(
            request, "descale_s_ptr", "descale_s"
        ),
        _attention_tensor_pointer(request, "scale_s_ptr", "scale_s"),
        _attention_tensor_pointer(request, "scale_o_ptr", "scale_o"),
    )
    forward = _attention_tuned_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_sdpa_fp8_fwd_kernel",
        dependencies=(0,),
        pointers=pointers,
        scalars=_attention_forward_scalars(request, fp8=True),
        constants_factory=lambda block_size: (
            _attention_forward_constants(
                request, fp8=True, block_size=block_size
            )
        ),
        grid_factory=lambda block_size: (
            (request.sequence_q + block_size - 1) // block_size,
            request.batch * request.heads,
            1,
        ),
    )
    return ExecutionPlan(
        stages=(zero, forward),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_WORKSPACE_SIZE,
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _attention_backward_scalars(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    return (
        _attention_scalar_argument(
            "attn_scale", "fp32", request.attn_scale
        ),
        _attention_scalar_argument("SQ", "i32", request.sequence_q),
        _attention_scalar_argument(
            "SKV", "i32", request.sequence_kv
        ),
        _attention_scalar_argument(
            "min_diag", "i32", request.min_diag
        ),
        _attention_scalar_argument(
            "max_diag", "i32", request.max_diag
        ),
    )


def _attention_backward_base_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    q = request.tensor("q")
    k = request.tensor("k")
    v = request.tensor("v")
    do = request.tensor("do")
    stats = request.tensor("stats")
    bias_strides = _attention_bias_strides(request)
    return {
        "stride_qb": q.strides[0],
        "stride_qh": q.strides[1],
        "stride_qm": q.strides[2],
        "stride_qd": q.strides[3],
        "stride_kb": k.strides[0],
        "stride_kh": k.strides[1],
        "stride_kn": k.strides[2],
        "stride_kd": k.strides[3],
        "stride_vb": v.strides[0],
        "stride_vh": v.strides[1],
        "stride_vn": v.strides[2],
        "stride_vd": v.strides[3],
        "stride_bias_b": bias_strides[0],
        "stride_bias_h": bias_strides[1],
        "stride_bias_m": bias_strides[2],
        "stride_bias_n": bias_strides[3],
        "stride_dob": do.strides[0],
        "stride_doh": do.strides[1],
        "stride_dom": do.strides[2],
        "stride_dod": do.strides[3],
        "stride_sb": stats.strides[0],
        "stride_sh": stats.strides[1],
        "stride_sm": stats.strides[2],
        "HEAD_DIM": request.head_dimension,
        "BLOCK_M": block_size,
        "BLOCK_N": block_size,
        "BLOCK_D_FULL": _attention_full_dimension_block(
            request.head_dimension
        ),
        "FULL_ATTENTION": False,
        "HAS_BIAS": request.has_bias,
        "BANDED": request.banded,
        "CAUSAL_TOP_LEFT": request.causal_top_left,
    }


def _attention_delta_strides(
    request: ParsedAttentionRequest,
) -> tuple[int, int, int]:
    return (
        request.heads * request.sequence_q,
        request.sequence_q,
        1,
    )


def _attention_dbias_reduces(
    request: ParsedAttentionRequest,
) -> bool:
    dbias = request.optional_tensor("dbias")
    return dbias is not None and (
        dbias.dimensions[0] != request.batch
        or dbias.dimensions[1] != request.heads
    )


def _attention_dq_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_backward_base_constants(
        request, block_size
    )
    o = request.tensor("o")
    dq = request.tensor("dq")
    dbias = request.optional_tensor("dbias")
    delta = _attention_delta_strides(request)
    dbias_strides = dbias.strides if dbias is not None else (0, 0, 0, 0)
    constants.update(
        {
            "HQ": request.heads,
            "q_per_k": request.q_per_k,
            "q_per_v": request.q_per_v,
            "stride_ob": o.strides[0],
            "stride_oh": o.strides[1],
            "stride_om": o.strides[2],
            "stride_od": o.strides[3],
            "stride_delta_b": delta[0],
            "stride_delta_h": delta[1],
            "stride_delta_m": delta[2],
            "stride_dqb": dq.strides[0],
            "stride_dqh": dq.strides[1],
            "stride_dqm": dq.strides[2],
            "stride_dqd": dq.strides[3],
            "stride_dbias_b": dbias_strides[0],
            "stride_dbias_h": dbias_strides[1],
            "stride_dbias_m": dbias_strides[2],
            "stride_dbias_n": dbias_strides[3],
            "V_DIM": request.value_dimension,
            "DBIAS_BATCHES": (
                dbias.dimensions[0] if dbias is not None else 1
            ),
            "DBIAS_HEADS": (
                dbias.dimensions[1] if dbias is not None else 1
            ),
            "BLOCK_D_OUT": _attention_output_dimension_block(
                request.head_dimension, block_size
            ),
            "BLOCK_DV": _attention_full_dimension_block(
                request.value_dimension
            ),
            "HAS_DBIAS": request.has_dbias,
            "DBIAS_REDUCE": _attention_dbias_reduces(request),
        }
    )
    return constants


def _attention_dq_pointers(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    if request.has_dbias:
        dbias_pointer = _attention_tensor_pointer(
            request, "dbias_ptr", "dbias"
        )
    else:
        dbias_pointer = _attention_tensor_pointer(
            request,
            "dbias_ptr",
            "dq",
            semantic_name="dbias_placeholder_dq",
        )
    return (
        _attention_tensor_pointer(request, "q_ptr", "q"),
        _attention_tensor_pointer(request, "k_ptr", "k"),
        _attention_tensor_pointer(request, "v_ptr", "v"),
        _attention_bias_pointer(request),
        _attention_tensor_pointer(request, "o_ptr", "o"),
        _attention_tensor_pointer(request, "do_ptr", "do"),
        _attention_tensor_pointer(request, "stats_ptr", "stats"),
        _attention_workspace_pointer("delta_ptr", "delta"),
        _attention_tensor_pointer(request, "dq_ptr", "dq"),
        dbias_pointer,
    )


def _attention_dkdv_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_backward_base_constants(
        request, block_size
    )
    dk = request.tensor("dk")
    dv = request.tensor("dv")
    delta = _attention_delta_strides(request)
    constants.update(
        {
            "HKV": request.key_heads,
            "stride_delta_b": delta[0],
            "stride_delta_h": delta[1],
            "stride_delta_m": delta[2],
            "stride_dkb": dk.strides[0],
            "stride_dkh": dk.strides[1],
            "stride_dkn": dk.strides[2],
            "stride_dkd": dk.strides[3],
            "stride_dvb": dv.strides[0],
            "stride_dvh": dv.strides[1],
            "stride_dvn": dv.strides[2],
            "stride_dvd": dv.strides[3],
            "Q_PER": request.q_per_k,
            "BLOCK_D_OUT": _attention_output_dimension_block(
                request.head_dimension, block_size
            ),
        }
    )
    return constants


def _attention_dkdv_pointers(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    return (
        _attention_tensor_pointer(request, "q_ptr", "q"),
        _attention_tensor_pointer(request, "k_ptr", "k"),
        _attention_tensor_pointer(request, "v_ptr", "v"),
        _attention_bias_pointer(request),
        _attention_tensor_pointer(request, "do_ptr", "do"),
        _attention_tensor_pointer(request, "stats_ptr", "stats"),
        _attention_workspace_pointer("delta_ptr", "delta"),
        _attention_tensor_pointer(request, "dk_ptr", "dk"),
        _attention_tensor_pointer(request, "dv_ptr", "dv"),
    )


def _attention_dk_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_backward_base_constants(
        request, block_size
    )
    dk = request.tensor("dk")
    delta = _attention_delta_strides(request)
    constants.update(
        {
            "HKV": request.key_heads,
            "stride_delta_b": delta[0],
            "stride_delta_h": delta[1],
            "stride_delta_m": delta[2],
            "stride_dkb": dk.strides[0],
            "stride_dkh": dk.strides[1],
            "stride_dkn": dk.strides[2],
            "stride_dkd": dk.strides[3],
            "V_DIM": request.value_dimension,
            "Q_PER": request.q_per_k,
            "BLOCK_D_OUT": _attention_output_dimension_block(
                request.head_dimension, block_size
            ),
            "BLOCK_DV": _attention_full_dimension_block(
                request.value_dimension
            ),
        }
    )
    return constants


def _attention_dk_pointers(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    return (
        _attention_tensor_pointer(request, "q_ptr", "q"),
        _attention_tensor_pointer(request, "k_ptr", "k"),
        _attention_tensor_pointer(request, "v_ptr", "v"),
        _attention_bias_pointer(request),
        _attention_tensor_pointer(request, "do_ptr", "do"),
        _attention_tensor_pointer(request, "stats_ptr", "stats"),
        _attention_workspace_pointer("delta_ptr", "delta"),
        _attention_tensor_pointer(request, "dk_ptr", "dk"),
    )


def _attention_dv_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_backward_base_constants(
        request, block_size
    )
    for name in (
        "stride_vb",
        "stride_vh",
        "stride_vn",
        "stride_vd",
    ):
        constants.pop(name)
    dv = request.tensor("dv")
    constants.update(
        {
            "HKV": request.value_heads,
            "stride_dvb": dv.strides[0],
            "stride_dvh": dv.strides[1],
            "stride_dvn": dv.strides[2],
            "stride_dvd": dv.strides[3],
            "V_DIM": request.value_dimension,
            "Q_PER": request.q_per_v,
            "BLOCK_DV_OUT": _attention_output_dimension_block(
                request.value_dimension, block_size
            ),
        }
    )
    return constants


def _attention_dv_pointers(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    return (
        _attention_tensor_pointer(request, "q_ptr", "q"),
        _attention_tensor_pointer(request, "k_ptr", "k"),
        _attention_bias_pointer(request),
        _attention_tensor_pointer(request, "do_ptr", "do"),
        _attention_tensor_pointer(request, "stats_ptr", "stats"),
        _attention_tensor_pointer(request, "dv_ptr", "dv"),
    )


def _attention_backward_plan(
    request: ParsedAttentionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    stages: list[KernelStage] = []
    dq_dependencies: tuple[int, ...] = ()
    dbias = request.optional_tensor("dbias")
    if _attention_dbias_reduces(request):
        if dbias is None or not is_row_major_contiguous(dbias):
            raise ValueError(
                "broadcast Attention dbias must be row-major contiguous"
            )
        dbias_elements = math.prod(dbias.dimensions)
        if dbias_elements > (1 << 31) - 1:
            raise ValueError("Attention dbias element count exceeds int32")
        zero = _attention_fixed_stage(
            request,
            source_sha256,
            stage_id=0,
            function="_zero_contiguous_kernel",
            dependencies=(),
            grid=((dbias_elements + 255) // 256, 1, 1),
            pointers=(
                _attention_tensor_pointer(request, "ptr", "dbias"),
            ),
            scalars=(
                _attention_scalar_argument(
                    "n_elements", "i32", dbias_elements
                ),
            ),
            constants={"BLOCK": 256},
        )
        stages.append(zero)
        dq_dependencies = (0,)
    dq_id = len(stages)
    dq = _attention_tuned_stage(
        request,
        source_sha256,
        stage_id=dq_id,
        function="_sdpa_bwd_dq_dbias_kernel",
        dependencies=dq_dependencies,
        pointers=_attention_dq_pointers(request),
        scalars=_attention_backward_scalars(request),
        constants_factory=lambda block_size: _attention_dq_constants(
            request, block_size
        ),
        grid_factory=lambda block_size: (
            (request.sequence_q + block_size - 1) // block_size,
            (
                request.head_dimension
                + _attention_output_dimension_block(
                    request.head_dimension, block_size
                )
                - 1
            )
            // _attention_output_dimension_block(
                request.head_dimension, block_size
            ),
            request.batch * request.heads,
        ),
    )
    stages.append(dq)
    gradient_id = len(stages)
    if (
        request.key_heads == request.value_heads
        and request.head_dimension == request.value_dimension
    ):
        stages.append(
            _attention_tuned_stage(
                request,
                source_sha256,
                stage_id=gradient_id,
                function="_sdpa_bwd_dkdv_kernel",
                dependencies=(dq_id,),
                pointers=_attention_dkdv_pointers(request),
                scalars=_attention_backward_scalars(request),
                constants_factory=lambda block_size: (
                    _attention_dkdv_constants(request, block_size)
                ),
                grid_factory=lambda block_size: (
                    (request.sequence_kv + block_size - 1)
                    // block_size,
                    (
                        request.head_dimension
                        + _attention_output_dimension_block(
                            request.head_dimension, block_size
                        )
                        - 1
                    )
                    // _attention_output_dimension_block(
                        request.head_dimension, block_size
                    ),
                    request.batch * request.key_heads,
                ),
            )
        )
    else:
        stages.append(
            _attention_tuned_stage(
                request,
                source_sha256,
                stage_id=gradient_id,
                function="_sdpa_bwd_dk_kernel",
                dependencies=(dq_id,),
                pointers=_attention_dk_pointers(request),
                scalars=_attention_backward_scalars(request),
                constants_factory=lambda block_size: (
                    _attention_dk_constants(request, block_size)
                ),
                grid_factory=lambda block_size: (
                    (request.sequence_kv + block_size - 1)
                    // block_size,
                    (
                        request.head_dimension
                        + _attention_output_dimension_block(
                            request.head_dimension, block_size
                        )
                        - 1
                    )
                    // _attention_output_dimension_block(
                        request.head_dimension, block_size
                    ),
                    request.batch * request.key_heads,
                ),
            )
        )
        stages.append(
            _attention_tuned_stage(
                request,
                source_sha256,
                stage_id=gradient_id + 1,
                function="_sdpa_bwd_dv_kernel",
                dependencies=(gradient_id,),
                pointers=_attention_dv_pointers(request),
                scalars=_attention_backward_scalars(request),
                constants_factory=lambda block_size: (
                    _attention_dv_constants(request, block_size)
                ),
                grid_factory=lambda block_size: (
                    (request.sequence_kv + block_size - 1)
                    // block_size,
                    (
                        request.value_dimension
                        + _attention_output_dimension_block(
                            request.value_dimension, block_size
                        )
                        - 1
                    )
                    // _attention_output_dimension_block(
                        request.value_dimension, block_size
                    ),
                    request.batch * request.value_heads,
                ),
            )
        )
    raw_delta_size = (
        4 * request.batch * request.heads * request.sequence_q
    )
    aligned_delta_size = (
        (raw_delta_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return ExecutionPlan(
        stages=tuple(stages),
        external_binding_uids=request.external_binding_uids,
        workspace_size=max(_WORKSPACE_SIZE, aligned_delta_size),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _attention_fp8_backward_base_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    q = request.tensor("q")
    k = request.tensor("k")
    v = request.tensor("v")
    o = request.tensor("o")
    do = request.tensor("do")
    stats = request.tensor("stats")
    return {
        "stride_qb": q.strides[0],
        "stride_qh": q.strides[1],
        "stride_qm": q.strides[2],
        "stride_qd": q.strides[3],
        "stride_kb": k.strides[0],
        "stride_kh": k.strides[1],
        "stride_kn": k.strides[2],
        "stride_kd": k.strides[3],
        "stride_vb": v.strides[0],
        "stride_vh": v.strides[1],
        "stride_vn": v.strides[2],
        "stride_vd": v.strides[3],
        "stride_ob": o.strides[0],
        "stride_oh": o.strides[1],
        "stride_om": o.strides[2],
        "stride_od": o.strides[3],
        "stride_dob": do.strides[0],
        "stride_doh": do.strides[1],
        "stride_dom": do.strides[2],
        "stride_dod": do.strides[3],
        "stride_sb": stats.strides[0],
        "stride_sh": stats.strides[1],
        "stride_sm": stats.strides[2],
        "HEAD_DIM": request.head_dimension,
        "BLOCK_M": block_size,
        "BLOCK_N": block_size,
        "BLOCK_D": _attention_full_dimension_block(
            request.head_dimension
        ),
        "BANDED": request.banded,
        "FULL_BLOCKS": False,
        "CAUSAL_TOP_LEFT": request.causal_top_left,
    }


def _attention_fp8_dq_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_fp8_backward_base_constants(
        request, block_size
    )
    dq = request.tensor("dq")
    constants.update(
        {
            "q_per_k": request.q_per_k,
            "q_per_v": request.q_per_v,
            "stride_dqb": dq.strides[0],
            "stride_dqh": dq.strides[1],
            "stride_dqm": dq.strides[2],
            "stride_dqd": dq.strides[3],
        }
    )
    return constants


def _attention_fp8_dq_scalars(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    return (
        _attention_scalar_argument(
            "attn_scale", "fp32", request.attn_scale
        ),
        _attention_scalar_argument("HQ", "i32", request.heads),
        _attention_scalar_argument("SQ", "i32", request.sequence_q),
        _attention_scalar_argument(
            "SKV", "i32", request.sequence_kv
        ),
        _attention_scalar_argument(
            "min_diag", "i32", request.min_diag
        ),
        _attention_scalar_argument(
            "max_diag", "i32", request.max_diag
        ),
    )


def _attention_fp8_dq_pointers(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    ports = (
        ("q_ptr", "q"),
        ("k_ptr", "k"),
        ("v_ptr", "v"),
        ("o_ptr", "o"),
        ("do_ptr", "do"),
        ("stats_ptr", "stats"),
        ("dq_ptr", "dq"),
        ("amax_dq_ptr", "amax_dq"),
        ("descale_q_ptr", "descale_q"),
        ("descale_k_ptr", "descale_k"),
        ("descale_v_ptr", "descale_v"),
        ("descale_o_ptr", "descale_o"),
        ("descale_do_ptr", "descale_do"),
        ("descale_dp_ptr", "descale_dp"),
        ("scale_dq_ptr", "scale_dq"),
        ("scale_dp_ptr", "scale_dp"),
    )
    return tuple(
        _attention_tensor_pointer(request, parameter, port)
        for parameter, port in ports
    )


def _attention_fp8_dkdv_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_fp8_backward_base_constants(
        request, block_size
    )
    dk = request.tensor("dk")
    dv = request.tensor("dv")
    constants.update(
        {
            "HKV": request.key_heads,
            "stride_dkb": dk.strides[0],
            "stride_dkh": dk.strides[1],
            "stride_dkn": dk.strides[2],
            "stride_dkd": dk.strides[3],
            "stride_dvb": dv.strides[0],
            "stride_dvh": dv.strides[1],
            "stride_dvn": dv.strides[2],
            "stride_dvd": dv.strides[3],
            "Q_PER": request.q_per_k,
        }
    )
    return constants


def _attention_fp8_dkdv_pointers(
    request: ParsedAttentionRequest,
) -> tuple[tuple[str, str, RuntimeArgument], ...]:
    ports = (
        ("q_ptr", "q"),
        ("k_ptr", "k"),
        ("v_ptr", "v"),
        ("o_ptr", "o"),
        ("do_ptr", "do"),
        ("stats_ptr", "stats"),
        ("dk_ptr", "dk"),
        ("dv_ptr", "dv"),
        ("amax_dk_ptr", "amax_dk"),
        ("amax_dv_ptr", "amax_dv"),
        ("amax_dp_ptr", "amax_dp"),
        ("descale_q_ptr", "descale_q"),
        ("descale_k_ptr", "descale_k"),
        ("descale_v_ptr", "descale_v"),
        ("descale_o_ptr", "descale_o"),
        ("descale_do_ptr", "descale_do"),
        ("descale_s_ptr", "descale_s"),
        ("descale_dp_ptr", "descale_dp"),
        ("scale_s_ptr", "scale_s"),
        ("scale_dk_ptr", "scale_dk"),
        ("scale_dv_ptr", "scale_dv"),
        ("scale_dp_ptr", "scale_dp"),
    )
    return tuple(
        _attention_tensor_pointer(request, parameter, port)
        for parameter, port in ports
    )


def _attention_fp8_backward_plan(
    request: ParsedAttentionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    zero_ports = (
        ("amax_dq_ptr", "amax_dq"),
        ("amax_dk_ptr", "amax_dk"),
        ("amax_dv_ptr", "amax_dv"),
        ("amax_dp_ptr", "amax_dp"),
    )
    zero = _attention_fixed_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_zero_sdpa_fp8_bwd_amax_kernel",
        dependencies=(),
        grid=(1, 1, 1),
        pointers=tuple(
            _attention_tensor_pointer(request, parameter, port)
            for parameter, port in zero_ports
        ),
    )
    dq = _attention_tuned_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_sdpa_fp8_bwd_dq_kernel",
        dependencies=(0,),
        pointers=_attention_fp8_dq_pointers(request),
        scalars=_attention_fp8_dq_scalars(request),
        constants_factory=lambda block_size: (
            _attention_fp8_dq_constants(request, block_size)
        ),
        grid_factory=lambda block_size: (
            (request.sequence_q + block_size - 1) // block_size,
            request.batch * request.heads,
            1,
        ),
    )
    dkdv = _attention_tuned_stage(
        request,
        source_sha256,
        stage_id=2,
        function="_sdpa_fp8_bwd_dkdv_kernel",
        dependencies=(1,),
        pointers=_attention_fp8_dkdv_pointers(request),
        scalars=_attention_backward_scalars(request),
        constants_factory=lambda block_size: (
            _attention_fp8_dkdv_constants(request, block_size)
        ),
        grid_factory=lambda block_size: (
            (request.sequence_kv + block_size - 1) // block_size,
            request.batch * request.key_heads,
            1,
        ),
    )
    return ExecutionPlan(
        stages=(zero, dq, dkdv),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_WORKSPACE_SIZE,
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _plan(
    request: ParsedCompilerRequest,
    source_sha256: str,
) -> ExecutionPlan:
    if isinstance(request, ParsedBinaryRequest):
        return _binary_plan(request, source_sha256)
    if isinstance(request, ParsedAddSquareRequest):
        return _add_square_plan(request, source_sha256)
    if isinstance(request, ParsedConvBiasReluRequest):
        return _conv_bias_relu_plan(request, source_sha256)
    if isinstance(request, ParsedUnaryRequest):
        if request.operation == "identity":
            return _identity_plan(request, source_sha256)
        return _unary_plan(request, source_sha256)
    if isinstance(request, ParsedTernaryRequest):
        return _ternary_plan(request, source_sha256)
    if isinstance(request, ParsedLayoutRequest):
        return _layout_plan(request, source_sha256)
    if isinstance(request, ParsedReductionRequest):
        return _reduction_plan(request, source_sha256)
    if isinstance(request, ParsedMatmulRequest):
        return _matmul_plan(request, source_sha256)
    if isinstance(request, ParsedConvolutionRequest):
        return _convolution_plan(request, source_sha256)
    if isinstance(request, ParsedNormalizationRequest):
        return _normalization_plan(request, source_sha256)
    if isinstance(request, ParsedBatchnormRequest):
        return _batchnorm_plan(request, source_sha256)
    if isinstance(request, ParsedBatchnormInferenceRequest):
        return _batchnorm_inference_plan(request, source_sha256)
    if isinstance(request, ParsedAttentionRequest):
        if request.operation == "sdpa":
            return _attention_forward_plan(request, source_sha256)
        if request.operation == "sdpa_backward":
            return _attention_backward_plan(request, source_sha256)
        if request.operation == "sdpa_fp8":
            return _attention_fp8_forward_plan(request, source_sha256)
        if request.operation == "sdpa_fp8_backward":
            return _attention_fp8_backward_plan(request, source_sha256)
        raise ValueError("unknown MThreads Attention operation")
    raise TypeError("unknown mthreads compiler request family")


def _write_and_sync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    if path.read_bytes() != payload:
        raise RuntimeError(f"artifact self-read differs: {path.name}")


def _safe_destination(
    output_directory: Path,
    *,
    request_path: Path,
    request_bytes: bytes,
) -> tuple[Path, bool]:
    expanded = output_directory.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    if ".." in expanded.parts:
        raise ValueError("artifact output path contains parent traversal")
    if expanded.is_symlink():
        raise ValueError("artifact output directory must not be a symlink")
    destination = expanded.resolve(strict=False)
    has_core_request = False
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(
                "artifact output directory already exists and is nonempty"
            )
        mode = destination.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("artifact output directory must not be a symlink")
        entries = list(destination.iterdir())
        if entries:
            expected_request = destination / "request.json"
            if (
                len(entries) != 1
                or entries[0] != expected_request
                or expected_request.is_symlink()
                or not expected_request.is_file()
                or request_path.resolve(strict=True)
                != expected_request.resolve(strict=True)
                or expected_request.read_bytes() != request_bytes
            ):
                raise ValueError(
                    "artifact output directory already exists and is nonempty"
                )
            has_core_request = True
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("artifact output parent is not a directory")
    return parent / destination.name, has_core_request


def _publish_artifact(
    output_directory: Path,
    *,
    request_path: Path,
    request_bytes: bytes,
    source_relative_path: str,
    source_bytes: bytes,
    manifest: dict[str, Any],
) -> Path:
    destination, has_core_request = _safe_destination(
        output_directory,
        request_path=request_path,
        request_bytes=request_bytes,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.",
            dir=destination.parent,
        )
    )
    published = False
    try:
        source_path = temporary / source_relative_path
        _write_and_sync(source_path, source_bytes)
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            source_digest != manifest["source_sha256"]
            or source_digest != manifest["files"][0]["sha256"]
            or source_path.stat().st_size != manifest["files"][0]["size"]
        ):
            raise RuntimeError("artifact source verification failed")
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_and_sync(temporary / "manifest.json", manifest_bytes)
        if has_core_request:
            _write_and_sync(temporary / "request.json", request_bytes)
        if json.loads(
            (temporary / "manifest.json").read_text(encoding="utf-8")
        ) != manifest:
            raise RuntimeError("artifact manifest self-read differs")

        for directory in (source_path.parent, temporary):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if has_core_request:
            backup = temporary.with_name(temporary.name + ".request-input")
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            if destination.exists():
                destination.rmdir()
            os.replace(temporary, destination)
        parent_descriptor = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        published = True
        return destination
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = SUPPORTED_ENGINE,
) -> dict[str, Any]:
    if execution_engine != SUPPORTED_ENGINE:
        raise ValueError("mthreads requires execution engine libtriton_jit")
    request_bytes = request_path.read_bytes()
    preliminary = json.loads(request_bytes)
    if not isinstance(preliminary, dict):
        raise ValueError("compiler request must be an object")
    target = preliminary.get("target")
    if not isinstance(target, str):
        raise ValueError("compiler request target is invalid")
    identity = compiler_identity(target, execution_engine)
    request = parse_compiler_request(
        request_bytes,
        expected_target=target,
        expected_identity=identity["identity_sha256"],
    )

    candidate = select_kernel_candidate("mthreads", request.operation)
    platform_kernel = isinstance(
        request,
        (
            ParsedBinaryRequest,
            ParsedAddSquareRequest,
            ParsedConvBiasReluRequest,
            ParsedBatchnormInferenceRequest,
        ),
    ) or isinstance(
        request,
        (ParsedConvolutionRequest, ParsedUnaryRequest, ParsedMatmulRequest),
    ) or (
        isinstance(request, ParsedLayoutRequest)
        and request.operation in {"reshape", "slice", "transpose"}
    )
    if candidate.ownership not in {"common", "platform"} or (
        candidate.ownership == "platform" and not platform_kernel
    ):
        raise ValueError(
            "mthreads compiler selected an unsupported kernel owner"
        )
    if isinstance(request, ParsedBinaryRequest):
        expected_source = "binary.py"
        source_relative_path = _BINARY_SOURCE_RELATIVE_PATH
        required_functions = {
            "binary_contiguous_kernel",
            "binary_strided_kernel",
        }
    elif isinstance(request, ParsedAddSquareRequest):
        expected_source = "composite.py"
        source_relative_path = _COMPOSITE_SOURCE_RELATIVE_PATH
        required_functions = {"add_square_tensor_kernel"}
    elif isinstance(request, ParsedConvBiasReluRequest):
        expected_source = "conv_bias_relu.py"
        source_relative_path = _CONV_BIAS_RELU_SOURCE_RELATIVE_PATH
        required_functions = {"conv_bias_relu_2d_kernel"}
    elif isinstance(request, ParsedUnaryRequest):
        if request.operation == "identity":
            expected_source = "identity.py"
            source_relative_path = _IDENTITY_SOURCE_RELATIVE_PATH
            required_functions = {
                "identity_contiguous_packed_kernel",
                "identity_contiguous_kernel",
                "identity_strided_kernel",
            }
        else:
            expected_source = "unary.py"
            source_relative_path = _UNARY_SOURCE_RELATIVE_PATH
            required_functions = {
                "unary_pointwise_contiguous_kernel",
                "unary_pointwise_strided_kernel",
            }
    elif isinstance(request, ParsedTernaryRequest):
        expected_source = "ternary.py"
        source_relative_path = _TERNARY_SOURCE_RELATIVE_PATH
        required_functions = {
            "binary_select_tensor_kernel",
            "binary_select_strided_kernel",
        }
    elif isinstance(request, ParsedLayoutRequest):
        expected_source = "layout.py"
        source_relative_path = _LAYOUT_SOURCE_RELATIVE_PATH
        required_functions = (
            {"reshape_contiguous_kernel", "layout_copy_kernel"}
            if request.operation == "reshape"
            else {"slice_copy_kernel", "layout_copy_kernel"}
            if request.operation == "slice"
            else {"transpose_physical_copy_kernel", "layout_copy_kernel"}
            if request.operation == "transpose"
            else {"layout_copy_kernel"}
        )
    elif isinstance(request, ParsedReductionRequest):
        expected_source = "reduction.py"
        source_relative_path = _REDUCTION_SOURCE_RELATIVE_PATH
        required_functions = {
            "reduction_2d_kernel",
            "reduction_3d_kernel",
            "reduction_strided_kernel",
        }
    elif isinstance(request, ParsedMatmulRequest):
        expected_source = "matmul.py"
        source_relative_path = _MATMUL_SOURCE_RELATIVE_PATH
        required_functions = {
            "matmul_descriptor_kernel",
            "_matmul_tle_consumer",
            "_matmul_tle_producer",
            "matmul_tle_kernel",
            "matmul_strided_kernel",
        }
    elif isinstance(request, ParsedConvolutionRequest):
        expected_source = "convolution.py"
        source_relative_path = _CONVOLUTION_SOURCE_RELATIVE_PATH
        if request.operation == "convolution_dgrad":
            required_functions = {"conv_dgrad_nd_kernel"}
        elif request.operation == "convolution_wgrad":
            required_functions = {
                "conv_wgrad_nd_kernel",
                "_conv_wgrad2d_p5_pack_image_kernel",
                "_conv_wgrad2d_p5_mm_kernel",
            }
        else:
            required_functions = {
                "conv1d_gemm_kernel",
                "conv2d_spatial_nchw_kernel",
                "conv3d_spatial_ncdhw_m_kernel",
                "_conv_fprop2d_im2col_kernel",
                "_conv_fprop2d_im2col_mm_kernel",
                "conv_dgrad_nd_kernel",
                "conv_wgrad_nd_kernel",
            }
    elif isinstance(request, ParsedNormalizationRequest):
        expected_source = "normalization.py"
        source_relative_path = _NORMALIZATION_SOURCE_RELATIVE_PATH
        required_functions = {
            "layer_norm_kernel"
            if request.operation == "layernorm"
            else "rms_norm_kernel"
        }
    elif isinstance(request, ParsedBatchnormRequest):
        expected_source = "normalization.py"
        source_relative_path = _NORMALIZATION_SOURCE_RELATIVE_PATH
        required_functions = {
            "batch_norm_nchw_kernel",
            "batch_norm_kernel",
        }
    elif isinstance(request, ParsedBatchnormInferenceRequest):
        expected_source = "normalization.py"
        source_relative_path = _NORMALIZATION_SOURCE_RELATIVE_PATH
        required_functions = {
            "batch_norm_inference_nchw_kernel",
            "batch_norm_inference_kernel",
        }
    elif isinstance(request, ParsedAttentionRequest):
        expected_source = "attention.py"
        source_relative_path = _ATTENTION_SOURCE_RELATIVE_PATH
        required_functions = {
            "sdpa": {"_sdpa_fwd_kernel"},
            "sdpa_backward": {
                "_zero_contiguous_kernel",
                "_sdpa_bwd_dq_dbias_kernel",
                "_sdpa_bwd_dkdv_kernel",
                "_sdpa_bwd_dk_kernel",
                "_sdpa_bwd_dv_kernel",
            },
            "sdpa_fp8": {
                "_zero_sdpa_fp8_fwd_amax_kernel",
                "_sdpa_fp8_fwd_kernel",
            },
            "sdpa_fp8_backward": {
                "_zero_sdpa_fp8_bwd_amax_kernel",
                "_sdpa_fp8_bwd_dq_kernel",
                "_sdpa_fp8_bwd_dkdv_kernel",
            },
        }[request.operation]
    else:
        raise TypeError("unknown mthreads compiler request family")
    if required_functions.difference(candidate.functions):
        raise ValueError(
            "kernel registry entry is incomplete"
        )
    source_path = resolve_kernel_source(
        _compiler_entry_path(), candidate
    )
    if (
        Path(candidate.source).is_absolute()
        or ".." in Path(candidate.source).parts
        or candidate.source != expected_source
    ):
        raise ValueError("common kernel source path is unsafe")
    source_bytes = _materialize_mthreads_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > (16 << 20):
        raise ValueError("common kernel source bytes are invalid")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    plan = _plan(request, source_sha256)
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request.flagdnn_version,
        "backend": "mthreads",
        "target": request.target,
        "engine": execution_engine,
        "request_sha256": request.request_sha256,
        "compiler_identity": request.compiler_identity,
        "source_sha256": source_sha256,
        "workspace_size": plan.workspace_size,
        "workspace_alignment": plan.workspace_alignment,
        "external_binding_uids": list(plan.external_binding_uids),
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(plan.stages),
            "stages": [stage.to_json() for stage in plan.stages],
        },
        "files": [
            {
                "path": source_relative_path,
                "size": len(source_bytes),
                "sha256": source_sha256,
            }
        ],
    }
    artifact_directory = _publish_artifact(
        output_directory,
        request_path=request_path,
        request_bytes=request_bytes,
        source_relative_path=source_relative_path,
        source_bytes=source_bytes,
        manifest=manifest,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "mthreads",
        "provider": PROVIDER_NAME,
        "node_count": preliminary["graph"]["node_count"],
        "stage_count": len(plan.stages),
        "target": request.target,
        "artifact_directory": str(artifact_directory),
        "workspace_size": plan.workspace_size,
        "execution_engine": execution_engine,
        "torch_loaded": "torch" in sys.modules,
    }
