"""Compile planned kernels and emit libtriton_jit launch metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import math

from flagdnn_codegen.kernel_registry import materialize_kernel_source
from flagdnn_codegen.kernel_registry import resolve_kernel_source
from flagdnn_codegen.kernel_registry import select_kernel_candidate
from triton.backends.compiler import GPUTarget
import triton

from ..dispatch.common import LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
from ..dispatch.selection import _kernel_configuration
from .io import _atomic_write, _load_generated_module
from ..dispatch.tensor import _build_argument_abi
from ..dispatch.tuning import (
    _prepare_tuning_variants,
    _default_compile_options,
    _FIXED_LAUNCH_KERNELS,
)
from .resources import _launch_scratch_size, _effective_jit_warps
from .tensor_map import resolve_tensor_maps
from .jit_cache import emit_jit_cache


def _libtriton_jit_runtime_signature(
    function: Any,
    runtime_signature: dict[str, str],
    argument_abi: list[dict[str, Any]],
) -> dict[str, str]:
    if (
        len(argument_abi) < 2
        or argument_abi[-2].get("kind") != "global_scratch_pointer"
        or argument_abi[-1].get("kind") != "profile_scratch_pointer"
    ):
        raise ValueError(
            "libtriton_jit argument ABI has invalid scratch slots"
        )

    runtime_names = [
        name for name in function.arg_names if name in runtime_signature
    ]
    runtime_arguments = argument_abi[:-2]
    if len(runtime_names) != len(runtime_arguments):
        raise ValueError(
            "libtriton_jit runtime signature and argument ABI disagree"
        )

    result = dict(runtime_signature)
    for name, argument in zip(runtime_names, runtime_arguments):
        kind = argument.get("kind")
        if kind not in {"tensor", "workspace_tensor"}:
            continue
        alignment = (
            16 if kind == "workspace_tensor" else argument.get("alignment", 1)
        )
        if alignment < 16:
            continue
        token = result[name]
        if not token.startswith("*") or ":" in token:
            raise ValueError(
                "libtriton_jit workspace argument is not a plain pointer"
            )
        # The execution engine enforces each external tensor's declared
        # alignment and the packed workspace's fixed 16-byte alignment.
        result[name] = f"{token}:16"
    return result


def _libtriton_jit_full_signature(
    function: Any,
    runtime_signature: dict[str, str],
    constants: dict[str, int | float | str | bool],
    *,
    cached: bool = False,
) -> str:
    argument_names = list(function.arg_names)
    if set(runtime_signature).union(constants) != set(argument_names):
        raise ValueError(
            "libtriton_jit signature does not cover every kernel argument"
        )

    tokens: list[str] = []
    for name in argument_names:
        if name in runtime_signature:
            token = runtime_signature[name]
            if (
                not isinstance(token, str)
                or not token
                or ("," in token and not cached)
            ):
                raise ValueError("libtriton_jit runtime signature is invalid")
            tokens.append(token)
            continue

        value = constants[name]
        if isinstance(value, bool):
            tokens.append("true" if value else "false")
        elif isinstance(value, int):
            tokens.append(str(value))
        elif isinstance(value, float) and math.isfinite(value):
            tokens.append(repr(value))
        else:
            raise ValueError("libtriton_jit supports only numeric constexprs")
    return ",".join(tokens)


def _libtriton_jit_launch(
    grid: tuple[int, int, int] | list[int],
    num_warps: int,
    warp_size: int,
    scratch_size: int = LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
) -> dict[str, Any]:
    return {
        "grid": [int(grid[0]), int(grid[1]), int(grid[2])],
        "block": [num_warps * warp_size, 1, 1],
        "cluster": [1, 1, 1],
        "shared_memory": 0,
        "num_ctas": 1,
        "global_scratch_size": scratch_size,
        "profile_scratch_size": 0,
    }


def _compile_graph_operation(
    *,
    stage_id: int,
    source_node_ids: list[int],
    dependencies: list[int],
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    workspace_layout: dict[int, tuple[int, int]],
    compiler_path: Path,
    output_directory: Path,
    target: GPUTarget,
    enable_autotune: bool,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError(
            "NVIDIA only supports the libtriton_jit execution engine"
        )
    function_name, signature, constants, grid, argument_layout = (
        _kernel_configuration(operation, parameters, tensors, int(target.arch))
    )
    candidate_operation = {
        "matmul_tf32_short_kernel": "matmul_tf32_short",
        "layer_norm_warp_kernel": "layernorm_warp",
        "rms_norm_warp_kernel": "rmsnorm_warp",
    }.get(function_name, operation)
    candidate = select_kernel_candidate("nvidia", candidate_operation)
    kernel_source_path = resolve_kernel_source(compiler_path, candidate)
    kernel_source_bytes = kernel_source_path.read_bytes()
    generated_source_bytes = materialize_kernel_source(
        kernel_source_path, candidate
    )
    has_tensor_maps = any(kind == "tensor_map" for kind, _ in argument_layout)
    if has_tensor_maps:
        # The dependency's constructor initializes its environment but its
        # static-signature helper only accepts ordinary TL JITFunction objects.
        # This host-only signature anchor also supports cached Gluon kernels;
        # it is never compiled or launched. The real cached variant warms CUDA.
        generated_source_bytes += (
            b"\n\n@triton.jit\ndef _flagdnn_jit_initialize():\n    pass\n"
        )
    if (
        not kernel_source_bytes
        or not generated_source_bytes
        or len(kernel_source_bytes) > (1 << 20)
        or len(generated_source_bytes) > (1 << 20)
    ):
        raise ValueError("kernel source size is invalid")

    generated_source = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_source, generated_source_bytes)
    module = _load_generated_module(generated_source, stage_id)
    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and configuration disagree")
    argument_abi = _build_argument_abi(
        argument_layout, tensors, parameters, workspace_layout
    )

    function = getattr(module, function_name)
    function.create_binder()
    use_compiled_cache = has_tensor_maps or function.is_gluon()
    if use_compiled_cache and not has_tensor_maps:
        generated_source_bytes += (
            b"\n\n@triton.jit\ndef _flagdnn_jit_initialize():\n    pass\n"
        )
        if len(generated_source_bytes) > (1 << 20):
            raise ValueError("kernel source size is invalid")
        _atomic_write(generated_source, generated_source_bytes)
    tuning_plan = None
    fixed_tuning_configuration = None
    default_compile_options = _default_compile_options(
        function_name, parameters, constants
    )
    if (
        enable_autotune
        and candidate.tuning is not None
        and function_name not in _FIXED_LAUNCH_KERNELS
    ):
        tuning_plan = _prepare_tuning_variants(
            compiler_path=compiler_path,
            candidate=candidate,
            operation=operation,
            function_name=function_name,
            parameters=parameters,
            constants=constants,
            default_grid=grid,
        )
        if len(tuning_plan[0]) == 1:
            (
                configurations,
                tuning_source_hash,
                candidate_identity,
                selected_table,
            ) = tuning_plan
            (
                selected_configuration,
                constants,
                default_compile_options,
                grid,
            ) = configurations[0]
            tuning = candidate.tuning
            fixed_tuning_configuration = {
                "schema_version": 1,
                "mode": "fixed",
                "source": tuning.source,
                "source_sha256": tuning_source_hash,
                "table": selected_table,
                "key": tuning.key,
                "strategy": tuning.strategy,
                "candidate_identity": candidate_identity,
                "config": selected_configuration,
            }
            tuning_plan = None

    kernel_source_hash = hashlib.sha256(kernel_source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_source_bytes).hexdigest()
    kernel_metadata = {
        "provider": candidate.provider,
        "ownership": candidate.ownership,
        "source": candidate.source,
        "function": function_name,
        "materialized_source": {
            "file": generated_source.name,
            "size": len(generated_source_bytes),
            "sha256": generated_source_hash,
        },
    }

    def make_jit_variant(
        variant_id: str,
        variant_constants: dict[str, int | float | str | bool],
        compile_options: dict[str, Any],
        variant_grid: tuple[int, int, int] | list[int],
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        unsupported_options = set(compile_options).difference(
            {"num_warps", "num_stages"}
        )
        if unsupported_options:
            names = ", ".join(sorted(unsupported_options))
            raise ValueError(
                "installed libtriton_jit raw API cannot consume "
                f"compile options: {names}"
            )
        num_warps = compile_options.get("num_warps")
        num_stages = compile_options.get("num_stages")
        if (
            isinstance(num_warps, bool)
            or not isinstance(num_warps, int)
            or num_warps <= 0
            or isinstance(num_stages, bool)
            or not isinstance(num_stages, int)
            or num_stages <= 0
        ):
            raise ValueError(
                "libtriton_jit compile options must be positive integers"
            )
        # Resource planning must precede both prepare and autotune launches.
        # Use the exact libtriton_jit AST/signature/options so its later
        # get_kernel call reuses Triton's compiled cache entry.
        variant_signature, variant_abi = (
            resolve_tensor_maps(
                signature,
                argument_abi,
                variant_constants,
                gluon=function.is_gluon(),
            )
            if has_tensor_maps
            else (signature, argument_abi)
        )
        jit_runtime_signature = _libtriton_jit_runtime_signature(
            function, variant_signature, variant_abi
        )
        resource_signature = {
            **variant_signature,
            **{name: "constexpr" for name in variant_constants},
        }
        resource_attrs = {}
        for argument_index, argument_name in enumerate(function.arg_names):
            token = jit_runtime_signature.get(argument_name, "")
            if token.endswith(":16"):
                resource_attrs[(argument_index,)] = [["tt.divisibility", 16]]
        resource_source = function.ASTSource(
            fn=function,
            signature=resource_signature,
            constexprs=variant_constants,
            attrs=resource_attrs,
        )
        resource_options = dict(compile_options)
        if function_name == "matmul_tf32_short_kernel":
            resource_options["default_dot_input_precision"] = "tf32"
        compiled_resource = triton.compile(
            resource_source, target=target, options=resource_options
        )
        effective_warps = _effective_jit_warps(
            compiled_resource.metadata, num_warps, use_compiled_cache
        )
        scratch_size = _launch_scratch_size(
            compiled_resource.metadata,
            tuple(variant_grid),
            LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
        )
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _libtriton_jit_full_signature(
                function,
                jit_runtime_signature,
                variant_constants,
                cached=use_compiled_cache,
            ),
            "compile_options": {
                # Cache execution needs the compiler's final CTA size. The
                # original tuning config still records the producer warp count.
                "num_warps": effective_warps,
                "num_stages": num_stages,
            },
            "argument_abi": variant_abi,
            "launch": _libtriton_jit_launch(
                variant_grid,
                effective_warps,
                int(target.warp_size),
                scratch_size,
            ),
        }
        if use_compiled_cache:
            result["compiled_cache"] = emit_jit_cache(
                compiled_resource, output_directory, stage_id, variant_id
            )
        if configuration is not None:
            result["config"] = configuration
        return result

    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": source_node_ids,
        "dependencies": dependencies,
        "operation": operation,
        "source_sha256": kernel_source_hash,
        "kernel": kernel_metadata,
    }
    if tuning_plan is not None:
        tuning = candidate.tuning
        if tuning is None:
            raise RuntimeError("autotune plan lost its tuning metadata")
        (
            prepared_variants,
            tuning_source_hash,
            candidate_identity,
            selected_table,
        ) = tuning_plan
        variants: list[dict[str, Any]] = []
        for variant_index, (
            configuration,
            variant_constants,
            compile_options,
            variant_grid,
        ) in enumerate(prepared_variants):
            variants.append(
                make_jit_variant(
                    f"config_{variant_index}",
                    variant_constants,
                    compile_options,
                    variant_grid,
                    configuration,
                )
            )
        jit_identity_payload = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                    "argument_abi": variant["argument_abi"],
                    "compiled_cache": variant.get("compiled_cache"),
                }
                for variant in variants
            ],
        }
        candidate_identity = hashlib.sha256(
            json.dumps(
                jit_identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": tuning.source,
            "source_sha256": tuning_source_hash,
            "table": selected_table,
            "key": tuning.key,
            "strategy": tuning.strategy,
            "warmup": tuning.warmup,
            "repetitions": tuning.repetitions,
            "candidate_identity": candidate_identity,
        }
    else:
        stage.update(
            make_jit_variant(
                "default",
                constants,
                default_compile_options,
                grid,
            )
        )
        if fixed_tuning_configuration is not None:
            stage["tuning_configuration"] = fixed_tuning_configuration
    return stage
