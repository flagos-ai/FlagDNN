"""Tuning-table validation and candidate launch variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import hashlib
import itertools
import json

from flagdnn_codegen.kernel_registry import KernelCandidate
from flagdnn_codegen.kernel_registry import resolve_tuning_source
import yaml

from .common import _ceil_div, _require_integer


_FIXED_LAUNCH_KERNELS = {
    "attention_partial_gradient_reduce_kernel",
    "conv_dgrad_direct_kernel",
    "conv_fprop_direct_kernel",
    "conv_wgrad_direct_kernel",
    "_zero_sdpa_fp8_fwd_amax_kernel",
    "_zero_sdpa_fp8_bwd_amax_kernel",
}


def _default_compile_options(
    function_name: str,
    parameters: dict[str, Any],
    constants: dict[str, int | float | str | bool],
) -> dict[str, Any]:
    """Select launch compiler options independently of artifact emission."""
    default_compile_options: dict[str, Any] = {
        "num_warps": 4,
        "num_stages": 1,
    }
    if (
        function_name
        in {"matmul_batched_contiguous_kernel", "matmul_strided_kernel"}
        and parameters.get("input_precision", 0) != 0
    ):
        default_compile_options = {"num_warps": 4, "num_stages": 3}
    elif function_name == "matmul_batched_tma_short_kernel":
        default_compile_options = (
            {"num_warps": 4, "num_stages": 3}
            if constants["BLOCK_M"] == 64
            else {"num_warps": 8, "num_stages": 4}
        )
    elif function_name in {
        "matmul_tf32_tma_kernel",
        "matmul_tf32_tma_direct_kernel",
        "matmul_tf32_tensor_map_kernel",
    }:
        default_compile_options = {"num_warps": 8, "num_stages": 3}
    elif function_name == "_conv_wgrad2d_batched_split_kernel":
        default_compile_options = {"num_warps": 4, "num_stages": 2}
    elif function_name in {
        "_sdpa_bwd_dq_dbias_kernel",
        "_sdpa_bwd_dkdv_kernel",
        "_sdpa_bwd_dk_kernel",
        "_sdpa_bwd_dv_kernel",
    }:
        # Match the fixed backward tuning tables. A one-stage MMA pipeline
        # accesses invalid shared memory for short FP16 tiles on SM90.
        default_compile_options = {"num_warps": 4, "num_stages": 2}
    elif function_name == "_sdpa_fp8_fwd_kernel":
        # Match the validated four-warp, BLOCK_N=128 FP8 tuning layouts.
        default_compile_options = {"num_warps": 4, "num_stages": 3}
    if function_name in {
        "_sdpa_fp8_bwd_dq_kernel",
        "_sdpa_fp8_bwd_dkdv_kernel",
    }:
        default_compile_options = {"num_warps": 4, "num_stages": 2}
    if function_name == "fp8_matmul_kernel":
        default_compile_options = {"num_warps": 4, "num_stages": 3}
    if function_name in {
        "compact_normalization_forward",
        "compact_batchnorm_backward",
    }:
        block = int(constants["BLOCK_SIZE"])
        default_compile_options = {
            "num_warps": (
                1
                if block <= 1024
                else 4 if block <= 8192 else 8 if block <= 32768 else 16
            ),
            "num_stages": 1,
        }
    if parameters.get("input_precision", 0) != 0 and function_name.startswith(
        "conv"
    ):
        default_compile_options = {"num_warps": 4, "num_stages": 3}
    if (
        function_name == "compact_batchnorm_backward"
        and int(constants["BLOCK_SIZE"]) > 32768
    ):
        # Limit per-thread state for full rows.
        default_compile_options["num_warps"] = 32
    if function_name == "conv_fprop_direct_kernel":
        default_compile_options = {"num_warps": 1, "num_stages": 1}
    return default_compile_options


def _canonical_tuning_value(value: Any, context: str) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} must contain JSON-compatible finite values"
        ) from error


def _flatten_tuning_param_map(
    value: object, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("tuning param_map must be a nonempty mapping")
    result: list[tuple[tuple[str, ...], str]] = []
    for output_name, source in value.items():
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("tuning param_map output names must be nonempty")
        output_path = (*path, output_name)
        if isinstance(source, dict):
            result.extend(_flatten_tuning_param_map(source, output_path))
        elif isinstance(source, str) and source:
            result.append((output_path, source))
        else:
            raise ValueError(
                "tuning param_map leaves must name a source parameter"
            )
    return result


def _assign_tuning_value(
    configuration: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    current = configuration
    for name in path[:-1]:
        existing = current.get(name)
        if existing is None:
            nested: dict[str, Any] = {}
            current[name] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(
                f"tuning output path collides at {'.'.join(path)}"
            )
    if path[-1] in current:
        raise ValueError(f"duplicate tuning output: {'.'.join(path)}")
    current[path[-1]] = copy.deepcopy(value)


def _tuning_parameter_values(
    generated: dict[str, Any], source_name: str
) -> list[Any]:
    if source_name not in generated:
        raise ValueError(
            f"tuning param_map references missing parameter: {source_name}"
        )
    source = generated[source_name]
    values = source if isinstance(source, list) else [source]
    if not values:
        raise ValueError(f"empty tuning parameter list: {source_name}")
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        canonical = _canonical_tuning_value(
            value, f"tuning parameter {source_name}"
        )
        if canonical not in seen:
            seen.add(canonical)
            unique.append(value)
    return unique


def _expand_generated_tuning_entry(
    generated: dict[str, Any]
) -> list[dict[str, Any]]:
    leaves = _flatten_tuning_param_map(generated.get("param_map"))
    output_paths: set[tuple[str, ...]] = set()
    source_names: list[str] = []
    for output_path, source_name in leaves:
        if output_path in output_paths:
            raise ValueError(
                f"duplicate tuning output: {'.'.join(output_path)}"
            )
        output_paths.add(output_path)
        if source_name not in source_names:
            source_names.append(source_name)

    dimensions = [
        _tuning_parameter_values(generated, source_name)
        for source_name in source_names
    ]
    source_set = set(source_names)
    literal_fields = {
        name: value
        for name, value in generated.items()
        if name not in {"gen", "param_map"} and name not in source_set
    }

    configurations: list[dict[str, Any]] = []
    for combination in itertools.product(*dimensions):
        selected = dict(zip(source_names, combination, strict=True))
        configuration: dict[str, Any] = {}
        for output_path, source_name in leaves:
            _assign_tuning_value(
                configuration, output_path, selected[source_name]
            )
        for name, value in literal_fields.items():
            _assign_tuning_value(configuration, (name,), value)
        configurations.append(configuration)
    return configurations


def _expand_tuning_table(entries: list[object]) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry:
            raise ValueError(
                f"tuning entry {entry_index} must be a nonempty mapping"
            )
        if entry.get("gen") is True:
            expanded = _expand_generated_tuning_entry(entry)
        elif "gen" in entry:
            raise ValueError(
                f"tuning entry {entry_index}.gen must be true when present"
            )
        else:
            expanded = [copy.deepcopy(entry)]
        for configuration in expanded:
            canonical = _canonical_tuning_value(
                configuration, f"tuning entry {entry_index}"
            )
            if canonical not in seen:
                seen.add(canonical)
                configurations.append(configuration)
    if not configurations:
        raise ValueError("kernel tuning table produced no configurations")
    return configurations


def _split_tuning_configuration(
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_meta = configuration.get("META")
    if not isinstance(raw_meta, dict) or not raw_meta:
        raise ValueError("tuning configuration META must be nonempty")
    meta = dict(raw_meta)
    nested_options: dict[str, Any] = {}
    for option_name in ("num_warps", "num_stages"):
        if option_name not in meta:
            continue
        if option_name in configuration:
            raise ValueError(
                f"tuning configuration defines {option_name} twice"
            )
        nested_options[option_name] = meta.pop(option_name)
    if not meta:
        raise ValueError("tuning configuration META has no kernel constants")
    for name, value in meta.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tuning META names must be nonempty strings")
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"tuning META.{name} must be a scalar")

    options: dict[str, Any] = {
        "num_warps": 4,
        "num_stages": 1,
    }
    # Preserve legacy generated search spaces whose launch options were
    # accidentally nested below META. They are compiler options, not Triton
    # constexpr parameters.
    options.update(nested_options)
    options.update(
        {
            name: value
            for name, value in configuration.items()
            if name != "META"
        }
    )
    for name, value in options.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tuning option names must be nonempty strings")
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"tuning option {name} must be a scalar")
    for required in ("num_warps", "num_stages"):
        value = options.get(required)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"tuning configuration {required} must be positive"
            )
    return meta, options


def _load_tuning_configurations(
    compiler_path: Path,
    candidate: KernelCandidate,
    *,
    table: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("kernel candidate has no tuning configuration")

    table_name = tuning.table if table is None else table

    tuning_source = resolve_tuning_source(compiler_path, candidate)
    tuning_bytes = tuning_source.read_bytes()
    document = yaml.safe_load(tuning_bytes)
    if not isinstance(document, dict):
        raise ValueError("kernel tuning source must be a YAML mapping")
    entries = document.get(table_name)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"missing tuning table: {table_name}")

    expanded_configurations = _expand_tuning_table(entries)
    configurations: list[dict[str, Any]] = []
    seen_configurations: set[str] = set()
    for raw_configuration in expanded_configurations:
        meta, options = _split_tuning_configuration(raw_configuration)
        configuration = {"META": meta, **options}
        canonical = _canonical_tuning_value(
            configuration, "normalized tuning configuration"
        )
        if canonical not in seen_configurations:
            seen_configurations.add(canonical)
            configurations.append(configuration)

    selected_tuning_payload = {
        "schema_version": 1,
        "table": table_name,
        "configurations": configurations,
    }
    selected_tuning_bytes = json.dumps(
        selected_tuning_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    tuning_source_hash = hashlib.sha256(selected_tuning_bytes).hexdigest()
    identity_payload = {
        "schema_version": 2,
        "source_sha256": tuning_source_hash,
        "kernel_ownership": candidate.ownership,
        "kernel_backend": candidate.backend,
        "table": table_name,
        "key": tuning.key,
        "strategy": tuning.strategy,
        "configurations": configurations,
    }
    canonical_identity = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        configurations,
        tuning_source_hash,
        hashlib.sha256(canonical_identity).hexdigest(),
    )


def _selected_tuning_table(
    candidate: KernelCandidate,
    operation: str,
    function_name: str,
    parameters: dict[str, Any],
    constants: dict[str, int | float | str | bool],
) -> str:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("kernel candidate has no tuning configuration")
    if function_name == "matrix_transpose_kernel":
        return "matrix_transpose"
    if function_name == "matmul_batched_tma_short_kernel":
        return "matmul_short_persistent"
    if function_name == "matmul_tf32_pack_b_kernel":
        return "matmul_tf32_pack"
    if function_name == "matmul_tf32_tma_kernel":
        return "matmul_tf32_tma"
    if function_name == "matmul_tf32_tma_direct_kernel":
        return "matmul_tf32_tma_direct"
    if function_name == "matmul_tf32_tensor_map_kernel":
        return "matmul_tf32_tensor_map"
    if function_name == "matmul_tf32_short_kernel":
        return "matmul_tf32_short"
    if (
        function_name == "matmul_strided_kernel"
        and parameters.get("_matmul_k_contiguous") is True
    ):
        return "matmul_fp32_k_contiguous"
    if (
        function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel"
        and parameters.get("_dgrad_k_contiguous") is True
    ):
        return "conv_dgrad_p5_k_contiguous"
    if function_name == "matmul_p5_split_k_reduce_kernel":
        return "unary"
    if constants.get("INPUT_IS_FLOAT32") is True and (
        function_name == "matmul_p5_split_k_kernel"
        or (
            function_name == "matmul_strided_kernel"
            and parameters.get("_fprop_im2col_matmul") is True
            and constants.get("B_STRIDE_K") == 1
        )
    ):
        return "conv_fprop_fp32_mm"
    if function_name == "_conv_wgrad2d_split_vector_reduce_kernel":
        return "reduction"
    attention_tables = {
        "_sdpa_fwd_kernel": "sdpa",
        "_zero_contiguous_kernel": "sdpa_backward_zero_delta",
        "_sdpa_bwd_dq_dbias_kernel": "sdpa_backward_dq",
        "_sdpa_bwd_dkdv_kernel": "sdpa_backward_dkdv",
        "_sdpa_bwd_dk_kernel": "sdpa_backward_dk",
        "_sdpa_bwd_dv_kernel": "sdpa_backward_dv",
        "_sdpa_fp8_fwd_kernel": "sdpa_fp8",
        "_sdpa_fp8_bwd_dq_kernel": "sdpa_fp8_backward_dq",
        "_sdpa_fp8_bwd_dkdv_kernel": "sdpa_fp8_backward_dkdv",
    }
    attention_table = attention_tables.get(function_name)
    if attention_table is not None and operation in {
        "sdpa",
        "sdpa_backward",
        "sdpa_fp8",
        "sdpa_fp8_backward",
    }:
        if (
            operation == "sdpa_backward"
            and parameters.get("_sdpa_bwd_fp32")
            and function_name != "_zero_contiguous_kernel"
        ):
            return attention_table + "_fp32"
        return attention_table
    if tuning.strategy != "convolution":
        return tuning.table
    if (
        function_name == "conv2d_spatial_nchw_kernel"
        and parameters.get("_fprop_small_reduction") is True
    ):
        return "conv_fprop_small_reduction_fp32"
    if function_name in {
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
    }:
        return "mm"
    if function_name == "_conv_wgrad2d_batched_split_kernel":
        return "conv_wgrad_short_split"
    if function_name == "_conv_wgrad2d_batched_tma_kernel":
        return "conv_wgrad_2d"

    forward_tables = {
        "conv1d_gemm_kernel": "conv1d_gemm_v3",
        "conv2d_1x1_nchw_pad0_kernel": "conv2d_1x1",
        "conv2d_spatial_nchw_kernel": "conv2d_spatial",
        "conv2d_im2col_nchw_kernel": "unary",
        "conv2d_im2col_nchw_transposed_kernel": "conv_fprop_im2col_transposed",
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel": "unary",
        "conv3d_spatial_ncdhw_m_kernel": "conv_fprop_3d",
        "conv_dgrad2d_1x1_nchw_kernel": "conv_dgrad_2d_1x1",
        "conv_dgrad2d_stride1_kernel": "conv_dgrad_2d_stride1",
        "conv_dgrad2d_pack_weight_kernel": "batch_norm",
        "zero_contiguous_kernel": "batch_norm",
        "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel": (
            "conv_dgrad_2d_stride2_pad1_3x3_tile2w"
        ),
        "conv_dgrad3d_pack_weight_kernel": "batch_norm",
        "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel": "conv_dgrad_3d",
        "conv_dgrad3d_packed_kernel": "conv_dgrad_3d_packed",
        "_conv_wgrad2d_1x1_direct_nodiv_kernel": "conv_wgrad_2d_1x1",
        "_conv_wgrad2d_1x1_split_nodiv_kernel": "conv_wgrad_2d_1x1",
        "_conv_wgrad2d_1x1_reduce_kernel": "conv_wgrad_2d_1x1",
        "_conv_wgrad2d_3tap_split_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_stride2_row4_split_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_reduce_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_col_split_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_col_reduce_kernel": "conv_wgrad_2d",
    }
    table = forward_tables.get(function_name)
    if table is not None:
        return table
    if function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel":
        return "conv_dgrad_2d_stride2_pad1_3x3_packed_mci"
    if function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel":
        if parameters.get("_dgrad_small_ci") is True:
            return "conv_dgrad_2d_stride2_pad1_3x3_tile4"
        return "conv_dgrad_2d_stride2_pad1_3x3_tile2w"
    if function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel":
        return "conv_dgrad_2d_stride2_pad1_3x3_tile4"
    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    if operation == "convolution_dgrad":
        return f"conv_dgrad_{spatial_rank}d"
    if operation == "convolution_wgrad":
        return f"conv_wgrad_{spatial_rank}d"
    raise ValueError(
        "convolution tuning strategy received an unknown kernel entry"
    )


def _translated_tuning_meta(
    function_name: str, meta: dict[str, Any]
) -> dict[str, Any]:
    result = dict(meta)
    alias: tuple[str, str] | None = None
    if function_name == "conv_dgrad_nd_kernel":
        alias = ("BLOCK_CO", "BLOCK_K")
    elif function_name == "conv_wgrad_nd_kernel":
        alias = ("BLOCK_CO", "BLOCK_OC")
    elif function_name in {
        "_conv_wgrad2d_col_split_kernel",
        "_conv_wgrad2d_col_reduce_kernel",
    }:
        alias = ("BLOCK_CI", "BLOCK_N")
    if alias is not None and alias[0] in result:
        if alias[1] in result:
            raise ValueError(
                f"tuning META defines both {alias[0]} and {alias[1]}"
            )
        result[alias[1]] = result.pop(alias[0])
    if function_name == "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel":
        result.pop("BLOCK_CI", None)
        result.pop("BLOCK_CO", None)
    if function_name in {
        "_conv_wgrad2d_1x1_reduce_kernel",
        "_conv_wgrad2d_reduce_kernel",
        "_conv_wgrad2d_col_reduce_kernel",
    }:
        result.pop("BLOCK_M", None)
    if function_name == "_conv_wgrad2d_stride2_row4_split_kernel":
        result.pop("BLOCK_M", None)
    if function_name in {
        "conv2d_im2col_nchw_kernel",
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
    }:
        result.pop("TILES_PER_PROGRAM", None)
    if function_name == "matmul_p5_split_k_reduce_kernel":
        result.pop("TILES_PER_PROGRAM", None)
    if function_name == "_zero_contiguous_kernel" and "BLOCK_ZERO" in result:
        result["BLOCK"] = result.pop("BLOCK_ZERO")
        result.pop("BLOCK_M", None)
        result.pop("BLOCK_D", None)
    return result


def _positive_tuning_integer(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"tuning META.{name} must be a positive integer")
    return value


def _autotune_variant_grid(
    *,
    strategy: str,
    key: str,
    function_name: str,
    parameters: dict[str, Any],
    constants: dict[str, int | float | str | bool],
    meta: dict[str, Any],
    default_grid: tuple[int, int, int],
) -> tuple[int, int, int]:
    key_value = _require_integer(parameters, key)
    if function_name == "matmul_tf32_pack_b_kernel":
        block = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(int(constants["K"]), block)
            * _ceil_div(int(constants["N"]), block),
            int(constants["BATCH"]),
            1,
        )
    if function_name == "matrix_transpose_kernel":
        block = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(int(constants["ROWS"]), block),
            _ceil_div(int(constants["COLUMNS"]), block),
            1,
        )
    if strategy == "attention":
        if function_name in {"_sdpa_fwd_kernel", "_sdpa_fp8_fwd_kernel"}:
            block_m = _positive_tuning_integer(meta, "BLOCK_M")
            _positive_tuning_integer(meta, "BLOCK_N")
            return (
                _ceil_div(_require_integer(parameters, "sequence_q"), block_m),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "heads"),
                1,
            )
        if function_name == "_zero_contiguous_kernel":
            block = _positive_tuning_integer(meta, "BLOCK")
            return (
                _ceil_div(
                    _require_integer(parameters, "dbias_elements"), block
                ),
                1,
                1,
            )
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        if function_name == "_sdpa_bwd_dq_dbias_kernel":
            block_d = _positive_tuning_integer(meta, "BLOCK_D_OUT")
            return (
                _ceil_div(_require_integer(parameters, "sequence_q"), block_m),
                _ceil_div(
                    _require_integer(parameters, "head_dimension"), block_d
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "heads"),
            )
        if function_name in {
            "_sdpa_bwd_dkdv_kernel",
            "_sdpa_bwd_dk_kernel",
        }:
            block_d = _positive_tuning_integer(meta, "BLOCK_D_OUT")
            return (
                _ceil_div(
                    _require_integer(parameters, "sequence_kv"), block_n
                ),
                _ceil_div(
                    _require_integer(parameters, "head_dimension"), block_d
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "key_heads")
                * parameters.get("_sdpa_partial_count", 1),
            )
        if function_name == "_sdpa_bwd_dv_kernel":
            block_dv = _positive_tuning_integer(meta, "BLOCK_DV_OUT")
            return (
                _ceil_div(
                    _require_integer(parameters, "sequence_kv"), block_n
                ),
                _ceil_div(
                    _require_integer(parameters, "value_dimension"), block_dv
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "value_heads"),
            )
        if function_name == "_sdpa_fp8_bwd_dq_kernel":
            return (
                _ceil_div(_require_integer(parameters, "sequence_q"), block_m),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "heads"),
                1,
            )
        if function_name == "_sdpa_fp8_bwd_dkdv_kernel":
            return (
                _ceil_div(
                    _require_integer(parameters, "sequence_kv"), block_n
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "key_heads")
                * parameters.get("_sdpa_partial_count", 1),
                1,
            )
        raise ValueError("attention tuning received an unknown kernel entry")
    if function_name == "matmul_p5_split_k_reduce_kernel":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "TOTAL"), block_size
            ),
            1,
            1,
        )
    if function_name == "conv2d_im2col_nchw_transposed_kernel":
        return (
            _ceil_div(
                int(constants["OH"]) * int(constants["OW"]),
                _positive_tuning_integer(meta, "BLOCK_M"),
            ),
            _ceil_div(
                int(constants["CIN_PER_GROUP"])
                * int(constants["KH"])
                * int(constants["KW"]),
                _positive_tuning_integer(meta, "BLOCK_K"),
            ),
            default_grid[2],
        )
    if function_name == "_conv_wgrad2d_split_vector_reduce_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        _positive_tuning_integer(meta, "BLOCK_N")
        return (
            _ceil_div(_positive_tuning_integer(constants, "TOTAL"), block_m),
            1,
            1,
        )
    if function_name == "zero_contiguous_kernel":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "TOTAL"), block_size
            ),
            1,
            1,
        )
    if function_name == "_conv_wgrad2d_p5_pack_image_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        _positive_tuning_integer(meta, "BLOCK_K")
        _positive_tuning_integer(meta, "GROUP_M")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m),
            _ceil_div(_positive_tuning_integer(constants, "N"), block_n),
            1,
        )
    if function_name == "_conv_wgrad2d_p5_mm_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        _positive_tuning_integer(meta, "BLOCK_K")
        _positive_tuning_integer(meta, "GROUP_M")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(_positive_tuning_integer(constants, "N"), block_n),
            1,
            1,
        )
    if strategy == "align32":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        if function_name == "batch_norm_inference_nchw_kernel":
            spatial = _require_integer(parameters, "spatial", minimum=1)
            channels = _require_integer(parameters, "channels", minimum=1)
            elements_per_batch = channels * spatial
            if key_value % elements_per_batch != 0:
                raise ValueError(
                    "BatchNorm Inference elements are not divisible by C*S"
                )
            block_s = min(1 << (spatial - 1).bit_length(), block_size)
            block_c = block_size // block_s
            return (
                (key_value // elements_per_batch)
                * _ceil_div(channels, block_c)
                * _ceil_div(spatial, block_s),
                1,
                1,
            )
        tiles_per_program = meta.get("TILES_PER_PROGRAM", 1)
        if (
            isinstance(tiles_per_program, bool)
            or not isinstance(tiles_per_program, int)
            or tiles_per_program <= 0
        ):
            raise ValueError(
                "tuning META.TILES_PER_PROGRAM must be a positive integer"
            )
        return (
            _ceil_div(key_value, block_size * tiles_per_program),
            1,
            1,
        )

    if strategy == "reduction":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        _positive_tuning_integer(meta, "BLOCK_N")
        return (_ceil_div(key_value, block_m), 1, 1)

    if strategy == "fixed_grid":
        if function_name in {"layer_norm_kernel", "rms_norm_kernel"}:
            return (
                _ceil_div(
                    _require_integer(parameters, "rows", minimum=1),
                    _positive_tuning_integer(constants, "ROWS_PER_PROGRAM"),
                ),
                1,
                1,
            )
        return tuple(int(value) for value in default_grid)

    if strategy == "matmul":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        _positive_tuning_integer(meta, "BLOCK_K")
        if function_name == "matmul_tf32_short_kernel":
            if _positive_tuning_integer(meta, "SLOTS") not in {3, 4}:
                raise ValueError(
                    "short-K MatMul requires three or four input slots"
                )
        else:
            _positive_tuning_integer(meta, "GROUP_M")
        m = _positive_tuning_integer(constants, "M")
        n = _positive_tuning_integer(constants, "N")
        if function_name in {
            "matmul_batched_tma_persistent_kernel",
            "matmul_batched_tma_short_kernel",
            "matmul_tf32_tma_kernel",
            "matmul_tf32_tma_direct_kernel",
            "matmul_tf32_tensor_map_kernel",
            "matmul_tf32_short_kernel",
        }:
            batch = _positive_tuning_integer(constants, "BATCH")
            persistent_grid = _positive_tuning_integer(
                constants, "PERSISTENT_GRID"
            )
            total_tiles = batch * _ceil_div(m, block_m) * _ceil_div(n, block_n)
            if function_name == "matmul_tf32_short_kernel":
                total_tiles //= 2
            return (min(total_tiles, persistent_grid), 1, 1)
        return (
            _ceil_div(m, block_m) * _ceil_div(n, block_n),
            int(default_grid[1]),
            int(default_grid[2]),
        )

    if strategy != "convolution":
        raise ValueError(f"unknown kernel tuning strategy: {strategy!r}")

    if function_name in {
        "conv2d_im2col_nchw_kernel",
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
    }:
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        output_area = _positive_tuning_integer(
            constants, "OH"
        ) * _positive_tuning_integer(constants, "OW")
        return (
            _ceil_div(output_area, block_size),
            int(default_grid[1]),
            1,
        )
    if function_name == "conv1d_gemm_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "conv2d_1x1_nchw_pad0_kernel",
        "conv2d_spatial_nchw_kernel",
    }:
        block_hw = _positive_tuning_integer(meta, "BLOCK_HW")
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        output_spatial = (
            _positive_tuning_integer(constants, "HW")
            if function_name == "conv2d_1x1_nchw_pad0_kernel"
            else (
                _positive_tuning_integer(constants, "OH")
                * _positive_tuning_integer(constants, "OW")
            )
        )
        return (
            _ceil_div(output_spatial, block_hw)
            * _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv3d_spatial_ncdhw_m_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_dgrad2d_1x1_nchw_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(_positive_tuning_integer(constants, "HW"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "conv_dgrad2d_stride1_kernel",
        "conv_dgrad_nd_kernel",
        "conv_dgrad3d_packed_kernel",
    }:
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "conv_dgrad2d_pack_weight_kernel",
        "conv_dgrad3d_pack_weight_kernel",
    }:
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "C_OUT")
                * _positive_tuning_integer(constants, "C_IN"),
                block_size,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        group_k = _positive_tuning_integer(constants, "GROUP_K")
        split_k_blocks = _ceil_div(
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            ),
            group_k,
        )
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            )
            * split_k_blocks,
            1,
            1,
        )
    if function_name in {
        "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel",
        "conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel",
        "conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel",
    }:
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "M"),
                block_m,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_wgrad_nd_kernel":
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "_conv_wgrad2d_3tap_split_kernel":
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "_conv_wgrad2d_batched_tma_kernel",
        "_conv_wgrad2d_batched_split_kernel",
    }:
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "C_OUT"),
                block_co,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIK"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "_conv_wgrad2d_1x1_direct_nodiv_kernel",
        "_conv_wgrad2d_1x1_split_nodiv_kernel",
        "_conv_wgrad2d_1x1_reduce_kernel",
        "_conv_wgrad2d_stride2_row4_split_kernel",
        "_conv_wgrad2d_reduce_kernel",
    }:
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "_conv_wgrad2d_col_split_kernel",
        "_conv_wgrad2d_col_reduce_kernel",
    }:
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        cik = (
            _positive_tuning_integer(constants, "CIN_PER_GROUP")
            * _positive_tuning_integer(constants, "KH")
            * _positive_tuning_integer(constants, "KW")
        )
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            )
            * _ceil_div(cik, block_n),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    raise ValueError(
        "convolution tuning strategy received an unknown kernel entry"
    )


def _prepare_tuning_variants(
    *,
    compiler_path: Path,
    candidate: KernelCandidate,
    operation: str,
    function_name: str,
    parameters: dict[str, Any],
    constants: dict[str, int | float | str | bool],
    default_grid: tuple[int, int, int],
) -> tuple[
    list[
        tuple[
            dict[str, Any],
            dict[str, int | float | str | bool],
            dict[str, Any],
            tuple[int, int, int],
        ]
    ],
    str,
    str,
    str,
]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("kernel candidate has no tuning configuration")

    selected_table = _selected_tuning_table(
        candidate, operation, function_name, parameters, constants
    )
    configurations, tuning_source_hash, _ = _load_tuning_configurations(
        compiler_path, candidate, table=selected_table
    )
    if function_name == "conv2d_im2col_nchw_kernel":
        for block_size in (512, 1024):
            configurations.append(
                {
                    "META": {"BLOCK_SIZE": block_size},
                    "num_warps": 2,
                    "num_stages": 1,
                }
            )
    if function_name == "matmul_batched_broadcast_a_kernel":
        for block_m, block_n, block_k, num_stages in (
            (64, 64, 32, 4),
            (64, 64, 64, 5),
            (32, 64, 64, 4),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": 8,
                    },
                    "num_warps": 4,
                    "num_stages": num_stages,
                }
            )
    if (
        function_name == "matmul_strided_kernel"
        and parameters.get("_fprop_im2col_matmul") is True
        and constants.get("B_STRIDE_K") != 1
    ):
        for num_warps in (4, 8):
            for num_stages in (3, 4, 5):
                configurations.append(
                    {
                        "META": {
                            "BLOCK_M": 64,
                            "BLOCK_N": 32,
                            "BLOCK_K": 64,
                            "GROUP_M": 8,
                        },
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                )
    if function_name == "_conv_wgrad2d_p5_pack_image_kernel":
        for block_m, block_n in (
            (32, 32),
            (16, 64),
            (32, 64),
            (64, 32),
            (16, 128),
            (32, 128),
            (16, 256),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    "num_warps": 4,
                    "num_stages": 1,
                }
            )
    if (
        function_name == "matmul_batched_contiguous_kernel"
        and constants.get("INPUT_IS_FLOAT32") is True
    ):
        for (
            block_m,
            block_n,
            block_k,
            group_m,
            num_warps,
            num_stages,
        ) in (
            (128, 256, 32, 4, 8, 3),
            (256, 128, 32, 4, 8, 3),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": group_m,
                    },
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
            )
    if function_name == "matmul_batched_tma_persistent_kernel":
        configurations.append(
            {
                "META": {
                    "PERSISTENT_GRID": 132,
                    "BLOCK_M": 128,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 16,
                },
                "num_warps": 8,
                "num_stages": 4,
            }
        )
    if function_name == "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel":
        configurations.append(
            {
                "META": {
                    "BLOCK_M": 32,
                    "BLOCK_CI": 64,
                    "BLOCK_CO": 64,
                },
                "num_warps": 4,
                "num_stages": 3,
            }
        )
    if function_name == "conv_dgrad2d_1x1_nchw_kernel":
        for block_m, block_ci, block_co, num_warps, num_stages in (
            (32, 32, 128, 4, 3),
            (64, 32, 64, 4, 3),
            (64, 64, 64, 4, 3),
            (32, 64, 64, 4, 2),
            (16, 64, 128, 4, 2),
            (16, 64, 128, 8, 2),
            (32, 64, 128, 4, 1),
            (32, 64, 128, 4, 2),
            (32, 64, 128, 4, 3),
            (32, 64, 128, 8, 1),
            (32, 64, 128, 8, 2),
            (32, 64, 128, 8, 3),
            (32, 128, 128, 8, 2),
            (32, 128, 128, 8, 3),
            (64, 64, 128, 8, 2),
            (128, 32, 64, 8, 2),
            (128, 32, 128, 8, 2),
            (128, 32, 64, 8, 3),
            (128, 32, 128, 8, 3),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_CI": block_ci,
                        "BLOCK_CO": block_co,
                    },
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
            )
    if (
        function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel"
        and parameters.get("_dgrad_p5_parity") is True
    ):
        for block_co in (128, 256):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": 64,
                        "BLOCK_CI": 64,
                        "BLOCK_CO": block_co,
                    },
                    "num_warps": 8,
                    "num_stages": 2,
                }
            )
    if (
        function_name == "conv_wgrad_nd_kernel"
        and parameters.get("spatial_rank") == 3
    ):
        configurations.append(
            {
                "META": {
                    "BLOCK_CO": 8,
                    "BLOCK_CI": 8,
                    "BLOCK_M": 128,
                },
                "num_warps": 4,
                "num_stages": 1,
            }
        )
        for block_m in (128, 256):
            for num_warps in (4, 8):
                for num_stages in (2, 3):
                    configurations.append(
                        {
                            "META": {
                                "BLOCK_CO": 8,
                                "BLOCK_CI": 8,
                                "BLOCK_M": block_m,
                            },
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
    if function_name in {"layer_norm_kernel", "rms_norm_kernel"}:
        extent = int(constants["N"])
        for warps in (1, 2, 4) if extent <= 1024 else (4, 8):
            configurations.append(
                {
                    "META": {
                        "BLOCK_SIZE": 1 << (extent - 1).bit_length(),
                        "ROWS_PER_PROGRAM": 1,
                    },
                    "num_warps": warps,
                    "num_stages": 1,
                }
            )
    if function_name == "layer_norm_kernel":
        extent = int(constants["N"])
        for warps in (1, 2, 4, 8) if extent <= 1024 else (4, 8):
            configurations.append(
                {
                    "META": {
                        "BLOCK_SIZE": 1 << (extent - 1).bit_length(),
                        "ROWS_PER_PROGRAM": 1,
                        "PAIRED_REDUCTION": True,
                    },
                    "num_warps": warps,
                    "num_stages": 1,
                }
            )
    if function_name == "rms_norm_kernel":
        extent = int(constants["N"])
        for grouped_rows, warps in (
            ((2, 2), (2, 4)) if extent <= 1024 else ((1, 16),)
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_SIZE": 1 << (extent - 1).bit_length(),
                        "ROWS_PER_PROGRAM": grouped_rows,
                    },
                    "num_warps": warps,
                    "num_stages": 1,
                }
            )
    if (
        function_name in {"layer_norm_kernel", "rms_norm_kernel"}
        and int(constants["N"]) <= 1024
    ):
        for rows_per_program in (4, 8):
            meta = {
                "BLOCK_SIZE": 1 << (int(constants["N"]) - 1).bit_length(),
                "ROWS_PER_PROGRAM": rows_per_program,
            }
            if function_name == "layer_norm_kernel":
                meta["PAIRED_REDUCTION"] = True
            configurations.append(
                {"META": meta, "num_warps": 4, "num_stages": 1}
            )
    if (
        function_name == "conv2d_spatial_nchw_kernel"
        and constants["X_STRIDE_C"] == 1
        and constants["W_STRIDE_C"] == 1
    ):
        for block_k in (32, 128):
            configurations.append(
                {
                    "META": {
                        "BLOCK_OC": 16,
                        "BLOCK_HW": 32,
                        "BLOCK_K": block_k,
                        "GROUP_M": 1,
                    },
                    "num_warps": 4,
                    "num_stages": 3,
                }
            )
    if (
        function_name == "matmul_batched_contiguous_kernel"
        and int(constants["K"]) >= 256
    ):
        for block_m, block_n in ((32, 32), (64, 32), (32, 64), (64, 64)):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": 128,
                        "GROUP_M": 8,
                    },
                    "num_warps": 4,
                    "num_stages": 3,
                }
            )
    key_value = _require_integer(parameters, tuning.key)
    variants: list[
        tuple[
            dict[str, Any],
            dict[str, int | float | str | bool],
            dict[str, Any],
            tuple[int, int, int],
        ]
    ] = []
    seen_configurations: set[str] = set()
    for configuration in configurations:
        raw_meta, compile_options = _split_tuning_configuration(configuration)
        if function_name in {
            "_conv_wgrad2d_1x1_reduce_kernel",
            "_conv_wgrad2d_reduce_kernel",
            "_conv_wgrad2d_col_reduce_kernel",
        }:
            compile_options = {**compile_options, "num_stages": 1}
        elif function_name == "_conv_wgrad2d_stride2_row4_split_kernel":
            compile_options = {**compile_options, "num_stages": 2}
        variant_meta = _translated_tuning_meta(function_name, raw_meta)
        if function_name in {"layer_norm_kernel", "rms_norm_kernel"}:
            variant_meta["BLOCK_SIZE"] = min(
                variant_meta["BLOCK_SIZE"],
                1 << (int(constants["N"]) - 1).bit_length(),
            )
        if function_name.startswith("_sdpa_bwd_"):
            for key, extent_name in (
                ("BLOCK_D_OUT", "HEAD_DIM"),
                ("BLOCK_DV_OUT", "V_DIM"),
            ):
                if key in variant_meta and extent_name in constants:
                    variant_meta[key] = min(
                        variant_meta[key],
                        max(
                            16 if parameters.get("_sdpa_bwd_fp32") else 64,
                            1
                            << (int(constants[extent_name]) - 1).bit_length(),
                        ),
                    )

        if function_name == "activation_backward_contiguous_kernel":
            variant_meta["MASK_TAIL"] = (
                key_value
                % _positive_tuning_integer(variant_meta, "BLOCK_SIZE")
                != 0
            )
        unknown_meta = set(variant_meta).difference(constants)
        if unknown_meta:
            names = ", ".join(sorted(unknown_meta))
            raise ValueError(f"tuning META is not a kernel constexpr: {names}")

        normalized_configuration = {
            "META": variant_meta,
            **compile_options,
        }
        canonical = _canonical_tuning_value(
            normalized_configuration,
            "prepared tuning configuration",
        )
        if canonical in seen_configurations:
            continue
        seen_configurations.add(canonical)

        variant_constants = dict(constants)
        variant_constants.update(variant_meta)
        variant_grid = _autotune_variant_grid(
            strategy=tuning.strategy,
            key=tuning.key,
            function_name=function_name,
            parameters=parameters,
            constants=variant_constants,
            meta=variant_meta,
            default_grid=default_grid,
        )
        variants.append(
            (
                normalized_configuration,
                variant_constants,
                compile_options,
                variant_grid,
            )
        )

    if len(variants) < 1 or len(variants) > 1024:
        raise ValueError(
            "kernel tuning plan must contain between 1 and 1024 "
            f"valid candidates; got {len(variants)}"
        )

    identity_payload = {
        "schema_version": 3,
        "source_sha256": tuning_source_hash,
        "kernel_ownership": candidate.ownership,
        "kernel_backend": candidate.backend,
        "kernel_provider": candidate.provider,
        "operation": operation,
        "function": function_name,
        "table": selected_table,
        "key": tuning.key,
        "key_value": key_value,
        "strategy": tuning.strategy,
        "constants": constants,
        "default_grid": list(default_grid),
        "configurations": [variant[0] for variant in variants],
    }
    canonical_identity = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        variants,
        tuning_source_hash,
        hashlib.sha256(canonical_identity).hexdigest(),
        selected_table,
    )
