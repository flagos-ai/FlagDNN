"""MThreads dispatch attention implementation."""

from __future__ import annotations

import math
from typing import Any

from .common import (
    _ATTENTION_SOURCE_RELATIVE_PATH,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _float32_token,
    _pointer_token,
    _scalar_f32_bits,
    _scalar_i32_bits,
)
from .graph import ParsedAttentionRequest
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import TensorSpec, is_row_major_contiguous
from .tuning import _load_tuning

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
        (
            runtime_tokens[name]
            if name in runtime_tokens
            else _attention_constexpr_token(constants[name])
        )
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
        RuntimeArgument("tensor", semantic_name or port, tensor.uid, None),
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
            RuntimeArgument("scalar_i32", name, None, _scalar_i32_bits(value)),
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
        else (f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}")
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
        "BLOCK_D": _attention_full_dimension_block(request.head_dimension),
        "BLOCK_DV": _attention_full_dimension_block(request.value_dimension),
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
        return _attention_workspace_pointer("stats_ptr", "stats_placeholder")
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
            _attention_tensor_pointer(request, "amax_s_ptr", "amax_s"),
            _attention_tensor_pointer(request, "amax_o_ptr", "amax_o"),
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
        _attention_tensor_pointer(request, "descale_q_ptr", "descale_q"),
        _attention_tensor_pointer(request, "descale_k_ptr", "descale_k"),
        _attention_tensor_pointer(request, "descale_v_ptr", "descale_v"),
        _attention_tensor_pointer(request, "descale_s_ptr", "descale_s"),
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
        _attention_scalar_argument("attn_scale", "fp32", request.attn_scale),
        _attention_scalar_argument("SQ", "i32", request.sequence_q),
        _attention_scalar_argument("SKV", "i32", request.sequence_kv),
        _attention_scalar_argument("min_diag", "i32", request.min_diag),
        _attention_scalar_argument("max_diag", "i32", request.max_diag),
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
    constants = _attention_backward_base_constants(request, block_size)
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
            "DBIAS_BATCHES": (dbias.dimensions[0] if dbias is not None else 1),
            "DBIAS_HEADS": (dbias.dimensions[1] if dbias is not None else 1),
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
    constants = _attention_backward_base_constants(request, block_size)
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
    constants = _attention_backward_base_constants(request, block_size)
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
    constants = _attention_backward_base_constants(request, block_size)
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
            pointers=(_attention_tensor_pointer(request, "ptr", "dbias"),),
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
                    (request.sequence_kv + block_size - 1) // block_size,
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
                    (request.sequence_kv + block_size - 1) // block_size,
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
                    (request.sequence_kv + block_size - 1) // block_size,
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
    raw_delta_size = 4 * request.batch * request.heads * request.sequence_q
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
        "BLOCK_D": _attention_full_dimension_block(request.head_dimension),
        "BANDED": request.banded,
        "FULL_BLOCKS": False,
        "CAUSAL_TOP_LEFT": request.causal_top_left,
    }


def _attention_fp8_dq_constants(
    request: ParsedAttentionRequest,
    block_size: int,
) -> dict[str, int | float | bool]:
    constants = _attention_fp8_backward_base_constants(request, block_size)
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
        _attention_scalar_argument("attn_scale", "fp32", request.attn_scale),
        _attention_scalar_argument("HQ", "i32", request.heads),
        _attention_scalar_argument("SQ", "i32", request.sequence_q),
        _attention_scalar_argument("SKV", "i32", request.sequence_kv),
        _attention_scalar_argument("min_diag", "i32", request.min_diag),
        _attention_scalar_argument("max_diag", "i32", request.max_diag),
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
    constants = _attention_fp8_backward_base_constants(request, block_size)
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
