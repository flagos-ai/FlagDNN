from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import _static_shape
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

# fp8 SDPA forward prepared path: bind every scalar kernel argument at plan
# time so graph replay only allocates outputs (incl. the zero-initialised amax
# accumulators) and launches the cached compiled kernel.


@register_prepared_run_fn("sdpa_fp8")
def _prepare_sdpa_fp8(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    import math

    import triton

    from flag_dnn.ops.sdpa import (
        _BOTTOM_RIGHT,
        _LOG2E,
        _TOP_LEFT,
        _UNBOUNDED_DIAG,
        _ensure_triton_tma_allocator,
    )
    from flag_dnn.ops.sdpa_fp8 import (
        _sdpa_fp8_fwd_fast_kernel,
        _sdpa_fp8_fwd_gqa_causal_tma_kernel,
        _sdpa_fp8_fwd_kernel,
        _sdpa_fp8_fwd_tma_kernel,
    )

    has_bias = bool(attrs.get("has_bias"))
    generate_stats = bool(attrs.get("generate_stats"))
    expected_inputs = 4 if has_bias else 3
    if len(input_specs) != expected_inputs:
        return None

    q_spec = input_specs[0]
    shapes = [_static_shape(spec) for spec in input_specs]
    if any(shape is None for shape in shapes):
        return None
    strides = [
        None if spec.stride is None else tuple(spec.stride)
        for spec in input_specs
    ]
    if any(stride is None for stride in strides):
        return None
    sdpa_input_checks = runtime_tensor_checks_from_specs(
        input_specs, tuple(range(expected_inputs))
    )
    if sdpa_input_checks is None:
        return None

    q_shape, k_shape, v_shape = shapes[0], shapes[1], shapes[2]
    q_stride, k_stride, v_stride = strides[0], strides[1], strides[2]
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return None
    batch, heads, sq, head_dim = q_shape
    skv = k_shape[2]
    v_dim = v_shape[3]
    if 0 in q_shape or 0 in k_shape or 0 in v_shape:
        return None
    if heads % k_shape[1] != 0 or heads % v_shape[1] != 0:
        return None

    out_dtype = torch_dtype(q_spec.dtype)
    out_shape = (batch, heads, sq, v_dim)
    o_stride = (heads * sq * v_dim, sq * v_dim, v_dim, 1)
    stats_shape = (batch, heads, sq, 1)
    stats_stride = (heads * sq, sq, 1)

    if has_bias:
        bias_shape, bias_stride_full = shapes[3], strides[3]
        if len(bias_shape) != 4:
            return None
        bias_stride = (
            bias_stride_full[0] if bias_shape[0] != 1 else 0,
            bias_stride_full[1] if bias_shape[1] != 1 else 0,
            bias_stride_full[2],
            bias_stride_full[3],
        )
    else:
        bias_stride = (0, 0, 0, 0)

    descale_q = float(attrs.get("descale_q"))
    descale_k = float(attrs.get("descale_k"))
    descale_v = float(attrs.get("descale_v"))
    descale_s = float(attrs.get("descale_s"))
    scale_s = float(attrs.get("scale_s"))
    scale_o = float(attrs.get("scale_o"))

    attn_scale = attrs.get("attn_scale")
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    alignment = attrs.get("diagonal_alignment")
    left = attrs.get("diagonal_band_left_bound")
    right = attrs.get("diagonal_band_right_bound")
    shift = skv - sq if alignment == _BOTTOM_RIGHT else 0
    min_diag = 1 - left + shift if left is not None else -_UNBOUNDED_DIAG
    max_diag = right + shift if right is not None else _UNBOUNDED_DIAG
    banded = left is not None or right is not None
    reverse_causal = alignment == _TOP_LEFT and left is None and right == 0

    qk_scale = attn_scale * descale_q * descale_k * _LOG2E
    sv_descale = descale_s * descale_v

    pure_causal = (
        banded and left is None and right == 0 and alignment == _TOP_LEFT
    )
    head_pow2 = head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    v_pow2 = v_dim >= 16 and (v_dim & (v_dim - 1)) == 0
    fast_ok = (
        not has_bias
        and head_dim == v_dim
        and head_pow2
        and v_pow2
        and (not banded or pure_causal)
    )
    if fast_ok:
        fast_stats_stride = stats_stride if generate_stats else (0, 0, 0)
        hkv = int(k_shape[1])
        hv = int(v_shape[1])
        q_per_k = heads // hkv
        q_per_v = heads // hv
        tma_amortizes = (
            skv >= 2048
            or (skv >= 1024 and not pure_causal)
            or (skv >= 1024 and pure_causal and generate_stats)
            or (
                sq == skv
                and not generate_stats
                and head_dim == 128
                and v_dim == 128
                and (
                    (skv == 512 and pure_causal)
                    or (skv in (256, 512) and not banded)
                )
            )
        )
        tma_ok = (
            tma_amortizes
            and q_stride[3] == 1
            and k_stride[3] == 1
            and v_stride[3] == 1
            and head_dim % 16 == 0
            and v_dim % 16 == 0
        )
        gqa_causal_tma_ok = (
            tma_ok
            and pure_causal
            and generate_stats
            and sq == skv
            and skv >= 1024
            and hkv == hv
            and heads > hkv
            and q_per_k <= 8
            and head_dim == 128
            and v_dim == 128
        )

        def make_fp8_fast_output(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, ...]:
            q = inputs[0]
            assert isinstance(q, torch.Tensor)
            o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
            if generate_stats:
                stats = torch.empty(
                    stats_shape, dtype=torch.float32, device=q.device
                )
            else:
                stats = o
            amax_s = torch.zeros(
                (1, 1, 1, 1), dtype=torch.float32, device=q.device
            )
            amax_o = torch.zeros(
                (1, 1, 1, 1), dtype=torch.float32, device=q.device
            )
            return o, stats, amax_s, amax_o

        def fp8_fast_runtime_args(
            inputs: Sequence[Any], output: Any
        ) -> tuple[Any, ...]:
            return (
                inputs[0],
                inputs[1],
                inputs[2],
                output[0],
                output[1],
                output[2],
                output[3],
            )

        def fp8_fast_result(output: Any) -> Any:
            if generate_stats:
                return output[0], output[1], output[2], output[3]
            return output[0], output[2], output[3]

        def fast_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return (triton.cdiv(sq, meta["BLOCK_M"]), batch * heads)

        fast_tail = (
            qk_scale,
            scale_s,
            sv_descale,
            scale_o,
            heads,
            sq,
            skv,
            q_per_k,
            q_per_v,
            *q_stride,
            *k_stride,
            *v_stride,
            *o_stride,
            *fast_stats_stride,
        )

        def build_fast_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            block_n = int(constexprs["BLOCK_N"])
            static_grid = (triton.cdiv(sq, block_m), batch * heads, 1)
            cached_args = fast_tail + (
                head_dim,
                v_dim,
                block_m,
                block_n,
                head_dim,
                v_dim,
                pure_causal,
                generate_stats,
            )
            return static_grid, cached_args

        if gqa_causal_tma_ok:
            gqa_tail = (
                qk_scale,
                scale_s,
                sv_descale,
                scale_o,
                hkv,
                sq,
                skv,
                q_per_k,
                *q_stride,
                *k_stride,
                *v_stride,
                *o_stride,
                *fast_stats_stride,
            )

            def gqa_tma_grid(meta: dict[str, Any]) -> tuple[int, int, int]:
                return (
                    triton.cdiv(sq, meta["BLOCK_M"]),
                    batch * hkv,
                    triton.cdiv(q_per_k, meta["BLOCK_H"]),
                )

            def build_gqa_tma_cached_call(
                constexprs: dict[str, Any],
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                block_m = int(constexprs["BLOCK_M"])
                block_h = int(constexprs["BLOCK_H"])
                block_n = int(constexprs["BLOCK_N"])
                static_grid = (
                    triton.cdiv(sq, block_m),
                    batch * hkv,
                    triton.cdiv(q_per_k, block_h),
                )
                cached_args = gqa_tail + (
                    head_dim,
                    v_dim,
                    block_m,
                    block_h,
                    block_n,
                    head_dim,
                    v_dim,
                    generate_stats,
                )
                return static_grid, cached_args

            return make_single_kernel_run_fn(
                PreparedSingleKernelRunSpec(
                    kernel=PreparedSingleKernelSpec(
                        kernel=_sdpa_fp8_fwd_gqa_causal_tma_kernel,
                        grid=gqa_tma_grid,
                        static_args=gqa_tail,
                        constexpr_kwargs=dict(
                            HEAD_DIM=head_dim,
                            V_DIM=v_dim,
                            BLOCK_D=head_dim,
                            BLOCK_DV=v_dim,
                            GENERATE_STATS=generate_stats,
                        ),
                        build_cached_call=build_gqa_tma_cached_call,
                    ),
                    input_checks=sdpa_input_checks,
                    output_factory=make_fp8_fast_output,
                    runtime_args=fp8_fast_runtime_args,
                    result=fp8_fast_result,
                    pre_launch=_ensure_triton_tma_allocator,
                ),
                default_run_fn,
            )

        fast_kernel = (
            _sdpa_fp8_fwd_tma_kernel if tma_ok else _sdpa_fp8_fwd_fast_kernel
        )
        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=fast_kernel,
                    grid=fast_grid,
                    static_args=fast_tail,
                    constexpr_kwargs=dict(
                        HEAD_DIM=head_dim,
                        V_DIM=v_dim,
                        BLOCK_D=head_dim,
                        BLOCK_DV=v_dim,
                        CAUSAL=pure_causal,
                        GENERATE_STATS=generate_stats,
                    ),
                    build_cached_call=build_fast_cached_call,
                ),
                input_checks=sdpa_input_checks,
                output_factory=make_fp8_fast_output,
                runtime_args=fp8_fast_runtime_args,
                result=fp8_fast_result,
                pre_launch=(_ensure_triton_tma_allocator if tma_ok else None),
            ),
            default_run_fn,
        )

    if not generate_stats:
        stats_stride = (0, 0, 0)

    constexpr_kwargs = dict(
        HEAD_DIM=head_dim,
        V_DIM=v_dim,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        BLOCK_DV=max(16, triton.next_power_of_2(v_dim)),
        HAS_BIAS=has_bias,
        BANDED=banded,
        GENERATE_STATS=generate_stats,
        REVERSE_CAUSAL=reverse_causal,
    )
    scalar_tail = (
        qk_scale,
        scale_s,
        sv_descale,
        scale_o,
        heads,
        sq,
        skv,
        heads // k_shape[1],
        heads // v_shape[1],
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *bias_stride,
        *o_stride,
        *stats_stride,
    )
    batch_heads = batch * heads

    def make_fp8_output(inputs: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        q = inputs[0]
        assert isinstance(q, torch.Tensor)
        o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        if generate_stats:
            stats = torch.empty(
                stats_shape, dtype=torch.float32, device=q.device
            )
        else:
            stats = o
        amax_s = torch.zeros(
            (1, 1, 1, 1), dtype=torch.float32, device=q.device
        )
        amax_o = torch.zeros(
            (1, 1, 1, 1), dtype=torch.float32, device=q.device
        )
        return o, stats, amax_s, amax_o

    def fp8_runtime_args(
        inputs: Sequence[Any], output: Any
    ) -> tuple[Any, ...]:
        bias = inputs[3] if has_bias else inputs[0]
        return (
            inputs[0],
            inputs[1],
            inputs[2],
            bias,
            output[0],
            output[1],
            output[2],
            output[3],
        )

    def fp8_result(output: Any) -> Any:
        if generate_stats:
            return output[0], output[1], output[2], output[3]
        return output[0], output[2], output[3]

    def tune_grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(sq, meta["BLOCK_M"]), batch_heads)

    def build_fp8_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_n = int(constexprs["BLOCK_N"])
        static_grid = (triton.cdiv(sq, block_m), batch_heads, 1)
        cached_args = scalar_tail + (
            head_dim,
            v_dim,
            block_m,
            block_n,
            constexpr_kwargs["BLOCK_D"],
            constexpr_kwargs["BLOCK_DV"],
            has_bias,
            banded,
            generate_stats,
            reverse_causal,
        )
        return static_grid, cached_args

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_sdpa_fp8_fwd_kernel,
                grid=tune_grid,
                static_args=scalar_tail,
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_fp8_cached_call,
            ),
            input_checks=sdpa_input_checks,
            output_factory=make_fp8_output,
            runtime_args=fp8_runtime_args,
            result=fp8_result,
        ),
        default_run_fn,
    )
