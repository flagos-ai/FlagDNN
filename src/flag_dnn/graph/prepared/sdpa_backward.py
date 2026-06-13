from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import torch

from flag_dnn.graph.prepared import (
    PreparedKernelPipelineSpec,
    PreparedPipelineStepSpec,
    RunFn,
    make_kernel_pipeline_run_fn,
    make_static_cached_call,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import _static_shape
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

# SDPA backward prepared paths


@register_prepared_run_fn("sdpa_backward")
def _prepare_sdpa_backward(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    import math

    import triton

    from flag_dnn.ops.sdpa import _BOTTOM_RIGHT, _TOP_LEFT, _UNBOUNDED_DIAG
    from flag_dnn.ops.sdpa_backward import (
        _sdpa_bwd_fused_atomic_causal_kernel,
        _sdpa_bwd_fused_atomic_causal_delta_tri_kernel,
        _sdpa_bwd_fused_atomic_causal_exact_delta_kernel,
        _sdpa_bwd_fused_atomic_causal_exact_kernel,
        _sdpa_bwd_fused_atomic_causal_tri_kernel,
        _sdpa_bwd_fused_atomic_gqa_causal_kernel,
        _sdpa_bwd_fused_atomic_gqa_causal_tri_kernel,
        _sdpa_bwd_fused_atomic_kernel,
        _single_tuned_config_kwargs,
        _zero_three_and_delta_kernel,
        _zero_three_contiguous_kernel,
        _zero_three_equal_and_delta_kernel,
    )

    if attrs.get("has_bias") or attrs.get("has_dbias"):
        return None
    if len(input_specs) != 6:
        return None

    shapes = [_static_shape(spec) for spec in input_specs]
    if any(shape is None for shape in shapes):
        return None
    strides = [
        None if spec.stride is None else tuple(spec.stride)
        for spec in input_specs
    ]
    if any(stride is None for stride in strides):
        return None
    bwd_input_checks = runtime_tensor_checks_from_specs(
        input_specs, tuple(range(6))
    )
    if bwd_input_checks is None:
        return None

    q_shape, k_shape, v_shape, o_shape, do_shape, stats_shape = cast(
        list[tuple[int, ...]], shapes
    )
    q_stride, k_stride, v_stride, o_stride, do_stride, stats_stride = cast(
        list[tuple[int, ...]], strides
    )
    if (
        len(q_shape) != 4
        or len(k_shape) != 4
        or len(v_shape) != 4
        or len(o_shape) != 4
        or len(do_shape) != 4
        or len(stats_shape) != 4
    ):
        return None
    batch, heads, sq, head_dim = q_shape
    kv_heads = k_shape[1]
    v_heads = v_shape[1]
    skv = k_shape[2]
    v_dim = v_shape[3]
    if 0 in q_shape or 0 in k_shape or 0 in v_shape:
        return None
    if k_shape[0] != batch or v_shape[0] != batch:
        return None
    if o_shape != (batch, heads, sq, v_dim) or do_shape != o_shape:
        return None
    if stats_shape != (batch, heads, sq, 1):
        return None
    if head_dim != v_dim or heads % kv_heads != 0 or heads % v_heads != 0:
        return None
    q_per_k = heads // kv_heads
    q_per_v = heads // v_heads
    if head_dim > 128 or sq > 1024 or skv > 1024:
        return None

    out_dtype = torch_dtype(input_specs[0].dtype)
    if out_dtype not in (torch.float16, torch.bfloat16):
        return None

    alignment = attrs.get("diagonal_alignment")
    left = attrs.get("diagonal_band_left_bound")
    right = attrs.get("diagonal_band_right_bound")
    shift = skv - sq if alignment == _BOTTOM_RIGHT else 0
    min_diag = 1 - left + shift if left is not None else -_UNBOUNDED_DIAG
    max_diag = right + shift if right is not None else _UNBOUNDED_DIAG
    banded = left is not None or right is not None
    causal_top_left = (
        alignment == _TOP_LEFT and left is None and right == 0 and sq == skv
    )
    if banded and not causal_top_left:
        return None
    use_fused_gqa = (
        causal_top_left
        and q_per_k == q_per_v
        and q_per_k > 1
        and sq <= 512
        and skv <= 512
    )
    if (q_per_k != 1 or q_per_v != 1) and not use_fused_gqa:
        return None

    attn_scale = attrs.get("attn_scale")
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    attn_scale = float(attn_scale)

    config_name = "sdpa_backward_fused_atomic"
    if use_fused_gqa:
        config_name = "sdpa_backward_fused_atomic_gqa_causal_d128"
    elif causal_top_left:
        config_name = (
            "sdpa_backward_fused_atomic_causal_d128"
            if head_dim > 64
            else "sdpa_backward_fused_atomic_causal"
        )
    fused_config = _single_tuned_config_kwargs(config_name)
    block_m = int(fused_config["BLOCK_M"])
    block_n = int(fused_config["BLOCK_N"])
    block_d = int(fused_config["BLOCK_D"])
    zero_block = 1024
    q_numel = batch * heads * sq * head_dim
    k_numel = batch * kv_heads * skv * head_dim
    v_numel = batch * v_heads * skv * v_dim
    zero_grid = (triton.cdiv(max(q_numel, k_numel, v_numel), zero_block), 1, 1)
    num_m_blocks = triton.cdiv(sq, block_m)
    num_n_blocks = triton.cdiv(skv, block_n)
    exact_causal = (
        causal_top_left
        and not use_fused_gqa
        and q_per_k == 1
        and q_per_v == 1
        and head_dim == block_d
        and v_dim == block_d
        and sq % block_m == 0
        and skv % block_n == 0
    )
    triangular_causal = (
        causal_top_left
        and block_m == block_n
        and sq == skv
        and not exact_causal
    )
    use_exact_delta = (
        exact_causal
        and head_dim == 128
        and q_numel == k_numel
        and q_numel == v_numel
        and q_numel % zero_block == 0
    )
    use_delta_fused = (
        triangular_causal
        and not use_fused_gqa
        and q_per_k == 1
        and head_dim > 64
        and sq >= 1024
    )
    if triangular_causal:
        fused_grid = (num_m_blocks * (num_m_blocks + 1) // 2, batch * heads, 1)
    else:
        fused_grid = (num_m_blocks, num_n_blocks, batch * heads)
    delta_stride = (heads * sq, sq, 1)
    delta_config = None
    zero_delta_grid = zero_grid
    if use_delta_fused or use_exact_delta:
        delta_config = _single_tuned_config_kwargs("sdpa_backward_zero_delta")
        zero_delta_grid = (
            max(
                zero_grid[0],
                triton.cdiv(batch * heads * sq, int(delta_config["BLOCK_M"])),
            ),
            1,
            1,
        )

    dense_tail = (
        attn_scale,
        heads,
        sq,
        skv,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
    )
    gqa_tail = (
        attn_scale,
        heads,
        q_per_k,
        sq,
        skv,
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
        banded,
        causal_top_left,
    )
    gqa_tri_tail = (
        attn_scale,
        heads,
        q_per_k,
        sq,
        skv,
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
        num_m_blocks,
        banded,
        causal_top_left,
    )
    causal_tri_tail = (
        attn_scale,
        heads,
        sq,
        skv,
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
        num_m_blocks,
        banded,
        causal_top_left,
    )
    causal_delta_tri_tail = (
        attn_scale,
        heads,
        sq,
        skv,
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *delta_stride,
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
        num_m_blocks,
        banded,
        causal_top_left,
    )
    causal_exact_delta_tail = (
        attn_scale,
        heads,
        sq,
        skv,
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *delta_stride,
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
        banded,
        causal_top_left,
    )
    causal_tail = (
        attn_scale,
        heads,
        sq,
        skv,
        min_diag,
        max_diag,
        *q_stride,
        *k_stride,
        *v_stride,
        *o_stride,
        *do_stride,
        stats_stride[0],
        stats_stride[1],
        stats_stride[2],
        *q_stride,
        *k_stride,
        *v_stride,
        head_dim,
        block_m,
        block_n,
        block_d,
        banded,
        causal_top_left,
    )

    def make_bwd_context(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        q = inputs[0]
        assert isinstance(q, torch.Tensor)
        dQ = torch.empty_strided(
            q_shape, q_stride, dtype=out_dtype, device=q.device
        )
        dK = torch.empty_strided(
            k_shape, k_stride, dtype=out_dtype, device=q.device
        )
        dV = torch.empty_strided(
            v_shape, v_stride, dtype=out_dtype, device=q.device
        )
        delta = None
        if use_delta_fused or use_exact_delta:
            delta = torch.empty(
                (batch, heads, sq), dtype=torch.float32, device=q.device
            )
        return dQ, dK, dV, delta

    def bwd_result(context: Any) -> tuple[Any, Any, Any]:
        return context[0], context[1], context[2]

    if use_exact_delta:
        assert delta_config is not None
        zero_step_grid = zero_delta_grid
        zero_static_args = (
            batch * heads * sq,
            heads,
            sq,
            *o_stride,
            *do_stride,
            *delta_stride,
        )
        zero_cached_args = zero_static_args + (
            delta_config["BLOCK_ZERO"],
            delta_config["BLOCK_M"],
            delta_config["BLOCK_D"],
        )

        def zero_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            delta = context[3]
            assert isinstance(delta, torch.Tensor)
            return (
                context[0],
                context[1],
                context[2],
                inputs[3],
                inputs[4],
                delta,
            )

        zero_step = PreparedPipelineStepSpec(
            kernel=_zero_three_equal_and_delta_kernel,
            grid=zero_step_grid,
            runtime_args=zero_runtime_args,
            static_args=zero_static_args,
            constexpr_kwargs=delta_config,
            build_cached_call=make_static_cached_call(
                zero_step_grid, zero_cached_args
            ),
        )
    elif use_delta_fused:
        assert delta_config is not None
        zero_step_grid = zero_delta_grid
        zero_static_args = (
            q_numel,
            k_numel,
            v_numel,
            batch * heads * sq,
            heads,
            sq,
            *o_stride,
            *do_stride,
            *delta_stride,
        )
        zero_cached_args = zero_static_args + (
            delta_config["BLOCK_ZERO"],
            delta_config["BLOCK_M"],
            delta_config["BLOCK_D"],
        )

        def zero_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            delta = context[3]
            assert isinstance(delta, torch.Tensor)
            return (
                context[0],
                context[1],
                context[2],
                inputs[3],
                inputs[4],
                delta,
            )

        zero_step = PreparedPipelineStepSpec(
            kernel=_zero_three_and_delta_kernel,
            grid=zero_step_grid,
            runtime_args=zero_runtime_args,
            static_args=zero_static_args,
            constexpr_kwargs=delta_config,
            build_cached_call=make_static_cached_call(
                zero_step_grid, zero_cached_args
            ),
        )
    else:
        zero_step_grid = zero_grid
        zero_static_args = (q_numel, k_numel, v_numel)
        zero_cached_args = zero_static_args + (zero_block,)

        def zero_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            return context[0], context[1], context[2]

        zero_step = PreparedPipelineStepSpec(
            kernel=_zero_three_contiguous_kernel,
            grid=zero_step_grid,
            runtime_args=zero_runtime_args,
            static_args=zero_static_args,
            constexpr_kwargs={"BLOCK": zero_block},
            build_cached_call=make_static_cached_call(
                zero_step_grid, zero_cached_args
            ),
        )

    def fused_runtime_args(
        inputs: Sequence[Any], context: Any
    ) -> tuple[Any, ...]:
        return (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            context[0],
            context[1],
            context[2],
        )

    def fused_delta_runtime_args(
        inputs: Sequence[Any], context: Any
    ) -> tuple[Any, ...]:
        delta = context[3]
        assert isinstance(delta, torch.Tensor)
        return (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            delta,
            context[0],
            context[1],
            context[2],
        )

    if use_exact_delta:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_causal_exact_delta_kernel,
            grid=fused_grid,
            runtime_args=fused_delta_runtime_args,
            static_args=causal_exact_delta_tail[:-6],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(
                fused_grid, causal_exact_delta_tail
            ),
        )
    elif exact_causal:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_causal_exact_kernel,
            grid=fused_grid,
            runtime_args=fused_runtime_args,
            static_args=causal_tail[:-6],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(fused_grid, causal_tail),
        )
    elif use_delta_fused:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_causal_delta_tri_kernel,
            grid=fused_grid,
            runtime_args=fused_delta_runtime_args,
            static_args=causal_delta_tri_tail[:-7],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "NUM_BLOCKS": num_m_blocks,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(
                fused_grid, causal_delta_tri_tail
            ),
        )
    elif triangular_causal and use_fused_gqa:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_gqa_causal_tri_kernel,
            grid=fused_grid,
            runtime_args=fused_runtime_args,
            static_args=gqa_tri_tail[:-7],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "NUM_BLOCKS": num_m_blocks,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(
                fused_grid, gqa_tri_tail
            ),
        )
    elif triangular_causal:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_causal_tri_kernel,
            grid=fused_grid,
            runtime_args=fused_runtime_args,
            static_args=causal_tri_tail[:-7],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "NUM_BLOCKS": num_m_blocks,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(
                fused_grid, causal_tri_tail
            ),
        )
    elif causal_top_left and use_fused_gqa:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_gqa_causal_kernel,
            grid=fused_grid,
            runtime_args=fused_runtime_args,
            static_args=gqa_tail[:-6],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(fused_grid, gqa_tail),
        )
    elif causal_top_left:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_causal_kernel,
            grid=fused_grid,
            runtime_args=fused_runtime_args,
            static_args=causal_tail[:-6],
            constexpr_kwargs={
                "HEAD_DIM": head_dim,
                "BANDED": banded,
                "CAUSAL_TOP_LEFT": causal_top_left,
                **fused_config,
            },
            build_cached_call=make_static_cached_call(fused_grid, causal_tail),
        )
    else:
        fused_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_fused_atomic_kernel,
            grid=fused_grid,
            runtime_args=fused_runtime_args,
            static_args=dense_tail[:-4],
            constexpr_kwargs={"HEAD_DIM": head_dim, **fused_config},
            build_cached_call=make_static_cached_call(fused_grid, dense_tail),
        )

    return make_kernel_pipeline_run_fn(
        PreparedKernelPipelineSpec(
            steps=(zero_step, fused_step),
            input_checks=bwd_input_checks,
            context_factory=make_bwd_context,
            result=bwd_result,
        ),
        default_run_fn,
    )
