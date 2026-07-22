# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import torch

from flag_dnn import runtime
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
from flag_dnn.utils.device_info import get_device_capability_for

# SDPA backward prepared paths


def _mloop_supported_on_device(
    inputs: Sequence[Any],
) -> bool:
    """Reject m-loop kernels on SM90 before they are launched."""
    q = inputs[0] if inputs else None
    if not isinstance(q, torch.Tensor):
        return False
    if runtime.device.vendor_name != "nvidia":
        return True
    capability = get_device_capability_for(q.device)
    return capability not in ((0, 0), (9, 0))


def _is_owner_compute_causal_d128(
    q_shape,
    k_shape,
    v_shape,
    o_shape,
    do_shape,
    stats_shape,
    q_stride,
    k_stride,
    v_stride,
    o_stride,
    do_stride,
    stats_stride,
    out_dtype,
    *,
    causal_top_left,
):
    if not causal_top_left or out_dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return False
    exact_shapes = {
        (
            (2, 16, 2048, 128),
            (2, 16, 2048, 128),
        ),
    }
    if (q_shape, k_shape) not in exact_shapes:
        return False
    if v_shape != k_shape or o_shape != q_shape or do_shape != q_shape:
        return False
    if stats_shape != (*q_shape[:3], 1):
        return False

    def bhsd_stride(shape):
        _, heads, seq, dim = shape
        return (heads * seq * dim, seq * dim, dim, 1)

    return (
        q_stride == bhsd_stride(q_shape)
        and k_stride == bhsd_stride(k_shape)
        and v_stride == bhsd_stride(v_shape)
        and o_stride == bhsd_stride(o_shape)
        and do_stride == bhsd_stride(do_shape)
        and stats_stride == (q_shape[1] * q_shape[2], q_shape[2], 1, 1)
    )


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
        _sdpa_bwd_delta_kernel,
        _sdpa_bwd_decode_dkdv_dq_atomic_kernel,
        _sdpa_bwd_dense_mloop_kernel,
        _sdpa_bwd_gqa_dkdv_atomic_causal_d128_kernel,
        _sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel,
        _sdpa_bwd_mloop_causal_d128_kernel,
        _sdpa_bwd_owner_causal_d128_kernel,
        _single_tuned_config_kwargs,
        _tuned_config_supported_on_device,
        _zero_two_contiguous_kernel,
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
    if head_dim > 128:
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
    owner_kind: Optional[str] = None
    if _is_owner_compute_causal_d128(
        q_shape,
        k_shape,
        v_shape,
        o_shape,
        do_shape,
        stats_shape,
        q_stride,
        k_stride,
        v_stride,
        o_stride,
        do_stride,
        stats_stride,
        out_dtype,
        causal_top_left=causal_top_left,
    ):
        owner_kind = "mha"
    if banded and not causal_top_left:
        return None
    use_fused_gqa = (
        causal_top_left
        and q_per_k == q_per_v
        and q_per_k > 1
        and sq <= 4096
        and skv <= 4096
    )
    if (q_per_k != 1 or q_per_v != 1) and not use_fused_gqa:
        return None
    small_supported = sq <= 1024 and skv <= 1024
    long_causal_supported = causal_top_left and sq <= 4096 and skv <= 4096
    dense_d32_supported = (
        not causal_top_left and head_dim <= 32 and sq <= 2048 and skv <= 2048
    )
    decode_supported = not causal_top_left and sq == 1 and skv <= 2048
    if not (
        small_supported
        or long_causal_supported
        or dense_d32_supported
        or decode_supported
    ):
        return None

    attn_scale = attrs.get("attn_scale")
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    attn_scale = float(attn_scale)

    config_device = input_specs[0].device
    config_name = "sdpa_backward_fused_atomic"
    if use_fused_gqa:
        config_name = "sdpa_backward_fused_atomic_gqa_causal_d128"
    elif causal_top_left:
        config_name = (
            "sdpa_backward_fused_atomic_causal_d128"
            if head_dim > 64
            else "sdpa_backward_fused_atomic_causal"
        )
    elif head_dim <= 32:
        config_name = "sdpa_backward_fused_atomic_d32"
    elif head_dim > 64:
        config_name = "sdpa_backward_fused_atomic_d128"
    fused_config = _single_tuned_config_kwargs(
        config_name, device=config_device
    )
    mloop_config = _single_tuned_config_kwargs(
        "sdpa_backward_mloop_causal_d128"
    )
    gqa_dq_config = _single_tuned_config_kwargs(
        "sdpa_backward_gqa_dq_delta_d128",
        device=config_device,
    )
    decode_config = _single_tuned_config_kwargs("sdpa_backward_decode_d128")
    owner_config = None
    if owner_kind is not None:
        owner_config = _single_tuned_config_kwargs(
            "sdpa_backward_owner_mha_causal_d128",
            device=config_device,
        )
    dense_mloop_config = _single_tuned_config_kwargs(
        "sdpa_backward_dense_mloop_d128"
        if head_dim > 64
        else "sdpa_backward_dense_mloop_d64"
    )

    def config_guard(*named_configs):
        def supported(inputs: Sequence[Any]) -> bool:
            q = inputs[0] if inputs else None
            if not isinstance(q, torch.Tensor):
                return False
            return all(
                _tuned_config_supported_on_device(name, config, q.device)
                for name, config in named_configs
            )

        return supported

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
    use_mloop_causal = exact_causal and head_dim == 128 and sq >= 2048
    use_dense_mloop = (
        not causal_top_left
        and q_per_k == 1
        and q_per_v == 1
        and head_dim in (64, 128)
        and head_dim == v_dim
        and head_dim == int(dense_mloop_config["BLOCK_D"])
        and sq == skv
        and sq <= 512
        and sq % int(dense_mloop_config["BLOCK_M"]) == 0
        and skv % int(dense_mloop_config["BLOCK_N"]) == 0
    )
    use_decode_d128 = (
        decode_supported
        and not causal_top_left
        and q_per_k == 1
        and q_per_v == 1
        and head_dim == 128
        and v_dim == 128
        and skv <= 2048
    )
    use_gqa_split_causal = (
        use_fused_gqa
        and causal_top_left
        and head_dim == 128
        and v_dim == 128
        and sq == skv
        and sq >= 4096
        and block_m == 64
        and block_n == 64
        and sq % block_n == 0
    )
    if triangular_causal:
        fused_grid = (num_m_blocks * (num_m_blocks + 1) // 2, batch * heads, 1)
    else:
        fused_grid = (num_m_blocks, num_n_blocks, batch * heads)
    delta_stride = (heads * sq, sq, 1)
    delta_config = None
    zero_delta_grid: tuple[int, int, int] = zero_grid
    if use_delta_fused or use_exact_delta or use_gqa_split_causal:
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
        if (
            owner_kind is not None
            or use_delta_fused
            or use_exact_delta
            or use_gqa_split_causal
        ):
            delta = torch.empty(
                (batch, heads, sq), dtype=torch.float32, device=q.device
            )
        return dQ, dK, dV, delta

    def bwd_result(context: Any) -> tuple[Any, Any, Any]:
        return context[0], context[1], context[2]

    if owner_kind is not None:
        assert owner_config is not None
        num_n_blocks = triton.cdiv(skv, owner_config["BLOCK_N_DKDV"])
        num_m_blocks = triton.cdiv(sq, owner_config["BLOCK_M_DQ"])
        owner_grid = (
            num_n_blocks + q_per_k * num_m_blocks,
            batch,
            kv_heads,
        )
        delta_grid = (
            triton.cdiv(sq, 64),
            batch * heads,
            1,
        )
        delta_tail = (
            heads,
            sq,
            *o_stride,
            *do_stride,
            *delta_stride,
            128,
            64,
            128,
        )
        owner_tail = (
            attn_scale,
            heads,
            q_per_k,
            sq,
            *q_stride,
            *k_stride,
            *v_stride,
            *do_stride,
            stats_stride[0],
            stats_stride[1],
            stats_stride[2],
            *delta_stride,
            *q_stride,
            *k_stride,
            *v_stride,
            num_n_blocks,
            num_m_blocks,
            owner_config["BLOCK_M_DKDV"],
            owner_config["BLOCK_N_DKDV"],
            owner_config["BLOCK_M_DQ"],
            owner_config["BLOCK_N_DQ"],
            owner_config["BLOCK_D"],
        )

        def owner_delta_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            delta = context[3]
            assert isinstance(delta, torch.Tensor)
            return inputs[3], inputs[4], delta

        def owner_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            delta = context[3]
            assert isinstance(delta, torch.Tensor)
            return (
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[4],
                inputs[5],
                delta,
                context[0],
                context[1],
                context[2],
            )

        delta_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_delta_kernel,
            grid=delta_grid,
            runtime_args=owner_delta_runtime_args,
            static_args=delta_tail[:-3],
            constexpr_kwargs={
                "V_DIM": 128,
                "BLOCK_M": 64,
                "BLOCK_DV": 128,
                "num_warps": 4,
                "num_stages": 2,
            },
            build_cached_call=make_static_cached_call(delta_grid, delta_tail),
        )
        owner_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_owner_causal_d128_kernel,
            grid=owner_grid,
            runtime_args=owner_runtime_args,
            static_args=owner_tail[:-5],
            constexpr_kwargs=owner_config,
            build_cached_call=make_static_cached_call(owner_grid, owner_tail),
        )
        return make_kernel_pipeline_run_fn(
            PreparedKernelPipelineSpec(
                steps=(delta_step, owner_step),
                input_checks=bwd_input_checks,
                context_factory=make_bwd_context,
                result=bwd_result,
                extra_check=config_guard(
                    (
                        "sdpa_backward_owner_mha_causal_d128",
                        owner_config,
                    )
                ),
            ),
            default_run_fn,
        )

    zero_step_grid: tuple[int, int, int]
    zero_static_args: tuple[Any, ...]
    zero_cached_args: tuple[Any, ...]

    if use_decode_d128:
        zero_step_grid = (triton.cdiv(q_numel, zero_block), 1, 1)
        zero_static_args = (q_numel, 0, 0)
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
    elif use_gqa_split_causal:
        assert delta_config is not None
        gqa_zero_block = int(delta_config["BLOCK_ZERO"])
        zero_step_grid = (
            triton.cdiv(max(k_numel, v_numel), gqa_zero_block),
            1,
            1,
        )
        zero_static_args = (k_numel, v_numel)
        zero_cached_args = zero_static_args + (gqa_zero_block,)

        def zero_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            return context[1], context[2]

        zero_step = PreparedPipelineStepSpec(
            kernel=_zero_two_contiguous_kernel,
            grid=zero_step_grid,
            runtime_args=zero_runtime_args,
            static_args=zero_static_args,
            constexpr_kwargs={"BLOCK": gqa_zero_block},
            build_cached_call=make_static_cached_call(
                zero_step_grid, zero_cached_args
            ),
        )
    elif use_mloop_causal or use_dense_mloop:
        zero_step_grid = (triton.cdiv(max(k_numel, v_numel), zero_block), 1, 1)
        zero_static_args = (k_numel, v_numel)
        zero_cached_args = zero_static_args + (zero_block,)

        def zero_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            return context[1], context[2]

        zero_step = PreparedPipelineStepSpec(
            kernel=_zero_two_contiguous_kernel,
            grid=zero_step_grid,
            runtime_args=zero_runtime_args,
            static_args=zero_static_args,
            constexpr_kwargs={"BLOCK": zero_block},
            build_cached_call=make_static_cached_call(
                zero_step_grid, zero_cached_args
            ),
        )
    elif use_exact_delta:
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

    if use_decode_d128:
        decode_grid = (
            triton.cdiv(skv, int(decode_config["BLOCK_N"])),
            batch * heads,
            1,
        )
        decode_tail = (
            attn_scale,
            heads,
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
            decode_config["BLOCK_N"],
            decode_config["BLOCK_D"],
        )
        decode_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_decode_dkdv_dq_atomic_kernel,
            grid=decode_grid,
            runtime_args=fused_runtime_args,
            static_args=decode_tail[:-2],
            constexpr_kwargs=decode_config,
            build_cached_call=make_static_cached_call(
                decode_grid, decode_tail
            ),
        )
        return make_kernel_pipeline_run_fn(
            PreparedKernelPipelineSpec(
                steps=(zero_step, decode_step),
                input_checks=bwd_input_checks,
                context_factory=make_bwd_context,
                result=bwd_result,
            ),
            default_run_fn,
        )

    if use_dense_mloop:
        dense_mloop_grid = (
            triton.cdiv(sq, int(dense_mloop_config["BLOCK_M"])),
            batch * heads,
            1,
        )
        dense_mloop_tail = (
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
            dense_mloop_config["BLOCK_M"],
            dense_mloop_config["BLOCK_N"],
            dense_mloop_config["BLOCK_D"],
        )
        dense_mloop_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_dense_mloop_kernel,
            grid=dense_mloop_grid,
            runtime_args=fused_runtime_args,
            static_args=dense_mloop_tail[:-3],
            constexpr_kwargs=dense_mloop_config,
            build_cached_call=make_static_cached_call(
                dense_mloop_grid, dense_mloop_tail
            ),
        )
        return make_kernel_pipeline_run_fn(
            PreparedKernelPipelineSpec(
                steps=(zero_step, dense_mloop_step),
                input_checks=bwd_input_checks,
                context_factory=make_bwd_context,
                result=bwd_result,
                extra_check=_mloop_supported_on_device,
            ),
            default_run_fn,
        )

    if use_mloop_causal:
        mloop_grid = (
            triton.cdiv(sq, int(mloop_config["BLOCK_M"])),
            batch * heads,
            1,
        )
        mloop_tail = (
            attn_scale,
            heads,
            q_per_k,
            sq,
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
            mloop_config["BLOCK_M"],
            mloop_config["BLOCK_N"],
            mloop_config["BLOCK_D"],
        )
        mloop_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_mloop_causal_d128_kernel,
            grid=mloop_grid,
            runtime_args=fused_runtime_args,
            static_args=mloop_tail[:-3],
            constexpr_kwargs=mloop_config,
            build_cached_call=make_static_cached_call(mloop_grid, mloop_tail),
        )
        return make_kernel_pipeline_run_fn(
            PreparedKernelPipelineSpec(
                steps=(zero_step, mloop_step),
                input_checks=bwd_input_checks,
                context_factory=make_bwd_context,
                result=bwd_result,
                extra_check=_mloop_supported_on_device,
            ),
            default_run_fn,
        )

    if use_gqa_split_causal:
        gqa_dq_block_m = int(gqa_dq_config["BLOCK_M"])
        gqa_dq_block_n = int(gqa_dq_config["BLOCK_N"])
        gqa_dq_block_d = int(gqa_dq_config["BLOCK_D"])
        if (
            gqa_dq_block_d != block_d
            or sq % gqa_dq_block_m != 0
            or skv % gqa_dq_block_n != 0
        ):
            return None
        gqa_split_dq_grid = (
            triton.cdiv(sq, gqa_dq_block_m),
            batch * heads,
            1,
        )
        gqa_split_dkdv_grid = (num_n_blocks, batch * heads, 1)
        gqa_split_dq_tail = (
            attn_scale,
            heads,
            q_per_k,
            sq,
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
            gqa_dq_block_m,
            gqa_dq_block_n,
            gqa_dq_block_d,
        )
        gqa_split_dkdv_tail = (
            attn_scale,
            heads,
            q_per_k,
            sq,
            *q_stride,
            *k_stride,
            *v_stride,
            *do_stride,
            stats_stride[0],
            stats_stride[1],
            stats_stride[2],
            *delta_stride,
            *k_stride,
            *v_stride,
            block_m,
            block_n,
            block_d,
        )

        def gqa_split_dq_runtime_args(
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
            )

        def gqa_split_dkdv_runtime_args(
            inputs: Sequence[Any], context: Any
        ) -> tuple[Any, ...]:
            delta = context[3]
            assert isinstance(delta, torch.Tensor)
            return (
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[4],
                inputs[5],
                delta,
                context[1],
                context[2],
            )

        gqa_split_dq_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel,
            grid=gqa_split_dq_grid,
            runtime_args=gqa_split_dq_runtime_args,
            static_args=gqa_split_dq_tail[:-3],
            constexpr_kwargs=gqa_dq_config,
            build_cached_call=make_static_cached_call(
                gqa_split_dq_grid, gqa_split_dq_tail
            ),
        )
        gqa_split_dkdv_step = PreparedPipelineStepSpec(
            kernel=_sdpa_bwd_gqa_dkdv_atomic_causal_d128_kernel,
            grid=gqa_split_dkdv_grid,
            runtime_args=gqa_split_dkdv_runtime_args,
            static_args=gqa_split_dkdv_tail[:-3],
            constexpr_kwargs=fused_config,
            build_cached_call=make_static_cached_call(
                gqa_split_dkdv_grid, gqa_split_dkdv_tail
            ),
        )
        return make_kernel_pipeline_run_fn(
            PreparedKernelPipelineSpec(
                steps=(zero_step, gqa_split_dq_step, gqa_split_dkdv_step),
                input_checks=bwd_input_checks,
                context_factory=make_bwd_context,
                result=bwd_result,
                extra_check=config_guard(
                    (
                        "sdpa_backward_fused_atomic_gqa_causal_d128",
                        fused_config,
                    ),
                    (
                        "sdpa_backward_gqa_dq_delta_d128",
                        gqa_dq_config,
                    ),
                ),
            ),
            default_run_fn,
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
            extra_check=(
                config_guard(
                    (
                        "sdpa_backward_fused_atomic_gqa_causal_d128",
                        fused_config,
                    )
                )
                if use_fused_gqa
                else None
            ),
        ),
        default_run_fn,
    )
