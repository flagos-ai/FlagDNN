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

from typing import Any, Optional, Sequence

import torch

from flag_dnn import runtime
from flag_dnn.graph.prepared import (
    PreparedKernelPipelineSpec,
    PreparedPipelineStepSpec,
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    make_kernel_pipeline_run_fn,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import _static_shape
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

# SDPA forward prepared paths


@register_prepared_run_fn("sdpa")
def _prepare_sdpa(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    backend_prepare = runtime.get_backend_hook("prepare_sdpa")
    if backend_prepare is not None:
        backend_run = backend_prepare(attrs, input_specs, default_run_fn)
        if backend_run is not None:
            return backend_run
    # Bind every scalar kernel argument at plan time so graph replay only
    # has to allocate outputs and launch the cached compiled kernel.
    import math

    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    from flag_dnn.ops.sdpa import (
        _BOTTOM_RIGHT,
        _LOG2E,
        _UNBOUNDED_DIAG,
        _sdpa_decode_combine_kernel,
        _ensure_triton_tma_allocator,
        _sdpa_decode_split_kernel,
        _sdpa_fwd_dense_exact_kernel,
        _sdpa_fwd_gqa_causal_desc_kernel,
        _sdpa_fwd_gqa_causal_kernel,
        _sdpa_fwd_mha_causal_desc_kernel,
        _sdpa_fwd_mha_causal_hostdesc_kernel,
        _sdpa_fwd_kernel,
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
    qkv_input_checks = runtime_tensor_checks_from_specs(input_specs, (0, 1, 2))
    if qkv_input_checks is None:
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
    reverse_causal = alignment != _BOTTOM_RIGHT and left is None and right == 0

    # The first replay goes through libentry/libtuner so the kernel is
    # autotuned per tune_configs.yaml; the chosen BLOCK_M/BLOCK_N are read
    # back to build the static argument list for direct CompiledKernel
    # dispatch on every later replay.
    constexpr_kwargs = dict(
        HEAD_DIM=head_dim,
        V_DIM=v_dim,
        ELEM_SIZE=out_dtype.itemsize,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        BLOCK_DV=max(16, triton.next_power_of_2(v_dim)),
        HAS_BIAS=has_bias,
        BANDED=banded,
        GENERATE_STATS=generate_stats,
        REVERSE_CAUSAL=reverse_causal,
    )
    scalar_tail = (
        attn_scale * _LOG2E,
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

    def make_stats_output(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = inputs[0]
        assert isinstance(q, torch.Tensor)
        o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        stats = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
        return o, stats

    def make_output_tensor(inputs: Sequence[Any]) -> torch.Tensor:
        q = inputs[0]
        assert isinstance(q, torch.Tensor)
        return torch.empty(out_shape, dtype=out_dtype, device=q.device)

    def qkv_stats_runtime_args(
        inputs: Sequence[Any], output: Any
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], inputs[2], output[0], output[1]

    def qkv_output_runtime_args(
        inputs: Sequence[Any], output: Any
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], inputs[2], output

    def make_sdpa_output(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = inputs[0]
        assert isinstance(q, torch.Tensor)
        o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        if generate_stats:
            stats = torch.empty(
                stats_shape, dtype=torch.float32, device=q.device
            )
        else:
            stats = o
        return o, stats

    def sdpa_runtime_args(
        inputs: Sequence[Any], output: Any
    ) -> tuple[Any, ...]:
        bias = inputs[3] if has_bias else inputs[0]
        return inputs[0], inputs[1], inputs[2], bias, output[0], output[1]

    def sdpa_result(output: Any) -> Any:
        if generate_stats:
            return output
        return output[0]

    use_decode = (
        sq == 1
        and not has_bias
        and not banded
        and out_dtype in (torch.float16, torch.bfloat16)
        and k_shape[1] == v_shape[1]
        and heads > k_shape[1]
    )
    if use_decode:
        hkv = k_shape[1]
        group = heads // hkv
        chunk = min(skv, 1024)
        splits = triton.cdiv(skv, chunk)
        block_g = max(1, triton.next_power_of_2(group))
        block_d = max(16, triton.next_power_of_2(head_dim))
        block_dv = max(16, triton.next_power_of_2(v_dim))
        part_last = v_dim + 2
        part_shape = (batch, heads, splits, part_last)
        part_stride = (
            heads * splits * part_last,
            splits * part_last,
            part_last,
            1,
        )
        split_grid = (splits, batch * hkv, 1)
        combine_grid = (batch_heads, 1, 1)
        split_constexpr = dict(
            GROUP=group,
            HEAD_DIM=head_dim,
            V_DIM=v_dim,
            ELEM_SIZE=out_dtype.itemsize,
            BLOCK_G=block_g,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
        )
        split_tail = (
            attn_scale * _LOG2E,
            hkv,
            skv,
            chunk,
            q_stride[0],
            q_stride[1],
            q_stride[3],
            *k_stride,
            *v_stride,
            *part_stride,
        )
        combine_tail = (
            heads,
            splits,
            *part_stride,
            o_stride[0],
            o_stride[1],
            o_stride[3],
            stats_stride[0],
            stats_stride[1],
            v_dim,
            max(1, triton.next_power_of_2(splits)),
            block_dv,
            generate_stats,
        )

        def make_decode_context(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = inputs[0]
            assert isinstance(q, torch.Tensor)
            o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
            if generate_stats:
                stats = torch.empty(
                    stats_shape, dtype=torch.float32, device=q.device
                )
            else:
                stats = o
            part = torch.empty_strided(
                part_shape, part_stride, dtype=torch.float32, device=q.device
            )
            return o, stats, part

        def split_runtime_args(
            inputs: Sequence[Any],
            context: Any,
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], inputs[2], context[2]

        def build_split_cached_call(
            metadata: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            block_n = int(metadata["BLOCK_N"])
            cached_args = split_tail + (
                group,
                head_dim,
                v_dim,
                out_dtype.itemsize,
                block_g,
                block_n,
                block_d,
                block_dv,
            )
            return split_grid, cached_args

        def combine_runtime_args(
            inputs: Sequence[Any],
            context: Any,
        ) -> tuple[Any, ...]:
            return context[2], context[0], context[1]

        def decode_result(
            context: Any,
        ) -> Any:
            if generate_stats:
                return context[0], context[1]
            return context[0]

        return make_kernel_pipeline_run_fn(
            PreparedKernelPipelineSpec(
                steps=(
                    PreparedPipelineStepSpec(
                        kernel=_sdpa_decode_split_kernel,
                        grid=split_grid,
                        runtime_args=split_runtime_args,
                        static_args=split_tail,
                        constexpr_kwargs=split_constexpr,
                        build_cached_call=build_split_cached_call,
                        first_launch_returns_metadata=True,
                    ),
                    PreparedPipelineStepSpec(
                        kernel=_sdpa_decode_combine_kernel,
                        grid=combine_grid,
                        runtime_args=combine_runtime_args,
                        static_args=combine_tail,
                    ),
                ),
                input_checks=qkv_input_checks,
                context_factory=make_decode_context,
                result=decode_result,
            ),
            default_run_fn,
        )

    use_gqa_causal = (
        sq > 1
        and sq == skv
        and not has_bias
        and generate_stats
        and banded
        and alignment != _BOTTOM_RIGHT
        and left is None
        and right == 0
        and out_dtype in (torch.float16, torch.bfloat16)
        and head_dim == 128
        and v_dim == 128
        and k_shape[1] == v_shape[1]
        and heads > k_shape[1]
        and heads // k_shape[1] <= 4
        and sq % 64 == 0
        and skv % 64 == 0
    )
    if use_gqa_causal:
        hkv = k_shape[1]
        group = heads // hkv
        gqa_constexpr = dict(
            HEAD_DIM=head_dim,
            V_DIM=v_dim,
            ELEM_SIZE=out_dtype.itemsize,
            BLOCK_D=head_dim,
            BLOCK_DV=v_dim,
        )
        gqa_tail = (
            attn_scale * _LOG2E,
            hkv,
            sq,
            skv,
            group,
            *q_stride,
            *k_stride,
            *v_stride,
            *o_stride,
            *stats_stride,
        )

        def gqa_grid(meta: dict[str, Any]) -> tuple[int, int, int]:
            return (
                triton.cdiv(sq, meta["BLOCK_M"]),
                batch * hkv,
                triton.cdiv(group, meta["BLOCK_H"]),
            )

        use_gqa_desc = (
            sq == 4096
            and skv == 4096
            and q_stride[3] == 1
            and k_stride[3] == 1
            and v_stride[3] == 1
        )

        def build_gqa_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            block_h = int(constexprs["BLOCK_H"])
            block_n = int(constexprs["BLOCK_N"])
            static_grid = (
                triton.cdiv(sq, block_m),
                batch * hkv,
                triton.cdiv(group, block_h),
            )
            cached_args = gqa_tail + (
                head_dim,
                v_dim,
                out_dtype.itemsize,
                block_m,
                block_h,
                block_n,
                head_dim,
                v_dim,
            )
            return static_grid, cached_args

        if use_gqa_desc:
            return make_single_kernel_run_fn(
                PreparedSingleKernelRunSpec(
                    kernel=PreparedSingleKernelSpec(
                        kernel=_sdpa_fwd_gqa_causal_desc_kernel,
                        grid=gqa_grid,
                        static_args=gqa_tail,
                        constexpr_kwargs=gqa_constexpr,
                        build_cached_call=build_gqa_cached_call,
                    ),
                    input_checks=qkv_input_checks,
                    output_factory=make_stats_output,
                    runtime_args=qkv_stats_runtime_args,
                    pre_launch=_ensure_triton_tma_allocator,
                ),
                default_run_fn,
            )

        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=_sdpa_fwd_gqa_causal_kernel,
                    grid=gqa_grid,
                    static_args=gqa_tail,
                    constexpr_kwargs=gqa_constexpr,
                    build_cached_call=build_gqa_cached_call,
                ),
                input_checks=qkv_input_checks,
                output_factory=make_stats_output,
                runtime_args=qkv_stats_runtime_args,
            ),
            default_run_fn,
        )

    use_mha_causal_desc = (
        sq == 2048
        and skv == 2048
        and not has_bias
        and generate_stats
        and banded
        and alignment != _BOTTOM_RIGHT
        and left is None
        and right == 0
        and out_dtype in (torch.float16, torch.bfloat16)
        and head_dim == 128
        and v_dim == 128
        and heads == k_shape[1]
        and heads == v_shape[1]
        and q_stride[3] == 1
        and k_stride[3] == 1
        and v_stride[3] == 1
    )
    if use_mha_causal_desc:
        use_mha_hostdesc = (
            batch == 2
            and heads == 16
            and k_shape[1] == 16
            and sq == 2048
            and skv == 2048
            and q_stride == (heads * sq * head_dim, sq * head_dim, head_dim, 1)
            and k_stride
            == (
                k_shape[1] * skv * head_dim,
                skv * head_dim,
                head_dim,
                1,
            )
            and v_stride == (v_shape[1] * skv * v_dim, skv * v_dim, v_dim, 1)
        )
        if use_mha_hostdesc:
            descriptor_key = None
            descriptor = None
            descriptor_shape = (batch * k_shape[1] * skv, head_dim)
            descriptor_stride = (head_dim, 1)
            descriptor_block = [64, head_dim]

            def get_k_descriptor(k: torch.Tensor):
                nonlocal descriptor_key, descriptor
                key = (
                    k.data_ptr(),
                    tuple(k.shape),
                    tuple(k.stride()),
                    k.dtype,
                    k.device.type,
                    k.device.index,
                )
                if descriptor is None or descriptor_key != key:
                    descriptor = TensorDescriptor(
                        k,
                        list(descriptor_shape),
                        list(descriptor_stride),
                        descriptor_block,
                    )
                    descriptor_key = key
                return descriptor

            def hostdesc_runtime_args(inputs, output):
                return (
                    inputs[0],
                    get_k_descriptor(inputs[1]),
                    inputs[2],
                    output[0],
                    output[1],
                )

            mha_host_tail = (
                attn_scale * _LOG2E,
                heads,
                sq,
                skv,
                *q_stride,
                *v_stride,
                *o_stride,
                *stats_stride,
            )
            mha_host_constexpr = {
                "HEAD_DIM": head_dim,
                "V_DIM": v_dim,
                "ELEM_SIZE": out_dtype.itemsize,
                "BLOCK_D": head_dim,
                "BLOCK_DV": v_dim,
            }

            def mha_host_grid(meta):
                return (batch_heads, triton.cdiv(sq, meta["BLOCK_M"]))

            def build_mha_host_cached_call(meta):
                block_m = int(meta["BLOCK_M"])
                block_n = int(meta["BLOCK_N"])
                return (
                    batch_heads,
                    triton.cdiv(sq, block_m),
                    1,
                ), mha_host_tail + (
                    head_dim,
                    v_dim,
                    out_dtype.itemsize,
                    block_m,
                    block_n,
                    head_dim,
                    v_dim,
                )

            return make_single_kernel_run_fn(
                PreparedSingleKernelRunSpec(
                    kernel=PreparedSingleKernelSpec(
                        kernel=_sdpa_fwd_mha_causal_hostdesc_kernel,
                        grid=mha_host_grid,
                        static_args=mha_host_tail,
                        constexpr_kwargs=mha_host_constexpr,
                        build_cached_call=build_mha_host_cached_call,
                    ),
                    input_checks=qkv_input_checks,
                    output_factory=make_stats_output,
                    runtime_args=hostdesc_runtime_args,
                    pre_launch=_ensure_triton_tma_allocator,
                ),
                default_run_fn,
            )

        mha_desc_constexpr = dict(
            HEAD_DIM=head_dim,
            V_DIM=v_dim,
            ELEM_SIZE=out_dtype.itemsize,
            BLOCK_D=head_dim,
            BLOCK_DV=v_dim,
        )
        mha_desc_tail = (
            attn_scale * _LOG2E,
            heads,
            sq,
            skv,
            *q_stride,
            *k_stride,
            *v_stride,
            *o_stride,
            *stats_stride,
        )

        def mha_desc_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return (batch_heads, triton.cdiv(sq, meta["BLOCK_M"]))

        def build_mha_desc_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            block_n = int(constexprs["BLOCK_N"])
            static_grid = (batch_heads, triton.cdiv(sq, block_m), 1)
            cached_args = mha_desc_tail + (
                head_dim,
                v_dim,
                out_dtype.itemsize,
                block_m,
                block_n,
                head_dim,
                v_dim,
            )
            return static_grid, cached_args

        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=_sdpa_fwd_mha_causal_desc_kernel,
                    grid=mha_desc_grid,
                    static_args=mha_desc_tail,
                    constexpr_kwargs=mha_desc_constexpr,
                    build_cached_call=build_mha_desc_cached_call,
                ),
                input_checks=qkv_input_checks,
                output_factory=make_stats_output,
                runtime_args=qkv_stats_runtime_args,
                pre_launch=_ensure_triton_tma_allocator,
            ),
            default_run_fn,
        )

    use_dense_exact = (
        sq > 1
        and not has_bias
        and not generate_stats
        and not banded
        and out_dtype in (torch.float16, torch.bfloat16)
        and head_dim == 128
        and v_dim == 128
        and sq % 64 == 0
        and skv % 64 == 0
    )
    if use_dense_exact:
        dense_constexpr = dict(
            HEAD_DIM=head_dim,
            V_DIM=v_dim,
            ELEM_SIZE=out_dtype.itemsize,
            BLOCK_D=head_dim,
            BLOCK_DV=v_dim,
        )
        dense_tail = (
            attn_scale * _LOG2E,
            heads,
            sq,
            skv,
            heads // k_shape[1],
            heads // v_shape[1],
            *q_stride,
            *k_stride,
            *v_stride,
            *o_stride,
        )

        def dense_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return (triton.cdiv(sq, meta["BLOCK_M"]), batch_heads)

        def build_dense_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            block_n = int(constexprs["BLOCK_N"])
            static_grid = (triton.cdiv(sq, block_m), batch_heads, 1)
            cached_args = dense_tail + (
                head_dim,
                v_dim,
                out_dtype.itemsize,
                block_m,
                block_n,
                head_dim,
                v_dim,
            )
            return static_grid, cached_args

        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=_sdpa_fwd_dense_exact_kernel,
                    grid=dense_grid,
                    static_args=dense_tail,
                    constexpr_kwargs=dense_constexpr,
                    build_cached_call=build_dense_cached_call,
                ),
                input_checks=qkv_input_checks,
                output_factory=make_output_tensor,
                runtime_args=qkv_output_runtime_args,
            ),
            default_run_fn,
        )

    def tune_grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(sq, meta["BLOCK_M"]), batch_heads)

    def build_sdpa_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_n = int(constexprs["BLOCK_N"])
        static_grid = (
            triton.cdiv(sq, block_m),
            batch_heads,
            1,
        )
        cached_args = scalar_tail + (
            head_dim,
            v_dim,
            out_dtype.itemsize,
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
                kernel=_sdpa_fwd_kernel,
                grid=tune_grid,
                static_args=scalar_tail,
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_sdpa_cached_call,
            ),
            input_checks=sdpa_input_checks,
            output_factory=make_sdpa_output,
            runtime_args=sdpa_runtime_args,
            result=sdpa_result,
        ),
        default_run_fn,
    )
