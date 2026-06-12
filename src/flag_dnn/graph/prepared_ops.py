from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, cast

import torch

from flag_dnn import runtime
from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

RunFn = Callable[[Sequence[Any], dict[str, Any]], Any]


def _require_runtime_backend(inputs: Sequence[Any], op_type: str) -> None:
    tensor_inputs = [
        value for value in inputs if isinstance(value, torch.Tensor)
    ]
    if not tensor_inputs or not all(
        is_runtime_device_tensor(value) for value in tensor_inputs
    ):
        raise NotImplementedError(
            f"FlagDNN graph {op_type} requires runtime device tensors; "
            "torch fallback is disabled"
        )


def _unsupported_triton_path(op_type: str, detail: str) -> None:
    raise NotImplementedError(
        f"FlagDNN graph {op_type} has no Triton path for {detail}; "
        "torch fallback is disabled"
    )


def prepare_run_fn(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> RunFn:
    prepared = _prepare_pointwise(op_type, attrs, input_specs, default_run_fn)
    if prepared is not None:
        return prepared
    if op_type == "sdpa":
        prepared = _prepare_sdpa(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return prepared
    if op_type == "sdpa_backward":
        prepared = _prepare_sdpa_backward(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return prepared
    if op_type == "conv_fprop":
        prepared = _prepare_conv_fprop(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return prepared
    if op_type == "conv_dgrad":
        prepared = _prepare_conv_dgrad(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return prepared
    if op_type == "conv_wgrad":
        prepared = _prepare_conv_wgrad(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return prepared
    return default_run_fn


def _static_shape(spec: TensorSpec) -> Optional[tuple[int, ...]]:
    shape = tuple(spec.shape)
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def _prepare_sdpa(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    # Bind every scalar kernel argument at plan time so graph replay only
    # has to allocate outputs and launch the cached compiled kernel.
    import math

    import triton

    from flag_dnn.ops.sdpa import (
        _BOTTOM_RIGHT,
        _LOG2E,
        _UNBOUNDED_DIAG,
        _sdpa_decode_combine_kernel,
        _sdpa_decode_split_kernel,
        _sdpa_fwd_dense_exact_kernel,
        _sdpa_fwd_gqa_causal_kernel,
        _sdpa_fwd_kernel,
    )
    from flag_dnn.runtime import torch_device_fn

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
        expected_bias_stride = bias_stride_full
    else:
        bias_stride = (0, 0, 0, 0)
        expected_bias_stride = None

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
        split_cache: list[Any] = [None, None]
        combine_cache: list[Any] = [None]

        def run_decode(
            inputs: Sequence[Any], run_attrs: dict[str, Any]
        ) -> Any:
            q, k, v = inputs[0], inputs[1], inputs[2]
            if (
                not isinstance(q, torch.Tensor)
                or q.stride() != q_stride
                or k.stride() != k_stride
                or v.stride() != v_stride
            ):
                return default_run_fn(inputs, run_attrs)
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

            split_kernel = split_cache[0]
            if split_kernel is None:
                with torch_device_fn.device(q.device):
                    split_kernel, split_meta = _sdpa_decode_split_kernel[
                        split_grid
                    ](q, k, v, part, *split_tail, **split_constexpr)
                block_n = int(split_meta["BLOCK_N"])
                split_cache[1] = split_tail + (
                    group,
                    head_dim,
                    v_dim,
                    out_dtype.itemsize,
                    block_g,
                    block_n,
                    block_d,
                    block_dv,
                )
                split_cache[0] = split_kernel
            else:
                split_kernel[split_grid](
                    q, k, v, part, *cast(tuple[Any, ...], split_cache[1])
                )

            combine_kernel = combine_cache[0]
            if combine_kernel is None:
                with torch_device_fn.device(q.device):
                    combine_cache[0] = _sdpa_decode_combine_kernel[
                        combine_grid
                    ](part, o, stats, *combine_tail)
            else:
                combine_kernel[combine_grid](part, o, stats, *combine_tail)
            if generate_stats:
                return o, stats
            return o

        return run_decode

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

        gqa_cache: list[Any] = [None, None, None]

        def run_gqa_causal(
            inputs: Sequence[Any], run_attrs: dict[str, Any]
        ) -> Any:
            q, k, v = inputs[0], inputs[1], inputs[2]
            if (
                not isinstance(q, torch.Tensor)
                or q.stride() != q_stride
                or k.stride() != k_stride
                or v.stride() != v_stride
            ):
                return default_run_fn(inputs, run_attrs)
            o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
            stats = torch.empty(
                stats_shape, dtype=torch.float32, device=q.device
            )
            kernel = gqa_cache[0]
            if kernel is None:
                with torch_device_fn.device(q.device):
                    kernel, constexprs = _sdpa_fwd_gqa_causal_kernel[gqa_grid](
                        q, k, v, o, stats, *gqa_tail, **gqa_constexpr
                    )
                block_m = int(constexprs["BLOCK_M"])
                block_h = int(constexprs["BLOCK_H"])
                block_n = int(constexprs["BLOCK_N"])
                gqa_cache[1] = (
                    triton.cdiv(sq, block_m),
                    batch * hkv,
                    triton.cdiv(group, block_h),
                )
                gqa_cache[2] = gqa_tail + (
                    head_dim,
                    v_dim,
                    out_dtype.itemsize,
                    block_m,
                    block_h,
                    block_n,
                    head_dim,
                    v_dim,
                )
                gqa_cache[0] = kernel
            else:
                kernel[gqa_cache[1]](
                    q, k, v, o, stats, *cast(tuple[Any, ...], gqa_cache[2])
                )
            return o, stats

        return run_gqa_causal

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

        dense_cache: list[Any] = [None, None, None]

        def run_dense_exact(
            inputs: Sequence[Any], run_attrs: dict[str, Any]
        ) -> Any:
            q, k, v = inputs[0], inputs[1], inputs[2]
            if (
                not isinstance(q, torch.Tensor)
                or q.stride() != q_stride
                or k.stride() != k_stride
                or v.stride() != v_stride
            ):
                return default_run_fn(inputs, run_attrs)
            o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
            kernel = dense_cache[0]
            if kernel is None:
                with torch_device_fn.device(q.device):
                    kernel, constexprs = _sdpa_fwd_dense_exact_kernel[
                        dense_grid
                    ](q, k, v, o, *dense_tail, **dense_constexpr)
                block_m = int(constexprs["BLOCK_M"])
                block_n = int(constexprs["BLOCK_N"])
                dense_cache[1] = (triton.cdiv(sq, block_m), batch_heads, 1)
                dense_cache[2] = dense_tail + (
                    head_dim,
                    v_dim,
                    out_dtype.itemsize,
                    block_m,
                    block_n,
                    head_dim,
                    v_dim,
                )
                dense_cache[0] = kernel
            else:
                kernel[dense_cache[1]](
                    q, k, v, o, *cast(tuple[Any, ...], dense_cache[2])
                )
            return o

        return run_dense_exact

    def tune_grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(sq, meta["BLOCK_M"]), batch_heads)

    # (CompiledKernel, static grid, positional args in signature order)
    kernel_cache: list[Any] = [None, None, None]

    def run_sdpa(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        q, k, v = inputs[0], inputs[1], inputs[2]
        if (
            not isinstance(q, torch.Tensor)
            or q.stride() != q_stride
            or k.stride() != k_stride
            or v.stride() != v_stride
        ):
            return default_run_fn(inputs, run_attrs)
        if has_bias:
            bias = inputs[3]
            if bias.stride() != expected_bias_stride:
                return default_run_fn(inputs, run_attrs)
        else:
            bias = q
        o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        if generate_stats:
            stats = torch.empty(
                stats_shape, dtype=torch.float32, device=q.device
            )
        else:
            stats = o
        kernel = kernel_cache[0]
        if kernel is None:
            with torch_device_fn.device(q.device):
                kernel, constexprs = _sdpa_fwd_kernel[tune_grid](
                    q,
                    k,
                    v,
                    bias,
                    o,
                    stats,
                    *scalar_tail,
                    **constexpr_kwargs,
                )
            block_m = int(constexprs["BLOCK_M"])
            block_n = int(constexprs["BLOCK_N"])
            kernel_cache[1] = (
                triton.cdiv(sq, block_m),
                batch_heads,
                1,
            )
            kernel_cache[2] = scalar_tail + (
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
            kernel_cache[0] = kernel
        else:
            kernel[kernel_cache[1]](q, k, v, bias, o, stats, *kernel_cache[2])
        if generate_stats:
            return o, stats
        return o

    return run_sdpa


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
    from flag_dnn.runtime import torch_device_fn

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
    zero_cache: list[Any] = [None]
    fused_cache: list[Any] = [None]

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        q, k, v, o, dO, stats = inputs[:6]
        if (
            not isinstance(q, torch.Tensor)
            or q.stride() != q_stride
            or k.stride() != k_stride
            or v.stride() != v_stride
            or o.stride() != o_stride
            or dO.stride() != do_stride
            or stats.stride() != stats_stride
        ):
            return default_run_fn(inputs, run_attrs)

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

        zero_kernel = zero_cache[0]
        if zero_kernel is None:
            with torch_device_fn.device(q.device):
                if use_exact_delta:
                    assert delta is not None and delta_config is not None
                    zero_cache[0] = _zero_three_equal_and_delta_kernel[
                        zero_delta_grid
                    ](
                        dQ,
                        dK,
                        dV,
                        o,
                        dO,
                        delta,
                        batch * heads * sq,
                        heads,
                        sq,
                        *o_stride,
                        *do_stride,
                        *delta_stride,
                        **delta_config,
                    )
                elif use_delta_fused:
                    assert delta is not None and delta_config is not None
                    zero_cache[0] = _zero_three_and_delta_kernel[
                        zero_delta_grid
                    ](
                        dQ,
                        dK,
                        dV,
                        o,
                        dO,
                        delta,
                        q_numel,
                        k_numel,
                        v_numel,
                        batch * heads * sq,
                        heads,
                        sq,
                        *o_stride,
                        *do_stride,
                        *delta_stride,
                        **delta_config,
                    )
                else:
                    zero_cache[0] = _zero_three_contiguous_kernel[zero_grid](
                        dQ,
                        dK,
                        dV,
                        q_numel,
                        k_numel,
                        v_numel,
                        BLOCK=zero_block,
                    )
        else:
            if use_exact_delta:
                assert delta is not None and delta_config is not None
                zero_kernel[zero_delta_grid](
                    dQ,
                    dK,
                    dV,
                    o,
                    dO,
                    delta,
                    batch * heads * sq,
                    heads,
                    sq,
                    *o_stride,
                    *do_stride,
                    *delta_stride,
                    delta_config["BLOCK_ZERO"],
                    delta_config["BLOCK_M"],
                    delta_config["BLOCK_D"],
                )
            elif use_delta_fused:
                assert delta is not None and delta_config is not None
                zero_kernel[zero_delta_grid](
                    dQ,
                    dK,
                    dV,
                    o,
                    dO,
                    delta,
                    q_numel,
                    k_numel,
                    v_numel,
                    batch * heads * sq,
                    heads,
                    sq,
                    *o_stride,
                    *do_stride,
                    *delta_stride,
                    delta_config["BLOCK_ZERO"],
                    delta_config["BLOCK_M"],
                    delta_config["BLOCK_D"],
                )
            else:
                zero_kernel[zero_grid](
                    dQ,
                    dK,
                    dV,
                    q_numel,
                    k_numel,
                    v_numel,
                    zero_block,
                )

        fused_kernel = fused_cache[0]
        if fused_kernel is None:
            with torch_device_fn.device(q.device):
                if use_exact_delta:
                    assert delta is not None
                    fused_cache[
                        0
                    ] = _sdpa_bwd_fused_atomic_causal_exact_delta_kernel[
                        fused_grid
                    ](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        delta,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                elif exact_causal:
                    fused_cache[
                        0
                    ] = _sdpa_bwd_fused_atomic_causal_exact_kernel[fused_grid](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                elif use_delta_fused:
                    assert delta is not None
                    fused_cache[
                        0
                    ] = _sdpa_bwd_fused_atomic_causal_delta_tri_kernel[
                        fused_grid
                    ](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        delta,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        NUM_BLOCKS=num_m_blocks,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                elif triangular_causal and use_fused_gqa:
                    fused_cache[
                        0
                    ] = _sdpa_bwd_fused_atomic_gqa_causal_tri_kernel[
                        fused_grid
                    ](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        NUM_BLOCKS=num_m_blocks,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                elif triangular_causal:
                    fused_cache[0] = _sdpa_bwd_fused_atomic_causal_tri_kernel[
                        fused_grid
                    ](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        NUM_BLOCKS=num_m_blocks,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                elif causal_top_left and use_fused_gqa:
                    fused_cache[0] = _sdpa_bwd_fused_atomic_gqa_causal_kernel[
                        fused_grid
                    ](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                elif causal_top_left:
                    fused_cache[0] = _sdpa_bwd_fused_atomic_causal_kernel[
                        fused_grid
                    ](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        BANDED=banded,
                        CAUSAL_TOP_LEFT=causal_top_left,
                        **fused_config,
                    )
                else:
                    fused_cache[0] = _sdpa_bwd_fused_atomic_kernel[fused_grid](
                        q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        dQ,
                        dK,
                        dV,
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
                        HEAD_DIM=head_dim,
                        **fused_config,
                    )
        elif use_exact_delta:
            assert delta is not None
            fused_kernel[fused_grid](
                q,
                k,
                v,
                o,
                dO,
                stats,
                delta,
                dQ,
                dK,
                dV,
                *causal_exact_delta_tail,
            )
        elif exact_causal:
            fused_kernel[fused_grid](
                q, k, v, o, dO, stats, dQ, dK, dV, *causal_tail
            )
        elif use_delta_fused:
            assert delta is not None
            fused_kernel[fused_grid](
                q,
                k,
                v,
                o,
                dO,
                stats,
                delta,
                dQ,
                dK,
                dV,
                *causal_delta_tri_tail,
            )
        elif triangular_causal and use_fused_gqa:
            fused_kernel[fused_grid](
                q, k, v, o, dO, stats, dQ, dK, dV, *gqa_tri_tail
            )
        elif triangular_causal:
            fused_kernel[fused_grid](
                q, k, v, o, dO, stats, dQ, dK, dV, *causal_tri_tail
            )
        elif causal_top_left and use_fused_gqa:
            fused_kernel[fused_grid](
                q, k, v, o, dO, stats, dQ, dK, dV, *gqa_tail
            )
        elif causal_top_left:
            fused_kernel[fused_grid](
                q, k, v, o, dO, stats, dQ, dK, dV, *causal_tail
            )
        else:
            fused_kernel[fused_grid](
                q, k, v, o, dO, stats, dQ, dK, dV, *dense_tail
            )

        return dQ, dK, dV

    return run


_POINTWISE_BINARY_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}

_POINTWISE_CMP_REVERSE = {
    "eq": "eq",
    "ne": "ne",
    "lt": "gt",
    "le": "ge",
    "gt": "lt",
    "ge": "le",
}

_POINTWISE_CMP_ALIASES = {
    "cmp_eq": "eq",
    "cmp_neq": "ne",
    "cmp_lt": "lt",
    "cmp_le": "le",
    "cmp_gt": "gt",
    "cmp_ge": "ge",
}


def _prepare_pointwise(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if not input_specs or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None

    actual_op_type = _POINTWISE_CMP_ALIASES.get(
        op_type, attrs.get("op_type", op_type)
    )
    if actual_op_type in _POINTWISE_BINARY_OPS:
        return _prepare_binary_pointwise(actual_op_type, attrs, default_run_fn)
    if actual_op_type == "pow":
        return _prepare_pow_pointwise(attrs, default_run_fn)
    if op_type == "abs":
        from flag_dnn.ops.abs import abs as abs_op

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "abs")
            return abs_op(inputs[0])

        return run
    if op_type == "sigmoid":
        from flag_dnn.ops.sigmoid import sigmoid

        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "sigmoid")
            return sigmoid(
                inputs[0], compute_data_type=compute_data_type, name=name
            )

        return run
    if op_type == "sigmoid_backward":
        from flag_dnn.ops.sigmoid_backward import sigmoid_backward

        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "sigmoid_backward")
            return sigmoid_backward(
                inputs[0],
                inputs[1],
                compute_data_type=compute_data_type,
                name=name,
            )

        return run
    return None


def _pointwise_operands(
    inputs: Sequence[Any], attrs: dict[str, Any]
) -> tuple[Any, Any]:
    left = inputs[0]
    if len(inputs) > 1:
        right = inputs[1]
    else:
        right = attrs["other"]
    if attrs.get("reverse"):
        return right, left
    return left, right


def _prepare_binary_pointwise(
    op_type: str, attrs: dict[str, Any], default_run_fn: RunFn
) -> RunFn:
    from flag_dnn.ops.binary import binary

    alpha = attrs.get("alpha", 1)
    rounding_mode = attrs.get("rounding_mode")

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, op_type)
        left, right = _pointwise_operands(inputs, attrs)
        if op_type == "add":
            if isinstance(left, torch.Tensor):
                return binary(left, right, alpha=alpha, op_type="add")
            if isinstance(right, torch.Tensor):
                return binary(right, left, alpha=alpha, op_type="add")
        elif op_type == "sub":
            if isinstance(left, torch.Tensor):
                return binary(left, right, alpha=alpha, op_type="sub")
        elif op_type == "mul":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="mul")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="mul")
        elif op_type == "div":
            if isinstance(left, torch.Tensor):
                return binary(
                    left,
                    right,
                    rounding_mode=rounding_mode,
                    op_type="div",
                )
        elif op_type == "max":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="max")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="max")
        else:
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type=op_type)
            if isinstance(right, torch.Tensor):
                return binary(
                    right,
                    left,
                    op_type=_POINTWISE_CMP_REVERSE[op_type],
                )
        _unsupported_triton_path(op_type, "operand combination")

    return run


def _prepare_pow_pointwise(
    attrs: dict[str, Any], default_run_fn: RunFn
) -> RunFn:
    from flag_dnn.ops.pow import pow as pow_op

    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "pow")
        left, right = _pointwise_operands(inputs, attrs)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return pow_op(
                left, right, compute_data_type=compute_data_type, name=name
            )
        _unsupported_triton_path("pow", "two scalar operands")

    return run


def _prepare_conv_dgrad(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    del default_run_fn
    if len(input_specs) < 2:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:2]):
        return None

    from flag_dnn.ops.conv_dgrad import conv_dgrad

    input_size = tuple(int(dim) for dim in attrs["input_size"])
    padding = attrs.get("padding")
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    stride = attrs.get("stride", 1)
    dilation = attrs.get("dilation", 1)
    convolution_mode = attrs.get("convolution_mode", "CROSS_CORRELATION")
    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")
    groups = int(attrs.get("groups", 1))
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "conv_dgrad")
        loss = inputs[0]
        key = (
            loss.device.type,
            loss.device.index,
            loss.dtype,
            input_size,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty(
                input_size, device=loss.device, dtype=loss.dtype
            )
            output_cache[key] = output
        return conv_dgrad(
            loss,
            inputs[1],
            input_size=input_size,
            padding=padding,
            pre_padding=pre_padding,
            post_padding=post_padding,
            stride=stride,
            dilation=dilation,
            convolution_mode=convolution_mode,
            compute_data_type=compute_data_type,
            name=name,
            groups=groups,
            _output=output,
        )

    return run


def _prepare_conv_wgrad(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    del default_run_fn
    if len(input_specs) < 2:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:2]):
        return None

    from flag_dnn.ops.conv_wgrad import (
        _conv_wgrad2d_1x1_atomic_nodiv_kernel,
        _conv_wgrad2d_1x1_reduce_kernel,
        _conv_wgrad2d_1x1_split_nodiv_kernel,
        _conv_wgrad2d_reduce_kernel,
        _conv_wgrad2d_stride2_3tap_atomic_kernel,
        _conv_wgrad2d_stride2_row4_split_kernel,
        _conv_wgrad_zero_kernel,
        conv_wgrad,
    )

    filter_size = tuple(int(dim) for dim in attrs["filter_size"])
    padding = attrs.get("padding")
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    stride = attrs.get("stride", 1)
    dilation = attrs.get("dilation", 1)
    convolution_mode = attrs.get("convolution_mode", "CROSS_CORRELATION")
    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")
    groups = int(attrs.get("groups", 1))
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    workspace_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    image_spec = input_specs[0]
    loss_spec = input_specs[1]
    if image_spec.device is not None and all(
        isinstance(dim, int) for dim in image_spec.shape + loss_spec.shape
    ):
        device = torch.device(image_spec.device)
        dtype = torch_dtype(image_spec.dtype)
        output_cache[(device.type, device.index, dtype, filter_size)] = (
            torch.empty(filter_size, device=device, dtype=dtype)
        )
        image_shape = tuple(int(dim) for dim in image_spec.shape)
        loss_shape = tuple(int(dim) for dim in loss_spec.shape)
        if (
            image_shape == (8, 64, 28, 28)
            and loss_shape == (8, 128, 28, 28)
            and filter_size == (128, 64, 1, 1)
            and groups == 1
        ):
            if dtype in (torch.float16, torch.bfloat16):
                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_1x1_nodiv_split_v7", 32, 128, 64),
                    )
                ] = torch.empty(
                    (32, 128, 64), device=device, dtype=partial_dtype
                )
        if (
            image_shape == (8, 64, 56, 56)
            and loss_shape == (8, 128, 28, 28)
            and filter_size == (128, 64, 3, 3)
            and groups == 1
        ):
            if dtype in (torch.float16, torch.bfloat16):
                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_stride2_row4_v1", 8, 128, 64, 9),
                    )
                ] = torch.empty(
                    (8, 128, 64, 9), device=device, dtype=partial_dtype
                )

        if (
            len(image_shape) == 4
            and len(loss_shape) == 4
            and len(filter_size) == 4
        ):
            stride_tuple = _tuple_n(stride, 2, "stride")
            dilation_tuple = _tuple_n(dilation, 2, "dilation")
            if pre_padding is None and post_padding is None:
                pad = _tuple_n(0 if padding is None else padding, 2, "padding")
                pre = post = pad
            else:
                pre = _tuple_n(pre_padding, 2, "pre_padding")
                post = _tuple_n(post_padding, 2, "post_padding")
            mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
            exact_1x1 = (
                image_shape == (8, 64, 28, 28)
                and loss_shape == (8, 128, 28, 28)
                and filter_size == (128, 64, 1, 1)
                and stride_tuple == (1, 1)
                and pre == (0, 0)
                and post == (0, 0)
                and dilation_tuple == (1, 1)
                and mode == "CROSS_CORRELATION"
                and groups == 1
            )
            if exact_1x1:
                output = output_cache[
                    (device.type, device.index, dtype, filter_size)
                ]
                if dtype == torch.float32:

                    def run_exact_1x1_atomic(
                        inputs: Sequence[Any], _attrs: dict[str, Any]
                    ) -> Any:
                        image, loss = inputs
                        _conv_wgrad_zero_kernel[(8,)](
                            output,
                            8192,
                            BLOCK=1024,
                            num_warps=4,
                        )
                        _conv_wgrad2d_1x1_atomic_nodiv_kernel[(8, 32)](
                            image,
                            loss,
                            output,
                            784,
                            64,
                            128,
                            50176,
                            784,
                            100352,
                            784,
                            64,
                            1,
                            4,
                            BLOCK_CO=16,
                            BLOCK_CI=64,
                            BLOCK_M=64,
                            num_warps=4,
                            num_stages=3,
                        )
                        return output

                    return run_exact_1x1_atomic

                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                partial = workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_1x1_nodiv_split_v7", 32, 128, 64),
                    )
                ]

                def run_exact_1x1_split(
                    inputs: Sequence[Any], _attrs: dict[str, Any]
                ) -> Any:
                    image, loss = inputs
                    _conv_wgrad2d_1x1_split_nodiv_kernel[(8, 32)](
                        image,
                        loss,
                        partial,
                        784,
                        128,
                        64,
                        128,
                        50176,
                        784,
                        100352,
                        784,
                        4,
                        BLOCK_CO=16,
                        BLOCK_CI=64,
                        BLOCK_M=256,
                        num_warps=8,
                        num_stages=3,
                    )
                    if dtype == torch.bfloat16:
                        _conv_wgrad2d_1x1_reduce_kernel[(64, 1)](
                            partial,
                            output,
                            128,
                            64,
                            128,
                            64,
                            1,
                            32,
                            BLOCK_CO=8,
                            BLOCK_CI=16,
                            num_warps=4,
                            num_stages=1,
                        )
                    else:
                        _conv_wgrad2d_1x1_reduce_kernel[(8, 1)](
                            partial,
                            output,
                            128,
                            64,
                            128,
                            64,
                            1,
                            32,
                            BLOCK_CO=16,
                            BLOCK_CI=64,
                            num_warps=8,
                            num_stages=1,
                        )
                    return output

                return run_exact_1x1_split

            exact_stride2 = (
                image_shape == (8, 64, 56, 56)
                and loss_shape == (8, 128, 28, 28)
                and filter_size == (128, 64, 3, 3)
                and stride_tuple == (2, 2)
                and pre == (1, 1)
                and post == (1, 1)
                and dilation_tuple == (1, 1)
                and mode == "CROSS_CORRELATION"
                and groups == 1
            )
            if exact_stride2:
                output = output_cache[
                    (device.type, device.index, dtype, filter_size)
                ]
                if dtype == torch.float32:

                    def run_exact_stride2_atomic(
                        inputs: Sequence[Any], _attrs: dict[str, Any]
                    ) -> Any:
                        image, loss = inputs
                        _conv_wgrad_zero_kernel[(72,)](
                            output,
                            73728,
                            BLOCK=1024,
                            num_warps=4,
                        )
                        _conv_wgrad2d_stride2_3tap_atomic_kernel[(16, 3, 8)](
                            image,
                            loss,
                            output,
                            6272,
                            56,
                            56,
                            28,
                            28,
                            128,
                            64,
                            200704,
                            3136,
                            56,
                            1,
                            100352,
                            784,
                            28,
                            1,
                            576,
                            9,
                            3,
                            1,
                            8,
                            BLOCK_CO=16,
                            BLOCK_CI=32,
                            BLOCK_M=16,
                            num_warps=2,
                            num_stages=3,
                        )
                        return output

                    return run_exact_stride2_atomic

                partial_dtype = (
                    dtype if dtype == torch.float16 else torch.float32
                )
                partial = workspace_cache[
                    (
                        device.type,
                        device.index,
                        partial_dtype,
                        ("2d_stride2_row4_v1", 8, 128, 64, 9),
                    )
                ]

                def run_exact_stride2(
                    inputs: Sequence[Any], _attrs: dict[str, Any]
                ) -> Any:
                    image, loss = inputs
                    _conv_wgrad2d_stride2_row4_split_kernel[(16, 3, 8)](
                        image,
                        loss,
                        partial,
                        128,
                        64,
                        128,
                        200704,
                        3136,
                        56,
                        1,
                        100352,
                        784,
                        28,
                        1,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        BLOCK_HW=128,
                        num_warps=4,
                        num_stages=2,
                    )
                    _conv_wgrad2d_reduce_kernel[(16, 9, 1)](
                        partial,
                        output,
                        128,
                        64,
                        128,
                        576,
                        9,
                        3,
                        1,
                        3,
                        3,
                        8,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        num_warps=4,
                        num_stages=1,
                    )
                    return output

                return run_exact_stride2

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "conv_wgrad")
        image = inputs[0]
        key = (
            image.device.type,
            image.device.index,
            image.dtype,
            filter_size,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty(
                filter_size, device=image.device, dtype=image.dtype
            )
            output_cache[key] = output
        return conv_wgrad(
            image,
            inputs[1],
            filter_size=filter_size,
            padding=padding,
            pre_padding=pre_padding,
            post_padding=post_padding,
            stride=stride,
            dilation=dilation,
            convolution_mode=convolution_mode,
            compute_data_type=compute_data_type,
            name=name,
            groups=groups,
            _output=output,
            _workspace=workspace_cache,
        )

    return run


def _prepare_conv_fprop(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) < 2:
        return None

    rank = _conv_rank(input_specs[0], input_specs[1])
    if rank not in (1, 2, 3):
        return None
    if not _is_cross_correlation(attrs.get("convolution_mode")):
        return None

    from flag_dnn.ops.conv1d import conv1d
    from flag_dnn.ops.conv2d import conv2d
    from flag_dnn.ops.conv3d import conv3d

    stride = _tuple_n(attrs.get("stride", 1), rank, "stride")
    dilation = _tuple_n(attrs.get("dilation", 1), rank, "dilation")
    groups = int(attrs.get("groups", 1))
    padding = _direct_padding(
        rank,
        attrs.get("padding"),
        attrs.get("pre_padding"),
        attrs.get("post_padding"),
    )
    if rank == 1:
        stride_arg = stride[0]
        dilation_arg = dilation[0]

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "conv_fprop")
            return conv1d(
                inputs[0],
                inputs[1],
                stride=stride_arg,
                padding=padding,
                dilation=dilation_arg,
                groups=groups,
            )

        return run

    if rank == 2:
        stride_2d = cast(tuple[int, int], stride)
        dilation_2d = cast(tuple[int, int], dilation)

        def run_conv2d(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "conv_fprop")
            return conv2d(
                inputs[0],
                inputs[1],
                stride=stride_2d,
                padding=padding,
                dilation=dilation_2d,
                groups=groups,
            )

        return run_conv2d

    stride_3d = cast(tuple[int, int, int], stride)
    dilation_3d = cast(tuple[int, int, int], dilation)

    def run_conv3d(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "conv_fprop")
        return conv3d(
            inputs[0],
            inputs[1],
            stride=stride_3d,
            padding=padding,
            dilation=dilation_3d,
            groups=groups,
        )

    return run_conv3d


def _conv_rank(image: TensorSpec, weight: TensorSpec) -> int:
    image_rank = len(image.shape)
    weight_rank = len(weight.shape)
    if image_rank == 2 and weight_rank == 3:
        return 1
    if image_rank >= 3 and image_rank == weight_rank:
        return image_rank - 2
    return -1


def _tuple_n(value: Any, rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(item) for item in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _direct_padding(
    rank: int,
    padding: Any,
    pre_padding: Any,
    post_padding: Any,
) -> Any:
    if pre_padding is None and post_padding is None:
        if padding is None:
            return 0
        if rank == 1 and not isinstance(padding, str):
            return _tuple_n(padding, 1, "padding")[0]
        return padding

    pre = _tuple_n(pre_padding, rank, "pre_padding")
    post = _tuple_n(post_padding, rank, "post_padding")
    if rank == 1:
        return (pre[0], post[0])
    if rank == 2:
        return (pre[0], post[0], pre[1], post[1])
    return (pre[0], post[0], pre[1], post[1], pre[2], post[2])


def _is_cross_correlation(convolution_mode: Any) -> bool:
    if convolution_mode is None:
        return True
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    return mode == "CROSS_CORRELATION"


def _is_runtime_device_spec(spec: TensorSpec) -> bool:
    if spec.device is None:
        return False
    device = str(spec.device)
    runtime_name = runtime.device.name
    return device == runtime_name or device.startswith(runtime_name + ":")
