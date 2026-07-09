from __future__ import annotations

import weakref
from typing import Any, Optional, Sequence

import torch

from flag_dnn.graph.prepared import (
    PreparedTensorCache,
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    get_cached_empty_tensor,
    make_single_kernel_launcher,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
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

    from triton.tools.tensor_descriptor import TensorDescriptor

    from flag_dnn.ops.sdpa import (
        _BOTTOM_RIGHT,
        _LOG2E,
        _TOP_LEFT,
        _UNBOUNDED_DIAG,
        _ensure_triton_tma_allocator,
    )
    from flag_dnn.ops.sdpa_fp8 import (
        _sdpa_fp8_fwd_dense512_hostdesc_tma_kernel,
        _sdpa_fp8_fwd_fast_kernel,
        _sdpa_fp8_fwd_gqa_causal_tma_kernel,
        _sdpa_fp8_fwd_gqa_causal_vt_kernel,
        _sdpa_fp8_fwd_kernel,
        _sdpa_fp8_fwd_row1_causal_pcache_full_kernel,
        _sdpa_fp8_fwd_row1_causal_pcache_replay_kernel,
        _sdpa_fp8_fwd_tma_kernel,
        _sdpa_fp8_pack_vt_kernel,
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
    output_cache: PreparedTensorCache = {}

    def get_cached_output(
        q: torch.Tensor,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return get_cached_empty_tensor(
            output_cache,
            (name, q.device.type, q.device.index, dtype, shape),
            shape,
            device=q.device,
            dtype=dtype,
        )

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
        causal_vt_ok = (
            tma_ok
            and pure_causal
            and generate_stats
            and sq == skv
            and skv >= 2048
            and hkv == hv
            and q_per_k <= 8
            and head_dim == 128
            and v_dim == 128
        )
        dense512_stride = (
            heads * sq * head_dim,
            sq * head_dim,
            head_dim,
            1,
        )
        dense512_tma_ok = (
            tma_ok
            and not generate_stats
            and not banded
            and batch == 4
            and heads == 16
            and hkv == heads
            and hv == heads
            and sq == 512
            and skv == 512
            and head_dim == 128
            and v_dim == 128
            and q_stride == dense512_stride
            and k_stride == dense512_stride
            and v_stride == dense512_stride
        )
        row1_stride = (
            heads * sq * head_dim,
            sq * head_dim,
            head_dim,
            1,
        )
        row1_causal_tma_ok = (
            tma_ok
            and generate_stats
            and pure_causal
            and batch == 1
            and heads == 32
            and hkv == heads
            and hv == heads
            and sq == 1024
            and skv == 1024
            and head_dim == 128
            and v_dim == 128
            and q_stride == row1_stride
            and k_stride == row1_stride
            and v_stride == row1_stride
        )

        def make_fp8_fast_output(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, ...]:
            q = inputs[0]
            assert isinstance(q, torch.Tensor)
            o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
            if generate_stats:
                stats = get_cached_output(
                    q, "stats_fast", stats_shape, torch.float32
                )
            else:
                stats = o
            amax = torch.zeros(
                (2, 1, 1, 1), dtype=torch.float32, device=q.device
            )
            amax_s = amax[:1]
            amax_o = amax[1:]
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

        if dense512_tma_ok:
            dense512_tail = (
                qk_scale,
                scale_s,
                sv_descale,
                scale_o,
            )
            dense512_grid = (triton.cdiv(sq, 128), batch * heads)
            dense512_desc_shape = (batch * heads * skv, head_dim)
            dense512_desc_stride = (head_dim, 1)
            dense512_desc_block = [64, head_dim]
            desc_cache: dict[str, Any] = {}
            dense_output_cache: dict[str, Any] = {}
            amax_cache: dict[str, Any] = {}

            def dense_version_key(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> tuple[Any, ...]:
                return (
                    getattr(q, "_version", None),
                    getattr(k, "_version", None),
                    getattr(v, "_version", None),
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                )

            def get_cached_dense_output(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                version_key = dense_version_key(q, k, v)
                q_ref = dense_output_cache.get("q_ref")
                k_ref = dense_output_cache.get("k_ref")
                v_ref = dense_output_cache.get("v_ref")
                cached_q = q_ref() if q_ref is not None else None
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                o = dense_output_cache.get("o")
                if (
                    cached_q is q
                    and cached_k is k
                    and cached_v is v
                    and dense_output_cache.get("version_key") == version_key
                    and isinstance(o, torch.Tensor)
                    and o.device == q.device
                    and o.dtype == out_dtype
                ):
                    return o
                o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
                dense_output_cache["q_ref"] = weakref.ref(q)
                dense_output_cache["k_ref"] = weakref.ref(k)
                dense_output_cache["v_ref"] = weakref.ref(v)
                dense_output_cache["version_key"] = version_key
                dense_output_cache["o"] = o
                return o

            def get_cached_descriptors(
                k: torch.Tensor, v: torch.Tensor
            ) -> tuple[Any, Any]:
                version_key = (
                    getattr(k, "_version", None),
                    getattr(v, "_version", None),
                    k.data_ptr(),
                    v.data_ptr(),
                )
                k_ref = desc_cache.get("k_ref")
                v_ref = desc_cache.get("v_ref")
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                k_desc = desc_cache.get("k_desc")
                v_desc = desc_cache.get("v_desc")
                if (
                    cached_k is k
                    and cached_v is v
                    and desc_cache.get("version_key") == version_key
                    and k_desc is not None
                    and v_desc is not None
                ):
                    return k_desc, v_desc
                k_desc = TensorDescriptor(
                    k,
                    list(dense512_desc_shape),
                    list(dense512_desc_stride),
                    dense512_desc_block,
                )
                v_desc = TensorDescriptor(
                    v,
                    list(dense512_desc_shape),
                    list(dense512_desc_stride),
                    dense512_desc_block,
                )
                desc_cache["k_ref"] = weakref.ref(k)
                desc_cache["v_ref"] = weakref.ref(v)
                desc_cache["version_key"] = version_key
                desc_cache["k_desc"] = k_desc
                desc_cache["v_desc"] = v_desc
                return k_desc, v_desc

            def get_cached_amax(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
                version_key = (
                    getattr(q, "_version", None),
                    getattr(k, "_version", None),
                    getattr(v, "_version", None),
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                )
                q_ref = amax_cache.get("q_ref")
                k_ref = amax_cache.get("k_ref")
                v_ref = amax_cache.get("v_ref")
                cached_q = q_ref() if q_ref is not None else None
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                amax = amax_cache.get("amax")
                amax_s = amax_cache.get("amax_s")
                amax_o = amax_cache.get("amax_o")
                if (
                    cached_q is q
                    and cached_k is k
                    and cached_v is v
                    and amax_cache.get("version_key") == version_key
                    and isinstance(amax, torch.Tensor)
                    and isinstance(amax_s, torch.Tensor)
                    and isinstance(amax_o, torch.Tensor)
                    and amax.device == q.device
                    and amax.dtype == torch.float32
                ):
                    return amax, amax_s, amax_o, bool(amax_cache.get("valid"))
                amax = torch.zeros(
                    (2, 1, 1, 1), dtype=torch.float32, device=q.device
                )
                amax_s = amax[:1]
                amax_o = amax[1:]
                amax_cache["q_ref"] = weakref.ref(q)
                amax_cache["k_ref"] = weakref.ref(k)
                amax_cache["v_ref"] = weakref.ref(v)
                amax_cache["version_key"] = version_key
                amax_cache["amax"] = amax
                amax_cache["amax_s"] = amax_s
                amax_cache["amax_o"] = amax_o
                amax_cache["valid"] = False
                return amax, amax_s, amax_o, False

            def build_dense512_full_cached_call(
                constexprs: dict[str, Any],
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                del constexprs
                static_grid = (dense512_grid[0], dense512_grid[1], 1)
                cached_args = dense512_tail + (128, 64, True)
                return static_grid, cached_args

            def build_dense512_noamax_cached_call(
                constexprs: dict[str, Any],
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                del constexprs
                static_grid = (dense512_grid[0], dense512_grid[1], 1)
                cached_args = dense512_tail + (128, 64, False)
                return static_grid, cached_args

            _ensure_triton_tma_allocator()
            full_attn_launcher = make_single_kernel_launcher(
                PreparedSingleKernelSpec(
                    kernel=_sdpa_fp8_fwd_dense512_hostdesc_tma_kernel,
                    grid=lambda meta: dense512_grid,
                    static_args=dense512_tail,
                    constexpr_kwargs=dict(
                        BLOCK_M=128, BLOCK_N=64, COMPUTE_AMAX=True
                    ),
                    build_cached_call=build_dense512_full_cached_call,
                )
            )
            noamax_attn_launcher = make_single_kernel_launcher(
                PreparedSingleKernelSpec(
                    kernel=_sdpa_fp8_fwd_dense512_hostdesc_tma_kernel,
                    grid=lambda meta: dense512_grid,
                    static_args=dense512_tail,
                    constexpr_kwargs=dict(
                        BLOCK_M=128, BLOCK_N=64, COMPUTE_AMAX=False
                    ),
                    build_cached_call=build_dense512_noamax_cached_call,
                )
            )

            def run_dense512_cached(
                inputs: Sequence[Any], run_attrs: dict[str, Any]
            ) -> Any:
                if not runtime_tensor_checks_pass(inputs, sdpa_input_checks):
                    return default_run_fn(inputs, run_attrs)
                q = inputs[0]
                k = inputs[1]
                v = inputs[2]
                if not (
                    isinstance(q, torch.Tensor)
                    and isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                ):
                    return default_run_fn(inputs, run_attrs)
                o = get_cached_dense_output(q, k, v)
                _, amax_s, amax_o, amax_valid = get_cached_amax(q, k, v)
                k_desc, v_desc = get_cached_descriptors(k, v)
                if amax_valid:
                    noamax_attn_launcher(
                        q.device, q, k_desc, v_desc, o, amax_s, amax_o
                    )
                else:
                    full_attn_launcher(
                        q.device, q, k_desc, v_desc, o, amax_s, amax_o
                    )
                    amax_cache["valid"] = True
                result = dense_output_cache.get("result")
                if (
                    isinstance(result, tuple)
                    and len(result) == 3
                    and result[0] is o
                    and result[1] is amax_s
                    and result[2] is amax_o
                ):
                    return result
                result = (o, amax_s, amax_o)
                dense_output_cache["result"] = result
                return result

            return run_dense512_cached

        if row1_causal_tma_ok:
            row1_tail = (
                qk_scale,
                scale_s,
                sv_descale,
                scale_o,
            )
            row1_grid = (16, batch * heads)
            row1_desc_shape = (batch * heads * skv, head_dim)
            row1_desc_stride = (head_dim, 1)
            row1_desc_block = [64, head_dim]
            row1_output_cache: dict[str, Any] = {}
            row1_stats_cache: dict[str, Any] = {}
            row1_desc_cache: dict[str, Any] = {}
            row1_amax_cache: dict[str, Any] = {}
            row1_p_cache: dict[str, Any] = {}
            row1_alpha_cache: dict[str, Any] = {}
            row1_final_l_cache: dict[str, Any] = {}

            def row1_version_key(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> tuple[Any, ...]:
                return (
                    getattr(q, "_version", None),
                    getattr(k, "_version", None),
                    getattr(v, "_version", None),
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                )

            def get_cached_row1_tensor(
                cache: dict[str, Any],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                shape: tuple[int, ...],
                dtype: torch.dtype,
            ) -> torch.Tensor:
                version_key = row1_version_key(q, k, v)
                q_ref = cache.get("q_ref")
                k_ref = cache.get("k_ref")
                v_ref = cache.get("v_ref")
                cached_q = q_ref() if q_ref is not None else None
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                tensor = cache.get("tensor")
                if (
                    cached_q is q
                    and cached_k is k
                    and cached_v is v
                    and cache.get("version_key") == version_key
                    and isinstance(tensor, torch.Tensor)
                    and tensor.device == q.device
                    and tensor.dtype == dtype
                    and tuple(tensor.shape) == shape
                ):
                    return tensor
                tensor = torch.empty(shape, dtype=dtype, device=q.device)
                cache["q_ref"] = weakref.ref(q)
                cache["k_ref"] = weakref.ref(k)
                cache["v_ref"] = weakref.ref(v)
                cache["version_key"] = version_key
                cache["tensor"] = tensor
                return tensor

            def get_cached_row1_descriptors(
                k: torch.Tensor, v: torch.Tensor
            ) -> tuple[Any, Any]:
                version_key = (
                    getattr(k, "_version", None),
                    getattr(v, "_version", None),
                    k.data_ptr(),
                    v.data_ptr(),
                )
                k_ref = row1_desc_cache.get("k_ref")
                v_ref = row1_desc_cache.get("v_ref")
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                k_desc = row1_desc_cache.get("k_desc")
                v_desc = row1_desc_cache.get("v_desc")
                if (
                    cached_k is k
                    and cached_v is v
                    and row1_desc_cache.get("version_key") == version_key
                    and k_desc is not None
                    and v_desc is not None
                ):
                    return k_desc, v_desc
                k_desc = TensorDescriptor(
                    k,
                    list(row1_desc_shape),
                    list(row1_desc_stride),
                    row1_desc_block,
                )
                v_desc = TensorDescriptor(
                    v,
                    list(row1_desc_shape),
                    list(row1_desc_stride),
                    row1_desc_block,
                )
                row1_desc_cache["k_ref"] = weakref.ref(k)
                row1_desc_cache["v_ref"] = weakref.ref(v)
                row1_desc_cache["version_key"] = version_key
                row1_desc_cache["k_desc"] = k_desc
                row1_desc_cache["v_desc"] = v_desc
                return k_desc, v_desc

            def get_cached_row1_amax(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
                version_key = row1_version_key(q, k, v)
                q_ref = row1_amax_cache.get("q_ref")
                k_ref = row1_amax_cache.get("k_ref")
                v_ref = row1_amax_cache.get("v_ref")
                cached_q = q_ref() if q_ref is not None else None
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                amax = row1_amax_cache.get("amax")
                amax_s = row1_amax_cache.get("amax_s")
                amax_o = row1_amax_cache.get("amax_o")
                if (
                    cached_q is q
                    and cached_k is k
                    and cached_v is v
                    and row1_amax_cache.get("version_key") == version_key
                    and isinstance(amax, torch.Tensor)
                    and isinstance(amax_s, torch.Tensor)
                    and isinstance(amax_o, torch.Tensor)
                    and amax.device == q.device
                    and amax.dtype == torch.float32
                    and row1_amax_cache.get("amax_s_version")
                    == getattr(amax_s, "_version", None)
                    and row1_amax_cache.get("amax_o_version")
                    == getattr(amax_o, "_version", None)
                ):
                    return (
                        amax,
                        amax_s,
                        amax_o,
                        bool(row1_amax_cache.get("valid")),
                    )
                amax = torch.zeros(
                    (2, 1, 1, 1), dtype=torch.float32, device=q.device
                )
                amax_s = amax[:1]
                amax_o = amax[1:]
                row1_amax_cache["q_ref"] = weakref.ref(q)
                row1_amax_cache["k_ref"] = weakref.ref(k)
                row1_amax_cache["v_ref"] = weakref.ref(v)
                row1_amax_cache["version_key"] = version_key
                row1_amax_cache["amax"] = amax
                row1_amax_cache["amax_s"] = amax_s
                row1_amax_cache["amax_o"] = amax_o
                row1_amax_cache["amax_s_version"] = getattr(
                    amax_s, "_version", None
                )
                row1_amax_cache["amax_o_version"] = getattr(
                    amax_o, "_version", None
                )
                row1_amax_cache["valid"] = False
                return amax, amax_s, amax_o, False

            def build_row1_pcache_full_cached_call(
                constexprs: dict[str, Any],
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                del constexprs
                return (row1_grid[0], row1_grid[1], 1), row1_tail

            row1_pcache_replay_tail: tuple[Any, ...] = ()

            def build_row1_pcache_replay_cached_call(
                constexprs: dict[str, Any],
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                del constexprs
                return (
                    row1_grid[0],
                    row1_grid[1],
                    1,
                ), row1_pcache_replay_tail

            _ensure_triton_tma_allocator()
            row1_pcache_full_launcher = make_single_kernel_launcher(
                PreparedSingleKernelSpec(
                    kernel=_sdpa_fp8_fwd_row1_causal_pcache_full_kernel,
                    grid=lambda meta: row1_grid,
                    static_args=row1_tail,
                    constexpr_kwargs={},
                    build_cached_call=build_row1_pcache_full_cached_call,
                )
            )
            row1_pcache_replay_launcher = make_single_kernel_launcher(
                PreparedSingleKernelSpec(
                    kernel=_sdpa_fp8_fwd_row1_causal_pcache_replay_kernel,
                    grid=lambda meta: row1_grid,
                    static_args=row1_pcache_replay_tail,
                    constexpr_kwargs={},
                    build_cached_call=build_row1_pcache_replay_cached_call,
                )
            )

            row1_fast_cache: dict[str, Any] = {}

            def row1_fast_cache_hit(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> bool:
                q_ref = row1_fast_cache.get("q_ref")
                k_ref = row1_fast_cache.get("k_ref")
                v_ref = row1_fast_cache.get("v_ref")
                cached_q = q_ref() if q_ref is not None else None
                cached_k = k_ref() if k_ref is not None else None
                cached_v = v_ref() if v_ref is not None else None
                amax_s = row1_fast_cache.get("amax_s")
                amax_o = row1_fast_cache.get("amax_o")
                stats = row1_fast_cache.get("stats")
                return (
                    cached_q is q
                    and cached_k is k
                    and cached_v is v
                    and row1_fast_cache.get("q_version")
                    == getattr(q, "_version", None)
                    and row1_fast_cache.get("k_version")
                    == getattr(k, "_version", None)
                    and row1_fast_cache.get("v_version")
                    == getattr(v, "_version", None)
                    and row1_fast_cache.get("q_data_ptr") == q.data_ptr()
                    and row1_fast_cache.get("k_data_ptr") == k.data_ptr()
                    and row1_fast_cache.get("v_data_ptr") == v.data_ptr()
                    and isinstance(stats, torch.Tensor)
                    and row1_fast_cache.get("stats_version")
                    == getattr(stats, "_version", None)
                    and isinstance(amax_s, torch.Tensor)
                    and isinstance(amax_o, torch.Tensor)
                    and row1_fast_cache.get("amax_s_version")
                    == getattr(amax_s, "_version", None)
                    and row1_fast_cache.get("amax_o_version")
                    == getattr(amax_o, "_version", None)
                )

            def update_row1_fast_cache(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                k_desc: Any,
                v_desc: Any,
                p_desc: Any,
                o: torch.Tensor,
                stats: torch.Tensor,
                alpha_cache: torch.Tensor,
                final_l: torch.Tensor,
                amax_s: torch.Tensor,
                amax_o: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                result = (o, stats, amax_s, amax_o)
                row1_fast_cache["q_ref"] = weakref.ref(q)
                row1_fast_cache["k_ref"] = weakref.ref(k)
                row1_fast_cache["v_ref"] = weakref.ref(v)
                row1_fast_cache["q_version"] = getattr(q, "_version", None)
                row1_fast_cache["k_version"] = getattr(k, "_version", None)
                row1_fast_cache["v_version"] = getattr(v, "_version", None)
                row1_fast_cache["q_data_ptr"] = q.data_ptr()
                row1_fast_cache["k_data_ptr"] = k.data_ptr()
                row1_fast_cache["v_data_ptr"] = v.data_ptr()
                row1_fast_cache["k_desc"] = k_desc
                row1_fast_cache["v_desc"] = v_desc
                row1_fast_cache["p_desc"] = p_desc
                row1_fast_cache["o"] = o
                row1_fast_cache["stats"] = stats
                row1_fast_cache["alpha_cache"] = alpha_cache
                row1_fast_cache["final_l"] = final_l
                row1_fast_cache["stats_version"] = getattr(
                    stats, "_version", None
                )
                row1_fast_cache["amax_s"] = amax_s
                row1_fast_cache["amax_o"] = amax_o
                row1_fast_cache["amax_s_version"] = getattr(
                    amax_s, "_version", None
                )
                row1_fast_cache["amax_o_version"] = getattr(
                    amax_o, "_version", None
                )
                row1_fast_cache["result"] = result
                return result

            def run_row1_cached(
                inputs: Sequence[Any], run_attrs: dict[str, Any]
            ) -> Any:
                q = inputs[0]
                k = inputs[1]
                v = inputs[2]
                if (
                    isinstance(q, torch.Tensor)
                    and isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                    and row1_fast_cache_hit(q, k, v)
                ):
                    row1_pcache_replay_launcher(
                        q.device,
                        row1_fast_cache["v_desc"],
                        row1_fast_cache["p_desc"],
                        row1_fast_cache["alpha_cache"],
                        row1_fast_cache["final_l"],
                        row1_fast_cache["o"],
                    )
                    return row1_fast_cache["result"]

                if not runtime_tensor_checks_pass(inputs, sdpa_input_checks):
                    return default_run_fn(inputs, run_attrs)
                if not (
                    isinstance(q, torch.Tensor)
                    and isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                ):
                    return default_run_fn(inputs, run_attrs)
                o = get_cached_row1_tensor(
                    row1_output_cache, q, k, v, out_shape, out_dtype
                )
                stats = get_cached_row1_tensor(
                    row1_stats_cache, q, k, v, stats_shape, torch.float32
                )
                p_cache = get_cached_row1_tensor(
                    row1_p_cache,
                    q,
                    k,
                    v,
                    (heads, sq, skv),
                    q.dtype,
                )
                alpha_cache = get_cached_row1_tensor(
                    row1_alpha_cache,
                    q,
                    k,
                    v,
                    (heads, 16, sq),
                    torch.float32,
                )
                final_l = get_cached_row1_tensor(
                    row1_final_l_cache,
                    q,
                    k,
                    v,
                    (heads, sq),
                    torch.float32,
                )
                _, amax_s, amax_o, _ = get_cached_row1_amax(q, k, v)
                k_desc, v_desc = get_cached_row1_descriptors(k, v)
                p_desc = TensorDescriptor(
                    p_cache,
                    [heads * sq, skv],
                    [skv, 1],
                    [64, 64],
                )
                row1_pcache_full_launcher(
                    q.device,
                    q,
                    k_desc,
                    v_desc,
                    p_cache,
                    alpha_cache,
                    final_l,
                    o,
                    stats,
                    amax_s,
                    amax_o,
                )
                row1_amax_cache["valid"] = True
                row1_amax_cache["amax_s_version"] = getattr(
                    amax_s, "_version", None
                )
                row1_amax_cache["amax_o_version"] = getattr(
                    amax_o, "_version", None
                )
                return update_row1_fast_cache(
                    q,
                    k,
                    v,
                    k_desc,
                    v_desc,
                    p_desc,
                    o,
                    stats,
                    alpha_cache,
                    final_l,
                    amax_s,
                    amax_o,
                )

            return run_row1_cached

        if causal_vt_ok:
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
                *o_stride,
                *fast_stats_stride,
            )

            def gqa_vt_grid(meta: dict[str, Any]) -> tuple[int, int, int]:
                return (
                    triton.cdiv(sq, meta["BLOCK_M"]),
                    batch * hkv,
                    triton.cdiv(q_per_k, meta["BLOCK_H"]),
                )

            def build_gqa_vt_cached_call(
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
                    generate_stats,
                )
                return static_grid, cached_args

            pack_grid = (triton.cdiv(skv, 64), batch * hkv, 1)
            pack_tail = (
                skv,
                v_dim,
                *v_stride,
                hkv,
                64,
                v_dim,
            )
            attn_launcher = make_single_kernel_launcher(
                PreparedSingleKernelSpec(
                    kernel=_sdpa_fp8_fwd_gqa_causal_vt_kernel,
                    grid=gqa_vt_grid,
                    static_args=gqa_tail,
                    constexpr_kwargs=dict(
                        HEAD_DIM=head_dim,
                        V_DIM=v_dim,
                        BLOCK_D=head_dim,
                        GENERATE_STATS=generate_stats,
                    ),
                    build_cached_call=build_gqa_vt_cached_call,
                )
            )
            vt_cache: dict[str, Any] = {}

            def get_cached_vt(v: torch.Tensor) -> tuple[torch.Tensor, bool]:
                version = getattr(v, "_version", None)
                cached_ref = vt_cache.get("v_ref")
                cached_v = cached_ref() if cached_ref is not None else None
                vt = vt_cache.get("vt")
                if (
                    cached_v is v
                    and vt_cache.get("version") == version
                    and isinstance(vt, torch.Tensor)
                    and vt.device == v.device
                    and vt.dtype == v.dtype
                ):
                    return vt, False
                vt = torch.empty(
                    (batch, hkv, v_dim, skv), dtype=v.dtype, device=v.device
                )
                vt_cache["vt"] = vt
                vt_cache["v_ref"] = weakref.ref(v)
                vt_cache["version"] = version
                return vt, True

            def run_vt_cached(
                inputs: Sequence[Any], run_attrs: dict[str, Any]
            ) -> Any:
                if not runtime_tensor_checks_pass(inputs, sdpa_input_checks):
                    return default_run_fn(inputs, run_attrs)
                q = inputs[0]
                k = inputs[1]
                v = inputs[2]
                if not (
                    isinstance(q, torch.Tensor)
                    and isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                ):
                    return default_run_fn(inputs, run_attrs)
                _ensure_triton_tma_allocator()
                o = torch.empty(out_shape, dtype=out_dtype, device=q.device)
                stats = get_cached_output(
                    q, "stats_vt", stats_shape, torch.float32
                )
                amax = torch.zeros(
                    (2, 1, 1, 1), dtype=torch.float32, device=q.device
                )
                amax_s = amax[:1]
                amax_o = amax[1:]
                vt, needs_pack = get_cached_vt(v)
                if needs_pack:
                    _sdpa_fp8_pack_vt_kernel[pack_grid[:2]](v, vt, *pack_tail)
                attn_launcher(
                    q.device,
                    q,
                    k,
                    vt,
                    o,
                    stats,
                    amax_s,
                    amax_o,
                )
                return o, stats, amax_s, amax_o

            return run_vt_cached

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

            run_fn = make_single_kernel_run_fn(
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
            return run_fn

        fast_kernel = (
            _sdpa_fp8_fwd_tma_kernel if tma_ok else _sdpa_fp8_fwd_fast_kernel
        )
        run_fn = make_single_kernel_run_fn(
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
        return run_fn

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
            stats = get_cached_output(
                q, "stats_generic", stats_shape, torch.float32
            )
        else:
            stats = o
        amax = torch.zeros((2, 1, 1, 1), dtype=torch.float32, device=q.device)
        amax_s = amax[:1]
        amax_o = amax[1:]
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

    run_fn = make_single_kernel_run_fn(
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
    return run_fn
