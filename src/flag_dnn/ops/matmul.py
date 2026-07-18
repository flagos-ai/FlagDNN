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

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.ops.mm import mm
from flag_dnn.ops.matmul_sm90 import launch_sm90_matmul_if_supported
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils.device_info import (
    get_device_capability_for,
    get_device_info,
)


_MATMUL_CONFIGS = runtime.get_tuned_config("matmul")
_MATMUL_PERSISTENT_CONFIGS = runtime.get_tuned_config("matmul_persistent")

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_MATMUL_TRITON_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    *_FP8_DTYPES,
)
_MATMUL_OUT_DTYPES = _MATMUL_TRITON_DTYPES
_MATMUL_OUT_DTYPE_ALIASES = {
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "e4m3": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "fp8_e5m2": torch.float8_e5m2,
    "e5m2": torch.float8_e5m2,
}


def _resolve_matmul_compute_mode(
    input_dtype: torch.dtype, compute_data_type
) -> str:
    if compute_data_type is None or compute_data_type is torch.float32:
        return "ieee" if input_dtype == torch.float32 else "float32"

    key = str(compute_data_type).lower().replace("torch.", "")
    if key in {"float", "float32", "fp32"}:
        return "ieee" if input_dtype == torch.float32 else "float32"
    if key == "tf32" and input_dtype == torch.float32:
        return "tf32"
    if key == "fast_float_for_fp8" and input_dtype in _FP8_DTYPES:
        return "fast_float_for_fp8"
    raise RuntimeError(
        f"unsupported matmul compute_data_type={compute_data_type!r} "
        f"for input dtype={input_dtype}"
    )


def _resolve_matmul_out_dtype(
    input_dtype: torch.dtype, out_dtype
) -> torch.dtype:
    if out_dtype is None:
        return input_dtype
    if out_dtype in _MATMUL_OUT_DTYPES:
        return out_dtype
    if isinstance(out_dtype, str):
        key = out_dtype.lower().replace("torch.", "")
        resolved = _MATMUL_OUT_DTYPE_ALIASES.get(key)
        if resolved is not None:
            return resolved
    raise RuntimeError(
        "matmul out_dtype must be fp16, bf16, fp32, fp8_e4m3, or "
        f"fp8_e5m2, got {out_dtype!r}"
    )


@triton.jit
def _round_fp32_to_tf32(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = bits + 0xFFF + ((bits >> 13) & 1)
    rounded &= 0xFFFFE000
    is_special = (bits & 0x7F800000) == 0x7F800000
    rounded = tl.where(is_special, bits, rounded)
    return rounded.to(tl.float32, bitcast=True)


@libentry()
@libtuner(
    configs=_MATMUL_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ROUND_F32_TO_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    elif GROUP_M >= (M + BLOCK_M - 1) // BLOCK_M:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

    a_base = a_ptr + bid * M * K
    b_base = b_ptr + bid * K * N
    c_base = c_ptr + bid * M * N
    a_block = tl.make_block_ptr(
        base=a_base,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_base,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K, BLOCK_K):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            a = tl.load(a_block, boundary_check=())
        else:
            a = tl.load(
                a_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            b = tl.load(b_block, boundary_check=())
        else:
            b = tl.load(
                b_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )
        if ROUND_F32_TO_TF32:
            a = _round_fp32_to_tf32(a)
            b = _round_fp32_to_tf32(b)
            acc += tl.dot(a, b, input_precision="tf32")
        else:
            acc += tl.dot(a, b)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=c_base,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    c = acc.to(c_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(c_block, c, boundary_check=())
    else:
        tl.store(c_block, c, boundary_check=(0, 1))


@triton.jit
def _batched_matmul_fp32_ieee_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid - pid_m * num_pid_n
    elif GROUP_M >= num_pid_m:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid - group_id * num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + bid * M * K + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + bid * K * N + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K, BLOCK_K):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
        offs_k += BLOCK_K

    c_ptrs = c_ptr + bid * M * N + offs_m[:, None] * N + offs_n[None, :]
    c = acc.to(c_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(c_ptrs, c)
    else:
        tl.store(
            c_ptrs,
            c,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@libentry()
@libtuner(
    configs=_MATMUL_PERSISTENT_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def _batched_matmul_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * BATCH
    tile_id = start_pid

    while tile_id < total_tiles:
        bid = tile_id // tiles_per_batch
        pid = tile_id - bid * tiles_per_batch
        if GROUP_M == 1:
            pid_m = pid // num_pid_n
            pid_n = pid - pid_m * num_pid_n
        elif GROUP_M >= num_pid_m:
            pid_m = pid % num_pid_m
            pid_n = pid // num_pid_m
        else:
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            pid_in_group = pid - group_id * num_pid_in_group
            pid_m = first_pid_m + (pid_in_group % group_size_m)
            pid_n = pid_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + bid * M * K + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + bid * K * N + offs_k[:, None] * N + offs_n[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, K, BLOCK_K):
            if M % BLOCK_M == 0 and K % BLOCK_K == 0:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
            if K % BLOCK_K == 0 and N % BLOCK_N == 0:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                )
            acc += tl.dot(a, b, input_precision="tf32")
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
            offs_k += BLOCK_K

        c_ptrs = c_ptr + bid * M * N + offs_m[:, None] * N + offs_n[None, :]
        c = acc.to(c_ptr.dtype.element_ty)
        if M % BLOCK_M == 0 and N % BLOCK_N == 0:
            tl.store(c_ptrs, c)
        else:
            tl.store(
                c_ptrs,
                c,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
        tile_id += NUM_SMS


def _batched_matmul_3d_out(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    compute_mode: str | None = None,
) -> torch.Tensor:
    if not A.is_contiguous() or not B.is_contiguous():
        raise NotImplementedError(
            "flag_dnn matmul batched path requires contiguous inputs"
        )
    if not C.is_contiguous():
        raise NotImplementedError(
            "flag_dnn matmul batched path requires contiguous output"
        )
    if A.device != B.device:
        raise RuntimeError(
            "matmul: input tensors must be on the same device, "
            f"but got {A.device} and {B.device}"
        )
    if A.dtype != B.dtype:
        raise RuntimeError(
            "expected mat1 and mat2 to have the same dtype, but got: "
            f"{A.dtype} != {B.dtype}"
        )
    if A.dtype not in _MATMUL_TRITON_DTYPES:
        raise NotImplementedError(
            f"flag_dnn batched matmul does not support dtype={A.dtype}"
        )
    batch, m, k = A.shape
    b_batch, b_k, n = B.shape
    if batch != b_batch or k != b_k:
        raise RuntimeError(
            f"matmul shape mismatch: {tuple(A.shape)} x {tuple(B.shape)}"
        )
    expected = (batch, m, n)
    if (
        tuple(C.shape) != expected
        or C.dtype not in _MATMUL_OUT_DTYPES
        or C.device != A.device
    ):
        raise RuntimeError("matmul output buffer shape/dtype/device mismatch")
    if C.numel() == 0:
        return C

    if compute_mode is None:
        compute_mode = "ieee" if A.dtype == torch.float32 else "float32"

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            batch,
        )

    exact_projection = batch == 16 and m == 2048 and n == 2048 and k == 512
    exact_long_k = batch == 32 and m == 1024 and n == 1024 and k == 4096

    with torch_device_fn.device(A.device):
        if launch_sm90_matmul_if_supported(
            A,
            B,
            C,
            compute_mode=compute_mode,
            capability=get_device_capability_for(A.device),
        ):
            return C
        if (
            exact_long_k
            and A.dtype in (torch.float16, torch.bfloat16)
            and C.dtype == A.dtype
            and compute_mode == "float32"
        ):
            fixed_grid = (triton.cdiv(m, 128) * triton.cdiv(n, 256), batch)
            _batched_matmul_kernel.fn.fn[fixed_grid](
                A,
                B,
                C,
                m,
                n,
                k,
                BLOCK_M=128,
                BLOCK_N=256,
                BLOCK_K=64,
                GROUP_M=4,
                ROUND_F32_TO_TF32=False,
                num_warps=8,
                num_stages=3,
            )
        elif A.dtype == torch.float32 and compute_mode == "ieee":
            block_m = 16 if m <= 16 else 32
            block_n = 32 if n <= 32 else 64
            block_k = 32 if k <= 64 else 64
            ieee_grid = (
                triton.cdiv(m, block_m) * triton.cdiv(n, block_n),
                batch,
            )
            _batched_matmul_fp32_ieee_kernel[ieee_grid](
                A,
                B,
                C,
                m,
                n,
                k,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=1,
                num_warps=4,
                num_stages=3,
            )
        elif (
            exact_projection
            and A.dtype in (torch.float16, torch.bfloat16)
            and C.dtype == A.dtype
            and compute_mode == "float32"
        ):
            fixed_grid = (triton.cdiv(m, 128) * triton.cdiv(n, 256), batch)
            _batched_matmul_kernel.fn.fn[fixed_grid](
                A,
                B,
                C,
                m,
                n,
                k,
                BLOCK_M=128,
                BLOCK_N=256,
                BLOCK_K=64,
                GROUP_M=16,
                ROUND_F32_TO_TF32=False,
                num_warps=8,
                num_stages=4,
            )
        elif (
            A.dtype in (torch.float16, torch.bfloat16)
            and C.dtype == A.dtype
            and compute_mode == "float32"
            and m == 512
            and n == 512
            and k == 512
        ):
            num_sms = get_device_info().sm_count

            def persistent_grid(meta):
                tiles = (
                    batch
                    * triton.cdiv(m, meta["BLOCK_M"])
                    * triton.cdiv(n, meta["BLOCK_N"])
                )
                return (min(tiles, num_sms),)

            _batched_matmul_persistent_kernel[persistent_grid](
                A,
                B,
                C,
                m,
                n,
                k,
                batch,
                num_sms,
            )
        else:
            _batched_matmul_kernel[grid](
                A,
                B,
                C,
                m,
                n,
                k,
                ROUND_F32_TO_TF32=(
                    A.dtype == torch.float32 and compute_mode == "tf32"
                ),
            )
    return C


def _batched_matmul_3d(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    compute_mode: str | None = None,
) -> torch.Tensor:
    batch, m, _ = A.shape
    n = int(B.shape[2])
    result_dtype = A.dtype if out_dtype is None else out_dtype
    C = torch.empty((batch, m, n), device=A.device, dtype=result_dtype)
    return _batched_matmul_3d_out(A, B, C, compute_mode=compute_mode)


_BATCHED_TRITON_DTYPES = _MATMUL_TRITON_DTYPES


def _prod(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _check_matmul_inputs(A: torch.Tensor, B: torch.Tensor) -> None:
    if A.dim() < 2 or B.dim() < 2:
        raise NotImplementedError(
            "flag_dnn matmul requires tensors with rank >= 2"
        )
    if A.shape[-1] != B.shape[-2]:
        raise RuntimeError(
            "mat1 and mat2 shapes cannot be multiplied "
            f"({A.shape[-2]}x{A.shape[-1]} and "
            f"{B.shape[-2]}x{B.shape[-1]})"
        )
    if A.device != B.device:
        raise RuntimeError(
            "matmul: input tensors must be on the same device, "
            f"but got {A.device} and {B.device}"
        )
    if A.dtype != B.dtype:
        raise RuntimeError(
            "expected mat1 and mat2 to have the same dtype, but got: "
            f"{A.dtype} != {B.dtype}"
        )
    if A.layout != torch.strided or B.layout != torch.strided:
        raise NotImplementedError(
            "flag_dnn matmul supports dense strided tensors only"
        )


def _broadcast_batch_shape(
    A: torch.Tensor, B: torch.Tensor
) -> tuple[int, ...]:
    try:
        return tuple(
            torch.broadcast_shapes(tuple(A.shape[:-2]), tuple(B.shape[:-2]))
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "matmul batch dimensions are not broadcastable: "
            f"{tuple(A.shape)} x {tuple(B.shape)}"
        ) from exc


def _as_batched_contiguous(
    tensor: torch.Tensor, batch_shape: tuple[int, ...]
) -> torch.Tensor:
    matrix_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
    batch_count = _prod(batch_shape)
    target_shape = batch_shape + matrix_shape

    if tuple(tensor.shape[:-2]) == batch_shape:
        batched = tensor.reshape(batch_count, *matrix_shape)
    else:
        batched = tensor.expand(target_shape).reshape(
            batch_count, *matrix_shape
        )
    return batched if batched.is_contiguous() else batched.contiguous()


def _batched_matmul_broadcast(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    compute_mode: str,
) -> torch.Tensor:
    if A.dtype not in _BATCHED_TRITON_DTYPES:
        raise NotImplementedError(
            "flag_dnn batched matmul supports fp16, bf16, fp32, "
            "fp8_e4m3, and fp8_e5m2 inputs"
        )

    batch_shape = _broadcast_batch_shape(A, B)
    m = int(A.shape[-2])
    n = int(B.shape[-1])
    A3 = _as_batched_contiguous(A, batch_shape)
    B3 = _as_batched_contiguous(B, batch_shape)
    C3 = _batched_matmul_3d(
        A3,
        B3,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )
    return C3.reshape(batch_shape + (m, n))


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    compute_data_type=None,
    out_dtype=None,
    padding: float = 0.0,
    name: str = "",
) -> torch.Tensor:
    del padding, name
    _check_matmul_inputs(A, B)

    if A.dtype not in _MATMUL_TRITON_DTYPES:
        if out_dtype is not None:
            raise RuntimeError(
                f"matmul out_dtype is unsupported for input dtype={A.dtype}"
            )
        if A.dim() == 2 and B.dim() == 2:
            return mm(A, B)
        return _batched_matmul_broadcast(
            A,
            B,
            out_dtype=A.dtype,
            compute_mode="float32",
        )

    compute_mode = _resolve_matmul_compute_mode(A.dtype, compute_data_type)
    result_dtype = _resolve_matmul_out_dtype(A.dtype, out_dtype)

    if A.dim() == 2 and B.dim() == 2:
        if (
            compute_mode != "tf32"
            and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and result_dtype in (A.dtype, torch.float32)
            and not (A.dtype == torch.float32 and result_dtype != A.dtype)
        ):
            mm_out_dtype = result_dtype if result_dtype != A.dtype else None
            return mm(A, B, out_dtype=mm_out_dtype)
        A3 = A.contiguous().reshape(1, A.shape[0], A.shape[1])
        B3 = B.contiguous().reshape(1, B.shape[0], B.shape[1])
        return _batched_matmul_3d(
            A3,
            B3,
            out_dtype=result_dtype,
            compute_mode=compute_mode,
        ).reshape(A.shape[0], B.shape[1])

    if (
        A.dim() == 3
        and B.dim() == 3
        and A.shape[0] == B.shape[0]
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        return _batched_matmul_3d(
            A,
            B,
            out_dtype=result_dtype,
            compute_mode=compute_mode,
        )

    return _batched_matmul_broadcast(
        A,
        B,
        out_dtype=result_dtype,
        compute_mode=compute_mode,
    )
