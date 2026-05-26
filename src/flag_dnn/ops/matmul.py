import torch
import triton
import triton.language as tl

from flag_dnn.ops.mm import mm
from flag_dnn.runtime import torch_device_fn


@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    batch,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + bid * stride_ab
    b_base = b_ptr + bid * stride_bb
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_base + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c_base = c_ptr + bid * stride_cb
    tl.store(
        c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _batched_matmul_3d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not A.is_contiguous() or not B.is_contiguous():
        raise NotImplementedError(
            "flag_dnn matmul batched path requires contiguous inputs"
        )
    batch, m, k = A.shape
    b_batch, b_k, n = B.shape
    if batch != b_batch or k != b_k:
        raise RuntimeError(
            f"matmul shape mismatch: {tuple(A.shape)} x {tuple(B.shape)}"
        )
    C = torch.empty((batch, m, n), device=A.device, dtype=A.dtype)
    if C.numel() == 0:
        return C

    block_m = 16 if m <= 16 else 32
    block_n = 32 if n <= 32 else 64
    block_k = 32 if k <= 64 else 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), batch)
    with torch_device_fn.device(A.device):
        _batched_matmul_kernel[grid](
            A,
            B,
            C,
            batch,
            m,
            n,
            k,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
        )
    return C


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    compute_data_type=None,
    padding: float = 0.0,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    if float(padding) != 0.0:
        raise NotImplementedError("flag_dnn matmul does not support padding")
    if A.dim() == 2 and B.dim() == 2:
        return mm(A, B)
    if A.dim() == 3 and B.dim() == 3 and A.shape[0] == B.shape[0]:
        return _batched_matmul_3d(A, B)
    if A.dim() < 2 or B.dim() < 2:
        raise NotImplementedError(
            "flag_dnn matmul requires tensors with rank >= 2"
        )
    # TODO: implement the remaining torch.matmul broadcasting cases in Triton.
    raise NotImplementedError(
        "flag_dnn matmul currently supports 2D x 2D and matching-batch "
        "3D x 3D inputs only"
    )
