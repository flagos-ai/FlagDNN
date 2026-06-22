import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.ops.mm import mm
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner


_MATMUL_CONFIGS = runtime.get_tuned_config("matmul")


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
    FAST_FP32_TO_FP16: tl.constexpr,
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
        if FAST_FP32_TO_FP16:
            a = a.to(tl.float16)
            b = b.to(tl.float16)
        acc += tl.dot(a, b, input_precision="tf32")
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

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            batch,
        )

    with torch_device_fn.device(A.device):
        _batched_matmul_kernel[grid](
            A,
            B,
            C,
            m,
            n,
            k,
            FAST_FP32_TO_FP16=(
                A.dtype == torch.float32 and m >= 512 and n >= 512 and k >= 512
            ),
        )
    return C


_BATCHED_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
    A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    if A.dtype not in _BATCHED_TRITON_DTYPES:
        raise NotImplementedError(
            "flag_dnn batched matmul supports fp16, bf16, and fp32 inputs"
        )

    batch_shape = _broadcast_batch_shape(A, B)
    m = int(A.shape[-2])
    n = int(B.shape[-1])
    A3 = _as_batched_contiguous(A, batch_shape)
    B3 = _as_batched_contiguous(B, batch_shape)
    C3 = _batched_matmul_3d(A3, B3)
    return C3.reshape(batch_shape + (m, n))


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    compute_data_type=None,
    padding: float = 0.0,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, padding, name
    _check_matmul_inputs(A, B)

    if A.dim() == 2 and B.dim() == 2:
        return mm(A, B)

    if (
        A.dim() == 3
        and B.dim() == 3
        and A.shape[0] == B.shape[0]
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        return _batched_matmul_3d(A, B)

    return _batched_matmul_broadcast(A, B)
