import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("elu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def elu_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    n_elements,  # 元素总数
    alpha,  # ELU alpha
    BLOCK_SIZE: tl.constexpr,
    USE_FP32_MATH: tl.constexpr,  # fp16/bf16/fp32 走 fp32 exp 计算
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if USE_FP32_MATH:
        x_math = x.to(tl.float32)
    else:
        x_math = x

    # ELU:
    # y = x,                        if x > 0
    # y = alpha * (exp(x) - 1),     otherwise
    y_math = tl.where(x_math > 0, x_math, alpha * (tl.exp(x_math) - 1.0))

    tl.store(y_ptr + offsets, y_math.to(y_ptr.dtype.element_ty), mask=mask)


def elu(
    input: torch.Tensor,
    alpha: float = 1.0,
    inplace: bool = False,
) -> torch.Tensor:
    logger.debug("FLAG_DNN ELU")

    alpha = float(alpha)

    orig_input = input
    need_copy_back = inplace and (not input.is_contiguous())

    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()

    if n_elements == 0:
        if inplace:
            return orig_input
        return torch.empty_like(input)

    if inplace:
        y = input
    else:
        y = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    use_fp32_math = input.dtype != torch.float64

    with torch_device_fn.device(input.device):
        elu_kernel[grid](
            input,
            y,
            n_elements,
            alpha,
            USE_FP32_MATH=use_fp32_math,
        )

    if need_copy_back:
        orig_input.copy_(y)
        return orig_input

    return y
