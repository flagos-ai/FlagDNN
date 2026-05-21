import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.type_utils import is_bool_dtype
from flag_dnn.ops.binary import (
    can_use_flat_output,
    empty_like_preserve_dense_layout,
    is_dense_flat_tensor,
)


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("abs"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def abs_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 计算绝对值 (底层直接处理符号位，无需提升至 f32)
    res = tl.abs(x)

    # 存回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def abs(
    input: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN ABS")

    if is_bool_dtype(input.dtype):
        raise NotImplementedError(
            f"\"abs\" not implemented for '{input.dtype}'"
        )

    if not is_dense_flat_tensor(input):
        raise NotImplementedError(
            "flag_dnn abs currently supports contiguous or NHWC "
            "channels-last input only"
        )

    out_dtype = input.dtype
    out_shape = input.shape

    # 输出内存分配
    if out is None:
        out = empty_like_preserve_dense_layout(input, out_dtype)
    else:
        assert (
            out.shape == out_shape
        ), f"out shape {out.shape} mismatch with input shape {out_shape}"
        out_dtype = out.dtype
        if not can_use_flat_output(out, input):
            raise NotImplementedError(
                "flag_dnn abs currently requires out to share input's "
                "dense flat layout"
            )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        abs_kernel[grid](input, out, n_elements)

    return out
