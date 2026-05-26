import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.ops.binary import (
    empty_like_preserve_dense_layout,
    is_dense_flat_tensor,
)
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("silu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def swish_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(beta * x)

    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def swish(
    input: torch.Tensor,
    swish_beta: float | None = None,
    *,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    logger.debug("FLAG_DNN SWISH")
    del compute_data_type, name

    if not is_dense_flat_tensor(input):
        raise NotImplementedError(
            "flag_dnn swish currently supports contiguous or NHWC "
            "channels-last input only"
        )

    n_elements = input.numel()
    if n_elements == 0:
        return empty_like_preserve_dense_layout(input, input.dtype)

    y = empty_like_preserve_dense_layout(input, input.dtype)
    beta = 1.0 if swish_beta is None else float(swish_beta)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        swish_kernel[grid](input, y, n_elements, beta)

    return y
