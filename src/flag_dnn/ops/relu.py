import logging
from typing import Union

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
    configs=runtime.get_tuned_config("relu"),
    key=["n"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + idx, y, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("FLAG_DNN RELU")

    assert x.is_contiguous(), "x must be contiguous"

    n = x.numel()
    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        relu_kernel[grid](x, y, n)

    return y
