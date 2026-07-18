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

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.ops.binary import (
    can_use_flat_output,
    empty_like_preserve_dense_layout,
    is_dense_flat_tensor,
)

logger = logging.getLogger(__name__)


_SIGMOID_CONFIGS = runtime.get_tuned_config("sigmoid")
_SIGMOID_FP64_CONFIGS = runtime.get_tuned_config("sigmoid")

# Triton's exp import path differs across versions.
if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 0):
    try:
        from triton.language.extra.libdevice import exp as triton_exp
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import exp as triton_exp
else:
    from triton.language.math import exp as triton_exp


@libentry()
@libtuner(
    configs=_SIGMOID_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.sigmoid(x)

    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_SIGMOID_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def sigmoid_fp64_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float64)
    y = 1.0 / (1.0 + triton_exp(-x))

    tl.store(y_ptr + offsets, y, mask=mask)


def sigmoid(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    logger.debug("FLAG_DNN SIGMOID")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn sigmoid does not support dtype={input.dtype}"
        )
    if not input.is_cuda:
        raise NotImplementedError(
            "flag_dnn sigmoid Triton implementation requires CUDA input"
        )
    if not is_dense_flat_tensor(input):
        raise NotImplementedError(
            "flag_dnn sigmoid currently supports contiguous or NHWC "
            "channels-last input only"
        )

    out_shape = input.shape
    if out is None:
        out = empty_like_preserve_dense_layout(input, input.dtype)
    else:
        if out.shape != out_shape:
            raise RuntimeError(
                f"out shape {out.shape} mismatch with input shape {out_shape}"
            )
        if out.dtype != input.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {input.dtype}, "
                f"but got {out.dtype} instead"
            )
        if out.device != input.device:
            raise RuntimeError(
                f"Expected out tensor to be on {input.device}, "
                f"but got {out.device} instead"
            )
        if not can_use_flat_output(out, input):
            raise NotImplementedError(
                "flag_dnn sigmoid currently requires out to share input's "
                "dense flat layout"
            )

    n_elements = input.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            sigmoid_fp64_kernel[grid](input, out, n_elements)
        else:
            sigmoid_kernel[grid](input, out, n_elements)

    return out
