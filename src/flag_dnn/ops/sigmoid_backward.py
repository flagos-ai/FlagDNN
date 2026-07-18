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
    has_same_dense_flat_layout,
)


logger = logging.getLogger(__name__)


_SIGMOID_BACKWARD_CONFIGS = runtime.get_tuned_config("sigmoid_backward")
_PORTABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@libtuner(
    configs=_SIGMOID_BACKWARD_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def sigmoid_backward_kernel(
    loss_ptr,
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    loss = tl.load(loss_ptr + offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.sigmoid(x)
    dx = loss * y * (1.0 - y)

    tl.store(out_ptr + offsets, dx.to(out_ptr.dtype.element_ty), mask=mask)


def sigmoid_backward(
    loss: torch.Tensor,
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    logger.debug("FLAG_DNN SIGMOID_BACKWARD")

    if loss.dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            "flag_dnn sigmoid_backward does not support "
            f"loss dtype={loss.dtype} on device={runtime.device.name}"
        )
    if input.dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            "flag_dnn sigmoid_backward does not support input "
            f"dtype={input.dtype} on device={runtime.device.name}"
        )
    if loss.device.type != runtime.device.name:
        raise RuntimeError(
            "flag_dnn sigmoid_backward expected loss on "
            f"{runtime.device.name}, got device={loss.device}"
        )
    if input.device.type != runtime.device.name:
        raise RuntimeError(
            "flag_dnn sigmoid_backward expected input on "
            f"{runtime.device.name}, got device={input.device}"
        )
    if loss.device != input.device:
        raise RuntimeError(
            f"Expected loss and input on same device, got {loss.device} and "
            f"{input.device}"
        )
    if not has_same_dense_flat_layout(loss, input):
        raise NotImplementedError(
            "flag_dnn sigmoid_backward currently requires loss and input to "
            "share a contiguous or NHWC channels-last dense flat layout"
        )

    out_dtype = (
        out.dtype if out is not None else torch.result_type(loss, input)
    )
    if out is None:
        out = empty_like_preserve_dense_layout(input, out_dtype)
    else:
        if not can_use_flat_output(out, input):
            raise NotImplementedError(
                "flag_dnn sigmoid_backward currently requires out to share "
                "input's dense flat layout"
            )

    n_elements = input.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        sigmoid_backward_kernel[grid](loss, input, out, n_elements)

    return out
