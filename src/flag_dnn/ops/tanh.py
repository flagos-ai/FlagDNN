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

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.triton_lang_helper import tl_extra_shim as libdevice

logger = logging.getLogger(__name__)


_TANH_CONFIGS = runtime.get_tuned_config("tanh")
_PORTABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_dense_flat_tensor(input: torch.Tensor) -> bool:
    return (
        input.is_contiguous()
        or (
            input.ndim == 4
            and input.is_contiguous(memory_format=torch.channels_last)
        )
        or (
            input.ndim == 5
            and input.is_contiguous(memory_format=torch.channels_last_3d)
        )
    )


@libentry()
@libtuner(
    configs=_TANH_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = libdevice.tanh(x)

    tl.store(
        y_ptr + offsets,
        y.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def tanh(input: torch.Tensor) -> torch.Tensor:
    logger.debug("FLAG_DNN TANH")

    if input.dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            f"flag_dnn tanh does not support dtype={input.dtype} "
            f"on device={runtime.device.name}"
        )
    if input.device.type != runtime.device.name:
        raise RuntimeError(
            f"flag_dnn tanh expected a {runtime.device.name} tensor, "
            f"got device={input.device}"
        )

    if not _is_dense_flat_tensor(input):
        input = input.contiguous()

    n_elements = input.numel()
    if n_elements == 0:
        return torch.empty_like(input)

    y = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        tanh_kernel[grid](input, y, n_elements)

    return y
