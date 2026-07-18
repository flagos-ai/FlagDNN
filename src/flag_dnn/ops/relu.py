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
    configs=runtime.get_tuned_config("relu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def relu_1d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    lower_clip,
    upper_clip,
    HAS_UPPER_CLIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    lower = x * 0.0 + lower_clip
    slope = x * 0.0 + negative_slope
    out = tl.where(x < lower, lower + slope * (x - lower), x)
    if HAS_UPPER_CLIP:
        out = tl.minimum(out, upper_clip)

    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def relu(
    input: torch.Tensor,
    inplace: bool = False,
    negative_slope: float | None = None,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
    *,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    logger.debug("FLAG_DNN RELU")
    del compute_data_type, name

    if not is_dense_flat_tensor(input):
        raise NotImplementedError(
            "flag_dnn relu currently supports contiguous or NHWC "
            "channels-last input only"
        )

    n_elements = input.numel()
    if n_elements == 0:
        if inplace:
            return input
        return empty_like_preserve_dense_layout(input, input.dtype)

    out = (
        input
        if inplace
        else empty_like_preserve_dense_layout(input, input.dtype)
    )

    slope = 0.0 if negative_slope is None else float(negative_slope)
    lower = 0.0 if lower_clip is None else float(lower_clip)
    upper = 0.0 if upper_clip is None else float(upper_clip)
    has_upper = upper_clip is not None

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        relu_1d_kernel[grid](
            input,
            out,
            n_elements,
            slope,
            lower,
            upper,
            HAS_UPPER_CLIP=has_upper,
        )

    return out
