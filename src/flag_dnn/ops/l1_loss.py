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


logger = logging.getLogger(__name__)

# ATen reduction codes
_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("abs"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def l1_loss_elementwise_kernel(
    input_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    tgt = tl.load(target_ptr + offsets, mask=mask).to(tl.float32)
    diff = inp - tgt
    res = tl.abs(diff)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def l1_loss_reduce_kernel(
    input_ptr,
    target_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Each block computes partial sum of absolute differences."""
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tgt = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = tl.abs(inp - tgt)
    partial = tl.sum(diff, axis=0)
    tl.store(partial_ptr + pid, partial)


def l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = _REDUCTION_MEAN,
) -> torch.Tensor:
    """ATen-compatible l1_loss. reduction: 0=none, 1=mean, 2=sum."""
    logger.debug(f"FLAG_DNN L1_LOSS (reduction={reduction})")

    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    n_elements = input.numel()
    out = torch.empty_like(input)

    if n_elements == 0:
        if reduction == _REDUCTION_NONE:
            return out
        from flag_dnn.ops.sum import sum as flag_sum

        return flag_sum(out)

    if reduction == _REDUCTION_NONE:

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(input.device):
            l1_loss_elementwise_kernel[grid](input, target, out, n_elements)
        return out

    # Fused reduction path for MEAN/SUM: avoids intermediate buffer
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 8192)
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    partial = torch.empty(num_blocks, dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        l1_loss_reduce_kernel[(num_blocks,)](
            input, target, partial, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )

    from flag_dnn.ops.sum import sum as flag_sum

    total = flag_sum(partial)
    if reduction == _REDUCTION_MEAN:
        result = total / n_elements
    else:  # _REDUCTION_SUM
        result = total
    return result.to(input.dtype)
