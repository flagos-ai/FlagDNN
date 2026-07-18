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
def mse_loss_elementwise_kernel(
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
    res = diff * diff

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def mse_loss_single_block_kernel(
    input_ptr,
    target_ptr,
    result_ptr,
    n_elements,
    inv_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-block reduction; stores result in native dtype."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tgt = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = inp - tgt
    total = tl.sum(diff * diff, axis=0) * inv_n
    tl.store(result_ptr, total.to(result_ptr.dtype.element_ty))


@libentry()
@triton.jit
def mse_loss_fused_kernel(
    input_ptr,
    target_ptr,
    result_ptr,
    n_elements,
    inv_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Multi-block fp32 atomic_add accumulation kernel."""
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tgt = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = inp - tgt
    partial = tl.sum(diff * diff, axis=0) * inv_n
    tl.atomic_add(result_ptr, partial)


@libentry()
@triton.jit
def _zero_scalar_kernel(result_ptr):
    tl.store(result_ptr, 0.0)


def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = _REDUCTION_MEAN,
) -> torch.Tensor:
    """ATen-compatible mse_loss. reduction: 0=none, 1=mean, 2=sum."""
    logger.debug(f"FLAG_DNN MSE_LOSS (reduction={reduction})")

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
            mse_loss_elementwise_kernel[grid](input, target, out, n_elements)
        return out

    inv_n = 1.0 / n_elements if reduction == _REDUCTION_MEAN else 1.0

    # For 16-bit dtypes, extend single-block threshold
    # to 32768 elements.
    # (64KB of data per block).  This ensures small tensors like [32, 1000] and
    # [8, 4096] avoid post-kernel type conversion overhead:
    #   - bf16: tl.atomic_add is not supported, so use
    #     fp32 + .to(bf16).
    #   - fp16: native atomic is supported but still benefits
    #     from eliminating conversion.
    # For fp32/fp64, larger BLOCK_SIZE degrades performance;
    # keep the 8192 cap.
    if input.dtype in (torch.float16, torch.bfloat16):
        single_block_threshold = 32768
    else:
        single_block_threshold = 8192

    single_block_size = min(
        triton.next_power_of_2(n_elements), single_block_threshold
    )
    if n_elements <= single_block_threshold:
        result = torch.empty(1, dtype=input.dtype, device=input.device)
        with torch_device_fn.device(input.device):
            mse_loss_single_block_kernel[(1,)](
                input,
                target,
                result,
                n_elements,
                inv_n,
                BLOCK_SIZE=single_block_size,
            )
        return result.squeeze()

    # Multi-block fp32 accumulation is safe for all dtypes
    # and large block counts.
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    result = torch.empty(1, dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        _zero_scalar_kernel[(1,)](result)
        mse_loss_fused_kernel[(num_blocks,)](
            input, target, result, n_elements, inv_n, BLOCK_SIZE=BLOCK_SIZE
        )
    result = result.squeeze()
    return result if input.dtype == torch.float32 else result.to(input.dtype)
