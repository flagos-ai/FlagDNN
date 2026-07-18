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

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def one_hot_zero_kernel(
    out_ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    tl.store(
        out_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.int64), mask=mask
    )


@triton.jit
def one_hot_scatter_kernel(
    input_ptr,
    out_ptr,
    N,
    C,
    BLOCK_N: tl.constexpr,
):
    """Scatter kernel: each CTA writes BLOCK_N hot entries into the output.
    Output must be pre-zeroed. This is O(N) writes vs O(N*C) for explicit fill,
    matching PyTorch's zeros+scatter_() strategy."""
    pid = tle.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    cls_idx = tl.load(input_ptr + n_offsets, mask=n_mask, other=0).to(tl.int64)
    # out[n, cls_idx[n]] = 1 for each n in this block
    out_offsets = n_offsets * C + cls_idx
    tl.store(
        out_ptr + out_offsets, tl.full([BLOCK_N], 1, tl.int64), mask=n_mask
    )


def one_hot(
    tensor: torch.Tensor,
    num_classes: int = -1,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ONE_HOT (num_classes={num_classes})")

    if tensor.dtype != torch.long:
        raise RuntimeError("one_hot is only applicable to index tensor.")

    if num_classes == -1:
        # TODO: replace this metadata query once FlagDNN has a Triton-backed
        # scalar shape-inference helper. The value is needed before allocating
        # the output tensor.
        num_classes = int(tensor.max().item()) + 1

    N = tensor.numel()
    C = num_classes

    out_shape = tensor.shape + (C,)

    out = torch.empty(out_shape, dtype=torch.long, device=tensor.device)
    out_1d = out.view(-1)

    if out.numel() == 0:
        return out

    BLOCK_ZERO = 1024
    with torch_device_fn.device(tensor.device):
        one_hot_zero_kernel[(triton.cdiv(out.numel(), BLOCK_ZERO),)](
            out_1d, out.numel(), BLOCK_SIZE=BLOCK_ZERO
        )

    if N == 0 or C == 0:
        return out

    flat_input = tensor.contiguous().view(-1)

    # Scatter: write 1 at each hot index; output initialization
    # is Triton-backed.
    BLOCK_N = 1024
    grid = (triton.cdiv(N, BLOCK_N),)
    with torch_device_fn.device(tensor.device):
        one_hot_scatter_kernel[grid](flat_input, out_1d, N, C, BLOCK_N=BLOCK_N)

    return out
