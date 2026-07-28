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

"""Ascend tanh orchestration for layouts not preserved by torch.empty_like."""

import torch
import triton

from flag_dnn import runtime
from flag_dnn.ops.tanh import _PORTABLE_DTYPES, tanh_kernel
from flag_dnn.runtime import torch_device_fn


def tanh(input: torch.Tensor) -> torch.Tensor:
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

    # torch_npu currently makes torch.empty_like(channels_last) contiguous.
    # Flattening the original physical storage into that output would permute
    # logical values, so normalize the input before the flat Triton launch.
    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()
    if n_elements == 0:
        return torch.empty_like(input)
    output = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        tanh_kernel[grid](input, output, n_elements)
    return output


__all__ = ("tanh",)
