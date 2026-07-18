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

from __future__ import annotations

from typing import Optional

import torch

from flag_dnn.ops.conv1d import conv1d
from flag_dnn.ops.silu import silu


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "identity",
) -> torch.Tensor:
    if x.dim() != 3 or weight.dim() != 2:
        raise ValueError(
            "causal_conv1d expects x shape (batch, dim, seq_len) "
            "and weight shape (dim, kernel_size)"
        )
    if x.device != weight.device or (
        bias is not None and bias.device != x.device
    ):
        raise ValueError("x, weight, and bias must be on the same device")
    if x.dtype != weight.dtype or (bias is not None and bias.dtype != x.dtype):
        raise TypeError("x, weight, and bias must have the same dtype")
    batch, dim, _ = x.shape
    del batch
    if weight.shape[0] != dim:
        raise ValueError(
            f"weight.shape[0] must match x.shape[1], "
            f"got {weight.shape[0]} and {dim}"
        )
    if bias is not None and bias.shape != (dim,):
        raise ValueError(
            f"bias must have shape ({dim},), got {tuple(bias.shape)}"
        )
    activation = str(activation).lower()
    if activation not in ("identity", "silu"):
        raise ValueError("activation must be 'identity' or 'silu'")

    kernel = int(weight.shape[1])
    conv_weight = weight.reshape(dim, 1, kernel).contiguous()
    y = conv1d(
        x,
        conv_weight,
        bias=bias,
        stride=1,
        padding=(kernel - 1, 0),
        dilation=1,
        groups=dim,
    )
    if activation == "silu":
        return silu(y)
    return y
