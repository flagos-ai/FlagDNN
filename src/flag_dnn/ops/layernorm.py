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

from typing import Any

import torch

from flag_dnn.ops.layer_norm import layer_norm_forward


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _trailing_normalized_shape(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    rank = input.dim()
    if scale.dim() > rank:
        raise RuntimeError("layernorm scale rank cannot exceed input rank")

    aligned_scale_shape = (1,) * (rank - scale.dim()) + tuple(scale.shape)
    axes = tuple(
        index
        for index, size in enumerate(aligned_scale_shape)
        if int(size) != 1
    )
    if not axes:
        axes = (rank - 1,)

    trailing_axes = tuple(range(rank - len(axes), rank))
    if axes != trailing_axes:
        raise NotImplementedError(
            "flag_dnn layernorm wraps layer_norm, which supports only "
            f"trailing normalized axes; scale implies axes={axes}"
        )

    normalized_shape = tuple(int(input.shape[axis]) for axis in axes)
    normalized_numel = 1
    for dim in normalized_shape:
        normalized_numel *= dim

    if scale.numel() != normalized_numel or bias.numel() != normalized_numel:
        raise NotImplementedError(
            "flag_dnn layernorm requires scale and bias to contain exactly "
            f"{normalized_numel} normalized values"
        )
    if not scale.is_contiguous() or not bias.is_contiguous():
        raise NotImplementedError(
            "flag_dnn layernorm currently requires contiguous scale and bias"
        )

    return normalized_shape, scale.reshape(-1), bias.reshape(-1)


def layernorm(
    norm_forward_phase,
    input: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    epsilon,
    *,
    compute_data_type=None,
    name: str = "",
):
    del norm_forward_phase, compute_data_type, name
    normalized_shape, weight, bias_flat = _trailing_normalized_shape(
        input, scale, bias
    )
    return layer_norm_forward(
        input,
        normalized_shape,
        weight=weight,
        bias=bias_flat,
        eps=_scalar(epsilon),
    )
