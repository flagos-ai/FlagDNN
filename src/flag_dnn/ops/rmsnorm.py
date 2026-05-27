from __future__ import annotations

from typing import Any, Optional

import torch

from flag_dnn.ops.rms_norm import rms_norm_forward


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _trailing_normalized_shape(
    input: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple[tuple[int, ...], torch.Tensor, Optional[torch.Tensor]]:
    rank = input.dim()
    if scale.dim() > rank:
        raise RuntimeError("rmsnorm scale rank cannot exceed input rank")

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
            "flag_dnn rmsnorm wraps rms_norm, which supports only "
            f"trailing normalized axes; scale implies axes={axes}"
        )

    normalized_shape = tuple(int(input.shape[axis]) for axis in axes)
    normalized_numel = 1
    for dim in normalized_shape:
        normalized_numel *= dim

    if scale.numel() != normalized_numel:
        raise NotImplementedError(
            "flag_dnn rmsnorm requires scale to contain exactly "
            f"{normalized_numel} normalized values"
        )
    if bias is not None and bias.numel() != normalized_numel:
        raise NotImplementedError(
            "flag_dnn rmsnorm requires bias to contain exactly "
            f"{normalized_numel} normalized values"
        )
    if not scale.is_contiguous() or (
        bias is not None and not bias.is_contiguous()
    ):
        raise NotImplementedError(
            "flag_dnn rmsnorm currently requires contiguous scale and bias"
        )

    return (
        normalized_shape,
        scale.reshape(-1),
        None if bias is None else bias.reshape(-1),
    )


def rmsnorm(
    norm_forward_phase,
    input: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epsilon=1e-5,
    *,
    compute_data_type=None,
    name: str = "",
):
    del norm_forward_phase, compute_data_type, name
    normalized_shape, weight, bias_flat = _trailing_normalized_shape(
        input, scale, bias
    )
    return rms_norm_forward(
        input,
        normalized_shape,
        weight=weight,
        bias=bias_flat,
        eps=_scalar(epsilon),
    )
