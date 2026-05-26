from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch

from flag_dnn.ops.conv1d import conv1d
from flag_dnn.ops.conv2d import conv2d
from flag_dnn.ops.conv3d import conv3d


def _spatial_rank(image: torch.Tensor, weight: torch.Tensor) -> int:
    if image.dim() == 2 and weight.dim() == 3:
        return 1
    if image.dim() >= 3 and weight.dim() == image.dim():
        return image.dim() - 2
    raise RuntimeError(
        "flag_dnn conv_fprop expects matching 1D/2D/3D convolution "
        f"shapes, got image dim={image.dim()} and weight dim={weight.dim()}"
    )


def _tuple_n(
    value: Union[int, Sequence[int]], rank: int, name: str
) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _normalize_convolution_mode(convolution_mode: Any) -> str:
    if convolution_mode is None:
        return "CROSS_CORRELATION"
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    if mode in ("CROSS_CORRELATION", "CONVOLUTION"):
        return mode
    raise RuntimeError(
        "convolution_mode must be CROSS_CORRELATION or CONVOLUTION"
    )


def _normalize_padding(
    rank: int,
    padding: Optional[Union[str, int, Sequence[int]]],
    pre_padding: Optional[Union[int, Sequence[int]]],
    post_padding: Optional[Union[int, Sequence[int]]],
) -> Union[str, Tuple[int, ...]]:
    if pre_padding is not None or post_padding is not None:
        if padding is not None:
            raise TypeError(
                "conv_fprop accepts either padding or pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                "conv_fprop requires both pre_padding and post_padding"
            )
        pre = _tuple_n(pre_padding, rank, "pre_padding")
        post = _tuple_n(post_padding, rank, "post_padding")
    else:
        if padding is None:
            padding = 0
        if isinstance(padding, str):
            return padding
        pre = post = _tuple_n(padding, rank, "padding")

    if rank == 1:
        return (pre[0], post[0])
    if rank == 2:
        return (pre[0], post[0], pre[1], post[1])
    if rank == 3:
        return (pre[0], post[0], pre[1], post[1], pre[2], post[2])
    raise NotImplementedError(
        "flag_dnn conv_fprop only supports ranks 1, 2, and 3"
    )


def conv_fprop(
    image: torch.Tensor,
    weight: torch.Tensor,
    padding: Optional[Union[str, int, Sequence[int]]] = None,
    *,
    pre_padding: Optional[Union[int, Sequence[int]]] = None,
    post_padding: Optional[Union[int, Sequence[int]]] = None,
    stride: Union[int, Sequence[int]] = 1,
    dilation: Union[int, Sequence[int]] = 1,
    convolution_mode: Any = "CROSS_CORRELATION",
    compute_data_type: Any = None,
    name: str = "",
    groups: int = 1,
) -> torch.Tensor:
    del compute_data_type, name
    mode = _normalize_convolution_mode(convolution_mode)
    rank = _spatial_rank(image, weight)
    if mode == "CONVOLUTION":
        weight = torch.flip(
            weight, dims=tuple(range(2, weight.dim()))
        ).contiguous()

    if rank == 1:
        return conv1d(
            image,
            weight,
            stride=_tuple_n(stride, 1, "stride"),
            padding=_normalize_padding(
                rank, padding, pre_padding, post_padding
            ),
            dilation=_tuple_n(dilation, 1, "dilation"),
            groups=groups,
        )
    if rank == 2:
        return conv2d(
            image,
            weight,
            stride=_tuple_n(stride, 2, "stride"),
            padding=_normalize_padding(
                rank, padding, pre_padding, post_padding
            ),
            dilation=_tuple_n(dilation, 2, "dilation"),
            groups=groups,
        )
    if rank == 3:
        return conv3d(
            image,
            weight,
            stride=_tuple_n(stride, 3, "stride"),
            padding=_normalize_padding(
                rank, padding, pre_padding, post_padding
            ),
            dilation=_tuple_n(dilation, 3, "dilation"),
            groups=groups,
        )
    raise RuntimeError(f"unsupported conv_fprop spatial rank: {rank}")
