import logging
import math
from typing import Tuple, Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def max_pool3d_kernel(
    x_ptr, y_ptr, idx_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    pad_d, pad_h, pad_w,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    DIL_D: tl.constexpr, DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    KERNEL_D: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OD * OH * OW
    mask = offsets < num_elements

    # 反推 3D 坐标 (n, c, od, oh, ow)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    od = (offsets // (OW * OH)) % OD
    c = (offsets // (OW * OH * OD)) % C
    n = offsets // (C * OW * OH * OD)

    # 输入基础偏移：每个元素所在的 batch 和 channel 起点
    x_base_idx = n * (C * D * H * W) + c * (D * H * W)

    d_start = od * STRIDE_D - pad_d
    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    input_dtype = x_ptr.dtype.element_ty
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=input_dtype)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 3D 窗口，三层静态循环展开
    for kd in tl.static_range(KERNEL_D):
        for kh in tl.static_range(KERNEL_H):
            for kw in tl.static_range(KERNEL_W):
                id_ = d_start + kd * DIL_D
                ih = h_start + kh * DIL_H
                iw = w_start + kw * DIL_W

                valid = (id_ >= 0) & (id_ < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                load_idx = x_base_idx + id_ * (H * W) + ih * W + iw

                val = tl.load(x_ptr + load_idx, mask=mask & valid, other=-float('inf'))

                update_mask = val > max_val
                max_val = tl.where(update_mask, val, max_val)
                
                if RETURN_INDICES:
                    # 3D 场景下的局部索引，即 D*H*W 展平后的绝对偏移
                    current_idx = id_ * (H * W) + ih * W + iw
                    max_idx = tl.where(update_mask, current_idx, max_idx)

    tl.store(y_ptr + offsets, max_val, mask=mask)
    
    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx, mask=mask)


def max_pool3d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(f"FLAG_DNN MAX_POOL3D (kernel={kernel_size}, return_indices={return_indices})")

    def _triple(x):
        return (x, x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _triple(kernel_size)
    stride = _triple(stride) if stride is not None else kernel_size
    padding = _triple(padding)
    dilation = _triple(dilation)

    assert input.ndim in [4, 5], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    N, C, D, H, W = input.shape

    def _out_size(L, pad, dil, k, s, ceil):
        out = (L + 2 * pad - dil * (k - 1) - 1) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OD = _out_size(D, padding[0], dilation[0], kernel_size[0], stride[0], ceil_mode)
    OH = _out_size(H, padding[1], dilation[1], kernel_size[1], stride[1], ceil_mode)
    OW = _out_size(W, padding[2], dilation[2], kernel_size[2], stride[2], ceil_mode)

    # ceil_mode 边缘丢弃
    if ceil_mode:
        if (OD - 1) * stride[0] >= D + padding[0]:
            OD -= 1
        if (OH - 1) * stride[1] >= H + padding[1]:
            OH -= 1
        if (OW - 1) * stride[2] >= W + padding[2]:
            OW -= 1

    x = input.contiguous()
    y = torch.empty((N, C, OD, OH, OW), dtype=x.dtype, device=x.device)
    
    idx = torch.empty((N, C, OD, OH, OW), dtype=torch.int64, device=x.device) if return_indices else None

    M = N * C * OD * OH * OW
    if M == 0:
        out_y = y.squeeze(0) if is_4d else y
        if return_indices:
            return out_y, (idx.squeeze(0) if is_4d else idx)
        return out_y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        max_pool3d_kernel[grid](
            x, y, idx,
            N, C, D, H, W,
            OD, OH, OW,
            padding[0], padding[1], padding[2],
            STRIDE_D=stride[0], STRIDE_H=stride[1], STRIDE_W=stride[2],
            DIL_D=dilation[0], DIL_H=dilation[1], DIL_W=dilation[2],
            KERNEL_D=kernel_size[0], KERNEL_H=kernel_size[1], KERNEL_W=kernel_size[2],
            RETURN_INDICES=return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )

    out_y = y.squeeze(0) if is_4d else y
    if return_indices:
        return out_y, (idx.squeeze(0) if is_4d else idx)
    return out_y