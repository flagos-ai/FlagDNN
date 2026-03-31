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
def max_pool2d_kernel(
    x_ptr, y_ptr, idx_ptr,
    N, C, H, W,
    OH, OW,
    pad_h, pad_w,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OH * OW
    mask = offsets < num_elements

    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (C * OW * OH)

    x_base_idx = n * (C * H * W) + c * (H * W)

    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 使用 tl.static_range 强制编译器在编译期展开循环
    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            # 编译器会在编译期直接算好这里的偏移常数，优化 dilation=1 的分支
            ih = h_start + kh * DIL_H
            iw = w_start + kw * DIL_W

            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            load_idx = x_base_idx + ih * W + iw

            val = tl.load(x_ptr + load_idx, mask=mask & valid, other=-float('inf')).to(tl.float32)

            update_mask = val > max_val
            max_val = tl.where(update_mask, val, max_val)
            
            if RETURN_INDICES:
                current_idx = ih * W + iw
                max_idx = tl.where(update_mask, current_idx, max_idx)

    tl.store(y_ptr + offsets, max_val.to(x_ptr.dtype.element_ty), mask=mask)
    
    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx, mask=mask)


def max_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(f"FLAG_DNN MAX_POOL2D (kernel={kernel_size}, return_indices={return_indices})")

    def _pair(x):
        return (x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _pair(kernel_size)
    stride = _pair(stride) if stride is not None else kernel_size
    padding = _pair(padding)
    dilation = _pair(dilation)

    assert input.ndim in [3, 4], "Input must be 3D or 4D"
    is_3d = input.ndim == 3
    if is_3d:
        input = input.unsqueeze(0)

    N, C, H, W = input.shape

    def _out_size(L, pad, dil, k, s, ceil):
        out = (L + 2 * pad - dil * (k - 1) - 1) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OH = _out_size(H, padding[0], dilation[0], kernel_size[0], stride[0], ceil_mode)
    OW = _out_size(W, padding[1], dilation[1], kernel_size[1], stride[1], ceil_mode)

    if ceil_mode:
        if (OH - 1) * stride[0] >= H + padding[0]:
            OH -= 1
        if (OW - 1) * stride[1] >= W + padding[1]:
            OW -= 1

    x = input.contiguous()
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    
    idx = torch.empty((N, C, OH, OW), dtype=torch.int64, device=x.device) if return_indices else None

    M = N * C * OH * OW
    if M == 0:
        out_y = y.squeeze(0) if is_3d else y
        if return_indices:
            return out_y, (idx.squeeze(0) if is_3d else idx)
        return out_y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        max_pool2d_kernel[grid](
            x, y, idx,
            N, C, H, W,
            OH, OW,
            padding[0], padding[1],
            STRIDE_H=stride[0], STRIDE_W=stride[1],
            DIL_H=dilation[0], DIL_W=dilation[1],
            KERNEL_H=kernel_size[0], KERNEL_W=kernel_size[1],
            RETURN_INDICES=return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )

    out_y = y.squeeze(0) if is_3d else y
    if return_indices:
        return out_y, (idx.squeeze(0) if is_3d else idx)
    return out_y