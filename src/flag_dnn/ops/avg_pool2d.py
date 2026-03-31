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
def avg_pool2d_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    OH, OW,
    pad_h, pad_w,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OH * OW
    mask = offsets < num_elements

    # 反推坐标
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (C * OW * OH)

    x_base_idx = n * (C * H * W) + c * (H * W)

    # 窗口在原图 (H, W) 上的物理起始坐标 (可能为负)
    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    # Triton 中统一升至 float32 防止累加精度溢出
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 使用 tl.static_range 强制编译器在编译期展开循环
    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            ih = h_start + kh
            iw = w_start + kw

            # 边界检查
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            load_idx = x_base_idx + ih * W + iw

            # 如果落在 padding 区域，补 0.0 (Average Pooling 的 padding 是补零)
            val = tl.load(x_ptr + load_idx, mask=mask & valid, other=0.0).to(tl.float32)
            sum_val += val

    # 计算均值的除数 (Divisor)
    if DIVISOR_OVERRIDE > 0:
        divisor = DIVISOR_OVERRIDE
    elif COUNT_INCLUDE_PAD:
        # KERNEL_H 和 KERNEL_W 现在是编译期常量
        hend_bounded = tl.where(h_start + KERNEL_H > H + pad_h, H + pad_h, h_start + KERNEL_H)
        pool_h = hend_bounded - h_start
        
        wend_bounded = tl.where(w_start + KERNEL_W > W + pad_w, W + pad_w, w_start + KERNEL_W)
        pool_w = wend_bounded - w_start
        
        divisor = pool_h * pool_w
    else:
        # 当不包含 padding 时，计算窗口与原图重叠的有效面积
        ih_start_clamp = tl.where(h_start < 0, 0, h_start)
        ih_end_clamp = tl.where(h_start + KERNEL_H > H, H, h_start + KERNEL_H)
        valid_h = ih_end_clamp - ih_start_clamp

        iw_start_clamp = tl.where(w_start < 0, 0, w_start)
        iw_end_clamp = tl.where(w_start + KERNEL_W > W, W, w_start + KERNEL_W)
        valid_w = iw_end_clamp - iw_start_clamp

        divisor = valid_h * valid_w
    
    # 防止除零错误 (如果极端的 padding 导致有效面积为 0)
    divisor = tl.where(divisor <= 0, 1, divisor)

    # 计算均值并写回
    avg_val = sum_val / divisor
    tl.store(y_ptr + offsets, avg_val.to(x_ptr.dtype.element_ty), mask=mask)


def avg_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN AVG_POOL2D (kernel={kernel_size}, count_include_pad={count_include_pad})")

    def _pair(x):
        return (x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _pair(kernel_size)
    stride = _pair(stride) if stride is not None else kernel_size
    padding = _pair(padding)

    assert input.ndim in [3, 4], "Input must be 3D or 4D"
    is_3d = input.ndim == 3
    if is_3d:
        input = input.unsqueeze(0)

    N, C, H, W = input.shape

    # --- 计算输出形状公式 ---
    def _out_size(L, pad, k, s, ceil):
        out = (L + 2 * pad - k) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OH = _out_size(H, padding[0], kernel_size[0], stride[0], ceil_mode)
    OW = _out_size(W, padding[1], kernel_size[1], stride[1], ceil_mode)

    # 边缘修正
    if ceil_mode:
        if (OH - 1) * stride[0] >= H + padding[0]:
            OH -= 1
        if (OW - 1) * stride[1] >= W + padding[1]:
            OW -= 1

    x = input.contiguous()
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)

    M = N * C * OH * OW
    if M == 0:
        return y.squeeze(0) if is_3d else y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    div_over = divisor_override if divisor_override is not None else -1

    with torch_device_fn.device(x.device):
        avg_pool2d_kernel[grid](
            x, y,
            N, C, H, W,
            OH, OW,
            padding[0], padding[1],
            STRIDE_H=stride[0], STRIDE_W=stride[1],
            KERNEL_H=kernel_size[0], KERNEL_W=kernel_size[1],
            COUNT_INCLUDE_PAD=count_include_pad,
            DIVISOR_OVERRIDE=div_over,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return y.squeeze(0) if is_3d else y