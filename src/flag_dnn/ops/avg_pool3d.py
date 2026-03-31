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
def avg_pool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    pad_d, pad_h, pad_w,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    KERNEL_D: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    HAS_DIVISOR_OVERRIDE: tl.constexpr, DIVISOR_OVERRIDE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OD * OH * OW
    mask = offsets < num_elements

    # 反推 3D 输出坐标 (n, c, od, oh, ow)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    od = (offsets // (OW * OH)) % OD
    c = (offsets // (OW * OH * OD)) % C
    n = offsets // (C * OW * OH * OD)

    x_base_idx = n * (C * D * H * W) + c * (D * H * W)

    d_start = od * STRIDE_D - pad_d
    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    # 动态计算除数
    if HAS_DIVISOR_OVERRIDE:
            pool_size = DIVISOR_OVERRIDE
    else:
        if COUNT_INCLUDE_PAD:
            # start 必定 >= -pad
            end_d = tl.minimum(d_start + KERNEL_D, D + pad_d)
            pool_d = end_d - d_start
            
            end_h = tl.minimum(h_start + KERNEL_H, H + pad_h)
            pool_h = end_h - h_start
            
            end_w = tl.minimum(w_start + KERNEL_W, W + pad_w)
            pool_w = end_w - w_start
        else:
            start_d = tl.maximum(d_start, 0)
            end_d = tl.minimum(d_start + KERNEL_D, D)
            pool_d = end_d - start_d
            
            start_h = tl.maximum(h_start, 0)
            end_h = tl.minimum(h_start + KERNEL_H, H)
            pool_h = end_h - start_h
            
            start_w = tl.maximum(w_start, 0)
            end_w = tl.minimum(w_start + KERNEL_W, W)
            pool_w = end_w - start_w
        
        pool_size = pool_d * pool_h * pool_w

    # 防止极端情况下除数为 0
    pool_size = tl.where(pool_size == 0, 1, pool_size)

    # 使用高精度初始化累加器
    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    # 3D 窗口，三层静态循环展开
    for kd in tl.static_range(KERNEL_D):
        for kh in tl.static_range(KERNEL_H):
            for kw in tl.static_range(KERNEL_W):
                id_ = d_start + kd
                ih = h_start + kh
                iw = w_start + kw

                valid = (id_ >= 0) & (id_ < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                load_idx = x_base_idx + id_ * (H * W) + ih * W + iw

                val = tl.load(x_ptr + load_idx, mask=mask & valid, other=0.0).to(ACC_DTYPE)
                sum_val += val

    # 计算平均值并转回原数据类型
    res = sum_val / pool_size
    tl.store(y_ptr + offsets, res.to(x_ptr.dtype.element_ty), mask=mask)


def avg_pool3d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN AVG_POOL3D (kernel={kernel_size}, divisor={divisor_override})")

    def _triple(x):
        return (x, x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _triple(kernel_size)
    stride = _triple(stride) if stride is not None else kernel_size
    padding = _triple(padding)

    assert input.ndim in [4, 5], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    N, C, D, H, W = input.shape

    def _out_size(L, pad, k, s, ceil):
        out = (L + 2 * pad - k) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OD = _out_size(D, padding[0], kernel_size[0], stride[0], ceil_mode)
    OH = _out_size(H, padding[1], kernel_size[1], stride[1], ceil_mode)
    OW = _out_size(W, padding[2], kernel_size[2], stride[2], ceil_mode)

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

    M = N * C * OD * OH * OW
    if M == 0:
        return y.squeeze(0) if is_4d else y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    acc_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32

    has_divisor_override = divisor_override is not None
    div_override_val = divisor_override if has_divisor_override else 1

    with torch_device_fn.device(x.device):
        avg_pool3d_kernel[grid](
            x, y,
            N, C, D, H, W,
            OD, OH, OW,
            padding[0], padding[1], padding[2],
            STRIDE_D=stride[0], STRIDE_H=stride[1], STRIDE_W=stride[2],
            KERNEL_D=kernel_size[0], KERNEL_H=kernel_size[1], KERNEL_W=kernel_size[2],
            COUNT_INCLUDE_PAD=count_include_pad,
            HAS_DIVISOR_OVERRIDE=has_divisor_override, DIVISOR_OVERRIDE=div_override_val,
            ACC_DTYPE=acc_dtype,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return y.squeeze(0) if is_4d else y