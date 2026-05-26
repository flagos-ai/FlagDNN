import logging
from typing import Optional, Union, List

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Nearest-neighbor 2D (4-D input: N, C, H, W)                        #
# ------------------------------------------------------------------ #
@triton.jit
def nearest2d_kernel(
    x_ptr,
    y_ptr,
    IH, IW,
    OH, OW,
    scale_h, scale_w,
    NC,
    BLOCK: tl.constexpr,
):
    """Grid: (NC, cdiv(OH*OW, BLOCK))"""
    pid_nc = tle.program_id(0)
    pid_hw = tle.program_id(1)

    ow_indices = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = ow_indices < OH * OW

    oh = ow_indices // OW
    ow = ow_indices % OW

    # Nearest neighbor: floor((oh + 0.5) * (IH / OH))
    ih = tl.minimum((oh * scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * scale_w).to(tl.int32), IW - 1)

    in_offset = pid_nc * IH * IW + ih * IW + iw
    out_offset = pid_nc * OH * OW + ow_indices

    val = tl.load(x_ptr + in_offset, mask=mask)
    tl.store(y_ptr + out_offset, val, mask=mask)


# ------------------------------------------------------------------ #
#  Bilinear 2D (4-D input: N, C, H, W)                                #
# ------------------------------------------------------------------ #
@triton.jit
def bilinear2d_kernel(
    x_ptr,
    y_ptr,
    IH, IW,
    OH, OW,
    scale_h, scale_w,
    NC,
    align_corners: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_hw = tle.program_id(1)

    ow_indices = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = ow_indices < OH * OW

    oh = ow_indices // OW
    ow = ow_indices % OW

    oh_f = oh.to(tl.float32)
    ow_f = ow.to(tl.float32)

    if align_corners:
        # Align by corner pixels
        if OH > 1:
            y = oh_f * (IH - 1).to(tl.float32) / (OH - 1).to(tl.float32)
        else:
            y = tl.zeros_like(oh_f)
        if OW > 1:
            x_coord = ow_f * (IW - 1).to(tl.float32) / (OW - 1).to(tl.float32)
        else:
            x_coord = tl.zeros_like(ow_f)
    else:
        # Align by half-pixel
        y = (oh_f + 0.5) * scale_h - 0.5
        x_coord = (ow_f + 0.5) * scale_w - 0.5

    y0 = tl.maximum(tl.math.floor(y).to(tl.int32), 0)
    x0 = tl.maximum(tl.math.floor(x_coord).to(tl.int32), 0)
    y1 = tl.minimum(y0 + 1, IH - 1)
    x1 = tl.minimum(x0 + 1, IW - 1)

    fy = y - tl.math.floor(y)
    fx = x_coord - tl.math.floor(x_coord)
    fy = tl.maximum(fy, 0.0)
    fx = tl.maximum(fx, 0.0)

    base = pid_nc * IH * IW
    v00 = tl.load(x_ptr + base + y0 * IW + x0, mask=mask).to(tl.float32)
    v01 = tl.load(x_ptr + base + y0 * IW + x1, mask=mask).to(tl.float32)
    v10 = tl.load(x_ptr + base + y1 * IW + x0, mask=mask).to(tl.float32)
    v11 = tl.load(x_ptr + base + y1 * IW + x1, mask=mask).to(tl.float32)

    out = (1.0 - fy) * (1.0 - fx) * v00         + (1.0 - fy) * fx * v01         + fy * (1.0 - fx) * v10         + fy * fx * v11

    tl.store(y_ptr + pid_nc * OH * OW + ow_indices, out.to(x_ptr.dtype.element_ty), mask=mask)


# ------------------------------------------------------------------ #
#  Nearest-neighbor 1D (3-D input: N, C, W)                           #
# ------------------------------------------------------------------ #
@triton.jit
def nearest1d_kernel(
    x_ptr,
    y_ptr,
    IW,
    OW,
    scale_w,
    NC,
    BLOCK: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_w = tle.program_id(1)

    ow_indices = pid_w * BLOCK + tl.arange(0, BLOCK)
    mask = ow_indices < OW

    iw = tl.minimum((ow_indices * scale_w).to(tl.int32), IW - 1)

    val = tl.load(x_ptr + pid_nc * IW + iw, mask=mask)
    tl.store(y_ptr + pid_nc * OW + ow_indices, val, mask=mask)


def _compute_output_size(input_size, size, scale_factor):
    """Compute the output spatial size."""
    spatial_dims = len(input_size) - 2
    if size is not None:
        if isinstance(size, int):
            return (size,) * spatial_dims
        return tuple(size)
    else:
        if isinstance(scale_factor, (int, float)):
            scales = (scale_factor,) * spatial_dims
        else:
            scales = tuple(scale_factor)
        return tuple(int(input_size[i + 2] * scales[i]) for i in range(spatial_dims))


def interpolate(
    input: torch.Tensor,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN INTERPOLATE (mode={mode}, size={size})")

    ndim = input.ndim

    # TODO: FlagDNN only has Triton kernels for nearest/bilinear 1D/2D here.
    # Keep this fallback marked until the missing interpolate modes/dimensions
    # have native kernels.
    if mode not in ("nearest", "bilinear") or ndim not in (3, 4) or antialias:
        import torch.nn.functional as F
        return F.interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    if mode == "bilinear" and ndim != 4:
        # TODO: no native bilinear kernel for non-4D input yet.
        import torch.nn.functional as F
        return F.interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    out_spatial = _compute_output_size(input.shape, size, scale_factor)

    if not input.is_contiguous():
        input = input.contiguous()

    N = input.shape[0]
    C = input.shape[1]
    NC = N * C

    BLOCK = 256

    if ndim == 3:
        # 3D: (N, C, W)
        IW = input.shape[2]
        OW = out_spatial[0]
        scale_w = IW / OW

        out = torch.empty((N, C, OW), dtype=input.dtype, device=input.device)
        grid = (NC, triton.cdiv(OW, BLOCK))
        with torch_device_fn.device(input.device):
            nearest1d_kernel[grid](input, out, IW, OW, scale_w, NC, BLOCK=BLOCK)

    elif ndim == 4:
        IH = input.shape[2]
        IW = input.shape[3]
        OH = out_spatial[0]
        OW = out_spatial[1]
        scale_h = IH / OH
        scale_w = IW / OW

        out = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)
        grid = (NC, triton.cdiv(OH * OW, BLOCK))

        with torch_device_fn.device(input.device):
            if mode == "nearest":
                nearest2d_kernel[grid](
                    input, out, IH, IW, OH, OW, scale_h, scale_w, NC, BLOCK=BLOCK
                )
            elif mode == "bilinear":
                ac = align_corners if align_corners is not None else False
                bilinear2d_kernel[grid](
                    input, out, IH, IW, OH, OW, scale_h, scale_w, NC,
                    align_corners=ac, BLOCK=BLOCK
                )

    return out
