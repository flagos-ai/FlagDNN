import os
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _pair(val):
    if isinstance(val, int) or isinstance(val, str):
        return val, val
    if len(val) != 2:
        raise RuntimeError(f"expected length 2, but got {val}")
    return int(val[0]), int(val[1])


def _conv_out_dim(
    input_size,
    pad,
    dilation,
    kernel,
    stride,
) -> int:
    if pad == "valid":
        pad = 0
    elif pad == "same":
        return (input_size + stride - 1) // stride
    return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


CONV2D_CASES = [
    ((32, 3, 224, 224), (64, 3, 7, 7), False, 2, 3, 1, 1),
    ((32, 64, 56, 56), (64, 64, 3, 3), False, 1, 1, 1, 1),
    ((32, 64, 56, 56), (128, 64, 3, 3), False, 2, 1, 1, 1),
    ((32, 128, 28, 28), (128, 128, 3, 3), False, 1, 1, 1, 1),
    ((32, 256, 14, 14), (256, 256, 3, 3), False, 1, 1, 1, 1),
    ((32, 512, 7, 7), (512, 512, 3, 3), False, 1, 1, 1, 1),
    ((32, 64, 56, 56), (64, 64, 1, 1), False, 1, 0, 1, 1),
    ((32, 64, 56, 56), (256, 64, 1, 1), False, 1, 0, 1, 1),
    ((32, 128, 28, 28), (256, 128, 1, 1), False, 1, 0, 1, 1),
    ((32, 256, 14, 14), (512, 256, 1, 1), False, 1, 0, 1, 1),
    ((16, 64, 56, 56), (64, 64, 3, 3), True, 1, 1, 1, 1),
    ((16, 128, 28, 28), (128, 128, 1, 1), True, 1, 0, 1, 1),
    ((16, 64, 56, 56), (64, 64, 3, 3), False, 1, 2, 2, 1),
    ((16, 128, 28, 28), (128, 128, 3, 3), False, 1, 4, 4, 1),
    ((16, 64, 56, 56), (128, 32, 3, 3), False, 1, 1, 1, 2),
    ((16, 128, 28, 28), (128, 16, 3, 3), False, 1, 1, 1, 8),
    ((16, 32, 112, 112), (32, 1, 3, 3), False, 1, 1, 1, 32),
    ((16, 64, 56, 56), (64, 1, 3, 3), False, 1, 1, 1, 64),
    ((16, 128, 28, 28), (128, 1, 5, 5), False, 1, 2, 1, 128),
]


def _is_sequence(val):
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


def _filtered_cases():
    only = os.getenv("FLAGDNN_CONV2D_PERF_SHAPE_IDS")
    if not only:
        return CONV2D_CASES
    selected = {int(item) for item in only.split(",") if item.strip()}
    return [shape for idx, shape in enumerate(CONV2D_CASES) if idx in selected]


def _normalize_case(case):
    if len(case) == 7 and _is_sequence(case[0]) and _is_sequence(case[1]):
        (
            input_shape,
            weight_shape,
            has_bias,
            stride,
            padding,
            dilation,
            groups,
        ) = case
        return (
            tuple(input_shape),
            tuple(weight_shape),
            bool(has_bias),
            stride,
            padding,
            dilation,
            groups,
        )

    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = case
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
    return input_shape, weight_shape, False, stride, padding, 1, groups


def _output_shape(input_shape, weight_shape, stride, padding, dilation):
    n, _, ih, iw = input_shape
    oc, _, kh, kw = weight_shape
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    oh = _conv_out_dim(ih, pad_h, dil_h, kh, stride_h)
    ow = _conv_out_dim(iw, pad_w, dil_w, kw, stride_w)
    return n, oc, oh, ow


def _estimate_bytes(case, dtype):
    input_shape, weight_shape, has_bias, stride, padding, dilation, _ = (
        _normalize_case(case)
    )
    output_shape = _output_shape(
        input_shape, weight_shape, stride, padding, dilation
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    tensor_numel = (
        torch.Size(input_shape).numel()
        + torch.Size(weight_shape).numel()
        + torch.Size(output_shape).numel()
    )
    if has_bias:
        tensor_numel += weight_shape[0]
    return tensor_numel * element_size


def conv2d_input_fn(case, dtype, device, max_peak_bytes):
    if _estimate_bytes(case, dtype) > max_peak_bytes:
        return
    input_shape, weight_shape, has_bias, stride, padding, dilation, groups = (
        _normalize_case(case)
    )
    x = torch.empty(input_shape, dtype=dtype, device=device).uniform_(
        -1.0, 1.0
    )
    w = torch.empty(weight_shape, dtype=dtype, device=device).uniform_(
        -1.0, 1.0
    )
    b = (
        torch.empty((weight_shape[0],), dtype=dtype, device=device).uniform_(
            -1.0, 1.0
        )
        if has_bias
        else None
    )
    yield x, w, b, stride, padding, dilation, groups


def conv2d_output_shape(args):
    x, w, _, stride, padding, dilation, _ = args
    return _output_shape(
        tuple(x.shape), tuple(w.shape), stride, padding, dilation
    )


def conv2d_flops(args):
    x, w, _, stride, padding, dilation, groups = args
    n, c_in, ih, iw = x.shape
    c_out, _, kh, kw = w.shape
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    oh = _conv_out_dim(ih, pad_h, dil_h, kh, stride_h)
    ow = _conv_out_dim(iw, pad_w, dil_w, kw, stride_w)
    return 2 * n * oh * ow * c_out * (c_in // groups) * kh * kw


@pytest.mark.conv2d
def test_conv2d():
    bench = base.ConvolutionBenchmark(
        op_name="conv2d",
        torch_op=F.conv2d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=conv2d_input_fn,
        output_shape_fn=conv2d_output_shape,
        flops_fn=conv2d_flops,
        cases=_filtered_cases(),
        max_peak_bytes=8 * 1024**3,
    )
    bench.run()
