import os
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _triple(val):
    if isinstance(val, int) or isinstance(val, str):
        return val, val, val
    if len(val) != 3:
        raise RuntimeError(f"expected length 3, but got {val}")
    return int(val[0]), int(val[1]), int(val[2])


def _conv_out_dim(input_size, pad, dilation, kernel, stride) -> int:
    if pad == "valid":
        pad = 0
    elif pad == "same":
        return (input_size + stride - 1) // stride
    return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


CONV3D_CASES = [
    ((4, 8, 16, 32, 32), (16, 8, 3, 3, 3), False, 1, 1, 1, 1),
    ((4, 16, 16, 24, 24), (32, 16, 3, 3, 3), False, 2, 1, 1, 1),
    ((4, 32, 16, 32, 32), (64, 32, 1, 1, 1), False, 1, 0, 1, 1),
    ((2, 16, 12, 20, 20), (32, 8, 3, 3, 3), True, 1, 1, 1, 2),
    ((2, 16, 12, 20, 20), (16, 1, 3, 3, 3), False, 1, 1, 1, 16),
]


def _is_sequence(val):
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


def _filtered_cases():
    only = os.getenv("FLAGDNN_CONV3D_PERF_SHAPE_IDS")
    if not only:
        return CONV3D_CASES
    selected = {int(item) for item in only.split(",") if item.strip()}
    return [shape for idx, shape in enumerate(CONV3D_CASES) if idx in selected]


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
        input_d,
        input_h,
        input_w,
        out_c,
        kernel_d,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = case
    input_shape = (batch, input_c, input_d, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_d, kernel_h, kernel_w)
    return input_shape, weight_shape, False, stride, padding, 1, groups


def _output_shape(input_shape, weight_shape, stride, padding, dilation):
    n, _, input_d, input_h, input_w = input_shape
    c_out, _, kernel_d, kernel_h, kernel_w = weight_shape
    stride_d, stride_h, stride_w = _triple(stride)
    pad_d, pad_h, pad_w = _triple(padding)
    dil_d, dil_h, dil_w = _triple(dilation)
    out_d = _conv_out_dim(input_d, pad_d, dil_d, kernel_d, stride_d)
    out_h = _conv_out_dim(input_h, pad_h, dil_h, kernel_h, stride_h)
    out_w = _conv_out_dim(input_w, pad_w, dil_w, kernel_w, stride_w)
    return n, c_out, out_d, out_h, out_w


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


def conv3d_input_fn(case, dtype, device, max_peak_bytes):
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


def conv3d_output_shape(args):
    x, w, _, stride, padding, dilation, _ = args
    return _output_shape(
        tuple(x.shape), tuple(w.shape), stride, padding, dilation
    )


def conv3d_flops(args):
    x, w, _, stride, padding, dilation, groups = args
    n, c_in, input_d, input_h, input_w = x.shape
    c_out, _, kernel_d, kernel_h, kernel_w = w.shape
    stride_d, stride_h, stride_w = _triple(stride)
    pad_d, pad_h, pad_w = _triple(padding)
    dil_d, dil_h, dil_w = _triple(dilation)
    out_d = _conv_out_dim(input_d, pad_d, dil_d, kernel_d, stride_d)
    out_h = _conv_out_dim(input_h, pad_h, dil_h, kernel_h, stride_h)
    out_w = _conv_out_dim(input_w, pad_w, dil_w, kernel_w, stride_w)
    return (
        2
        * n
        * out_d
        * out_h
        * out_w
        * c_out
        * (c_in // groups)
        * kernel_d
        * kernel_h
        * kernel_w
    )


@pytest.mark.conv3d
def test_conv3d():
    bench = base.ConvolutionBenchmark(
        op_name="conv3d",
        torch_op=F.conv3d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=conv3d_input_fn,
        output_shape_fn=conv3d_output_shape,
        flops_fn=conv3d_flops,
        cases=_filtered_cases(),
        max_peak_bytes=8 * 1024**3,
    )
    bench.run()
