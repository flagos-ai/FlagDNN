from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _single(val):
    if isinstance(val, int) or isinstance(val, str):
        return val
    if len(val) != 1:
        raise RuntimeError(f"expected length 1, but got {val}")
    return int(val[0])


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


CONV1D_CASES = [
    ((16, 16, 127), (32, 16, 3), False, 1, 1, 1, 1),
    ((8, 32, 257), (64, 32, 5), True, 2, 2, 1, 1),
    ((32, 64, 1024), (64, 64, 3), False, 1, 1, 1, 1),
    ((16, 64, 2048), (128, 64, 7), False, 2, 3, 1, 1),
    ((8, 128, 4096), (128, 128, 3), True, 1, 2, 2, 1),
    ((32, 64, 1024), (128, 64, 1), False, 1, 0, 1, 1),
    ((16, 128, 2048), (256, 128, 1), True, 1, 0, 1, 1),
    ((16, 64, 1024), (128, 32, 3), False, 1, 1, 1, 2),
    ((16, 128, 1024), (128, 16, 3), False, 1, 1, 1, 8),
    ((32, 64, 1024), (64, 1, 5), False, 1, 2, 1, 64),
    ((64, 2048), (128, 64, 3), True, 1, 1, 1, 1),
]

CONV1D_FP64_CASES = [
    ((16, 16, 127), (32, 16, 3), False, 1, 1, 1, 1),
    ((8, 32, 257), (64, 32, 5), True, 2, 2, 1, 1),
    ((4, 16, 513), (32, 16, 7), False, 1, 3, 1, 1),
    ((4, 32, 512), (64, 32, 1), True, 1, 0, 1, 1),
    ((4, 32, 384), (64, 16, 3), False, 1, 1, 1, 2),
    ((4, 32, 512), (32, 1, 5), False, 1, 2, 1, 32),
    ((32, 512), (64, 32, 3), True, 1, 1, 1, 1),
]


def _is_sequence(val):
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


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

    batch, input_c, input_l, out_c, kernel, stride, padding, groups = case
    input_shape = (batch, input_c, input_l)
    weight_shape = (out_c, input_c // groups, kernel)
    return input_shape, weight_shape, False, stride, padding, 1, groups


def _output_shape(input_shape, weight_shape, stride, padding, dilation):
    c_out, _, kernel = weight_shape
    stride_w = _single(stride)
    pad_w = _single(padding)
    dil_w = _single(dilation)
    out_l = _conv_out_dim(input_shape[-1], pad_w, dil_w, kernel, stride_w)
    if len(input_shape) == 2:
        return (c_out, out_l)
    return (input_shape[0], c_out, out_l)


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


def conv1d_input_fn(case, dtype, device, max_peak_bytes):
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


def conv1d_output_shape(args):
    x, w, _, stride, padding, dilation, _ = args
    return _output_shape(
        tuple(x.shape), tuple(w.shape), stride, padding, dilation
    )


def conv1d_flops(args):
    x, w, _, stride, padding, dilation, groups = args
    c_out, _, kernel = w.shape
    stride_w = _single(stride)
    pad_w = _single(padding)
    dil_w = _single(dilation)
    out_l = _conv_out_dim(x.shape[-1], pad_w, dil_w, kernel, stride_w)
    batch = x.shape[0] if x.dim() == 3 else 1
    c_in = x.shape[-2]
    return 2 * batch * out_l * c_out * (c_in // groups) * kernel


@pytest.mark.conv1d
def test_conv1d():
    bench = base.ConvolutionBenchmark(
        op_name="conv1d",
        torch_op=F.conv1d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=conv1d_input_fn,
        output_shape_fn=conv1d_output_shape,
        flops_fn=conv1d_flops,
        cases=CONV1D_CASES,
        fp64_cases=CONV1D_FP64_CASES,
        max_peak_bytes=4 * 1024**3,
    )
    bench.run()
