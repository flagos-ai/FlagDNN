# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import cudnn
except (ImportError, OSError) as exc:
    pytest.skip(
        f"cuDNN frontend cannot be imported: {exc}",
        allow_module_level=True,
    )

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

CUDNN_COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def get_cudnn():
    return cudnn


def cudnn_data_type(dtype):
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    if dtype == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    if dtype == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    if dtype == torch.float64:
        return cudnn.data_type.DOUBLE
    if dtype == torch.bool:
        return cudnn.data_type.BOOLEAN
    raise TypeError(f"Unsupported dtype for cuDNN frontend: {dtype}")


def cudnn_graph(dtype, cudnn_handle):
    return cudnn.pygraph(
        io_data_type=cudnn_data_type(dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )


def skip_unsupported_cudnn_graph(exc, op_name):
    message = str(exc)
    if (
        isinstance(exc, cudnn.cudnnGraphNotSupportedError)
        or "CUDNN_STATUS_NOT_SUPPORTED" in message
        or "No valid engine configs" in message
    ):
        pytest.skip(f"cuDNN frontend does not support {op_name}: {exc}")
    raise exc


def execute_cudnn_graph(
    graph,
    exec_tensors,
    output_value,
    output_template,
    cudnn_handle,
    op_name,
    skip_unsupported=True,
):
    output_value.set_output(True).set_data_type(
        cudnn_data_type(output_template.dtype)
    )
    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        if skip_unsupported:
            skip_unsupported_cudnn_graph(exc, op_name)
        raise

    output = torch.empty_strided(
        tuple(output_value.get_dim()),
        tuple(output_value.get_stride()),
        device=output_template.device,
        dtype=output_template.dtype,
    )
    workspace = torch.empty(
        graph.get_workspace_size(),
        device=output_template.device,
        dtype=torch.uint8,
    )
    exec_tensors[output_value] = output
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return output


def to_pair(value):
    if isinstance(value, int):
        return [value, value]
    return list(value)


def to_spatial_tuple(value, rank):
    if isinstance(value, int):
        return (value,) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise RuntimeError(f"expected length {rank}, got {value}")
    return result


def normalize_conv_padding(
    rank, padding=0, pre_padding=None, post_padding=None
):
    if pre_padding is not None or post_padding is not None:
        if pre_padding is None or post_padding is None:
            raise RuntimeError(
                "both pre_padding and post_padding are required"
            )
        return (
            to_spatial_tuple(pre_padding, rank),
            to_spatial_tuple(post_padding, rank),
        )
    pad = to_spatial_tuple(padding, rank)
    return pad, pad


def conv_out_dim(input_size, pad_before, pad_after, dilation, kernel, stride):
    return (
        input_size + pad_before + pad_after - dilation * (kernel - 1) - 1
    ) // stride + 1


def conv1d_output_shape(
    input_shape,
    weight_shape,
    stride,
    padding=0,
    dilation=1,
    pre_padding=None,
    post_padding=None,
):
    stride = to_spatial_tuple(stride, 1)
    dilation = to_spatial_tuple(dilation, 1)
    pre, post = normalize_conv_padding(
        1, padding, pre_padding=pre_padding, post_padding=post_padding
    )
    n, _, length = input_shape
    c_out, _, kernel = weight_shape
    out_l = conv_out_dim(
        length, pre[0], post[0], dilation[0], kernel, stride[0]
    )
    return n, c_out, out_l


def conv2d_output_shape(
    input_shape,
    weight_shape,
    stride,
    padding=0,
    dilation=1,
    pre_padding=None,
    post_padding=None,
):
    stride = to_spatial_tuple(stride, 2)
    dilation = to_spatial_tuple(dilation, 2)
    pre, post = normalize_conv_padding(
        2, padding, pre_padding=pre_padding, post_padding=post_padding
    )
    n, _, h, w = input_shape
    c_out, _, kh, kw = weight_shape
    oh = conv_out_dim(h, pre[0], post[0], dilation[0], kh, stride[0])
    ow = conv_out_dim(w, pre[1], post[1], dilation[1], kw, stride[1])
    return n, c_out, oh, ow


def conv3d_output_shape(
    input_shape,
    weight_shape,
    stride,
    padding=0,
    dilation=1,
    pre_padding=None,
    post_padding=None,
):
    stride = to_spatial_tuple(stride, 3)
    dilation = to_spatial_tuple(dilation, 3)
    pre, post = normalize_conv_padding(
        3, padding, pre_padding=pre_padding, post_padding=post_padding
    )
    n, _, d, h, w = input_shape
    c_out, _, kd, kh, kw = weight_shape
    od = conv_out_dim(d, pre[0], post[0], dilation[0], kd, stride[0])
    oh = conv_out_dim(h, pre[1], post[1], dilation[1], kh, stride[1])
    ow = conv_out_dim(w, pre[2], post[2], dilation[2], kw, stride[2])
    return n, c_out, od, oh, ow


def channels_last_randn(shape, dtype, device):
    return torch.randn(shape, device=device, dtype=dtype).contiguous(
        memory_format=torch.channels_last
    )


def conv1d_output_template(
    input_shape,
    weight_shape,
    stride,
    padding,
    dilation,
    dtype,
    device,
    pre_padding=None,
    post_padding=None,
):
    shape = conv1d_output_shape(
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )
    # cuDNN frontend infers NCW output with NWC-style dense strides.
    return torch.empty_strided(
        shape,
        (shape[1] * shape[2], 1, shape[1]),
        device=device,
        dtype=dtype,
    )


def conv3d_output_template(
    input_shape,
    weight_shape,
    stride,
    padding,
    dilation,
    dtype,
    device,
    pre_padding=None,
    post_padding=None,
):
    return torch.empty(
        conv3d_output_shape(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            pre_padding=pre_padding,
            post_padding=post_padding,
        ),
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last_3d)


def conv2d_output_template(
    input_shape,
    weight_shape,
    stride,
    padding,
    dilation,
    dtype,
    device,
    pre_padding=None,
    post_padding=None,
):
    return torch.empty(
        conv2d_output_shape(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            pre_padding=pre_padding,
            post_padding=post_padding,
        ),
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)
