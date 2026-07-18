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

import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    conv1d_output_shape,
    conv2d_output_shape,
    conv3d_output_shape,
    cudnn_graph,
    execute_cudnn_graph,
    get_cudnn,
    to_spatial_tuple,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils

CONV_WGRAD_CASES = (
    # 1D wgrad cases.
    ((2, 4, 16), (6, 4, 3), 1, 1, None, None, 1),
    ((2, 4, 19), (7, 4, 3), 1, None, (2,), (0,), 2),
    # 2D wgrad cases.
    ((2, 8, 16, 16), (16, 8, 3, 3), 1, 1, None, None, 1),
    ((1, 4, 15, 17), (6, 4, 3, 5), (2, 1), (1, 2), None, None, 1),
    ((2, 3, 8, 8), (5, 3, 1, 1), 1, 0, None, None, 1),
    ((2, 4, 12, 13), (7, 4, 3, 3), (1, 2), None, (1, 0), (2, 3), 1),
    ((1, 5, 19, 21), (9, 5, 3, 3), 1, None, (2, 1), (0, 3), (2, 1)),
    # 3D wgrad cases.
    ((1, 2, 5, 6, 7), (4, 2, 3, 3, 3), 1, 1, None, None, 1),
    ((1, 2, 6, 7, 8), (3, 2, 2, 3, 3), 1, None, (1, 0, 1), (0, 1, 2), 1),
)


def _spatial_rank_from_shape(shape):
    return len(shape) - 2


def _to_cudnn_spatial(value, rank):
    return list(to_spatial_tuple(value, rank))


def _to_cudnn_convolution_mode(convolution_mode):
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    cudnn = get_cudnn()
    if mode == "CROSS_CORRELATION":
        return cudnn._compiled_module.convolution_mode.CROSS_CORRELATION
    if mode == "CONVOLUTION":
        return cudnn._compiled_module.convolution_mode.CONVOLUTION
    raise RuntimeError(
        "convolution_mode must be CROSS_CORRELATION or CONVOLUTION"
    )


def _conv_loss_shape(
    input_shape,
    filter_size,
    stride,
    padding,
    dilation,
    pre_padding=None,
    post_padding=None,
):
    rank = _spatial_rank_from_shape(input_shape)
    if rank == 1:
        return conv1d_output_shape(
            input_shape,
            filter_size,
            stride,
            padding,
            dilation,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )
    if rank == 2:
        return conv2d_output_shape(
            input_shape,
            filter_size,
            stride,
            padding,
            dilation,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )
    if rank == 3:
        return conv3d_output_shape(
            input_shape,
            filter_size,
            stride,
            padding,
            dilation,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )
    raise NotImplementedError("conv_wgrad tests support ranks 1, 2, and 3")


def _randn_conv_tensor(shape, dtype, device):
    return torch.randn(shape, device=device, dtype=dtype).contiguous()


def _cudnn_conv_wgrad(
    image,
    loss,
    filter_size,
    stride,
    padding,
    pre_padding,
    post_padding,
    dilation,
    cudnn_handle,
    convolution_mode="CROSS_CORRELATION",
):
    rank = image.dim() - 2
    graph = cudnn_graph(image.dtype, cudnn_handle)
    image_tensor = graph.tensor_like(image)
    loss_tensor = graph.tensor_like(loss)
    conv_kwargs = {
        "image": image_tensor,
        "loss": loss_tensor,
        "stride": _to_cudnn_spatial(stride, rank),
        "dilation": _to_cudnn_spatial(dilation, rank),
        "convolution_mode": _to_cudnn_convolution_mode(convolution_mode),
        "name": "conv_wgrad",
    }
    if pre_padding is None:
        conv_kwargs["padding"] = _to_cudnn_spatial(
            0 if padding is None else padding, rank
        )
    else:
        conv_kwargs["pre_padding"] = _to_cudnn_spatial(pre_padding, rank)
        conv_kwargs["post_padding"] = _to_cudnn_spatial(post_padding, rank)

    dw_tensor = graph.conv_wgrad(**conv_kwargs)
    output_template = torch.empty(
        filter_size, device=image.device, dtype=image.dtype
    )
    dw_tensor.set_dim(list(output_template.shape)).set_stride(
        list(output_template.stride())
    )
    return execute_cudnn_graph(
        graph,
        {image_tensor: image, loss_tensor: loss},
        dw_tensor,
        output_template,
        cudnn_handle,
        "conv_wgrad",
    )


def _run_flag_dnn_conv_wgrad_graph(
    image,
    loss,
    filter_size,
    stride,
    padding,
    pre_padding,
    post_padding,
    dilation,
    convolution_mode="CROSS_CORRELATION",
):
    rank = image.dim() - 2

    @flag_dnn.graph
    def flag_dnn_conv_wgrad_graph(image, loss):
        kwargs = {
            "stride": stride,
            "dilation": dilation,
            "convolution_mode": convolution_mode,
            "compute_data_type": "FLOAT",
            "name": "conv_wgrad",
        }
        if pre_padding is None:
            kwargs["padding"] = padding
        else:
            kwargs["pre_padding"] = pre_padding
            kwargs["post_padding"] = post_padding
        return flag_dnn.conv_wgrad(image, loss, filter_size, **kwargs)

    compiled = flag_dnn.compile(
        flag_dnn_conv_wgrad_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(image, "image"),
            flag_dnn.TensorSpec.from_tensor(loss, "loss"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["conv_wgrad"]
    attrs = compiled.graph.nodes[0].attrs
    assert attrs["filter_size"] == tuple(filter_size)
    assert attrs["stride"] == tuple(to_spatial_tuple(stride, rank))
    assert attrs["dilation"] == tuple(to_spatial_tuple(dilation, rank))
    if pre_padding is not None:
        assert attrs["pre_padding"] == tuple(
            to_spatial_tuple(pre_padding, rank)
        )
        assert attrs["post_padding"] == tuple(
            to_spatial_tuple(post_padding, rank)
        )
    return compiled.run(image.clone(), loss.clone())


@pytest.mark.graph
@pytest.mark.conv_wgrad
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", CONV_WGRAD_CASES)
def test_conv_wgrad_flag_dnn(
    cudnn_handle,
    dtype,
    case,
):
    torch.manual_seed(0)
    (
        input_shape,
        filter_size,
        stride,
        padding,
        pre_padding,
        post_padding,
        dilation,
    ) = case
    device = flag_dnn.device
    loss_shape = _conv_loss_shape(
        input_shape,
        filter_size,
        stride,
        0 if padding is None else padding,
        dilation,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )

    image = _randn_conv_tensor(input_shape, dtype, device)
    loss = _randn_conv_tensor(loss_shape, dtype, device)

    cudnn_out = _cudnn_conv_wgrad(
        image,
        loss,
        filter_size,
        stride,
        padding,
        pre_padding,
        post_padding,
        dilation,
        cudnn_handle,
    )
    flag_dnn_out = _run_flag_dnn_conv_wgrad_graph(
        image,
        loss,
        filter_size,
        stride,
        padding,
        pre_padding,
        post_padding,
        dilation,
    )

    atol = 8e-2 if dtype == torch.float16 else 2e-1
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)


@pytest.mark.graph
@pytest.mark.conv_wgrad
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
def test_conv_wgrad_convolution_mode(cudnn_handle, dtype):
    torch.manual_seed(0)
    device = flag_dnn.device
    input_shape = (2, 4, 8, 9)
    filter_size = (6, 4, 3, 3)
    stride = 1
    padding = 1
    dilation = 1
    loss_shape = conv2d_output_shape(
        input_shape, filter_size, stride, padding, dilation
    )
    image = torch.randn(input_shape, device=device, dtype=dtype)
    loss = torch.randn(loss_shape, device=device, dtype=dtype)

    cudnn_out = _cudnn_conv_wgrad(
        image,
        loss,
        filter_size,
        stride,
        padding,
        None,
        None,
        dilation,
        cudnn_handle,
        convolution_mode="CONVOLUTION",
    )
    flag_dnn_out = _run_flag_dnn_conv_wgrad_graph(
        image,
        loss,
        filter_size,
        stride,
        padding,
        None,
        None,
        dilation,
        convolution_mode="CONVOLUTION",
    )

    atol = 8e-2 if dtype == torch.float16 else 2e-1
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
