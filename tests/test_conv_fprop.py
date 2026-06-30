import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    channels_last_randn,
    conv1d_output_template,
    conv2d_output_template,
    conv3d_output_template,
    cudnn_graph,
    execute_cudnn_graph,
    to_spatial_tuple,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils


CONV_FPROP_CASES = (
    # 1D fprop cases.
    ((2, 4, 16), (6, 4, 3), 1, 1, None, None, 1),
    ((2, 4, 19), (7, 4, 3), 1, None, (2,), (0,), 2),
    # 2D fprop cases.
    ((2, 8, 16, 16), (16, 8, 3, 3), 1, 1, None, None, 1),
    ((1, 4, 15, 17), (6, 4, 3, 5), (2, 1), (1, 2), None, None, 1),
    ((2, 3, 8, 8), (5, 3, 1, 1), 1, 0, None, None, 1),
    ((2, 4, 12, 13), (7, 4, 3, 3), (1, 2), None, (1, 0), (2, 3), 1),
    ((1, 5, 19, 21), (9, 5, 3, 3), 1, None, (2, 1), (0, 3), (2, 1)),
    # 3D fprop cases.
    ((1, 2, 5, 6, 7), (4, 2, 3, 3, 3), 1, 1, None, None, 1),
    ((1, 2, 6, 7, 8), (3, 2, 2, 3, 3), 1, None, (1, 0, 1), (0, 1, 2), 1),
)


def _spatial_rank(tensor):
    return tensor.dim() - 2


def _to_cudnn_spatial(value, rank):
    return list(to_spatial_tuple(value, rank))


def _conv_fprop_output_template(
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
    rank = len(input_shape) - 2
    if rank == 1:
        return conv1d_output_template(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            dtype,
            device,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )
    if rank == 2:
        return conv2d_output_template(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            dtype,
            device,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )
    if rank == 3:
        return conv3d_output_template(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            dtype,
            device,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )
    raise NotImplementedError("conv_fprop tests support ranks 1, 2, and 3")


def _randn_conv_tensor(shape, dtype, device):
    if len(shape) == 4:
        return channels_last_randn(shape, dtype, device)
    if len(shape) == 5:
        return torch.randn(shape, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last_3d
        )
    return torch.randn(shape, device=device, dtype=dtype)


def _cudnn_conv_fprop(
    x,
    weight,
    stride,
    padding,
    pre_padding,
    post_padding,
    dilation,
    cudnn_handle,
):
    rank = _spatial_rank(x)
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    weight_tensor = graph.tensor_like(weight)
    conv_kwargs = {
        "image": x_tensor,
        "weight": weight_tensor,
        "stride": _to_cudnn_spatial(stride, rank),
        "dilation": _to_cudnn_spatial(dilation, rank),
        "name": "conv_fprop",
    }
    if pre_padding is None:
        conv_kwargs["padding"] = _to_cudnn_spatial(
            0 if padding is None else padding, rank
        )
    else:
        conv_kwargs["pre_padding"] = _to_cudnn_spatial(pre_padding, rank)
        conv_kwargs["post_padding"] = _to_cudnn_spatial(post_padding, rank)
    y_tensor = graph.conv_fprop(**conv_kwargs)

    output_template = _conv_fprop_output_template(
        tuple(x.shape),
        tuple(weight.shape),
        stride,
        0 if padding is None else padding,
        dilation,
        x.dtype,
        x.device,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, weight_tensor: weight},
        y_tensor,
        output_template,
        cudnn_handle,
        "conv_fprop",
    )


def _run_flag_dnn_conv_fprop_graph(
    x, weight, stride, padding, pre_padding, post_padding, dilation
):
    rank = _spatial_rank(x)

    @flag_dnn.graph
    def flag_dnn_conv_fprop_graph(x, weight):
        kwargs = {
            "stride": stride,
            "dilation": dilation,
            "convolution_mode": "CROSS_CORRELATION",
            "compute_data_type": "FLOAT",
            "name": "conv_fprop",
        }
        if pre_padding is None:
            kwargs["padding"] = padding
        else:
            kwargs["pre_padding"] = pre_padding
            kwargs["post_padding"] = post_padding
        return flag_dnn.conv_fprop(x, weight, **kwargs)

    compiled = flag_dnn.compile(
        flag_dnn_conv_fprop_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(weight, "weight"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["conv_fprop"]
    attrs = compiled.graph.nodes[0].attrs
    assert attrs["stride"] == tuple(to_spatial_tuple(stride, rank))
    assert attrs["dilation"] == tuple(to_spatial_tuple(dilation, rank))
    if pre_padding is not None:
        assert attrs["pre_padding"] == tuple(
            to_spatial_tuple(pre_padding, rank)
        )
        assert attrs["post_padding"] == tuple(
            to_spatial_tuple(post_padding, rank)
        )
    return compiled.run(x.clone(), weight.clone())


@pytest.mark.graph
@pytest.mark.conv_fprop
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", CONV_FPROP_CASES)
def test_conv_fprop_flag_dnn(
    cudnn_handle,
    dtype,
    case,
):
    torch.manual_seed(0)
    (
        input_shape,
        weight_shape,
        stride,
        padding,
        pre_padding,
        post_padding,
        dilation,
    ) = case
    device = flag_dnn.device

    x = _randn_conv_tensor(input_shape, dtype, device)
    weight = _randn_conv_tensor(weight_shape, dtype, device)

    cudnn_out = _cudnn_conv_fprop(
        x,
        weight,
        stride,
        padding,
        pre_padding,
        post_padding,
        dilation,
        cudnn_handle,
    )
    flag_dnn_out = _run_flag_dnn_conv_fprop_graph(
        x, weight, stride, padding, pre_padding, post_padding, dilation
    )

    atol = 2e-2 if dtype == torch.float16 else 5e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
