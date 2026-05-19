import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    channels_last_randn,
    conv2d_output_template,
    cudnn_graph,
    execute_cudnn_graph,
    to_pair,
)
import torch

import flag_dnn
from tests_graph import consts
from tests import accuracy_utils as utils


def _cudnn_conv2d(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    cudnn_handle,
):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    weight_tensor = graph.tensor_like(weight)
    y_tensor = graph.conv_fprop(
        image=x_tensor,
        weight=weight_tensor,
        padding=to_pair(padding),
        stride=to_pair(stride),
        dilation=to_pair(dilation),
    )
    exec_tensors = {x_tensor: x, weight_tensor: weight}

    if bias is not None:
        bias_4d = bias.view(1, -1, 1, 1).contiguous(
            memory_format=torch.channels_last
        )
        bias_tensor = graph.tensor_like(bias_4d)
        y_tensor = graph.bias(name="bias", input=y_tensor, bias=bias_tensor)
        exec_tensors[bias_tensor] = bias_4d

    output_template = conv2d_output_template(
        tuple(x.shape),
        tuple(weight.shape),
        stride,
        padding,
        dilation,
        x.dtype,
        x.device,
    )
    return execute_cudnn_graph(
        graph,
        exec_tensors,
        y_tensor,
        output_template,
        cudnn_handle,
        "conv2d",
    )


def _run_flag_dnn_conv2d_graph(x, weight, bias, stride, padding, dilation):
    if bias is None:

        @flag_dnn.graph
        def flag_dnn_conv2d_graph(x, weight):
            return flag_dnn.conv2d(
                x,
                weight,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
            )

        compiled = flag_dnn.compile(
            flag_dnn_conv2d_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(weight, "weight"),
            ],
            options={"cache": None},
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["conv2d"]
        return compiled.run(x.clone(), weight.clone())

    @flag_dnn.graph
    def flag_dnn_conv2d_graph(x, weight, bias):
        return flag_dnn.conv2d(
            x,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

    compiled = flag_dnn.compile(
        flag_dnn_conv2d_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(weight, "weight"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["conv2d"]
    assert len(compiled.graph.nodes[0].inputs) == 3
    return compiled.run(x.clone(), weight.clone(), bias.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.graph
@pytest.mark.conv2d
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.CONV2D_CASES)
def test_graph_conv2d_flag_dnn_matches_cudnn_frontend(
    cudnn_handle,
    dtype,
    case,
):
    torch.manual_seed(0)
    input_shape, weight_shape, has_bias, stride, padding, dilation = case
    device = flag_dnn.device

    x = channels_last_randn(input_shape, dtype, device)
    weight = channels_last_randn(weight_shape, dtype, device)
    bias = (
        torch.randn(weight_shape[0], device=device, dtype=dtype)
        if has_bias
        else None
    )

    cudnn_out = _cudnn_conv2d(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        cudnn_handle,
    )
    flag_dnn_out = _run_flag_dnn_conv2d_graph(
        x, weight, bias, stride, padding, dilation
    )

    atol = 2e-2 if dtype == torch.float16 else 5e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
