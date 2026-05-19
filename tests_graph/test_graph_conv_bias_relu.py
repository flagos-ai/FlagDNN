import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    channels_last_randn,
    conv2d_output_template,
    cudnn,
    cudnn_graph,
    execute_cudnn_graph,
    to_pair,
)
import torch

import flag_dnn
from tests_graph import consts
from tests import accuracy_utils as utils


def _cudnn_conv_bias_relu(
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

    bias_4d = bias.view(1, -1, 1, 1).contiguous(
        memory_format=torch.channels_last
    )
    bias_tensor = graph.tensor_like(bias_4d)
    y_tensor = graph.bias(name="bias", input=y_tensor, bias=bias_tensor)
    y_tensor = graph.relu(
        input=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="relu",
    )

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
        {x_tensor: x, weight_tensor: weight, bias_tensor: bias_4d},
        y_tensor,
        output_template,
        cudnn_handle,
        "conv_bias_relu",
    )


def _run_flag_dnn_conv_bias_relu_graph(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
):
    @flag_dnn.graph
    def flag_dnn_conv_bias_relu_graph(x, weight, bias):
        y = flag_dnn.conv2d(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        y = flag_dnn.bias_add(y, bias)
        return flag_dnn.relu(y)

    compiled = flag_dnn.compile(
        flag_dnn_conv_bias_relu_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(weight, "weight"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "fused_conv2d_bias_relu"
    ]
    assert (
        compiled.plan.debug_info["fusion"]["conv2d_bias_activation_fused"] == 1
    )
    return compiled.run(x.clone(), weight.clone(), bias.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.graph
@pytest.mark.conv2d
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.CONV_BIAS_RELU_CASES)
def test_graph_conv_bias_relu_matches_cudnn_frontend(
    cudnn_handle,
    dtype,
    case,
):
    torch.manual_seed(0)
    input_shape, weight_shape, stride, padding, dilation = case
    device = flag_dnn.device

    x = channels_last_randn(input_shape, dtype, device)
    weight = channels_last_randn(weight_shape, dtype, device)
    bias = torch.randn(weight_shape[0], device=device, dtype=dtype)

    cudnn_out = _cudnn_conv_bias_relu(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        cudnn_handle,
    )
    flag_dnn_graph_out = _run_flag_dnn_conv_bias_relu_graph(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
    )

    atol = 2e-2 if dtype == torch.float16 else 5e-2
    utils.gems_assert_close(flag_dnn_graph_out, cudnn_out, dtype, atol=atol)
