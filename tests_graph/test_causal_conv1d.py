import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils

COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _reference_causal_conv1d(x, weight, bias, activation):
    kernel = weight.shape[1]
    conv_weight = weight.reshape(weight.shape[0], 1, kernel)
    y = F.conv1d(
        F.pad(x, (kernel - 1, 0)),
        conv_weight,
        bias=bias,
        groups=x.shape[1],
    )
    if activation == "silu":
        y = F.silu(y)
    return y


def _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation):
    @flag_dnn.graph
    def flag_dnn_causal_conv1d_graph(x, weight, bias):
        return flag_dnn.causal_conv1d(
            x, weight, bias=bias, activation=activation
        )

    compiled = flag_dnn.compile(
        flag_dnn_causal_conv1d_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(weight, "weight"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["causal_conv1d"]
    return compiled.run(x.clone(), weight.clone(), bias.clone())


@pytest.mark.causal_conv1d
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
@pytest.mark.parametrize("shape_kernel", [((2, 4, 16), 3), ((3, 8, 33), 5)])
@pytest.mark.parametrize("activation", ["identity", "silu"])
def test_graph_causal_conv1d_matches_torch(dtype, shape_kernel, activation):
    torch.manual_seed(0)
    shape, kernel = shape_kernel
    x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)
    weight = torch.randn(
        (shape[1], kernel), device=flag_dnn.device, dtype=dtype
    )
    bias = torch.randn((shape[1],), device=flag_dnn.device, dtype=dtype)
    expected = _reference_causal_conv1d(x, weight, bias, activation)
    actual = _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, expected, dtype, atol=atol)
