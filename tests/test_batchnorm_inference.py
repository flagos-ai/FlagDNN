import pytest
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests.cudnn_legacy import (
    CudnnLegacyError,
    cudnn_batchnorm_inference,
)
from tests import consts

CUDNN_COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _params(channels):
    mean = torch.randn(
        (1, channels, 1, 1), dtype=torch.float32, device=flag_dnn.device
    )
    inv_var = (
        torch.rand(
            (1, channels, 1, 1), dtype=torch.float32, device=flag_dnn.device
        )
        + 0.5
    )
    scale = torch.randn(
        (1, channels, 1, 1), dtype=torch.float32, device=flag_dnn.device
    )
    bias = torch.randn(
        (1, channels, 1, 1), dtype=torch.float32, device=flag_dnn.device
    )
    return mean, inv_var, scale, bias


def _run_flag_dnn_batchnorm_inference_graph(x, mean, inv_var, scale, bias):
    @flag_dnn.graph
    def flag_dnn_batchnorm_inference_graph(x, mean, inv_var, scale, bias):
        return flag_dnn.batchnorm_inference(
            x,
            mean,
            inv_var,
            scale,
            bias,
            compute_data_type="float32",
            name="batchnorm_inference",
        )

    compiled = flag_dnn.compile(
        flag_dnn_batchnorm_inference_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(mean, "mean"),
            flag_dnn.TensorSpec.from_tensor(inv_var, "inv_var"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "batchnorm_inference"
    ]
    return compiled.run(x.clone(), mean, inv_var, scale, bias)


@pytest.mark.cudnn_legacy
@pytest.mark.batchnorm_inference
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.BATCHNORM_INFERENCE_CASES)
def test_batchnorm_inference(
    dtype,
    shape,
):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device).contiguous()
    mean, inv_var, scale, bias = _params(shape[1])

    try:
        cudnn_out = cudnn_batchnorm_inference(x, mean, inv_var, scale, bias)
    except CudnnLegacyError as exc:
        pytest.skip(f"legacy cuDNN batchnorm_inference unavailable: {exc}")

    flag_dnn_out = _run_flag_dnn_batchnorm_inference_graph(
        x, mean, inv_var, scale, bias
    )

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
