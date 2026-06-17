import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_data_type,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests_graph import consts


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _cudnn_mode(mode):
    return getattr(cudnn.reduction_mode, mode)


def _input(shape, mode, dtype):
    if mode == "MUL":
        return 0.9 + 0.2 * torch.rand(
            shape, dtype=dtype, device=flag_dnn.device
        )
    return torch.randn(shape, dtype=dtype, device=flag_dnn.device)


def _cudnn_reduction(x, dim, mode, cudnn_handle):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(x.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    x_tensor = graph.tensor_like(x)
    y_shape = list(x.shape)
    y_shape[dim] = 1
    y_tensor = graph.reduction(
        input=x_tensor,
        mode=_cudnn_mode(mode),
        compute_data_type=cudnn.data_type.FLOAT,
        name="reduction",
    )
    y_tensor.set_dim(y_shape).set_stride(_contiguous_stride(y_shape))
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty(tuple(y_shape), device=x.device, dtype=x.dtype),
        cudnn_handle,
        "reduction",
    )


def _run_flag_dnn_reduction_graph(x, dim, mode):
    @flag_dnn.graph
    def flag_dnn_reduction_graph(x):
        return flag_dnn.reduction(
            x,
            mode=mode,
            dim=dim,
            keepdim=True,
            compute_data_type="float32",
            name="reduction",
        )

    compiled = flag_dnn.compile(
        flag_dnn_reduction_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["reduction"]
    return compiled.run(x.clone())


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape, dim, mode", consts.REDUCTION_CASES)
def test_reduction(cudnn_handle, dtype, shape, dim, mode):
    torch.manual_seed(0)
    x = _input(shape, mode, dtype).contiguous()

    cudnn_out = _cudnn_reduction(x, dim, mode, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_reduction_graph(x, dim, mode)

    atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mode", ["ADD", "AVG", "MUL"])
def test_reduction_single_case_matches_cudnn(cudnn_handle, mode):
    torch.manual_seed(0)
    x = _input((2, 4, 8, 8), mode, torch.float32).contiguous()
    cudnn_out = _cudnn_reduction(x, 1, mode, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_reduction_graph(x, 1, mode)
    torch.testing.assert_close(flag_dnn_out, cudnn_out, atol=1e-4, rtol=1e-4)
