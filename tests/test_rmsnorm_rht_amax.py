import pytest
import torch

import flag_dnn
from tests.base import get_cudnn


def _cudnn_rmsnorm_rht_amax(x, w, eps, rows_per_cta):
    cudnn = get_cudnn()
    try:
        wrapper = getattr(cudnn, "rmsnorm_rht_amax_wrapper_sm100")
    except (AttributeError, ImportError, RuntimeError) as exc:
        pytest.skip(f"cuDNN RMSNorm RHT amax wrapper is unavailable: {exc}")
    try:
        out = wrapper(x, w, eps=eps, rows_per_cta=rows_per_cta)
        torch.cuda.synchronize()
        return out
    except (AssertionError, ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(
            "cuDNN RMSNorm RHT amax wrapper is unsupported here: " f"{exc}"
        )


def _run_graph(x, w, eps, rows_per_cta):
    @flag_dnn.graph
    def fn(x, w):
        return flag_dnn.rmsnorm_rht_amax_wrapper_sm100(
            x, w, eps=eps, rows_per_cta=rows_per_cta
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(w, "w"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "rmsnorm_rht_amax_wrapper_sm100"
    ]
    return compiled.run(x.clone(), w.clone())


def _assert_output_close(actual, cudnn_out):
    assert set(actual.keys()) == {"o_tensor", "amax_tensor"}
    assert set(cudnn_out.keys()) == {"o_tensor", "amax_tensor"}
    torch.testing.assert_close(
        actual["o_tensor"].float(),
        cudnn_out["o_tensor"].float(),
        atol=4e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual["amax_tensor"],
        cudnn_out["amax_tensor"],
        atol=2e-3,
        rtol=1e-3,
    )


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows_per_cta", [2, 4])
def test_rmsnorm_rht_amax_matches_cudnn(rows_per_cta):
    torch.manual_seed(10)
    m = 8
    n = 2048
    eps = 1e-5
    x = torch.randn((m, n), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n,), device=flag_dnn.device, dtype=torch.bfloat16)

    cudnn_out = _cudnn_rmsnorm_rht_amax(x, w, eps, rows_per_cta)
    actual = _run_graph(x, w, eps, rows_per_cta)
    _assert_output_close(actual, cudnn_out)


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_rht_amax_squeezes_trailing_unit_dims():
    torch.manual_seed(11)
    m = 4
    n = 2048
    eps = 1e-5
    rows_per_cta = 2
    x = torch.randn((m, n, 1), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n, 1), device=flag_dnn.device, dtype=torch.bfloat16)

    cudnn_out = _cudnn_rmsnorm_rht_amax(x, w, eps, rows_per_cta)
    actual = _run_graph(x, w, eps, rows_per_cta)

    assert actual["o_tensor"].shape == (m, n)
    assert actual["amax_tensor"].shape == (m // rows_per_cta,)
    _assert_output_close(actual, cudnn_out)
