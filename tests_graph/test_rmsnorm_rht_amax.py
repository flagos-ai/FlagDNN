import math

import pytest
import torch

import flag_dnn


def _hadamard_matrix(n: int, *, device: torch.device) -> torch.Tensor:
    matrix = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while matrix.shape[0] < n:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix


def _reference(
    x: torch.Tensor, w: torch.Tensor, eps: float, rows_per_cta: int
):
    m, n = x.shape
    x_f32 = x.float()
    rms = torch.sqrt((x_f32 * x_f32).mean(dim=-1, keepdim=True) + eps)
    y = x_f32 / rms * w.float().unsqueeze(0)

    had_block = 16
    hadamard = _hadamard_matrix(had_block, device=x.device) / math.sqrt(
        had_block
    )
    y = y.view(m, n // had_block, had_block)
    y = torch.matmul(y, hadamard).view(m, n)

    num_ctas = m // rows_per_cta
    amax = y.abs().view(num_ctas, rows_per_cta, n).amax(dim=(1, 2))
    return y.to(torch.bfloat16), amax.to(torch.float32)


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


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows_per_cta", [2, 4])
def test_graph_rmsnorm_rht_amax_matches_reference(rows_per_cta):
    torch.manual_seed(10)
    m = 8
    n = 2048
    eps = 1e-5
    x = torch.randn((m, n), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n,), device=flag_dnn.device, dtype=torch.bfloat16)

    actual = _run_graph(x, w, eps, rows_per_cta)
    expected_o, expected_amax = _reference(x, w, eps, rows_per_cta)

    assert set(actual.keys()) == {"o_tensor", "amax_tensor"}
    torch.testing.assert_close(
        actual["o_tensor"].float(), expected_o.float(), atol=4e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual["amax_tensor"], expected_amax, atol=2e-3, rtol=1e-3
    )


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_rmsnorm_rht_amax_squeezes_trailing_unit_dims():
    torch.manual_seed(11)
    m = 4
    n = 2048
    eps = 1e-5
    rows_per_cta = 2
    x = torch.randn((m, n, 1), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n, 1), device=flag_dnn.device, dtype=torch.bfloat16)

    actual = _run_graph(x, w, eps, rows_per_cta)
    expected_o, expected_amax = _reference(
        x.squeeze(-1), w.squeeze(-1), eps, rows_per_cta
    )

    assert actual["o_tensor"].shape == (m, n)
    assert actual["amax_tensor"].shape == (m // rows_per_cta,)
    torch.testing.assert_close(
        actual["o_tensor"].float(), expected_o.float(), atol=4e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual["amax_tensor"], expected_amax, atol=2e-3, rtol=1e-3
    )
