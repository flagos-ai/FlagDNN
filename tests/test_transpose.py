import pytest
import torch

import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    BOOL_DTYPES = [torch.bool]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES
    BOOL_DTYPES = utils.BOOL_TYPES


TRANSPOSE_CASES = (
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4, 5), (0, 2, 3, 1)),
    ((4, 1, 5), (-1, 0, 1)),
)


def _make_input(shape, dtype):
    if dtype is torch.bool:
        return torch.randint(
            0, 2, shape, dtype=torch.int8, device=flag_dnn.device
        ).bool()
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    return torch.randint(-9, 10, shape, dtype=dtype, device=flag_dnn.device)


def _assert_equal(out, ref):
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    utils.gems_assert_equal(out, ref)


@pytest.mark.transpose
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES)
@pytest.mark.parametrize(("shape", "permutation"), TRANSPOSE_CASES)
def test_accuracy_transpose(dtype, shape, permutation):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = _make_input(shape, dtype)

    out = flag_dnn.transpose(inp, permutation)
    ref = torch.permute(inp, permutation)

    _assert_equal(out, ref)


@pytest.mark.transpose
def test_transpose_dim_swap_and_out_argument():
    inp = torch.randn((2, 3, 4), device=flag_dnn.device)
    out = torch.empty((4, 3, 2), device=flag_dnn.device)

    with pytest.raises(NotImplementedError, match="view-only graph utility"):
        flag_dnn.transpose(inp, 0, 2, out=out)


@pytest.mark.transpose
def test_graph_transpose_capture_and_run():
    x = torch.randn((2, 3, 4), device=flag_dnn.device)

    @flag_dnn.graph
    def fn(x):
        return flag_dnn.transpose(
            x,
            (2, 0, 1),
            compute_data_type="float32",
            name="transpose",
        )

    compiled = flag_dnn.compile(fn, inputs=[x], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["transpose"]
    assert compiled.graph.nodes[0].attrs["permutation"] == (2, 0, 1)
    out = compiled.run(x)
    utils.gems_assert_equal(out, torch.permute(x, (2, 0, 1)))
