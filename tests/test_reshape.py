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


RESHAPE_CASES = (
    ((2, 3, 4), (6, 4)),
    ((2, 3, 4), (-1, 2, 4)),
    ((4, 1, 5), (2, 10)),
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


@pytest.mark.reshape
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES)
@pytest.mark.parametrize(("shape", "new_shape"), RESHAPE_CASES)
def test_accuracy_reshape(dtype, shape, new_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = _make_input(shape, dtype)

    out = flag_dnn.reshape(inp, new_shape)
    ref = torch.reshape(inp, new_shape)

    _assert_equal(out, ref)


@pytest.mark.reshape
def test_reshape_out_argument():
    inp = torch.randn((2, 3, 4), device=flag_dnn.device)
    out = torch.empty((6, 4), device=flag_dnn.device)

    result = flag_dnn.reshape(inp, (6, 4), out=out)

    assert result is out
    utils.gems_assert_equal(out, torch.reshape(inp, (6, 4)))


@pytest.mark.reshape
def test_graph_reshape_capture_and_run():
    x = torch.randn((2, 3, 4), device=flag_dnn.device)

    @flag_dnn.graph
    def fn(x):
        return flag_dnn.reshape(
            x,
            (6, 4),
            name="reshape",
            reshape_mode="LOGICAL",
        )

    compiled = flag_dnn.compile(fn, inputs=[x], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["reshape"]
    assert compiled.graph.nodes[0].attrs["shape"] == (6, 4)
    out = compiled.run(x)
    utils.gems_assert_equal(out, torch.reshape(x, (6, 4)))
