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


CONCATENATE_CASES = (
    (((2, 3, 4), (2, 5, 4), (2, 1, 4)), 1),
    (((1, 2), (3, 2), (4, 2)), 0),
    (((2, 3, 1), (2, 3, 5)), -1),
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


@pytest.mark.concatenate
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES)
@pytest.mark.parametrize(("shapes", "axis"), CONCATENATE_CASES)
def test_accuracy_concatenate(dtype, shapes, axis):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inputs = [_make_input(shape, dtype) for shape in shapes]

    out = flag_dnn.concatenate(inputs, axis=axis)
    ref = torch.cat(tuple(inputs), dim=axis)

    _assert_equal(out, ref)


@pytest.mark.concatenate
def test_concatenate_out_argument():
    a = torch.randn((2, 3, 4), device=flag_dnn.device)
    b = torch.randn((2, 5, 4), device=flag_dnn.device)
    out = torch.empty((2, 8, 4), device=flag_dnn.device)

    result = flag_dnn.concatenate([a, b], axis=1, out=out)

    assert result is out
    utils.gems_assert_equal(out, torch.cat((a, b), dim=1))


@pytest.mark.concatenate
def test_graph_concatenate_capture_and_run():
    a = torch.randn((2, 3, 4), device=flag_dnn.device)
    b = torch.randn((2, 5, 4), device=flag_dnn.device)

    @flag_dnn.graph
    def fn(a, b):
        return flag_dnn.concatenate([a, b], axis=1, name="concatenate")

    compiled = flag_dnn.compile(fn, inputs=[a, b], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["concatenate"]
    assert compiled.graph.nodes[0].attrs["axis"] == 1
    out = compiled.run(a, b)
    utils.gems_assert_equal(out, torch.cat((a, b), dim=1))
