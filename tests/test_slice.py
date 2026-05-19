import builtins

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


SLICE_CASES = (
    (
        (2, 4, 5),
        (
            builtins.slice(None),
            builtins.slice(1, None),
            builtins.slice(None, None, 2),
        ),
    ),
    (
        (4, 6, 8),
        (
            builtins.slice(1, 4),
            builtins.slice(None, None, 2),
            builtins.slice(2, 8, 3),
        ),
    ),
    (
        (3, 5, 7, 2),
        (
            builtins.slice(None),
            builtins.slice(1, 5, 2),
            builtins.slice(None),
            builtins.slice(None),
        ),
    ),
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


@pytest.mark.slice
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES)
@pytest.mark.parametrize(("shape", "slices"), SLICE_CASES)
def test_accuracy_slice(dtype, shape, slices):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = _make_input(shape, dtype)

    out = flag_dnn.slice(inp, slices)
    ref = inp[tuple(slices)]

    _assert_equal(out, ref)


@pytest.mark.slice
def test_slice_out_argument():
    inp = torch.randn((2, 4, 5), device=flag_dnn.device)
    slices = [
        builtins.slice(None),
        builtins.slice(1, 4, 2),
        builtins.slice(None),
    ]
    out = torch.empty((2, 2, 5), device=flag_dnn.device)

    result = flag_dnn.slice(inp, slices, out=out)

    assert result is out
    utils.gems_assert_equal(out, inp[tuple(slices)])


@pytest.mark.slice
def test_graph_slice_capture_and_run():
    x = torch.randn((2, 4, 5), device=flag_dnn.device)
    slices = [
        builtins.slice(None),
        builtins.slice(1, 4, 2),
        builtins.slice(None),
    ]

    @flag_dnn.graph
    def fn(x):
        return flag_dnn.slice(
            x,
            slices,
            compute_data_type="float32",
            name="slice",
        )

    compiled = flag_dnn.compile(fn, inputs=[x], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["slice"]
    out = compiled.run(x)
    utils.gems_assert_equal(out, x[tuple(slices)])
