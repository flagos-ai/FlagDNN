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


IDENTITY_SHAPES = utils.POINTWISE_SHAPES


def _make_input(shape, dtype):
    if dtype is torch.bool:
        return torch.randint(
            0, 2, shape, dtype=torch.int8, device=flag_dnn.device
        ).bool()
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    return torch.randint(-9, 10, shape, dtype=dtype, device=flag_dnn.device)


@pytest.mark.identity
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES)
@pytest.mark.parametrize("shape", IDENTITY_SHAPES)
def test_accuracy_identity(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = _make_input(shape, dtype)
    ref = utils.to_reference(inp, ref_kind=None)

    out = flag_dnn.identity(inp)

    assert out.dtype == inp.dtype
    assert out.shape == inp.shape
    if inp.numel() > 0:
        assert out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(out, ref)


@pytest.mark.identity
def test_identity_out_argument():
    inp = torch.randn((2, 3, 4), device=flag_dnn.device)
    out = torch.empty_like(inp)

    result = flag_dnn.identity(inp, out=out)

    assert result is out
    utils.gems_assert_equal(out, inp)


@pytest.mark.identity
def test_graph_identity_capture_and_run():
    x = torch.randn((2, 3, 4), device=flag_dnn.device)

    @flag_dnn.graph
    def fn(x):
        return flag_dnn.identity(
            x,
            compute_data_type="float32",
            name="identity",
        )

    compiled = flag_dnn.compile(fn, inputs=[x], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["identity"]
    assert compiled.graph.nodes[0].attrs["name"] == "identity"

    out = compiled.run(x)
    if x.numel() > 0:
        assert out.data_ptr() != x.data_ptr()
    utils.gems_assert_equal(out, x)
