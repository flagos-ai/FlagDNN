import pytest
import torch

import flag_dnn
from . import accuracy_utils as utils


def _reference_gen_index(shape, axis, dtype):
    ndim = len(shape)
    normalized_axis = axis if axis >= 0 else axis + ndim
    view_shape = [1] * ndim
    view_shape[normalized_axis] = shape[normalized_axis]
    return (
        torch.arange(
            shape[normalized_axis],
            device=flag_dnn.device,
            dtype=dtype,
        )
        .reshape(view_shape)
        .expand(shape)
        .clone()
    )


@pytest.mark.gen_index
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_accuracy_gen_index(axis):
    inp = torch.empty((2, 3, 4), device=flag_dnn.device)
    ref = _reference_gen_index(tuple(inp.shape), axis, torch.int32)

    out = flag_dnn.gen_index(inp, axis)

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    utils.gems_assert_equal(out, ref)


@pytest.mark.gen_index
def test_gen_index_out_argument_and_dtype():
    inp = torch.empty((2, 3, 4), device=flag_dnn.device)
    out = torch.empty(inp.shape, dtype=torch.int64, device=flag_dnn.device)

    result = flag_dnn.gen_index(inp, 1, out=out, compute_data_type=torch.int64)

    assert result is out
    assert out.dtype == torch.int64
    ref = _reference_gen_index(tuple(inp.shape), 1, torch.int64)
    utils.gems_assert_equal(out, ref)


@pytest.mark.gen_index
def test_graph_gen_index_capture_and_run():
    x = torch.empty((2, 3, 4), device=flag_dnn.device)

    @flag_dnn.graph
    def fn(x):
        return flag_dnn.gen_index(
            x,
            axis=-2,
            compute_data_type="int64",
            name="gen_index",
        )

    compiled = flag_dnn.compile(fn, inputs=[x], options={"cache": None})

    assert [node.op_type for node in compiled.graph.nodes] == ["gen_index"]
    assert compiled.graph.nodes[0].attrs["axis"] == 1
    out = compiled.run(x)
    assert out.dtype == torch.int64
    ref = _reference_gen_index(tuple(x.shape), 1, torch.int64)
    utils.gems_assert_equal(out, ref)
