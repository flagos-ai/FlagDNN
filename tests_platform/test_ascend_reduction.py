import pytest
import torch

import flag_dnn

torch_npu = pytest.importorskip(
    "torch_npu", reason="Ascend reduction tests require torch_npu"
)


DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MODES = (
    "ADD",
    "AVG",
    "MUL",
    "NORM1",
    "NORM2",
    "MIN",
    "MAX",
    "AMAX",
    "MUL_NO_ZEROS",
)
_ASCEND_AVAILABLE = (
    flag_dnn.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)
pytestmark = pytest.mark.skipif(
    not _ASCEND_AVAILABLE,
    reason="reduction tests require an available Ascend NPU",
)


@pytest.fixture(scope="module", autouse=True)
def select_npu():
    torch.npu.set_device(0)
    yield
    torch.npu.synchronize()


def _normalized_dims(input_cpu, dim):
    if dim is None:
        values = tuple(range(input_cpu.ndim))
    elif isinstance(dim, int):
        values = (dim,)
    else:
        values = tuple(dim)
    return tuple(sorted({item % input_cpu.ndim for item in values}))


def _product_reference(work, dims, keepdim):
    result = work
    ordered = dims if keepdim else tuple(sorted(dims, reverse=True))
    for axis in ordered:
        result = torch.prod(result, dim=axis, keepdim=keepdim)
    return result


def _reference_reduction(input_cpu, mode, dim, keepdim, dtype=None):
    dims = _normalized_dims(input_cpu, dim)
    target_dtype = input_cpu.dtype if dtype is None else dtype
    work = input_cpu.to(torch.float32)
    if not dims:
        result = work
    elif mode == "ADD":
        result = torch.sum(work, dim=dims, keepdim=keepdim)
    elif mode == "AVG":
        result = torch.mean(work, dim=dims, keepdim=keepdim)
    elif mode == "MUL":
        result = _product_reference(work, dims, keepdim)
    elif mode == "MUL_NO_ZEROS":
        values = torch.where(work == 0, torch.ones_like(work), work)
        result = _product_reference(values, dims, keepdim)
    elif mode == "NORM1":
        result = torch.sum(work.abs(), dim=dims, keepdim=keepdim)
    elif mode == "NORM2":
        result = torch.sqrt(
            torch.sum(work.square(), dim=dims, keepdim=keepdim)
        )
    elif mode == "MIN":
        result = torch.amin(work, dim=dims, keepdim=keepdim)
    elif mode == "MAX":
        result = torch.amax(work, dim=dims, keepdim=keepdim)
    elif mode == "AMAX":
        result = torch.amax(work.abs(), dim=dims, keepdim=keepdim)
    else:
        raise AssertionError(f"unhandled reference mode: {mode}")
    return result.to(target_dtype)


def _assert_close(actual, expected, dtype):
    assert actual.device.type == "npu"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.npu.synchronize()
    tolerance = 1e-4 if dtype == torch.float32 else 8e-2
    torch.testing.assert_close(
        actual.detach().cpu(),
        expected,
        atol=tolerance,
        rtol=tolerance,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ("ADD", "AVG", "NORM1", "NORM2"))
def test_ascend_reduction_sum_and_norm_modes(dtype, mode):
    input_cpu = torch.linspace(-1.5, 2.0, 120).reshape(2, 3, 4, 5).to(dtype)
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"),
        mode,
        dim=(1, 2),
        keepdim=True,
    )
    expected = _reference_reduction(input_cpu, mode, dim=(1, 2), keepdim=True)
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ("MUL", "MUL_NO_ZEROS"))
def test_ascend_reduction_product_modes(dtype, mode):
    input_cpu = (
        torch.linspace(0.999, 1.001, 264).reshape(2, 3, 44, 1).to(dtype)
    )
    if mode == "MUL_NO_ZEROS":
        input_cpu[:, 1, 2, :] = 0
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"),
        mode,
        dim=(1, 2),
        keepdim=True,
    )
    expected = _reference_reduction(input_cpu, mode, dim=(1, 2), keepdim=True)
    _assert_close(actual, expected, dtype)


def test_ascend_reduction_product_zero_semantics_differ():
    input_cpu = torch.tensor([[0.0, 2.0, 3.0]], dtype=torch.float32)
    input_npu = input_cpu.to("npu:0")
    product = flag_dnn.reduction(input_npu, "MUL", dim=1, keepdim=False)
    no_zeros = flag_dnn.reduction(
        input_npu, "MUL_NO_ZEROS", dim=1, keepdim=False
    )
    _assert_close(product, torch.tensor([0.0]), torch.float32)
    _assert_close(no_zeros, torch.tensor([6.0]), torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ("MIN", "MAX", "AMAX"))
def test_ascend_reduction_extrema_modes(dtype, mode):
    input_cpu = torch.linspace(-3.0, 2.5, 120).reshape(2, 3, 4, 5).to(dtype)
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"),
        mode,
        dim=(1, 2),
        keepdim=True,
    )
    expected = _reference_reduction(input_cpu, mode, dim=(1, 2), keepdim=True)
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize(
    "mode, dim, keepdim",
    (
        ("ADD", 3, False),
        ("AVG", -1, True),
        ("NORM1", (1, 2), False),
        ("MAX", (0, 2), True),
        ("AMAX", None, False),
        ("ADD", (1, -3, 1), True),
    ),
)
def test_ascend_reduction_dimension_layouts(mode, dim, keepdim):
    input_cpu = torch.linspace(-1.5, 2.0, 120).reshape(2, 3, 4, 5)
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"), mode, dim=dim, keepdim=keepdim
    )
    expected = _reference_reduction(input_cpu, mode, dim, keepdim)
    _assert_close(actual, expected, torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("keepdim", (False, True))
def test_ascend_reduction_keepdim_for_all_dtypes(dtype, keepdim):
    input_cpu = torch.linspace(-1.0, 1.0, 120).reshape(2, 3, 4, 5).to(dtype)
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"),
        "ADD",
        dim=(1, 2),
        keepdim=keepdim,
    )
    expected = _reference_reduction(
        input_cpu, "ADD", dim=(1, 2), keepdim=keepdim
    )
    _assert_close(actual, expected, dtype)


def test_ascend_reduction_empty_dimension_list_is_identity():
    input_cpu = (
        torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4).to(torch.float16)
    )
    input_npu = input_cpu.to("npu:0")
    actual = flag_dnn.reduction(input_npu, "ADD", dim=[])
    assert actual.data_ptr() == input_npu.data_ptr()
    _assert_close(actual, input_cpu, torch.float16)


def test_ascend_reduction_explicit_output_dtype():
    input_cpu = (
        torch.linspace(-1.0, 1.0, 120).reshape(2, 3, 4, 5).to(torch.float16)
    )
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"),
        "ADD",
        dim=(1, 2),
        keepdim=False,
        dtype=torch.float32,
    )
    expected = _reference_reduction(
        input_cpu,
        "ADD",
        dim=(1, 2),
        keepdim=False,
        dtype=torch.float32,
    )
    _assert_close(actual, expected, torch.float32)


@pytest.mark.parametrize(
    "alias, canonical",
    (("SUM", "ADD"), ("MEAN", "AVG"), ("PROD", "MUL")),
)
def test_ascend_reduction_preserves_mode_aliases(alias, canonical):
    input_cpu = torch.linspace(0.98, 1.02, 120).reshape(2, 3, 4, 5)
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"), alias, dim=(1, 2), keepdim=True
    )
    expected = _reference_reduction(
        input_cpu, canonical, dim=(1, 2), keepdim=True
    )
    _assert_close(actual, expected, torch.float32)


@pytest.mark.parametrize("dim", (1, -2))
def test_ascend_reduction_accepts_int_dimensions(dim):
    input_cpu = torch.linspace(-2.0, 2.0, 64).reshape(2, 4, 8)
    actual = flag_dnn.reduction(
        input_cpu.to("npu:0"), "MAX", dim=dim, keepdim=True
    )
    expected = _reference_reduction(input_cpu, "MAX", dim, keepdim=True)
    _assert_close(actual, expected, torch.float32)


def test_ascend_reduction_rejects_fp64_before_launch():
    input_cpu = torch.ones((2, 4), dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="float64"):
        flag_dnn.reduction(input_cpu, "MAX", dim=(1,))


def test_ascend_reduction_rejects_fp64_output_before_launch():
    input_npu = torch.ones((2, 4), device="npu:0")
    with pytest.raises(NotImplementedError, match="float64"):
        flag_dnn.reduction(input_npu, "MAX", dim=(1,), dtype=torch.float64)


def test_ascend_reduction_rejects_cpu_before_launch():
    input_cpu = torch.ones((2, 4), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="npu"):
        flag_dnn.reduction(input_cpu, "MAX", dim=(1,))


def test_ascend_reduction_rejects_out_of_range_dimension():
    input_npu = torch.ones((2, 4), device="npu:0")
    with pytest.raises(IndexError, match="Dimension out of range"):
        flag_dnn.reduction(input_npu, "MAX", dim=2)


_EMPTY_IDENTITIES = {
    "ADD": 0.0,
    "AVG": float("nan"),
    "MUL": 1.0,
    "NORM1": 0.0,
    "NORM2": 0.0,
    "MUL_NO_ZEROS": 1.0,
}
_EMPTY_EXTREMA = ("MIN", "MAX", "AMAX")


@pytest.mark.parametrize("mode", MODES)
def test_ascend_reduction_empty_reduced_extent(mode):
    input_npu = torch.empty((2, 0, 3), device="npu:0")
    if mode in _EMPTY_EXTREMA:
        with pytest.raises(RuntimeError, match="empty"):
            flag_dnn.reduction(input_npu, mode, dim=1, keepdim=True)
        return

    actual = flag_dnn.reduction(input_npu, mode, dim=1, keepdim=True)
    expected = torch.full(
        (2, 1, 3), _EMPTY_IDENTITIES[mode], dtype=torch.float32
    )
    _assert_close(actual, expected, torch.float32)


@pytest.mark.parametrize("mode", MODES)
def test_ascend_reduction_empty_non_reduced_axis(mode):
    input_npu = torch.empty((0, 3), device="npu:0")
    actual = flag_dnn.reduction(input_npu, mode, dim=1, keepdim=False)
    expected = torch.empty((0,), dtype=torch.float32)
    _assert_close(actual, expected, torch.float32)


@pytest.mark.parametrize("mode", MODES)
def test_ascend_reduction_graph(mode):
    if mode in ("MUL", "MUL_NO_ZEROS"):
        input_cpu = torch.linspace(0.98, 1.02, 120).reshape(2, 3, 4, 5)
        input_cpu[:, 1, 2, :] = 0
    else:
        input_cpu = torch.linspace(-1.5, 2.0, 120).reshape(2, 3, 4, 5)
    input_npu = input_cpu.to("npu:0")

    @flag_dnn.graph
    def reduction_graph(x):
        return flag_dnn.reduction(
            x,
            mode,
            dim=(1, 2),
            keepdim=True,
            compute_data_type="float32",
            name=f"reduction_{mode.lower()}",
        )

    compiled = flag_dnn.compile(
        reduction_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(input_npu, "x")],
        options={"backend": "npu", "cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["reduction"]
    actual = compiled.run(input_npu.clone())
    expected = _reference_reduction(input_cpu, mode, dim=(1, 2), keepdim=True)
    _assert_close(actual, expected, torch.float32)
