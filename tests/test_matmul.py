import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_data_type,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_MATMUL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    *_FP8_DTYPES,
)


def _matmul_output_shape(a, b):
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return (*batch_shape, a.shape[-2], b.shape[-1])


def _cudnn_tensor(graph, value):
    if value.dtype in _FP8_DTYPES:
        return graph.tensor(
            dim=tuple(value.shape),
            stride=tuple(value.stride()),
            data_type=cudnn_data_type(value.dtype),
        )
    return graph.tensor_like(value)


def _cudnn_matmul(
    a,
    b,
    cudnn_handle,
    *,
    out_dtype=None,
    compute_mode="float32",
):
    out_dtype = a.dtype if out_dtype is None else out_dtype
    compute_type = (
        cudnn.data_type.FAST_FLOAT_FOR_FP8
        if compute_mode == "fast_float_for_fp8"
        else cudnn.data_type.FLOAT
    )
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(a.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=compute_type,
        handle=cudnn_handle,
    )
    a_tensor = _cudnn_tensor(graph, a)
    b_tensor = _cudnn_tensor(graph, b)
    matmul_tensor = graph.matmul(
        A=a_tensor,
        B=b_tensor,
        compute_data_type=compute_type,
        name="matmul",
    )
    if a.dtype in _FP8_DTYPES:
        matmul_tensor.set_data_type(cudnn.data_type.FLOAT)
    y_tensor = graph.identity(
        input=matmul_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="output_cast",
    )
    y_tensor.set_output(True).set_data_type(cudnn_data_type(out_dtype))
    return execute_cudnn_graph(
        graph,
        {a_tensor: a, b_tensor: b},
        y_tensor,
        torch.empty(
            _matmul_output_shape(a, b),
            device=a.device,
            dtype=out_dtype,
        ),
        cudnn_handle,
        "matmul",
    )


def _cudnn_matmul_materialized_batch(
    a,
    b,
    cudnn_handle,
    *,
    out_dtype,
    compute_mode,
):
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_matrix_shape = tuple(a.shape[-2:])
    b_matrix_shape = tuple(b.shape[-2:])
    a_batched = (
        a.expand((*batch_shape, *a_matrix_shape))
        .reshape(-1, *a_matrix_shape)
        .contiguous()
    )
    b_batched = (
        b.expand((*batch_shape, *b_matrix_shape))
        .reshape(-1, *b_matrix_shape)
        .contiguous()
    )
    output = _cudnn_matmul(
        a_batched,
        b_batched,
        cudnn_handle,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )
    return output.reshape(_matmul_output_shape(a, b))


def _compile_flag_dnn_matmul_graph(
    a,
    b,
    *,
    out_dtype=None,
    compute_mode="float32",
):
    @flag_dnn.graph
    def flag_dnn_matmul_graph(a, b):
        return flag_dnn.matmul(
            a,
            b,
            compute_data_type=compute_mode,
            out_dtype=out_dtype,
            name="matmul",
        )

    compiled = flag_dnn.compile(
        flag_dnn_matmul_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(a, "a"),
            flag_dnn.TensorSpec.from_tensor(b, "b"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]
    return compiled


def _run_flag_dnn_matmul_graph(
    a,
    b,
    *,
    out_dtype=None,
    compute_mode="float32",
):
    compiled = _compile_flag_dnn_matmul_graph(
        a, b, out_dtype=out_dtype, compute_mode=compute_mode
    )
    return compiled.run(a.clone(), b.clone())


def _torch_fp32_matmul(a, b):
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        return torch.matmul(a, b)
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape_pair", consts.MATMUL_CASES)
def test_matmul(cudnn_handle, dtype, shape_pair):
    torch.manual_seed(0)
    a_shape, b_shape = shape_pair
    a = torch.randn(a_shape, dtype=dtype, device=flag_dnn.device)
    b = torch.randn(b_shape, dtype=dtype, device=flag_dnn.device)

    flag_dnn_out = _run_flag_dnn_matmul_graph(a, b)

    if dtype == torch.bfloat16:
        expected = _cudnn_matmul(a, b, cudnn_handle)
        atol = 1e-1
    elif dtype == torch.float16:
        expected = _cudnn_matmul(a, b, cudnn_handle)
        atol = 5e-2
    else:
        expected = _torch_fp32_matmul(a, b)
        atol = 2e-4
    utils.gems_assert_close(flag_dnn_out, expected, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "shape_pair",
    (
        ((1, 512, 512), (1, 512, 512)),
        ((1, 1024, 512), (1, 512, 1024)),
    ),
    ids=("general_512", "direct_1024"),
)
def test_matmul_fp32_ieee_dispatch(shape_pair):
    torch.manual_seed(7)
    a_shape, b_shape = shape_pair
    a = torch.randn(a_shape, dtype=torch.float32, device=flag_dnn.device)
    b = torch.randn(b_shape, dtype=torch.float32, device=flag_dnn.device)

    expected = _torch_fp32_matmul(a, b)
    flag_dnn_out = _run_flag_dnn_matmul_graph(a, b)

    utils.gems_assert_close(flag_dnn_out, expected, torch.float32, atol=2e-4)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _MATMUL_DTYPES)
@pytest.mark.parametrize("out_dtype", _MATMUL_DTYPES)
def test_matmul_input_output_dtype_matrix(
    cudnn_handle,
    input_dtype,
    out_dtype,
):
    torch.manual_seed(11)
    a = torch.randint(-1, 2, (1, 32, 32), device=flag_dnn.device).to(
        input_dtype
    )
    b = torch.randint(-1, 2, (1, 32, 32), device=flag_dnn.device).to(
        input_dtype
    )
    compute_mode = (
        "fast_float_for_fp8" if input_dtype in _FP8_DTYPES else "float32"
    )

    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=out_dtype,
        compute_mode=compute_mode,
    )

    assert actual.dtype == out_dtype
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=0, rtol=0
    )


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_fp32_compute_modes(cudnn_handle):
    torch.manual_seed(13)
    a = torch.randn((1, 128, 512), dtype=torch.float32, device=flag_dnn.device)
    b = torch.randn((1, 512, 128), dtype=torch.float32, device=flag_dnn.device)

    ieee = _run_flag_dnn_matmul_graph(a, b, compute_mode="float32")
    tf32 = _run_flag_dnn_matmul_graph(a, b, compute_mode="tf32")
    expected_ieee = _torch_fp32_matmul(a, b)
    expected_tf32 = _cudnn_matmul(a, b, cudnn_handle, compute_mode="tf32")

    utils.gems_assert_close(ieee, expected_ieee, torch.float32, atol=2e-4)
    utils.gems_assert_close(tf32, expected_tf32, torch.float32, atol=2e-4)
    assert (ieee - tf32).abs().max().item() > 1e-4


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _FP8_DTYPES)
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    (
        ((32, 64), (64, 24)),
        ((2, 32, 64), (2, 64, 24)),
        ((2, 1, 32, 64), (3, 64, 24)),
    ),
    ids=("2d", "prepared_3d", "broadcast"),
)
def test_matmul_fp8_rank_routes_match_cudnn(
    cudnn_handle,
    input_dtype,
    a_shape,
    b_shape,
):
    torch.manual_seed(17)
    a = torch.randint(-1, 2, a_shape, device=flag_dnn.device).to(input_dtype)
    b = torch.randint(-1, 2, b_shape, device=flag_dnn.device).to(input_dtype)

    expected = _cudnn_matmul_materialized_batch(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _FP8_DTYPES)
def test_matmul_fp8_k_contiguous_b_matches_cudnn(
    cudnn_handle,
    input_dtype,
):
    torch.manual_seed(19)
    a = torch.randint(-1, 2, (2, 32, 64), device=flag_dnn.device).to(
        input_dtype
    )
    b_storage = torch.randint(-1, 2, (2, 24, 64), device=flag_dnn.device).to(
        input_dtype
    )
    b = b_storage.transpose(-2, -1)
    assert not b.is_contiguous()
    assert b.stride(-2) == 1

    expected = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    actual = _run_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", _FP8_DTYPES)
def test_matmul_fp8_prepared_cache_preserves_output_dtype(
    cudnn_handle,
    input_dtype,
):
    torch.manual_seed(23)
    a = torch.randint(-1, 2, (2, 32, 64), device=flag_dnn.device).to(
        input_dtype
    )
    b = torch.randint(-1, 2, (2, 64, 24), device=flag_dnn.device).to(
        input_dtype
    )
    compiled = _compile_flag_dnn_matmul_graph(
        a,
        b,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )

    expected_first = _cudnn_matmul(
        a,
        b,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    first = compiled.run(a.clone(), b.clone())
    first_snapshot = first.clone()

    a_second = torch.flip(a.float(), dims=(-1,)).to(input_dtype)
    b_second = torch.roll(b.float(), shifts=1, dims=-1).to(input_dtype)
    expected_second = _cudnn_matmul(
        a_second,
        b_second,
        cudnn_handle,
        out_dtype=torch.float32,
        compute_mode="fast_float_for_fp8",
    )
    second = compiled.run(a_second, b_second)

    assert first.dtype == second.dtype == torch.float32
    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(first_snapshot, expected_first, atol=0, rtol=0)
    torch.testing.assert_close(second, expected_second, atol=0, rtol=0)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("input_dtype", "compute_mode"),
    (
        (torch.float16, "tf32"),
        (torch.float32, "fast_float_for_fp8"),
    ),
)
def test_matmul_rejects_incompatible_compute_mode(
    input_dtype,
    compute_mode,
):
    a = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=input_dtype)
    b = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=input_dtype)

    with pytest.raises(
        RuntimeError, match="unsupported matmul compute_data_type"
    ) as exc:
        _run_flag_dnn_matmul_graph(a, b, compute_mode=compute_mode)

    message = str(exc.value)
    assert compute_mode in message
    assert str(input_dtype) in message


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_rejects_unsupported_output_dtype():
    a = torch.ones((1, 8, 8), device=flag_dnn.device)
    b = torch.ones((1, 8, 8), device=flag_dnn.device)

    with pytest.raises(RuntimeError, match="matmul out_dtype must be"):
        _run_flag_dnn_matmul_graph(a, b, out_dtype=torch.float64)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_rejects_mixed_input_dtypes():
    a = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=torch.float16)
    b = torch.ones((1, 8, 8), device=flag_dnn.device, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="same dtype"):
        _run_flag_dnn_matmul_graph(a, b)
