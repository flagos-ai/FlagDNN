import pytest
import torch

import flag_dnn


MM_CASES = [
    (0, 3, 4),
    (2, 0, 4),
    (2, 3, 0),
    (1, 1, 1),
    (2, 3, 4),
    (7, 5, 3),
    (16, 17, 15),
    (32, 64, 16),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    if dtype in (torch.float32, torch.complex64):
        return dict(rtol=1e-4, atol=1e-4)
    return dict(rtol=1e-10, atol=1e-10)


def make_tensor(shape, dtype):
    if dtype in (torch.complex64, torch.complex128):
        real_dtype = (
            torch.float32 if dtype == torch.complex64 else torch.float64
        )
        real = torch.empty(shape, dtype=real_dtype, device=flag_dnn.device)
        imag = torch.empty(shape, dtype=real_dtype, device=flag_dnn.device)
        real.uniform_(-1.0, 1.0)
        imag.uniform_(-1.0, 1.0)
        return torch.complex(real, imag)

    return torch.empty(shape, dtype=dtype, device=flag_dnn.device).uniform_(
        -1.0, 1.0
    )


@pytest.mark.mm
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
@pytest.mark.parametrize("m, k, n", MM_CASES)
@pytest.mark.parametrize("use_out", [False, True])
def test_accuracy_mm(dtype, m, k, n, use_out):
    if dtype in (torch.float64, torch.complex128):
        if not flag_dnn.runtime.device.support_fp64:
            pytest.skip("Device does not support float64")

    a = make_tensor((m, k), dtype)
    b = make_tensor((k, n), dtype)
    a_ref = a.clone()
    b_ref = b.clone()
    a_custom = a.clone()
    b_custom = b.clone()

    out_ref = torch.mm(a_ref, b_ref)

    if use_out:
        out_buf = torch.empty((m, n), dtype=dtype, device=flag_dnn.device)
        with flag_dnn.use_dnn():
            out_custom = torch.mm(a_custom, b_custom, out=out_buf)

        assert out_custom.data_ptr() == out_buf.data_ptr()
        torch.testing.assert_close(out_buf, out_ref, **get_tol(dtype))
    else:
        with flag_dnn.use_dnn():
            out_custom = torch.mm(a_custom, b_custom)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))
    torch.testing.assert_close(a_custom, a_ref, **get_tol(dtype))
    torch.testing.assert_close(b_custom, b_ref, **get_tol(dtype))


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_dtype_fp32(dtype):
    a = make_tensor((8, 9), dtype)
    b = make_tensor((9, 7), dtype)
    out = torch.empty((8, 7), dtype=torch.float32, device=flag_dnn.device)

    ref = torch.mm(a, b, out_dtype=torch.float32)
    got = flag_dnn.ops.mm(a, b, out_dtype=torch.float32, out=out)

    assert got.data_ptr() == out.data_ptr()
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.mm
def test_mm_non_contiguous_inputs_and_out():
    dtype = torch.float32
    a = torch.randn((5, 7), dtype=dtype, device=flag_dnn.device).t()
    b = torch.randn((3, 5), dtype=dtype, device=flag_dnn.device).t()
    out_base = torch.empty((3, 7), dtype=dtype, device=flag_dnn.device)
    out = out_base.t()

    ref = torch.mm(a, b)
    with flag_dnn.use_dnn():
        got = torch.mm(a, b, out=out)

    assert got.data_ptr() == out.data_ptr()
    assert not out.is_contiguous()
    torch.testing.assert_close(got, ref, **get_tol(dtype))


@pytest.mark.mm
def test_mm_nan_inf_equal_nan():
    a = torch.tensor(
        [[float("nan"), 1.0], [float("inf"), -0.0]],
        dtype=torch.float32,
        device=flag_dnn.device,
    )
    b = torch.tensor(
        [[2.0, -1.0], [3.0, float("inf")]],
        dtype=torch.float32,
        device=flag_dnn.device,
    )

    ref = torch.mm(a, b)
    with flag_dnn.use_dnn():
        got = torch.mm(a, b)

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.mm
def test_mm_invalid_inputs():
    a = torch.randn((2, 3), device=flag_dnn.device)
    b = torch.randn((2, 4), device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(RuntimeError):
            torch.mm(a, b)

    a3 = torch.randn((1, 2, 3), device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(RuntimeError):
            torch.mm(a3, torch.randn((3, 4), device=flag_dnn.device))

    int_a = torch.randint(0, 4, (2, 3), device=flag_dnn.device)
    int_b = torch.randint(0, 4, (3, 4), device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(NotImplementedError):
            torch.mm(int_a, int_b)


@pytest.mark.mm
def test_mm_mismatched_dtype():
    a = torch.randn((2, 3), dtype=torch.float16, device=flag_dnn.device)
    b = torch.randn((3, 4), dtype=torch.float32, device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(RuntimeError):
            torch.mm(a, b)
