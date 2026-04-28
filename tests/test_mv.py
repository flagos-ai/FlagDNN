import pytest
import torch
import flag_dnn


# (matrix_shape, use_out) 的组合测试用例
MV_CASES = [
    ((1, 1), False),
    ((4, 4), False),
    ((16, 32), False),
    ((32, 16), False),
    ((128, 64), False),
    ((128, 63), False),
    ((64, 128), False),
    ((63, 128), False),
    ((63, 127), False),
    ((2, 3), False),
    ((3, 2), False),
    ((0, 4), False),
    ((4, 0), False),
    ((1, 128), False),
    ((128, 1), False),
    ((1, 1), True),
    ((4, 4), True),
    ((16, 32), True),
    ((32, 16), True),
    ((128, 64), True),
    ((64, 128), True),
    ((2, 3), True),
    ((3, 2), True),
    ((0, 4), True),
    ((4, 0), True),
    ((1, 128), True),
    ((128, 1), True),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=1e-4)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.mv
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("matrix_shape, use_out", MV_CASES)
def test_accuracy_mv(dtype, matrix_shape, use_out):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    m, n = matrix_shape

    mat = torch.randn(matrix_shape, dtype=dtype, device=flag_dnn.device) * 5.0
    vec = torch.randn((n,), dtype=dtype, device=flag_dnn.device) * 5.0

    mat_ref = mat.clone()
    vec_ref = vec.clone()
    mat_custom = mat.clone()
    vec_custom = vec.clone()

    if use_out:
        out_ref_buf = torch.empty((m,), dtype=dtype, device=flag_dnn.device)
        out_custom_buf = torch.empty((m,), dtype=dtype, device=flag_dnn.device)

        out_ref = torch.mv(mat_ref, vec_ref, out=out_ref_buf)

        with flag_dnn.use_dnn():
            out_custom = torch.mv(mat_custom, vec_custom, out=out_custom_buf)

        torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

        assert out_custom.data_ptr() == out_custom_buf.data_ptr(), (
            "out is provided, but returned tensor does not share "
            "the output buffer memory."
        )

        torch.testing.assert_close(
            out_custom_buf, out_ref_buf, **get_tol(dtype)
        )
    else:
        out_ref = torch.mv(mat_ref, vec_ref)

        with flag_dnn.use_dnn():
            out_custom = torch.mv(mat_custom, vec_custom)

        torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

        if out_custom.numel() > 0:
            assert (
                out_custom.data_ptr() != mat_custom.data_ptr()
            ), "Output unexpectedly shares memory with input matrix."
            assert (
                out_custom.data_ptr() != vec_custom.data_ptr()
            ), "Output unexpectedly shares memory with input vector."

    torch.testing.assert_close(mat_custom, mat, **get_tol(dtype))
    torch.testing.assert_close(vec_custom, vec, **get_tol(dtype))
