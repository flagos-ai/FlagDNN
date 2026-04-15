import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, lower, upper, training, inplace)
RRELU_CASES = [
    ((1,), 1.0 / 8, 1.0 / 3, False, False),
    ((1,), 1.0 / 8, 1.0 / 3, False, True),
    ((16,), 1.0 / 8, 1.0 / 3, False, False),
    ((16,), 1.0 / 8, 1.0 / 3, False, True),
    ((1024,), 0.1, 0.3, False, False),
    ((1024,), 0.1, 0.3, False, True),
    ((2, 3), 1.0 / 8, 1.0 / 3, False, False),
    ((2, 3), 1.0 / 8, 1.0 / 3, False, True),
    ((4, 8, 16), 0.05, 0.4, False, False),
    ((4, 8, 16), 0.05, 0.4, False, True),
    ((2, 3, 32, 32), 1.0 / 8, 1.0 / 3, False, False),
    ((2, 3, 32, 32), 1.0 / 8, 1.0 / 3, False, True),
    ((1, 128, 64, 64), 0.2, 0.4, False, False),
    ((1, 128, 64, 64), 0.2, 0.4, False, True),
    ((0,), 1.0 / 8, 1.0 / 3, False, False),
    ((0,), 1.0 / 8, 1.0 / 3, False, True),
    ((0, 3), 1.0 / 8, 1.0 / 3, False, False),
    ((0, 3), 1.0 / 8, 1.0 / 3, False, True),
]

RRELU_TRAINING_CASES = [
    ((16,), 1.0 / 8, 1.0 / 3, True, False),
    ((16,), 1.0 / 8, 1.0 / 3, True, True),
    ((1024,), 0.1, 0.3, True, False),
    ((1024,), 0.1, 0.3, True, True),
    ((2, 3), 1.0 / 8, 1.0 / 3, True, False),
    ((2, 3), 1.0 / 8, 1.0 / 3, True, True),
    ((4, 8, 16), 0.05, 0.4, True, False),
    ((4, 8, 16), 0.05, 0.4, True, True),
    ((2, 3, 32, 32), 1.0 / 8, 1.0 / 3, True, False),
    ((2, 3, 32, 32), 1.0 / 8, 1.0 / 3, True, True),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    if dtype == torch.float64:
        return dict(rtol=1e-12, atol=1e-12)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.rrelu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, lower, upper, training, inplace", RRELU_CASES)
def test_accuracy_rrelu_inference(
    dtype, shape, lower, upper, training, inplace
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.rrelu(
        x_ref, lower=lower, upper=upper, training=training, inplace=inplace
    )

    with flag_dnn.use_dnn():
        out_custom = F.rrelu(
            x_custom,
            lower=lower,
            upper=upper,
            training=training,
            inplace=inplace,
        )

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    if inplace:
        assert out_custom.data_ptr() == x_custom.data_ptr(), (
            "Inplace flag is True, but output is not modifying "
            "the input tensor directly."
        )
        torch.testing.assert_close(x_custom, x_ref, **get_tol(dtype))
    else:
        if x.numel() > 0:
            assert out_custom.data_ptr() != x_custom.data_ptr(), (
                "Inplace flag is False, but output is modifying "
                "the input tensor memory."
            )
        torch.testing.assert_close(x_custom, x, **get_tol(dtype))


@pytest.mark.rrelu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "shape, lower, upper, training, inplace", RRELU_TRAINING_CASES
)
def test_accuracy_rrelu_training_properties(
    dtype, shape, lower, upper, training, inplace
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 让输入里同时有正值和负值，避免全正 / 全负过于单一
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_before = x.clone()
    x_custom = x.clone()

    with flag_dnn.use_dnn():
        out_custom = F.rrelu(
            x_custom,
            lower=lower,
            upper=upper,
            training=training,
            inplace=inplace,
        )

    if inplace:
        assert out_custom.data_ptr() == x_custom.data_ptr(), (
            "Inplace flag is True, but output is not modifying "
            "the input tensor directly."
        )
    else:
        if x.numel() > 0:
            assert out_custom.data_ptr() != x_custom.data_ptr(), (
                "Inplace flag is False, but output is modifying "
                "the input tensor memory."
            )
        torch.testing.assert_close(x_custom, x_before, **get_tol(dtype))

    # 正值位置必须保持不变
    pos_mask = x_before > 0
    if pos_mask.any():
        torch.testing.assert_close(
            out_custom[pos_mask], x_before[pos_mask], **get_tol(dtype)
        )

    # 负值位置：输出 / 输入 的 slope 应落在 [lower, upper]
    neg_mask = x_before < 0
    if neg_mask.any():
        slopes = out_custom[neg_mask] / x_before[neg_mask]
        tol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-6
        assert torch.all(
            slopes >= (lower - tol)
        ), f"Found slope smaller than lower bound {lower}"
        assert torch.all(
            slopes <= (upper + tol)
        ), f"Found slope larger than upper bound {upper}"


@pytest.mark.rrelu
def test_rrelu_invalid_bounds():
    x = torch.randn((8, 8), dtype=torch.float32, device=flag_dnn.device)

    with pytest.raises(Exception):
        F.rrelu(x, lower=0.5, upper=0.1, training=False, inplace=False)

    with flag_dnn.use_dnn():
        with pytest.raises(Exception):
            F.rrelu(x, lower=0.5, upper=0.1, training=False, inplace=False)
