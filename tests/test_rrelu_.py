import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, lower, upper, training)
RRELU_CASES = [
    ((1,), 1.0 / 8, 1.0 / 3, False),
    ((16,), 1.0 / 8, 1.0 / 3, False),
    ((1024,), 0.1, 0.3, False),
    ((2, 3), 1.0 / 8, 1.0 / 3, False),
    ((4, 8, 16), 0.05, 0.4, False),
    ((2, 3, 32, 32), 1.0 / 8, 1.0 / 3, False),
    ((1, 128, 64, 64), 0.2, 0.4, False),
    ((0,), 1.0 / 8, 1.0 / 3, False),
    ((0, 3), 1.0 / 8, 1.0 / 3, False),
]

RRELU_TRAINING_CASES = [
    ((16,), 1.0 / 8, 1.0 / 3, True),
    ((1024,), 0.1, 0.3, True),
    ((2, 3), 1.0 / 8, 1.0 / 3, True),
    ((4, 8, 16), 0.05, 0.4, True),
    ((2, 3, 32, 32), 1.0 / 8, 1.0 / 3, True),
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


@pytest.mark.rrelu_
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, lower, upper, training", RRELU_CASES)
def test_accuracy_rrelu__inference(dtype, shape, lower, upper, training):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.rrelu_(x_ref, lower=lower, upper=upper, training=training)

    with flag_dnn.use_dnn():
        out_custom = F.rrelu_(
            x_custom,
            lower=lower,
            upper=upper,
            training=training,
        )

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    torch.testing.assert_close(x_custom, x_ref, **get_tol(dtype))


@pytest.mark.rrelu_
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, lower, upper, training", RRELU_TRAINING_CASES)
def test_accuracy_rrelu__training_properties(
    dtype, shape, lower, upper, training
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 让输入里同时有正值和负值，避免全正 / 全负过于单一
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_before = x.clone()
    x_custom = x.clone()

    with flag_dnn.use_dnn():
        out_custom = F.rrelu_(
            x_custom,
            lower=lower,
            upper=upper,
            training=training,
        )

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )

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


@pytest.mark.rrelu_
def test_rrelu__invalid_bounds():
    x = torch.randn((8, 8), dtype=torch.float32, device=flag_dnn.device)

    with pytest.raises(Exception):
        F.rrelu_(x, lower=0.5, upper=0.1, training=False)

    with flag_dnn.use_dnn():
        with pytest.raises(Exception):
            F.rrelu_(x, lower=0.5, upper=0.1, training=False)
