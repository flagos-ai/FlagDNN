import pytest
import torch
import flag_dnn


# (shape, dim, keepdim) 的组合测试用例
MEAN_CASES = [
    # 全局归约
    ((1024,), None, False),
    ((2, 3, 4, 5), None, True),
    # 单维度归约
    ((128, 256), 0, False),
    ((128, 256), 1, True),
    ((2, 3, 4, 5), 2, False),
    ((2, 3, 4, 5), -1, True),
    # 多维度归约
    ((2, 3, 4, 5), (1, 3), False),
    ((10, 20, 30), (0, 1), True),
    # 大 N 归约
    ((2, 10000), 1, False),
]


def _get_tolerances(dtype):
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:
        return 1e-4, 1e-4


@pytest.mark.mean
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, dim, keepdim", MEAN_CASES)
def test_accuracy_mean(dtype, shape, dim, keepdim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)

    ref_out = torch.mean(x, dim=dim, keepdim=keepdim)
    with flag_dnn.use_dnn():
        out = torch.mean(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.mean
def test_accuracy_mean_with_out_param():
    """测试带有 out 参数的原地写入"""
    x = torch.randn((10, 20), dtype=torch.float32, device=flag_dnn.device)

    # 预先分配好一个符合预期的 out tensor
    ref_out = torch.empty((10,), dtype=torch.float32, device=flag_dnn.device)
    custom_out = torch.empty(
        (10,), dtype=torch.float32, device=flag_dnn.device
    )

    # 填充一些脏数据验证是否会被覆盖
    custom_out.fill_(-999.0)

    torch.mean(x, dim=1, out=ref_out)
    with flag_dnn.use_dnn():
        torch.mean(x, dim=1, out=custom_out)

    torch.testing.assert_close(custom_out, ref_out)


@pytest.mark.mean
@pytest.mark.parametrize(
    "input_dtype, out_dtype",
    [
        (torch.float16, torch.float32),
        (torch.int32, torch.float32),  # 整数求均值，必须指定浮点 dtype
    ],
)
def test_accuracy_mean_dtype_promotion(input_dtype, out_dtype):
    """测试带有 dtype 参数的计算"""
    if input_dtype.is_floating_point:
        x = torch.randn((10, 20), dtype=input_dtype, device=flag_dnn.device)
    else:
        x = torch.randint(
            -10, 10, (10, 20), dtype=input_dtype, device=flag_dnn.device
        )

    ref_out = torch.mean(x, dim=1, dtype=out_dtype)
    with flag_dnn.use_dnn():
        out = torch.mean(x, dim=1, dtype=out_dtype)

    assert out.dtype == out_dtype
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.mean
def test_accuracy_mean_empty_tensor():
    """边界测试：空张量必须产生 NaN，且不能崩溃"""
    # 针对被规约的维度是 0 的情况
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)

    ref_out = torch.mean(x, dim=1)
    with flag_dnn.use_dnn():
        out = torch.mean(x, dim=1)

    # 注意这里必须加 equal_nan=True，否则 NaN != NaN 会导致测试失败
    torch.testing.assert_close(out, ref_out, equal_nan=True)

    # 针对规约别的维度，但某个不相关维度是 0 的情况
    ref_out_2 = torch.mean(x, dim=2)
    with flag_dnn.use_dnn():
        out_2 = torch.mean(x, dim=2)
    torch.testing.assert_close(out_2, ref_out_2, equal_nan=True)
