import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, kernel_size, stride, padding)
PARAMS = [
    ((2, 3, 32, 32), 2, 2, 0),  # 标准下采样
    ((1, 16, 28, 28), 3, 1, 1),  # 保持原图尺寸 (带 padding)
    ((4, 8, 15, 15), 3, 2, 1),  # 奇数尺寸步长跨越
    ((2, 4, 32, 32), (3, 5), (2, 1), 0),  # 不对称核和步长
    ((16, 14, 14), 2, 2, 0),  # 3D 张量输入
]


@pytest.mark.avg_pool2d
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, kernel_size, stride, padding", PARAMS)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
@pytest.mark.parametrize("divisor_override", [None, 4])
def test_accuracy_avg_pool2d(
    dtype,
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # AvgPool 对极端大值比较敏感，所以使用正常的 randn 即可
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 容差设置 (因为包含累加运算，不能像 max_pool 那样设置 1e-6)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_y = F.avg_pool2d(
        x,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    with flag_dnn.use_dnn():
        y = F.avg_pool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_avg_pool2d_empty_tensor(dtype):
    # PyTorch 规定空张量的 Batch 维可以为 0，但空间维度 (H, W) 不能为 0
    shape = (0, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_y = F.avg_pool2d(x, 2, 2)
    with flag_dnn.use_dnn():
        y = F.avg_pool2d(x, 2, 2)

    assert y.shape == ref_y.shape
    assert y.numel() == 0


@pytest.mark.avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_avg_pool2d_large_values(dtype):
    # 测试大数值累加是否会因为溢出导致结果不一致
    shape = (2, 3, 32, 32)
    # FP16 的最大值是 65504，我们用 1e3 级别的数据，保证单个数值不溢出，
    # 但 3x3 (9个) 或 5x5 (25个) 的窗口累加时极易逼近或超过极限，考验底层的 FP32 累加机制
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 1000.0

    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-2
    else:
        rtol, atol = 1e-5, 1e-4

    ref_y = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    with flag_dnn.use_dnn():
        y = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_avg_pool2d_mixed_values(dtype):
    # 测试大数吃小数（极端比例混合）时的精度表现
    shape = (2, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 制造巨大的数值落差
    x[..., ::2, ::2] *= 1000.0  # 偶数位放大
    x[..., 1::2, 1::2] *= 0.001  # 奇数位缩小

    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-2
    else:
        rtol, atol = 1e-5, 1e-4

    ref_y = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
    with flag_dnn.use_dnn():
        y = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
