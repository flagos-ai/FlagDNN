import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# BatchNorm 要求至少 2D，通常是 (N, C), (N, C, L) 或 (N, C, H, W)
SHAPES = [(32, 16), (4, 8, 32), (2, 4, 16, 16), (1, 64, 8, 8)]


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_accuracy_batch_norm(dtype, shape, training, affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # BatchNorm 在 FP16/BF16 下容易产生较大的累积误差，所以放宽精度
    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 2e-2
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5

    C = shape[1]
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 2.0 + 1.0

    running_mean = torch.randn(C, dtype=dtype, device=flag_dnn.device)
    running_var = (
        torch.abs(torch.randn(C, dtype=dtype, device=flag_dnn.device)) + 0.1
    )

    if affine:
        weight = torch.randn(C, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(C, dtype=dtype, device=flag_dnn.device)
    else:
        weight = None
        bias = None

    # 因为 training=True 会 in-place 修改 running_mean 和 running_var
    # 我们需要深拷贝一份给自定义算子使用，以免影响对比基准
    ref_running_mean = running_mean.clone()
    ref_running_var = running_var.clone()
    test_running_mean = running_mean.clone()
    test_running_var = running_var.clone()

    ref_y = F.batch_norm(
        x,
        ref_running_mean,
        ref_running_var,
        weight=weight,
        bias=bias,
        training=training,
    )

    with flag_dnn.use_dnn():
        y = F.batch_norm(
            x,
            test_running_mean,
            test_running_var,
            weight=weight,
            bias=bias,
            training=training,
        )

    # 验证前向传播精度
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)

    # 如果是训练模式，还需要验证 running_stats 是否同步更新一致
    if training:
        torch.testing.assert_close(
            test_running_mean, ref_running_mean, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            test_running_var, ref_running_var, rtol=rtol, atol=atol
        )


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_batch_norm_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量
    shape = (0, 4, 16)
    C = shape[1]
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    running_mean = torch.zeros(C, dtype=dtype, device=flag_dnn.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_dnn.device)

    F.batch_norm(x, running_mean, running_var, training=False)
    with flag_dnn.use_dnn():
        y = F.batch_norm(x, running_mean, running_var, training=False)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device

    # 因为没有元素，不需要对比 close，能跑过并保持属性一致即可


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("training", [False, True])
def test_accuracy_batch_norm_large_values(dtype, training):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试极大的数值，验证线性平移 (x - mean) 是否能正常工作且不损失过多精度
    shape = (4, 16, 32)
    C = shape[1]

    if dtype in [torch.float16, torch.bfloat16]:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0
            + 100.0
        )
        running_mean = (
            torch.randn(C, dtype=dtype, device=flag_dnn.device) * 10.0
        )
        running_var = (
            torch.abs(torch.randn(C, dtype=dtype, device=flag_dnn.device))
            * 5.0
            + 0.1
        )
    else:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 1000.0
            + 10000.0
        )
        running_mean = (
            torch.randn(C, dtype=dtype, device=flag_dnn.device) * 100.0
        )
        running_var = (
            torch.abs(torch.randn(C, dtype=dtype, device=flag_dnn.device))
            * 10.0
            + 0.1
        )

    # 宽容度设置
    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_running_mean, test_running_mean = (
        running_mean.clone(),
        running_mean.clone(),
    )
    ref_running_var, test_running_var = (
        running_var.clone(),
        running_var.clone(),
    )

    ref_y = F.batch_norm(
        x, ref_running_mean, ref_running_var, training=training
    )
    with flag_dnn.use_dnn():
        y = F.batch_norm(
            x, test_running_mean, test_running_var, training=training
        )

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("training", [False, True])
def test_accuracy_batch_norm_mixed_values(dtype, training):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 纯正负混合的常规正态分布
    shape = (4, 16, 32)
    C = shape[1]
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 2e-2
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5

    running_mean = torch.zeros(C, dtype=dtype, device=flag_dnn.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_dnn.device)

    ref_running_mean, test_running_mean = (
        running_mean.clone(),
        running_mean.clone(),
    )
    ref_running_var, test_running_var = (
        running_var.clone(),
        running_var.clone(),
    )

    ref_y = F.batch_norm(
        x, ref_running_mean, ref_running_var, training=training
    )
    with flag_dnn.use_dnn():
        y = F.batch_norm(
            x, test_running_mean, test_running_var, training=training
        )

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_batch_norm_small_variance(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 专门针对 BatchNorm 命门的测试：所有输入几乎一样，方差趋近于 0
    # 验证 eps (epsilon) 是否成功防止了除以 0 的错误 (NaN/Inf)
    shape = (4, 8, 16)
    C = shape[1]

    # 极小的噪声
    x = torch.zeros(shape, dtype=dtype, device=flag_dnn.device)
    if dtype in [torch.float32, torch.float64]:
        x += torch.randn_like(x) * 1e-3

    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 2e-2
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5

    running_mean = torch.zeros(C, dtype=dtype, device=flag_dnn.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_dnn.device)

    # 必须使用 training=True 才能触发根据输入动态计算方差的逻辑
    ref_y = F.batch_norm(
        x, running_mean.clone(), running_var.clone(), training=True
    )
    with flag_dnn.use_dnn():
        y = F.batch_norm(
            x, running_mean.clone(), running_var.clone(), training=True
        )

    # 只要输出不是 NaN 且能对齐 PyTorch 的结果，说明 eps 处理正确
    assert not torch.isnan(
        y
    ).any(), "Output contains NaN due to division by zero!"
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
