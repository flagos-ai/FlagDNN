import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# 1D 的参数格式：(shape, kernel_size, stride, padding, dilation)
PARAMS = [
    ((2, 3, 32), 2, 2, 0, 1),             # 标准 2 降采样
    ((1, 16, 28), 3, 1, 1, 1),            # 保持原图尺寸
    ((4, 8, 15), 3, 2, 1, 1),             # 奇数尺寸的步长跨越
    ((2, 4, 32), 5, 1, 0, 1),             # 较大的一维卷积核测试
    ((2, 3, 32), 3, 2, 0, 2),             # 带空洞率 (Dilation)
    ((16, 14), 2, 2, 0, 1),               # 2D 张量输入 (无 Batch(N) 维度)
]


@pytest.mark.max_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, kernel_size, stride, padding, dilation", PARAMS)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("return_indices", [False, True])
def test_accuracy_max_pool1d(dtype, shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 减少相同最大值的出现概率，保证 indices 的唯一对比性
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.max_pool1d(
        x, kernel_size, stride=stride, padding=padding, dilation=dilation,
        ceil_mode=ceil_mode, return_indices=return_indices
    )
    
    with flag_dnn.use_dnn():
        out = F.max_pool1d(
            x, kernel_size, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, return_indices=return_indices
        )

    # 容差设置：因为 MaxPool 只做选择不参与算术运算，应该完全一致，设 0 完全可以。
    # 为了防止某些架构底层的极小扰动，给一个 1e-6 的底线
    rtol, atol = 1e-6, 1e-6

    if return_indices:
        y, idx = out
        ref_y, ref_idx = ref_out
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
        
        torch.testing.assert_close(idx, ref_idx, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.max_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_max_pool1d_empty_tensor(dtype):
    shape = (0, 3, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.max_pool1d(x, 2, 2)
    with flag_dnn.use_dnn():
        out = F.max_pool1d(x, 2, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0