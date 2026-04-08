import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# 3D 参数格式：(shape, kernel_size, stride, padding, dilation)
PARAMS = [
    ((2, 3, 8, 32, 32), 2, 2, 0, 1),              # 标准 2x2x2 降采样
    ((1, 8, 4, 16, 16), 3, 1, 1, 1),              # 保持原图尺寸 (Padding=1)
    ((2, 4, 5, 15, 15), 3, 2, 1, 1),              # 奇数尺寸的步长跨越
    ((1, 2, 8, 16, 16), (2, 3, 5), (1, 2, 1), 0, 1),  # 不对称的 Kernel 和 Stride
    ((2, 3, 6, 16, 16), 3, 2, 0, 2),              # 带空洞率 (Dilation)
    ((4, 5, 14, 14), 2, 2, 0, 1),                 # 4D 张量输入 (无 Batch 维度 N)
]


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, kernel_size, stride, padding, dilation", PARAMS)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("return_indices", [False, True])
def test_accuracy_max_pool3d(dtype, shape, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 减少相同最大值的出现概率，保证 indices 的绝对唯一性
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.max_pool3d(
        x, kernel_size, stride=stride, padding=padding, dilation=dilation,
        ceil_mode=ceil_mode, return_indices=return_indices
    )
    
    with flag_dnn.use_dnn():
        out = F.max_pool3d(
            x, kernel_size, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, return_indices=return_indices
        )

    rtol, atol = 1e-6, 1e-6

    if return_indices:
        y, idx = out
        ref_y, ref_idx = ref_out
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
        
        torch.testing.assert_close(idx, ref_idx, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_max_pool3d_empty_tensor(dtype):
    # D, H, W 至少一个维度的尺寸导致输出 M=0 的情况
    shape = (0, 3, 4, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.max_pool3d(x, 2, 2)
    with flag_dnn.use_dnn():
        out = F.max_pool3d(x, 2, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0