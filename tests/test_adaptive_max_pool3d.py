import pytest
import torch
import torch._C._nn as F
import flag_dnn


# adaptive_max_pool3d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 8, 16, 16), (4, 8, 8)),  # 标准 3D 降采样
    ((1, 8, 5, 14, 14), 7),  # 输出尺寸为单 int
    ((2, 4, 7, 15, 15), (3, 5, 7)),  # 三个维度不同尺寸
    ((1, 2, 4, 8, 8), (5, 10, 10)),  # 上采样 (输出尺寸大于输入)
    ((4, 5, 10, 20, 20), 1),  # 3D 全局池化 (Global Max Pooling)
    ((3, 8, 14, 14), (4, 7, 7)),  # 4D 张量输入 (无 Batch 维度 N)
    ((1, 2, 8, 8, 8), (8, 8, 8)),  # output == input (原样输出)
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_max_pool3d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_max_pool3d(x, output_size)

    with flag_dnn.use_dnn():
        out = F.adaptive_max_pool3d(x, output_size)

    # 容差设置：Max Pool 仅拷贝数据无数学运算，因此所有 dtype 皆可要求严苛的精确匹配
    rtol, atol = 1e-5, 1e-5

    out_vals, out_indices = out
    ref_vals, ref_indices = ref_out

    # 验证数值正确性
    torch.testing.assert_close(out_vals, ref_vals, rtol=rtol, atol=atol)
    # 验证索引正确性 (必须完全一致)
    torch.testing.assert_close(out_indices, ref_indices, rtol=0, atol=0)


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_adaptive_max_pool3d_empty_tensor(dtype):
    # D, H, W 至少一个维度的尺寸导致输出 M=0 的情况
    shape = (0, 3, 4, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_max_pool3d(x, 2)
    with flag_dnn.use_dnn():
        out = F.adaptive_max_pool3d(x, 2)

    out_vals, out_indices = out
    ref_vals, ref_indices = ref_out
    assert out_vals.shape == ref_vals.shape
    assert out_indices.shape == ref_indices.shape
    assert out_vals.numel() == 0
    assert out_indices.numel() == 0
