import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, threshold_val, value_val) 的组合测试用例
THRESHOLD_CASES = [
    # 1. 基础与经典形状
    ((1024,), 0.0, 0.0),  # 经典 ReLU 行为
    ((1024,), 1.5, 99.0),
    # 2. 非对齐的形状
    ((1023,), -1.0, 5.0),
    ((17, 31), 0.5, -2.5),
    ((13, 17, 19, 23), 0.0, 42.0),  # 4D 全质数形状，绝对的非 2 的幂次方
    # 3. 多维与高维张量
    ((4, 8, 16), 2.0, 0.0),
    ((2, 3, 4, 5), -0.5, 10.0),
    (
        (1, 1, 1, 1, 1),
        0.5,
        -0.5,
    ),  # 5D 张量，测试维度降维成 1D 时的正确性
    # 4. 大尺度张量 (Stress Testing)
    ((1024 * 1024,), 1.0, -1.0),  # 百万级一维元素，测试 Grid 拆分机制
    (
        (64, 128, 256),
        0.5,
        0.0,
    ),  # 大体积 3D 张量，测试大吞吐量
    # 5. 极端条件判断 (All or Nothing)
    ((64, 64), 100.0, -1.0),  # 阈值极大 (100.0)：几乎所有元素都会被替换
    (
        (64, 64),
        -100.0,
        -1.0,
    ),  # 阈值极小 (-100.0)：几乎没有任何元素会被替换
    # 6. 特殊替换值
    (
        (256, 256),
        0.0,
        float("-inf"),
    ),  # 替换为负无穷 (常用于 Attention Masking)
    ((256, 256), 0.0, float("inf")),  # 替换为正无穷
    # 7. 极端边界情况
    ((1,), 0.0, 1.0),  # 单元素
    (
        (0,),
        0.0,
        0.0,
    ),  # 空张量 (Empty Tensor)，测试 Kernel 是否正确 skip
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.threshold_
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, threshold_val, value_val", THRESHOLD_CASES)
def test_accuracy_threshold_(dtype, shape, threshold_val, value_val):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.threshold_(x_ref, threshold_val, value_val)

    with flag_dnn.use_dnn():
        out_custom = F.threshold_(x_custom, threshold_val, value_val)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    torch.testing.assert_close(x_custom, x_ref, **get_tol(dtype))
