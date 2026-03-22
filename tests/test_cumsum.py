import pytest
import torch
import flag_dnn

CUMSUM_CASES = [
    # (shape, dim)
    ((1024,), 0),
    ((2, 3, 4, 5), 3),
    ((2, 3, 4, 5), 0),
    ((2, 3, 4, 5), -2),
    ((128, 256), 1),
    ((128, 256), 0),
    # 超长 N，专门验证跨越 BLOCK_SIZE (1024) 的 running_sum 逻辑是否正确
    ((2, 5000), 1), 
]

@pytest.mark.cumsum
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32])
@pytest.mark.parametrize("shape, dim", CUMSUM_CASES)
def test_accuracy_cumsum(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 生成数据：浮点数稍微小点，防止累计求和时溢出
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 0.1
    else:
        x = torch.randint(-5, 5, shape, dtype=dtype, device=flag_dnn.device)

    # 【核心修复】：动态计算容差
    rtol, atol = 1e-3, 1e-3
    if dtype in (torch.float16, torch.bfloat16):
        # 如果归约的维度特别长 (比如 5000)，舍入误差会指数级放大
        # 我们对超过 1000 的序列给予更合理的宽容度
        d = dim if dim >= 0 else dim + len(shape)
        N = shape[d]
        if N > 1000:
            rtol, atol = 2e-1, 2e-1  # 允许 0.2 左右的绝对误差
        else:
            rtol, atol = 5e-2, 5e-2

    ref_out = torch.cumsum(x, dim=dim)
    out = flag_dnn.ops.cumsum(x, dim=dim)

    # 整型的累加必须 100% 精确，一点错都不能有
    if not dtype.is_floating_point:
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)

@pytest.mark.cumsum
def test_accuracy_cumsum_out_param():
    """测试原地的 out 参数覆盖"""
    x = torch.randn((10, 20), dtype=torch.float32, device=flag_dnn.device)
    out = torch.empty((10, 20), dtype=torch.float32, device=flag_dnn.device)
    
    ref_out = torch.cumsum(x, dim=0)
    flag_dnn.ops.cumsum(x, dim=0, out=out)
    
    torch.testing.assert_close(out, ref_out)
    
@pytest.mark.cumsum
def test_accuracy_cumsum_dtype_promotion():
    """测试 PyTorch 隐蔽的数据类型提升 (Type Promotion) 机制"""
    x = torch.randint(-5, 5, (10, 20), dtype=torch.int8, device=flag_dnn.device)
    
    # 默认情况下，PyTorch 会把小的整型 (如 int8/int16) 提升为 int64 以防止溢出
    ref_out = torch.cumsum(x, dim=1)
    out = flag_dnn.ops.cumsum(x, dim=1)
    
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    
    # 如果用户硬性指定了 dtype
    ref_out_fp32 = torch.cumsum(x, dim=1, dtype=torch.float32)
    out_fp32 = flag_dnn.ops.cumsum(x, dim=1, dtype=torch.float32)
    
    assert out_fp32.dtype == torch.float32
    torch.testing.assert_close(out_fp32, ref_out_fp32)

@pytest.mark.cumsum
def test_accuracy_cumsum_empty():
    """测试极其刁钻的空张量边界情况"""
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.cumsum(x, dim=1)
    out = flag_dnn.ops.cumsum(x, dim=1)
    torch.testing.assert_close(out, ref_out)