import pytest
import torch
import flag_dnn


# (shape, dim, keepdim) 的组合测试用例
SUM_CASES = [
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
    
    # 大 N 归约 (测试 Kernel 内部的 for 循环累加是否正确)
    ((2, 10000), 1, False),
]

def _get_tolerances(dtype):
    # 累加操作的误差会随元素增多而放大，容差需要适当放宽
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:
        return 1e-4, 1e-4


@pytest.mark.sum
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, dim, keepdim", SUM_CASES)
def test_accuracy_sum(dtype, shape, dim, keepdim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    
    ref_out = torch.sum(x, dim=dim, keepdim=keepdim)
    with flag_dnn.use_dnn():
        out = torch.sum(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.sum
@pytest.mark.parametrize("input_dtype, out_dtype", [
    (torch.float16, torch.float32), 
    (torch.int32, torch.int64),    # 测试整数的类型推导
])
def test_accuracy_sum_dtype_promotion(input_dtype, out_dtype):
    """测试带有 dtype 参数的类型提升"""
    if input_dtype.is_floating_point:
        x = torch.randn((10, 20), dtype=input_dtype, device=flag_dnn.device)
    else:
        x = torch.randint(-10, 10, (10, 20), dtype=input_dtype, device=flag_dnn.device)

    ref_out = torch.sum(x, dim=1, dtype=out_dtype)
    with flag_dnn.use_dnn():
        out = torch.sum(x, dim=1, dtype=out_dtype)

    assert out.dtype == out_dtype
    # 整数必须精确相等
    if not out_dtype.is_floating_point:
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.sum
def test_accuracy_sum_empty_tensor():
    """边界测试：空张量"""
    x = torch.randn((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    
    ref_out = torch.sum(x, dim=2)
    with flag_dnn.use_dnn():
        out = torch.sum(x, dim=2)
    
    torch.testing.assert_close(out, ref_out)