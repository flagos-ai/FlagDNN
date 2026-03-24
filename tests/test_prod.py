import pytest
import torch

import flag_dnn

# (shape, dim, keepdim) 组合测试用例
# 注意：prod 不支持 tuple 形式的多维度归约
PROD_CASES = [
    # 全局归约
    ((1024,), None, False),
    ((2, 3, 4, 5), None, True),
    
    # 单维度归约
    ((128, 256), 0, False),
    ((128, 256), 1, True),
    ((2, 3, 4, 5), 2, False),
    ((2, 3, 4, 5), -1, True),
    
    # 大 N 归约
    ((2, 5000), 1, False),
]

def _get_tolerances(dtype):
    # 乘积的浮点误差会迅速放大，所以这里的容差比 sum 稍微放宽一些
    if dtype == torch.bfloat16:
        return 1e-1, 1e-1
    elif dtype == torch.float16:
        return 5e-2, 5e-2
    else:
        return 1e-3, 1e-3


@pytest.mark.prod
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, dim, keepdim", PROD_CASES)
def test_accuracy_prod(dtype, shape, dim, keepdim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 乘积操作很容易发生指数级溢出/下溢，所以用较小的方差生成数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 0.5 + 1.0

    rtol, atol = _get_tolerances(dtype)
    
    # 绕过 PyTorch 原生 prod API 对 dim=None 和 keepdim 的限制
    if dim is None:
        ref_out = torch.prod(x)  # PyTorch 全局求积只能这么调用
        if keepdim:
            ref_out = ref_out.view([1] * x.ndim)  # 手动补齐 keepdim 的形状
    else:
        ref_out = torch.prod(x, dim=dim, keepdim=keepdim)

    out = flag_dnn.ops.prod(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.prod
@pytest.mark.parametrize("input_dtype, out_dtype", [
    (torch.float16, torch.float32), 
    (torch.int32, torch.float64), 
])
def test_accuracy_prod_dtype_promotion(input_dtype, out_dtype):
    """测试带有 dtype 参数的计算，防止数据类型溢出"""
    if input_dtype.is_floating_point:
        x = torch.randn((10, 20), dtype=input_dtype, device=flag_dnn.device) * 0.5
    else:
        x = torch.randint(-2, 3, (10, 20), dtype=input_dtype, device=flag_dnn.device)

    ref_out = torch.prod(x, dim=1, dtype=out_dtype)
    out = flag_dnn.ops.prod(x, dim=1, dtype=out_dtype)

    assert out.dtype == out_dtype
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.prod
def test_accuracy_prod_empty_tensor():
    """边界测试：空张量的乘积必须产生 1 (而不是 0)"""
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    
    ref_out = torch.prod(x, dim=1)
    out = flag_dnn.ops.prod(x, dim=1)
    
    torch.testing.assert_close(out, ref_out)