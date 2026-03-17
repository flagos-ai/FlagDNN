import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from .accuracy_utils import gems_assert_close

SHAPES = [(32,), (1024,), (2, 16), (4, 8, 32), (2, 4, 16, 16)]
DIMS = [None, -1, 0, 1, 2]

@pytest.mark.softmax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
def test_accuracy_softmax(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 如果指定的 dim 超出了当前 shape 的维度范围，则跳过
    if dim is not None and (dim >= len(shape) or dim < -len(shape)):
        pytest.skip(f"Dimension {dim} is out of bounds for shape {shape}")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    # Softmax 包含 exp 和 sum，在低精度下累积误差较大
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    # 我们的算子对 dim=None 的处理默认是 -1
    # 为了防止 PyTorch F.softmax 报 warning，显式转换为 -1 供 PyTorch 参照
    ref_dim = -1 if dim is None else dim
    
    ref_y = F.softmax(x, dim=ref_dim)
    y = flag_dnn.ops.softmax(x, dim=dim)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 0])
def test_accuracy_softmax_empty_tensor(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量
    shape = (0, 4, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2 
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3   
    else:
        rtol, atol = 1e-5, 1e-5   
    
    ref_y = F.softmax(x, dim=dim)
    y = flag_dnn.ops.softmax(x, dim=dim)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 1])
def test_accuracy_softmax_large_values(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 专门测试大数值 (如 100 左右)，验证算子内部减去最大值的防溢出机制是否生效
    shape = (4, 8, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0 + 100.0

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2 
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3   
    else:
        rtol, atol = 1e-5, 1e-5   
    
    ref_y = F.softmax(x, dim=dim)
    y = flag_dnn.ops.softmax(x, dim=dim)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_accuracy_softmax_mixed_values(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合常规数据测试
    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2 
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3   
    else:
        rtol, atol = 1e-5, 1e-5   
    
    ref_y = F.softmax(x, dim=dim)
    y = flag_dnn.ops.softmax(x, dim=dim)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)