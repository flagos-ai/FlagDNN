import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# 定义常用的形状和它们对应的归一化形状 (shape, normalized_shape)
SHAPES_AND_NORM_SHAPES = [
    ((32,), (32,)),                           # 1D 张量，全局归一化
    ((1024,), (1024,)),                       # 1D 大张量
    ((2, 16), (16,)),                         # 2D 张量，归一化最后一维
    ((4, 8, 32), (32,)),                      # 3D 张量，归一化最后一维 (类似 NLP 中的 SeqLen)
    ((4, 8, 32), (8, 32)),                    # 3D 张量，归一化最后两维
    ((2, 4, 16, 16), (16, 16)),               # 4D 张量，归一化空间维度 (类似 CV)
    ((2, 4, 16, 16), (4, 16, 16)),            # 4D 张量，归一化除 Batch 外的所有维度
]


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, normalized_shape", SHAPES_AND_NORM_SHAPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm(dtype, shape, normalized_shape, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    # 构建仿射变换参数
    weight, bias = None, None
    if elementwise_affine:
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
    
    ref_y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm_empty_tensor(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量
    shape = (0, 4, 16)
    normalized_shape = (16,)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2 
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3   
    else:
        rtol, atol = 1e-5, 1e-5   
    
    weight, bias = None, None
    if elementwise_affine:
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
    
    ref_y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)




@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm_large_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 32)
    normalized_shape = (32,)
    
    # 专门测试大数值，验证方差计算时是否发生了严重的精度丢失
    # 由于 16 位浮点数的尾数限制，这里的大数值设计需要针对 dtype 区分
    if dtype in [torch.float16, torch.bfloat16]:
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0 + 100.0
    else:
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 100.0 + 1000.0

    # 针对大数值平移，适当放宽低精度类型的阈值
    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2  # BF16 尾数极短，允许更大的误差
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2  # FP16 允许更大的误差
    else:
        rtol, atol = 5e-4, 5e-4   
    
    weight, bias = None, None
    if elementwise_affine:
        # 权重和偏置也需要适配大数值场景，避免乘加溢出
        if dtype in [torch.float16, torch.bfloat16]:
            weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device) * 2.0
            bias = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device) * 10.0
        else:
            weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device) * 10.0
            bias = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device) * 100.0
    
    ref_y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm_mixed_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合常规数据测试
    shape = (4, 8, 16)
    normalized_shape = (8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2 
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3   
    else:
        rtol, atol = 1e-5, 1e-5   
    
    weight, bias = None, None
    if elementwise_affine:
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
    
    ref_y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)