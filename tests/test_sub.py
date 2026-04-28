import pytest
import torch
import flag_dnn


SHAPES = [
    (),
    (1,),
    (17,),
    (32,),
    (127,),
    (1024,),
    (5333,),
    (17, 31),
    (4, 8, 16),
    (2, 3, 4, 5),
    (1, 64, 7, 7),
    (1024 * 1024,),
]

BROADCAST_SHAPES = [
    ((4, 4), (4,)),  # 1D broadcast to 2D
    ((2, 3, 4), (3, 1)),  # 内部维度广播
    ((1, 5), (5, 5)),  # 单一维度扩展
    ((2, 1, 4, 1), (1, 3, 1, 5)),  # 复杂高维双向广播
    ((), (17, 31)),  # 标量 Tensor 广播到矩阵
]

INTEGER_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


@pytest.mark.sub
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sub(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3  # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5  # FP32 和 FP64 保持严格

    ref_out = torch.sub(x, y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sub_integer_dtype(dtype, shape):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    ref_out = torch.sub(x, y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    assert out.dtype == dtype
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.sub
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_sub_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_out = torch.sub(x, y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.sub
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_sub_scalar(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    scalar = 3.14

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3  # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5  # FP32 和 FP64 保持严格

    ref_out = torch.sub(x, scalar)
    with flag_dnn.use_dnn():
        out = torch.sub(x, scalar)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.sub
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_sub_alpha(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    alpha = 2.5

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3  # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5  # FP32 和 FP64 保持严格

    ref_out = torch.sub(x, y, alpha=alpha)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y, alpha=alpha)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.sub
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_sub_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(other_shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3  # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5  # FP32 和 FP64 保持严格

    ref_out = torch.sub(x, y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)
