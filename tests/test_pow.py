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


def _get_positive_tensor(shape, dtype, device):
    """
    生成严格为正数的张量。
    对于 pow 运算，如果底数为负数且指数为小数，会产生 NaN 或复数。
    我们限制底数为正，确保能够进行稳定的精度比对。
    """
    return torch.abs(torch.randn(shape, dtype=dtype, device=device)) + 0.5


@pytest.mark.pow
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_pow_tensor(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 底数必须为正数
    x = _get_positive_tensor(shape, dtype, flag_dnn.device)
    # 指数可以用普通随机数 (负指数即为取倒数)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3  # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5  # FP32 和 FP64 保持严格

    ref_out = torch.pow(x, y)
    with flag_dnn.use_dnn():
        out = torch.pow(x, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.pow
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_pow_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.pow(x, y)
    with flag_dnn.use_dnn():
        out = torch.pow(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.pow
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_pow_scalar_exponent(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # Tensor base, scalar exponent
    x = _get_positive_tensor((100,), dtype, flag_dnn.device)
    scalar_exp = 2.5

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.pow(x, scalar_exp)
    with flag_dnn.use_dnn():
        out = torch.pow(x, scalar_exp)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.pow
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_pow_scalar_base(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # Scalar base, tensor exponent
    scalar_base = 3.14
    y = torch.randn(100, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.pow(scalar_base, y)
    with flag_dnn.use_dnn():
        out = torch.pow(scalar_base, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.pow
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_pow_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = _get_positive_tensor(input_shape, dtype, flag_dnn.device)
    y = torch.randn(other_shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.pow(x, y)
    with flag_dnn.use_dnn():
        out = torch.pow(x, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.pow
def test_accuracy_pow_integer_and_bool_dtype():
    x_int = torch.tensor([2, 3, 4], dtype=torch.int32, device=flag_dnn.device)
    x_bool = torch.tensor(
        [True, False, True], dtype=torch.bool, device=flag_dnn.device
    )

    ref_int = torch.pow(x_int, 2.0)
    ref_bool = torch.pow(x_bool, 2)
    with flag_dnn.use_dnn():
        out_int = torch.pow(x_int, 2.0)
        out_bool = torch.pow(x_bool, 2)

    assert out_int.dtype == torch.float32
    assert out_bool.dtype == torch.int64
    torch.testing.assert_close(out_int, ref_int, rtol=0, atol=0)
    torch.testing.assert_close(out_bool, ref_bool, rtol=0, atol=0)
