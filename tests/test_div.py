import pytest
import torch

import flag_dnn

from .accuracy_utils import gems_assert_close


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]

BROADCAST_SHAPES = [
    ((4, 4), (4,)),              # 1D broadcast to 2D
    ((2, 3, 4), (3, 1)),         # 内部维度广播
    ((1, 5), (5, 5)),            # 单一维度扩展
]


def _get_safe_divisor(shape, dtype, device):
    """
    生成安全的除数张量，避免随机出极小的值（如 1e-4）导致除法结果过大，
    从而破坏 FP16/BF16 的相对/绝对容差验证。
    """
    y = torch.randn(shape, dtype=dtype, device=device)
    # 将绝对值小于 0.1 的元素强制设为 1.0 (保留原本的符号会更好，这里为简单稳定直接赋固定值或带符号偏移)
    y[y.abs() < 0.1] = 1.0
    return y


@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_div(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = _get_safe_divisor(shape, dtype, flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格

    ref_out = torch.div(x, y)
    out = flag_dnn.ops.div(x, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_accuracy_div_empty_tensor(dtype):
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

    ref_out = torch.div(x, y)
    out = flag_dnn.ops.div(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_accuracy_div_scalar(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    scalar = 3.14  # 使用足够大的固定安全除数

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.div(x, scalar)
    out = flag_dnn.ops.div(x, scalar)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_accuracy_div_rounding_mode(dtype, rounding_mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    y = _get_safe_divisor((100,), dtype, flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.div(x, y, rounding_mode=rounding_mode)
    out = flag_dnn.ops.div(x, y, rounding_mode=rounding_mode)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_div_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    y = _get_safe_divisor(other_shape, dtype, flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_out = torch.div(x, y)
    out = flag_dnn.ops.div(x, y)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)