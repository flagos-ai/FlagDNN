import pytest
import torch
import flag_dnn


# (input_shape, other_spec) 的组合测试用例
# other_spec 可以是 tuple (代表形状) 或者数值 (代表标量)
EQ_CASES = [
    # 相同形状
    ((1024,), (1024,)),
    ((2, 3, 4), (2, 3, 4)),
    # 标量比较
    ((128, 256), 0.5),
    ((10, 10), 0),
    # Broadcasting 广播机制
    ((10, 1), (1, 20)),  # 互相扩展
    ((2, 3, 4), (4,)),  # 向前补齐
    ((1, 3, 1, 5), (2, 1, 4, 1)),  # 复杂高维广播
]


@pytest.mark.eq
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32]
)
@pytest.mark.parametrize("input_shape, other_spec", EQ_CASES)
def test_accuracy_eq(dtype, input_shape, other_spec):
    # 初始化 input
    if dtype == torch.int32:
        x = torch.randint(-5, 5, input_shape, dtype=dtype, device=flag_dnn.device)
    else:
        x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)

    # 初始化 other
    if isinstance(other_spec, tuple):
        if dtype == torch.int32:
            y = torch.randint(-5, 5, other_spec, dtype=dtype, device=flag_dnn.device)
        else:
            y = torch.randn(other_spec, dtype=dtype, device=flag_dnn.device)
            # 为了制造相等的条件，随机将一部分 y 赋值为 x 的对应切片 (如果形状允许)
            if input_shape == other_spec:
                mask = torch.rand(input_shape, device=flag_dnn.device) > 0.5
                y = torch.where(mask, x, y)
    else:
        y = torch.tensor(other_spec, dtype=dtype).item()
        # y = other_spec

    ref_out = torch.eq(x, y)
    with flag_dnn.use_dnn():
        out = torch.eq(x, y)

    # eq 操作返回必须是 bool 类型
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, ref_out)


@pytest.mark.eq
def test_accuracy_eq_with_out_param():
    """测试带有 out 参数的原地写入"""
    x = torch.tensor([1.0, 2.0, 3.0], device=flag_dnn.device)
    y = torch.tensor([1.0, 0.0, 3.0], device=flag_dnn.device)

    # 预分配
    ref_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)
    custom_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)

    # 填充脏数据，保证写入真正生效
    custom_out.fill_(True)

    torch.eq(x, y, out=ref_out)
    with flag_dnn.use_dnn():
        torch.eq(x, y, out=custom_out)

    torch.testing.assert_close(custom_out, ref_out)


@pytest.mark.eq
def test_accuracy_eq_dtype_promotion():
    """测试数据类型提升 (Type Promotion)"""
    x = torch.tensor([1, 2, 3], dtype=torch.int32, device=flag_dnn.device)
    y = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32, device=flag_dnn.device)

    ref_out = torch.eq(x, y)
    with flag_dnn.use_dnn():
        out = torch.eq(x, y)

    torch.testing.assert_close(out, ref_out)


@pytest.mark.eq
def test_accuracy_eq_empty_tensor():
    """边界测试：空张量的广播与比较"""
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    y = torch.empty(
        (1, 0, 1), dtype=torch.float32, device=flag_dnn.device
    )  # 支持广播的空张量

    ref_out = torch.eq(x, y)
    with flag_dnn.use_dnn():
        out = torch.eq(x, y)

    assert out.shape == ref_out.shape
    assert out.shape == (2, 0, 3)
    torch.testing.assert_close(out, ref_out)
