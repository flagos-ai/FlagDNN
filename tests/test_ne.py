import pytest
import torch

import flag_dnn

# (input_shape, other_spec) 的组合测试用例
NE_CASES = [
    # 相同形状
    ((1024,), (1024,)),
    ((2, 3, 4), (2, 3, 4)),
    
    # 标量比较 (包括刚才报错的浮点标量截断用例，这次一定能过！)
    ((128, 256), 0.5),
    ((10, 10), 0),
    
    # Broadcasting 广播机制
    ((10, 1), (1, 20)),
    ((2, 3, 4), (4,)),
    ((1, 3, 1, 5), (2, 1, 4, 1)),
]

@pytest.mark.ne
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
@pytest.mark.parametrize("input_shape, other_spec", NE_CASES)
def test_accuracy_ne(dtype, input_shape, other_spec):
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
            # 为了制造相等的条件，随机将一部分 y 赋值为 x 的对应切片 (从而让 != 产生 False)
            if input_shape == other_spec:
                mask = torch.rand(input_shape, device=flag_dnn.device) > 0.5
                y = torch.where(mask, x, y)
    else:
        y = other_spec

    ref_out = torch.ne(x, y)
    out = flag_dnn.ops.ne(x, y)

    # ne 操作返回必须是 bool 类型
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, ref_out)


@pytest.mark.ne
def test_accuracy_ne_with_out_param():
    """测试带有 out 参数的原地写入"""
    x = torch.tensor([1.0, 2.0, 3.0], device=flag_dnn.device)
    y = torch.tensor([1.0, 0.0, 3.0], device=flag_dnn.device)
    
    # 预分配
    ref_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)
    custom_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)
    
    # 填充脏数据
    custom_out.fill_(False) 
    
    torch.ne(x, y, out=ref_out)
    flag_dnn.ops.ne(x, y, out=custom_out)
    
    torch.testing.assert_close(custom_out, ref_out)


@pytest.mark.ne
def test_accuracy_ne_dtype_promotion():
    """测试数据类型提升 (Type Promotion)"""
    x = torch.tensor([1, 2, 3], dtype=torch.int32, device=flag_dnn.device)
    y = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32, device=flag_dnn.device)

    ref_out = torch.ne(x, y)
    out = flag_dnn.ops.ne(x, y)

    torch.testing.assert_close(out, ref_out)


@pytest.mark.ne
def test_accuracy_ne_empty_tensor():
    """边界测试：空张量的广播与比较"""
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    y = torch.empty((1, 0, 1), dtype=torch.float32, device=flag_dnn.device)
    
    ref_out = torch.ne(x, y)
    out = flag_dnn.ops.ne(x, y)
    
    assert out.shape == ref_out.shape
    assert out.shape == (2, 0, 3)
    torch.testing.assert_close(out, ref_out)