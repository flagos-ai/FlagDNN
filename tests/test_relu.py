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

INTEGER_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


@pytest.mark.relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # Inplace 测试必须隔离显存
    ref_x = x.clone()
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    # ReLU 无精度损失，直接卡死容差
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    if inplace:
        assert (
            y.data_ptr() == test_x.data_ptr()
        ), "Inplace operation failed to reuse memory pointer."


@pytest.mark.relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_empty_tensor(dtype, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试多维度的空张量
    x = torch.empty((2, 0, 3), dtype=dtype, device=flag_dnn.device)

    ref_x = x.clone()
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    assert y.shape == (2, 0, 3)
    assert y.dtype == dtype
    assert y.device == test_x.device
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@pytest.mark.relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_negative_values(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 纯负数测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) - 2.0

    ref_x = x.clone()
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()


@pytest.mark.relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_positive_values(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 纯正数测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) + 2.0

    ref_x = x.clone()
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()


@pytest.mark.relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_mixed_values(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合正负数测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = x.clone()
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()


@pytest.mark.relu
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_integer_dtype(dtype, shape, inplace):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    ref_x = x.clone()
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    assert y.dtype == dtype
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()
