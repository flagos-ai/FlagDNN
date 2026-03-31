import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from .accuracy_utils import gems_assert_close

# (shape, output_size)
PARAMS = [
    ((2, 3, 32, 32), (1, 1)),                 # 全局最大池化
    ((1, 16, 28, 28), 14),                    # 降维到 14x14
    ((4, 8, 15, 15), (7, 5)),                 # 非对称目标尺寸
    ((2, 4, 32, 32), (None, 16)),             # 保持 H 尺寸不变
    ((16, 14, 14), (2, 2)),                   # 3D 张量输入
]

@pytest.mark.adaptive_max_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, output_size", PARAMS)
@pytest.mark.parametrize("return_indices", [False, True])
def test_accuracy_adaptive_max_pool2d(dtype, shape, output_size, return_indices):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # MaxPool 本身不做乘加，直接比较，所以可以要求绝对的零误差
    ref_out = F.adaptive_max_pool2d(x, output_size, return_indices=return_indices)
    out = flag_dnn.ops.adaptive_max_pool2d(x, output_size, return_indices=return_indices)

    if return_indices:
        ref_y, ref_idx = ref_out
        y, idx = out
        torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
        torch.testing.assert_close(idx, ref_idx, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.adaptive_max_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("return_indices", [False, True])
def test_accuracy_adaptive_max_pool2d_empty_tensor(dtype, return_indices):
    shape = (0, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_max_pool2d(x, (2, 2), return_indices=return_indices)
    out = flag_dnn.ops.adaptive_max_pool2d(x, (2, 2), return_indices=return_indices)

    if return_indices:
        ref_y, ref_idx = ref_out
        y, idx = out
        assert y.shape == ref_y.shape
        assert y.numel() == 0
        assert idx.shape == ref_idx.shape
        assert idx.numel() == 0
    else:
        assert out.shape == ref_out.shape
        assert out.numel() == 0