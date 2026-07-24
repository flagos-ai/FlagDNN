# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_dnn
from tests import consts
from tests.norm_test_utils import (
    _assert_outputs_close,
    _compile_layernorm,
    run_batchnorm_test,
    run_layernorm_test,
    run_rmsnorm_test,
)


@pytest.mark.layernorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_layernorm_multi_output(dnn_reference, dtype):
    run_layernorm_test(dnn_reference, dtype)


@pytest.mark.layernorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_layernorm_wide_single_tile(dnn_reference, dtype):
    torch.manual_seed(11)
    epsilon = 1e-3
    x = torch.randn(
        (2, 4, 4096), device=flag_dnn.device, dtype=dtype
    ).contiguous()
    scale = torch.randn(
        (1, 1, 4096), device=x.device, dtype=dtype
    ).contiguous()
    bias = torch.randn((1, 1, 4096), device=x.device, dtype=dtype).contiguous()
    expected = dnn_reference.run(
        "layernorm", "TRAINING", x, scale, bias, epsilon
    )
    actual = _compile_layernorm(x, scale, bias, epsilon).run(x, scale, bias)
    dnn_reference.synchronize()
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    _assert_outputs_close(actual, expected, dtype, atol)


@pytest.mark.rmsnorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_rmsnorm_multi_output(dnn_reference, dtype):
    run_rmsnorm_test(dnn_reference, dtype)


@pytest.mark.batchnorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_batchnorm_multi_output(dnn_reference, dtype):
    run_batchnorm_test(dnn_reference, dtype)


@pytest.mark.parametrize("op_name", ("layernorm", "rmsnorm"))
def test_norm_reference_rejects_noncontiguous_parameters(
    dnn_reference, op_name
):
    if dnn_reference.vendor_name != "nvidia":
        pytest.skip("contiguity contract is NVIDIA-reference-specific")

    x = torch.randn((2, 5, 17), device=flag_dnn.device, dtype=torch.float16)
    scale_storage = torch.randn((17, 2), device=x.device, dtype=x.dtype)
    scale = scale_storage[:, 0]
    bias = torch.randn((17,), device=x.device, dtype=x.dtype)
    assert not scale.is_contiguous()

    with pytest.raises(ValueError, match="requires contiguous tensors"):
        dnn_reference.prepare(
            op_name,
            "TRAINING",
            x,
            scale,
            bias,
            1e-5,
        )


def test_batchnorm_reference_accepts_channels_last_input(dnn_reference):
    if dnn_reference.vendor_name != "nvidia":
        pytest.skip("channels-last regression is NVIDIA-reference-specific")

    x = torch.randn(
        (2, 8, 8, 8), device=flag_dnn.device, dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    params = [
        torch.ones((1, 8, 1, 1), device=x.device, dtype=x.dtype),
        torch.zeros((1, 8, 1, 1), device=x.device, dtype=x.dtype),
        torch.zeros((1, 8, 1, 1), device=x.device, dtype=torch.float32),
        torch.ones((1, 8, 1, 1), device=x.device, dtype=torch.float32),
    ]
    assert not x.is_contiguous()
    assert x.is_contiguous(memory_format=torch.channels_last)

    prepared = dnn_reference.prepare("batchnorm", x, *params, 1e-5, 0.1)
    try:
        outputs = prepared.run()
        dnn_reference.synchronize()
        assert all(torch.isfinite(output).all() for output in outputs)
    finally:
        prepared.close()


def test_batchnorm_reference_still_rejects_noncontiguous_parameters(
    dnn_reference,
):
    if dnn_reference.vendor_name != "nvidia":
        pytest.skip("contiguity contract is NVIDIA-reference-specific")

    x = torch.randn(
        (2, 8, 8, 8), device=flag_dnn.device, dtype=torch.float16
    ).to(memory_format=torch.channels_last)
    scale_storage = torch.ones((1, 8, 1, 2), device=x.device, dtype=x.dtype)
    scale = scale_storage[..., :1]
    params = [
        scale,
        torch.zeros_like(scale),
        torch.zeros((1, 8, 1, 1), device=x.device, dtype=torch.float32),
        torch.ones((1, 8, 1, 1), device=x.device, dtype=torch.float32),
    ]
    assert not scale.is_contiguous()

    with pytest.raises(ValueError, match="requires contiguous tensors"):
        dnn_reference.prepare("batchnorm", x, *params, 1e-5, 0.1)
