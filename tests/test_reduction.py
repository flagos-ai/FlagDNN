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

import math

import pytest
import torch

import flag_dnn
from devtools.dnn_reference.interfaces import DnnReferenceNotSupportedError
from tests import consts
from tests.reduction_test_utils import run_reduction_test


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape,dim,mode", consts.REDUCTION_CASES)
def test_reduction(dnn_reference, dtype, shape, dim, mode):
    torch.manual_seed(0)
    run_reduction_test(dnn_reference, dtype, shape, dim, mode)


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("mode", ("ADD", "AVG", "MUL"))
def test_reduction_channels_last_repeated(dnn_reference, dtype, mode):
    shape = (2, 3, 5, 5)
    axis = 1
    x = torch.linspace(
        -1.1,
        1.1,
        math.prod(shape),
        device=flag_dnn.device,
        dtype=torch.float32,
    ).to(dtype)
    x = x.reshape(shape).contiguous(memory_format=torch.channels_last)
    x[0, 0, 0, 0] = 0
    is_nvidia = dnn_reference.vendor_name == "nvidia"
    if is_nvidia:
        views = tuple(x.select(axis, index) for index in range(shape[axis]))
        assert any(view.data_ptr() % 16 for view in views)

    if is_nvidia and mode == "MUL":
        with pytest.raises(
            DnnReferenceNotSupportedError,
            match="no exact unaligned MUL reduction fallback",
        ):
            dnn_reference.prepare("reduction", x, mode, dim=axis, keepdim=True)
        return

    prepared = dnn_reference.prepare(
        "reduction", x, mode, dim=axis, keepdim=True
    )
    try:
        if is_nvidia and not (mode == "ADD" and dtype == torch.float32):
            assert prepared.reference_name == "cuDNN standard composite"
        assert prepared.output.shape == (2, 1, 5, 5)
        method = {
            "ADD": torch.sum,
            "AVG": torch.mean,
            "MUL": torch.prod,
        }[mode]
        expected = method(x, dim=axis, keepdim=True)
        for _ in range(3):
            actual = prepared.run()
            dnn_reference.synchronize()
            torch.testing.assert_close(
                actual,
                expected,
                atol=8e-2 if dtype == torch.bfloat16 else 5e-2,
                rtol=1e-2,
            )
    finally:
        prepared.close()


@pytest.mark.reduction
@pytest.mark.graph
def test_reduction_all_dimensions_returns_scalar(dnn_reference):
    x = torch.linspace(
        -1.0,
        1.1,
        8,
        device=flag_dnn.device,
        dtype=torch.float32,
    )

    actual = dnn_reference.run("reduction", x, "ADD", dim=0, keepdim=False)
    expected = torch.sum(x, dim=0, keepdim=False)

    assert actual.shape == expected.shape == torch.Size([])
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_reduction_rejects_unaligned_input_entrance(dnn_reference, dtype):
    if dnn_reference.vendor_name != "nvidia":
        pytest.skip("16-byte entrance alignment is cuDNN-specific")

    storage = torch.randn((2, 4, 8, 9), device=flag_dnn.device, dtype=dtype)
    x = storage[..., 1:]
    assert x.data_ptr() % 16

    with pytest.raises(
        DnnReferenceNotSupportedError,
        match="16-byte-aligned input entrance",
    ):
        dnn_reference.prepare("reduction", x, "ADD", dim=1, keepdim=True)
