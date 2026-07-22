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


def _cudnn_rmsnorm_rht_amax(dnn_reference, x, w, eps, rows_per_cta):
    assert dnn_reference.supports("rmsnorm_rht_amax_wrapper_sm100", x.dtype)
    return dnn_reference.run(
        "rmsnorm_rht_amax_wrapper_sm100",
        x,
        w,
        eps=eps,
        rows_per_cta=rows_per_cta,
    )


def _run_graph(x, w, eps, rows_per_cta):
    @flag_dnn.graph
    def fn(x, w):
        return flag_dnn.rmsnorm_rht_amax_wrapper_sm100(
            x, w, eps=eps, rows_per_cta=rows_per_cta
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(w, "w"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "rmsnorm_rht_amax_wrapper_sm100"
    ]
    return compiled.run(x.clone(), w.clone())


def _assert_output_close(actual, cudnn_out):
    assert set(actual.keys()) == {"o_tensor", "amax_tensor"}
    assert set(cudnn_out.keys()) == {"o_tensor", "amax_tensor"}
    torch.testing.assert_close(
        actual["o_tensor"].float(),
        cudnn_out["o_tensor"].float(),
        atol=4e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual["amax_tensor"],
        cudnn_out["amax_tensor"],
        atol=2e-3,
        rtol=1e-3,
    )


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows_per_cta", [2, 4])
def test_rmsnorm_rht_amax_matches_cudnn(dnn_reference, rows_per_cta):
    torch.manual_seed(10)
    m = 8
    n = 2048
    eps = 1e-5
    x = torch.randn((m, n), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n,), device=flag_dnn.device, dtype=torch.bfloat16)

    cudnn_out = _cudnn_rmsnorm_rht_amax(dnn_reference, x, w, eps, rows_per_cta)
    actual = _run_graph(x, w, eps, rows_per_cta)
    _assert_output_close(actual, cudnn_out)


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_rht_amax_squeezes_trailing_unit_dims(dnn_reference):
    torch.manual_seed(11)
    m = 4
    n = 2048
    eps = 1e-5
    rows_per_cta = 2
    x = torch.randn((m, n, 1), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n, 1), device=flag_dnn.device, dtype=torch.bfloat16)

    cudnn_out = _cudnn_rmsnorm_rht_amax(dnn_reference, x, w, eps, rows_per_cta)
    actual = _run_graph(x, w, eps, rows_per_cta)

    assert actual["o_tensor"].shape == (m, n)
    assert actual["amax_tensor"].shape == (m // rows_per_cta,)
    _assert_output_close(actual, cudnn_out)


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_rht_amax_h100_reference_is_standard_frontend(dnn_reference):
    if torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("SM100+ uses the native cuDNN wrapper")
    x = torch.randn((4, 2048), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((2048,), device=x.device, dtype=x.dtype)
    prepared = dnn_reference.prepare(
        "rmsnorm_rht_amax_wrapper_sm100",
        x,
        w,
        rows_per_cta=2,
    )
    try:
        assert prepared.reference_name == "cuDNN standard composite"
    finally:
        prepared.close()


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_rht_amax_default_rows_per_cta(dnn_reference):
    torch.manual_seed(12)
    m = 1184
    n = 2048
    eps = 1e-5
    x = torch.randn((m, n), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((n,), device=x.device, dtype=x.dtype)

    expected = _cudnn_rmsnorm_rht_amax(dnn_reference, x, w, eps, None)
    actual = _run_graph(x, w, eps, None)

    assert actual["amax_tensor"].shape == (148,)
    assert expected["amax_tensor"].shape == (148,)
    _assert_output_close(actual, expected)


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_rht_amax_native_rejection_falls_back(
    dnn_reference, monkeypatch
):
    from devtools.dnn_reference.providers.nvidia_ops import (
        rmsnorm_rht_amax as provider_module,
    )

    def reject_native(*_args, **_kwargs):
        raise ValueError("native wrapper rejected this configuration")

    monkeypatch.setattr(
        provider_module.torch.cuda,
        "get_device_capability",
        lambda _device=None: (10, 0),
    )
    monkeypatch.setitem(
        provider_module.cudnn.__dict__,
        "rmsnorm_rht_amax_wrapper_sm100",
        reject_native,
    )
    x = torch.randn((4, 2048), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((2048,), device=x.device, dtype=x.dtype)
    prepared = dnn_reference.prepare(
        "rmsnorm_rht_amax_wrapper_sm100", x, w, rows_per_cta=2
    )
    try:
        assert prepared.reference_name == "cuDNN standard composite"
    finally:
        prepared.close()


@pytest.mark.rmsnorm_rht_amax
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_rht_amax_native_runtime_failure_is_not_retried(
    dnn_reference, monkeypatch
):
    from devtools.dnn_reference.providers.nvidia_ops import (
        rmsnorm_rht_amax as provider_module,
    )

    def fail_native(*_args, **_kwargs):
        raise RuntimeError("native execution failed")

    monkeypatch.setattr(
        provider_module.torch.cuda,
        "get_device_capability",
        lambda _device=None: (10, 0),
    )
    monkeypatch.setitem(
        provider_module.cudnn.__dict__,
        "rmsnorm_rht_amax_wrapper_sm100",
        fail_native,
    )
    x = torch.randn((4, 2048), device=flag_dnn.device, dtype=torch.bfloat16)
    w = torch.randn((2048,), device=x.device, dtype=x.dtype)

    with pytest.raises(RuntimeError, match="native execution failed"):
        dnn_reference.prepare(
            "rmsnorm_rht_amax_wrapper_sm100",
            x,
            w,
            rows_per_cta=2,
        )
