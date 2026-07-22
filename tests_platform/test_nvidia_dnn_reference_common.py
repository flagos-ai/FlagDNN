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

from devtools.dnn_reference.interfaces import (
    DnnProviderUnavailableError,
    DnnReferenceNotSupportedError,
)

try:
    import cudnn as _cudnn  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(
        f"cuDNN frontend cannot be imported: {exc}",
        allow_module_level=True,
    )
from devtools.dnn_reference.providers.nvidia_ops import common  # noqa: E402


build_cudnn_graph = common.build_cudnn_graph


class _FailingGraph:
    def __init__(self, message):
        self.message = message

    def build(self, _heuristics):
        raise RuntimeError(self.message)


def test_bad_param_is_not_converted_to_capability_skip():
    graph = _FailingGraph("cudnn_status: CUDNN_STATUS_BAD_PARAM")
    with pytest.raises(RuntimeError, match="CUDNN_STATUS_BAD_PARAM") as caught:
        build_cudnn_graph(graph, "malformed")
    assert type(caught.value) is RuntimeError


def test_not_supported_is_reported_as_reference_capability():
    graph = _FailingGraph("cudnn_status: CUDNN_STATUS_NOT_SUPPORTED")
    with pytest.raises(DnnReferenceNotSupportedError, match="not_supported"):
        build_cudnn_graph(graph, "not_supported")


@pytest.mark.parametrize("backend_version", (92400, 92401))
def test_hopper_cudnn_924_fp32_sdpa_is_rejected_before_execute(
    monkeypatch, backend_version
):
    monkeypatch.setattr(
        common.cudnn, "backend_version", lambda: backend_version
    )
    monkeypatch.setattr(
        common.torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0),
    )

    tensor = torch.empty(1, dtype=torch.float32)
    with pytest.raises(
        DnnReferenceNotSupportedError, match="cannot safely execute"
    ):
        common.require_cudnn_sdpa_execution_supported(tensor, "sdpa")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_hopper_cudnn_924_low_precision_sdpa_remains_supported(
    monkeypatch, dtype
):
    monkeypatch.setattr(common.cudnn, "backend_version", lambda: 92400)
    monkeypatch.setattr(
        common.torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0),
    )
    common.require_cudnn_sdpa_execution_supported(
        torch.empty(1, dtype=dtype), "sdpa"
    )


def test_nvidia_context_reports_backend_unavailable(monkeypatch):
    def fail_backend():
        raise OSError("backend shared library is unavailable")

    monkeypatch.setattr(common.cudnn, "backend_version", fail_backend)
    with pytest.raises(
        DnnProviderUnavailableError,
        match="cuDNN runtime is unavailable",
    ):
        common.NvidiaContext()


def test_expanded_layout_uses_nonoverlapping_output():
    reference = torch.empty((1, 4)).expand(3, 4)

    output = common.empty_output_like_layout(
        reference, tuple(reference.shape), reference.dtype
    )

    assert output.is_contiguous()
    assert all(
        size <= 1 or stride > 0
        for size, stride in zip(output.shape, output.stride())
    )
