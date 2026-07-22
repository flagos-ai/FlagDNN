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

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def cudnn_handle():
    from benchmark.base import get_cudnn

    cudnn = get_cudnn()
    import torch

    try:
        cudnn.backend_version()
    except Exception as exc:
        pytest.skip(f"cuDNN backend is not available: {exc}")

    handle = cudnn.create_handle()
    cudnn.set_stream(
        handle=handle,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    try:
        yield handle
    finally:
        cudnn.destroy_handle(handle)


@pytest.fixture()
def dnn_baseline():
    from benchmark.baselines import create_baseline
    from devtools.dnn_reference import (
        DnnProviderNotImplementedError,
        DnnProviderUnavailableError,
    )

    try:
        baseline = create_baseline()
    except (
        DnnProviderNotImplementedError,
        DnnProviderUnavailableError,
    ) as exc:
        pytest.skip(str(exc))

    try:
        yield baseline
    finally:
        baseline.close()
