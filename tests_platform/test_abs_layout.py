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

"""Cross-platform layout regression tests for the abs operator."""

import importlib
from typing import Any

import torch

from flag_dnn.ops.binary import empty_like_preserve_dense_layout


def test_empty_like_preserve_dense_layout_uses_source_shape_and_stride(
    monkeypatch,
):
    source = torch.empty(
        (2, 3, 4, 5),
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    requested_dtype = torch.float64
    real_empty_strided = torch.empty_strided
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_empty_strided(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((args, kwargs))
        return real_empty_strided(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_strided", record_empty_strided)

    result = empty_like_preserve_dense_layout(source, requested_dtype)

    assert calls == [
        (
            (source.shape, source.stride()),
            {"device": source.device, "dtype": requested_dtype},
        )
    ]
    assert result.shape == source.shape
    assert result.stride() == source.stride()
    assert result.dtype == requested_dtype
    assert result.device == source.device


def test_abs_uses_layout_preserving_allocator(monkeypatch):
    abs_module = importlib.import_module("flag_dnn.ops.abs")
    source = torch.empty(
        (2, 3, 0, 5),
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    output = torch.empty_strided(
        source.shape,
        source.stride(),
        dtype=source.dtype,
        device=source.device,
    )
    calls = []

    def allocate(actual_source, dtype):
        calls.append((actual_source, dtype))
        return output

    monkeypatch.setattr(
        abs_module,
        "empty_like_preserve_dense_layout",
        allocate,
    )

    result = abs_module.abs(source)

    assert result is output
    assert calls == [(source, source.dtype)]
