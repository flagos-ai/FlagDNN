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

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence


class BenchmarkTimer(Protocol):
    def measure_pair(
        self,
        first: Callable[[], object],
        second: Callable[[], object],
    ) -> tuple[float, float]: ...


def _contains_device_type(inputs: Sequence[Any], device_type: str) -> bool:
    return any(
        getattr(getattr(item, "device", None), "type", None) == device_type
        for item in inputs
    )


def create_timer(vendor_name: str, inputs: Sequence[Any]) -> BenchmarkTimer:
    if vendor_name == "ascend" and _contains_device_type(inputs, "npu"):
        from .ascend import AscendEventTimer

        return AscendEventTimer()

    from .default import TritonBenchmarkTimer

    return TritonBenchmarkTimer()


__all__ = ("BenchmarkTimer", "create_timer")
