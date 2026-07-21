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

import os
import statistics
from typing import Callable

import triton

from benchmark import consts


def lower_percentile(values, quantile=0.2):
    """Return a measured sample without interpolating event timestamps."""
    if not values:
        raise ValueError("cannot summarize an empty timing sample")
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def _format_timing_distribution(values) -> str:
    ordered = sorted(values)
    return (
        f"min={ordered[0]:.6f}, "
        f"p20={lower_percentile(ordered):.6f}, "
        f"median={statistics.median(ordered):.6f}, "
        f"max={ordered[-1]:.6f}"
    )


class AscendEventTimer:
    """Paired NPU event timer that excludes host submission bubbles."""

    def measure_pair(
        self,
        first: Callable[[], object],
        second: Callable[[], object],
    ) -> tuple[float, float]:
        device_interface = triton.runtime.driver.active.get_device_interface()
        active_driver = triton.runtime.driver.active
        first()
        second()
        device_interface.synchronize()

        for _ in range(max(1, consts.bench_warmup())):
            first()
            second()
        device_interface.synchronize()

        repeat = max(1, consts.bench_repeat())
        cache = active_driver.get_empty_cache_for_benchmark()
        queue_guard_clears = 4
        for _ in range(queue_guard_clears):
            active_driver.clear_cache(cache)
        device_interface.synchronize()
        first_start = [
            device_interface.Event(enable_timing=True) for _ in range(repeat)
        ]
        first_end = [
            device_interface.Event(enable_timing=True) for _ in range(repeat)
        ]
        second_start = [
            device_interface.Event(enable_timing=True) for _ in range(repeat)
        ]
        second_end = [
            device_interface.Event(enable_timing=True) for _ in range(repeat)
        ]

        def record(function, start, end) -> None:
            for _ in range(queue_guard_clears):
                active_driver.clear_cache(cache)
            start.record()
            function()
            end.record()

        for index in range(repeat):
            if index % 2 == 0:
                record(first, first_start[index], first_end[index])
                record(second, second_start[index], second_end[index])
            else:
                record(second, second_start[index], second_end[index])
                record(first, first_start[index], first_end[index])

        device_interface.synchronize()
        first_times = [
            start.elapsed_time(end)
            for start, end in zip(first_start, first_end)
        ]
        second_times = [
            start.elapsed_time(end)
            for start, end in zip(second_start, second_end)
        ]
        if os.getenv("FLAGDNN_PERF_DEBUG_TIMING") == "1":
            print(
                "Ascend timing samples: "
                f"baseline({_format_timing_distribution(first_times)}), "
                f"FlagDNN({_format_timing_distribution(second_times)})"
            )
        return statistics.median(first_times), statistics.median(second_times)
