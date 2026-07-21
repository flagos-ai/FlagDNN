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

from typing import Callable

import triton

from benchmark import consts


def bench_ms(function: Callable[[], object]) -> float:
    return triton.testing.do_bench(
        function,
        warmup=consts.bench_warmup(),
        rep=consts.bench_repeat(),
        return_mode="median",
    )


class TritonBenchmarkTimer:
    """Default device timer supplied by the active Triton provider."""

    def measure_pair(
        self,
        first: Callable[[], object],
        second: Callable[[], object],
    ) -> tuple[float, float]:
        return bench_ms(first), bench_ms(second)
